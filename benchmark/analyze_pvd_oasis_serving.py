"""Analyze live formal requests, preserving startup and steady-state scopes."""
import argparse
import json
from pathlib import Path
import re
import statistics

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('--comparison', choices=('pipeline', 'v-search', 'v-latency'), default='pipeline')
a = p.parse_args()
root = a.directory
if (root / 'comparison.json').exists():
    assert json.loads((root / 'comparison.json').read_text())['comparison'] == a.comparison
online = json.loads((root / 'online.json').read_text())
configs = [json.loads(path.read_text()) for path in sorted(root.glob('*_config.json'))]
if configs:
    comparison_config = {k: v for k, v in configs[0].items() if k != 'overlap'}
    assert all({k: v for k, v in config.items() if k != 'overlap'} == comparison_config
               for config in configs), 'comparison has configuration differences beyond overlap'
    if a.comparison in ('v-search', 'v-latency'):
        assert all(config == configs[0] and config['overlap'] is True for config in configs), (
            'V search comparison requires identical overlapped Decode configs')
summary, modes = {}, ({'serial': [], 'overlap': []} if a.comparison == 'pipeline'
                     else {'baseline': [], 'optimized': []})
for arm, rows in online.items():
    log = (root / (arm + '_d.log')).read_text()
    records = {}
    for rid, step, total, wait in re.findall(
        r'PVD Oasis forward rid=(\S+) step=(\d+) total_ms=([\d.]+) layer_wait_ms=([\d.]+)', log):
        records.setdefault(rid, {}).setdefault('forward', []).append(dict(
            step=int(step), total_ms=float(total), wait_ms=float(wait)))
    for rid, tokens, seconds in re.findall(
        r'PVD Oasis initialized rid=(\S+) prompt_tokens=(\d+) seconds=([\d.]+)', log):
        records.setdefault(rid, {}).update(prompt_tokens=int(tokens), initialization_seconds=float(seconds))
    for rid, ms in re.findall(r'PVD Oasis draft rid=(\S+) ms=([\d.]+)', log):
        records.setdefault(rid, {}).setdefault('draft_ms', []).append(float(ms))
    for rid, data in re.findall(r'PVD Oasis trace rid=(\S+) data=(\{[^\n]+\})', log):
        records.setdefault(rid, {})['trace'] = json.loads(data)
    selected = []
    for row in rows:
        assert row['status'] == 200 and not row['error'], 'request failed'
        final = row['events'][-1]['event']
        rid = final['meta_info']['id']
        record = records[rid]
        forwards = record['forward']
        assert len(forwards) == row['completion_tokens'] - 1
        assert [f['step'] for f in forwards] == list(range(len(forwards)))
        trace = record.get('trace')
        if trace is not None:
            assert len(trace['layers']) == (len(forwards) - 1) * 28
            assert len(trace['transport']) == len(forwards) * 28
            assert {(t['step'], t['layer']) for t in trace['layers']} == {
                (s, l) for s in range(len(forwards) - 1) for l in range(28)}
        selected.append(dict(case=row['case'], rid=rid,
            prompt_sha256=row['prompt_sha256'], output_ids=final['output_ids'],
            output_sha256=row['output_sha256'], completion_tokens=row['completion_tokens'],
            cached_tokens=final['meta_info']['cached_tokens'],
            wall_seconds=row['wall_seconds'], first_event_seconds=row['first_event_seconds'],
            stream_seconds=row['wall_seconds'] - row['first_event_seconds'],
            **record))
    summary[arm] = selected
    mode = ('overlap' if arm.startswith('overlap') else 'serial') if a.comparison == 'pipeline' else (
        'optimized' if arm.startswith('opt') else 'baseline')
    modes[mode].extend(selected)


def median(values):
    return statistics.median(values) if values else None


aggregate = {}
for mode, rows in modes.items():
    if not rows:
        continue
    forwards = [f for r in rows for f in r['forward'] if f['step'] > 0]
    first = [r['forward'][0]['total_ms'] for r in rows]
    transport = [t for r in rows if 'trace' in r for t in r['trace']['transport'][28:]]
    layers = [t for r in rows if 'trace' in r for t in r['trace']['layers']]
    aggregate[mode] = dict(requests=len(rows),
        wall_seconds=median([r['wall_seconds'] for r in rows]),
        first_event_seconds=median([r['first_event_seconds'] for r in rows]),
        stream_seconds=median([r['stream_seconds'] for r in rows]),
        initialization_seconds=median([r['initialization_seconds'] for r in rows]),
        first_decode_ms=median(first),
        steady_forward_ms=median([f['total_ms'] for f in forwards]),
        steady_layer_wait_sum_ms=median([f['wait_ms'] for f in forwards]),
        steady_foreground_except_wait_ms=median([f['total_ms'] - f['wait_ms'] for f in forwards]),
        # First EAGLE proposal initializes Prompt cache during admission.
        steady_draft_ms=median([t for r in rows for t in r['draft_ms'][1:]]),
        layer_rpc_ms=median([t['rpc_seconds'] * 1000 for t in transport]),
        layer_queue_ms=median([t['queue_seconds'] * 1000 for t in layers]),
        layer_service_ms=median([t['service_seconds'] * 1000 for t in layers]),
        layer_consumer_wait_ms=median([t['consumer_wait_seconds'] * 1000 for t in layers]),
        network_sparse_bytes_per_request=median([sum(t['remote_rows'] * 512 for t in r['trace']['transport'])
            for r in rows if 'trace' in r]),
        startup_full_kv_bytes_per_request=median([r['prompt_tokens'] * 28 * 4 * 128 * 2 * 2 for r in rows]),
        actual_forward_count=sum(len(r['forward']) for r in rows),
        layer_consume_count=len(layers))

by_case = {}
for arm, rows in summary.items():
    for row in rows:
        by_case.setdefault(row['case'], []).append((arm, row))
output_identity = {case: dict(
    prompt_identical=len({r['prompt_sha256'] for _, r in rows}) == 1,
    output_ids_identical=len({tuple(r['output_ids']) for _, r in rows}) == 1,
    text_identical=len({r['output_sha256'] for _, r in rows}) == 1,
    arms=[arm for arm, _ in rows]) for case, rows in by_case.items()}
result = dict(aggregate=aggregate, output_identity=output_identity, requests=summary,
    comparison_scope=('live paired serial vs live paired per-layer overlap' if a.comparison == 'pipeline'
        else ('live overlapped paired with baseline vs partial-head cached V search'
            if a.comparison == 'v-search' else 'live overlapped paired with cached V baseline vs bounded RMM pool and host candidates')) + '; full initial KV retained')
(root / 'summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(dict(aggregate=aggregate, output_identity=output_identity), indent=2))
