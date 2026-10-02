"""Analyze live formal requests, preserving startup and steady-state scopes."""
import argparse
import json
from pathlib import Path
import re
import shlex
import statistics

from pvd_oasis_delivery_validation import (
    DELIVERY_COMPARISONS, delivery_flags, validate_delivery_profiles,
    validate_io_snapshot,
)

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('--comparison', choices=('pipeline', 'v-search', 'v-latency', 'v-host-query', 'v-io', 'v-pack') + DELIVERY_COMPARISONS, default='pipeline')
a = p.parse_args()
root = a.directory
if (root / 'comparison.json').exists():
    assert json.loads((root / 'comparison.json').read_text())['comparison'] == a.comparison
online = json.loads((root / 'online.json').read_text())
configs = [json.loads(path.read_text()) for path in sorted(root.glob('*_config.json'))]
if a.comparison in ('v-pack',) + DELIVERY_COMPARISONS:
    assert len(configs) == len(online), 'missing/extra comparison configs'
if configs:
    allowed_difference = {'v-io': 'reuse_io', 'v-combine': 'combine_reserve_start',
                          'v-slots': 'reuse_receive_slots'}.get(a.comparison, 'overlap')
    comparison_config = {k: v for k, v in configs[0].items() if k != allowed_difference}
    assert all({k: v for k, v in config.items() if k != allowed_difference} == comparison_config
               for config in configs), 'comparison has unrelated configuration differences'
    if a.comparison == 'v-io':
        for path in sorted(root.glob('*_config.json')):
            config=json.loads(path.read_text())
            assert config['overlap'] is True
            assert config['reuse_io'] is path.name.startswith('opt'), 'IO knob does not match comparison arm'
    if a.comparison in ('v-search', 'v-latency', 'v-host-query', 'v-pack'):
        assert all(config == configs[0] and config['overlap'] is True for config in configs), (
            'V search comparison requires identical overlapped Decode configs')
    if a.comparison == 'v-pack':
        assert all(config['reuse_io'] is False for config in configs), 'packing comparison changed HTTP mode'
    if a.comparison in DELIVERY_COMPARISONS:
        for path in sorted(root.glob('*_config.json')):
            config = json.loads(path.read_text())
            arm = path.name.removesuffix('_config.json')
            combine, slots = delivery_flags(a.comparison, arm)
            assert config['overlap'] is True and config['reuse_io'] is False
            assert config['combine_reserve_start'] is combine and config['reuse_receive_slots'] is slots
            assert config['workers'] == 2 and config['top_k'] == 4 and config['capacity'] == 32
        declared = json.loads((root / 'comparison.json').read_text())
        assert list(online) == declared['arms'] == ['base_a', 'opt_a', 'opt_b', 'base_b']
        cases = [int(value) for value in declared['cases'].split(',')]
        assert len(cases) == len(set(cases)) == 2 and declared['tokens'] == 16
        for arm, rows in online.items():
            assert [row['case'] for row in rows] == cases
            warm = json.loads((root / f'{arm}_warmup.json').read_text())
            assert [row['case'] for row in warm] == [99991, 99992]
            assert all(row['status'] == 200 and not row['error'] and row['completion_tokens'] == 16 for row in warm)
        for role in ('p', 'v', 'd', 'gateway'):
            launch_envs = []
            for arm in online:
                command = (root / f'{arm}_{role}.launch').read_text()
                tokens = shlex.split(command.split('; bash ', 1)[0])
                assert tokens.pop(0) == 'export'
                env = dict(token.split('=', 1) for token in tokens)
                for name, expected in (('PVD_PARTIAL_GROUP_SEARCH', '1'), ('PVD_HOST_CANDIDATES', '1'),
                                       ('PVD_NATIVE_POOL', '1'), ('PVD_HOST_QUERY_VALIDATION', '0'),
                                       ('PVD_TRITON_SPARSE_PACKING', '0'), ('PVD_DIRECT_PD_BOOTSTRAP', '0'),
                                       ('PVD_GATE_INITIAL_FANIN_ON_INDEX', '1'), ('PVD_CAGRA_ITOPK_SIZE', '2048')):
                    assert env[name] == expected, (arm, role, name)
                launch_envs.append({key: value for key, value in env.items()
                                    if key not in ('PVD_RUN_TAG', 'PVD_OASIS_CONFIG')})
            assert all(env == launch_envs[0] for env in launch_envs), (role, 'unrelated launch difference')
if a.comparison == 'v-pack':
    pack_modes = json.loads((root / 'pack_modes.json').read_text())
    assert set(pack_modes) == set(online), 'missing/extra actual packing mode arms'
    for arm, proof in pack_modes.items():
        expected = 'triton' if arm.startswith('opt') else 'torch'
        assert len(proof['ranks']) == 2 and {r['rank'] for r in proof['ranks']} == {0, 1}
        for rank in proof['ranks']:
            assert rank['sparse_pack_kernel'] == expected
            assert rank['health_file'] == f"{arm}_v_rank{rank['rank']}_health.json"
            raw = json.loads((root / rank['health_file']).read_text())
            assert type(raw['rank']) is int and raw['rank'] == rank['rank'] and raw['ready'] is True
            assert raw['device'] == rank['device'] == f"cuda:{rank['rank']}"
            assert raw['sparse_pack_kernel'] == expected
            assert raw['sparse_packing_mode'] == 'cuda_synchronous_experimental'
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
        if a.comparison in ('v-io', 'v-pack') + DELIVERY_COMPARISONS:
            io=trace['io']
            reuse=a.comparison == 'v-io' and arm.startswith('opt')
            jobs=len(forwards)*28
            validate_io_snapshot(io, reuse_io=reuse, jobs=jobs)
        if a.comparison in DELIVERY_COMPARISONS:
            assert record['prompt_tokens'] == 2159 and row['completion_tokens'] == 16
            assert final['meta_info']['cached_tokens'] == 0
            record['delivery_validation'] = validate_delivery_profiles(trace, comparison=a.comparison, arm=arm)
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
scope = {
    'pipeline': 'live paired serial vs live paired per-layer overlap',
    'v-search': 'live overlapped paired with baseline vs partial-head cached V search',
    'v-latency': 'live overlapped paired with cached V baseline vs bounded RMM pool and host candidates',
    'v-host-query': 'live overlapped paired with pooled host candidates and GPU vs private CPU finite-Q proof',
    'v-io': 'live overlapped paired with identical V and per-job vs request-scoped HTTP clients',
    'v-pack': 'live overlapped paired with identical V search and per-job HTTP; Torch vs Triton sparse packing',
    'v-combine': 'live overlapped paired with identical fast V search, per-job HTTP and per-delivery registration; separate vs combined reserve/start',
    'v-slots': 'live overlapped paired with identical fast V search, per-job HTTP and separate reserve/start; per-delivery vs request-owned physical receive registrations',
}[a.comparison] + '; full initial KV retained'
if a.comparison in DELIVERY_COMPARISONS:
    assert all(item['prompt_identical'] and item['output_ids_identical'] and item['text_identical']
               for item in output_identity.values()), 'delivery comparison changed prompt or actual output'
result = dict(aggregate=aggregate, output_identity=output_identity, requests=summary,
    comparison_scope=scope)
(root / 'summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(dict(aggregate=aggregate, output_identity=output_identity), indent=2))
