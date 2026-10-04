"""Analyze live formal requests, preserving startup and steady-state scopes."""
import argparse
import json
from pathlib import Path
import re
import shlex
import statistics

from pvd_oasis_delivery_validation import (
    DELIVERY_COMPARISONS, comparison_workers, delivery_flags, validate_delivery_profiles,
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
                          'v-slots': 'reuse_receive_slots', 'v-workers': 'workers',
                          'v-direct-sparse': '__no_config_difference__',
                          'v-pack-fence': '__no_config_difference__',
                          'd-gpu-bank': 'gpu_receive_to_bank',
                          'd-stages': 'staged_transport',
                          'd-workspace': 'attention_workspace',
                          'd-batch-install': 'batched_bank_install',
                          'd-cache-install': 'batched_cache_install',
                          'v-contiguous': 'sort_missing_tokens'}.get(a.comparison, 'overlap')
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
            assert config['workers'] == comparison_workers(a.comparison, arm)
            assert config['top_k'] == 4 and config['capacity'] == 32
            if a.comparison == 'd-gpu-bank':
                assert config['gpu_receive_to_bank'] is arm.startswith('opt')
            if a.comparison == 'd-stages':
                assert config['staged_transport'] is arm.startswith('opt')
                assert config['gpu_receive_to_bank'] is False
            if a.comparison == 'd-workspace':
                assert config['attention_workspace'] is arm.startswith('opt')
                assert config['staged_transport'] is config['gpu_receive_to_bank'] is False
            if a.comparison == 'v-contiguous':
                assert config['sort_missing_tokens'] is arm.startswith('opt')
                assert config['staged_transport'] is config['gpu_receive_to_bank'] is config['attention_workspace'] is False
            if a.comparison == 'd-batch-install':
                assert config['batched_bank_install'] is arm.startswith('opt')
                assert config['staged_transport'] is config['gpu_receive_to_bank'] is config['attention_workspace'] is config['sort_missing_tokens'] is False
            if a.comparison == 'd-cache-install':
                assert config['batched_cache_install'] is arm.startswith('opt')
                assert config['batched_bank_install'] is config['staged_transport'] is config['gpu_receive_to_bank'] is config['attention_workspace'] is config['sort_missing_tokens'] is False
            if a.comparison == 'v-pack-fence':
                assert config['batched_cache_install'] is config['batched_bank_install'] is config['staged_transport'] is config['gpu_receive_to_bank'] is config['attention_workspace'] is config['sort_missing_tokens'] is False
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
                                    if key not in ('PVD_RUN_TAG', 'PVD_OASIS_CONFIG')
                                    and not (a.comparison == 'v-direct-sparse'
                                             and role == 'v' and key == 'PVD_DIRECT_SPARSE_BATCH_PUT')
                                    and not (a.comparison == 'v-contiguous'
                                             and role == 'v' and key == 'PVD_CONTIGUOUS_SPARSE_PACKING')
                                    and not (a.comparison == 'v-pack-fence'
                                             and role == 'v' and key == 'PVD_REUSE_SPARSE_PACK_FENCE')})
                if a.comparison == 'v-direct-sparse':
                    assert env['PVD_DIRECT_SPARSE_BATCH_PUT'] == str(int(
                        role == 'v' and arm.startswith('opt')))
                if a.comparison == 'v-contiguous':
                    assert env['PVD_CONTIGUOUS_SPARSE_PACKING'] == str(int(
                        role == 'v' and arm.startswith('opt')))
                if a.comparison in ('d-batch-install', 'd-cache-install'):
                    assert env['PVD_CONTIGUOUS_SPARSE_PACKING'] == env['PVD_DIRECT_SPARSE_BATCH_PUT'] == '0'
                if a.comparison == 'v-pack-fence':
                    assert env['PVD_REUSE_SPARSE_PACK_FENCE'] == str(int(role == 'v' and arm.startswith('opt')))
                    assert env['PVD_CONTIGUOUS_SPARSE_PACKING'] == env['PVD_DIRECT_SPARSE_BATCH_PUT'] == '0'
            assert all(env == launch_envs[0] for env in launch_envs), (role, 'unrelated launch difference')
if a.comparison in ('v-pack', 'v-contiguous', 'd-batch-install', 'd-cache-install', 'v-pack-fence'):
    pack_modes = json.loads((root / 'pack_modes.json').read_text())
    assert set(pack_modes) == set(online), 'missing/extra actual packing mode arms'
    for arm, proof in pack_modes.items():
        expected = ('torch_contiguous_runs' if a.comparison == 'v-contiguous' else 'triton') if arm.startswith('opt') and a.comparison in ('v-pack', 'v-contiguous') else 'torch'
        assert len(proof['ranks']) == 2 and {r['rank'] for r in proof['ranks']} == {0, 1}
        for rank in proof['ranks']:
            assert rank['sparse_pack_kernel'] == expected
            assert rank['health_file'] == f"{arm}_v_rank{rank['rank']}_health.json"
            raw = json.loads((root / rank['health_file']).read_text())
            assert type(raw['rank']) is int and raw['rank'] == rank['rank'] and raw['ready'] is True
            assert raw['device'] == rank['device'] == f"cuda:{rank['rank']}"
            assert raw['sparse_pack_kernel'] == expected
            assert raw['sparse_packing_mode'] == 'cuda_synchronous_experimental'
            if a.comparison == 'v-pack-fence':
                assert raw['reuse_sparse_pack_fence'] is arm.startswith('opt')
if a.comparison == 'v-pack-fence':
    sources = json.loads((root / 'source_hashes.json').read_text())
    for role in ('v', 'd'):
        assert re.fullmatch('[0-9a-f]{64}', sources[role]['python/sglang/srt/disaggregation/pvd/v_source_profile.py'])
if a.comparison in ('d-batch-install', 'd-cache-install'):
    sources = json.loads((root / 'source_hashes.json').read_text())
    helper = 'python/sglang/srt/disaggregation/pvd/' + ('oasis_bank_install.py' if a.comparison == 'd-batch-install' else 'oasis_cache_install.py')
    assert re.fullmatch('[0-9a-f]{64}', sources['d'][helper]), 'missing deployed install source proof'
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
        token_times = [event['seconds'] for event in row['events']]
        assert len(token_times) == row['completion_tokens']
        assert [event['event']['meta_info']['completion_tokens'] for event in row['events']] == list(range(1, len(token_times) + 1))
        assert all(later >= earlier for earlier, later in zip(token_times, token_times[1:]))
        intervals_ms = [(later - earlier) * 1000 for earlier, later in zip(token_times, token_times[1:])]
        steady_tokens = len(forwards) - 1
        wait_sum_ms = sum(forward['wait_ms'] for forward in forwards if forward['step'] > 0)
        record.update(client_token_event_seconds=token_times,
                      client_inter_token_ms=intervals_ms,
                      client_tpot_ms=(token_times[-1] - token_times[0]) * 1000 / (len(token_times) - 1),
                      steady_tokens=steady_tokens, steady_wait_sum_ms=wait_sum_ms,
                      steady_wait_mean_ms_per_token=wait_sum_ms / steady_tokens)
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

if a.comparison == 'v-pack-fence':
    # Native adapter counters are process-wide, so include full bootstrap PUTs.
    # They supplement the per-delivery profiles and deployed adapter source hash.
    for arm, rows in summary.items():
        for rank in (0, 1):
            before = json.loads((root / f'{arm}_v_rank{rank}_warmed_health.json').read_text())
            after = json.loads((root / f'{arm}_v_rank{rank}_after_health.json').read_text())
            sparse_count = sum(item['rank'] == rank for row in rows
                for transport in row['trace']['transport'] for item in transport['deliveries'])
            for state in (before, after):
                assert state['rank'] == rank and state['isolated_reason'] is None
                assert state['reuse_sparse_pack_fence'] is arm.startswith('opt')
            first, last = before['transport']['submit_timing'], after['transport']['submit_timing']
            assert last['cuda_sync_calls'] - first['cuda_sync_calls'] >= sparse_count
            assert last['native_submit_calls'] - first['native_submit_calls'] >= sparse_count


def median(values):
    return statistics.median(values) if values else None


def mean(values):
    return statistics.fmean(values) if values else None


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
        client_tpot_mean_ms=mean([r['client_tpot_ms'] for r in rows]),
        client_tpot_median_ms=median([r['client_tpot_ms'] for r in rows]),
        client_inter_token_median_ms=median([interval for r in rows for interval in r['client_inter_token_ms']]),
        initialization_seconds=median([r['initialization_seconds'] for r in rows]),
        first_decode_ms=median(first),
        steady_forward_ms=median([f['total_ms'] for f in forwards]),
        steady_forward_mean_ms=mean([f['total_ms'] for f in forwards]),
        steady_layer_wait_sum_ms=median([f['wait_ms'] for f in forwards]),
        steady_wait_mean_ms_per_token=mean([f['wait_ms'] for f in forwards]),
        steady_request_wait_sum_ms=median([r['steady_wait_sum_ms'] for r in rows]),
        steady_request_wait_sum_mean_ms=mean([r['steady_wait_sum_ms'] for r in rows]),
        steady_foreground_except_wait_ms=median([f['total_ms'] - f['wait_ms'] for f in forwards]),
        # First EAGLE proposal initializes Prompt cache during admission.
        steady_draft_ms=median([t for r in rows for t in r['draft_ms'][1:]]),
        layer_rpc_ms=median([t['rpc_seconds'] * 1000 for t in transport]),
        layer_queue_ms=median([t['queue_seconds'] * 1000 for t in layers]),
        layer_queue_mean_ms=mean([t['queue_seconds'] * 1000 for t in layers]),
        layer_service_ms=median([t['service_seconds'] * 1000 for t in layers]),
        layer_service_mean_ms=mean([t['service_seconds'] * 1000 for t in layers]),
        layer_consumer_wait_ms=median([t['consumer_wait_seconds'] * 1000 for t in layers]),
        network_sparse_bytes_per_request=median([sum(t['remote_rows'] * 512 for t in r['trace']['transport'])
            for r in rows if 'trace' in r]),
        startup_full_kv_bytes_per_request=median([r['prompt_tokens'] * 28 * 4 * 128 * 2 * 2 for r in rows]),
        actual_forward_count=sum(len(r['forward']) for r in rows),
        layer_consume_count=len(layers))
    if a.comparison in ('d-batch-install', 'd-cache-install'):
        installs = [t['bank_install'] for t in transport]
        aggregate[mode].update(
            layer_install_mean_ms=mean([p['install_seconds'] * 1000 for p in installs]),
            layer_install_median_ms=median([p['install_seconds'] * 1000 for p in installs]),
            steady_kv_h2d_bytes=sum(p['kv_h2d_bytes'] for p in installs),
            steady_kv_h2d_calls=sum(p['kv_h2d_calls'] for p in installs),
            steady_resident_gather_calls=sum(p['resident_gather_calls'] for p in installs),
            steady_resident_scatter_calls=sum(p['resident_scatter_calls'] for p in installs),
            steady_cpu_scatter_calls=sum(p['cpu_scatter_calls'] for p in installs))
    if a.comparison == 'd-cache-install':
        deliveries = [p for t in transport for p in t['deliveries']]
        aggregate[mode].update(
            delivery_cache_copy_mean_ms=mean([p['cache_copy_seconds'] * 1000 for p in deliveries]),
            delivery_cache_copy_median_ms=median([p['cache_copy_seconds'] * 1000 for p in deliveries]),
            steady_cache_rows=sum(p['cache_installed_rows'] for p in deliveries),
            steady_cache_kv_bytes=sum(p['cache_kv_bytes'] for p in deliveries),
            steady_cache_row_clones=sum(p['cache_row_clones'] for p in deliveries),
            steady_cache_kv_copy_calls=sum(p['cache_kv_copy_calls'] for p in deliveries),
            steady_cache_valid_write_calls=sum(p['cache_valid_write_calls'] for p in deliveries))
    if a.comparison == 'v-pack-fence':
        source_profiles = [p['v_source'] for t in transport for p in t['deliveries']]
        aggregate[mode].update(
            steady_v_source_deliveries=len(source_profiles),
            steady_v_outer_fences_reused=sum(p['outer_fence_reused'] for p in source_profiles),
            steady_v_source_phases={name: dict(
                calls=sum(p['phases'][name]['calls'] for p in source_profiles),
                mean_ms=mean([p['phases'][name]['seconds'] * 1000 for p in source_profiles]),
                median_ms=median([p['phases'][name]['seconds'] * 1000 for p in source_profiles]))
                for name in source_profiles[0]['phases']},
            v_source_timing_scope='per completed rank delivery; pack is launch wall time, adapter submit includes its mandatory source fence; phase medians are not additive')

by_case = {}
for arm, rows in summary.items():
    for row in rows:
        by_case.setdefault(row['case'], []).append((arm, row))
output_identity = {case: dict(
    prompt_identical=len({r['prompt_sha256'] for _, r in rows}) == 1,
    output_ids_identical=len({tuple(r['output_ids']) for _, r in rows}) == 1,
    text_identical=len({r['output_sha256'] for _, r in rows}) == 1,
    arms=[arm for arm, _ in rows]) for case, rows in by_case.items()}
if a.comparison in ('d-batch-install', 'd-cache-install', 'v-pack-fence'):
    # Fewer returned/installed rows must not masquerade as an installation win.
    for case, items in by_case.items():
        signatures = [[(t['remote_rows'], t['bank_install']['selected_rows'],
                        t['bank_install']['resident_rows'], t['bank_install']['cpu_rows'],
                        t['bank_install']['kv_h2d_bytes']) for t in r['trace']['transport']]
                      for _, r in items]
        assert all(s == signatures[0] for s in signatures), (case, 'candidate or byte budgets changed')
scope = {
    'pipeline': 'live paired serial vs live paired per-layer overlap',
    'v-search': 'live overlapped paired with baseline vs partial-head cached V search',
    'v-latency': 'live overlapped paired with cached V baseline vs bounded RMM pool and host candidates',
    'v-host-query': 'live overlapped paired with pooled host candidates and GPU vs private CPU finite-Q proof',
    'v-io': 'live overlapped paired with identical V and per-job vs request-scoped HTTP clients',
    'v-pack': 'live overlapped paired with identical V search and per-job HTTP; Torch vs Triton sparse packing',
    'v-combine': 'live overlapped paired with identical fast V search, per-job HTTP and per-delivery registration; separate vs combined reserve/start',
    'v-slots': 'live overlapped paired with identical fast V search, per-job HTTP and separate reserve/start; per-delivery vs request-owned physical receive registrations',
    'v-workers': 'live overlapped paired with identical fast V search, per-job HTTP, separate reserve/start and per-delivery registration; two vs four callback workers',
    'v-direct-sparse': 'live overlapped paired with identical fast V search and D; staged packed PUT vs direct registered Entry scatter batch PUT',
    'd-gpu-bank': 'live overlapped paired with identical packed-PUT V; CPU cache round trip vs direct GPU bank rows with owned asynchronous historical backup',
    'd-stages': 'live overlapped paired with identical packed-PUT V and CPU historical cache; full-chain callbacks vs bounded persistent search/delivery/install stages',
    'd-workspace': 'live overlapped paired with identical packed-PUT V and CPU history; original variable-span attention vs bounded preallocated workspace, no serving CUDA graphs',
    'v-contiguous': 'live overlapped paired with identical fast V search and selected banks; row packing vs contiguous-run packing with wire-only miss sorting',
    'd-batch-install': 'live overlapped paired with identical packed-PUT V and CPU cache; per-head vs bounded all-head KV bank installation',
    'd-cache-install': 'live overlapped paired with identical packed-PUT V and per-head GPU banks; per-token clone/copy vs grouped monotonic CPU-cache install after unchanged D2H fence',
    'v-pack-fence': 'live overlapped paired with identical fast V/CAGRA, D, Torch packed PUT and candidate/byte budgets; original store fences vs reuse of successful pack fence for outer preparation, mandatory adapter fence retained',
}[a.comparison] + '; full initial KV retained'
if a.comparison in DELIVERY_COMPARISONS:
    assert all(item['prompt_identical'] and item['output_ids_identical'] and item['text_identical']
               for item in output_identity.values()), 'delivery comparison changed prompt or actual output'
result = dict(aggregate=aggregate, output_identity=output_identity, requests=summary,
    comparison_scope=scope, timing_scope=dict(
        client_tpot='per request (last token event minus first token event)/(completion_tokens-1); aggregate mean and median of request values',
        client_inter_token_median='median of individual token-event intervals; different from median request TPOT',
        steady_wait='step>0 only; mean per token and median of actual summed waits per request',
        layer_means='arithmetic means of completed consumed-layer callbacks; warmups and initial-bank priming excluded'))
(root / 'summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(dict(aggregate=aggregate, output_identity=output_identity), indent=2))
