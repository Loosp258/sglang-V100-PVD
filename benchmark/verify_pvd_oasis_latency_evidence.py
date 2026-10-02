"""Verify portable evidence hashes, archive CRCs, completed work and cleanup."""
import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shlex
from statistics import mean, median
import subprocess
import tarfile

from pvd_oasis_delivery_validation import (
    DELIVERY_COMPARISONS, comparison_workers, delivery_flags, validate_delivery_profiles,
    validate_delivery_snapshot, validate_io_snapshot,
)


def canonical(data, mode):
    if mode == 'lf':
        return data.replace(b'\r\n', b'\n')
    if mode != 'raw':
        raise ValueError('unsupported artifact normalization')
    return data


def archive_files(path):
    """Read an evidence archive without extraction or duplicate-name ambiguity."""
    with tarfile.open(path, 'r:gz') as archive:
        files = {}
        for member in archive:
            if member.isfile():
                assert member.name not in files, ('duplicate evidence member', member.name)
                files[member.name] = archive.extractfile(member).read()
        return files


def unique_file(files, filename):
    matches = [name for name in files if name.endswith('/' + filename)]
    assert len(matches) == 1, (filename, 'one complete evidence record required')
    return files[matches[0]]


def json_file(files, filename):
    return json.loads(unique_file(files, filename))


def zero_gpu_csv(data):
    rows = data.splitlines()
    assert len(rows) == 2, 'both GPUs require recorded cleanup proof'
    assert [int(row.split(',')[0]) for row in rows] == [0, 1]
    assert all(int(row.split(',')[1].split()[0]) == 0 for row in rows)


def client_token_evidence(complete, online):
    """Measure actual streamed token intervals, excluding admission/TTFT."""
    assert online['status'] == 200 and online['error'] is None
    assert type(online['completion_tokens']) is int and online['completion_tokens'] == 16
    events = online['events']
    assert len(events) == 16, 'one actual event for every output token required'
    times = [item['seconds'] for item in events]
    assert all(type(value) in (int, float) and math.isfinite(value) and value >= 0
               for value in times)
    assert times == sorted(times), 'client token events reordered'
    for count, item in enumerate(events, 1):
        event = item['event']
        assert type(event['meta_info']['completion_tokens']) is int
        assert event['meta_info']['completion_tokens'] == count
        assert event['meta_info']['id'] == complete['rid']
        assert event['output_ids'] == complete['output_ids'][:count]
    for field in ('case', 'prompt_sha256', 'output_sha256', 'completion_tokens',
                  'wall_seconds', 'first_event_seconds'):
        assert online[field] == complete[field], (field, 'actual client event identity differs')
    assert times[0] == online['first_event_seconds'] and times[-1] <= online['wall_seconds']
    intervals = [(last - first) * 1000 for first, last in zip(times, times[1:])]
    if 'client_token_event_seconds' in complete:
        assert complete['client_token_event_seconds'] == times
    if 'client_inter_token_ms' in complete:
        assert complete['client_inter_token_ms'] == intervals
    tpot = (times[-1] - times[0]) * 1000 / 15
    if 'client_tpot_ms' in complete:
        assert math.isclose(complete['client_tpot_ms'], tpot, abs_tol=1e-9)
    return dict(client_tpot_ms=tpot,
                steady_client_tpot_ms=(times[-1] - times[1]) * 1000 / 14,
                first_client_interval_ms=(times[1] - times[0]) * 1000,
                event_count=16, client_intervals=15, steady_client_intervals=14)


def callback_evidence(trace, forwards, *, workers):
    """Reconstruct executor occupancy from callback start/ready endpoints.

    Occupied slots include blocking RPC time; this is not hardware utilization.
    The work window divides the observed service sum by the worker count. It
    is a fixed-duration capacity measure, not an end-to-end latency prediction.
    """
    assert type(workers) is int and workers in (2, 4)
    assert len(forwards) == 15 and all(type(item['step']) is int for item in forwards)
    assert [item['step'] for item in forwards] == list(range(15))
    assert all(type(item['total_ms']) in (int, float) and math.isfinite(item['total_ms'])
               and item['total_ms'] >= 0 for item in forwards)
    assert all(type(item['wait_ms']) in (int, float) and math.isfinite(item['wait_ms'])
               and 0 <= item['wait_ms'] <= item['total_ms'] for item in forwards)
    assert forwards[0]['wait_ms'] == 0
    layers = trace['layers']
    expected = {(step, layer) for step in range(14) for layer in range(28)}
    assert len(layers) == 392 and {(item['step'], item['layer']) for item in layers} == expected
    assert all(type(item['step']) is int and type(item['layer']) is int for item in layers)
    assert [(item['step'], item['layer']) for item in layers] == sorted(expected)
    assert [item['consumed'] for item in layers] == sorted(item['consumed'] for item in layers)
    assert len(trace['transport']) == 420
    steady_transport = trace['transport'][28:]
    assert {(item['step'], item['layer']) for item in steady_transport} == expected
    endpoints = []
    for item in layers:
        for field in ('published', 'worker_start', 'ready', 'consumed', 'queue_seconds',
                      'service_seconds', 'consumer_wait_seconds'):
            value = item[field]
            assert type(value) in (int, float) and math.isfinite(value) and value >= 0, field
        assert item['published'] <= item['worker_start'] < item['ready'] <= item['consumed']
        assert math.isclose(item['queue_seconds'], item['worker_start'] - item['published'], abs_tol=1e-9)
        assert math.isclose(item['service_seconds'], item['ready'] - item['worker_start'], abs_tol=1e-9)
        assert type(item['ready_before_consume']) is bool
        wait_start = item['consumed'] - item['consumer_wait_seconds']
        assert item['ready_before_consume'] == (item['ready'] <= wait_start)
        endpoints.extend(((item['worker_start'], 1), (item['ready'], -1)))
    endpoints.sort()  # A completion at the same timestamp precedes the next start.
    active = peak = 0
    area = 0.0
    previous = endpoints[0][0]
    for timestamp, delta in endpoints:
        area += active * (timestamp - previous)
        active += delta
        assert 0 <= active <= workers, 'actual executor concurrency exceeds configuration'
        peak = max(peak, active)
        previous = timestamp
    assert active == 0
    service = sum(item['service_seconds'] for item in layers)
    assert math.isclose(area, service, abs_tol=1e-8)
    span = endpoints[-1][0] - endpoints[0][0]
    wait_ms = sum(item['wait_ms'] for item in forwards[1:])
    raw_wait_ms = sum(item['consumer_wait_seconds'] for item in layers) * 1000
    assert abs(raw_wait_ms - wait_ms) <= 14 * 0.00051, 'forward wait differs from actual layer waits'
    return dict(workers=workers, consumed_callbacks=392, steady_tokens=14,
                executor_peak=peak, service_seconds=service, worker_window_seconds=span,
                worker_occupancy=service / (workers * span),
                mean_service_ms=service * 1000 / 392,
                mean_queue_ms=mean(item['queue_seconds'] for item in layers) * 1000,
                work_window_ms_per_token=service * 1000 / workers / 14,
                kv_wait_mean_ms=wait_ms / 14,
                kv_wait_step_median_ms=median(item['wait_ms'] for item in forwards[1:]),
                kv_wait_request_sum_ms=wait_ms,
                ready_before_consume_fraction=mean(item['ready_before_consume'] for item in layers))


def worker_timing_evidence(full, online, comparison):
    """Independent request and mode statistics from complete raw observations."""
    requests = {}
    for arm, rows in full['requests'].items():
        actual = online[arm]
        assert len(actual) == len(rows) == 2
        requests[arm] = []
        workers = comparison_workers(comparison, arm)
        for complete, client in zip(rows, actual):
            evidence = callback_evidence(complete['trace'], complete['forward'], workers=workers)
            assert evidence['executor_peak'] == workers, 'configured worker concurrency not demonstrated'
            evidence.update(client_token_evidence(complete, client))
            evidence.update(case=complete['case'], rid=complete['rid'])
            requests[arm].append(evidence)
    aggregate = {}
    for mode in ('baseline', 'optimized'):
        rows = [row for arm, values in requests.items()
                if arm.startswith('opt') is (mode == 'optimized') for row in values]
        forwards = [item for arm, values in full['requests'].items()
                    if arm.startswith('opt') is (mode == 'optimized')
                    for row in values for item in row['forward'][1:]]
        assert len(rows) == 4 and len(forwards) == 56
        workers = rows[0]['workers']
        service = sum(row['service_seconds'] for row in rows)
        span = sum(row['worker_window_seconds'] for row in rows)
        aggregate[mode] = dict(requests=4, steady_tokens=56, consumed_callbacks=1568,
            workers=workers, executor_peaks=[row['executor_peak'] for row in rows],
            client_tpot_mean_ms=mean(row['client_tpot_ms'] for row in rows),
            client_tpot_request_median_ms=median(row['client_tpot_ms'] for row in rows),
            steady_client_tpot_mean_ms=mean(row['steady_client_tpot_ms'] for row in rows),
            steady_client_tpot_request_median_ms=median(row['steady_client_tpot_ms'] for row in rows),
            kv_wait_mean_ms=mean(item['wait_ms'] for item in forwards),
            kv_wait_step_median_ms=median(item['wait_ms'] for item in forwards),
            kv_wait_request_sum_median_ms=median(row['kv_wait_request_sum_ms'] for row in rows),
            mean_service_ms=service * 1000 / 1568,
            mean_queue_ms=mean(row['mean_queue_ms'] for row in rows),
            work_window_ms_per_token=service * 1000 / workers / 56,
            worker_occupancy=service / (workers * span),
            ready_before_consume_fraction=mean(row['ready_before_consume_fraction'] for row in rows))
    return dict(aggregate=aggregate, requests=requests,
        scope='four formal requests per mode; client TPOT uses 15 actual stream intervals; steady client/forward waiting uses 14 tokens per request; executor occupancy includes callback RPC blocking and is not hardware utilization')


def verify_native_slots(native):
    """Validate the complete local-session byte gate, not just its count label."""
    assert native['status'] == 'passed' and native['transport'] == 'mooncake_local_session'
    assert native['device'] == 'cuda:1' and native['current_device'] == 'cuda:0'
    assert native['two_executor_pools'] is True
    assert native['bootstrap_and_decode_thread_ids_reusable'] is True
    rows = [1, 2, 8, 16, 32, 64]
    assert native['rows'] == rows
    assert type(native['exact_byte_cases']) is int and native['exact_byte_cases'] == 48
    observations = native['observations']
    assert len(observations) == 48
    expected = {(round_id, worker, rank, count)
                for round_id in range(2) for worker in range(2)
                for rank in range(2) for count in rows}
    observed, generations, regions = set(), set(), {0: set(), 1: set()}
    registrations = 0
    for item in observations:
        names = ('executor_round', 'worker_slot', 'rank', 'rows', 'nbytes',
                 'transferred_bytes', 'worker_thread', 'physical_register_calls')
        assert all(type(item[name]) is int for name in names), 'typed native counters required'
        key = tuple(item[name] for name in ('executor_round', 'worker_slot', 'rank', 'rows'))
        assert key in expected and key not in observed, 'missing, duplicate or wrong native byte case'
        observed.add(key)
        assert item['nbytes'] == item['transferred_bytes'] == item['rows'] * 512
        assert item['terminal_state'] == 'terminal_success' and item['exact_bytes'] is True
        assert isinstance(item['generation'], str) and item['generation']
        assert item['generation'] not in generations, 'native logical generation replay'
        generations.add(item['generation'])
        assert isinstance(item['region_id'], str) and item['region_id']
        regions[item['rank']].add(item['region_id'])
        assert item['physical_register_calls'] in (0, 1)
        registrations += item['physical_register_calls']
        for name in ('allocate_seconds', 'register_seconds'):
            assert type(item[name]) in (int, float) and math.isfinite(item[name]) and item[name] >= 0
    assert observed == expected and registrations == 4
    assert len(regions[0]) == len(regions[1]) == 2 and regions[0].isdisjoint(regions[1])
    for name, closed, physical_bytes in (('before_close', False, 131072), ('after_close', True, 0)):
        state = native[name]
        assert state['closed'] is closed and state['closing'] is closed and state['quarantine'] is None
        for field, value in (('slots_per_rank', 2), ('capacity_bytes', 32768),
                             ('physical_register_calls', 4), ('physical_registrations', 4),
                             ('acquired_leases', 48), ('returned_leases', 48),
                             ('leased_slots', 0), ('unknown_slots', 0),
                             ('physical_slots', 0 if closed else 4), ('physical_bytes', physical_bytes),
                             ('physical_release_calls', 4 if closed else 0),
                             ('physical_releases', 4 if closed else 0)):
            assert type(state[field]) is int and state[field] == value, (name, field)
    for name in ('used_staging_bytes', 'used_inflight', 'reservations'):
        assert type(native['transfer_budget'][name]) is int and native['transfer_budget'][name] == 0
    health = native['receive_health']
    assert health['healthy'] is True
    assert health['registered_destinations'] == health['quarantined_collisions'] == 0
    assert health['rails'], 'native receive rail proof required'
    for rail in health['rails'].values():
        assert rail['healthy'] is True and rail['registration_unknown_reason'] is None
        assert rail['registered_regions'] == 0
        lifecycle = rail['lifecycle']
        assert lifecycle['quarantined'] is False and lifecycle['quarantine_reason'] is None
        for name in ('used_staging_bytes', 'used_inflight', 'reservations',
                     'tracked_transfers', 'unknown_transfers'):
            assert type(lifecycle[name]) is int and lifecycle[name] == 0


def verify_delivery_gate(folder, manifest, source_hashes):
    for name in ('gate.tar.gz', 'implementation.json', 'gate_count_record.json'):
        assert name in manifest, (name, 'complete delivery gate must be hashed')
    gate = archive_files(folder / 'gate.tar.gz')
    bundle = unique_file(gate, 'deployed.tar.gz')
    raw = gzip.decompress(bundle)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        deployed = {}
        for member in archive:
            if member.isfile():
                assert member.name not in deployed, 'duplicate source in deployment bundle'
                deployed[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    expected = json_file(gate, 'local_source_hashes.json')
    assert deployed == expected and deployed, 'bundle bytes differ from gate source hashes'
    assert set(source_hashes) == {'v', 'd'}, 'both serving role source proofs required'
    prefix = 'python/sglang/srt/disaggregation/pvd/'
    delivery = {prefix + name for name in (
        'client.py', 'sparse_receiver.py', 'cuda_sparse_receiver.py', 'protocol.py',
        'control_server.py', 'vector_store.py', 'mooncake_engine.py',
        'transfer_engine.py', 'oasis_receive_slots.py')}
    v_required = delivery | {prefix + name for name in (
        'cagra_backend.py', 'prompt_index.py', 'server.py', 'cagra_kv_update.py',
        'cagra_kv_prepare.py', 'cagra_search_batch.py', 'index_search.py')}
    d_required = delivery | {name for name in deployed
                             if name.startswith(prefix + 'oasis') and name.endswith('.py')}
    d_required |= {'python/sglang/srt/server_args.py', 'python/sglang/srt/models/qwen2.py',
                   'python/sglang/srt/managers/scheduler.py',
                   'python/sglang/srt/managers/scheduler_components/batch_result_processor.py',
                   'python/sglang/srt/disaggregation/decode.py'}
    for role, required in (('v', v_required), ('d', d_required)):
        assert json_file(gate, role + '_source_hashes.json') == expected
        assert set(source_hashes[role]) == required, (role, 'incomplete serving source gate')
        assert all(deployed.get(name) == digest for name, digest in source_hashes[role].items()), role
        zero_gpu_csv(unique_file(gate, role + '_initial_gpu.txt').decode('utf-8'))
        zero_gpu_csv(unique_file(gate, role + '_final_gpu.txt').decode('utf-8'))
    implementation = json.loads((folder / 'implementation.json').read_text(encoding='utf-8'))
    assert implementation['bundle_sha256'] == hashlib.sha256(bundle).hexdigest()
    assert implementation['deployed_files'] == len(deployed)
    assert implementation['serving_sources_match_gate'] is True
    assert implementation['live_working_tree_sources_used'] is False
    status = json_file(gate, 'unit_status.json')
    assert type(status['exit_code']) is int and status['exit_code'] == 0
    unit_text = unique_file(gate, 'unit.txt').decode('utf-8')
    summaries = [line for line in unit_text.splitlines()
                 if re.search(r'\d+ passed', line) and re.search(r'in [\d.]+s', line)]
    assert len(summaries) == 1
    counts = {kind: int(number) for number, kind in re.findall(
        r'(\d+) (passed|failed|skipped|errors?|warnings?)', summaries[0])}
    assert counts.get('passed', 0) > 0 and not any(counts.get(name, 0) for name in ('failed', 'error', 'errors'))
    record = json.loads((folder / 'gate_count_record.json').read_text(encoding='utf-8'))
    assert record['unit']['counts'] == counts and record['unit']['exact_summary'] == summaries[0]
    assert record['unit']['exit_code'] == 0
    native_status = json_file(gate, 'native_status.json')
    assert type(native_status['exit_code']) is int and native_status['exit_code'] == 0
    native = json_file(gate, 'native.json')
    verify_native_slots(native)
    assert record['native_full_observations_saved'] is True
    assert record['native'] == {name: value for name, value in native.items() if name != 'observations'}
    return deployed


def verify_worker_cpu_gate(folder, manifest, deployed):
    """Keep the repeated CPU lifecycle gate distinct from native acceptance."""
    for name in ('worker_cpu_gate.tar.gz', 'worker_cpu_gate_record.json'):
        assert name in manifest, (name, 'worker CPU recheck evidence must be hashed')
    files = archive_files(folder / 'worker_cpu_gate.tar.gz')
    assert json_file(files, 'status.json') == {'exit_code': 0}
    text = unique_file(files, 'unit.txt').decode('utf-8')
    summaries = [line for line in text.splitlines()
                 if re.search(r'\d+ passed', line) and re.search(r'in [\d.]+s', line)]
    assert len(summaries) == 1
    counts = {kind: int(number) for number, kind in re.findall(
        r'(\d+) (passed|failed|skipped|errors?|warnings?)', summaries[0])}
    assert counts == {'passed': 48, 'warning': 1}
    scope = json_file(files, 'scope.json')
    assert scope['frozen_serving_commit'] == '9b8b5dc0c'
    assert scope['new_mirror_tests'] is scope['gpu_used'] is scope['ssh_used'] is False
    assert scope['test_scope'] == 'existing CPU causal/lifecycle regressions; not native four-worker acceptance'
    assert scope['tests'] == ['test_pvd_oasis_attention.py', 'test_pvd_oasis_request.py',
        'test_pvd_oasis_pipeline.py', 'test_pvd_oasis_serving.py', 'test_pvd_oasis_transport_io.py',
        'test_pvd_oasis_receive_slot_records.py']
    hashes = json_file(files, 'source_hashes.json')
    prefix = 'python/sglang/srt/disaggregation/pvd/'
    assert set(hashes) == {prefix + name for name in (
        'oasis_attention.py', 'oasis_pipeline.py', 'oasis_request.py', 'oasis_transport.py',
        'oasis_startup.py', 'oasis_receive_slots.py', 'sparse_receiver.py',
        'cuda_sparse_receiver.py', 'oasis_sglang.py')}
    assert all(deployed.get(name) == digest for name, digest in hashes.items()), 'worker CPU serving source differs from deployed gate'
    record = json.loads((folder / 'worker_cpu_gate_record.json').read_text(encoding='utf-8'))
    assert record['counts'] == counts and record['exact_summary'] == summaries[0]
    assert record['exit_code'] == 0 and record['scope'] == scope
    assert record['source_files_match_deployment'] == len(hashes)


def verify_delivery_launches(files, arms, *, direct_sparse=False):
    """Recheck archived launch modes and warmup inputs independently of summaries."""
    warmups = []
    for arm in arms:
        rows = json_file(files, arm + '_warmup.json')
        assert [row['case'] for row in rows] == [99991, 99992]
        assert all(row['status'] == 200 and not row['error'] and row['completion_tokens'] == 16
                   for row in rows)
        warmups.append([row['prompt_sha256'] for row in rows])
    assert all(values == warmups[0] for values in warmups), 'warmup prompts changed between arms'
    flags = dict(PVD_PARTIAL_GROUP_SEARCH='1', PVD_HOST_CANDIDATES='1', PVD_NATIVE_POOL='1',
                 PVD_HOST_QUERY_VALIDATION='0', PVD_TRITON_SPARSE_PACKING='0',
                 PVD_DIRECT_PD_BOOTSTRAP='0', PVD_GATE_INITIAL_FANIN_ON_INDEX='1',
                 PVD_CAGRA_ITOPK_SIZE='2048', PVD_CAGRA_GROUP_HEADS='4',
                 PVD_CAGRA_KV_ROUTING_EDGES='2', PVD_MODE='oasis')
    for role in ('p', 'v', 'd', 'gateway'):
        environments = []
        for arm in arms:
            command = unique_file(files, arm + '_' + role + '.launch').decode('utf-8')
            tokens = shlex.split(command.split('; bash ', 1)[0])
            assert tokens.pop(0) == 'export'
            pairs = [token.split('=', 1) for token in tokens]
            assert all(len(pair) == 2 for pair in pairs)
            env = dict(pairs)
            assert len(env) == len(pairs), 'duplicate launch variable'
            assert all(env[name] == value for name, value in flags.items()), (arm, role, 'changed fixed serving mode')
            if direct_sparse:
                assert env['PVD_DIRECT_SPARSE_BATCH_PUT'] == str(int(role == 'v' and arm.startswith('opt')))
            environments.append({name: value for name, value in env.items()
                                 if name not in ('PVD_RUN_TAG', 'PVD_OASIS_CONFIG')
                                 and not (direct_sparse and role == 'v' and name == 'PVD_DIRECT_SPARSE_BATCH_PUT')})
        assert all(env == environments[0] for env in environments), (role, 'unrelated launch changes')


def verify(folder, *, check_git=False):
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    repo = Path(__file__).resolve().parents[1]
    for name, entry in manifest.items():
        mode = entry.get('normalization', 'raw')
        data = canonical((folder / name).read_bytes(), mode)
        assert len(data) == entry['bytes'], name
        assert hashlib.sha256(data).hexdigest() == entry['sha256'], name
        if check_git:
            relative = (folder / name).resolve().relative_to(repo).as_posix()
            blob = subprocess.check_output(['git', '-c', f'safe.directory={repo.as_posix()}',
                                            '-C', str(repo), 'show', 'HEAD:' + relative])
            blob = canonical(blob, mode)
            assert len(blob) == entry['bytes'] and hashlib.sha256(blob).hexdigest() == entry['sha256'], name
        if name.endswith('.tar.gz'):
            # Check full gzip CRC/truncation before reading tar, without extraction.
            raw = gzip.decompress(data)
            with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
                for member in archive:
                    if member.isfile():
                        archive.extractfile(member).read()
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    assert all(item['prompt_identical'] and item['output_ids_identical'] and item['text_identical']
               for item in summary['output_identity'].values())
    assert len(summary['requests']) == 4
    comparison = json.loads((folder / 'comparison.json').read_text(encoding='utf-8'))
    is_io = comparison['comparison'] == 'v-io'
    is_pack = comparison['comparison'] == 'v-pack'
    is_workers = comparison['comparison'] == 'v-workers'
    is_direct_sparse = comparison['comparison'] == 'v-direct-sparse'
    is_gpu_bank = comparison['comparison'] == 'd-gpu-bank'
    is_stages = comparison['comparison'] == 'd-stages'
    is_workspace = comparison['comparison'] == 'd-workspace'
    is_delivery = comparison['comparison'] in DELIVERY_COMPARISONS
    if is_io or is_pack or is_delivery:
        expected_arms = ['base_a', 'opt_a', 'opt_b', 'base_b']
        assert comparison['arms'] == expected_arms, 'comparison requires the recorded ABBA order'
        assert type(comparison['tokens']) is int and comparison['tokens'] == 16
        cases = [int(value) for value in comparison['cases'].split(',')]
        assert len(cases) == len(set(cases)) == 2, 'comparison requires two distinct declared cases'
        assert list(summary['requests']) == expected_arms, 'incomplete or reordered comparison arms'
        assert set(summary['output_identity']) == {str(case) for case in cases}
        for case in cases:
            assert summary['output_identity'][str(case)]['arms'] == expected_arms
        for arm, rows in summary['requests'].items():
            assert [row['case'] for row in rows] == cases, (arm, 'mismatched request cases')
    if is_pack:
        assert 'pack_modes.json' in manifest, 'packing runtime evidence must be hashed'
        pack_modes = json.loads((folder / 'pack_modes.json').read_text(encoding='utf-8'))
        assert set(pack_modes) == set(expected_arms), 'missing or extra packing arms'
        for arm in expected_arms:
            mode = pack_modes[arm]
            assert mode['source'] == 'GET /internal/health from both V rank endpoints; full responses saved'
            ranks = mode['ranks']
            assert [item['rank'] for item in ranks] == [0, 1], (arm, 'both V ranks required')
            expected_kernel = 'triton' if arm.startswith('opt') else 'torch'
            for item in ranks:
                rank = item['rank']
                assert type(rank) is int
                assert item['sparse_pack_kernel'] == expected_kernel
                assert item['device'] == f'cuda:{rank}'
                health_file = f'{arm}_v_rank{rank}_health.json'
                assert item['health_file'] == health_file
                assert health_file in manifest, (arm, rank, 'full health response must be hashed')
                health = json.loads((folder / health_file).read_text(encoding='utf-8'))
                assert type(health['rank']) is int and health['rank'] == rank
                assert health['ready'] is True
                assert health['device'] == item['device']
                assert health['sparse_packing_mode'] == 'cuda_synchronous_experimental'
                assert health['sparse_pack_kernel'] == expected_kernel
    for arm, rows in summary['requests'].items():
        assert len(rows) == 2
        for row in rows:
            assert row['completion_tokens'] == 16 and row['cached_tokens'] == 0
            assert row['trace_counts']['layers'] == 392
            assert row['trace_counts']['transport'] == 420
            if is_io or is_pack or is_delivery:
                # Keep this request snapshot in the compact summary as well as
                # the complete raw D trace. Counts prove actual reuse, not just
                # a launch flag or the presence of one shared client object.
                snapshot = row.get('io')
                reuse = is_io and arm.startswith('opt')
                validate_io_snapshot(snapshot, reuse_io=reuse, jobs=420)
                if is_delivery:
                    assert row['prompt_tokens'] == 2159
                    validate_delivery_snapshot(snapshot, comparison=comparison['comparison'], arm=arm)
    assert all(all(int(line.split(',')[1].split()[0]) == 0 for line in text.splitlines())
               for text in json.loads((folder / 'final_gpu_memory.json').read_text()).values())
    with tarfile.open(folder / 'raw.tar.gz', 'r:gz') as archive:
        names = {member.name: member for member in archive if member.isfile()}
        owned = next(name for name in names if name.endswith('/owned.json'))
        cleanup = next(name for name in names if name.endswith('/cleanup_errors.json'))
        assert json.loads(archive.extractfile(names[owned]).read()) == {}
        assert json.loads(archive.extractfile(names[cleanup]).read()) == []
        if is_delivery:
            raw_files = {name: archive.extractfile(member).read() for name, member in names.items()}
            verify_delivery_launches(raw_files, expected_arms, direct_sparse=is_direct_sparse)
            assert json_file(raw_files, 'comparison.json') == comparison
            final_gpu = json.loads((folder / 'final_gpu_memory.json').read_text(encoding='utf-8'))
            assert set(final_gpu) == {'p', 'v', 'd'}, 'all six GPUs require cleanup proof'
            assert json_file(raw_files, 'final_gpu_memory.json') == final_gpu
            for text in final_gpu.values():
                zero_gpu_csv(text)
            summary_names = [name for name in names if name.endswith('/summary.json')]
            assert len(summary_names) == 1, 'one complete raw summary is required'
            full = json.loads(archive.extractfile(names[summary_names[0]]).read())
            assert list(full['requests']) == expected_arms
            configs = {}
            for arm in expected_arms:
                config_names = [name for name in names if name.endswith('/' + arm + '_config.json')]
                assert len(config_names) == 1
                config = json.loads(archive.extractfile(names[config_names[0]]).read())
                configs[arm] = config
                combine, slots = delivery_flags(comparison['comparison'], arm)
                assert config['combine_reserve_start'] is combine and config['reuse_receive_slots'] is slots
                assert config['reuse_io'] is False and config['overlap'] is True
                assert type(config['workers']) is int
                assert config['workers'] == comparison_workers(comparison['comparison'], arm)
                assert config['capacity'] == 32 and config['top_k'] == 4
                assert config['max_new'] == 16
                if is_gpu_bank:
                    assert config['gpu_receive_to_bank'] is arm.startswith('opt')
                if is_stages:
                    assert config['staged_transport'] is arm.startswith('opt')
                    assert config['gpu_receive_to_bank'] is False
                if is_workspace:
                    assert config['attention_workspace'] is arm.startswith('opt')
                    assert config['staged_transport'] is config['gpu_receive_to_bank'] is False
                assert [row['case'] for row in full['requests'][arm]] == cases
                for complete, compact in zip(full['requests'][arm], summary['requests'][arm]):
                    assert complete['rid'] == compact['rid']
                    for field in ('case', 'prompt_sha256', 'output_ids', 'output_sha256',
                                  'completion_tokens', 'cached_tokens', 'wall_seconds', 'first_event_seconds'):
                        assert complete[field] == compact[field], (arm, field, 'compact request differs from raw')
                    if 'steady_wait_sum_ms' in compact:
                        assert compact['steady_wait_sum_ms'] == sum(
                            item['wait_ms'] for item in complete['forward'] if item['step'] > 0)
                    trace = complete['trace']
                    assert len(trace['layers']) == 392 and len(trace['transport']) == 420
                    assert trace['io'] == compact['io'], 'compact counters differ from complete raw trace'
                    validate_io_snapshot(trace['io'], reuse_io=False, jobs=420)
                    proof = validate_delivery_profiles(trace, comparison=comparison['comparison'], arm=arm)
                    if 'delivery_validation' in compact:
                        assert compact['delivery_validation'] == proof
            allowed = {'v-combine': 'combine_reserve_start', 'v-slots': 'reuse_receive_slots',
                       'v-workers': 'workers', 'v-direct-sparse': '__no_config_difference__',
                       'd-gpu-bank': 'gpu_receive_to_bank',
                       'd-stages': 'staged_transport',
                       'd-workspace': 'attention_workspace'}[comparison['comparison']]
            fixed = {name: value for name, value in configs[expected_arms[0]].items() if name != allowed}
            assert all({name: value for name, value in config.items() if name != allowed} == fixed
                       for config in configs.values()), 'unrelated configuration changes in raw evidence'
            for filename in ('v_search_summary.json', 'rpc_summary.json', 'source_hashes.json', 'comparison.json'):
                assert filename in manifest, (filename, 'delivery evidence must be hashed')
                assert json_file(raw_files, filename) == json.loads(
                    (folder / filename).read_text(encoding='utf-8')), (filename, 'compact evidence differs from raw')
            sources = json.loads((folder / 'source_hashes.json').read_text(encoding='utf-8'))
            if is_direct_sparse or is_gpu_bank or is_stages or is_workspace:
                from pvd_oasis_direct_sparse_evidence import gate_proof, runtime_proof
                deployed = gate_proof(archive_files(folder / 'gate.tar.gz'), sources, json_file, unique_file,
                                      gpu_bank=is_gpu_bank, staged=is_stages, workspace=is_workspace)
                if is_direct_sparse:
                    assert 'direct_sparse_summary.json' in manifest
                    assert runtime_proof(full, lambda name: json_file(raw_files, name)) == json.loads(
                        (folder / 'direct_sparse_summary.json').read_text())
            else:
                deployed = verify_delivery_gate(folder, manifest, sources)
            search = json.loads((folder / 'v_search_summary.json').read_text(encoding='utf-8'))
            rpc = json.loads((folder / 'rpc_summary.json').read_text(encoding='utf-8'))
            for mode in ('baseline', 'optimized'):
                assert search['aggregate'][mode]['batches'] == rpc['aggregate'][mode]['search_rpc_count'] == 3136
                assert search['aggregate'][mode]['paths'] == ['grouped_cagra_partial_batched_host']
            if is_stages:
                from pvd_oasis_stage_evidence import stage_timing_evidence
                assert 'stage_timing_summary.json' in manifest
                assert stage_timing_evidence(full, json_file(raw_files, 'online.json')) == json.loads(
                    (folder / 'stage_timing_summary.json').read_text(encoding='utf-8'))
            if is_workers or is_direct_sparse or is_gpu_bank or is_workspace:
                if is_workers:
                    verify_worker_cpu_gate(folder, manifest, deployed)
                assert 'worker_timing_summary.json' in manifest, 'actual worker/client timings must be hashed'
                online = json_file(raw_files, 'online.json')
                assert list(online) == expected_arms
                for arm in expected_arms:
                    log = unique_file(raw_files, arm + '_d.log').decode('utf-8')
                    traced = re.findall(r'PVD Oasis trace rid=(\S+) data=(\{[^\n]+\})', log)
                    for complete in full['requests'][arm]:
                        records = [json.loads(data) for rid, data in traced if rid == complete['rid']]
                        assert len(records) == 1 and records[0] == complete['trace'], 'summary callbacks differ from actual D trace'
                        forwards = [dict(step=int(step), total_ms=float(total), wait_ms=float(wait))
                                    for rid, step, total, wait in re.findall(
                                        r'PVD Oasis forward rid=(\S+) step=(\d+) total_ms=([\d.]+) layer_wait_ms=([\d.]+)', log)
                                    if rid == complete['rid']]
                        assert forwards == complete['forward'], 'summary forward timings differ from actual D log'
                timing = worker_timing_evidence(full, online, comparison['comparison'])
                preserved = json.loads((folder / 'worker_timing_summary.json').read_text(encoding='utf-8'))
                assert preserved == timing, 'preserved worker timing differs from actual raw endpoints/events'
                for arm, rows in timing['requests'].items():
                    worker_rows = rpc['requests'][arm]
                    assert len(worker_rows) == len(rows)
                    for observed, profiled in zip(rows, worker_rows):
                        assert profiled['case'] == observed['case']
                        assert profiled['workers'] == observed['workers']
                        assert math.isclose(profiled['service_seconds'], observed['service_seconds'], abs_tol=1e-9)
                        assert math.isclose(profiled['worker_window_seconds'], observed['worker_window_seconds'], abs_tol=1e-9)
                        assert math.isclose(profiled['worker_utilization'], observed['worker_occupancy'], abs_tol=1e-9)
    return dict(passed=True, files=len(manifest), formal_requests=8, git_blobs_checked=check_git)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--check-git', action='store_true')
    a = p.parse_args()
    print(json.dumps(verify(a.directory, check_git=a.check_git)))
