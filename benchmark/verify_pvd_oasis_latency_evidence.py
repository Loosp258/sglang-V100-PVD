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
import subprocess
import tarfile

from pvd_oasis_delivery_validation import (
    DELIVERY_COMPARISONS, delivery_flags, validate_delivery_profiles,
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


def verify_delivery_launches(files, arms):
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
            environments.append({name: value for name, value in env.items()
                                 if name not in ('PVD_RUN_TAG', 'PVD_OASIS_CONFIG')})
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
            verify_delivery_launches(raw_files, expected_arms)
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
                assert config['workers'] == 2 and config['capacity'] == 32 and config['top_k'] == 4
                assert config['max_new'] == 16
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
            allowed = 'combine_reserve_start' if comparison['comparison'] == 'v-combine' else 'reuse_receive_slots'
            fixed = {name: value for name, value in configs[expected_arms[0]].items() if name != allowed}
            assert all({name: value for name, value in config.items() if name != allowed} == fixed
                       for config in configs.values()), 'unrelated configuration changes in raw evidence'
            for filename in ('v_search_summary.json', 'rpc_summary.json', 'source_hashes.json', 'comparison.json'):
                assert filename in manifest, (filename, 'delivery evidence must be hashed')
                assert json_file(raw_files, filename) == json.loads(
                    (folder / filename).read_text(encoding='utf-8')), (filename, 'compact evidence differs from raw')
            sources = json.loads((folder / 'source_hashes.json').read_text(encoding='utf-8'))
            verify_delivery_gate(folder, manifest, sources)
            search = json.loads((folder / 'v_search_summary.json').read_text(encoding='utf-8'))
            rpc = json.loads((folder / 'rpc_summary.json').read_text(encoding='utf-8'))
            for mode in ('baseline', 'optimized'):
                assert search['aggregate'][mode]['batches'] == rpc['aggregate'][mode]['search_rpc_count'] == 3136
                assert search['aggregate'][mode]['paths'] == ['grouped_cagra_partial_batched_host']
    return dict(passed=True, files=len(manifest), formal_requests=8, git_blobs_checked=check_git)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--check-git', action='store_true')
    a = p.parse_args()
    print(json.dumps(verify(a.directory, check_git=a.check_git)))
