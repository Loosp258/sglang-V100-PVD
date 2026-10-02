"""Separate qualification for direct Entry scatter, using archived observations."""
import hashlib
import io
import json
import tarfile


def native_proof(native, *, expected_mode='direct_sparse_batch_put'):
    assert native['status'] == 'passed'
    assert native['mode'] == expected_mode
    assert native['transport'] == 'mooncake_local_session_scatter'
    assert native['exact_byte_cases'] == len(native['observations']) == 48
    for row in native['observations']:
        assert row['exact_bytes'] is row['terminal_success'] is row['cleanup_complete'] is True
        assert row['nbytes'] == row['expected_bytes'] == row['transferred_bytes'] == row['rows'] * 512
        assert row['slices'] == row['rows'] * 2 <= 128
        assert row['staging_registration_count'] == 0
        assert row['source_registration_count'] == 1
        assert row['original_source_region_id'] == row['source_region_id']
        assert row['source_original_mr'] is row['destination_sentinel_exact'] is True
        assert row['source_pins_observed'] == row['tracked_transfers_observed'] == 2
        assert row['remote_offsets'] == list(range(0, row['nbytes'], 256))
        assert sum(item['length'] for item in row['source_slices']) == row['nbytes']
        assert all(row['allocation_offset'] <= item['offset'] < item['offset'] + item['length']
                   <= row['allocation_offset'] + row['source_entry_bytes']
                   for item in row['source_slices'])
        assert row['device'] == 'cuda:1' and row['current_device'] == 'cuda:0'
        if expected_mode == 'gpu_receive_to_bank_with_async_backup':
            bank = row['gpu_bank']
            assert bank['gpu_bank_exact'] is bank['cpu_backup_exact'] is True
            assert bank['original_mr_retired_before_cpu_publication'] is True
            assert bank['cache_valid_before_publication'] is False
            assert bank['backup_before']['completed'] == 0
            assert bank['backup_before']['charged_bytes'] == 2 * row['nbytes']
            assert bank['backup_after']['closed'] is True
            assert bank['backup_after']['quarantined'] is False
            assert bank['backup_after']['completed'] == bank['backup_after']['submitted'] == 1
            assert bank['backup_after']['rows_copied'] == row['rows']
            assert all(bank['backup_after'][name] == 0 for name in (
                'pending_rows', 'retained_owners', 'charged_bytes'))
    assert native['source_physical_register_calls'] == 2
    assert native['staging_registration_count'] == 0
    assert native['immutable_source_bytes_exact'] is native['all_owners_retired'] is True
    assert native['all_sources_unregistered'] is native['all_destinations_unregistered'] is True
    assert native['transfer_budget']['used_staging_bytes'] == native['transfer_budget']['used_inflight'] == 0
    assert native['after_close']['physical_bytes'] == 0
    assert native['after_close']['closed'] is True
    assert native['after_close']['unknown_slots'] == 0
    return native


def runtime_proof(full, load_json):
    proof = {}
    for arm, rows in full['requests'].items():
        proof[arm] = []
        for rank in (0, 1):
            states = {phase: load_json(f'{arm}_v_rank{rank}_{phase}_health.json')
                      for phase in ('before', 'warmed', 'after')}
            optimized = arm.startswith('opt')
            for state in states.values():
                assert state['rank'] == rank and state['device'] == f'cuda:{rank}'
                assert state['ready'] is True and state['isolated_reason'] is None
                assert state['direct_sparse_batch_put'] == dict(enabled=optimized, max_slices=128)
            profiles = [item for row in rows for transport in row['trace']['transport']
                        for item in transport['deliveries'] if item['rank'] == rank]
            assert profiles
            before = states['warmed']['metrics']['counters']
            after = states['after']['metrics']['counters']
            batches = after.get('vector_sparse_batch_submissions', 0) - before.get('vector_sparse_batch_submissions', 0)
            slices = after.get('vector_sparse_batch_slices', 0) - before.get('vector_sparse_batch_slices', 0)
            assert batches == (len(profiles) if optimized else 0)
            assert slices == (sum(2 * item['remote_rows'] for item in profiles) if optimized else 0)
            assert states['after']['quarantined_index_sources'] == 0
            lifecycle = states['after']['transport']['lifecycle']
            assert lifecycle['quarantined'] is False and lifecycle['quarantine_reason'] is None
            assert all(lifecycle[name] == 0 for name in (
                'used_staging_bytes', 'used_inflight', 'reservations',
                'tracked_transfers', 'unknown_transfers'))
            assert states['after']['transport']['registered_regions'] == 1
            assert states['after']['pending_release_entries'] == 0
            timing_before = states['warmed']['transport']['submit_timing']
            timing_after = states['after']['transport']['submit_timing']
            proof[arm].append(dict(rank=rank, enabled=optimized, formal_deliveries=len(profiles),
                actual_batch_submissions=batches, actual_slices=slices,
                formal_bytes=sum(item['nbytes'] for item in profiles),
                submit_timing_delta={name: timing_after[name] - timing_before[name]
                                     for name in timing_after},
                retirement_proven=True,
                timing_scope='actual native adapter counters during both formal requests, including initial full-KV fan-in'))
    return proof


def gate_proof(gate_files, sources, read_json, unique_file, *, gpu_bank=False):
    bundle = unique_file(gate_files, 'deployed.tar.gz')
    with tarfile.open(fileobj=io.BytesIO(bundle), mode='r:gz') as archive:
        deployed = {item.name: hashlib.sha256(archive.extractfile(item).read()).hexdigest()
                    for item in archive if item.isfile()}
    assert deployed == read_json(gate_files, 'local_source_hashes.json')
    for role in ('v', 'd'):
        assert read_json(gate_files, role + '_source_hashes.json') == deployed
        assert all(deployed.get(name) == digest for name, digest in sources[role].items())
    assert read_json(gate_files, 'unit_status.json') == {'exit_code': 0}
    assert read_json(gate_files, 'native_status.json') == {'exit_code': 0}
    assert read_json(gate_files, 'owned.json') == {}
    assert read_json(gate_files, 'cleanup_errors.json') == []
    for role in ('p', 'v', 'd'):
        for phase in ('initial', 'final'):
            text = unique_file(gate_files, role + '_' + phase + '_gpu.txt').decode()
            assert len(text.splitlines()) == 2
            assert all(int(line.split(',')[1].split()[0]) == 0 for line in text.splitlines())
    native_proof(read_json(gate_files, 'native.json'), expected_mode=(
        'gpu_receive_to_bank_with_async_backup' if gpu_bank else 'direct_sparse_batch_put'))
    return deployed
