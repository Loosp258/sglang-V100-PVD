"""Verify portable evidence hashes, archive CRCs, completed work and cleanup."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
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
                assert [row['case'] for row in full['requests'][arm]] == cases
                for complete, compact in zip(full['requests'][arm], summary['requests'][arm]):
                    assert complete['rid'] == compact['rid']
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
