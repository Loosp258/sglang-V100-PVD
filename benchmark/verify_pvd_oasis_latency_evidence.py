"""Verify portable evidence hashes, archive CRCs, completed work and cleanup."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


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
    if is_io or is_pack:
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
            if is_io or is_pack:
                # Keep this request snapshot in the compact summary as well as
                # the complete raw D trace. Counts prove actual reuse, not just
                # a launch flag or the presence of one shared client object.
                snapshot = row.get('io')
                assert isinstance(snapshot, dict), (arm, row['case'], 'missing IO snapshot')
                reuse = is_io and arm.startswith('opt')
                assert snapshot['reuse_io'] is reuse
                assert snapshot['manager_io_loop_reused'] is reuse
                assert snapshot['shared_close_submitted'] is reuse
                assert snapshot['closed'] is True and snapshot['closing'] is True
                for name in ('job_count', 'worker_loops_created',
                             'search_clients_created', 'control_clients_created',
                             'search_sessions_created', 'control_sessions_created'):
                    assert type(snapshot[name]) is int, (arm, row['case'], name)
                assert snapshot['job_count'] == snapshot['worker_loops_created'] == 420
                clients = 2 if reuse else 840
                assert snapshot['search_clients_created'] == clients
                assert snapshot['control_clients_created'] == clients
                assert snapshot['search_sessions_created'] == clients
                if reuse:
                    assert snapshot['control_sessions_created'] == 2
                else:
                    # Cache hits create no control HTTP session on that rank;
                    # baseline control clients are still created for every job.
                    assert 0 < snapshot['control_sessions_created'] <= 840
    assert all(all(int(line.split(',')[1].split()[0]) == 0 for line in text.splitlines())
               for text in json.loads((folder / 'final_gpu_memory.json').read_text()).values())
    with tarfile.open(folder / 'raw.tar.gz', 'r:gz') as archive:
        names = {member.name: member for member in archive if member.isfile()}
        owned = next(name for name in names if name.endswith('/owned.json'))
        cleanup = next(name for name in names if name.endswith('/cleanup_errors.json'))
        assert json.loads(archive.extractfile(names[owned]).read()) == {}
        assert json.loads(archive.extractfile(names[cleanup]).read()) == []
    return dict(passed=True, files=len(manifest), formal_requests=8, git_blobs_checked=check_git)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--check-git', action='store_true')
    a = p.parse_args()
    print(json.dumps(verify(a.directory, check_git=a.check_git)))
