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
    for arm, rows in summary['requests'].items():
        assert len(rows) == 2
        for row in rows:
            assert row['completion_tokens'] == 16 and row['cached_tokens'] == 0
            assert row['trace_counts']['layers'] == 392
            assert row['trace_counts']['transport'] == 420
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
