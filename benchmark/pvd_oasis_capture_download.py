"""Recover completed diagnostic artifacts in bounded SSH chunks, without GPU work."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
SSH = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3',
       '-o', 'IPQoS=none',
       '-o', 'HostKeyAlias=clgpu019.clemson.cloudlab.us', '-i', '/home/loosp/.ssh/cloudlab_pub_wsl',
       'Yizhzhu@130.127.134.33']
CHECKOUT = '/proj/llm-course-PG0/Yizhzhu-node2-sglang-pvd/validation/pvd-oasis-alignment-20261002'


def call(command, timeout=90):
    result = subprocess.run(SSH + [command], capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    return result.stdout.decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9_]+', args.tag):
        raise ValueError('filename-safe existing tag required')
    out = (ROOT / 'artifacts' / args.tag).resolve()
    assert out.is_relative_to(ROOT / 'artifacts') and out.is_dir()
    assert json.loads((out / 'owned.json').read_text()) == {}, 'stop owned services before artifact recovery'
    remote = CHECKOUT + '/artifacts/' + args.tag
    archive = remote + '/capture.tar.gz'
    started = time.time()
    call('test -s ' + shlex.quote(archive) + ' || tar -czf ' + shlex.quote(archive) + ' -C ' + shlex.quote(remote) + ' capture')
    proof = call('sha256sum ' + shlex.quote(archive)).split()[0]
    size = int(call('stat -c %s ' + shlex.quote(archive)).strip())
    chunks = out / 'download_parts_1m'
    chunks.mkdir(exist_ok=False)
    chunk_bytes = 1 << 20
    count = (size + chunk_bytes - 1) // chunk_bytes
    records = []
    # Reuse full MiB ranges already received, including complete prefixes of
    # interrupted reads. Final archive SHA-256 is the acceptance proof.
    old_parts = out / 'download_parts'
    for path in sorted(old_parts.glob('*.part')):
        old_index = int(path.stem)
        with path.open('rb') as source:
            for subindex in range(path.stat().st_size // chunk_bytes):
                index = old_index * 4 + subindex
                if index >= count:
                    break
                data = source.read(chunk_bytes)
                destination = chunks / f'{index:04d}.part'
                destination.write_bytes(data)
                records.append(dict(index=index, source='local prefix of prior 4 MiB range; final SHA required',
                    exit_code=0, bytes=len(data), expected_bytes=min(chunk_bytes, size - index * chunk_bytes)))

    def download(index):
        path = chunks / f'{index:04d}.part'
        command = 'dd if=' + shlex.quote(archive) + f' bs={chunk_bytes} skip={index} count=1 status=none'
        tick = time.time()
        with path.open('wb') as stream:
            result = subprocess.run(SSH + [command], stdout=stream, stderr=subprocess.PIPE, timeout=120)
        (chunks / f'{index:04d}.stderr').write_bytes(result.stderr)
        expected = min(chunk_bytes, size - index * chunk_bytes)
        record = dict(index=index, exit_code=result.returncode, bytes=path.stat().st_size,
                      expected_bytes=expected, started_unix=tick, finished_unix=time.time())
        (chunks / f'{index:04d}.json').write_text(json.dumps(record))
        if result.returncode or path.stat().st_size != expected:
            raise RuntimeError('incomplete artifact chunk: ' + str(index))
        return record

    # Read-only independent ranges; never concurrent GPU experiments.
    with ThreadPoolExecutor(max_workers=3) as pool:
        done = {record['index'] for record in records}
        print('reused artifact ranges', len(done), '/', count, flush=True)
        futures = [pool.submit(download, index) for index in range(count) if index not in done]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print('artifact chunks', len(records), '/', count, flush=True)
    destination = out / 'capture.tar.gz'
    digest = hashlib.sha256()
    with destination.open('wb') as target:
        for index in range(count):
            with (chunks / f'{index:04d}.part').open('rb') as source:
                for data in iter(lambda: source.read(1 << 20), b''):
                    digest.update(data)
                    target.write(data)
    assert destination.stat().st_size == size and digest.hexdigest() == proof
    with tarfile.open(destination, 'r:gz') as source:
        for member in source:
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError('unexpected artifact entry type')
            path = (out / member.name).resolve()
            assert path.is_relative_to(out / 'capture')
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as contents, path.open('wb') as target:
                for data in iter(lambda: contents.read(1 << 20), b''):
                    target.write(data)
    result = dict(schema='pvd-ready-collection-recovery-v1',
        reason='single binary SSH capture collection timed out at 240 seconds after successful performance trials',
        collection_only=True, no_serving_restart=True, archive_bytes=size, archive_sha256=proof,
        chunks=count, chunk_bytes=chunk_bytes, chunk_workers=3, chunk_records=sorted(records, key=lambda r: r['index']),
        started_unix=started, finished_unix=time.time())
    (out / 'collection_recovery.json').write_text(json.dumps(result, indent=2) + '\n')
    print('artifact recovery verified', size, proof, flush=True)


if __name__ == '__main__':
    main()
