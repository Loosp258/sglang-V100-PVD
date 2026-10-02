"""Freeze and qualify one isolated delivery implementation before live ABBA.

WSL only. All outputs stay in checkout/artifacts; only recorded process groups
are stopped. No Git push. Native and CPU gates are separate observations.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
HOSTS = {'p': ('130.127.134.34', 'clgpu020.clemson.cloudlab.us', 0),
         'v': ('130.127.134.35', 'clgpu021.clemson.cloudlab.us', 1),
         'd': ('130.127.134.33', 'clgpu019.clemson.cloudlab.us', 2)}
CHECKOUTS = {
 'p': '/proj/llm-course-PG0/Yizhzhu-node0-sglang-pvd/validation/pvd-direct-20260929',
 'v': '/proj/llm-course-PG0/Yizhzhu-node1-sglang-pvd/validation/pvd-oasis-v-search-20261002',
 'd': '/proj/llm-course-PG0/Yizhzhu-node2-sglang-pvd/validation/pvd-oasis-alignment-20261002'}


def ssh(role, command, *, data=None, timeout=60, binary=False):
    ip, host, _ = HOSTS[role]
    result = subprocess.run(['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes',
        '-o', 'IPQoS=none', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=10',
        '-o', 'ServerAliveCountMax=3', '-o', 'HostKeyAlias=' + host,
        '-i', '/home/loosp/.ssh/cloudlab_pub_wsl', 'Yizhzhu@' + ip, command],
        input=data, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(role + ': ' + result.stdout.decode(errors='replace')
                           + result.stderr.decode(errors='replace'))
    return result.stdout if binary else result.stdout.decode()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--native', required=True, help='project-relative benchmark Python')
    parser.add_argument('--tests', nargs='+', required=True, help='CPU test filenames')
    args = parser.parse_args()
    assert re.fullmatch('[a-z0-9_]+', args.tag)
    native = (ROOT / args.native).resolve()
    assert native.is_file() and native.is_relative_to(ROOT / 'benchmark')
    out = ROOT / 'artifacts' / args.tag
    out.mkdir(exist_ok=False)
    pid = None
    errors = []
    try:
        for role in HOSTS:
            memory = ssh(role, 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader')
            (out / f'{role}_initial_gpu.txt').write_text(memory)
            assert all(int(line.split(',')[1].split()[0]) == 0 for line in memory.splitlines()), role
        paths = set((ROOT / 'python/sglang/srt/disaggregation/pvd').glob('*.py'))
        paths.update((ROOT / 'test/registered/disaggregation').glob('test_pvd*.py'))
        paths.update(ROOT / name for name in (
            'python/sglang/srt/server_args.py', 'python/sglang/srt/models/qwen2.py',
            'python/sglang/srt/managers/scheduler.py',
            'python/sglang/srt/managers/scheduler_components/batch_result_processor.py',
            'python/sglang/srt/disaggregation/decode.py',
            'test/registered/disaggregation/conftest.py',
            'test/registered/disaggregation/cloudlab_pvd_new_lease.sh'))
        paths.add(native)
        if args.native in ('benchmark/pvd_oasis_gpu_bank_native.py', 'benchmark/pvd_oasis_stages_native.py',
                           'benchmark/pvd_oasis_workspace_native.py'):
            paths.add(ROOT / 'benchmark/pvd_direct_sparse_batch_native.py')
        hashes = {}
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
            for path in sorted(paths):
                if not path.is_file():
                    continue
                name = path.relative_to(ROOT).as_posix()
                data = path.read_bytes().replace(b'\r\n', b'\n')
                hashes[name] = hashlib.sha256(data).hexdigest()
                item = tarfile.TarInfo(name)
                item.size = len(data)
                archive.addfile(item, io.BytesIO(data))
        (out / 'deployed.tar.gz').write_bytes(buffer.getvalue())
        save_json(out / 'local_source_hashes.json', hashes)
        for role in ('v', 'd'):
            remote = CHECKOUTS[role]
            remote_out = remote + '/artifacts/' + args.tag
            ssh(role, 'test ! -e ' + shlex.quote(remote_out) + ' && mkdir -p '
                + shlex.quote(remote_out + '/tmp'))
            ssh(role, 'tar -xzf - -C ' + shlex.quote(remote), data=buffer.getvalue())
            output = ssh(role, 'cd ' + shlex.quote(remote) + ' && sha256sum '
                         + ' '.join(shlex.quote(name) for name in hashes))
            actual = {line.split(None, 1)[1]: line.split(None, 1)[0] for line in output.splitlines()}
            save_json(out / f'{role}_source_hashes.json', actual)
            assert actual == hashes, role
        remote = CHECKOUTS['d']
        remote_out = remote + '/artifacts/' + args.tag
        python = '/proj/llm-course-PG0/Yizhzhu-node2-sglang-pvd/conda-envs/sglang-v100/bin/python'
        tests = ['test/registered/disaggregation/' + name for name in args.tests]
        assert all(re.fullmatch(r'test_pvd[a-z0-9_]+\.py', name) for name in args.tests)
        commands = {
            'unit': 'CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python '
                    + shlex.quote(python) + ' -B -m pytest -q ' + ' '.join(map(shlex.quote, tests)),
            'native': shlex.quote(python) + ' -B ' + shlex.quote(args.native)
                    + ' --device cuda:1 --current-device cuda:0 --hostname 10.10.1.3'
                    + ' --rails mlx5_0 mlx5_0 --output ' + shlex.quote(remote_out + '/native.json')}
        save_json(out / 'configuration.json', vars(args))
        for kind, command in commands.items():
            script = ('#!/usr/bin/env bash\nset +e\nsource /users/Yizhzhu/.sglang-v100-pvd-env.sh\n'
                      'cd ' + shlex.quote(remote) + '\nexport TMPDIR=' + shlex.quote(remote_out + '/tmp')
                      + '\n' + command + '\nstatus=$?\nprintf \'{"exit_code":%s}\\n\' "$status" > '
                      + shlex.quote(remote_out + '/' + kind + '_status.json') + '\nexit "$status"\n')
            ssh('d', 'cat > ' + shlex.quote(remote_out + '/' + kind + '.sh'), data=script.encode())
            launched = ssh('d', 'nohup setsid bash ' + shlex.quote(remote_out + '/' + kind + '.sh')
                           + ' > ' + shlex.quote(remote_out + '/' + kind + '.txt') + ' 2>&1 < /dev/null & echo $!')
            pid = int(launched.strip())
            save_json(out / 'owned.json', {'d': pid, 'stage': kind})
            deadline = time.monotonic() + 600
            while True:
                exists = ssh('d', 'test -f ' + shlex.quote(remote_out + '/' + kind + '_status.json')
                             + ' && echo done || true').strip()
                if exists:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(kind + ' gate timeout; recorded PGID will be stopped')
                time.sleep(2)
            text = ssh('d', 'cat ' + shlex.quote(remote_out + '/' + kind + '.txt'))
            (out / (kind + '.txt')).write_text(text)
            status = json.loads(ssh('d', 'cat ' + shlex.quote(remote_out + '/' + kind + '_status.json')))
            save_json(out / (kind + '_status.json'), status)
            ssh('d', f'kill -TERM -- -{pid} 2>/dev/null || true')
            pid = None
            save_json(out / 'owned.json', {})
            print(kind, status, text[-1800:], flush=True)
            if kind == 'native':
                observed = ssh('d', 'cat ' + shlex.quote(remote_out + '/native.json') + ' 2>/dev/null || true')
                if observed:
                    (out / 'native.json').write_text(observed)
            assert status == {'exit_code': 0}, kind
    finally:
        if pid is not None:
            try:
                ssh('d', f'kill -TERM -- -{pid} 2>/dev/null || true')
                save_json(out / 'owned.json', {})
            except Exception as error:
                errors.append(str(error))
        for role in HOSTS:
            try:
                for _ in range(30):
                    memory = ssh(role, 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader')
                    if all(int(line.split(',')[1].split()[0]) == 0 for line in memory.splitlines()):
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError('owned gate did not drain GPUs: ' + role)
                (out / f'{role}_final_gpu.txt').write_text(memory)
            except Exception as error:
                errors.append(str(error))
        save_json(out / 'cleanup_errors.json', errors)
        if errors:
            raise RuntimeError(errors)


if __name__ == '__main__':
    main()
