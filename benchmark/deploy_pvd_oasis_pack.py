"""Isolated V deploy, source proof, regression and idle two-GPU byte gates."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True)
a = p.parse_args()
if not re.fullmatch(r'[a-z0-9_]+', a.tag):
    p.error('fresh filename-safe lowercase tag required')
root = Path(__file__).resolve().parents[1]
out = root / 'artifacts' / a.tag
out.mkdir(parents=True, exist_ok=False)
remote = '/proj/llm-course-PG0/Yizhzhu-node1-sglang-pvd/validation/pvd-oasis-v-search-20261002'
ssh = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3',
       '-o', 'HostKeyAlias=clgpu021.clemson.cloudlab.us', '-i',
       '/home/loosp/.ssh/cloudlab_pub_wsl', 'Yizhzhu@130.127.134.35']

def call(command, *, data=None, check=True, timeout=180):
    r = subprocess.run(ssh + [command], input=data, capture_output=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(r.stderr.decode(errors='replace') + r.stdout.decode(errors='replace'))
    return r

modules = ('cagra_backend.py', 'prompt_index.py', 'control_server.py', 'vector_store.py',
           'server.py', 'cagra_kv_update.py', 'cagra_kv_prepare.py', 'cagra_search_batch.py',
           'index_search.py', 'sparse_copy.py', 'sparse_pack_plan.py', 'sparse_payload.py',
           'sparse_delivery.py', 'sparse_receiver.py', 'triton_sparse_pack.py')
source_names = ['python/sglang/srt/disaggregation/pvd/' + n for n in modules]
tests = ['sparse_copy', 'sparse_pack_plan', 'cuda_sparse_packing', 'sparse_store_delivery',
         'sparse_delivery', 'sparse_payload', 'sparse_receiver']
test_names = ['test/registered/disaggregation/test_pvd_' + n + '.py' for n in tests]
names = source_names + test_names + ['benchmark/probe_pvd_oasis_sparse_pack.py',
         'test/registered/disaggregation/run_pvd_sparse_pack_gpu_smoke.py']
initial = call('nvidia-smi --query-gpu=index,memory.used --format=csv,noheader').stdout.decode()
(out / 'initial_gpu.txt').write_text(initial)
if any(int(line.split(',')[1].split()[0]) for line in initial.splitlines()):
    raise RuntimeError('V GPUs occupied; no deployment or GPU test started')
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w:gz') as archive:
    for name in names:
        data = (root / name).read_bytes().replace(b'\r\n', b'\n')
        entry = tarfile.TarInfo(name); entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
(out / 'deployed.tar.gz').write_bytes(buf.getvalue())
call('tar -xzf - -C ' + remote, data=buf.getvalue())
hashes = {}
for line in call('sha256sum ' + ' '.join(remote + '/' + n for n in names)).stdout.decode().splitlines():
    digest, path = line.split(None, 1)
    name = path.removeprefix(remote + '/')
    assert digest == hashlib.sha256((root / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest(), name
    hashes[name] = digest
(out / 'source_hashes.json').write_text(json.dumps(hashes, indent=2))
(out / 'remote_head.txt').write_bytes(call('git -C ' + remote + ' rev-parse HEAD').stdout)

env = ('source /users/Yizhzhu/.sglang-v100-pvd-env.sh; '
       'export PYTHONPATH=' + remote + '/python:$SGLANG_PVD_ROOT/deps/pvd-cagra-joint25:'
       '$SGLANG_PVD_ROOT/deps/pvd-cagra-extend25-venv/lib/python3.12/site-packages; '
       'task_site="$SGLANG_PVD_ROOT/deps/pvd-cagra-extend25-venv/lib/python3.12/site-packages"; '
       'task_nv="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"; '
       'export LD_LIBRARY_PATH="$task_nv/cublas/lib:$task_nv/cusolver/lib:$task_nv/cusparse/lib:'
       '$task_nv/nvjitlink/lib:$task_nv/cuda_runtime/lib:$task_site/libcuvs/lib64:'
       '$task_site/libraft/lib64:${LD_LIBRARY_PATH:-}"; cd ' + remote + '; ')

def run(label, command):
    filename = '/tmp/' + a.tag + '_' + label + '.txt'
    full = env + command + ' > ' + shlex.quote(filename) + ' 2>&1'
    (out / (label + '.command')).write_text(full)
    r = call('bash -c ' + shlex.quote(full), check=False, timeout=240)
    (out / (label + '_ssh_stderr.txt')).write_bytes(r.stderr)
    (out / (label + '_status.json')).write_text(json.dumps({'exit_code': r.returncode}))
    data = gzip.decompress(call('gzip -c -- ' + shlex.quote(filename)).stdout)
    (out / (label + '.txt')).write_bytes(data)
    print(label, 'exit', r.returncode, 'bytes', len(data), flush=True)
    if r.returncode:
        raise RuntimeError(label + ' failed; complete output preserved')
    return data

try:
    run('unit', '"$CONDA_PREFIX/bin/python" -m pytest -q ' + ' '.join(test_names))
    for device, current in ((0, 1), (1, 0)):
        data = run('smoke' + str(device), '"$CONDA_PREFIX/bin/python" '
                   'test/registered/disaggregation/run_pvd_sparse_pack_gpu_smoke.py '
                   f'--device cuda:{device} --current-device cuda:{current}')
        result = json.loads(data)
        assert len(result['checks']) == 12 and result['current_device'] == current
    data = run('probe', '"$CONDA_PREFIX/bin/python" benchmark/probe_pvd_oasis_sparse_pack.py '
               '--devices 0,1 --page-sizes 1,2 --warmup 2 --repeats 10')
    result = json.loads(data)
    assert result['status'] == 'passed' and len(result['devices']) == 4
    (out / 'probe.json').write_text(json.dumps(result, indent=2))
finally:
    final = call('nvidia-smi --query-gpu=index,memory.used --format=csv,noheader').stdout.decode()
    (out / 'final_gpu.txt').write_text(final)
    if any(int(line.split(',')[1].split()[0]) for line in final.splitlines()):
        raise RuntimeError('native gate GPU ownership remains; do not start serving trial')
print('native sparse packing gates passed; both V GPUs empty', flush=True)
