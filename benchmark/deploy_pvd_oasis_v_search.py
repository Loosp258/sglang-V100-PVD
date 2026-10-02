"""Upload to the already-created isolated V search checkout, then unit/native gates."""
import io
import argparse
from pathlib import Path
import subprocess
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument('--focus-layer', type=int)
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
probe_tag = 'native' if args.focus_layer is None else 'focus'

root = Path(__file__).resolve().parents[1]
remote = '/proj/llm-course-PG0/Yizhzhu-node1-sglang-pvd/validation/pvd-oasis-v-search-20261002'
ssh = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
    '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3',
    '-o', 'HostKeyAlias=clgpu021.clemson.cloudlab.us', '-i', '/home/loosp/.ssh/cloudlab_pub_wsl',
    'Yizhzhu@130.127.134.35']
names = ['python/sglang/srt/disaggregation/pvd/' + n for n in
    ('prompt_index.py', 'server.py', 'cagra_search_batch.py')]
names += ['test/registered/disaggregation/' + n for n in
    ('test_pvd_cagra_search_batch.py', 'test_pvd_grouped_search_cache.py', 'test_pvd_cagra_kv_update.py')]
names += ['benchmark/pvd_oasis_partial_search_probe.py', 'benchmark/pvd_cagra_joint_manager_probe.py']
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w') as archive:
    for name in names:
        data = (root / name).read_bytes().replace(b'\r\n', b'\n')
        info = tarfile.TarInfo(name); info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
subprocess.run(ssh + ['tar -xf - -C ' + remote], input=buffer.getvalue(), check=True)
command = ('source /users/Yizhzhu/.sglang-v100-pvd-env.sh; '
    'export PYTHONPATH=' + remote + '/python:$SGLANG_PVD_ROOT/deps/pvd-cagra-joint25:'
    '$SGLANG_PVD_ROOT/deps/pvd-cagra-extend25-venv/lib/python3.12/site-packages; '
    'task_site="$SGLANG_PVD_ROOT/deps/pvd-cagra-extend25-venv/lib/python3.12/site-packages"; '
    'task_nv="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"; '
    'export LD_LIBRARY_PATH="$task_nv/cublas/lib:$task_nv/cusolver/lib:$task_nv/cusparse/lib:'
    '$task_nv/nvjitlink/lib:$task_nv/cuda_runtime/lib:$task_site/libcuvs/lib64:'
    '$task_site/libraft/lib64:${LD_LIBRARY_PATH:-}"; '
    'cd ' + remote + '; "$CONDA_PREFIX/bin/python" benchmark/pvd_oasis_partial_search_probe.py '
    '--fixtures /tmp/pvd_extend_real_direct_rank0.pt /tmp/pvd_extend_real_direct_rank1.pt '
    '--repeats ' + str(args.repeats) + (' --focus-layer ' + str(args.focus_layer) if args.focus_layer is not None else '') +
    ' --output /tmp/oasis-partial-search-' + probe_tag + '.json > /tmp/oasis-partial-search-' + probe_tag + '.log 2>&1; '
    'task_probe_status=$?; cat /tmp/oasis-partial-search-' + probe_tag + '.log; exit "$task_probe_status"')
subprocess.run(ssh + ['bash -c ' + __import__('shlex').quote(command)], check=True)
