"""WSL helper: upload the isolated checkout's changed sources for validation."""
import io
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/proj/llm-course-PG0/Yizhzhu-node2-sglang-pvd/validation/pvd-oasis-alignment-20261002'
SSH = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o',
       'HostKeyAlias=clgpu019.clemson.cloudlab.us', '-i', '/home/loosp/.ssh/cloudlab_pub_wsl',
       'Yizhzhu@130.127.134.33']
GIT = ['git.exe', '-C', 'D:/code/sglang-V100-PVD-oasiskv']
paths = set(subprocess.check_output(GIT + ['diff', '53adb880d', '--name-only'], text=True).splitlines())
paths.update(subprocess.check_output(GIT + ['ls-files', '--others', '--exclude-standard'], text=True).splitlines())
paths.update(subprocess.check_output(GIT + ['ls-files', 'python/sglang/srt/disaggregation/pvd/oasis*',
    'test/registered/disaggregation/*oasis*', 'python/sglang/srt/models/qwen2.py'], text=True).splitlines())
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w') as archive:
    for name in sorted(paths):
        path = ROOT / name
        if path.is_file():
            data = path.read_bytes().replace(b'\r\n', b'\n')
            info = tarfile.TarInfo(name); info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
subprocess.run(SSH + ['tar -xf - -C ' + REMOTE], input=buffer.getvalue(), check=True)
print('uploaded', len(paths), 'files', flush=True)
command = ('cd ' + REMOTE + '; PYTHONPATH=python '
    '/proj/llm-course-PG0/Yizhzhu-node2-sglang-pvd/conda-envs/sglang-v100/bin/python -m pytest -q '
    'test/registered/disaggregation/test_pvd_oasis_attention.py '
    'test/registered/disaggregation/test_pvd_oasis_request.py '
    'test/registered/disaggregation/test_pvd_oasis_pipeline.py '
    'test/registered/disaggregation/test_pvd_oasis_serving.py '
    'test/registered/disaggregation/test_pvd_prompt_index.py '
    'test/registered/disaggregation/test_pvd_prompt_chunks.py '
    '> /tmp/oasis-serving-unit.log 2>&1 && cat /tmp/oasis-serving-unit.log')
subprocess.run(SSH + [command], check=True)
