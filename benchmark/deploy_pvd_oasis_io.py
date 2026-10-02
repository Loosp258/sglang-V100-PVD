"""Deploy only reviewed request-scoped I/O changes to the isolated D checkout."""
import io
from pathlib import Path
import subprocess
import tarfile

root=Path(__file__).resolve().parents[1]
remote='/proj/llm-course-PG0/Yizhzhu-node2-sglang-pvd/validation/pvd-oasis-alignment-20261002'
ssh=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=15',
     '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3',
     '-o','HostKeyAlias=clgpu019.clemson.cloudlab.us','-i','/home/loosp/.ssh/cloudlab_pub_wsl',
     'Yizhzhu@130.127.134.33']
names=['python/sglang/srt/disaggregation/pvd/oasis_transport.py',
       'python/sglang/srt/disaggregation/pvd/oasis_startup.py',
       'test/registered/disaggregation/test_pvd_oasis_transport_io.py']
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as archive:
    for name in names:
        data=(root/name).read_bytes().replace(b'\r\n',b'\n')
        info=tarfile.TarInfo(name);info.size=len(data)
        archive.addfile(info,io.BytesIO(data))
subprocess.run(ssh+['tar -xzf - -C '+remote],input=buf.getvalue(),check=True)
tests=['oasis_attention','oasis_request','oasis_pipeline','oasis_serving',
       'oasis_transport_io','cagra_search_batch','cagra_backend','grouped_search_cache',
       'four_head_grouping','prompt_index','prompt_chunks','search_client','control_background_io']
command=('cd '+remote+'; PYTHONPATH=python '
         '/proj/llm-course-PG0/Yizhzhu-node2-sglang-pvd/conda-envs/sglang-v100/bin/python -m pytest -q '+
         ' '.join('test/registered/disaggregation/test_pvd_'+name+'.py' for name in tests)+
         ' > /tmp/oasis-io-unit.log 2>&1; task_unit_status=$?; '
         'cat /tmp/oasis-io-unit.log; exit "$task_unit_status"')
subprocess.run(ssh+[command],check=True)
