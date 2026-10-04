"""CPU packing and synchronous fake PUT byte oracle; no native/GPU latency claim."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import verify_pvd_selected_views_real_kv as replay
from sglang.srt.disaggregation.pvd.sparse_copy import copy_sparse_kv_into
from sglang.srt.disaggregation.pvd.sparse_source_slots import SparseSourceSlotPool
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine, MemorySlice
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

ROOT=Path(__file__).resolve().parents[1]


def verify(path):
    sources,known,jobs=replay.prepare(torch.load(path,map_location='cpu',weights_only=True))
    before=[replay.byte_digest(s) for s in sources]
    observations={}
    for mode in ('baseline','pooled'):
        engine=FakeTransferEngine();budget=TransferBudget(1<<20,8)
        pools={rank:SparseSourceSlotPool(engine,budget,device='cpu',rank=rank,rail=f'r{rank}',
            slots=2,capacity_bytes=32768) for rank in (0,1)}
        phases={};registers=allocates=0;peak=0
        for phase,items in jobs.items():
            digest=hashlib.sha256();manifests=hashlib.sha256();rows=views=0
            old_registers=registers;old_allocates=allocates
            pool_before={r:p.snapshot()['physical_register_calls'] for r,p in pools.items()}
            for index,job in enumerate(items):
                nbytes=job['manifest'].nbytes;rank=job['shard'].rank;lease=None
                owner=f'{mode}:{phase}:{index}'
                if mode=='pooled':
                    lease=pools[rank].acquire(owner,nbytes);staging=lease.buffer;local=lease.local
                else:
                    budget.reserve(owner,nbytes,0)
                    staging=torch.empty(nbytes,dtype=torch.uint8);allocates+=1
                    reg=engine.register_memory(staging,endpoint='source',rank=rank,rail=f'r{rank}');registers+=1
                    local=MemorySlice(reg,0,nbytes)
                peak=max(peak,budget.snapshot()['used_staging_bytes'])
                views+=copy_sparse_kv_into(job['source'],staging,manifest=job['manifest'],layout=job['layout'],
                    shard=job['shard'],entry_transfer_id='entry',index_version='index',id_mapping_version='map')
                target=engine.register_memory(job['target'],endpoint='D',rank=rank,rail=f'r{rank}')
                handle=engine.submit_put(local,target.descriptor)
                assert engine.cleanup_complete(handle) and handle.transferred_bytes==nbytes
                assert torch.equal(job['target'],job['oracle'])
                rows+=sum(len(s.token_ids) for s in job['manifest'].specs)
                digest.update(job['target'].numpy().tobytes());manifests.update(job['manifest'].fingerprint.encode())
                if lease:lease.release_after_proof()
                else:engine.release_memory(reg);budget.release(owner)
                engine.release_memory(target)
            cold=sum(p.snapshot()['physical_register_calls']-pool_before[r] for r,p in pools.items())
            phases[phase]=dict(deliveries=len(items),rows=rows,bytes=rows*512,
                wire_sha256=digest.hexdigest(),manifest_sha256=manifests.hexdigest(),source_component_views=views,
                physical_source_register_calls=cold if mode=='pooled' else registers-old_registers,
                physical_source_allocate_calls=cold if mode=='pooled' else allocates-old_allocates)
        preclose={r:p.snapshot() for r,p in pools.items()}
        for pool in pools.values():pool.close()
        assert budget.snapshot()['used_staging_bytes']==0
        observations[mode]=dict(phases=phases,peak_charged_source_bytes=peak,pools_before_close=preclose,
            pools_after_close={r:p.snapshot() for r,p in pools.items()},budget_after_close=budget.snapshot())
    for phase in jobs:
        base,new=[observations[m]['phases'][phase] for m in ('baseline','pooled')]
        for key in ('deliveries','rows','bytes','wire_sha256','manifest_sha256','source_component_views'):
            assert base[key]==new[key]
    assert before==[replay.byte_digest(s) for s in sources]
    return dict(case=int(path.parent.name),input_path=path.relative_to(ROOT).as_posix(),
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),consumed_banks=420,
        known_rows=int(known.sum()),uncaptured_poison_rows=known.numel()-int(known.sum()),
        source_bytes_unchanged=True,selected_wire_bytes_exact=True,observations=observations)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();capture,output=a.capture.resolve(),a.output.resolve()
    if not capture.is_relative_to(ROOT/'artifacts') or not output.is_relative_to(ROOT/'artifacts') or output.exists():
        raise ValueError('project-local captures and fresh output required')
    source_files=[Path(__file__),ROOT/'benchmark/verify_pvd_selected_views_real_kv.py']+[
        ROOT/'python/sglang/srt/disaggregation/pvd'/name for name in
        ('sparse_source_slots.py','sparse_copy.py','sparse_payload.py','sparse_delivery.py','protocol.py','transfer_engine.py','transfer_lifecycle.py')]
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in source_files}
    torch.set_num_threads(1)
    result=dict(schema='pvd-source-slots-real-kv-cpu-v1',cpu_only=True,native_gpu_decode_tested=False,
        fake_transport_is_explicit=True,source_is_declared_cpu_reconstruction=True,
        uncaptured_rows_poisoned_and_never_selected=True,latency_gain_claimed=False,
        cases=[verify(capture/str(case)/'trajectory.pt') for case in (99401,99402)],source_hashes=hashes)
    assert hashes=={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in source_files}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({c['case']:{m:{phase:o['physical_source_register_calls'] for phase,o in
        c['observations'][m]['phases'].items()} for m in ('baseline','pooled')} for c in result['cases']},indent=2))


if __name__=='__main__':main()
