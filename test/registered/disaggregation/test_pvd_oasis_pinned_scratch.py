"""CPU owner policy tests and separately skipped actual CUDA copy test."""
import contextlib
import threading
from types import SimpleNamespace
import pytest
import torch
from sglang.srt.disaggregation.pvd.oasis_pinned_scratch import PinnedScratchPool, scratch_bytes
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from sglang.srt.disaggregation.pvd.oasis_pipeline import LayerLookahead
from test_pvd_oasis_transport_io import transport, background_io


def cpu_allocate(capacity):
    # Explicit CPU policy; production allocator always requests pinned memory.
    return (torch.empty((28,128)),torch.empty((4,capacity,2,128),dtype=torch.float16),
            torch.empty(2*capacity*512,dtype=torch.uint8))


def test_reuse_and_pending_completion_keep_exact_byte_charge(monkeypatch):
    budget=TransferBudget(1<<20,4)
    pool=PinnedScratchPool(budget,capacity=32,slots=1)
    monkeypatch.setattr(pool,'_allocate',lambda:cpu_allocate(32))
    first=pool.acquire();address=first.query.data_ptr();first.begin()
    with pytest.raises(RuntimeError): pool.acquire()
    with pytest.raises(RuntimeError): pool.close()
    first.finish(SimpleNamespace(synchronize=lambda:None))
    second=pool.acquire()
    assert second.query.data_ptr()==address and pool.snapshot()['allocations']==1
    assert budget.snapshot()['used_staging_bytes']==scratch_bytes(32)
    second.abort(SimpleNamespace(synchronize=lambda:None))
    pool.close();assert budget.snapshot()['used_staging_bytes']==0


@pytest.mark.parametrize('phase',['event','stream'])
def test_unknown_local_completion_retains_scratch_and_budget(monkeypatch,phase):
    pool=PinnedScratchPool(TransferBudget(1<<20,4),capacity=4)
    monkeypatch.setattr(pool,'_allocate',lambda:cpu_allocate(4))
    lease=pool.acquire();lease.begin()
    def fail(): raise RuntimeError('CUDA completion unknown policy')
    with pytest.raises(RuntimeError):
        (lease.finish if phase=='event' else lease.abort)(SimpleNamespace(synchronize=fail))
    assert lease.active and pool.snapshot()['unknown'] and pool.snapshot()['live']==1
    with pytest.raises(RuntimeError): pool.close()
    with pytest.raises(RuntimeError): pool.acquire()
    assert pool.budget.snapshot()['used_staging_bytes']==scratch_bytes(4)


def test_full_job_publishes_event_bank_before_blocked_completion_and_reuses_scratch(monkeypatch,background_io):
    owner=transport(monkeypatch,background_io,reuse_io=False,reuse_pinned_scratch=True,event_bank_ready=True)
    monkeypatch.setattr(owner.pinned_pool,'_allocate',lambda:cpu_allocate(4))
    entered,release=threading.Event(),threading.Event()
    events=[]
    class Event:
        def __init__(self): events.append(self);self.ordinal=len(events)
        def record(self,*args): pass
        def synchronize(self):
            if self.ordinal==2:
                entered.set();assert release.wait(5)
    class Stream:
        def wait_event(self,event): pass
        def synchronize(self): pass
    monkeypatch.setattr(torch.cuda,'Stream',lambda **kwargs:Stream())
    monkeypatch.setattr(torch.cuda,'Event',Event)
    monkeypatch.setattr(torch.cuda,'current_stream',lambda *args:Stream())
    monkeypatch.setattr(torch.cuda,'device',lambda *args:contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda,'stream',lambda *args:contextlib.nullcontext())
    for head in range(4):
        owner.cache[0][head][1]=torch.full((2,128),head+1,dtype=torch.float16)
    async def fetch(state,*args):
        state.update(gpu_readers=[],fresh_gpu_rows={},delivery_profiles=[])
        return ((1,),(1,),(1,),(1,)),4,0
    monkeypatch.setattr(owner,'_select_and_fetch',fetch)
    lookahead=LayerLookahead('request','incarnation',layers=28,timeout=5)
    try:
        lookahead.publish(0,0,owner.job(torch.zeros((28,128)),None))
        bank=lookahead.consume(0,0)
        assert entered.wait(5) and bank.completion is events[1]
        assert owner.pinned_pool.snapshot()['live']==1
        assert torch.equal(bank.keys[:,0,0],torch.tensor([1,2,3,4],dtype=torch.float16))
        release.set()
        # Join the original owned cleanup before a sequential bootstrap job.
        owner.cleanup.close()
        from sglang.srt.disaggregation.pvd.oasis_ready_cleanup import OwnedReadyCleanup
        owner.cleanup=OwnedReadyCleanup(workers=2,capacity=56)
        lookahead.publish(1,0,owner.job(torch.ones((28,128)),bank))
        assert lookahead.consume(1,0).ids==bank.ids
        lookahead.close();owner.close()
        assert owner.pinned_pool.snapshot()['allocations']==1
        assert owner.pinned_pool.snapshot()['returned']==2
        assert owner.manager.transfer_budget.snapshot()['used_staging_bytes']==0
    finally:
        release.set();lookahead.close()
        if not owner.closed:owner.close()


def test_fused_receive_copy_uses_exact_prefix_of_existing_host_scratch(monkeypatch):
    import asyncio
    from test_pvd_oasis_receive_slot_records import slot_case, writer
    from test_pvd_fused_receive_slots import prepared
    from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient
    from sglang.srt.disaggregation.pvd.fused_search_delivery import start_fused
    async def run():
        async with slot_case(monkeypatch) as c:
            pool=PinnedScratchPool(c.budget,capacity=4)
            monkeypatch.setattr(pool,'_allocate',lambda:cpu_allocate(4))
            lease=pool.acquire();lease.begin();lease.receive.fill_(123)
            c.registry.pinned_scratch=lease
            search=PVDShardSearchClient(c.client.base_url,binary_queries=True)
            try:
                record,requests=prepared(c,search,'scratch-copy')
                _,ready=await start_fused(record,search,requests)
                assert not ready
                c.engine.finish(writer(c,record).transfer_handle)
                assert await record.poll()
                cache=[{},{}];record.copy_to_cache(cache)
                assert torch.equal(lease.receive[:record.manifest.nbytes],record._buffer)
                assert torch.all(lease.receive[record.manifest.nbytes:]==123)
                await record.ack();assert await record.close()
                lease.abort(SimpleNamespace(synchronize=lambda:None))
                pool.close()
            finally:
                await search.close()
    asyncio.run(run())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual pinned CUDA copy required')
def test_actual_cuda_pinned_copy_consumer_event_and_retirement():
    pool=PinnedScratchPool(TransferBudget(1<<20,4),capacity=4)
    lease=pool.acquire();lease.begin();lease.query.fill_(3)
    producer,consumer=torch.cuda.Stream(),torch.cuda.Stream()
    with torch.cuda.stream(producer):
        gpu=lease.query.to('cuda',non_blocking=True)+1
        event=torch.cuda.Event();event.record()
    with torch.cuda.stream(consumer):
        consumer.wait_event(event)
        output=gpu*2
    lease.finish(event)
    consumer.synchronize()
    assert torch.equal(output.cpu(),torch.full((28,128),8.0))
    pool.close()
