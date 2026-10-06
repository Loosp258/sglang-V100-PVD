"""Deterministic CPU event policy and a separately skipped real CUDA copy gate."""
import asyncio
import contextlib
import threading
from types import SimpleNamespace
import pytest
import torch
from sglang.srt.disaggregation.pvd.async_cuda_completion import wait_local_cuda_event
from sglang.srt.disaggregation.pvd.oasis_pinned_scratch import PinnedScratchPool,scratch_bytes
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_oasis_pinned_scratch import cpu_allocate


@pytest.mark.parametrize('cancel',[False,True])
def test_pending_event_yields_other_work_and_cancel_joins(monkeypatch,cancel):
    monkeypatch.setattr(torch.cuda,'device',lambda _:contextlib.nullcontext())
    async def run():
        release=asyncio.Event();queries=[]
        event=SimpleNamespace(query=lambda:queries.append(threading.get_ident()) or release.is_set())
        task=asyncio.create_task(wait_local_cuda_event(event,'cuda:0'))
        await asyncio.sleep(.003)
        assert queries and not task.done()
        if cancel:
            task.cancel();await asyncio.sleep(.003)
            assert not task.done()
        release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):await task
        else:await task
        assert set(queries)=={threading.get_ident()}
    asyncio.run(run())


def test_completed_scratch_two_rank_regions_are_separate_and_charged(monkeypatch):
    pool=PinnedScratchPool(TransferBudget(1<<20,4),capacity=4,slots=1,receive_ranks=2)
    monkeypatch.setattr(pool,'_allocate',lambda:cpu_allocate(4,2))
    lease=pool.acquire();lease.begin()
    first,second=lease.receive[:4096],lease.receive[4096:]
    first.fill_(17);second.fill_(29)
    assert first.numel()==second.numel()==4096 and first.data_ptr()!=second.data_ptr()
    assert torch.all(first==17) and torch.all(second==29)
    assert pool.budget.snapshot()['used_staging_bytes']==scratch_bytes(4,2)
    lease.finish_completed(SimpleNamespace(query=lambda:True))
    pool.close();assert pool.budget.snapshot()['used_staging_bytes']==0


def test_async_cache_copy_does_not_install_or_ack_until_event(monkeypatch):
    from test_pvd_oasis_receive_slot_records import slot_case,prepare
    from test_pvd_oasis_async_jobs import cuda_policy
    async def run():
        async with slot_case(monkeypatch) as c:
            cuda_policy(monkeypatch)
            record=prepare(c)
            assert not await record.start()
            delivery=c.store.entries[c.entry.key].deliveries[record.identity.transfer_id]
            c.engine.finish(delivery.transfer_handle)
            assert await record.poll()
            release=asyncio.Event()
            class Event:
                def record(self):pass
                def query(self):return release.is_set()
            monkeypatch.setattr(torch.cuda,'Event',Event)
            cache=[{} for _ in range(4)]
            task=asyncio.create_task(record.copy_to_cache_async(cache,stream=SimpleNamespace(synchronize=lambda:None)))
            await asyncio.sleep(.003)
            assert not record._installed and not any(cache)
            assert c.pool.snapshot()['leased_slots']==1 and record._cache_copy_owners
            release.set();await task
            assert record._installed and not record._cache_copy_owners
            for spec in record.manifest.specs:
                for token in spec.token_ids:
                    assert torch.equal(cache[spec.kv_head][token],torch.stack((c.source.k_buffer[0][token,spec.kv_head],c.source.v_buffer[0][token,spec.kv_head])))
            await record.ack();assert await record.close()
    asyncio.run(run())


def test_async_cuda_config_requires_async_owner(monkeypatch,tmp_path):
    from test_pvd_delivery_followup_config import load,config
    cfg=config(5)
    cfg['async_cuda_completion']=True
    with pytest.raises(ValueError,match='async layer jobs'):load(monkeypatch,tmp_path,cfg)
    cfg.update(async_layer_jobs=True,ready_before_cleanup=True,request_scratch_bytes=2<<20)
    assert load(monkeypatch,tmp_path,cfg)['async_cuda_completion'] is True


def test_unknown_event_keeps_scratch_and_budget(monkeypatch):
    pool=PinnedScratchPool(TransferBudget(1<<20,4),capacity=4,receive_ranks=2)
    monkeypatch.setattr(pool,'_allocate',lambda:cpu_allocate(4,2))
    lease=pool.acquire();lease.begin()
    def unknown():raise RuntimeError('CUDA event query failed')
    with pytest.raises(RuntimeError):lease.finish_completed(SimpleNamespace(query=unknown))
    assert lease.active and pool.snapshot()['unknown']
    assert pool.budget.snapshot()['used_staging_bytes']==scratch_bytes(4,2)
    with pytest.raises(RuntimeError):pool.close()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA event/copy gate requires GPU')
def test_real_cuda_d2h_event_owns_host_until_copy_complete():
    async def run():
        source=torch.arange(32768,device='cuda',dtype=torch.float32)
        host=torch.empty_like(source,device='cpu',pin_memory=True)
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):
            host.copy_(source,non_blocking=True);event=torch.cuda.Event();event.record()
        await wait_local_cuda_event(event,source.device)
        assert torch.equal(host,torch.arange(32768,dtype=torch.float32))
    asyncio.run(run())
