"""Two real CPU V servers/channels and native-policy copies through async D jobs.

CUDA ordering/copies and native PUT completion are explicit CPU policy doubles.
All selection, identity, packed FP16 bytes, slot leases and network I/O are real.
"""
import asyncio
import threading
import time
from types import SimpleNamespace

from aiohttp.test_utils import TestServer
import pytest
import torch

from sglang.srt.disaggregation.pvd import oasis_transport as oasis
from sglang.srt.disaggregation.pvd.control_server import HttpShardClient, create_shard_app
from sglang.srt.disaggregation.pvd.oasis_pipeline import LayerLookahead
from sglang.srt.disaggregation.pvd.protocol import KVEntryKey, KVEntryManifest
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_oasis_async_jobs import cuda_policy
from test_pvd_oasis_transport_io import background_io
from test_pvd_oasis_pinned_scratch import cpu_allocate
from test_pvd_prompt_index import make_store, manager, SPACE
from test_pvd_prompt_vectors import FakePool, storage_layout, pack_shard


@pytest.mark.parametrize('pinned',[False,True])
@pytest.mark.parametrize('async_cuda',[False,True])
def test_actual_async_two_rank_pipeline_and_slot_budget_with_blocked_acks(monkeypatch,background_io,pinned,async_cuda):
    # CPU byte qualification only: avoid massive tiny-op thread-pool overhead.
    prior_threads=torch.get_num_threads();torch.set_num_threads(1)
    source=FakePool(layers=28,heads=4,dim=128)
    layout=storage_layout(source)
    entry=KVEntryManifest(KVEntryKey.new('model','async'),layout,8,
        [pack_shard(source,layout,rank=r,prompt_tokens=8)[1] for r in (0,1)])
    stores=[]
    for rank in (0,1):
        store=make_store(entry,index=manager(metric='ip'),rank=rank)
        store.transfer_engine.lifecycle_manager=SimpleNamespace(budget=TransferBudget(1<<22,16))
        stored=store.create_entry(entry);store.begin_p_write(entry.key)
        packed,shard,_=pack_shard(source,layout,rank=rank,prompt_tokens=8)
        offset=stored.allocation.start_page*store.page_bytes
        store.pool[offset:offset+shard.expected_bytes]=packed.tensor
        store.commit_p_write(entry.key,shard.expected_bytes);store.progress_prompt_indexes()
        stores.append(store)
    servers=[]
    finishing=True
    async def complete_puts():
        while finishing:
            for store in stores:
                with store._lock:
                    deliveries=tuple(store.entries[entry.key].deliveries.values())
                for delivery in deliveries:
                    handle=delivery.transfer_handle
                    if handle is not None and handle.transfer_id in store.transfer_engine.pending:
                        store.transfer_engine.finish(handle)
            await asyncio.sleep(0.001)
    async def start_servers():
        for store in stores:
            server=TestServer(create_shard_app(store));await server.start_server();servers.append(server)
        return asyncio.create_task(complete_puts())
    completion=background_io.submit(start_servers()).result(5)
    release=threading.Event()
    calls=[]
    original_ack=HttpShardClient.ack_delivery
    async def blocked_ack(client,*args):
        calls.append(('ack',threading.get_ident()))
        while not release.is_set():await asyncio.sleep(0.001)
        return await original_ack(client,*args)
    monkeypatch.setattr(HttpShardClient,'ack_delivery',blocked_ack)
    original_registry,original_pool=oasis.OasisCUDAReceiveRegistry,oasis.OasisReceiveSlotPool
    cuda_policy(monkeypatch)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    def pool_factory(engine,budget,**kwargs):
        value=original_pool(engine,budget,**{**kwargs,'device':'cuda:0'})
        value.device=torch.device('cpu')
        return value
    def registry_factory(engine,budget,**kwargs):
        pool=kwargs.pop('receive_pool')
        value=original_registry(engine,budget,**{**kwargs,'device':'cuda:0','receive_pool':None})
        value.device=torch.device('cpu');value.receive_pool=pool
        value.ordering=SimpleNamespace(prepare=lambda _:None,after_remote_write=lambda _:None)
        return value
    monkeypatch.setattr(oasis,'OasisReceiveSlotPool',pool_factory)
    monkeypatch.setattr(oasis,'OasisCUDAReceiveRegistry',registry_factory)
    resources=SimpleNamespace(control=background_io,worker_epoch='D-async',
        transfer_budget=TransferBudget(1<<22,16),sparse_receive_engine=stores[0].transfer_engine)
    monkeypatch.setattr(resources.sparse_receive_engine,'health',lambda:{'healthy':True,'session_id':'D'})
    routes=tuple(SimpleNamespace(rank=s.rank,rail=s.rail,sender_epoch=s.worker_epoch,url=str(h.make_url('')))
        for s,h in zip(stores,servers,strict=True))
    owner=oasis.OasisLayerTransport(resources,SimpleNamespace(manifest=entry,shards=routes),
        request_id='req',incarnation='inc',device='cpu',vector_space=SPACE,capacity=4,max_new=4,top_k=2,
        timeout=5,ready_before_cleanup=True,binary_queries=True,fused_search_delivery=True,
        binary_control_channel=True,compact_cache_snapshots=True,fused_zero_miss_proof=True,
        reuse_receive_slots=True,reuse_pinned_scratch=pinned,event_bank_ready=pinned,async_layer_jobs=True,
        async_cuda_completion=async_cuda)
    if pinned:monkeypatch.setattr(owner.pinned_pool,'_allocate',lambda:cpu_allocate(4,2 if async_cuda else 1))
    lookahead=LayerLookahead('req','inc',layers=28,timeout=5)
    banks=[]
    try:
        for layer in range(4):
            query=torch.cat([source.k_buffer[layer][3,h].float().repeat(7,1) for h in range(4)])
            lookahead.publish(0,layer,owner.job(query,None))
            bank=lookahead.consume(0,layer);banks.append(bank)
            for head,ids in enumerate(bank.ids):
                assert torch.equal(bank.keys[head,:len(ids)],source.k_buffer[layer][list(ids),head])
                assert torch.equal(bank.values[head,:len(ids)],source.v_buffer[layer][list(ids),head])
        assert owner.async_jobs.snapshot()['cleanup_peak']==2 and len(owner.workers)==4
        assert owner.receive_pool.snapshot()['leased_slots']==8
        assert owner.receive_pool.snapshot()['physical_register_calls']==8
        release.set()
        deadline=time.monotonic()+5
        while owner.async_jobs.snapshot()['live']:
            assert time.monotonic()<deadline;time.sleep(0.001)
        before=sum(s.transfer_engine.total_put_bytes for s in stores)
        for layer in range(4):
            query=torch.cat([source.k_buffer[layer][3,h].float().repeat(7,1) for h in range(4)])
            lookahead.publish(1,layer,owner.job(query,banks[layer]))
            assert lookahead.consume(1,layer).ids==banks[layer].ids
        lookahead.close();owner.close()
        assert sum(s.transfer_engine.total_put_bytes for s in stores)==before
        assert owner.receive_pool.snapshot()['physical_register_calls']==8
        assert owner.receive_pool.snapshot()['physical_release_calls']==8
        assert resources.transfer_budget.snapshot()['used_staging_bytes']==0
        assert resources.transfer_budget.snapshot()['used_inflight']==0
        assert owner.async_jobs.snapshot()['active_peak']<=2
        assert owner.async_jobs.snapshot()['cleanup_peak']==2
        assert len(calls)==8 and all(c[1]==owner.async_jobs.thread.ident for c in calls)
        assert not owner.workers and owner.closed
    finally:
        release.set();lookahead.close()
        if not owner.closed:owner.close()
        async def stop_servers():
            nonlocal finishing
            finishing=False;await completion
            for server in servers:await server.close()
        background_io.submit(stop_servers()).result(5)
        for store in stores:store.close()
        torch.set_num_threads(prior_threads)
