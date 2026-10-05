"""Real CPU index/HTTP/packing; delayed fake native PUT is explicit."""
import asyncio
from dataclasses import replace
import threading
import sys
from types import SimpleNamespace
import torch
import pytest
from sglang.srt.disaggregation.pvd.control_server import HttpShardClient
from sglang.srt.disaggregation.pvd.fused_search_delivery import (
    prepare_fused,start_fused,choose_wire,selection_digest,allocation_bytes)
from sglang.srt.disaggregation.pvd.oasis_transport import OasisCPUReceiveRegistry, OasisCUDAReceiveRegistry
from sglang.srt.disaggregation.pvd.oasis_transport import OasisLayerTransport, _HeadCPUCache
from sglang.srt.disaggregation.pvd.oasis_pipeline import LayerTicket
from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient,SearchScope
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_prompt_index import manager,stored_entry,ident,shard_client
from test_pvd_prompt_index import make_store
from test_pvd_prompt_vectors import pack_shard


def setup(rank):
    index=manager();store,manifest,pool,layout=stored_entry(index,rank=rank)
    store.transfer_engine.lifecycle_manager=SimpleNamespace(budget=TransferBudget(1<<20,4))
    store.progress_prompt_indexes()
    heads=[rank*2,rank*2+1]
    identities=[ident(manifest.key.transfer_id,kv_head=h) for h in heads]
    record=index._entries[manifest.key.transfer_id]
    queries=[record.vectors[(0,h)].vectors[3:4].tolist() for h in heads]
    scope=SearchScope(manifest.prompt_token_count,layout.page_size,layout.head_dim,'l2')
    requests=[(identity,q,2,scope) for identity,q in zip(identities,queries,strict=True)]
    selection=dict(request_id='request',incarnation='incarnation',operation_id='lookahead:0:0',
        target_tokens=1,entry_transfer_id=manifest.key.transfer_id,layout_fingerprint=layout.fingerprint,
        layer=0,heads=heads,capacity=4,max_new=4,prompt_tokens=manifest.prompt_token_count,
        dtype=layout.kv_dtype,head_dim=layout.head_dim,resident=[[],[]],cached=[[],[]])
    return index,store,manifest,pool,layout,requests,selection


async def prepared(store,manifest,requests,selection,http,*,cuda=False,binary=False):
    search=PVDShardSearchClient(str(http.make_url('')),binary_queries=binary)
    control=HttpShardClient(store.rank,str(http.make_url('')),timeout_seconds=1)
    batch=dict(batch_protocol='pvd.search.batch.v1',batch_id='batch',items=[
        search._prepare_search(identity,queries=q,top_k=k,scope=s)[0] for identity,q,k,s in requests])
    budget=TransferBudget(1<<20,4)
    registry=(OasisCUDAReceiveRegistry(store.transfer_engine,budget,receiver_epoch='D',device='cuda:0')
        if cuda else OasisCPUReceiveRegistry(store.transfer_engine,budget,receiver_epoch='D'))
    record=prepare_fused(registry,selection,batch,key=manifest.key,rank=store.rank,rail=store.rail,
        endpoint='D',sender_epoch=store.worker_epoch,client=control,binary_queries=binary)
    return search,control,budget,registry,record


@pytest.mark.parametrize('rank',[0,1])
@pytest.mark.parametrize('mode',['miss','mixed','hit'])
@pytest.mark.parametrize('binary',[False,True])
def test_both_logical_ranks_http_dynamic_prefix_exact_bytes_and_zero_miss(rank,mode,binary):
    async def run():
        _,store,manifest,pool,layout,requests,selection=setup(rank)
        if mode in ('mixed','hit'): selection['cached'][0]=list(range(manifest.prompt_token_count))
        if mode=='hit': selection['cached'][1]=list(range(manifest.prompt_token_count))
        async with shard_client(store) as http:
            search,control,budget,registry,record=await prepared(store,manifest,requests,selection,http,binary=binary)
            old=await search.search_many(requests)
            # Independent baseline policy uses original resident selector.
            from sglang.srt.disaggregation.pvd.oasis_pipeline import select_resident
            expected=tuple(select_resident(tuple(t for _,t in sorted(zip(r.scores,r.token_ids),reverse=True)),
                (),capacity=4,max_new=4) for r in old)
            chosen,ready=await start_fused(record,search,requests)
            assert chosen==expected and record.profile['fused_calls']==1
            physical=record._registration
            if mode=='hit':
                assert ready and not store.transfer_engine.pending
                assert record.fused_no_miss and await record.close()
            else:
                assert not ready
                assert record.manifest.nbytes < physical.descriptor.length
                assert record._registration is physical and record._buffer.data_ptr()==physical.buffer.data_ptr()
                delivery=store.entries[manifest.key].deliveries[record.identity.transfer_id]
                store.transfer_engine.finish(delivery.transfer_handle)
                assert await record.poll()
                cache=[{} for _ in range(4)]
                record.copy_to_cache(cache)
                for head in selection['heads']:
                    missing=[t for t in expected[head-rank*2] if t not in selection['cached'][head-rank*2]]
                    assert tuple(cache[head])==tuple(missing)
                    for token in missing:
                        oracle=torch.stack((pool.k_buffer[0][token,head],pool.v_buffer[0][token,head]))
                        assert torch.equal(cache[head][token],oracle)
                assert store.transfer_engine.total_put_bytes==record.manifest.nbytes
                assert record._registration is physical
                await record.ack();assert await record.close()
            assert not registry._records and budget.snapshot()['used_staging_bytes']==0
            await search.close();await control.close()
        store.close()
    asyncio.run(run())


@pytest.mark.parametrize('damage',['chosen','manifest','digest','short_terminal','failed_native'])
def test_bad_dynamic_reply_or_native_proof_never_installs_and_retains_live_writer(damage):
    async def run():
        _,store,manifest,_,_,requests,selection=setup(0)
        async with shard_client(store) as http:
            search,control,budget,registry,record=await prepared(store,manifest,requests,selection,http)
            original=search._post_json
            async def corrupt(path,payload,**kwargs):
                reply=await original(path,payload,**kwargs)
                if damage=='chosen': reply['chosen'][0]=[7]
                if damage=='manifest': reply['manifest']['specs'][0]['token_ids']=[7]
                if damage=='digest': reply['selection_digest']='bad'
                return reply
            search._post_json=corrupt
            if damage in ('chosen','manifest','digest'):
                with pytest.raises(ValueError): await start_fused(record,search,requests)
                assert not record._ready and not await record.close()
            else:
                chosen,ready=await start_fused(record,search,requests)
                assert not ready
            delivery=store.entries[manifest.key].deliveries[record.identity.transfer_id]
            store.transfer_engine.finish(delivery.transfer_handle,success=damage!='failed_native')
            if damage=='short_terminal': delivery.transfer_handle.transferred_bytes-=1
            if damage in ('short_terminal','failed_native'):
                with pytest.raises(ValueError): await record.poll()
            assert not record._installed and await record.close()
            assert not registry._records and budget.snapshot()['used_staging_bytes']==0
            await search.close();await control.close()
        store.close()
    asyncio.run(run())


@pytest.mark.skipif(not torch.cuda.is_available() or sys.platform!='linux',reason='Linux CUDA receive ordering required; fake PUT only')
def test_actual_cuda_dynamic_prefix_keeps_original_physical_registration_until_copy():
    async def run():
        _,store,manifest,pool,_,requests,selection=setup(0)
        async with shard_client(store) as http:
            search,control,budget,registry,record=await prepared(store,manifest,requests,selection,http,cuda=True)
            physical=record._registration
            chosen,ready=await start_fused(record,search,requests)
            assert not ready and record._registration is physical
            delivery=store.entries[manifest.key].deliveries[record.identity.transfer_id]
            store.transfer_engine.finish(delivery.transfer_handle)
            assert await record.poll()
            with torch.cuda.stream(torch.cuda.Stream()):
                cache=[{} for _ in range(4)];record.copy_to_cache(cache)
            for head,ids in enumerate(chosen):
                for token in ids:
                    assert torch.equal(cache[head][token],torch.stack((pool.k_buffer[0][token,head],pool.v_buffer[0][token,head])))
            assert record._source_guard.value is physical
            await record.ack();assert await record.close()
            assert record._source_guard.value is None and budget.snapshot()['used_staging_bytes']==0
            await search.close();await control.close()
        store.close()
    asyncio.run(run())


def test_lost_fused_response_preserves_physical_owner_until_original_identity_fence():
    async def run():
        _,store,manifest,_,_,requests,selection=setup(0)
        async with shard_client(store) as http:
            search,control,budget,registry,record=await prepared(store,manifest,requests,selection,http)
            original=search._post_json
            async def lost(path,payload,**kwargs):
                reply=await original(path,payload,**kwargs)
                if path.endswith('search-deliver'): raise TimeoutError('lost response after native submit')
                return reply
            search._post_json=lost
            with pytest.raises(TimeoutError): await start_fused(record,search,requests)
            assert record._published and not await record.close()
            assert record._registration is not None and registry._records
            delivery=store.entries[manifest.key].deliveries[record.identity.transfer_id]
            store.transfer_engine.finish(delivery.transfer_handle)
            assert await record.close() and budget.snapshot()['used_staging_bytes']==0
            await search.close();await control.close()
        store.close()
    asyncio.run(run())


def test_fence_while_search_is_running_blocks_late_reserve_and_put(monkeypatch):
    async def run():
        index,store,manifest,_,_,requests,selection=setup(0)
        entered,release=threading.Event(),threading.Event()
        original=index.backend.search
        def slow(*args,**kwargs):
            entered.set();assert release.wait(5)
            return original(*args,**kwargs)
        monkeypatch.setattr(index.backend,'search',slow)
        async with shard_client(store) as http:
            search,control,budget,registry,record=await prepared(store,manifest,requests,selection,http)
            pending=asyncio.create_task(start_fused(record,search,requests))
            try:
                assert await asyncio.to_thread(entered.wait,5)
                # close serialized by record lock until the pending RPC exits;
                # direct trusted fence proves absence independently of its await.
                proof=await control.fence_delivery(record.identity)
                assert proof['fenced']
                release.set()
                with pytest.raises(Exception): await pending
                assert await record.close() and not store.transfer_engine.pending
                assert budget.snapshot()['used_staging_bytes']==0
            finally:
                release.set();await asyncio.gather(pending,return_exceptions=True)
                await search.close();await control.close()
        store.close()
    asyncio.run(run())


@pytest.mark.parametrize('change',['heads','capacity','cache','layer','entry','query_shape'])
def test_capability_mismatch_rejected_before_registration(change):
    _,store,manifest,_,_,requests,selection=setup(0)
    search=PVDShardSearchClient('http://localhost:1')
    batch=dict(batch_protocol='pvd.search.batch.v1',batch_id='b',items=[
        search._prepare_search(identity,queries=q,top_k=k,scope=s)[0] for identity,q,k,s in requests])
    if change=='heads': selection['heads']=[0,2]
    if change=='capacity': selection['capacity']=33
    if change=='cache': selection['cached'][0]=[manifest.prompt_token_count]
    if change=='layer': selection['layer']=1
    if change=='entry': selection['entry_transfer_id']='different'
    if change=='query_shape': batch['items'][0]['queries']=[[1]]
    budget=TransferBudget(1<<20,4)
    registry=OasisCPUReceiveRegistry(store.transfer_engine,budget,receiver_epoch='D')
    with pytest.raises(ValueError):
        prepare_fused(registry,selection,batch,key=manifest.key,rank=0,rail=store.rail,
            endpoint='D',sender_epoch=store.worker_epoch,client=object())
    assert not registry._records and budget.snapshot()['used_staging_bytes']==0
    store.close()


@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('binary',[False,True])
def test_actual_two_shard_select_fetch_cache_misses_then_hits(monkeypatch,packed,binary):
    monkeypatch.setenv('PVD_PACKED_QUERY_BATCH','1' if packed else '0')
    async def run():
        index0,store0,manifest,pool,layout,_,_=setup(0)
        index1=manager();store1=make_store(manifest,index=index1,rank=1)
        store1.transfer_engine.lifecycle_manager=SimpleNamespace(budget=TransferBudget(1<<20,4))
        entry=store1.create_entry(manifest);store1.begin_p_write(manifest.key)
        packed,shard,_=pack_shard(pool,layout,rank=1,prompt_tokens=manifest.prompt_token_count)
        offset=entry.allocation.start_page*store1.page_bytes
        store1.pool[offset:offset+shard.expected_bytes]=packed.tensor
        store1.commit_p_write(manifest.key,shard.expected_bytes);store1.progress_prompt_indexes()
        async with shard_client(store0) as http0, shard_client(store1) as http1:
            routes=tuple(SimpleNamespace(rank=s.rank,rail=s.rail,sender_epoch=s.worker_epoch,url=str(h.make_url('')))
                for s,h in ((store0,http0),(store1,http1)))
            owner=object.__new__(OasisLayerTransport)
            owner.selected=SimpleNamespace(manifest=manifest,shards=routes)
            owner.lock=threading.Lock();owner.versions={};owner.capacity=4;owner.max_new=4;owner.top_k=2
            owner.prompt_tokens=manifest.prompt_token_count;owner.vector_space='target/model-8b'
            owner.scope=SearchScope(manifest.prompt_token_count,layout.page_size,layout.head_dim,'l2')
            owner.incarnation='incarnation';owner.timeout=2;owner.fused_search_delivery=True;owner.binary_queries=binary
            owner.endpoints={0:'D0',1:'D1'};owner.quarantined=False
            owner._cache_valid=torch.zeros((layout.num_layers,4,manifest.prompt_token_count),dtype=torch.bool)
            rows=torch.empty((layout.num_layers,4,manifest.prompt_token_count,2,layout.head_dim),dtype=torch.float16)
            owner.cache=[[_HeadCPUCache(rows[l,h],owner._cache_valid[l,h]) for h in range(4)] for l in range(layout.num_layers)]
            budget=TransferBudget(1<<20,4)
            state=dict(search={r.rank:PVDShardSearchClient(r.url,binary_queries=binary) for r in routes},
                control={r.rank:HttpShardClient(r.rank,r.url) for r in routes},
                registry=OasisCPUReceiveRegistry(store0.transfer_engine,budget,receiver_epoch='D'))
            queries=[]
            for head in range(4):
                index=index0 if head<2 else index1
                q=index._entries[manifest.key.transfer_id].vectors[(0,head)].vectors[3].tolist()
                queries.extend([q]*7)
            finish=True
            async def complete_native_fake():
                while finish:
                    for s in (store0,store1):
                        for delivery in tuple(s.entries[manifest.key].deliveries.values()):
                            handle=delivery.transfer_handle
                            if handle is not None and handle.transfer_id in s.transfer_engine.pending:
                                s.transfer_engine.finish(handle)
                    await asyncio.sleep(0.001)
            task=asyncio.create_task(complete_native_fake())
            try:
                chosen,remote,_=await owner._select_and_fetch(state,LayerTicket('request','incarnation',0,0),queries,None,False)
                assert remote==8 and len(state['delivery_profiles'])==len(state['fused_profiles'])==2
                for head,ids in enumerate(chosen):
                    for token in ids:
                        assert torch.equal(owner.cache[0][head][token],torch.stack((pool.k_buffer[0][token,head],pool.v_buffer[0][token,head])))
                before=sum(s.transfer_engine.total_put_bytes for s in (store0,store1))
                again,remote,_=await owner._select_and_fetch(state,LayerTicket('request','incarnation',1,0),queries,None,False)
                assert again==chosen and remote==0 and not state['delivery_profiles']
                assert len(state['fused_profiles'])==2 and all(p['nbytes']==0 for p in state['fused_profiles'])
                assert sum(s.transfer_engine.total_put_bytes for s in (store0,store1))==before
                assert not state['registry']._records and budget.snapshot()['used_staging_bytes']==0
            finally:
                finish=False;await task
                await owner._close_clients(state)
        store0.close();store1.close()
    asyncio.run(run())


def test_identical_rpc_replay_submits_once_and_changed_scope_is_refused():
    async def run():
        _,store,manifest,_,_,requests,selection=setup(0)
        async with shard_client(store) as http:
            search,control,budget,registry,record=await prepared(store,manifest,requests,selection,http)
            chosen,ready=await start_fused(record,search,requests)
            request=dict(protocol='pvd.search-delivery.v1',selection=record.fused_scope,
                search=record.fused_search,identity=record.identity.to_dict(),destination=record._registration.descriptor.to_dict())
            reply=await search._post_json('/internal/v1/indexes/search-deliver',request)
            assert reply['chosen']==[list(ids) for ids in chosen] and len(store.transfer_engine.pending)==1
            changed={**request,'selection':{**request['selection'],'operation_id':'other'}}
            with pytest.raises(Exception): await search._post_json('/internal/v1/indexes/search-deliver',changed)
            with pytest.raises(ValueError,match='already published'): await start_fused(record,search,requests)
            delivery=store.entries[manifest.key].deliveries[record.identity.transfer_id]
            store.transfer_engine.finish(delivery.transfer_handle)
            assert await record.poll() and await record.close()
            assert budget.snapshot()['used_staging_bytes']==0
            await search.close();await control.close()
        store.close()
    asyncio.run(run())
