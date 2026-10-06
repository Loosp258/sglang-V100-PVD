"""Lost V cache state: real channel fences old generation and D resends full."""
import asyncio
import threading
from types import SimpleNamespace
import pytest
import torch
from sglang.srt.disaggregation.pvd.cache_delta import CacheDeltaSender,CacheDeltaReceiver
from sglang.srt.disaggregation.pvd.oasis_transport import OasisLayerTransport,_HeadCPUCache
from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient,SearchScope
from test_pvd_oasis_receive_slot_records import slot_case
from test_pvd_prompt_index import ident


def test_lost_receiver_state_requires_absent_fence_and_new_identity(monkeypatch):
    calls=[];original=CacheDeltaReceiver.prepare
    def lose(self,scope,delta):
        calls.append(delta)
        if len(calls)==2:self.states.clear()
        return original(self,scope,delta)
    monkeypatch.setattr(CacheDeltaReceiver,'prepare',lose)
    async def run():
        async with slot_case(monkeypatch) as c:
            search=PVDShardSearchClient(c.client.base_url,binary_queries=True,binary_control_channel=True)
            owner=object.__new__(OasisLayerTransport)
            owner.selected=SimpleNamespace(manifest=c.entry)
            owner.capacity=owner.max_new=4;owner.prompt_tokens=8;owner.timeout=5
            owner.compact_cache_snapshots=owner.binary_queries=owner.ready_before_cleanup=True
            owner.fused_zero_miss_proof=owner.channel_cleanup=owner.parallel_owned_cleanup=True
            owner.cache_delta_sender=CacheDeltaSender(8)
            owner._cache_valid=torch.zeros((1,2,8),dtype=torch.bool)
            owner.cache=[[_HeadCPUCache(torch.empty((8,2,128),dtype=torch.float16),owner._cache_valid[0,h],
                lambda token,h=h:owner.cache_delta_sender.publish(0,h,token)) for h in (0,1)]]
            owner.versions={};owner.lock=threading.Lock();owner.quarantined=False
            owner.endpoints={0:'D'};owner.incarnation='inc'
            ticket=SimpleNamespace(request_id='req',incarnation='inc',step=0,layer=0)
            route=SimpleNamespace(rank=0,rail=c.store.rail,sender_epoch=c.store.worker_epoch)
            requests=[(ident(c.entry.key.transfer_id,kv_head=h),[[1.0]*128],2,
                SearchScope(8,c.entry.layout.page_size,128,'l2')) for h in (0,1)]
            def state():return dict(registry=c.registry,search={0:search},control={0:c.client},fused_profiles=[],delivery_profiles=[])
            registrations=[];register=c.registry._prepare_registration
            def capture(record,**kwargs):
                result=register(record,**kwargs);registrations.append(record);return result
            monkeypatch.setattr(c.registry,'_prepare_registration',capture)
            try:
                first=state()
                task=asyncio.create_task(owner._fused_fetch(first,ticket,route,requests,None,False))
                while not c.engine.pending:await asyncio.sleep(.001)
                delivery=next(iter(c.store.entries[c.entry.key].deliveries.values()))
                c.engine.finish(delivery.transfer_handle)
                chosen,rows=await task;assert rows==4
                await owner._finish_owned_cleanup(first)
                before=c.engine.total_put_bytes
                second=state();selected,rows=await owner._fused_fetch(second,ticket,route,requests,None,False)
                assert selected==chosen and rows==0 and c.engine.total_put_bytes==before
                assert len(calls)==3 and len(registrations)==3
                old,new=registrations[1:]
                assert old.identity.transfer_id!=new.identity.transfer_id
                assert old.identity.generation!=new.identity.generation
                assert old._absent_write_closed and old._closed and new._closed
                assert (old.identity.key,old.identity.transfer_id) in c.store._absent_write_fences
                assert new.profile['cache_delta_resyncs']==1
                assert not c.registry._records and c.pool.snapshot()['leased_slots']==0
                third=state();assert (await owner._fused_fetch(third,ticket,route,requests,None,False))[1]==0
                assert all(s['full'] is None for s in calls[-1]['states'])
                assert search._channel.snapshot()['connections']==1
            finally:await search.close()
    asyncio.run(run())


def test_cache_delta_config_requires_exact_channel_and_scratch(monkeypatch,tmp_path):
    from test_pvd_delivery_followup_config import load,config
    cfg=config(5);cfg['cache_delta_snapshots']=True
    assert load(monkeypatch,tmp_path,cfg)['cache_delta_snapshots'] is True
    cfg['compact_cache_snapshots']=False
    with pytest.raises(ValueError,match='cache deltas'):load(monkeypatch,tmp_path,cfg)
    cfg['compact_cache_snapshots']=True;cfg['request_scratch_bytes']=1
    with pytest.raises(ValueError,match='scratch'):load(monkeypatch,tmp_path,cfg)
