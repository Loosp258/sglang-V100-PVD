"""Real CPU HTTP/bytes with declared CUDA policy doubles; not RDMA proof."""
import asyncio
import pytest
from sglang.srt.disaggregation.pvd.fused_search_delivery import prepare_fused, start_fused
from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient, SearchScope
from test_pvd_prompt_index import ident
from test_pvd_oasis_receive_slot_records import slot_case, writer


def prepared(c, client, op, *, cached=False, zero_miss_proof=False):
    scope = dict(request_id='req', incarnation='inc', operation_id=op,
        target_tokens=1, entry_transfer_id=c.entry.key.transfer_id,
        layout_fingerprint=c.entry.layout.fingerprint, layer=0, heads=[0,1],
        capacity=4, max_new=4, prompt_tokens=8, dtype='torch.float16', head_dim=128,
        resident=[[],[]], cached=[list(range(8)) if cached else [] for _ in range(2)])
    requests = [(ident(c.entry.key.transfer_id,kv_head=h), [[1.0]*128], 2,
                 SearchScope(8,c.entry.layout.page_size,128,'l2')) for h in (0,1)]
    search = dict(batch_protocol='pvd.search.batch.v1',batch_id=op,items=[
        client._prepare_search(i,queries=q,top_k=k,scope=s)[0] for i,q,k,s in requests])
    record = prepare_fused(c.registry,scope,search,key=c.entry.key,rank=0,
        rail=c.store.rail,endpoint='D',sender_epoch=c.store.worker_epoch,
        client=c.client,binary_queries=client.binary_queries,zero_miss_proof=zero_miss_proof)
    return record, requests


@pytest.mark.parametrize('binary',[False,True])
def test_dynamic_prefix_miss_hit_miss_reuses_one_physical_mr(monkeypatch,binary):
    async def run():
        async with slot_case(monkeypatch) as c:
            search = PVDShardSearchClient(c.client.base_url,binary_queries=binary)
            try:
                previous = None
                for n, hit in enumerate((False,True,False)):
                    record,requests = prepared(c,search,f'op{n}',cached=hit)
                    physical = record._slot_lease._slot.registration
                    assert c.pool.snapshot()['physical_register_calls']==1
                    if previous is not None:
                        assert previous.region_id==record.identity.region_id
                        with pytest.raises(ValueError): previous.validate_destination(record._registration.descriptor)
                    _,ready = await start_fused(record,search,requests)
                    if not hit:
                        assert not ready
                        c.engine.finish(writer(c,record).transfer_handle)
                        assert await record.poll()
                        cache = [{},{}];record.copy_to_cache(cache)
                        for h in (0,1):
                            for token,value in cache[h].items():
                                import torch
                                oracle=torch.stack((c.source.k_buffer[0][token,h],c.source.v_buffer[0][token,h]))
                                assert torch.equal(value,oracle)
                        await record.ack()
                    else:
                        assert ready and record.fused_no_miss
                    old_destination=getattr(record,'_wire_destination',record._registration.descriptor)
                    assert await record.close()
                    assert c.pool.snapshot()['leased_slots']==0
                    assert c.budget.snapshot()['used_staging_bytes']==4096
                    assert c.budget.snapshot()['used_inflight']==0
                    assert physical not in c.released_objects
                    previous=record.identity
                    # Identical replay may return the retired record, but must
                    # never launch a second writer against this physical MR.
                    before=c.engine.total_put_bytes
                    if hit:
                        with pytest.raises(Exception):
                            await c.client.reserve_delivery(c.entry.key,previous.transfer_id,old_destination)
                    else:
                        await c.client.reserve_delivery(c.entry.key,previous.transfer_id,old_destination)
                        reply=await c.client.start_delivery(c.entry.key,previous.transfer_id)
                        assert reply['state'] in ('acked','released')
                    assert c.engine.total_put_bytes==before and not c.engine.pending
                c.pool.close()
                assert c.pool.snapshot()['physical_release_calls']==1
                assert c.budget.snapshot()['used_staging_bytes']==0
            finally:
                await search.close()
    asyncio.run(run())


def test_lost_binary_reply_keeps_lease_until_exact_remote_fence(monkeypatch):
    async def run():
        async with slot_case(monkeypatch) as c:
            search=PVDShardSearchClient(c.client.base_url,binary_queries=True)
            record,requests=prepared(c,search,'lost')
            original=search._post_json
            async def lost(path,payload,**options):
                await original(path,payload,**options)
                raise TimeoutError('reply lost after submit')
            search._post_json=lost
            try:
                with pytest.raises(TimeoutError): await start_fused(record,search,requests)
                assert not await record.close()
                assert c.pool.snapshot()['leased_slots']==1
                with pytest.raises(Exception): c.pool.close()
                c.engine.finish(writer(c,record).transfer_handle)
                assert await record.close()
                assert c.pool.snapshot()['leased_slots']==0
            finally:
                await search.close()
    asyncio.run(run())
