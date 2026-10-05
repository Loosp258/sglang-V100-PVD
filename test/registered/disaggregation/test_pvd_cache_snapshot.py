"""Compact metadata preserves cache membership and the selected wire exactly."""
import base64
import numpy as np
import pytest
from sglang.srt.disaggregation.pvd.cache_snapshot import pack_cache_snapshot, cache_ids
from sglang.srt.disaggregation.pvd.fused_search_delivery import choose_wire
from test_pvd_fused_search_delivery import setup, prepared
from test_pvd_prompt_index import shard_client
from sglang.srt.disaggregation.pvd.fused_search_delivery import start_fused


@pytest.mark.parametrize('tokens',[1,7,8,9,2159,32768])
@pytest.mark.parametrize('stride',[1,3,127])
def test_exact_owned_sets_at_prompt_and_padding_boundaries(tokens,stride):
    valid=np.zeros(tokens,dtype=np.bool_);valid[::stride]=True
    expected=np.flatnonzero(valid)
    encoded=pack_cache_snapshot(valid)
    valid.fill(False)
    assert np.array_equal(cache_ids(encoded,tokens),expected)


@pytest.mark.parametrize('damage',['padding','length','order','duplicate','bounds','extra','base64','encoding'])
def test_invalid_snapshot_cannot_exclude_remote_cache_rows(damage):
    snapshot=dict(encoding='bitset-le-base64-v1',data=base64.b64encode(b'\x01\x00').decode())
    if damage=='padding': snapshot['data']=base64.b64encode(b'\x01\x80').decode()
    if damage=='length': snapshot['data']=''
    if damage in ('order','duplicate','bounds'):
        ids={'order':[4,2],'duplicate':[2,2],'bounds':[16]}[damage]
        snapshot=dict(encoding='u16le-base64-v1',data=base64.b64encode(np.array(ids,dtype='<u2').tobytes()).decode())
    if damage=='extra': snapshot['extra']=1
    if damage=='base64': snapshot['data']='!!!!'
    if damage=='encoding': snapshot['encoding']='unknown'
    with pytest.raises(ValueError): cache_ids(snapshot,9)


@pytest.mark.parametrize('binary',[False,True])
def test_real_http_bitmap_hit_and_mixed_miss_match_list_policy(binary):
    import asyncio
    async def run():
        _,store,manifest,_,_,requests,scope=setup(0)
        scope['cached'][0]=dict(encoding='bitset-le-base64-v1',data='/w==')
        async with shard_client(store) as http:
            search,control,budget,registry,record=await prepared(store,manifest,requests,scope,http,binary=binary)
            try:
                chosen,ready=await start_fused(record,search,requests)
                plain={**scope,'cached':[list(range(8)),[]]}
                old_ids,old_wire=choose_wire(plain,record.fused_results)
                assert chosen==old_ids and record.manifest==old_wire
                delivery=store.entries[manifest.key].deliveries[record.identity.transfer_id]
                store.transfer_engine.finish(delivery.transfer_handle)
                assert await record.poll()
                cache=[{},{}];record.copy_to_cache(cache)
                assert not cache[0] and cache[1]
                await record.ack();assert await record.close()
            finally:
                await search.close();await control.close()
        store.close()
    asyncio.run(run())
