"""Actual f32 bytes and local HTTP searches; no GPU timing claim."""
import asyncio
from dataclasses import replace
import json
import struct
import numpy as np
import pytest
from sglang.srt.disaggregation.pvd.search_wire import pack_binary_batch, unpack_binary_batch
from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient
from test_pvd_search_client import fixture
from test_pvd_prompt_index import shard_client


def payload(values):
    return dict(batch_protocol='pvd.search.batch.v1',batch_id='b',items=[dict(
        search_protocol='pvd.search.v1', search_id='s', transfer_id='entry',vector_space='q',
        positional_encoding='rope_applied',layer=0,kv_head=0,top_k=4,queries=values)])


def test_binary_owned_f32_exact_bits_and_no_python_float_materialization():
    values=np.arange(7*128,dtype=np.float32).reshape(7,128)/3
    values[0,:3]=[0,-0.0,np.finfo(np.float32).tiny]
    raw=pack_binary_batch(payload(values))
    assert raw[-values.nbytes:] == values.astype('<f4').tobytes()
    decoded=unpack_binary_batch(raw)['items'][0]['query_values']
    assert decoded.tobytes() == values.tobytes() and decoded.flags.writeable
    values.fill(99)
    assert decoded[0,0] == 0


@pytest.mark.parametrize('damage',['magic','meta','short','trailing','nan','shape','mixed'])
def test_malformed_binary_refused_before_search(damage):
    raw=pack_binary_batch(payload(np.ones((7,128),dtype=np.float32)))
    if damage == 'magic': raw=b'badmagic'+raw[8:]
    if damage == 'meta': raw=raw[:8]+struct.pack('<I',999999)+raw[12:]
    if damage == 'short': raw=raw[:-1]
    if damage == 'trailing': raw+=b'\0'
    if damage == 'nan': raw=raw[:-4]+struct.pack('<f',float('nan'))
    if damage in ('shape','mixed'):
        size=struct.unpack_from('<I',raw,8)[0];meta=json.loads(raw[12:12+size])
        meta['items'][0].update({'query_dim':100001} if damage=='shape' else {'queries':[[1]]})
        encoded=json.dumps(meta).encode();raw=raw[:8]+struct.pack('<I',len(encoded))+encoded+raw[12+size:]
    with pytest.raises(ValueError): unpack_binary_batch(raw)


@pytest.mark.parametrize('values',[np.ones((65,1)),np.ones((1,100001)),np.ones((1,2),dtype=bool),np.array([[float('inf')]])])
def test_pack_bounds_and_dtype(values):
    with pytest.raises(ValueError): pack_binary_batch(payload(values))


def test_real_http_binary_and_json_match_unpinned_and_pinned_head_results():
    async def run():
        index,store,identity,query,scope=fixture()
        other=replace(identity,kv_head=1)
        other_query=index._entries[identity.entry_transfer_id].vectors[(0,1)].vectors[3:4].numpy()
        array=np.asarray(query,dtype=np.float32)
        async with shard_client(store) as http:
            baseline=PVDShardSearchClient(str(http.make_url('')))
            binary=PVDShardSearchClient(str(http.make_url('')),binary_queries=True)
            try:
                old=await baseline.search_many([(identity,query,1,scope),(other,other_query.tolist(),1,scope)])
                new=await binary.search_many([(identity,array,1,scope),(other,other_query,1,scope)])
                assert old == new
                pin=replace(identity,expected_index_version=old[0].index_version,
                    expected_id_mapping_version=old[0].id_mapping_version)
                assert (await binary.search_many([(pin,array,1,scope)]))[0] == await baseline.search(pin,queries=query,top_k=1,scope=scope)
                invalid=np.full_like(array,np.nan)
                with pytest.raises(ValueError): await binary.search_many([(pin,invalid,1,scope)])
            finally:
                await baseline.close();await binary.close()
    asyncio.run(run())
