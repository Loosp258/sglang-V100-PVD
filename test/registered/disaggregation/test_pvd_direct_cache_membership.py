"""Fixed candidate semantics, strict encodings, real CPU HTTP and no bitmap expansion."""
import base64
import numpy as np
import pytest
from sglang.srt.disaggregation.pvd.cache_snapshot import (
    pack_cache_snapshot, cache_ids, decode_cache_snapshot, cache_hits)
from sglang.srt.disaggregation.pvd.fused_search_delivery import choose_wire, selection_digest
from sglang.srt.disaggregation.pvd.search_wire import freeze_binary_search


@pytest.mark.parametrize('n',[1,7,8,9,2159,8192,32768])
@pytest.mark.parametrize('stride',[1,3,127])
@pytest.mark.parametrize('encoding',['automatic','list','sparse','bitmap'])
def test_membership_matches_original_sets_without_expansion(n,stride,encoding,monkeypatch):
    valid=np.zeros(n,dtype=bool);valid[::stride]=True
    ids=np.flatnonzero(valid)
    if encoding=='list':snapshot=ids.tolist()[::-1]
    elif encoding=='sparse':snapshot=dict(encoding='u16le-base64-v1',data=base64.b64encode(ids.astype('<u2').tobytes()).decode())
    elif encoding=='bitmap':snapshot=dict(encoding='bitset-le-base64-v1',data=base64.b64encode(np.packbits(valid,bitorder='little').tobytes()).decode())
    else:snapshot=pack_cache_snapshot(valid)
    expected=set(map(int,cache_ids(snapshot,n)))
    tokens=np.random.default_rng(45).integers(0,n,size=32).tolist()
    decoded=decode_cache_snapshot(snapshot,n)
    def fail(*args,**kwargs):raise AssertionError('must not expand Prompt IDs')
    monkeypatch.setattr(np,'flatnonzero',fail);monkeypatch.setattr(np,'unpackbits',fail)
    assert cache_hits(decoded,tokens,n)==tuple(t in expected for t in tokens)
    assert cache_hits(decoded,[],n)==()


@pytest.mark.parametrize('damage',['padding','short','order','duplicate','bounds','base64','noncanonical','extra','bool'])
def test_malformed_cache_cannot_claim_a_hit(damage):
    snapshot=dict(encoding='bitset-le-base64-v1',data='AQA=')
    if damage=='padding':snapshot['data']='AYA='
    elif damage=='short':snapshot['data']='AQ=='
    elif damage in ('order','duplicate','bounds'):
        ids={'order':[4,2],'duplicate':[2,2],'bounds':[9]}[damage]
        snapshot=dict(encoding='u16le-base64-v1',data=base64.b64encode(np.array(ids,dtype='<u2').tobytes()).decode())
    elif damage=='base64':snapshot['data']='!!!!'
    elif damage=='noncanonical':snapshot['data']='AQB='
    elif damage=='extra':snapshot['extra']=1
    else:snapshot=[True]
    with pytest.raises(ValueError):decode_cache_snapshot(snapshot,9)


def test_chosen_manifest_and_digest_equal_old_dense_policy(monkeypatch):
    from test_pvd_fused_wire import fixture
    store,scope,search=fixture()
    search=freeze_binary_search(search)
    try:
        scope['cached']=[dict(encoding='bitset-le-base64-v1',data='/w=='),[1,3]]
        results=[dict(index_version='i',id_mapping_version='m',layer=0,kv_head=h,
            token_ids=[0,1,2,3,4],scores=[1.,2.,2.,0.,-1.]) for h in (0,1)]
        monkeypatch.delenv('PVD_DIRECT_CACHE_MEMBERSHIP',raising=False)
        old=choose_wire(scope,results);digest=selection_digest(scope,search)
        monkeypatch.setenv('PVD_DIRECT_CACHE_MEMBERSHIP','1')
        def fail(*args,**kwargs):raise AssertionError('must not expand Prompt IDs')
        monkeypatch.setattr(np,'flatnonzero',fail);monkeypatch.setattr(np,'unpackbits',fail)
        assert choose_wire(scope,results)==old
        assert selection_digest(scope,search)==digest
    finally:store.close()


@pytest.mark.parametrize('rank',[0,1])
def test_real_cpu_channel_keeps_exact_bytes_and_zero_miss_policy(monkeypatch,rank):
    from test_pvd_fused_search_delivery import test_both_logical_ranks_http_dynamic_prefix_exact_bytes_and_zero_miss as run
    monkeypatch.setenv('PVD_DIRECT_CACHE_MEMBERSHIP','1')
    for mode in ('miss','mixed','hit'):run(rank,mode,True,True)
