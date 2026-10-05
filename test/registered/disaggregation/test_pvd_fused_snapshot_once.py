"""Frozen binary Q encoding counts and exact compatibility, using real arrays."""
import copy
import numpy as np
import pytest

from sglang.srt.disaggregation.pvd import search_wire as wire
from sglang.srt.disaggregation.pvd.fused_search_delivery import selection_digest


def case():
    scope=dict(request_id='request',incarnation='inc',operation_id='op',target_tokens=1,
        entry_transfer_id='entry',layout_fingerprint='layout',layer=0,heads=[0,1],
        capacity=4,max_new=4,prompt_tokens=8,dtype='torch.float16',head_dim=128,
        resident=[[],[]],cached=[[],[]])
    q=np.arange(7*128,dtype=np.float32).reshape(7,128)/11
    q[0,0]=-0.0
    search=dict(batch_protocol='pvd.search.batch.v1',batch_id='batch',items=[
        dict(kv_head=h,layer=0,transfer_id='entry',search_id=f's{h}',queries=q.copy()) for h in (0,1)])
    return scope,search


def test_once_encoding_readonly_byte_owner_and_legacy_digest(monkeypatch):
    scope, search=case()
    original=copy.deepcopy(search)
    calls=[]
    pack=wire.pack_binary_batch
    def counted(value):
        calls.append(value)
        return pack(value)
    monkeypatch.setattr(wire,'pack_binary_batch',counted)
    snapshot=wire.freeze_binary_search(search)
    digest=selection_digest(scope,snapshot)
    payload=dict(protocol='pvd.search-delivery.v1',selection=scope,search=snapshot.search,
        identity={'generation':'g'},destination={'region_id':'r'})
    encoded=wire.pack_binary_fused(payload,snapshot=snapshot)
    received=wire.unpack_binary_fused(encoded,frozen_search=True)
    assert selection_digest(received['selection'],received['search'])==digest
    assert len(calls)==1
    assert snapshot.search['items'][0]['query_values'].flags.writeable is False
    with pytest.raises(ValueError): snapshot.search['items'][0]['query_values'][0,0]=1
    search['items'][0]['queries'].fill(100)
    search['batch_id']='mutated'
    assert wire.pack_binary_fused(payload,snapshot=snapshot)==encoded
    assert selection_digest(scope,snapshot)==digest
    legacy=wire.unpack_binary_batch(wire.binary_search_snapshot(original))
    assert selection_digest(scope,legacy)==digest
    assert wire.pack_binary_fused({**payload,'search':legacy})==encoded
    parsed=wire.unpack_binary_fused(encoded)
    for item, old in zip(parsed['search']['items'],original['items'],strict=True):
        assert item['query_values'].tobytes()==old['queries'].tobytes()


@pytest.mark.parametrize('damage',['trailing','truncated','nan','shape'])
def test_once_path_keeps_original_wire_validation(damage):
    scope,search=case()
    if damage=='nan':
        search['items'][0]['queries'][0,0]=np.nan
        with pytest.raises(ValueError): wire.freeze_binary_search(search)
        return
    if damage=='shape':
        search['items'][0]['queries']=np.ones(128)
        with pytest.raises(ValueError): wire.freeze_binary_search(search)
        return
    snapshot=wire.freeze_binary_search(search)
    encoded=wire.pack_binary_fused(dict(search=snapshot.search,selection=scope),snapshot=snapshot)
    with pytest.raises(ValueError): wire.unpack_binary_fused(encoded+b'x' if damage=='trailing' else encoded[:-1])
