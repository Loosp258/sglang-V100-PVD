"""Exact nested binary Q and authorization binding; no GPU substitute."""
import json
import numpy as np
import pytest
from sglang.srt.disaggregation.pvd.search_wire import (
    pack_binary_fused, unpack_binary_fused, unpack_binary_batch, binary_search_snapshot)
from sglang.srt.disaggregation.pvd.fused_search_delivery import selection_digest
from test_pvd_fused_search_delivery import setup
from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient


def fixture():
    _, store, _, _, _, requests, scope = setup(0)
    client = PVDShardSearchClient('http://unused', binary_queries=True)
    search = dict(batch_protocol='pvd.search.batch.v1', batch_id='batch', items=[
        client._prepare_search(i, queries=np.array(q, dtype=np.float32), top_k=k, scope=s)[0]
        for i, q, k, s in requests])
    return store, scope, search


def test_nested_q_bytes_owned_and_digest_bound_to_exact_q_and_selection():
    store, scope, search = fixture()
    try:
        frozen = unpack_binary_batch(binary_search_snapshot(search))
        digest = selection_digest(scope, frozen)
        payload = dict(protocol='pvd.search-delivery.v1', selection=scope, search=frozen,
                       identity={}, destination={})
        raw = pack_binary_fused(payload)
        parsed = unpack_binary_fused(raw)
        assert selection_digest(parsed['selection'], parsed['search']) == digest
        for before, after in zip(frozen['items'], parsed['search']['items'], strict=True):
            assert before['query_values'].tobytes() == after['query_values'].tobytes()
        frozen['items'][0]['query_values'].fill(42)
        assert selection_digest(scope, frozen) != digest
        assert selection_digest(scope, parsed['search']) == digest
        changed = {**scope, 'operation_id': 'changed'}
        assert selection_digest(changed, parsed['search']) != digest
    finally:
        store.close()


@pytest.mark.parametrize('damage', ['short', 'trailing', 'nan', 'nested_items'])
def test_bad_nested_payload_refused(damage):
    store, scope, search = fixture()
    try:
        payload = dict(protocol='pvd.search-delivery.v1', selection=scope, search=search,
                       identity={}, destination={})
        if damage == 'nested_items':
            import struct
            raw = pack_binary_fused(payload)
            size = struct.unpack_from('<I', raw, 8)[0]
            meta = json.loads(raw[12:12+size]); meta['search']['items'] = []
            encoded = json.dumps(meta).encode()
            raw = raw[:8] + struct.pack('<I', len(encoded)) + encoded + raw[12+size:]
        elif damage == 'nan':
            raw = pack_binary_fused(payload)[:-4] + np.float32(np.nan).tobytes()
        else:
            raw = pack_binary_fused(payload)
            raw = raw[:-1] if damage == 'short' else raw+b'x'
        with pytest.raises(ValueError):
            unpack_binary_fused(raw)
    finally:
        store.close()
