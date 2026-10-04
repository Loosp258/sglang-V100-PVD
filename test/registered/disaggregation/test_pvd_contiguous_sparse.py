"""Exact payload/order and cache semantics for contiguous copies; CPU gates."""

import asyncio
from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.pvd.oasis_pipeline import LayerTicket
from sglang.srt.disaggregation.pvd.oasis_transport import OasisLayerTransport
from sglang.srt.disaggregation.pvd.sparse_copy import copy_sparse_kv_into
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from sglang.srt.disaggregation.pvd.sparse_token_runs import consecutive_token_runs
from test_pvd_sparse_copy import expected, setup
from test_pvd_sparse_store_delivery import destination, ready


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('ids', [(0, 1, 2, 3, 4, 9), (9, 8, 7, 6),
                               (4, 5, 0, 1, 8, 9), (9, 0, 1, 2, 3), (7, 0, 9)])
def test_strided_contiguous_copy_exact_bytes_order_and_no_gather(rank, dtype, ids, monkeypatch):
    pool, source, kwargs = setup(rank, dtype)
    old = kwargs['manifest']
    manifest = replace(old, specs=(replace(old.specs[0], token_ids=ids),
                                  replace(old.specs[1], kv_head=rank * 2, token_ids=ids)))
    kwargs['manifest'] = manifest
    target = torch.empty(manifest.nbytes, dtype=torch.uint8)
    wanted, original = expected(pool, manifest), source.clone()
    copied = []
    copy = torch.Tensor.copy_

    def record(tensor, other, **options):
        copied.append(tuple(other.shape))
        return copy(tensor, other, **options)

    def allocate(*args, **options):
        raise AssertionError('no additional gather or payload allocation permitted')

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'copy_', record)
        for name in ('empty', 'empty_like', 'zeros', 'cat', 'stack', 'index_select', 'gather'):
            patch.setattr(torch, name, allocate)
        copy_sparse_kv_into(source, target, **kwargs, contiguous_runs=True)
    torch.testing.assert_close(source, original, rtol=0, atol=0)
    for payload, value in zip(manifest.payload_views(target), wanted, strict=True):
        torch.testing.assert_close(payload.tensor, value, rtol=0, atol=0)
        payload.close()
    # Independent boundary count; includes K/V and both manifest groups.
    boundaries = 1 + sum(b != a + 1 for a, b in zip(ids, ids[1:]))
    assert len(copied) == 4 * boundaries
    if (0, 1) in tuple(zip(ids, ids[1:])):
        assert any(len(shape) == 2 for shape in copied)


@pytest.mark.parametrize('mode', ['invalid_later', 'alias', 'nonbool', 'fused'])
def test_bad_contiguous_contract_never_partly_writes(mode):
    _, source, kwargs = setup()
    target = torch.full((kwargs['manifest'].nbytes,), 211, dtype=torch.uint8)
    options = dict(contiguous_runs=True)
    if mode == 'invalid_later':
        manifest = kwargs['manifest']
        kwargs['manifest'] = replace(manifest, specs=(manifest.specs[0],
            replace(manifest.specs[1], token_ids=(10, 11))))
    elif mode == 'alias':
        target = source[:target.numel()]
    elif mode == 'nonbool':
        options['contiguous_runs'] = 1
    else:
        options['fused_workspace'] = object()
    before = target.clone()
    with pytest.raises(SparsePayloadError):
        copy_sparse_kv_into(source, target, **kwargs, **options)
    assert torch.equal(target, before)


def test_failed_contiguous_copy_keeps_source_and_partial_destination(monkeypatch):
    _, source, kwargs = setup()
    manifest = kwargs['manifest']
    kwargs['manifest'] = replace(manifest, specs=tuple(
        replace(spec, token_ids=(0, 1, 2, 8, 9)) for spec in manifest.specs))
    target = torch.full((kwargs['manifest'].nbytes,), 211, dtype=torch.uint8)
    before, calls, original = source.clone(), [], torch.Tensor.copy_

    def fail(tensor, other, **options):
        calls.append(tuple(other.shape))
        if len(calls) == 2:
            raise RuntimeError('partial copy failed')
        return original(tensor, other, **options)

    monkeypatch.setattr(torch.Tensor, 'copy_', fail)
    with pytest.raises(RuntimeError, match='partial copy'):
        copy_sparse_kv_into(source, target, **kwargs, contiguous_runs=True)
    assert calls == [(3, kwargs['layout'].head_dim), (2, kwargs['layout'].head_dim)]
    assert torch.equal(source, before) and target.eq(211).any() and not target.eq(211).all()


def test_real_store_staging_and_ack_path_preserve_bytes_and_budget():
    store, entry, pool, manifest, budget = ready()
    manifest = replace(manifest, specs=tuple(
        replace(spec, token_ids=(0, 1, 2, 3, 4, 5)) for spec in manifest.specs))
    store.contiguous_sparse_packing = True  # explicit CPU test-only store
    target = destination(store, manifest)
    engine = store.transfer_engine
    try:
        delivery = store.reserve_delivery(entry.key, 'contiguous-cpu', target.descriptor)
        store.start_delivery(entry.key, delivery.delivery_id)
        assert delivery.transfer_handle is not None, delivery.error
        assert budget.snapshot()['used_staging_bytes'] == manifest.nbytes
        engine.finish(delivery.transfer_handle)
        store.poll_delivery(entry.key, delivery.delivery_id)
        store.ack_delivery(entry.key, delivery.delivery_id)
        assert budget.snapshot()['used_staging_bytes'] == 0
        for payload, value in zip(manifest.payload_views(target.buffer), expected(pool, manifest), strict=True):
            assert torch.equal(payload.tensor, value)
            payload.close()
        assert engine.total_put_bytes == manifest.nbytes
    finally:
        store.close()
        engine.release_memory(target)


def selection(sort_wire):
    """Exercise actual selection and post-receive cache code with a CPU RPC double."""
    transport = object.__new__(OasisLayerTransport)
    routes = tuple(SimpleNamespace(rank=r, rail='test', sender_epoch='V') for r in (0, 1))
    transport.selected = SimpleNamespace(shards=routes, manifest=SimpleNamespace(
        key=SimpleNamespace(transfer_id='entry'), layout=SimpleNamespace(fingerprint='layout')))
    transport.cache = [[{1: head * 100 + 1} for head in range(4)]]
    transport.lock, transport.versions = threading.Lock(), {}
    transport.capacity, transport.max_new, transport.top_k, transport.prompt_tokens = 8, 8, 4, 10
    transport.vector_space, transport.scope, transport.incarnation = 'target-Q', None, 'inc'
    transport.endpoints, transport.timeout = {0: 'D0', 1: 'D1'}, 1
    transport.gpu_backups, transport.gpu_receive_to_bank = None, False
    transport.sort_missing_tokens = sort_wire
    wires = []

    class Search:
        async def search_many(self, requests):
            return [SimpleNamespace(index_version='index', id_mapping_version='map',
                    scores=(4, 3, 2, 1), token_ids=(5, 1, 2, 4)) for _ in requests]

    class Record:
        profile = {}
        async def start(self):
            return True
        def copy_to_cache(self, cache):
            for spec in self.manifest.specs:
                for token in spec.token_ids:
                    assert token not in cache[spec.kv_head]
                    cache[spec.kv_head][token] = spec.kv_head * 100 + token
        async def ack(self):
            return None
        async def close(self):
            return True

    class Registry:
        def prepare(self, wire, **kwargs):
            wires.append(wire)
            record = Record()
            record.manifest = wire
            return record

    state = dict(search={r: Search() for r in (0, 1)}, registry=Registry(),
                 control={0: None, 1: None})
    ticket = LayerTicket('request', 'inc', 0, 0)
    chosen, rows, _ = asyncio.run(transport._select_and_fetch(
        state, ticket, [[0] * 128] * 28, None, True))
    return chosen, rows, wires, transport.cache, state['delivery_profiles']


def test_sorted_wire_preserves_ranked_bank_cpu_cache_and_byte_budget():
    base, opt = selection(False), selection(True)
    assert base[0] == opt[0] == ((5, 1, 2, 4),) * 4
    assert base[1] == opt[1] == 12 and base[3] == opt[3]
    assert [w.nbytes for w in base[2]] == [w.nbytes for w in opt[2]] == [3072, 3072]
    assert all(s.token_ids == (5, 2, 4) for w in base[2] for s in w.specs)
    assert all(s.token_ids == (2, 4, 5) for w in opt[2] for s in w.specs)
    assert all(p['wire_ids_sorted'] for p in opt[4])
    assert sum(p['wire_runs'] for p in base[4]) == 12
    assert sum(p['wire_runs'] for p in opt[4]) == 8


def test_empty_token_run_iterator_has_no_ranges():
    assert tuple(consecutive_token_runs(())) == ()
