"""Real CPU indexed cache writes; protocol stubs are explicit, not RDMA proof."""

import asyncio
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.pvd.oasis_cache_install import install_cpu_payloads, cache_install_tensor_bound
from sglang.srt.disaggregation.pvd.oasis_transport import _HeadCPUCache, OasisLayerTransport, OasisCUDAReceiveRecord, OasisCPUReceiveRecord
from sglang.srt.disaggregation.pvd.cuda_receive_ordering import CUDAReceiveOrdering
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
from test_pvd_oasis_receive_slot_records import slot_case, prepare, ready


def payload_case(ids=((7, 1, 4), (2, 9)), heads=(0, 3)):
    specs = tuple(SparseKVSpec('r', 'i', 'op', 1, 'entry', 'index', 'map', 'layout',
        2, head, tuple(tokens)) for head, tokens in zip(heads, ids))
    manifest = SparseDeliveryManifest(specs, 'torch.float16', 128)
    wire = torch.arange(manifest.nbytes // 2, dtype=torch.int64).remainder(701).to(torch.float16)
    wire = wire.view(torch.uint8)
    payloads = manifest.payload_views(wire)
    caches = [_HeadCPUCache(torch.full((12, 2, 128), -1, dtype=torch.float16),
        torch.zeros(12, dtype=torch.bool)) for _ in range(4)]
    return manifest, wire, payloads, caches


@pytest.mark.parametrize('ids', [((7, 1, 4), (2, 9)), ((0,), (11,)),
    ((1, 2, 3, 4), (8, 9, 10, 11))])
def test_batch_copy_exact_and_source_can_retire_without_clones(monkeypatch, ids):
    manifest, wire, payloads, cache = payload_case(ids)
    expected = [p.tensor.clone() for p in payloads]
    def no_clone(*args, **kwargs):
        raise AssertionError('batch cache path made a KV clone')
    monkeypatch.setattr(torch.Tensor, 'clone', no_clone)
    profile = install_cpu_payloads(payloads, cache, capacity=12, retained=[])
    assert profile['cache_row_clones'] == 0
    assert profile['cache_kv_copy_calls'] == profile['cache_valid_write_calls'] == 2
    assert profile['cache_kv_bytes'] == manifest.nbytes
    for payload, tensor in zip(payloads, expected):
        head, tokens = payload.spec.kv_head, payload.spec.token_ids
        assert torch.equal(cache[head].rows[list(tokens)], tensor.permute(1, 0, 2))
        mask = torch.zeros(12, dtype=torch.bool)
        mask[list(tokens)] = True
        assert torch.equal(cache[head].valid, mask)
        assert torch.all(cache[head].rows[~mask] == -1)
        payload.close()
    wire.zero_()  # independent destination owns bytes before ACK
    for head, tokens, tensor in zip((0, 3), ids, expected):
        assert torch.equal(cache[head].rows[list(tokens)], tensor.permute(1, 0, 2))


@pytest.mark.parametrize('bad', ['cached', 'bounds', 'alias', 'dtype', 'cache_dtype', 'layer', 'duplicate_head', 'capacity'])
def test_invalid_later_group_rejects_before_first_write(bad):
    manifest, wire, payloads, cache = payload_case()
    capacity = 12
    if bad == 'cached': cache[3].valid[9] = True
    if bad == 'bounds':
        payloads = (payloads[0], SimpleNamespace(spec=replace(payloads[1].spec, token_ids=(2, 12)),
            tensor=payloads[1].tensor))
    if bad == 'alias':
        payloads = (payloads[0], SimpleNamespace(spec=payloads[1].spec,
            tensor=cache[0].rows[:2].permute(1, 0, 2)))
    if bad == 'dtype':
        payloads = (payloads[0], SimpleNamespace(spec=payloads[1].spec, tensor=payloads[1].tensor.float()))
    if bad == 'cache_dtype': cache[3].rows = cache[3].rows.float()
    if bad == 'layer':
        payloads = (payloads[0], SimpleNamespace(spec=replace(payloads[1].spec, layer=3), tensor=payloads[1].tensor))
    if bad == 'duplicate_head':
        payloads = (payloads[0], SimpleNamespace(spec=replace(payloads[1].spec, kv_head=0), tensor=payloads[1].tensor))
    if bad == 'capacity': capacity = 2
    with pytest.raises((ValueError, RuntimeError)):
        install_cpu_payloads(payloads, cache, capacity=capacity, retained=[])
    assert not cache[0].valid.any() and torch.all(cache[0].rows == -1)
    assert torch.all(cache[3].rows == -1)


def test_partial_copy_does_not_mark_rows_valid_and_retains_all_sources(monkeypatch):
    manifest, wire, payloads, cache = payload_case()
    original = torch.Tensor.index_copy_
    def fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError('CPU copy returned unknown')
    monkeypatch.setattr(torch.Tensor, 'index_copy_', fail)
    retained = []
    with pytest.raises(RuntimeError, match='unknown'):
        install_cpu_payloads(payloads, cache, capacity=12, retained=retained)
    assert not any(c.valid.any() for c in cache)
    source_storage = wire.untyped_storage().data_ptr()
    assert sum(isinstance(x, torch.Tensor) and x.untyped_storage().data_ptr() == source_storage
        for x in retained) == 2
    assert sum(isinstance(x, torch.Tensor) and x.dtype == torch.int64 for x in retained) == 2


def test_cpu_policy_cuda_receiver_installs_before_ack_and_retires_sources(monkeypatch):
    async def run():
        async with slot_case(monkeypatch) as c:
            # Explicit CPU CUDA-ordering fixture; real tensors/HTTP/ACK/cache.
            c.registry.batched_cache_install = True
            c.registry.cache_capacity = 8
            cache = [_HeadCPUCache(torch.zeros((8, 2, 128), dtype=torch.float16),
                torch.zeros(8, dtype=torch.bool)) for _ in range(4)]
            r = prepare(c)
            with pytest.raises(RuntimeError, match='terminal-success'):
                r.copy_to_cache(cache)
            await ready(c, r)
            expected = [(p.spec, p.tensor.clone()) for p in r.manifest.payload_views(r._buffer)]
            r.copy_to_cache(cache)
            assert r._installed and r.profile['cache_install_complete']
            assert r.profile['cache_install_mode'] == 'batched'
            assert not r._cache_copy_owners
            await r.ack()
            assert await r.close()
            assert c.registry.snapshot() == {}
            for spec, value in expected:
                assert torch.equal(cache[spec.kv_head].rows[list(spec.token_ids)], value.permute(1, 0, 2))
                assert cache[spec.kv_head].valid[list(spec.token_ids)].all()
    asyncio.run(run())


def test_cpu_policy_partial_install_refuses_ack(monkeypatch):
    async def run():
        async with slot_case(monkeypatch) as c:
            c.registry.batched_cache_install = True
            cache = [_HeadCPUCache(torch.zeros((8, 2, 128), dtype=torch.float16),
                torch.zeros(8, dtype=torch.bool)) for _ in range(4)]
            r = prepare(c)
            await ready(c, r)
            def fail(*args, **kwargs):
                raise RuntimeError('injected cache copy failure')
            monkeypatch.setattr(torch.Tensor, 'index_copy_', fail)
            with pytest.raises(RuntimeError, match='injected'):
                r.copy_to_cache(cache)
            assert not r._installed and not any(e.valid.any() for e in cache)
            with pytest.raises(ValueError, match='must install'):
                await r.ack()
            assert await r.close()  # remote and local drain already proven
    asyncio.run(run())


def test_cache_admission_refuses_mixing_and_undercharged_scratch():
    layout = SimpleNamespace(num_layers=28, total_kv_heads=4, kv_heads_per_rank=2,
        head_dim=128, kv_dtype='torch.float16')
    selected = SimpleNamespace(manifest=SimpleNamespace(layout=layout),
        shards=tuple(SimpleNamespace(rank=r) for r in (0, 1)))
    kwargs = dict(request_id='r', incarnation='i', device='cpu', vector_space='Q',
        capacity=32, max_new=16, top_k=4, batched_cache_install=True)
    for extra in ({'batched_bank_install': True}, {'gpu_receive_to_bank': True},
                  {'staged_transport': True}, {'install_scratch_bytes': 2 * cache_install_tensor_bound(32) - 1}):
        with pytest.raises(ValueError, match='batched cache install'):
            OasisLayerTransport(None, selected, **kwargs, **extra)


@pytest.mark.parametrize('dtype', ['float16', 'bfloat16', 'float32'])
def test_baseline_cpu_receive_keeps_existing_small_shape_contract(dtype):
    manifest, wire, payloads, _ = payload_case()
    manifest = replace(manifest, dtype='torch.' + dtype, head_dim=4)
    rows = manifest.nbytes // torch.empty((), dtype=getattr(torch, dtype)).element_size()
    source = torch.arange(rows).to(getattr(torch, dtype)).view(torch.uint8)
    registry = SimpleNamespace(combine_reserve_start=False, _owner=lambda: None, batched_cache_install=False)
    record = OasisCPUReceiveRecord(registry, manifest, SimpleNamespace(transfer_id='fixture'), None)
    record._buffer, record._ready, record._safe = source, True, True
    cache = [{} for _ in range(4)]
    record.copy_to_cache(cache)
    assert record.profile['cache_kv_bytes'] == manifest.nbytes
    assert record.profile['cache_d2h_fenced'] is False


@pytest.mark.parametrize('bad', [None, 'mode', 'd2h', 'complete', 'bytes', 'clone', 'copies', 'index'])
def test_live_cache_profile_gate_refuses_unproven_or_changed_budget(bad):
    spec = importlib.util.spec_from_file_location('delivery_validation',
        Path(__file__).resolve().parents[3] / 'benchmark/pvd_oasis_delivery_validation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = dict(cache_install_mode='batched', cache_install_complete=True, cache_d2h_fenced=True,
        cache_installed_rows=5, cache_groups=2, cache_kv_bytes=2560, nbytes=2560,
        remote_rows=5, cache_row_clones=0, cache_kv_copy_calls=2,
        cache_valid_write_calls=2, cache_index_bytes=40,
        cache_tensor_bound_bytes=cache_install_tensor_bound(32))
    changes = dict(mode=('cache_install_mode', 'rows'), d2h=('cache_d2h_fenced', False),
        complete=('cache_install_complete', False), bytes=('cache_kv_bytes', 2559),
        clone=('cache_row_clones', 5), copies=('cache_kv_copy_calls', 5), index=('cache_index_bytes', 0))
    if bad:
        name, value = changes[bad]
        profile[name] = value
        with pytest.raises(AssertionError):
            module.validate_cache_install_profile(profile, batched=True)
    else:
        module.validate_cache_install_profile(profile, batched=True)


def test_cpu_policy_d2h_unknown_retains_destination_owners_and_budget(monkeypatch):
    async def run():
        async with slot_case(monkeypatch) as c:
            c.registry.batched_cache_install = True
            cache = [_HeadCPUCache(torch.zeros((8, 2, 128), dtype=torch.float16),
                torch.zeros(8, dtype=torch.bool)) for _ in range(4)]
            r = prepare(c)
            await ready(c, r)
            def fail():
                raise RuntimeError('D2H fence completion unknown')
            monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: SimpleNamespace(synchronize=fail))
            with pytest.raises(RuntimeError, match='D2H fence'):
                r.copy_to_cache(cache)
            assert not r._installed and not any(e.valid.any() for e in cache)
            assert r._cache_copy_owners and r._local_unknown == 'Oasis cache D2H completion unknown'
            assert not await r.close()
            assert c.budget.snapshot()['used_staging_bytes'] == 4096
            assert c.budget.snapshot()['used_inflight'] == 1
            assert c.pool.snapshot()['unknown_slots'] == 1
    asyncio.run(run())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real CUDA required')
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('ids', [((7, 1, 4), (2, 9)), ((0,), (11,)), ((1, 2, 3), (8, 9, 10))])
def test_real_cuda_d2h_nondefault_stream_and_cpu_owned_bytes(batched, ids):
    # Actual CUDA ordering/D2H/install. Source is local Torch, readiness is an
    # explicit fixture, so this is not native RDMA or terminal-proof validation.
    manifest, wire, payloads, cache = payload_case(ids)
    expected = [(p.spec, p.tensor.clone()) for p in payloads]
    device = torch.device('cuda', torch.cuda.current_device())
    registry = SimpleNamespace(combine_reserve_start=False, _owner=lambda: None,
        device=device, ordering=CUDAReceiveOrdering(device),
        batched_cache_install=batched, cache_capacity=12)
    record = OasisCUDAReceiveRecord(registry, manifest, SimpleNamespace(transfer_id='fixture'), None)
    record._buffer = wire.to(device)
    record._registration = SimpleNamespace(buffer=record._buffer)
    record._ready = record._safe = True
    with torch.cuda.stream(torch.cuda.Stream(device=device)):
        record.copy_to_cache(cache)
    assert record._ordered and record._installed and not record._cache_copy_owners
    assert record.profile['cache_install_mode'] == ('batched' if batched else 'rows')
    record._buffer.zero_()
    for spec, value in expected:
        assert torch.equal(cache[spec.kv_head].rows[list(spec.token_ids)], value.permute(1, 0, 2))
        assert cache[spec.kv_head].valid[list(spec.token_ids)].all()
