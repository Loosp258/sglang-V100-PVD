"""Actual Torch bank assembly; CPU byte proof and separate genuine CUDA gates."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.pvd.oasis_bank_install import install_batched_bank, install_tensor_bound
from sglang.srt.disaggregation.pvd.oasis_qwen import PromptBank
from sglang.srt.disaggregation.pvd.oasis_transport import _HeadCPUCache, OasisLayerTransport


def source_cache():
    # Head/token/component distinctions expose incorrect flattened indexing.
    rows = torch.arange(4 * 12 * 2 * 128).reshape(4, 12, 2, 128)
    rows = ((rows % 997) - 498).to(torch.float16)
    return [_HeadCPUCache(rows[h].clone(), torch.ones(12, dtype=torch.bool)) for h in range(4)]


def oracle(ids, cache, device='cpu'):
    width = max(map(len, ids))
    keys = torch.zeros((4, width, 128), dtype=torch.float16)
    values, valid = torch.zeros_like(keys), torch.zeros((4, width), dtype=torch.bool)
    for h, tokens in enumerate(ids):
        for i, t in enumerate(tokens):
            keys[h, i], values[h, i] = cache[h][t][0], cache[h][t][1]
            valid[h, i] = True
    return PromptBank(ids, keys.to(device), values.to(device), valid.to(device))


CASES = [
    (((), (), (), ()), None),
    (((7, 1, 4), (4,), (), (1, 2)), None),
    (((7, 1, 4), (4,), (), (1, 2)), ((4, 7), (1, 4, 8), (9,), (2,))),
    (((1, 2), (9, 8), (4,), (7,)), ((2, 1, 3), (8, 9), (4,), (7,))),
    (((8, 7), (), (2, 1), (5, 4)), ((1, 2), (9,), (), (6,))),
]


@pytest.mark.parametrize('chosen,old', CASES)
def test_exact_mixed_head_layout_and_kv_budget(chosen, old):
    cache = source_cache()
    resident = None if old is None else oracle(old, cache)
    expected = oracle(chosen, cache)
    retained = []
    k, v, mask, profile = install_batched_bank(chosen, resident, cache, device='cpu',
        capacity=12, prompt_tokens=12, retained=retained)
    assert torch.equal(k, expected.keys) and torch.equal(v, expected.values)
    assert torch.equal(mask, expected.valid)
    hits = sum(t in old[h] for h, ids in enumerate(chosen) for t in ids) if old else 0
    misses = sum(map(len, chosen)) - hits
    assert profile['resident_rows'] == hits and profile['cpu_rows'] == misses
    assert profile['kv_h2d_bytes'] == misses * 512
    assert profile['kv_h2d_calls'] == int(misses > 0)
    assert profile['resident_gather_calls'] == profile['resident_scatter_calls'] == 2 * int(hits > 0)
    assert profile['cpu_scatter_calls'] == 2 * int(misses > 0)
    assert profile['index_metadata_bytes'] == (2 * hits + misses) * 8
    assert not profile['cuda']
    assert any(x is k for x in retained) and any(x is v for x in retained)


def test_resident_hits_do_not_read_cpu_cache_or_copy_padding():
    cache = source_cache()
    old = ((8, 7, 6), (5,), (), (3, 2))
    resident = oracle(old, cache)
    # Padding deliberately differs; only logical old ids can be gathered.
    resident.keys[~resident.valid] = 1000
    resident.values[~resident.valid] = -1000
    chosen = ((6, 8), (5,), (), (2,))
    expected = oracle(chosen, cache)
    k, v, valid, stats = install_batched_bank(chosen, resident, ({},) * 4,
        device='cpu', capacity=12, prompt_tokens=12, retained=[])
    assert torch.equal(k, expected.keys) and torch.equal(v, expected.values)
    assert torch.equal(valid, expected.valid) and stats['cpu_rows'] == 0


@pytest.mark.parametrize('bad', ['duplicate', 'negative', 'bool', 'bound', 'heads', 'uncached', 'dtype', 'resident'])
def test_invalid_later_head_rejects_before_allocation(monkeypatch, bad):
    cache = source_cache()
    chosen, resident = [(1,), (), (), (2,)], None
    if bad == 'duplicate': chosen[3] = (2, 2)
    if bad == 'negative': chosen[3] = (-1,)
    if bad == 'bool': chosen[3] = (True,)
    if bad == 'bound': chosen[3] = (12,)
    if bad == 'heads': chosen = chosen[:3]
    if bad == 'uncached': cache[3].valid[2] = False
    if bad == 'dtype': cache[3] = {2: torch.zeros((2, 128), dtype=torch.float32)}
    if bad == 'resident':
        resident = oracle(((1,), (), (), (2,)), cache)
        resident.values = resident.values.float()
    def no_allocation(*args, **kwargs):
        raise AssertionError('output allocated before validation')
    monkeypatch.setattr(torch, 'zeros', no_allocation)
    retained = []
    with pytest.raises((ValueError, KeyError)):
        install_batched_bank(chosen, resident, cache, device='cpu', capacity=12,
            prompt_tokens=12, retained=retained)
    assert not retained


@pytest.mark.parametrize('fail_at', [1, 2, 3, 4])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='real CUDA required'))])
def test_partial_install_retains_owners_before_exception(monkeypatch, fail_at, device):
    cache = source_cache()
    resident = oracle(((1,), (2,), (3,), (4,)), cache, device)
    chosen = ((1, 8), (2,), (9,), (4, 10))
    original = torch.Tensor.index_copy_
    calls = 0
    used = []
    def fail(self, dim, indexes, source):
        nonlocal calls
        calls += 1
        used.append((self, source))
        if calls == fail_at:
            raise RuntimeError('injected partial install')
        return original(self, dim, indexes, source)
    monkeypatch.setattr(torch.Tensor, 'index_copy_', fail)
    retained = []
    with pytest.raises(RuntimeError, match='partial install'):
        install_batched_bank(chosen, resident, cache, device=device, capacity=12,
            prompt_tokens=12, retained=retained)
    assert any(x is resident for x in retained)
    storage = {x.untyped_storage().data_ptr() for x in retained if isinstance(x, torch.Tensor)}
    assert all(x.untyped_storage().data_ptr() in storage for pair in used for x in pair)
    if device == 'cuda':
        torch.cuda.synchronize()  # retention above precedes caller drain proof
    assert torch.equal(resident.keys.cpu(), oracle(resident.ids, cache).keys)


def test_admission_rejects_mixed_modes_and_insufficient_scratch():
    layout = SimpleNamespace(num_layers=28, total_kv_heads=4, kv_heads_per_rank=2,
        head_dim=128, kv_dtype='torch.float16')
    selected = SimpleNamespace(manifest=SimpleNamespace(layout=layout),
        shards=tuple(SimpleNamespace(rank=r) for r in (0, 1)))
    kw = dict(request_id='r', incarnation='i', device='cpu', vector_space='Q',
        capacity=32, max_new=16, top_k=4, batched_bank_install=True)
    for extra in ({'gpu_receive_to_bank': True}, {'staged_transport': True},
                  {'install_scratch_bytes': 2 * install_tensor_bound(32) - 1}):
        with pytest.raises(ValueError, match='batched install'):
            OasisLayerTransport(None, selected, **kw, **extra)
    with pytest.raises(ValueError, match='routes required'):
        OasisLayerTransport(None, selected, **dict(kw, batched_bank_install=1))


@pytest.mark.parametrize('changed', [None, 'mode', 'cuda', 'completion_proven', 'kv_h2d_bytes',
    'kv_h2d_calls', 'resident_gather_calls', 'cpu_rows', 'tensor_bound_bytes', 'install_seconds'])
def test_live_gate_requires_actual_fenced_operation_budget(changed):
    spec = importlib.util.spec_from_file_location('delivery_validation',
        Path(__file__).resolve().parents[3] / 'benchmark/pvd_oasis_delivery_validation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = dict(mode='batched', cuda=True, completion_proven=True, install_seconds=.005,
        selected_rows=16, resident_rows=12, cpu_rows=4, kv_h2d_bytes=2048, kv_h2d_calls=1,
        resident_gather_calls=2, resident_scatter_calls=2, cpu_scatter_calls=2,
        tensor_bound_bytes=install_tensor_bound(32), index_metadata_bytes=28 * 8)
    bad = dict(mode='per_head', cuda=False, completion_proven=False, kv_h2d_bytes=2049,
        kv_h2d_calls=4, resident_gather_calls=8, cpu_rows=3, tensor_bound_bytes=32 << 20,
        install_seconds=float('nan'))
    if changed:
        profile[changed] = bad[changed]
        with pytest.raises(AssertionError):
            module.validate_bank_install_profile(profile, batched=True)
    else:
        module.validate_bank_install_profile(profile, batched=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real CUDA required')
@pytest.mark.parametrize('chosen,old', CASES[1:])
def test_real_cuda_nondefault_stream_fence_and_bits(chosen, old):
    cache = source_cache()
    device = torch.device('cuda', torch.cuda.current_device())
    resident = None if old is None else oracle(old, cache, device)
    expected = oracle(chosen, cache)
    producer = torch.cuda.Event()
    producer.record()
    if resident is not None:
        resident.completion = producer
    stream = torch.cuda.Stream(device=device)
    retained = []
    with torch.cuda.stream(stream):
        k, v, mask, stats = install_batched_bank(chosen, resident, cache,
            device=device, capacity=12, prompt_tokens=12, retained=retained)
        complete = torch.cuda.Event()
        complete.record()
    complete.synchronize()  # same completion proof as serving caller
    assert stats['cuda']
    assert torch.equal(k.cpu(), expected.keys) and torch.equal(v.cpu(), expected.values)
    assert torch.equal(mask.cpu(), expected.valid)
    assert any(isinstance(x, torch.Tensor) and x.device.type == 'cpu'
               and x.is_pinned() for x in retained)
