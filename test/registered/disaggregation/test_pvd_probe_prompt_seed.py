"""Private probe Prompt-KV seeding: identity, copy, and source lifetime."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.cuda_model_attention import CUDAModelPools
from sglang.srt.disaggregation.pvd.prediction import (
    CommittedPrefix,
    PredictionConfigError,
)
from sglang.srt.disaggregation.pvd.target_probe import _LlamaTargetProbeCore
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferBudget,
)


class Pool:
    def __init__(self, *, fill, device="cpu"):
        self.keys = [
            torch.full(
                (8, 1, 2), float(fill + layer), dtype=torch.float32, device=device
            )
            for layer in range(2)
        ]
        self.values = [
            torch.full(
                (8, 1, 2), float(fill + layer + 10), dtype=torch.float32, device=device
            )
            for layer in range(2)
        ]
        for layer in range(2):
            for row in range(8):
                self.keys[layer][row] += row
                self.values[layer][row] += row

    def get_key_buffer(self, layer):
        return self.keys[layer]

    def get_value_buffer(self, layer):
        return self.values[layer]


class Allocator:
    def __init__(self):
        self.allocated = False
        self.mapping = None

    def alloc_kv(self, count):
        assert count == 3
        self.allocated = True
        return [1, 2, 3]

    def write_mapping(self, slot, start, rows):
        self.mapping = (slot, start, tuple(rows))


def case(device="cpu"):
    req = SimpleNamespace(rid="request", req_pool_idx=1, origin_input_ids=(11, 12, 13))
    mapping = torch.zeros((2, 8), dtype=torch.int32, device=device)
    mapping[1, :3] = torch.tensor([3, 5, 7], dtype=torch.int32, device=device)
    source = CUDAModelPools(
        SimpleNamespace(req_to_token=mapping), Pool(fill=20, device=device)
    )
    owner = ResourceGuard(source, lambda: None)
    probe = _LlamaTargetProbeCore.__new__(_LlamaTargetProbeCore)
    probe.prefix_budget = TransferBudget(1024, 1)
    probe._prefix_caches = {"request": SimpleNamespace(req=req)}
    probe._prompt_seed_source = None
    probe.device = device
    probe.dtype = torch.float32
    probe.kv_heads, probe.head_dim, probe.layers, probe.max_tokens = 1, 2, 2, 16
    probe.transient_bytes_bound = 1024
    probe._drain_private = (
        (lambda: torch.cuda.synchronize(device)) if device != "cpu" else (lambda: None)
    )
    record = SimpleNamespace(req=req, tokens=(), rows=[], slot=1)
    resources = SimpleNamespace(pool=Pool(fill=0, device=device), allocator=Allocator())
    prefix = CommittedPrefix("request", (11, 12, 13, 42), 0, "v1")
    return probe, req, source, owner, record, resources, prefix


def test_prompt_seed_copies_only_verified_rows_into_private_kv():
    probe, req, source, owner, record, resources, prefix = case()
    with probe.prompt_seed_scope(req, owner):
        probe._seed_cached_prompt(prefix, record, resources)
        assert owner.value is source
        owner.request_release()
        assert owner.value is source  # The active seed pin prevents retirement.
    assert owner.value is None
    assert resources.allocator.mapping == (1, 0, (1, 2, 3))
    assert record.tokens == req.origin_input_ids
    for layer in range(2):
        for getter in ("get_key_buffer", "get_value_buffer"):
            src = getattr(source.kv_pool, getter)(layer)
            dst = getattr(resources.pool, getter)(layer)
            torch.testing.assert_close(dst[1:4], src[[3, 5, 7]])
            dst[1] = -99
            assert not torch.equal(dst[1], src[3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_prompt_seed_copies_private_rows_on_real_cuda():
    probe, req, source, owner, record, resources, prefix = case("cuda:0")
    with probe.prompt_seed_scope(req, owner):
        probe._seed_cached_prompt(prefix, record, resources)
    for layer in range(2):
        for getter in ("get_key_buffer", "get_value_buffer"):
            src = getattr(source.kv_pool, getter)(layer)
            dst = getattr(resources.pool, getter)(layer)
            torch.testing.assert_close(dst[1:4], src[[3, 5, 7]])


@pytest.mark.parametrize("bad_rows", [(3, 3, 7), (0, 5, 7)])
def test_prompt_seed_rejects_invalid_source_before_allocation(bad_rows):
    probe, req, source, owner, record, resources, prefix = case()
    source.req_pool.req_to_token[1, :3] = torch.tensor(bad_rows)
    with probe.prompt_seed_scope(req, owner):
        with pytest.raises(PredictionConfigError, match="invalid source rows"):
            probe._seed_cached_prompt(prefix, record, resources)
    assert not resources.allocator.allocated


def test_prompt_seed_rejects_foreign_prefix_and_alias():
    probe, req, source, owner, record, resources, prefix = case()
    with probe.prompt_seed_scope(req, owner):
        with pytest.raises(PredictionConfigError, match="identity"):
            probe._seed_cached_prompt(
                CommittedPrefix("request", (99, 12, 13, 42), 0, "v1"),
                record,
                resources,
            )
        resources.pool = source.kv_pool
        with pytest.raises(PredictionConfigError, match="layout"):
            probe._seed_cached_prompt(prefix, record, resources)
    assert not resources.allocator.allocated


def test_prompt_seed_rejects_unreserved_scratch_before_allocating():
    probe, req, _, owner, record, resources, prefix = case()
    probe.transient_bytes_bound = 1
    with probe.prompt_seed_scope(req, owner):
        with pytest.raises(PredictionConfigError, match="scratch bound"):
            probe._seed_cached_prompt(prefix, record, resources)
    assert not resources.allocator.allocated


def test_prompt_seed_scope_requires_exact_request_and_guard():
    probe, req, _, owner, _, _, _ = case()
    with pytest.raises(PredictionConfigError, match="exact live"):
        with probe.prompt_seed_scope(SimpleNamespace(rid=req.rid), owner):
            pass
    with pytest.raises(PredictionConfigError, match="exact live"):
        with probe.prompt_seed_scope(req, object()):
            pass
