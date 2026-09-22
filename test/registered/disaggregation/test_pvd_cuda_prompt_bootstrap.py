"""Index-independent Prompt import with real CPU tensors, no CUDA/RDMA."""

from dataclasses import replace

import pytest
import torch
from sglang.srt.disaggregation.pvd.cuda_prompt_bootstrap import (
    CUDAPromptBootstrap,
    CUDAPromptPoolSource,
)
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferBudget,
    TransferCapacityError,
)
from test_pvd_cuda_model_attention import fixture
from test_pvd_cuda_runtime_group import make_group


def setup(monkeypatch, limit=65536):
    group, bank_budget = make_group(monkeypatch)
    c = fixture(monkeypatch)
    c.req.req_to_token[1, :4] = torch.tensor([8, 3, 6, 1])
    budget = TransferBudget(limit, 2)
    importer = CUDAPromptBootstrap(group, execution_lock=c.lock, staging_budget=budget)
    monkeypatch.setattr(importer, "_synchronize", lambda: None)
    source = CUDAPromptPoolSource(group._banks[0].identity, 1, 4, c.owner)
    return c, group, importer, source, budget, bank_budget


def test_full_prompt_import_needs_no_index_and_preserves_absolute_positions(
    monkeypatch,
):
    c, group, importer, source, budget, _ = setup(monkeypatch)
    original_k, original_v = c.pool.k.clone(), c.pool.v.clone()
    epoch = importer.install(source)
    assert epoch.target_tokens == 0 and group.can_decode(0)
    assert budget.snapshot()["reservations"] == 0 and not importer._held
    permit = group.runtime.begin_forward(0)
    with group.read(0, 0) as groups:
        for (_, head), (spec, tensor) in groups.items():
            assert spec.token_ids == (0, 1, 2, 3)
            assert spec.index_version == "full-prompt:no-index"
            expected = torch.stack(
                (c.pool.k[[8, 3, 6, 1], head], c.pool.v[[8, 3, 6, 1], head])
            )
            torch.testing.assert_close(tensor, expected, atol=0, rtol=0)
    assert group.runtime.finish_forward(permit, readers_drained=True, succeeded=True)
    torch.testing.assert_close(c.pool.k, original_k, atol=0, rtol=0)
    torch.testing.assert_close(c.pool.v, original_v, atol=0, rtol=0)
    with pytest.raises(InstallProtocolError, match="consumed"):
        importer.install(source)
    group.close()


@pytest.mark.parametrize("fault", ["entry", "count", "slot", "rows", "padding", "dtype"])
def test_bad_source_never_makes_decode_ready(monkeypatch, fault):
    c, group, importer, source, budget, _ = setup(monkeypatch)
    if fault == "entry":
        source = replace(source, identity=("r", "inc", "foreign", "layout"))
    elif fault == "count":
        source = replace(source, prompt_tokens=3)
    elif fault == "slot":
        source = replace(source, slot=True)
    elif fault == "rows":
        c.req.req_to_token[1, 1] = c.req.req_to_token[1, 0]
    elif fault == "padding":
        c.req.req_to_token[1, 1] = 0
    else:
        c.pool.k = c.pool.k.to(torch.float16)
    with pytest.raises(InstallProtocolError):
        importer.install(source)
    assert not group.can_decode(0)
    assert budget.snapshot()["reservations"] == 0
    assert not importer._quarantined
    group.close()


def test_capacity_refusal_precedes_pool_pin_and_allocation(monkeypatch):
    c, group, importer, source, budget, _ = setup(monkeypatch, limit=1)
    with pytest.raises(TransferCapacityError):
        importer.install(source)
    assert not importer._used and not importer._held
    assert budget.snapshot()["reservations"] == 0
    c.owner.request_release()
    assert c.released == [True]
    group.close()


@pytest.mark.parametrize("fail_at", [1, 2])
def test_copy_or_retirement_unknown_retains_pool_staging_and_lock(monkeypatch, fail_at):
    c, group, importer, source, budget, bank_budget = setup(monkeypatch)
    calls = 0

    def sync():
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise RuntimeError("completion unknown")

    monkeypatch.setattr(importer, "_synchronize", sync)
    with pytest.raises(RuntimeError, match="completion unknown"):
        importer.install(source)
    c.owner.request_release()
    assert importer._quarantined and importer._held["source"] is source
    assert budget.snapshot()["used_staging_bytes"] > 0 and not c.released
    assert not group.can_decode(0)
    with pytest.raises(InstallProtocolError, match="quarantined"):
        importer.install(source)
    assert calls == fail_at, "UNKNOWN must not be probed again to justify refund"


def test_bank_budget_refusal_drains_staging_without_leak(monkeypatch):
    c, group, importer, source, budget, _ = setup(monkeypatch)
    group._banks[0].budget = TransferBudget(1, 1)
    with pytest.raises(TransferCapacityError):
        importer.install(source)
    assert budget.snapshot()["used_staging_bytes"] == 0
    assert not importer._held and not importer._quarantined
    assert not group.can_decode(0)
    c.owner.request_release()
    assert c.released == [True]
    group.close()
