"""CPU contract/numerical tests, not serving kernel or CUDA fence evidence."""

import math
from dataclasses import replace

import pytest
import torch
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.sparse_payload import (
    SparseKVPayload,
    SparseKVSpec,
    SparsePayloadError,
)
from sglang.srt.disaggregation.pvd.sparse_working_set import (
    CPUSparseWorkingSet,
    reference_attention,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferBudget,
    TransferCapacityError,
)


def fixture(capacity=4096):
    budget = TransferBudget(capacity, 3)
    bank = CPUSparseWorkingSet(
        request_id="r",
        incarnation="inc",
        entry_transfer_id="entry",
        layout_fingerprint="layout",
        expected_groups=((0, 0), (0, 1)),
        prompt_tokens=4,
        head_dim=3,
        max_union_tokens=3,
        budget=budget,
    )
    generator = torch.Generator().manual_seed(41)
    full = torch.randn(2, 4, 2, 3, generator=generator)
    return bank, budget, full


def payloads(full, boundary=0, tokens=(0, 1, 2, 3)):
    return [
        SparseKVPayload(
            SparseKVSpec(
                "r",
                "inc",
                f"op-{boundary}",
                boundary,
                "entry",
                "index",
                "mapping",
                "layout",
                0,
                head,
                tokens,
            ),
            full[:, list(tokens), head].clone(),
        )
        for head in range(2)
    ]


def used(budget):
    return budget.snapshot()["used_staging_bytes"]


def test_stage_owns_copy_switch_requires_boundary_and_reader_release():
    bank, budget, full = fixture()
    initial = payloads(full)
    bank.stage(initial)
    for payload in initial:
        payload.tensor.zero_()
        payload.close()
    with pytest.raises(SparsePayloadError, match="not installed"), bank.read():
        pytest.fail("uninstalled bank exposed")
    bank.install(0)
    initial_bytes = used(budget)
    bank.stage(payloads(full, 4, (1, 3)))
    assert used(budget) == initial_bytes + initial_bytes // 2
    with bank.read() as groups:
        torch.testing.assert_close(groups[(0, 0)][1], full[:, :, 0])
        with pytest.raises(SparsePayloadError, match="forward"):
            bank.install(4)
        with pytest.raises(SparsePayloadError, match="forward"):
            bank.close()
    for wrong in (3, 5, True):
        with pytest.raises(SparsePayloadError, match="exact prepared boundary"):
            bank.install(wrong)
    bank.install(4)
    assert used(budget) == initial_bytes // 2
    with bank.read() as groups:
        assert groups[(0, 1)][0].token_ids == (1, 3)
    bank.close()
    bank.close()
    assert used(budget) == 0


def test_cancel_next_preserves_current_and_releases_only_next():
    bank, budget, full = fixture()
    bank.stage(payloads(full))
    bank.install(0)
    initial_bytes = used(budget)
    bank.stage(payloads(full, 4, (0, 2)))
    with pytest.raises(SparsePayloadError, match="already staged"):
        bank.stage(payloads(full, 4, (0,)))
    bank.discard_next()
    assert used(budget) == initial_bytes
    with bank.read() as groups:
        assert groups[(0, 0)][0].target_tokens == 0
    bank.close()


@pytest.mark.parametrize(
    "defect", ["missing", "identity", "mixed", "initial_subset", "dtype"]
)
def test_bad_initial_bank_rejected_before_reservation(defect):
    bank, budget, full = fixture()
    rows = payloads(full)
    if defect == "missing":
        rows.pop()
    elif defect == "identity":
        rows[0] = SparseKVPayload(
            replace(rows[0].spec, incarnation="other"), rows[0].tensor
        )
    elif defect == "mixed":
        rows[0] = SparseKVPayload(
            replace(rows[0].spec, operation_id="other"), rows[0].tensor
        )
    elif defect == "initial_subset":
        rows = payloads(full, tokens=(0, 1))
    else:
        rows[0] = SparseKVPayload(rows[0].spec, rows[0].tensor.half())
    with pytest.raises(SparsePayloadError):
        bank.stage(rows)
    assert used(budget) == 0


@pytest.mark.parametrize("boundary,tokens", [(0, (0,)), (4, (0, 1, 2, 3))])
def test_refresh_rejects_stale_boundary_or_union_overflow(boundary, tokens):
    bank, budget, full = fixture()
    bank.stage(payloads(full))
    bank.install(0)
    before = used(budget)
    with pytest.raises(SparsePayloadError, match="stale boundary or union capacity"):
        bank.stage(payloads(full, boundary, tokens))
    assert used(budget) == before
    bank.close()


def test_capacity_failure_keeps_old_bank():
    bank, budget, full = fixture(192)  # full initial bank only
    bank.stage(payloads(full))
    bank.install(0)
    with pytest.raises(TransferCapacityError):
        bank.stage(payloads(full, 4, (0, 1)))
    assert used(budget) == 192
    with bank.read() as groups:
        torch.testing.assert_close(groups[(0, 0)][1], full[:, :, 0])
    bank.close()


def test_partial_copy_failure_releases_new_reservation(monkeypatch):
    bank, budget, full = fixture()
    rows = payloads(full)
    original = torch.Tensor.clone
    calls = 0

    def clone(tensor, *args, **kwargs):
        nonlocal calls
        calls += 1
        assert used(budget) == 192
        if calls == 2:
            raise RuntimeError("copy failed")
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", clone)
    with pytest.raises(RuntimeError, match="copy failed"):
        bank.stage(rows)
    assert used(budget) == 0


@pytest.mark.parametrize("sparse", [False, True])
def test_attention_matches_independent_softmax_preserves_generated_and_masks_positions(
    sparse,
):
    bank, budget, full = fixture()
    bank.stage(payloads(full))
    bank.install(0)
    tokens = (0, 1, 2, 3)
    if sparse:
        tokens = (3, 1)  # deliberately not compact-position order
        bank.stage(payloads(full, 4, tokens))
        bank.install(4)
    generator = torch.Generator().manual_seed(77)
    q = torch.randn(4, 3, generator=generator)
    generated_k = torch.randn(3, 2, 3, generator=generator)
    generated_v = torch.randn(3, 2, 3, generator=generator)
    generated_v[2] = 1000  # future KV must never enter attention
    before_k, before_v = generated_k.clone(), generated_v.clone()
    expected = []
    for head in range(4):
        kv_head = head // 2
        keys = torch.cat((full[0, list(tokens), kv_head], generated_k[:2, kv_head]))
        values = torch.cat((full[1, list(tokens), kv_head], generated_v[:2, kv_head]))
        scores = (keys @ q[head]) / math.sqrt(3)
        expected.append(scores.softmax(dim=0) @ values)
    with bank.read() as groups:
        actual = reference_attention(
            groups,
            q,
            layer=0,
            query_position=5,
            mapping=QueryHeadMapping(4, 2),
            prompt_tokens=4,
            generated_positions=(4, 5, 6),
            generated_k=generated_k,
            generated_v=generated_v,
        )
    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(generated_k, before_k, rtol=0, atol=0)
    torch.testing.assert_close(generated_v, before_v, rtol=0, atol=0)
    bank.close()
    assert used(budget) == 0


def test_absolute_prompt_positions_not_compacted_row_numbers():
    bank, _, full = fixture()
    bank.stage(payloads(full))
    bank.install(0)
    bank.stage(payloads(full, 4, (3, 1)))
    bank.install(4)
    with bank.read() as groups:
        output = reference_attention(
            groups,
            torch.ones(4, 3),
            layer=0,
            query_position=1,
            mapping=QueryHeadMapping(4, 2),
            prompt_tokens=4,
            generated_positions=(),
            generated_k=torch.empty(0, 2, 3),
            generated_v=torch.empty(0, 2, 3),
        )
    torch.testing.assert_close(output, full[1, 1].repeat_interleave(2, dim=0))
    bank.close()
