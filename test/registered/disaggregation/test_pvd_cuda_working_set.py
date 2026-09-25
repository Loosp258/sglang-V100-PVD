"""Ownership tests use CPU storage explicitly. Separate CUDA cases are not emulated."""

import threading

import pytest
import torch
from sglang.srt.disaggregation.pvd.cuda_working_set import CUDASparseWorkingSet
from sglang.srt.disaggregation.pvd.sparse_install import (
    CPUInstallGroup,
    InstallProtocolError,
)
from sglang.srt.disaggregation.pvd.sparse_payload import (
    SparseKVPayload,
    SparsePayloadError,
)
from sglang.srt.disaggregation.pvd.sparse_working_set import CPUSparseWorkingSet
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferBudget,
    TransferCapacityError,
)
from test_pvd_sparse_working_set import payloads


def options():
    return {
        "request_id": "r",
        "incarnation": "inc",
        "entry_transfer_id": "entry",
        "layout_fingerprint": "layout",
        "expected_groups": ((0, 0), (0, 1)),
        "prompt_tokens": 4,
        "head_dim": 3,
        "max_union_tokens": 3,
    }


def packed_payloads(
    boundary=0, tokens=(0, 1, 2, 3), *, device="cpu", dtype=torch.float32
):
    full = torch.arange(48, dtype=torch.float32).reshape(2, 4, 2, 3)
    source = payloads(full, boundary, tokens)
    backing = torch.cat([p.tensor.reshape(-1) for p in source]).to(
        device=device, dtype=dtype
    )
    size = source[0].tensor.numel()
    views = [
        SparseKVPayload(p.spec, backing[i * size : (i + 1) * size].view(p.tensor.shape))
        for i, p in enumerate(source)
    ]
    released = []
    guard = ResourceGuard(backing, lambda: released.append(True))
    return views, guard, released


def policy_bank(monkeypatch, *, limit=4096, contiguous_stage_copy=False):
    # No CUDA allocation/forward is executed. Exercise ownership on real CPU data.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    budget = TransferBudget(limit, 3)
    bank = CUDASparseWorkingSet(
        device="cuda:0",
        dtype=torch.float32,
        budget=budget,
        contiguous_stage_copy=contiguous_stage_copy,
        **options(),
    )
    bank.device = torch.device("cpu")
    syncs = []
    monkeypatch.setattr(bank, "_synchronize", lambda: syncs.append(True))
    return bank, budget, syncs


def test_opt_in_contiguous_bank_copy_clones_once_and_owns_its_bytes(monkeypatch):
    bank, budget, syncs = policy_bank(monkeypatch, contiguous_stage_copy=True)
    rows, guard, _ = packed_payloads()
    source = guard.value
    original = torch.Tensor.clone
    cloned = []

    def count_clone(tensor, *args, **kwargs):
        cloned.append(tensor.numel() * tensor.element_size())
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", count_clone)
    bank.stage(rows, source_guard=guard)
    assert cloned == [source.numel() * source.element_size()]
    assert syncs == [True]
    assert budget.snapshot()["used_staging_bytes"] == 192
    bank.install(0)
    expected = rows[0].tensor.detach().clone()
    source.zero_()
    with bank.read() as groups:
        assert torch.equal(groups[(0, 0)][1], expected)
        assert (
            groups[(0, 0)][1].untyped_storage().data_ptr()
            == groups[(0, 1)][1].untyped_storage().data_ptr()
        )
    bank.close()
    guard.request_release()
    assert budget.snapshot()["used_staging_bytes"] == 0


def test_opt_in_contiguous_bank_copy_refuses_gap_before_cuda_copy(monkeypatch):
    bank, budget, _ = policy_bank(monkeypatch, contiguous_stage_copy=True)
    rows, _, _ = packed_payloads()
    count = rows[0].tensor.numel()
    backing = torch.empty(2 * count + 1, dtype=torch.float32)
    backing[:count].copy_(rows[0].tensor.flatten())
    backing[count + 1 :].copy_(rows[1].tensor.flatten())
    separated = [
        SparseKVPayload(rows[0].spec, backing[:count].view(rows[0].tensor.shape)),
        SparseKVPayload(rows[1].spec, backing[count + 1 :].view(rows[1].tensor.shape)),
    ]
    guard = ResourceGuard(backing, lambda: None)
    with pytest.raises(SparsePayloadError, match="exact source extent"):
        bank.stage(separated, source_guard=guard)
    assert budget.snapshot()["used_staging_bytes"] == 0
    guard.request_release()
    bank.close()


def test_opt_in_contiguous_bank_copy_retains_storage_on_unknown_fence(monkeypatch):
    bank, budget, _ = policy_bank(monkeypatch, contiguous_stage_copy=True)
    rows, guard, released = packed_payloads()

    def unknown_completion():
        raise RuntimeError("unknown")

    monkeypatch.setattr(bank, "_synchronize", unknown_completion)
    with pytest.raises(RuntimeError, match="unknown"):
        bank.stage(rows, source_guard=guard)
    guard.request_release()
    assert not released
    assert bank._pending_copy is not None
    assert bank._retained_stage is not None
    assert bank._retained_source_guard is not None
    assert budget.snapshot()["used_staging_bytes"] == 192


def test_stage_source_pin_is_held_through_completion_and_readers_block_switch(
    monkeypatch,
):
    bank, budget, syncs = policy_bank(monkeypatch)
    initial, guard, released = packed_payloads()

    def synchronize():
        assert budget.snapshot()["used_staging_bytes"] == 192
        guard.request_release()
        assert not released and guard.value is not None
        syncs.append(True)

    monkeypatch.setattr(bank, "_synchronize", synchronize)
    bank.stage(initial, source_guard=guard)
    assert released == [True] and guard.value is None
    assert syncs == [True]
    bank.install(0)
    monkeypatch.setattr(bank, "_synchronize", lambda: syncs.append(True))
    next_rows, next_guard, _ = packed_payloads(4, (1, 3))
    bank.stage(next_rows, source_guard=next_guard)
    assert budget.snapshot()["used_staging_bytes"] == 288
    with bank.read() as groups:
        assert groups[(0, 0)][0].token_ids == (0, 1, 2, 3)
        with pytest.raises(SparsePayloadError, match="forward"):
            bank.install(4)
    assert len(syncs) == 3
    bank.install(4)
    assert budget.snapshot()["used_staging_bytes"] == 96
    bank.close()
    bank.close()
    next_guard.request_release()
    assert budget.snapshot()["used_staging_bytes"] == 0
    assert not isinstance(bank, CPUSparseWorkingSet)


@pytest.mark.parametrize("drain_fails", [False, True])
def test_partial_copy_failure_drains_before_refund_or_quarantines(
    monkeypatch, drain_fails
):
    bank, budget, _ = policy_bank(monkeypatch)
    rows, guard, released = packed_payloads()
    original = torch.Tensor.clone
    calls = []

    def clone(tensor, *a, **kw):
        calls.append(True)
        if len(calls) == 2:
            guard.request_release()
            assert not released
            raise RuntimeError("partial copy")
        return original(tensor, *a, **kw)

    def synchronize():
        assert not released
        assert budget.snapshot()["used_staging_bytes"] == 192
        if drain_fails:
            raise RuntimeError("completion unknown")

    monkeypatch.setattr(torch.Tensor, "clone", clone)
    monkeypatch.setattr(bank, "_synchronize", synchronize)
    with pytest.raises(
        RuntimeError, match="completion unknown" if drain_fails else "partial copy"
    ):
        bank.stage(rows, source_guard=guard)
    if drain_fails:
        assert bank.snapshot()["retained_stage"]
        assert bank.snapshot()["source_guard_held"]
        assert not released
        assert budget.snapshot()["used_staging_bytes"] == 192
        for action in (bank.close, bank.discard_next, lambda: bank.install(0)):
            with pytest.raises(SparsePayloadError, match="quarantined"):
                action()
    else:
        assert released == [True]
        assert bank.snapshot()["quarantine"] is None
        assert budget.snapshot()["used_staging_bytes"] == 0
        bank.close()


def test_reader_completion_unknown_never_drops_reader_or_current_charge(monkeypatch):
    bank, budget, _ = policy_bank(monkeypatch)
    rows, guard, _ = packed_payloads()
    bank.stage(rows, source_guard=guard)
    bank.install(0)

    def fail():
        raise RuntimeError("reader completion unknown")

    monkeypatch.setattr(bank, "_synchronize", fail)
    with pytest.raises(RuntimeError, match="reader completion"), bank.read():
        pass
    assert bank.snapshot()["readers"] == 1
    assert budget.snapshot()["used_staging_bytes"] == 192
    with pytest.raises(SparsePayloadError, match="quarantined"):
        bank.close()
    guard.request_release()  # input copy was drained before the read, not current storage


@pytest.mark.parametrize(
    "bad", ["foreign_storage", "outside_guard", "not_payload", "dtype"]
)
def test_bad_source_refused_without_bank_charge_or_copy(monkeypatch, bad):
    bank, budget, syncs = policy_bank(monkeypatch)
    rows, guard, released = packed_payloads()
    if bad == "foreign_storage":
        rows[0] = SparseKVPayload(rows[0].spec, rows[0].tensor.clone())
    elif bad == "outside_guard":
        guard = ResourceGuard(guard.value[:1], lambda: released.append(True))
    elif bad == "not_payload":
        rows[0] = object()
    else:
        bank.dtype = torch.float16
    with pytest.raises(SparsePayloadError):
        bank.stage(rows, source_guard=guard)
    assert not syncs
    assert budget.snapshot()["used_staging_bytes"] == 0
    guard.request_release()
    assert released == [True]


def test_next_capacity_refusal_keeps_current_and_releases_input_pin(monkeypatch):
    bank, budget, _ = policy_bank(monkeypatch, limit=192)
    rows, guard, _ = packed_payloads()
    bank.stage(rows, source_guard=guard)
    bank.install(0)
    next_rows, next_guard, released = packed_payloads(4, (1, 3))
    with pytest.raises(TransferCapacityError):
        bank.stage(next_rows, source_guard=next_guard)
    next_guard.request_release()
    assert released == [True]
    assert bank.snapshot()["current_boundary"] == 0
    assert budget.snapshot()["used_staging_bytes"] == 192
    bank.close()
    guard.request_release()


def test_foreign_owner_thread_refused(monkeypatch):
    bank, _, _ = policy_bank(monkeypatch)
    errors = []

    def run():
        try:
            bank.close()
        except SparsePayloadError as exc:
            errors.append(str(exc))

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=5)
    assert errors and "owner thread" in errors[0]
    bank.close()


def test_completed_copies_with_unknown_fence_keep_all_owners(monkeypatch):
    bank, budget, _ = policy_bank(monkeypatch)
    rows, guard, released = packed_payloads()

    def fail():
        guard.request_release()
        raise RuntimeError("unknown fence")

    monkeypatch.setattr(bank, "_synchronize", fail)
    with pytest.raises(RuntimeError, match="unknown fence"):
        bank.stage(rows, source_guard=guard)
    assert len(bank._retained_stage[1]) == 2
    assert bank.snapshot()["next_boundary"] is None
    assert bank.snapshot()["source_guard_held"]
    assert budget.snapshot()["used_staging_bytes"] == 192
    assert not released


def test_source_release_exception_quarantines_staged_bank(monkeypatch):
    bank, budget, _ = policy_bank(monkeypatch)
    rows, original_guard, _ = packed_payloads()

    def fail():
        raise RuntimeError("unregister failed")

    guard = ResourceGuard(original_guard.value, fail)
    monkeypatch.setattr(bank, "_synchronize", guard.request_release)
    with pytest.raises(RuntimeError, match="unregister failed"):
        bank.stage(rows, source_guard=guard)
    assert bank.snapshot()["quarantine"] == "source release outcome unknown"
    assert bank.snapshot()["source_guard_held"]
    assert bank.snapshot()["next_boundary"] == 0
    assert budget.snapshot()["used_staging_bytes"] == 192
    with pytest.raises(SparsePayloadError, match="quarantined"):
        bank.install(0)


def test_exception_in_reader_still_drains_before_releasing_lease(monkeypatch):
    bank, budget, syncs = policy_bank(monkeypatch)
    rows, guard, _ = packed_payloads()
    bank.stage(rows, source_guard=guard)
    bank.install(0)
    with pytest.raises(RuntimeError, match="forward failed"), bank.read():
        raise RuntimeError("forward failed")
    assert len(syncs) == 2
    assert bank.snapshot()["readers"] == 0
    bank.close()
    guard.request_release()
    assert budget.snapshot()["used_staging_bytes"] == 0


def test_cuda_bank_cannot_enter_cpu_install_group(monkeypatch):
    bank, _, _ = policy_bank(monkeypatch)
    with pytest.raises(InstallProtocolError, match="CPU"):
        CPUInstallGroup({0: bank}, interval=4, lead_tokens=1)
    bank.close()


def test_no_cuda_is_an_explicit_failure(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SparsePayloadError, match="unavailable"):
        CUDASparseWorkingSet(
            device="cuda:0",
            dtype=torch.float32,
            budget=TransferBudget(1024, 2),
            **options(),
        )


@pytest.mark.parametrize("device", ["cpu", "cuda", "meta"])
def test_constructor_never_guesses_device_or_falls_back(device):
    with pytest.raises(SparsePayloadError, match="indexed CUDA"):
        CUDASparseWorkingSet(
            device=device,
            dtype=torch.float32,
            budget=TransferBudget(1024, 2),
            **options(),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="real CUDA bank unverified without CUDA"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_real_cuda_bank_copy_stream_reader_and_switch(dtype):
    budget = TransferBudget(4096, 3)
    bank = CUDASparseWorkingSet(
        device="cuda:0", dtype=dtype, budget=budget, **options()
    )
    rows, guard, released = packed_payloads(device="cuda:0", dtype=dtype)
    expected = rows[0].tensor.clone()
    stream = torch.cuda.Stream(device="cuda:0")
    stream.wait_stream(torch.cuda.current_stream(device="cuda:0"))
    with torch.cuda.stream(stream):
        bank.stage(rows, source_guard=guard)
        bank.install(0)
        rows[0].tensor.zero_()
        with bank.read() as groups:
            observed = groups[(0, 0)][1].clone()
        complete = torch.cuda.Event()
        complete.record(stream)
    complete.synchronize()
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    next_rows, next_guard, _ = packed_payloads(4, (1, 3), device="cuda:0", dtype=dtype)
    bank.stage(next_rows, source_guard=next_guard)
    bank.install(4)
    bank.close()
    guard.request_release()
    next_guard.request_release()
    assert released == [True]
    assert budget.snapshot()["used_staging_bytes"] == 0
