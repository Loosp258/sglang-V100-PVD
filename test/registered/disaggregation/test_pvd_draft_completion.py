"""CPU failure injection for draft completion/ownership, not CUDA evidence."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from sglang.srt.disaggregation.pvd.draft_sglang import (
    DraftPlacement,
    DraftWorkerError,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferCapacityError
from test_pvd_draft_sglang import FakeAllocator, FakeExecutor, factory, prefix, provider


class FenceExecutor(FakeExecutor):
    def __init__(self, events, fail_at=0, fail_forward=False):
        super().__init__()
        self.events = events
        self.fail_at = fail_at
        self.fail_forward = fail_forward
        self.drains = 0

    def drain(self):
        self.drains += 1
        self.events.append("drain")
        if self.drains == self.fail_at:
            raise RuntimeError("completion unknown")

    def forward(self, inputs):
        if self.fail_forward:
            raise RuntimeError("forward failed after launch")
        return super().forward(inputs)


class OrderedAllocator(FakeAllocator):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def clear_mapping(self, index):
        self.events.append("clear")
        super().clear_mapping(index)

    def free_kv(self, rows):
        self.events.append("free-kv")
        super().free_kv(rows)

    def free_request(self, index):
        self.events.append("free-request")
        super().free_request(index)


def setup(fail_at=0, fail_forward=False):
    events = []
    executor = FenceExecutor(events, fail_at, fail_forward)
    allocator = OrderedAllocator(events)
    made = provider(
        factory(executor=executor, allocator=allocator),
        placement=DraftPlacement(
            scratch_budget_bytes=1 << 20,
            persistent_budget_bytes=1 << 20,
            max_concurrent_branches=3,
        ),
    )
    return made, executor, allocator, events


def test_release_fences_work_mapping_clear_and_allocator_updates():
    made, _, allocator, events = setup()
    with made.branch():
        made.predict(prefix(), 2)
    assert events == ["drain", "clear", "drain", "free-kv", "free-request", "drain"]
    assert not allocator.live_kv and not allocator.live_requests
    assert made.scratch_budget.snapshot()["used_staging_bytes"] == 0


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_unknown_completion_stops_whole_shared_provider_and_keeps_owner(fail_at):
    made, executor, allocator, events = setup(fail_at)
    with pytest.raises(DraftWorkerError, match="could not be released"):
        with made.branch():
            made.predict(prefix(), 2)
    assert made.degraded and len(made.quarantined) == 1
    assert made.scratch_budget.snapshot()["used_staging_bytes"] > 0
    record = made._retained[made.quarantined[0].branch_id]
    assert record.handle is not None and not record.handle.released
    before = list(events)
    with pytest.raises(DraftWorkerError):
        record.handle.release()
    assert events == before, "UNKNOWN cleanup must not retry a partial free"
    if fail_at < 3:
        assert allocator.live_kv and allocator.live_requests
        assert "free-kv" not in events and "free-request" not in events
    # More admission slots exist, but the allocator/device is shared.
    with pytest.raises(TransferCapacityError, match="quarantined"):
        with made.branch():
            made.predict(prefix(), 2)
    assert executor.drains == fail_at


def test_failed_forward_cleans_up_before_execution_lock_is_released():
    made, executor, allocator, _ = setup(fail_forward=True)
    original = executor.drain

    def drain():
        assert made._execution_lock.locked()
        original()

    executor.drain = drain
    with made.branch():
        with pytest.raises(RuntimeError, match="forward failed"):
            made.predict(prefix(), 2)
        # Even when the caller catches the forward error inside the branch.
        assert not allocator.live_kv and not allocator.live_requests
        assert made.active_branches == 0
    assert executor.drains == 3
    assert not made.degraded


def test_already_admitted_branch_cannot_execute_after_peer_cleanup_fails():
    made, executor, allocator, _ = setup(fail_at=1)
    admitted, proceed = Event(), Event()

    def peer():
        with pytest.raises(DraftWorkerError, match="quarantined"):
            with made.branch():
                admitted.set()
                assert proceed.wait(5)
                made.predict(prefix(), 2)

    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(peer)
        try:
            assert admitted.wait(5)
            with pytest.raises(DraftWorkerError, match="could not be released"):
                with made.branch():
                    made.predict(prefix(), 2)
            requests, calls = allocator.next_request, len(executor.calls)
        finally:
            proceed.set()
        future.result(timeout=5)
    assert allocator.next_request == requests == 1
    assert len(executor.calls) == calls
    assert len(made.quarantined) == len(made._retained) == 2
    assert made.active_branches == 0


def test_executor_without_completion_contract_is_refused_before_allocation():
    from sglang.srt.disaggregation.pvd.draft_sglang import DraftCapabilityError

    made, executor, allocator, _ = setup()
    executor.drain = None
    with pytest.raises(DraftCapabilityError, match="completion drain"):
        with made.branch():
            made.predict(prefix(), 2)
    assert allocator.next_request == 0 and not made.degraded


@pytest.mark.parametrize("device", ["cuda", "meta"])
def test_real_adapter_refuses_ambiguous_or_unsupported_fence_device(device):
    from sglang.srt.disaggregation.pvd.draft_sglang import DraftCapabilityError
    from test_pvd_draft_forward import adapter

    with pytest.raises(DraftCapabilityError, match="explicitly indexed CUDA"):
        adapter(device=device)


def test_real_adapter_fences_its_selected_device_and_propagates_unknown(monkeypatch):
    import torch
    from test_pvd_draft_forward import adapter

    seen = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: seen.append(device))
    adapter(device="cpu").drain()
    assert seen == []
    made = adapter(device="cuda:3")
    made.drain()
    assert seen == [torch.device("cuda:3")]

    def fail(device):
        raise RuntimeError("device completion unknown")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    with pytest.raises(RuntimeError, match="completion unknown"):
        made.drain()


@pytest.mark.parametrize("runner_fails", [False, True])
def test_adapter_unknown_retains_batch_and_refuses_later_forward(
    monkeypatch, runner_fails
):
    from types import SimpleNamespace

    import torch
    from test_pvd_draft_forward import adapter, extend_inputs

    batch = object()
    logits = torch.zeros(8)

    def forward(value):
        assert value is batch
        if runner_fails:
            raise RuntimeError("failed after launch")
        return logits

    made = adapter(SimpleNamespace(forward=forward), device="cuda:2")
    monkeypatch.setattr(made, "build_forward_batch", lambda inputs: batch)

    def fail(device):
        assert device == torch.device("cuda:2")
        assert made._pending_owners[0] is batch
        raise RuntimeError("fence failed")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    with pytest.raises(RuntimeError, match="adapter quarantined"):
        made.forward(extend_inputs())
    assert made._pending_owners[0] is batch
    assert len(made._pending_owners) == (1 if runner_fails else 2)
    if not runner_fails:
        assert made._pending_owners[1] is logits
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    with pytest.raises(RuntimeError, match="adapter quarantined"):
        made.forward(extend_inputs())
    with pytest.raises(RuntimeError, match="adapter quarantined"):
        made.drain()
    assert made.forward_count == 1 and made._pending_owners[0] is batch


@pytest.mark.parametrize("bad_call", [1, 2])
def test_malformed_kv_allocation_still_belongs_to_handle_until_cleanup(bad_call):
    from sglang.srt.disaggregation.pvd.draft_sglang import DraftLifecycleError

    made, _, allocator, _ = setup()
    original = allocator.alloc_kv
    count = 0

    def alloc(size):
        nonlocal count
        count += 1
        return original(size + 1 if count == bad_call else size)

    allocator.alloc_kv = alloc
    with pytest.raises(DraftLifecycleError, match="allocator"):
        with made.branch():
            made.predict(prefix(), 2)
    assert not allocator.live_kv and not allocator.live_requests
    assert not made.degraded
