import dataclasses
import threading

import pytest

from sglang.srt.disaggregation.pvd.transfer_engine import TransferHandle
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferBudget,
    TransferCapacityError,
    TransportState,
)


def test_guard_retains_source_until_transport_unpins():
    released = []
    source = object()
    guard = ResourceGuard(source, lambda: released.append(source))
    guard.pin("write-1")
    guard.request_release()
    assert released == []
    guard.unpin("write-1")
    guard.unpin("write-1")
    assert released == [source]


def test_guard_waits_for_every_owner_and_owner_pins_are_idempotent():
    released = []
    guard = ResourceGuard(object(), lambda: released.append(True))
    guard.pin("write-1")
    guard.pin("write-1")
    guard.pin("write-2")
    guard.request_release()
    guard.unpin("write-1")
    assert released == []
    guard.unpin("write-2")
    assert released == [True]


def test_guard_failed_callback_retains_value_and_can_be_retried():
    source = object()
    attempts = []

    def release():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("release failed")

    guard = ResourceGuard(source, release)
    assert guard.value is source
    with pytest.raises(RuntimeError, match="release failed"):
        guard.request_release()
    assert guard.value is source
    guard.request_release()
    assert attempts == [True, True]
    assert guard.value is None


def test_guard_rejects_new_pins_after_release_is_requested():
    guard = ResourceGuard(object(), lambda: None)
    guard.pin("write-1")
    guard.request_release()
    with pytest.raises(RuntimeError):
        guard.pin("write-2")
    guard.unpin("write-1")
    with pytest.raises(RuntimeError):
        guard.pin("write-3")


def test_guard_callback_runs_outside_lock_and_releasing_rejects_new_pins():
    callback_started = threading.Event()
    allow_callback_to_finish = threading.Event()
    guard = None

    def release():
        callback_started.set()
        assert guard.value is not None
        assert allow_callback_to_finish.wait(timeout=5)

    guard = ResourceGuard(object(), release)
    release_thread = threading.Thread(target=guard.request_release)
    release_thread.start()
    assert callback_started.wait(timeout=5)
    with pytest.raises(RuntimeError):
        guard.pin("late-owner")
    allow_callback_to_finish.set()
    release_thread.join(timeout=5)
    assert not release_thread.is_alive()


def test_budget_rejects_before_allocation():
    budget = TransferBudget(staging_bytes=64, max_inflight=1)
    budget.reserve("a", 64, 1)
    with pytest.raises(TransferCapacityError):
        budget.reserve("b", 1, 1)
    budget.release("a")
    budget.reserve("b", 64, 1)


def test_budget_owner_reservation_is_idempotent_but_cannot_change():
    budget = TransferBudget(staging_bytes=64, max_inflight=2)
    budget.reserve("a", 32, 1)
    budget.reserve("a", 32, 1)
    assert budget.snapshot()["used_staging_bytes"] == 32
    assert budget.snapshot()["used_inflight"] == 1
    with pytest.raises(ValueError):
        budget.reserve("a", 16, 1)


@pytest.mark.parametrize("byte_count,slots", [(0, 1), (1, 0), (0, 0)])
def test_budget_can_reserve_zero_in_either_dimension(byte_count, slots):
    budget = TransferBudget(staging_bytes=1, max_inflight=1)
    budget.reserve("a", byte_count, slots)
    assert budget.snapshot()["used_staging_bytes"] == byte_count
    assert budget.snapshot()["used_inflight"] == slots


@pytest.mark.parametrize(
    "staging_bytes,max_inflight", [(-1, 1), (1, -1), (0, 1), (1, 0)]
)
def test_budget_limits_must_be_positive(staging_bytes, max_inflight):
    with pytest.raises(ValueError):
        TransferBudget(staging_bytes=staging_bytes, max_inflight=max_inflight)


@pytest.mark.parametrize("byte_count,slots", [(-1, 0), (0, -1)])
def test_budget_rejects_negative_reservations(byte_count, slots):
    budget = TransferBudget(staging_bytes=1, max_inflight=1)
    with pytest.raises(ValueError):
        budget.reserve("a", byte_count, slots)


def test_budget_release_is_idempotent():
    budget = TransferBudget(staging_bytes=8, max_inflight=1)
    budget.reserve("a", 8, 1)
    budget.release("a")
    budget.release("a")
    assert budget.snapshot()["used_staging_bytes"] == 0
    assert budget.snapshot()["used_inflight"] == 0


def test_unknown_transport_is_not_safe_and_budget_stays_reserved_until_release():
    budget = TransferBudget(staging_bytes=8, max_inflight=1)
    budget.reserve("a", 8, 1)
    handle = TransferHandle("transfer", transport_state=TransportState.UNKNOWN)
    assert not handle.transport_state.is_locally_safe_to_release
    assert budget.snapshot()["used_staging_bytes"] == 8
    budget.release("a")
    assert budget.snapshot()["used_staging_bytes"] == 0


@pytest.mark.parametrize(
    "state,expected",
    [
        (TransportState.NOT_SUBMITTED, True),
        (TransportState.IN_FLIGHT, False),
        (TransportState.DRAINING, False),
        (TransportState.TERMINAL_SUCCESS, True),
        (TransportState.TERMINAL_FAILED, True),
        (TransportState.UNKNOWN, False),
    ],
)
def test_transport_state_local_release_safety(state, expected):
    assert state.is_locally_safe_to_release is expected


def test_transfer_handle_field_is_appended_with_safe_default():
    legacy = TransferHandle("transfer", "legacy-status", 7, "error", "backend")
    assert legacy.backend_handle == "backend"
    assert legacy.transport_state is TransportState.NOT_SUBMITTED
    assert [field.name for field in dataclasses.fields(TransferHandle)][-1] == (
        "transport_state"
    )
