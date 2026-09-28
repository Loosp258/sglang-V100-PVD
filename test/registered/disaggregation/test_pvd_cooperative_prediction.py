import threading

import pytest

from sglang.srt.disaggregation.pvd.cooperative_prediction import (
    CooperativePredictionError,
    CooperativePredictionStepper,
    PredictionStep,
)


class _Arbiter:
    def __init__(self):
        self._lease = None

    @property
    def busy(self):
        return self._lease is not None

    def acquire(self):
        if self.busy:
            raise RuntimeError("busy")
        self._lease = object()
        return self._lease

    def release(self, lease):
        assert lease is self._lease
        self._lease = None


def test_one_forward_per_turn_releases_both_target_owners():
    lock, arbiter = threading.RLock(), _Arbiter()
    stepper = CooperativePredictionStepper(lock, arbiter)
    forwards = []
    values = iter(("draft-token-1", "draft-token-2"))

    def step(forward_once):
        forwards.append(forward_once("batch"))
        return PredictionStep(done=len(forwards) == 2, value=forwards[-1])

    stepper.submit(
        "req-a",
        "prefix-1",
        step=step,
        forward=lambda batch: next(values),
    )

    first = stepper.run_one_after_decode(10)
    assert first.status == "progressed"
    assert first.step_index == 0
    assert first.value is None
    assert stepper.pending == 1
    assert not arbiter.busy
    assert lock.acquire(blocking=False)
    lock.release()

    second = stepper.run_one_after_decode(11)
    assert second.status == "completed"
    assert second.value == "draft-token-2"
    assert stepper.pending == 0
    assert forwards == ["draft-token-1", "draft-token-2"]
    assert not arbiter.busy
    assert lock.acquire(blocking=False)
    lock.release()


def test_busy_target_owner_yields_without_waiting_or_running_step():
    lock, arbiter = threading.Lock(), _Arbiter()
    stepper = CooperativePredictionStepper(lock, arbiter)
    called = []
    stepper.submit(
        "req-a",
        1,
        step=lambda forward: (called.append(True), PredictionStep(False))[1],
        forward=lambda batch: batch,
    )

    lock.acquire()
    try:
        result = stepper.run_one_after_decode(1)
    finally:
        lock.release()

    assert result.status == "busy"
    assert not arbiter.busy
    assert called == []


def test_stale_prefix_is_dropped_without_forwarding():
    lock, arbiter = threading.RLock(), _Arbiter()
    stepper = CooperativePredictionStepper(lock, arbiter)
    forwards = []
    stepper.submit(
        "req-a",
        "old-prefix",
        step=lambda forward: PredictionStep(done=True),
        forward=lambda batch: forwards.append(batch),
        prefix_is_current=lambda: False,
    )

    result = stepper.run_one_after_decode(1)

    assert result.status == "stale"
    assert result.request_id == "req-a"
    assert forwards == []
    assert stepper.pending == 0
    assert not arbiter.busy


def test_unfinished_jobs_rotate_and_turn_cannot_be_reused():
    lock, arbiter = threading.RLock(), _Arbiter()
    stepper = CooperativePredictionStepper(lock, arbiter)
    calls = []

    def make_step(request_id):
        def step(forward):
            calls.append(request_id)
            forward(request_id)
            return PredictionStep(done=request_id == "req-b")

        return step

    for request_id in ("req-a", "req-b"):
        stepper.submit(
            request_id,
            1,
            step=make_step(request_id),
            forward=lambda batch: batch,
        )

    assert stepper.run_one_after_decode(5).request_id == "req-a"
    assert stepper.run_one_after_decode(6).request_id == "req-b"
    assert calls == ["req-a", "req-b"]
    with pytest.raises(CooperativePredictionError, match="once per increasing"):
        stepper.run_one_after_decode(6)


def test_second_forward_in_one_step_quarantines_target_ownership():
    lock, arbiter = threading.Lock(), _Arbiter()
    stepper = CooperativePredictionStepper(lock, arbiter)
    stepper.submit(
        "req-a",
        1,
        step=lambda forward: (
            forward("first"),
            forward("second"),
            PredictionStep(done=True),
        )[-1],
        forward=lambda batch: batch,
    )

    with pytest.raises(CooperativePredictionError, match="more than one"):
        stepper.run_one_after_decode(1)

    assert arbiter.busy
    assert not lock.acquire(blocking=False)
    with pytest.raises(CooperativePredictionError, match="quarantined"):
        stepper.run_one_after_decode(2)
