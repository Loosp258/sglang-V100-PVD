"""CPU lifecycle tests for the private PVD prediction worker."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest
from sglang.srt.disaggregation.pvd.concurrent_prediction_worker import (
    ConcurrentPredictionWorker,
    PredictionJob,
    PredictionWorkerBusyError,
    PredictionWorkerError,
    PredictionWorkerQuarantinedError,
    PredictionWorkerTimeout,
)


def _job(request_id: str = "request-1") -> PredictionJob:
    return PredictionJob(
        request_id=request_id,
        incarnation="incarnation-1",
        prefix_version="prefix-v1",
        committed_position=3,
        prefix_tokens=(11, 12, 13, 14),
        committed_query_positions=(1, 3),
    )


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        threading.Event().wait(0.001)


def _poll_until_done(worker, future, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not future.done():
        worker.poll()
        if time.monotonic() >= deadline:
            raise AssertionError("worker completion was not delivered before timeout")
        threading.Event().wait(0.001)


def test_job_captures_immutable_prefix_and_bounded_query_positions():
    job = _job()
    assert job.prefix_tokens == (11, 12, 13, 14)
    assert job.committed_query_positions == (1, 3)

    with pytest.raises(ValueError, match="prefix_tokens"):
        PredictionJob("r", "i", "v", 0, [1, 2])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="committed_query_positions"):
        PredictionJob("r", "i", "v", 1, (1, 2), (2,))
    with pytest.raises(ValueError, match="committed_query_positions"):
        PredictionJob("r", "i", "v", 1, (1, 2), (True,))


def test_callback_context_and_fence_are_worker_owned_and_poll_publishes_on_owner():
    main_thread_id = threading.get_ident()
    stream = object()
    calls: list[tuple[str, int]] = []
    callback_entered = threading.Event()
    callback_release = threading.Event()
    fence_entered = threading.Event()
    fence_release = threading.Event()
    done_callback_thread: list[int] = []

    @contextmanager
    def stream_context(actual_stream):
        assert actual_stream is stream
        calls.append(("context-enter", threading.get_ident()))
        try:
            yield
        finally:
            calls.append(("context-exit", threading.get_ident()))

    def callback(job, context):
        assert job.committed_query_positions == (1, 3)
        assert context.stream is stream
        calls.append(("callback", threading.get_ident()))
        callback_entered.set()
        assert callback_release.wait(2.0)
        return ("q0", "q1")

    def completion_fence(actual_stream):
        assert actual_stream is stream
        calls.append(("fence", threading.get_ident()))
        fence_entered.set()
        assert fence_release.wait(2.0)

    worker = ConcurrentPredictionWorker(
        callback,
        stream_factory=lambda: stream,
        context_factory=stream_context,
        completion_fence=completion_fence,
    )
    try:
        assert worker.owner_thread_id != main_thread_id
        future = worker.submit(_job())
        future.add_done_callback(
            lambda _future: done_callback_thread.append(threading.get_ident())
        )

        assert callback_entered.wait(2.0)
        assert worker.busy
        assert worker.snapshot()["busy"] is True
        with pytest.raises(PredictionWorkerBusyError):
            worker.submit(_job("request-2"))

        callback_release.set()
        assert fence_entered.wait(2.0)
        assert not future.done()
        assert worker.busy  # completion fence and owner poll have not retired it
        fence_release.set()

        _poll_until_done(worker, future)
        assert future.result() == ("q0", "q1")
        assert not worker.busy
        assert done_callback_thread == [main_thread_id]
        assert [name for name, _ in calls] == [
            "context-enter",
            "callback",
            "context-exit",
            "fence",
        ]
        assert all(thread_id == worker.owner_thread_id for _, thread_id in calls)
    finally:
        callback_release.set()
        fence_release.set()
        worker.close()


def test_cooperative_cancel_waits_for_callback_cleanup_and_fence_then_reuses_worker():
    callback_entered = threading.Event()
    allow_callback_exit = threading.Event()
    cleanup_finished = threading.Event()
    fence_entered = threading.Event()
    allow_fence_exit = threading.Event()

    def callback(_job, context):
        callback_entered.set()
        assert allow_callback_exit.wait(2.0)
        try:
            context.raise_if_cancelled()
        finally:
            cleanup_finished.set()

    def completion_fence(_stream):
        fence_entered.set()
        assert allow_fence_exit.wait(2.0)

    worker = ConcurrentPredictionWorker(callback, completion_fence=completion_fence)
    try:
        future = worker.submit(_job())
        assert callback_entered.wait(2.0)
        assert future.cancel()
        assert not future.done()
        assert not future.cancelled()

        allow_callback_exit.set()
        assert cleanup_finished.wait(2.0)
        assert fence_entered.wait(2.0)
        assert not future.done()
        assert worker.busy

        allow_fence_exit.set()
        _poll_until_done(worker, future)
        assert future.cancelled()
        assert worker.state == "idle"
        assert not worker.busy

        next_future = worker.submit(_job("request-2"))
        _poll_until_done(worker, next_future)
        assert next_future.result() is None
    finally:
        allow_callback_exit.set()
        allow_fence_exit.set()
        worker.close()


def test_callback_exception_quarantines_worker_and_close_does_not_deadlock():
    fence_finished = threading.Event()

    def callback(_job, _context):
        raise ValueError("private prediction failed")

    def completion_fence(_stream):
        fence_finished.set()

    worker = ConcurrentPredictionWorker(callback, completion_fence=completion_fence)
    future = worker.submit(_job())
    assert fence_finished.wait(2.0)
    _poll_until_done(worker, future)

    error = future.exception()
    assert isinstance(error, ValueError)
    assert str(error) == "private prediction failed"
    assert worker.state == "quarantined"
    assert not worker.busy
    with pytest.raises(PredictionWorkerQuarantinedError):
        worker.submit(_job("request-2"))

    worker.close(timeout=1.0)
    assert worker.snapshot()["thread_alive"] is False
    assert worker.state == "quarantined"


def test_fence_exception_is_quarantined_and_reported_as_future_exception():
    def callback(_job, _context):
        return ("q",)

    def completion_fence(_stream):
        raise RuntimeError("stream fence failed")

    worker = ConcurrentPredictionWorker(callback, completion_fence=completion_fence)
    future = worker.submit(_job())
    _poll_until_done(worker, future)
    assert isinstance(future.exception(), RuntimeError)
    assert str(future.exception()) == "stream fence failed"
    assert worker.state == "quarantined"
    worker.close(timeout=1.0)


def test_close_timeout_keeps_active_job_until_callback_and_fence_drain():
    callback_entered = threading.Event()
    release_callback = threading.Event()
    fence_finished = threading.Event()

    def callback(_job, _context):
        callback_entered.set()
        assert release_callback.wait(2.0)

    def completion_fence(_stream):
        fence_finished.set()

    worker = ConcurrentPredictionWorker(callback, completion_fence=completion_fence)
    future = worker.submit(_job())
    assert callback_entered.wait(2.0)

    with pytest.raises(PredictionWorkerTimeout):
        worker.close(timeout=0.01)
    assert worker.busy
    assert not future.done()

    release_callback.set()
    assert fence_finished.wait(2.0)
    worker.close(timeout=2.0)
    assert future.cancelled()
    assert worker.state == "closed"
    assert not worker.busy


def test_scheduler_thread_owns_submit_and_poll():
    worker = ConcurrentPredictionWorker(lambda _job, _context: "ok")
    errors: list[PredictionWorkerError] = []
    try:
        thread = threading.Thread(
            target=lambda: _capture_error(worker.poll, errors), daemon=True
        )
        thread.start()
        thread.join(2.0)
        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], PredictionWorkerError)
        assert "Scheduler thread" in str(errors[0])
    finally:
        worker.close()


def test_graceful_close_retires_private_cache_on_worker_stream():
    owner_id = threading.get_ident()
    stream = object()
    calls = []

    @contextmanager
    def stream_context(actual_stream):
        assert actual_stream is stream
        calls.append("enter")
        try:
            yield
        finally:
            calls.append("exit")

    def retire():
        assert threading.get_ident() != owner_id
        calls.append("retire")

    def fence(actual_stream):
        assert actual_stream is stream
        calls.append("fence")

    worker = ConcurrentPredictionWorker(
        lambda _job, _context: "ok",
        stream_factory=lambda: stream,
        context_factory=stream_context,
        completion_fence=fence,
        shutdown_callback=retire,
    )
    worker.close()
    worker.close()
    assert worker.state == "closed"
    assert calls == ["enter", "retire", "exit", "fence"]


def test_quarantined_worker_keeps_private_cache_owner():
    retired = []

    def fail(_job, _context):
        raise RuntimeError("failed forward")

    worker = ConcurrentPredictionWorker(
        fail,
        shutdown_callback=lambda: retired.append(True),
    )
    future = worker.submit(_job())
    _poll_until_done(worker, future)
    worker.close()
    assert worker.state == "quarantined"
    assert retired == []


def test_shutdown_cleanup_failure_quarantines_worker_after_fence():
    calls = []

    def retire():
        calls.append("retire")
        raise RuntimeError("cleanup failed")

    worker = ConcurrentPredictionWorker(
        lambda _job, _context: "ok",
        completion_fence=lambda _stream: calls.append("fence"),
        shutdown_callback=retire,
    )
    worker.close()
    assert calls == ["retire", "fence"]
    assert worker.state == "quarantined"
    assert worker.snapshot()["quarantine_error"] == "RuntimeError"


def _capture_error(callback, errors: list[PredictionWorkerError]) -> None:
    try:
        callback()
    except PredictionWorkerError as exc:
        errors.append(exc)
