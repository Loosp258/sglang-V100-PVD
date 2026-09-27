"""Single-owner background execution for bounded private PVD prediction.

This module is an execution primitive, not serving integration. A worker owns
one prediction callback from entry through its completion fence. Requests and
results cross the thread boundary as immutable snapshots and Future values;
the Scheduler thread must call :meth:`ConcurrentPredictionWorker.poll` to
complete those Futures. A cancellation request never releases callback-owned
resources early.

The callback must keep all mutable Req/KV/model branch state on the worker
thread, check cancellation at safe boundaries, and return only after its
private resource cleanup is complete. The worker then fences its stream
before publishing the result. Any callback, context, or fence failure
quarantines the worker and retains its stream-side owners.
"""

from __future__ import annotations

import inspect
import math
import queue
import threading
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, TimeoutError
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from typing_extensions import Self

_T = TypeVar("_T")


class PredictionWorkerError(RuntimeError):
    """Base error for the private prediction worker."""


class PredictionWorkerBusyError(PredictionWorkerError):
    """The worker still owns a submitted or not-yet-polled prediction."""


class PredictionWorkerClosedError(PredictionWorkerError):
    """The worker is closing or closed."""


class PredictionWorkerQuarantinedError(PredictionWorkerError):
    """The worker cannot be reused after uncertain execution or cleanup."""


class PredictionWorkerStartupError(PredictionWorkerError):
    """The worker thread could not initialize its stream."""


class PredictionCancelledError(CancelledError):
    """Cooperative callback cancellation requested at a safe boundary."""


class PredictionWorkerTimeout(TimeoutError, PredictionWorkerError):
    """The worker did not drain before a bounded close timeout."""


@dataclass(frozen=True, slots=True)
class PredictionJob:
    """Immutable request incarnation and committed-prefix snapshot.

    ``prefix_tokens`` must be a tuple so a scheduler cannot change the
    prediction input after submission. No live ``Req`` or allocator object
    belongs in this message.
    """

    request_id: str
    incarnation: str
    prefix_version: str
    committed_position: int
    prefix_tokens: tuple[int, ...]
    committed_query_positions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ("incarnation", self.incarnation),
            ("prefix_version", self.prefix_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if type(self.committed_position) is not int or self.committed_position < 0:
            raise ValueError("committed_position must be a nonnegative integer")
        if (
            not isinstance(self.prefix_tokens, tuple)
            or not self.prefix_tokens
            or any(type(token) is not int or token < 0 for token in self.prefix_tokens)
        ):
            raise ValueError("prefix_tokens must be a nonempty tuple of token IDs")
        if self.committed_position >= len(self.prefix_tokens):
            raise ValueError("committed_position must identify a token in the prefix")
        if not isinstance(self.committed_query_positions, tuple) or any(
            type(position) is not int
            or position < 0
            or position >= len(self.prefix_tokens)
            for position in self.committed_query_positions
        ):
            raise ValueError(
                "committed_query_positions must be a tuple of positions in the prefix"
            )


class PredictionContext:
    """Worker-local stream and cooperative cancellation view for a callback."""

    __slots__ = ("_cancel_event", "job", "stream")

    def __init__(
        self,
        job: PredictionJob,
        stream: Any,
        cancel_event: threading.Event,
    ) -> None:
        self.job = job
        self.stream = stream
        self._cancel_event = cancel_event

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancel_requested:
            raise PredictionCancelledError(
                f"prediction cancelled for request {self.job.request_id}"
            )


class PredictionFuture(Generic[_T]):
    """Future facade whose cancellation waits for worker cleanup and polling.

    Calling ``cancel()`` signals the callback. The underlying standard Future
    becomes cancelled only when the Scheduler thread polls a completion after
    callback cleanup and the worker's stream fence.
    """

    __slots__ = ("_future", "_worker", "job")

    def __init__(self, worker: ConcurrentPredictionWorker, job: PredictionJob):
        self.job = job
        self._worker = worker
        self._future: Future[_T] = Future()

    def cancel(self) -> bool:
        return self._worker.cancel(self)

    def cancelled(self) -> bool:
        return self._future.cancelled()

    def done(self) -> bool:
        return self._future.done()

    def running(self) -> bool:
        return self._future.running()

    def result(self, timeout: float | None = None) -> _T:
        return self._future.result(timeout=timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self._future.exception(timeout=timeout)

    def add_done_callback(self, fn: Callable[[PredictionFuture[_T]], Any]) -> None:
        if not callable(fn):
            raise TypeError("done callback must be callable")
        self._future.add_done_callback(lambda _future: fn(self))

    def _set_result(self, value: _T) -> None:
        self._future.set_result(value)

    def _set_exception(self, error: BaseException) -> None:
        self._future.set_exception(error)

    def _set_cancelled(self) -> None:
        self._future.cancel()

    def __repr__(self) -> str:
        return (
            f"PredictionFuture(request_id={self.job.request_id!r}, "
            f"done={self.done()}, cancelled={self.cancelled()})"
        )


@dataclass(slots=True)
class _Task:
    job: PredictionJob
    future: PredictionFuture[Any]
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass(frozen=True, slots=True)
class _Completion:
    task: _Task
    value: Any = None
    error: BaseException | None = None
    cancelled: bool = False


def _default_stream_context(stream: Any) -> AbstractContextManager[Any]:
    del stream
    return nullcontext()


def _default_completion_fence(stream: Any) -> None:
    """Synchronize only the worker stream, never the whole CUDA device."""
    if stream is None:
        return
    synchronize = getattr(stream, "synchronize", None)
    if not callable(synchronize):
        raise PredictionWorkerError(
            "a worker stream without synchronize() needs an explicit completion_fence"
        )
    synchronize()


class ConcurrentPredictionWorker:
    """Single-flight prediction thread with Scheduler-owned Future completion.

    ``callback(job, context)`` runs entirely on the worker thread. The optional
    stream factory also runs there once at startup. ``context_factory(stream)``
    creates a context manager around each callback (for example,
    ``torch.cuda.stream(stream)``). ``completion_fence(stream)`` runs after
    callback/context cleanup and before a result is made pollable.

    Only the construction thread may submit, cancel, poll, or close. The worker
    exposes its own ``owner_thread_id`` for startup code that must bind
    thread-affine predictor state after the thread starts.
    """

    def __init__(
        self,
        callback: Callable[[PredictionJob, PredictionContext], _T],
        *,
        stream_factory: Callable[[], Any] | None = None,
        context_factory: Callable[[Any], AbstractContextManager[Any]] | None = None,
        completion_fence: Callable[[Any], None] | None = None,
        max_prefix_tokens: int = 65536,
        startup_timeout: float = 10.0,
        name: str = "pvd-prediction-worker",
    ) -> None:
        if (
            not callable(callback)
            or inspect.iscoroutinefunction(callback)
            or (stream_factory is not None and not callable(stream_factory))
            or (context_factory is not None and not callable(context_factory))
            or (completion_fence is not None and not callable(completion_fence))
        ):
            raise PredictionWorkerError("synchronous worker callbacks are required")
        if type(max_prefix_tokens) is not int or max_prefix_tokens <= 0:
            raise PredictionWorkerError("max_prefix_tokens must be positive")
        if (
            isinstance(startup_timeout, bool)
            or not isinstance(startup_timeout, (int, float))
            or not math.isfinite(startup_timeout)
            or startup_timeout <= 0
        ):
            raise PredictionWorkerError("startup_timeout must be finite and positive")
        if not isinstance(name, str) or not name.strip():
            raise PredictionWorkerError("worker thread name is required")

        self._callback = callback
        self._stream_factory = stream_factory
        self._context_factory = context_factory or _default_stream_context
        self._completion_fence = completion_fence or _default_completion_fence
        self.max_prefix_tokens = max_prefix_tokens
        self._scheduler_thread_id = threading.get_ident()
        self._worker_thread_id: int | None = None
        self._lock = threading.Lock()
        self._commands: queue.Queue[_Task] = queue.Queue(maxsize=1)
        self._completions: queue.SimpleQueue[_Completion] = queue.SimpleQueue()
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._closing = False
        self._state = "starting"
        self._stream: Any = None
        self._active: _Task | None = None
        self._quarantine_error: BaseException | None = None
        self._quarantine_retained: tuple[Any, ...] | None = None
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=name,
            daemon=False,
        )
        self._thread.start()
        if not self._ready.wait(float(startup_timeout)):
            with self._lock:
                self._closing = True
            self._thread.join(float(startup_timeout))
            raise PredictionWorkerStartupError("worker thread startup timed out")
        if self._startup_error is not None:
            self._thread.join(float(startup_timeout))
            raise PredictionWorkerStartupError(
                f"worker stream initialization failed: {self._startup_error}"
            ) from self._startup_error

    @property
    def owner_thread_id(self) -> int:
        """The worker thread identity, available after constructor startup."""
        self._ready.wait()
        ident = self._worker_thread_id
        if ident is None:
            raise PredictionWorkerStartupError("worker thread has no identity")
        return ident

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active is not None

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def _owner(self) -> None:
        if threading.get_ident() != self._scheduler_thread_id:
            raise PredictionWorkerError(
                "submit, cancel, poll, and close belong to the Scheduler thread"
            )

    def submit(self, job: PredictionJob) -> PredictionFuture[Any]:
        self._owner()
        if not isinstance(job, PredictionJob):
            raise PredictionWorkerError("an immutable PredictionJob is required")
        if len(job.prefix_tokens) > self.max_prefix_tokens:
            raise PredictionWorkerError("prediction prefix exceeds its token bound")
        future = PredictionFuture(self, job)
        task = _Task(job, future)
        with self._lock:
            if self._state == "quarantined":
                raise PredictionWorkerQuarantinedError(
                    "prediction worker is quarantined"
                )
            if self._closing or self._state in ("closed", "closing"):
                raise PredictionWorkerClosedError("prediction worker is closing")
            if self._active is not None:
                raise PredictionWorkerBusyError(
                    "the previous prediction has not been polled and retired"
                )
            self._active = task
            self._state = "running"
            try:
                self._commands.put_nowait(task)
            except queue.Full as exc:  # Defensive invariant check.
                self._active = None
                self._state = "idle"
                raise PredictionWorkerBusyError(
                    "prediction command queue is full"
                ) from exc
        return future

    def cancel(self, future: PredictionFuture[Any]) -> bool:
        """Request cancellation without completing its Future or releasing it."""
        self._owner()
        if not isinstance(future, PredictionFuture) or future._worker is not self:
            raise PredictionWorkerError("Future belongs to another prediction worker")
        if future.done():
            return False
        with self._lock:
            task = self._active
            if task is None or task.future is not future:
                return False
            if task.cancel_event.is_set():
                return True
            task.cancel_event.set()
            return True

    def poll(self) -> int:
        """Deliver completed Futures on the Scheduler thread; return count."""
        self._owner()
        delivered = 0
        while True:
            try:
                completion = self._completions.get_nowait()
            except queue.Empty:
                break

            task = completion.task
            with self._lock:
                if self._active is not task:
                    self._quarantine_locked(
                        PredictionWorkerError("completion does not own the active job"),
                        (task, self._stream),
                    )
                    error = self._quarantine_error
                else:
                    self._active = None
                    error = self._quarantine_error or completion.error
                    if self._state not in ("quarantined", "closing", "closed"):
                        self._state = "idle"

            if error is not None:
                task.future._set_exception(error)
            elif completion.cancelled:
                task.future._set_cancelled()
            else:
                task.future._set_result(completion.value)
            delivered += 1
        return delivered

    def snapshot(self) -> dict[str, Any]:
        """Return bounded owner-thread state without exposing payloads."""
        self._owner()
        with self._lock:
            active = self._active
            return {
                "state": self._state,
                "busy": active is not None,
                "request_id": None if active is None else active.job.request_id,
                "incarnation": None if active is None else active.job.incarnation,
                "cancel_requested": (
                    False if active is None else active.cancel_event.is_set()
                ),
                "owner_thread_id": self._worker_thread_id,
                "quarantine_error": (
                    None
                    if self._quarantine_error is None
                    else type(self._quarantine_error).__name__
                ),
                "thread_alive": self._thread.is_alive(),
            }

    def close(self, *, timeout: float | None = 5.0) -> None:
        """Stop submissions, cancel active work, fence, poll, and join.

        On timeout the worker and all active owners remain retained; call
        ``close``/``poll`` again after the cooperative callback drains. A
        callback or fence exception exits the worker and quarantines it, so
        close still joins promptly and never waits on a dead worker.
        """
        self._owner()
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise PredictionWorkerError("close timeout must be nonnegative or None")
        with self._lock:
            if self._state != "quarantined":
                self._state = "closing"
            self._closing = True
            if self._active is not None:
                self._active.cancel_event.set()
        self._thread.join(timeout)
        self.poll()
        if self._thread.is_alive():
            raise PredictionWorkerTimeout(
                "prediction callback has not drained; worker ownership is retained"
            )
        with self._lock:
            if self._state != "quarantined":
                self._state = "closed"
                self._stream = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _run(self) -> None:
        self._worker_thread_id = threading.get_ident()
        try:
            if self._stream_factory is not None:
                self._stream = self._stream_factory()
        except BaseException as exc:  # noqa: BLE001 - quarantine every thread exit.
            with self._lock:
                self._startup_error = exc
                self._quarantine_locked(exc, (self._stream_factory,))
                self._state = "quarantined"
            self._ready.set()
            self._stopped.set()
            return
        with self._lock:
            if not self._closing:
                self._state = "idle"
        self._ready.set()

        try:
            while True:
                with self._lock:
                    if self._closing and self._commands.empty():
                        break
                try:
                    task = self._commands.get(timeout=0.05)
                except queue.Empty:
                    continue
                if not self._run_task(task):
                    break
        except BaseException as exc:  # noqa: BLE001 - quarantine every thread exit.
            with self._lock:
                active = self._active
                retained = (active, self._stream, self._callback)
                self._quarantine_locked(exc, retained)
                self._state = "quarantined"
            if active is not None:
                self._completions.put(_Completion(active, error=exc))
        finally:
            with self._lock:
                if self._state not in ("quarantined", "closed"):
                    self._state = "closed" if self._closing else "idle"
            self._stopped.set()

    def _run_task(self, task: _Task) -> bool:
        if task.cancel_event.is_set():
            self._completions.put(_Completion(task, cancelled=True))
            return True

        context: PredictionContext | None = None
        context_manager: Any = None
        value: Any = None
        error: BaseException | None = None
        fence_error: BaseException | None = None
        try:
            context = PredictionContext(task.job, self._stream, task.cancel_event)
            context_manager = self._context_factory(self._stream)
            if not callable(
                getattr(context_manager, "__enter__", None)
            ) or not callable(getattr(context_manager, "__exit__", None)):
                raise PredictionWorkerError(
                    "context_factory must return a synchronous context manager"
                )
            with context_manager:
                value = self._callback(task.job, context)
                if inspect.isawaitable(value):
                    close = getattr(value, "close", None)
                    if callable(close):
                        close()
                    raise PredictionWorkerError(
                        "prediction callback must return synchronously"
                    )
        except BaseException as exc:  # noqa: BLE001 - preserve callback failures.
            error = exc
        finally:
            # This fence also runs after callback/context errors because a
            # failing forward may have enqueued device work before raising.
            try:
                self._completion_fence(self._stream)
            except BaseException as exc:  # noqa: BLE001 - fence failure is terminal.
                fence_error = exc

        if error is not None or fence_error is not None:
            cancelled = (
                isinstance(error, PredictionCancelledError)
                and task.cancel_event.is_set()
                and fence_error is None
            )
            primary = (
                fence_error
                if isinstance(error, PredictionCancelledError)
                and fence_error is not None
                else error or fence_error
            )
            if error is not None and fence_error is not None:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note(f"completion fence also failed: {fence_error}")
            if not cancelled or fence_error is not None:
                with self._lock:
                    self._quarantine_locked(
                        primary,
                        (
                            task,
                            context,
                            context_manager,
                            value,
                            self._stream,
                            self._callback,
                        ),
                    )
                    self._state = "quarantined"
            self._completions.put(
                _Completion(
                    task, error=None if cancelled else primary, cancelled=cancelled
                )
            )
            return cancelled

        if task.cancel_event.is_set():
            # Do not publish a result built from a request being retired.
            self._completions.put(_Completion(task, cancelled=True))
        else:
            self._completions.put(_Completion(task, value=value))
        return True

    def _quarantine_locked(
        self, error: BaseException, retained: tuple[Any, ...]
    ) -> None:
        self._quarantine_error = error
        # Keep stream-side owners alive when completion or cleanup is unknown.
        self._quarantine_retained = retained
        self._state = "quarantined"


__all__ = [
    "ConcurrentPredictionWorker",
    "PredictionCancelledError",
    "PredictionContext",
    "PredictionFuture",
    "PredictionJob",
    "PredictionWorkerBusyError",
    "PredictionWorkerClosedError",
    "PredictionWorkerError",
    "PredictionWorkerQuarantinedError",
    "PredictionWorkerStartupError",
    "PredictionWorkerTimeout",
]
