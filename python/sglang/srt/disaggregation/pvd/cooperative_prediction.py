"""One-forward-per-turn prediction work owned by the Scheduler thread.

This module is an execution seam, not serving integration. The caller invokes
``run_one_after_decode`` once after its ordinary Decode batch has completed.
Each queued job supplies a resumable ``step`` function that may call the
provided ``forward_once`` at most once. The forward callback must synchronously
fence its ModelRunner work before returning (``DraftForwardAdapter.forward``
already does this).

The shared target lock and arbiter are acquired nonblocking for one step and
released before a successful call returns. They are never held over a Scheduler
turn boundary. If a step fails after dispatch may have started, both owners are
retained and the stepper is quarantined; reusing uncertain model state is not
safe.
"""

from __future__ import annotations

import inspect
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Optional


class CooperativePredictionError(RuntimeError):
    """The cooperative prediction queue cannot safely continue."""


@dataclass(frozen=True)
class PredictionStep:
    """Result of one bounded scheduler-owned step.

    ``done=False`` keeps the job eligible for a later turn. ``value`` is
    published only when ``done=True``; intermediate tensors should remain
    owned by the job's private state, not escape through this result.
    """

    done: bool
    value: Any = None

    def __post_init__(self):
        if type(self.done) is not bool:
            raise CooperativePredictionError("step completion must be boolean")


@dataclass(frozen=True)
class PredictionTurn:
    """Observable outcome for one Scheduler turn."""

    status: str
    request_id: Optional[str] = None
    step_index: Optional[int] = None
    value: Any = None


@dataclass
class _PredictionJob:
    request_id: str
    prefix_version: Any
    step: Callable[[Callable[[Any], Any]], PredictionStep]
    forward: Callable[[Any], Any]
    prefix_is_current: Callable[[], bool]
    step_index: int = 0


class CooperativePredictionStepper:
    """Fair, single-owner pump for resumable prediction steps.

    ``forward`` should be a one-call adapter around ``ModelRunner.forward``.
    The step function receives a guarded wrapper rather than the raw callback,
    so a programming error cannot launch two forwards in one Scheduler turn.
    At most one job advances on each strictly increasing ``turn_id``.
    """

    def __init__(self, execution_lock, arbiter, *, max_pending_jobs: int = 8):
        if (
            not callable(getattr(execution_lock, "acquire", None))
            or not callable(getattr(execution_lock, "release", None))
            or not callable(getattr(arbiter, "acquire", None))
            or not callable(getattr(arbiter, "release", None))
            or not hasattr(arbiter, "busy")
        ):
            raise CooperativePredictionError(
                "shared execution lock and target arbiter are required"
            )
        if type(max_pending_jobs) is not int or max_pending_jobs <= 0:
            raise CooperativePredictionError("max_pending_jobs must be positive")
        self.execution_lock = execution_lock
        self.arbiter = arbiter
        self.max_pending_jobs = max_pending_jobs
        self._owner_thread = threading.get_ident()
        self._jobs: Deque[_PredictionJob] = deque()
        self._request_ids = set()
        self._last_turn_id = None
        self._stepping = False
        self._quarantine = None
        self._held_arbiter_lease = None
        self._held_execution_lock = False

    def _owner(self):
        if threading.get_ident() != self._owner_thread:
            raise CooperativePredictionError(
                "prediction steps belong to their Scheduler thread"
            )
        owner = getattr(self.arbiter, "owner", None)
        if callable(owner):
            owner()
        if self._quarantine is not None:
            raise CooperativePredictionError(
                "prediction stepper is quarantined after uncertain execution"
            )

    def submit(
        self,
        request_id: str,
        prefix_version: Any,
        *,
        step: Callable[[Callable[[Any], Any]], PredictionStep],
        forward: Callable[[Any], Any],
        prefix_is_current: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Queue one resumable request branch without executing model work."""

        self._owner()
        if (
            not isinstance(request_id, str)
            or not request_id.strip()
            or prefix_version is None
            or not callable(step)
            or inspect.iscoroutinefunction(step)
            or not callable(forward)
            or inspect.iscoroutinefunction(forward)
            or (
                prefix_is_current is not None
                and (
                    not callable(prefix_is_current)
                    or inspect.iscoroutinefunction(prefix_is_current)
                )
            )
        ):
            raise CooperativePredictionError(
                "request identity, prefix version and synchronous callbacks required"
            )
        if request_id in self._request_ids:
            raise CooperativePredictionError(
                "a request already has a queued prediction branch"
            )
        if len(self._jobs) >= self.max_pending_jobs:
            raise CooperativePredictionError("prediction queue is at capacity")
        self._jobs.append(
            _PredictionJob(
                request_id,
                prefix_version,
                step,
                forward,
                prefix_is_current or (lambda: True),
            )
        )
        self._request_ids.add(request_id)

    @property
    def pending(self) -> int:
        self._owner()
        return len(self._jobs)

    def run_one_after_decode(self, turn_id: int) -> PredictionTurn:
        """Advance no more than one job step after the turn's formal Decode.

        A busy target owner yields immediately instead of waiting. The caller
        can still enter its next ordinary scheduling turn. Request prefixes
        are revalidated before every step; a stale branch is discarded without
        forwarding and the next queued request waits until a later turn.
        """

        self._owner()
        if type(turn_id) is not int or turn_id < 0:
            raise CooperativePredictionError("turn_id must be non-negative")
        if self._last_turn_id is not None and turn_id <= self._last_turn_id:
            raise CooperativePredictionError(
                "prediction pump may run only once per increasing Scheduler turn"
            )
        self._last_turn_id = turn_id
        if not self._jobs:
            return PredictionTurn("idle")
        if self._stepping:
            raise CooperativePredictionError("prediction steps cannot reenter")

        job = self._jobs[0]
        if not job.prefix_is_current():
            self._jobs.popleft()
            self._request_ids.remove(job.request_id)
            return PredictionTurn("stale", job.request_id, job.step_index)

        if self.arbiter.busy:
            return PredictionTurn("busy", job.request_id, job.step_index)
        try:
            lease = self.arbiter.acquire()
        except Exception as exc:
            # The scheduler thread is the arbiter owner. A same-thread busy
            # lease is ordinary backpressure; other arbiter failures are not.
            if self.arbiter.busy:
                return PredictionTurn("busy", job.request_id, job.step_index)
            raise CooperativePredictionError("target arbiter acquisition failed") from exc

        if not self.execution_lock.acquire(blocking=False):
            self.arbiter.release(lease)
            return PredictionTurn("busy", job.request_id, job.step_index)

        self._stepping = True
        forward_calls = 0

        def forward_once(batch):
            nonlocal forward_calls
            if forward_calls:
                raise CooperativePredictionError(
                    "one prediction step attempted more than one ModelRunner forward"
                )
            forward_calls += 1
            return job.forward(batch)

        try:
            result = job.step(forward_once)
            if inspect.isawaitable(result):
                close = getattr(result, "close", None)
                if callable(close):
                    close()
                raise CooperativePredictionError(
                    "prediction step must finish synchronously"
                )
            if not isinstance(result, PredictionStep):
                raise CooperativePredictionError(
                    "prediction step must return PredictionStep"
                )
            if result.done:
                self._jobs.popleft()
                self._request_ids.remove(job.request_id)
                status = "completed"
            else:
                self._jobs.rotate(-1)
                status = "progressed"
            job.step_index += 1
        except BaseException:
            # A failed ModelRunner call may have enqueued CUDA work without a
            # completion proof. Retain both ownership gates and refuse reuse.
            self._quarantine = job
            self._held_arbiter_lease = lease
            self._held_execution_lock = True
            raise
        finally:
            self._stepping = False

        try:
            self.execution_lock.release()
            self.arbiter.release(lease)
        except BaseException as exc:
            self._quarantine = job
            # The release state is uncertain; do not advertise this device as
            # available to another target forward.
            self._held_arbiter_lease = lease
            self._held_execution_lock = True
            raise CooperativePredictionError(
                "prediction step ownership could not be released"
            ) from exc

        return PredictionTurn(
            status,
            job.request_id,
            job.step_index - 1,
            result.value if result.done else None,
        )
