"""Owner-thread CPU scheduler seam; NOT wired to production Scheduler.

No Req objects, sampler, native callbacks, transport fences or GPU admission.
The driver must determine EOS/length limits and provide real rank install ACKs.
One arbiter must be shared by all lifecycles using the same target execution
context. The offline probe's process-global context forbids parallel targets.
"""

import asyncio
import math
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

from sglang.srt.disaggregation.pvd.cpu_prefetch_request import CPUPrefetchRequest
from sglang.srt.disaggregation.pvd.prediction import CommittedPrefix


class LifecycleError(ValueError):
    pass


class TargetExecutionArbiter:
    """Owner-thread lease; cancellation cannot release an executing forward."""

    def __init__(self):
        self._owner_id = threading.get_ident()
        self._lease = None

    def owner(self):
        if threading.get_ident() != self._owner_id:
            raise LifecycleError("target execution belongs to its owner thread")

    @property
    def busy(self):
        self.owner()
        return self._lease is not None

    def acquire(self):
        self.owner()
        if self.busy:
            raise LifecycleError("target execution is busy")
        self._lease = object()
        return self._lease

    def release(self, lease):
        self.owner()
        if lease is not self._lease or lease is None:
            raise LifecycleError("stale target execution lease")
        self._lease = None


@dataclass(frozen=True)
class DecodePermit:
    incarnation: str
    committed_tokens: int
    input_token: int
    query_position: int
    operation_id: str


class CPUDecodeLifecycle:
    """Waiting -> admitted -> running/refresh wait -> finished/aborted.

    Counts advance only on matching successful one-token target completion.
    Initial full KV must already be installed before admit(). Refresh timeout
    is explicitly configured per launch, checked by poll(), and includes the
    wait for installation. Cancellation retains execution/resource ownership
    until completion/drain; it is not a remote-write cancellation proof.
    """

    def __init__(
        self, request_id, prompt, first_token, *, arbiter, clock=time.monotonic
    ):
        if not isinstance(arbiter, TargetExecutionArbiter):
            raise LifecycleError("explicit shared target arbiter required")
        arbiter.owner()
        if (
            not isinstance(prompt, tuple)
            or not prompt
            or any(type(t) is not int or t < 0 for t in prompt)
        ):
            raise LifecycleError("immutable nonempty Prompt token tuple required")
        self._token(first_token)
        if not isinstance(request_id, str) or not request_id.strip():
            raise LifecycleError("request identity required")
        self.request_id, self.prompt = request_id, prompt
        self.arbiter, self._clock = arbiter, clock
        self._outputs = [first_token]  # P's token, never a D clock tick
        self.incarnation = uuid.uuid4().hex
        self.state, self.reason = "waiting", None
        self.controller = None
        self._permit = self._decode_lease = None
        self._batch_owner = None
        self._refresh = self._refresh_lease = self._deadline = None
        self._refresh_ready = False

    @staticmethod
    def _token(token):
        if type(token) is not int or token < 0:
            raise LifecycleError("a nonnegative integer token is required")

    @property
    def committed_tokens(self):
        self.arbiter.owner()
        return len(self._outputs) - 1

    @property
    def outputs(self):
        self.arbiter.owner()
        return tuple(self._outputs)

    def snapshot(self):
        self.arbiter.owner()
        if self.state != "running" or self._permit is not None:
            raise LifecycleError(
                "snapshot requires an admitted request between forwards"
            )
        return CommittedPrefix(
            self.request_id,
            self.prompt + self.outputs,
            self.committed_tokens,
            f"{self.incarnation}:{self.committed_tokens}",
        )

    def admit(self, controller):
        self.arbiter.owner()
        if self.state != "waiting" or not isinstance(controller, CPUPrefetchRequest):
            raise LifecycleError(
                "admission requires a waiting request and CPU controller"
            )
        metadata = controller.group.describe_banks()
        state = controller.group.coordinator.snapshot()
        if (
            controller.group.coordinator.identity[0] != self.request_id
            or state["state"] != "idle"
            or state["installed_tokens"] != 0
            or not controller.group.can_decode(0)
            or any(m["prompt_tokens"] != len(self.prompt) for m in metadata.values())
        ):
            raise LifecycleError(
                "initial complete Prompt installation does not match admission"
            )
        if getattr(controller, "_lifecycle_claimed", False):
            raise LifecycleError("CPU controller already belongs to a lifecycle")
        controller._lifecycle_claimed = True
        self.controller = controller
        self.state = "running"

    def _release_refresh_lease(self):
        if self._refresh_lease is not None:
            self.arbiter.release(self._refresh_lease)
            self._refresh_lease = None

    @contextmanager
    def _capture_scope(self):
        try:
            if self.state != "running" or self._refresh_lease is None:
                raise LifecycleError("probe dispatch is stale")
            yield
        finally:
            self._release_refresh_lease()

    def launch_refresh(
        self, *, query_positions, clients, pack_source=None, timeout_seconds
    ):
        self.poll()
        if (
            self.state != "running"
            or self._permit is not None
            or self._refresh is not None
        ):
            raise LifecycleError(
                "refresh requires an idle admitted request without another refresh"
            )
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise LifecycleError("explicit finite positive refresh timeout required")
        state = self.controller.group.coordinator.snapshot()
        if self.committed_tokens < state["next_boundary"] - state["lead_tokens"]:
            raise LifecycleError("prefetch window has not started")
        loop = asyncio.get_running_loop()
        prefix = self.snapshot()
        # Reserve BEFORE task dispatch. D cannot advance the snapshot's count
        # while the queued task has not entered its synchronous probe yet.
        self._refresh_lease = self.arbiter.acquire()
        self._deadline = self._clock() + timeout_seconds
        coroutine = self.controller.refresh(
            prefix,
            query_positions=query_positions,
            clients=clients,
            pack_source=pack_source,
            execution_scope=self._capture_scope,
        )
        try:
            self._refresh = loop.create_task(coroutine)
        except BaseException:
            coroutine.close()
            self._release_refresh_lease()
            self._deadline = None
            raise
        return self._refresh

    def poll(self):
        self.arbiter.owner()
        if self._refresh is not None and self._refresh.done():
            self._release_refresh_lease()  # includes cancel-before-first-dispatch
            try:
                self._refresh.result()
            except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
                # Task errors become terminal lifecycle status, not silent success.
                if self.state == "running":
                    self.terminate("refresh failed: " + type(exc).__name__)
            else:
                self._refresh_ready = self.state == "running"
        if (
            self.state == "running"
            and self._deadline is not None
            and self._clock() >= self._deadline
        ):
            self.terminate("refresh timeout")
        return self.state

    def can_decode(self):
        self.poll()
        return (
            self.state == "running"
            and self._permit is None
            and not self.arbiter.busy
            and self.controller.can_decode(self.committed_tokens)
        )

    def begin_decode(self):
        if not self.can_decode():
            raise LifecycleError("request must wait or abort before Decode")
        self._decode_lease = self.arbiter.acquire()
        self._permit = DecodePermit(
            self.incarnation,
            self.committed_tokens,
            self._outputs[-1],
            len(self.prompt) + self.committed_tokens,
            uuid.uuid4().hex,
        )
        return self._permit

    def _match(self, permit):
        self.arbiter.owner()
        if self._batch_owner is not None:
            raise LifecycleError(
                "batch-owned completion must go through its dispatcher"
            )
        if permit is None or permit is not self._permit:
            raise LifecycleError("stale or foreign Decode completion")

    def _retire_decode(self):
        self.arbiter.release(self._decode_lease)
        self._decode_lease = self._permit = None

    def complete_decode(self, permit, token, *, finished=False):
        self._match(permit)
        self._token(token)
        if type(finished) is not bool:
            raise LifecycleError("finished must be an explicit bool")
        self.poll()
        if self.state != "running":
            self._retire_decode()  # result after cancel/EOS/timeout is discarded
            return False
        self._outputs.append(token)
        self._retire_decode()
        if finished:
            self.terminate("EOS or output limit", finished=True)
        return True

    def fail_decode(self, permit, reason):
        self._match(permit)
        self.terminate(reason)
        self._retire_decode()  # called only after the actual execution has ended

    def try_install(self, rank_counts):
        self.poll()
        if (
            self.state != "running"
            or self._permit is not None
            or self.arbiter.busy
            or not self._refresh_ready
        ):
            return False
        if any(
            type(n) is not int or n != self.committed_tokens
            for n in rank_counts.values()
        ):
            raise LifecycleError("rank counts must match actually committed D tokens")
        if self.committed_tokens != self.controller.pending_install_boundary:
            return False
        try:
            installed = self.controller.try_install(rank_counts)
        except BaseException:
            self.terminate("installation failed")
            raise
        if installed:
            self._refresh = self._deadline = None
            self._refresh_ready = False
        return installed

    def terminate(self, reason, *, finished=False):
        self.arbiter.owner()
        if self.state in ("finished", "aborted"):
            return
        if (
            type(finished) is not bool
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise LifecycleError(
                "explicit termination reason and boolean status required"
            )
        self.state, self.reason = ("finished" if finished else "aborted"), reason
        if self.controller is not None:
            self.controller.cancel(reason)
        if self._refresh is not None:
            self._refresh.cancel()
        # Deliberately retain target lease for any in-flight forward/probe.

    async def close(self):
        self.arbiter.owner()
        self.terminate("closed")
        if self._permit is not None:
            raise LifecycleError("Decode completion must drain before resource close")
        if self._refresh is not None:
            await asyncio.gather(self._refresh, return_exceptions=True)
            self.poll()
        if self.controller is not None:
            await self.controller.aclose()
