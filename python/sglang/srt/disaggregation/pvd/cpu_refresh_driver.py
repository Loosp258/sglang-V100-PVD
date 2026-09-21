"""Owner-polled CPU request refresh driver, not a production Scheduler loop.

The caller polls between forwards and yields to asyncio for HTTP progress.
Initial complete Prompt admission stays in CPUDecodeLifecycle. CPU rank counts
below are local group mirrors, NEVER distributed TP acknowledgements/fences.
"""

import math
from dataclasses import dataclass
from types import MappingProxyType

from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import (
    CPUDecodeLifecycle,
    LifecycleError,
    TargetExecutionArbiter,
)


@dataclass(frozen=True)
class RefreshLaunch:
    request_id: str
    task: object
    query_source: str
    query_position: int


@dataclass(frozen=True)
class RefreshProgress:
    launched: tuple[RefreshLaunch, ...]
    installed: tuple[str, ...]
    aborted: tuple[str, ...]


@dataclass(frozen=True)
class _Registration:
    life: CPUDecodeLifecycle
    clients: object
    pack_source: object
    timeout: float


class CPURefreshDriver:
    """One arbiter, independent clocks, at most one capture launch per poll.

    Query policy is explicit: all configured layers/Q heads at the last token
    position of the coming committed boundary. A missed window uses the last
    actual prefix token. Short drafts cannot silently substitute an earlier Q.
    Existing in-flight queries are never cancelled/replaced at their boundary.
    """

    def __init__(self, arbiter):
        if not isinstance(arbiter, TargetExecutionArbiter):
            raise LifecycleError("explicit shared target arbiter required")
        arbiter.owner()
        self.arbiter = arbiter
        self._records = {}

    def register(self, life, *, clients, pack_source=None, timeout_seconds):
        self.arbiter.owner()
        if (
            not isinstance(life, CPUDecodeLifecycle)
            or life.arbiter is not self.arbiter
            or life.state != "running"
            or life._permit is not None
            or life._refresh is not None
            or life.request_id in self._records
        ):
            raise LifecycleError("unique admitted idle request required")
        ranks = life.controller.group.describe_banks()
        if set(clients) != set(ranks) or any(type(rank) is not int for rank in clients):
            raise LifecycleError("search client membership must match local CPU ranks")
        if (
            (life.controller.delivery is None and not callable(pack_source))
            or (life.controller.delivery is not None and pack_source is not None)
        ) or (
            type(timeout_seconds) not in (float, int)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise LifecycleError("source scope and finite positive timeout required")
        lead = life.controller.group.coordinator.snapshot()["lead_tokens"]
        if life.controller.pipeline.draft_config.predict_tokens < lead:
            raise LifecycleError(
                "draft horizon must cover the whole configured lead window"
            )
        self._records[life.request_id] = _Registration(
            life, MappingProxyType(dict(clients)), pack_source, float(timeout_seconds)
        )

    def progress(self):
        """Poll all requests, install at exact boundaries, launch one due probe.

        Does not sleep, change batch membership, append output, run a forward,
        or infer native transfer completion. Whole-selected-batch wait-all is
        still enforced by CPUBatchDispatcher.begin().
        """
        self.arbiter.owner()
        installed, aborted, due = [], [], []
        for record in self._records.values():
            life = record.life
            life.poll()
            if life.state == "aborted":
                aborted.append(life.request_id)
                continue
            if life.state != "running":
                continue
            state = life.controller.group.coordinator.snapshot()
            n, boundary = life.committed_tokens, state["next_boundary"]
            pending_boundary = life.controller.pending_install_boundary
            if life._refresh is not None and pending_boundary is not None:
                # APPLIED can advance next_boundary while this round still
                # waits for RESUMED/Delivery finalization at its old count.
                boundary = pending_boundary
            if n > boundary:
                life.terminate("Decode crossed an uninstalled refresh boundary")
                aborted.append(life.request_id)
                continue
            if life._permit is not None:
                continue
            if life._refresh is not None:
                # Ready early: keep current bank. Late: wait for original query.
                if n == boundary and life.try_install(
                    {rank: n for rank in life.controller.group.describe_banks()}
                ):
                    installed.append(life.request_id)
                continue
            if n >= boundary - state["lead_tokens"]:
                due.append((record, boundary - n))
        launched = []
        if due and not self.arbiter.busy:
            # Requests already in-flight are excluded. Launch one capture, let
            # it release the target lease before another request is considered.
            record, distance = due[0]
            life = record.life
            prefix = life.snapshot()
            position = len(prefix.tokens) + distance - 1
            task = life.launch_refresh(
                query_positions=(position,),
                clients=record.clients,
                pack_source=record.pack_source,
                timeout_seconds=record.timeout,
            )
            launched.append(
                RefreshLaunch(
                    life.request_id,
                    task,
                    "committed" if distance == 0 else "predicted",
                    position,
                )
            )
        return RefreshProgress(tuple(launched), tuple(installed), tuple(aborted))

    async def remove(self, life):
        """Cancel/drain this request before forgetting its registration."""
        self.arbiter.owner()
        record = self._records.get(life.request_id)
        if record is None or record.life is not life:
            raise LifecycleError("cannot remove an unregistered request incarnation")
        await life.close()
        del self._records[life.request_id]

    async def close(self):
        self.arbiter.owner()
        if any(r.life._permit is not None for r in self._records.values()):
            raise LifecycleError(
                "all dispatched Decode work must drain before driver close"
            )
        for record in tuple(self._records.values()):
            await self.remove(record.life)
