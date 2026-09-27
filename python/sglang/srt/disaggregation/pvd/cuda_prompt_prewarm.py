"""Bounded D Req-arrival owner for prompt-only sidecar cache warmup.

The ticket only warms the private target/draft prefix caches. Its Q reply is
discarded, and the ticket never authorizes retrieval, installation, or Decode.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from array import array
from dataclasses import dataclass

from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError
from sglang.srt.disaggregation.pvd.prediction import CommittedPrefix
from sglang.srt.disaggregation.pvd.probe_lane_identity import (
    ProbeLaneCheckpointIdentity,
)
from sglang.srt.disaggregation.pvd.probe_lane_protocol import (
    MAX_LANE_PREFIX_TOKENS,
    ProbeLaneTicket,
)
from sglang.srt.disaggregation.pvd.probe_lane_unix import ProbeLaneUnixClient
from sglang.srt.disaggregation.pvd.probe_search import ProbeWindow

logger = logging.getLogger(__name__)


def _safe_monotonic():
    try:
        return time.monotonic()
    except Exception:  # noqa: BLE001 - diagnostic timing is best effort.
        return None


def _elapsed_ms(started, finished):
    if started is None or finished is None:
        return None
    try:
        return max(0.0, finished - started) * 1000
    except Exception:  # noqa: BLE001 - diagnostic timing is best effort.
        return None


def _format_ms(value):
    return "-" if value is None else f"{value:.3f}"


def _log_prewarm_event(
    stage,
    owner,
    *,
    level=logging.INFO,
    schedule_delay_ms=None,
    request_duration_ms=None,
    total_duration_ms=None,
    error_type=None,
):
    """Emit request-identity-only timing; diagnostics must stay fail-open."""
    try:
        logger.log(
            level,
            "PVD timeline event=prompt_only_sidecar_prewarm stage=%s "
            "request_id=%s entry_transfer_id=%s utc_epoch_s=%.3f "
            "schedule_delay_ms=%s request_duration_ms=%s "
            "total_duration_ms=%s error_type=%s",
            stage,
            owner.request_id,
            owner.entry_transfer_id,
            time.time_ns() / 1_000_000_000,
            _format_ms(schedule_delay_ms),
            _format_ms(request_duration_ms),
            _format_ms(total_duration_ms),
            error_type or "-",
        )
    except Exception:  # noqa: BLE001 - diagnostics must be fail-open.
        return


@dataclass
class CUDAPromptPrewarmOwner:
    """One exact provisional Req/session pair held through initial admission."""

    req: object
    session: object
    request_id: str
    delivery_id: str
    entry_transfer_id: str
    receiver_epoch: str
    prompt: tuple[int, ...]
    ticket: ProbeLaneTicket
    scheduled_at_monotonic: float | None = None
    task: asyncio.Task | None = None
    reconciled: bool = False
    state: str = "pending"


class CUDAPromptPrewarmer:
    """Serialize early tickets around the sidecar's single active cache identity."""

    def __init__(
        self,
        driver,
        *,
        lane_client,
        checkpoint,
        target_model_id,
        probe_config,
        head_dim,
        timeout_seconds,
        max_prefix_tokens,
    ):
        if (
            not isinstance(lane_client, ProbeLaneUnixClient)
            or not isinstance(checkpoint, ProbeLaneCheckpointIdentity)
            or not isinstance(target_model_id, str)
            or not target_model_id.strip()
            or getattr(probe_config, "target_model_id", None) != target_model_id
            or type(head_dim) is not int
            or head_dim <= 0
            or type(timeout_seconds) not in (int, float)
            or timeout_seconds <= 0
            or type(max_prefix_tokens) is not int
            or not 1 <= max_prefix_tokens < MAX_LANE_PREFIX_TOKENS
            or not isinstance(getattr(probe_config, "layers", None), tuple)
            or not probe_config.layers
            or type(getattr(probe_config, "head_start", None)) is not int
            or type(getattr(probe_config, "head_count", None)) is not int
            or probe_config.head_count <= 0
        ):
            raise LifecycleError(
                "bounded sidecar prompt prewarm configuration required"
            )
        driver._owner()
        self.driver = driver
        self.lane_client = lane_client
        self.checkpoint = checkpoint
        self.target_model_id = target_model_id
        self.probe_config = probe_config
        self.head_dim = head_dim
        self.timeout_seconds = float(timeout_seconds)
        self.max_prefix_tokens = min(max_prefix_tokens, driver.max_prefix_tokens)
        self._owner: CUDAPromptPrewarmOwner | None = None
        self._orphan_tasks: set[asyncio.Task] = set()
        self._closed = False
        self._stats = {
            "started": 0,
            "skipped_busy": 0,
            "skipped_bounds": 0,
            "skipped_invalid": 0,
            "failed": 0,
            "cancelled": 0,
            "completed": 0,
            "retired": 0,
        }

    def _collect_orphans(self):
        self._orphan_tasks = {task for task in self._orphan_tasks if not task.done()}

    @property
    def pending(self):
        self.driver._owner()
        self._collect_orphans()
        return self._owner is not None or bool(self._orphan_tasks)

    @property
    def owner(self):
        self.driver._owner()
        return self._owner

    def snapshot(self):
        self.driver._owner()
        self._collect_orphans()
        return {
            "pending": self._owner is not None or bool(self._orphan_tasks),
            "owner_state": None if self._owner is None else self._owner.state,
            **self._stats,
        }

    def _ticket(self, req, session):
        from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeSession

        manager = getattr(session, "manager", None)
        delivery_id = getattr(req, "pvd_delivery_id", None)
        entry_transfer_id = getattr(getattr(session, "key", None), "transfer_id", None)
        prompt_values = getattr(req, "origin_input_ids", None)
        if (
            not isinstance(session, PVDDecodeSession)
            or session.req is not req
            or manager is None
            or manager.key_for(req) != session.key
            or manager.worker_epoch != session.receiver_epoch
            or getattr(req, "pvd_transfer_id", None) != entry_transfer_id
            or not isinstance(delivery_id, str)
            or not delivery_id
            or session.consumer_id != delivery_id
            or session.clock.round != 0
            or session.clock.pending is not None
            or session._closed
            or req.finished()
            or req.is_retracted
            or not isinstance(prompt_values, (list, tuple, array))
        ):
            raise LifecycleError("exact live D Req and initial PVD session required")
        prompt = tuple(prompt_values)
        # Leave room for P's first token so the later append ticket cannot
        # cross the sidecar's hard 16,384-token protocol cap.
        if (
            not prompt
            or len(prompt) > self.max_prefix_tokens
            or len(prompt) + 1 > MAX_LANE_PREFIX_TOKENS
            or any(
                type(token) is not int or not 0 <= token <= 0xFFFFFFFF
                for token in prompt
            )
        ):
            raise LifecycleError("prompt is outside the bounded sidecar lane")
        epoch = session.receiver_epoch
        version = f"{epoch}:prompt-only:{delivery_id}"
        prefix = CommittedPrefix(req.rid, prompt, 0, version)
        window = ProbeWindow(
            incarnation=epoch,
            operation_id=uuid.uuid4().hex,
            entry_transfer_id=entry_transfer_id,
            prefix=prefix,
            target_tokens=0,
            query_positions=(len(prompt),),
            query_source="predicted",
        )
        config = self.probe_config
        layers = tuple(sorted(set(config.layers)))
        head_start, head_count = config.head_start, config.head_count
        reply_bytes = len(layers) * head_count * self.head_dim * 4
        ticket = ProbeLaneTicket.issue(
            window,
            target_model_id=self.target_model_id,
            weights_sha256=self.checkpoint.weights_sha256,
            tokenizer_sha256=self.checkpoint.tokenizer_sha256,
            layers=layers,
            head_start=head_start,
            head_count=head_count,
            head_dim=self.head_dim,
            max_reply_bytes=reply_bytes,
            deadline_monotonic=time.monotonic() + self.timeout_seconds,
        )
        return CUDAPromptPrewarmOwner(
            req=req,
            session=session,
            request_id=req.rid,
            delivery_id=delivery_id,
            entry_transfer_id=entry_transfer_id,
            receiver_epoch=epoch,
            prompt=prompt,
            ticket=ticket,
        )

    async def _run(self, owner):
        run_started = _safe_monotonic()
        scheduled_at = owner.scheduled_at_monotonic
        schedule_delay_ms = _elapsed_ms(scheduled_at, run_started)
        _log_prewarm_event(
            "request_started", owner, schedule_delay_ms=schedule_delay_ms
        )
        try:
            async with self.lane_client.request(owner.ticket):
                # The bounded Q response has no serving consumer by design.
                pass
        except asyncio.CancelledError:
            owner.state = "cancelled"
            finished = _safe_monotonic()
            _log_prewarm_event(
                "cancelled",
                owner,
                schedule_delay_ms=schedule_delay_ms,
                request_duration_ms=_elapsed_ms(run_started, finished),
                total_duration_ms=_elapsed_ms(
                    scheduled_at if scheduled_at is not None else run_started,
                    finished,
                ),
            )
            raise
        except Exception as exc:  # noqa: BLE001 - optional sidecar work is fail-open.
            # This is optional sidecar cache work. Admission's ordinary n=0
            # ticket remains the fallback when the exact receipt arrives.
            owner.state = "failed"
            self._stats["failed"] += 1
            finished = _safe_monotonic()
            _log_prewarm_event(
                "failed",
                owner,
                level=logging.WARNING,
                schedule_delay_ms=schedule_delay_ms,
                request_duration_ms=_elapsed_ms(run_started, finished),
                total_duration_ms=_elapsed_ms(
                    scheduled_at if scheduled_at is not None else run_started,
                    finished,
                ),
                error_type=type(exc).__name__,
            )
            return False
        owner.state = "completed"
        self._stats["completed"] += 1
        finished = _safe_monotonic()
        _log_prewarm_event(
            "completed",
            owner,
            schedule_delay_ms=schedule_delay_ms,
            request_duration_ms=_elapsed_ms(run_started, finished),
            total_duration_ms=_elapsed_ms(
                scheduled_at if scheduled_at is not None else run_started,
                finished,
            ),
        )
        return True

    def start(self, req, session):
        """Schedule at most one early request; every malformed case fails open."""
        self.driver._owner()
        self._collect_orphans()
        if self._closed or self._owner is not None or self._orphan_tasks:
            self._stats["skipped_busy"] += 1
            return None
        try:
            owner = self._ticket(req, session)
        except Exception as exc:  # noqa: BLE001 - eligibility is optional.
            if "outside the bounded sidecar lane" in str(exc):
                self._stats["skipped_bounds"] += 1
            else:
                self._stats["skipped_invalid"] += 1
            return None
        try:
            owner.scheduled_at_monotonic = _safe_monotonic()
            coroutine = self._run(owner)
            try:
                task = self.driver._loop.create_task(coroutine)
            except BaseException:
                coroutine.close()
                raise
            owner.task = task
            self._owner = owner
            self._stats["started"] += 1
            _log_prewarm_event("scheduled", owner)
            return owner
        except Exception as exc:  # noqa: BLE001 - optional task setup is fail-open.
            self._stats["failed"] += 1
            _log_prewarm_event(
                "schedule_failed",
                owner,
                level=logging.WARNING,
                error_type=type(exc).__name__,
            )
            return None

    def owns(self, owner, req, session):
        self.driver._owner()
        return (
            isinstance(owner, CUDAPromptPrewarmOwner)
            and self._owner is owner
            and owner.req is req
            and owner.session is session
            and owner.request_id == req.rid
            and owner.entry_transfer_id == session.key.transfer_id
            and owner.receiver_epoch == session.receiver_epoch
            and owner.reconciled
        )

    def reconcile(self, session):
        """Bind the provisional ticket to the exact completed receipt, if any."""
        self.driver._owner()
        owner = self._owner
        if owner is None or owner.session is not session:
            return None
        req = session.req
        try:
            receipt = session.require_initial_prompt()
        except Exception:  # noqa: BLE001 - preserve the normal admission path.
            # A provisional prewarm must never turn a normal admission failure
            # into a new failure mode. Drop it and let the ordinary n=0 path run.
            self.cancel(req, session=session)
            return None
        if (
            owner.req is not req
            or owner.request_id != receipt.request_id
            or owner.delivery_id != req.pvd_delivery_id
            or owner.entry_transfer_id != receipt.key.transfer_id
            or owner.receiver_epoch != receipt.receiver_epoch
            or owner.prompt != receipt.prompt
            or receipt.outputs != tuple(req.output_ids)
        ):
            self.cancel(req, session=session)
            return None
        owner.reconciled = True
        return owner

    def complete(self, owner):
        """Retire after initial refresh installation and early work have drained."""
        self.driver._owner()
        if self._owner is owner:
            if not owner.reconciled or owner.task is None or not owner.task.done():
                raise LifecycleError("reconciled completed prewarm owner required")
            owner.reconciled = False
            owner.state = "retired"
            self._stats["retired"] += 1
            self._owner = None

    def cancel(self, req, *, session=None):
        """Drop exact owner provenance; an in-flight remote Q is discarded."""
        self.driver._owner()
        owner = self._owner
        if (
            owner is None
            or owner.req is not req
            or (session is not None and owner.session is not session)
        ):
            return False
        self._owner = None
        owner.reconciled = False
        if owner.state != "cancelled":
            owner.state = "cancelled"
            self._stats["cancelled"] += 1
        if owner.task is not None and not owner.task.done():
            owner.task.cancel()
            self._orphan_tasks.add(owner.task)
        return True

    def close_drained(self):
        self.driver._owner()
        self._collect_orphans()
        if self._owner is not None or self._orphan_tasks:
            raise LifecycleError(
                "early sidecar prompt owner must drain before shutdown"
            )
        self._closed = True

    def begin_shutdown(self):
        """Stop early admission and cancel its owner without blocking the loop."""
        self.driver._owner()
        self._closed = True
        owner = self._owner
        if owner is not None:
            self.cancel(owner.req, session=owner.session)
