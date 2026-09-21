"""Bounded, owner-polled rank control. No tensor/MR ownership or serving hook.

Transport threads enqueue bytes or latch peer loss; only the owner advances the
exchange. send/stop_peer must enqueue nonblocking, request-scoped notifications.
A successful callback is NOT a peer ACK, cleanup proof, or CUDA/RDMA fence.
"""

import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

from sglang.srt.disaggregation.pvd.rank_install_wire import (
    MAX_FRAME_BYTES,
    RankInstallExchange,
    RankInstallMessage,
)
from sglang.srt.disaggregation.pvd.sparse_install import (
    InstallEpoch,
    InstallProtocolError,
)


@dataclass(frozen=True)
class RankForwardPermit:
    """Owner-local execution identity; not a wire message or native fence."""

    operation_id: str
    identity: tuple[str, str, str]
    committed_tokens: int
    installed_epoch: InstallEpoch


class RankInstallRuntime:
    def __init__(
        self,
        exchange,
        *,
        send,
        stop_peer,
        max_pending_events,
        max_pending_bytes,
        clock=time.monotonic,
    ):
        if (
            not isinstance(exchange, RankInstallExchange)
            or not all(callable(fn) for fn in (send, stop_peer, clock))
            or any(
                type(n) is not int or n <= 0
                for n in (max_pending_events, max_pending_bytes)
            )
        ):
            raise InstallProtocolError(
                "exchange, callbacks and positive inbox bounds required"
            )
        exchange.coordinator._owner()
        self.exchange = exchange
        self._send, self._stop, self._clock = send, stop_peer, clock
        self._max_events, self._max_bytes = max_pending_events, max_pending_bytes
        self._lock = threading.Lock()
        self._queue, self._bytes, self._fault = deque(), 0, None
        self._accepting, self._progressing = True, False
        self._notifying = False
        self._epoch = self._deadline = self._last_time = None
        self._phase, self._reason = "idle", None
        self._stop_pending, self._previous = set(), {}
        self._forward = None

    def _bound(self, rank, epoch):
        return type(rank) is int and self.exchange.peer_epochs.get(rank) == epoch

    def post(self, raw, *, peer_rank, peer_epoch):
        """Thread-safe, no parsing/callbacks/state-machine work on this thread."""
        with self._lock:
            if not self._accepting or not self._bound(peer_rank, peer_epoch):
                return False
            if type(raw) is not bytes or not 0 < len(raw) <= MAX_FRAME_BYTES:
                self._fault = "invalid rank-control frame"
            elif (
                len(self._queue) >= self._max_events
                or self._bytes + len(raw) > self._max_bytes
            ):
                self._fault = "rank-control inbox capacity exceeded"
            else:
                self._queue.append((peer_rank, raw))
                self._bytes += len(raw)
                return True
            self._accepting = False
            return False

    def peer_lost(self, *, peer_rank, peer_epoch):
        with self._lock:
            if not self._accepting or not self._bound(peer_rank, peer_epoch):
                return False
            self._fault = "bound rank control channel lost"
            self._accepting = False
            return True

    def _now(self):
        value = self._clock()
        if (
            type(value) not in (float, int)
            or not math.isfinite(value)
            or (self._last_time is not None and value < self._last_time)
        ):
            raise InstallProtocolError("finite monotonic control clock required")
        self._last_time = value
        return value

    def begin(self, decode_tokens, *, timeout_seconds):
        self.exchange.coordinator._owner()
        if self._progressing or self._phase != "idle" or self._forward is not None:
            raise InstallProtocolError("idle owner runtime required")
        with self._lock:
            if self._fault or self._queue:
                raise InstallProtocolError("drain pending control before a new round")
        if (
            type(timeout_seconds) not in (float, int)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise InstallProtocolError("finite positive round timeout required")
        deadline = self._now() + timeout_seconds
        if not math.isfinite(deadline):
            raise InstallProtocolError("round deadline overflow")
        epoch = self.exchange.begin(decode_tokens)
        self._epoch, self._deadline, self._phase = epoch, deadline, "preparing"
        return epoch

    def _fail(self, reason):
        if self._phase != "failed":
            self.exchange.coordinator.cancel(reason)
            self._phase, self._reason, self._deadline = "failed", reason, None
            self._stop_pending = set(self.exchange.peer_epochs)
            with self._lock:
                self._accepting = False
                self._queue.clear()  # control bytes only; never data/MR ownership
                self._bytes = 0

    def _notify_stops(self):
        if self._notifying:
            return
        self._notifying = True
        try:
            for rank in sorted(self._stop_pending):
                try:
                    self._stop(rank, self.exchange.coordinator.identity, self._reason)
                except Exception:  # noqa: BLE001, S112 -- pending rank remains visible for retry
                    continue
                self._stop_pending.remove(rank)
        finally:
            self._notifying = False

    def _guard(self):
        with self._lock:
            fault = self._fault
        if fault:
            self._fail(fault)
        if self._phase == "failed":
            return False
        try:
            now = self._now()
        except Exception:  # noqa: BLE001 -- external clock failure cannot reopen admission
            self._fail("rank-control clock failed")
            return False
        if self._deadline is not None and now >= self._deadline:
            self._fail("rank installation round timed out")
            return False
        return True

    def _send_commands(self, commands):
        for rank, raw in commands.items():
            if not self._guard():
                return
            try:
                self._send(rank, raw)
            except Exception:  # noqa: BLE001 -- may already be queued, fail entire request
                self._fail("rank-control send failed")
                return

    def progress(self, *, max_events=64):
        """Bounded work, no sleep or retries of installation/resume commands.

        One absolute deadline includes PREPARED through RESUMED. Duplicate
        traffic cannot extend it. Missing messages time out the whole request.
        Automatic retries/failover are deliberately outside this runtime.
        """
        self.exchange.coordinator._owner()
        if self._progressing or type(max_events) is not int or max_events <= 0:
            raise InstallProtocolError("nonreentrant positive progress bound required")
        self._progressing = True
        try:
            for _ in range(max_events):
                if not self._guard():
                    break
                with self._lock:
                    if not self._queue:
                        break
                    rank, raw = self._queue.popleft()
                    self._bytes -= len(raw)
                try:
                    message = RankInstallMessage.decode(raw)
                    if (
                        message.peer_epoch == self.exchange.peer_epochs[rank]
                        and message.receipt == self._previous.get(rank)
                        and message.kind in ("prepared", "parked", "applied", "resumed")
                    ):
                        continue  # exact last completed round only; bounded replay history
                    self.exchange.receive(raw, peer_rank=rank)
                    if self.exchange.coordinator.snapshot()["state"] == "failed":
                        self._fail("rank reported installation failure")
                except (ValueError, RuntimeError):
                    self._fail("invalid rank installation event")
                    break
            with self._lock:
                pending = bool(self._queue)
            if (
                self._guard()
                and not pending
                and self._epoch is not None
                and self._forward is None
            ):
                if self._phase == "preparing":
                    commands = self.exchange.install_commands(self._epoch)
                    if commands:
                        self._phase = "installing"
                        self._send_commands(commands)
                elif self._phase == "installing":
                    if self.exchange.coordinator.snapshot()["completed"] == self._epoch:
                        self._phase = "resuming"
                        self._send_commands(self.exchange.resume_commands(self._epoch))
                elif self._phase == "resuming" and self.exchange.resume_complete(
                    self._epoch
                ):
                    self._previous = dict(self.exchange._resume_receipts)
                    self._epoch = self._deadline = None
                    self._phase = "idle"
            if not self._guard():
                self._notify_stops()
        finally:
            self._progressing = False

    def can_decode(self, decode_tokens):
        self.exchange.coordinator._owner()
        if self._progressing or not self._guard():
            return False
        if self._forward is not None:
            return False
        with self._lock:
            if self._queue:
                return False  # process potential failure notifications before dispatch
        return self.exchange.can_decode(decode_tokens)

    def begin_forward(self, decode_tokens):
        """Acquire one request execution slot after normal global admission.

        An already launched refresh can progress while this slot is owned, but
        INSTALL cannot be sent. The caller must also hold actual bank readers
        and target-worker execution ownership; this metadata ticket replaces
        neither. It never samples tokens or changes the committed-token clock.
        """
        if not self.can_decode(decode_tokens):
            raise InstallProtocolError("request must wait or abort before forward")
        self._forward = RankForwardPermit(
            uuid.uuid4().hex,
            self.exchange.coordinator.identity,
            decode_tokens,
            self.exchange.coordinator.snapshot()["completed"],
        )
        return self._forward

    def finish_forward(self, permit, *, readers_drained, succeeded):
        """Retire exactly this execution and decide whether to accept its output.

        Call only AFTER all actual execution/readers drain. The booleans are
        caller assertions, never evidence manufactured by this adapter. False
        means discard the output and abort this request; no token is appended.
        Apply the decision synchronously on the owner thread. Cancellation or
        timeout by itself NEVER retires the in-flight ticket.
        """
        self.exchange.coordinator._owner()
        if permit is None or permit is not self._forward or self._progressing:
            raise InstallProtocolError("stale or foreign forward completion")
        if readers_drained is not True or type(succeeded) is not bool:
            raise InstallProtocolError(
                "explicit execution drain and success status required"
            )
        self.progress(max_events=self._max_events)  # ticket still blocks INSTALL
        with self._lock:
            unresolved = bool(self._queue) or self._fault is not None
        if not succeeded:
            self._fail("target forward failed")
        elif unresolved:
            self._fail("unresolved rank control at forward completion")
        elif (
            self.exchange.coordinator.snapshot()["completed"] != permit.installed_epoch
        ):
            self._fail("installed bank changed during target forward")
        accepted = self._guard()
        self._forward = None
        if not accepted:
            self._notify_stops()
        return accepted

    def cancel(self):
        self.exchange.coordinator._owner()
        self._fail("rank installation request cancelled")
        self._notify_stops()

    def snapshot(self):
        self.exchange.coordinator._owner()
        with self._lock:
            queued, size = len(self._queue), self._bytes
        return {
            "phase": self._phase,
            "epoch": self._epoch,
            "deadline": self._deadline,
            "reason": self._reason,
            "pending_events": queued,
            "pending_bytes": size,
            "stop_notifications_pending": tuple(sorted(self._stop_pending)),
            "resource_cleanup_proven": False,
            "forward_operation_id": self._forward.operation_id
            if self._forward
            else None,
        }
