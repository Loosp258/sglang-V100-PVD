"""Transport-independent resource lifetime and capacity primitives for PVD."""

from __future__ import annotations

import enum
import threading
from typing import Any, Callable, Dict, Optional, Tuple


class TransportState(str, enum.Enum):
    NOT_SUBMITTED = "not_submitted"
    IN_FLIGHT = "in_flight"
    DRAINING = "draining"
    TERMINAL_SUCCESS = "terminal_success"
    TERMINAL_FAILED = "terminal_failed"
    UNKNOWN = "unknown"

    @property
    def is_locally_safe_to_release(self) -> bool:
        return self in {
            TransportState.NOT_SUBMITTED,
            TransportState.TERMINAL_SUCCESS,
            TransportState.TERMINAL_FAILED,
        }


class GuardUnpinOutcome(str, enum.Enum):
    """Atomic post-unpin ownership and release-callback state."""

    OWNERS_REMAIN = "owners_remain"
    RELEASE_NOT_REQUESTED = "release_not_requested"
    RELEASE_IN_PROGRESS = "release_in_progress"
    RELEASED = "released"


class ResourceGuard:
    """Keep a resource alive until release is requested and all owners unpin."""

    def __init__(self, value: Any, release: Callable[[], None]) -> None:
        self._value: Optional[Any] = value
        self._release = release
        self._owners: set[str] = set()
        self._release_requested = False
        self._releasing = False
        self._released = False
        self._lock = threading.Lock()

    @property
    def value(self) -> Optional[Any]:
        with self._lock:
            return self._value

    def pin(self, owner: str) -> None:
        with self._lock:
            if self._release_requested or self._releasing or self._released:
                raise RuntimeError("resource release has already begun")
            self._owners.add(owner)

    def unpin(self, owner: str) -> GuardUnpinOutcome:
        with self._lock:
            self._owners.discard(owner)
            if self._released:
                return GuardUnpinOutcome.RELEASED
            if self._releasing:
                return GuardUnpinOutcome.RELEASE_IN_PROGRESS
            if self._owners:
                return GuardUnpinOutcome.OWNERS_REMAIN
            if not self._release_requested:
                return GuardUnpinOutcome.RELEASE_NOT_REQUESTED
            self._releasing = True
            should_release = True
        if should_release:
            self._run_release()
        return GuardUnpinOutcome.RELEASED

    def request_release(self) -> None:
        with self._lock:
            self._release_requested = True
            should_release = self._begin_release_locked()
        if should_release:
            self._run_release()

    def _begin_release_locked(self) -> bool:
        if (
            self._release_requested
            and not self._owners
            and not self._released
            and not self._releasing
        ):
            self._releasing = True
            return True
        return False

    def _run_release(self) -> None:
        try:
            self._release()
        except Exception:
            with self._lock:
                self._releasing = False
            raise
        with self._lock:
            self._released = True
            self._releasing = False
            self._value = None
            self._release = None


class TransferCapacityError(RuntimeError):
    """Raised when a reservation would exceed transfer capacity."""


class TransferBudget:
    """Thread-safe owner-keyed accounting for staging memory and transfers."""

    def __init__(self, staging_bytes: int, max_inflight: int) -> None:
        if staging_bytes <= 0 or max_inflight <= 0:
            raise ValueError("transfer budget limits must be positive")
        self._staging_bytes = staging_bytes
        self._max_inflight = max_inflight
        self._reservations: Dict[str, Tuple[int, int]] = {}
        self._used_staging_bytes = 0
        self._used_inflight = 0
        self._lock = threading.Lock()

    def reserve(self, owner: str, byte_count: int, slots: int) -> None:
        if byte_count < 0 or slots < 0:
            raise ValueError("reservation values must be non-negative")
        with self._lock:
            existing = self._reservations.get(owner)
            requested = (byte_count, slots)
            if existing is not None:
                if existing != requested:
                    raise ValueError("owner already has a different reservation")
                return
            if (
                self._used_staging_bytes + byte_count > self._staging_bytes
                or self._used_inflight + slots > self._max_inflight
            ):
                raise TransferCapacityError("transfer capacity exceeded")
            self._reservations[owner] = requested
            self._used_staging_bytes += byte_count
            self._used_inflight += slots

    def release(self, owner: str) -> None:
        with self._lock:
            reservation = self._reservations.pop(owner, None)
            if reservation is None:
                return
            byte_count, slots = reservation
            self._used_staging_bytes -= byte_count
            self._used_inflight -= slots

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "staging_bytes": self._staging_bytes,
                "max_inflight": self._max_inflight,
                "used_staging_bytes": self._used_staging_bytes,
                "used_inflight": self._used_inflight,
                "reservations": len(self._reservations),
            }


class TransferLifecycleManager:
    """Shared native-write ownership, independent of business cancellation.

    This manager charges transfer slots only. Staging allocations must reserve
    bytes separately, under their allocation owner, before allocating memory.
    A byte count here is the PUT size, not the size/lifetime of its allocation.
    """

    def __init__(self, budget: TransferBudget) -> None:
        self.budget = budget
        self._transfers = {}
        # Native submit and a transition to UNKNOWN share this gate.  A lost
        # native handle requires process-level quarantine, so no admitted
        # transfer may slip through to native submission after that point.
        self._lock = threading.RLock()
        self._quarantine_reason: Optional[str] = None

    def attach(self, handle, source_guard: ResourceGuard, byte_count: int) -> None:
        if byte_count <= 0:
            raise ValueError("transfer size must be positive")
        with handle._lock:
            with self._lock:
                if self._quarantine_reason is not None:
                    raise RuntimeError(
                        "PVD native transport is quarantined: "
                        f"{self._quarantine_reason}"
                    )
                if handle.transfer_id in self._transfers:
                    raise ValueError("transfer is already attached")
                self.budget.reserve(handle.transfer_id, 0, 1)
                try:
                    source_guard.pin(handle.transfer_id)
                except Exception:
                    self.budget.release(handle.transfer_id)
                    raise
                self._transfers[handle.transfer_id] = (handle, source_guard, byte_count)

    def submit_native(self, handle, submit: Callable[[], Any]) -> None:
        """Invoke native submission under the quarantine gate.

        ``attach`` reserves a slot before this method, but an earlier
        submission can become untrackable before this handle reaches native.
        Holding the gate across both calls makes that race fail locally with
        ``NOT_SUBMITTED`` rather than creating another unsafe native write.
        """
        from sglang.srt.disaggregation.pvd.transfer_engine import TransferStatus

        with handle._lock:
            with self._lock:
                record = self._transfers.get(handle.transfer_id)
                if record is None:
                    raise RuntimeError("transfer was not attached")
                if self._quarantine_reason is not None:
                    self._discard_unsubmitted_locked(handle, record)
                    handle.status = TransferStatus.FAILED
                    handle.error = (
                        "PVD native transport is quarantined: "
                        f"{self._quarantine_reason}"
                    )
                    return
                try:
                    native_id = submit()
                except Exception as exc:
                    self._mark_unknown_locked(handle, record, f"native submit raised: {exc}")
                    return
                if not isinstance(native_id, int) or native_id <= 0:
                    self._mark_unknown_locked(
                        handle, record, "native submit returned no trackable handle"
                    )
                    return
                handle.backend_handle = native_id
                handle.transport_state = TransportState.IN_FLIGHT

    def mark_unknown(self, handle, reason: str) -> None:
        with handle._lock:
            with self._lock:
                record = self._transfers.get(handle.transfer_id)
                self._mark_unknown_locked(handle, record, reason)

    def _mark_unknown_locked(self, handle, record, reason: str) -> None:
        from sglang.srt.disaggregation.pvd.transfer_engine import TransferStatus

        if handle.transport_state in (
            TransportState.TERMINAL_SUCCESS, TransportState.TERMINAL_FAILED
        ):
            return
        handle.transport_state = TransportState.UNKNOWN
        handle.error = reason
        if handle.status == TransferStatus.PENDING:
            handle.status = TransferStatus.FAILED
        if self._quarantine_reason is None:
            self._quarantine_reason = reason
        if record is not None:
            # This pins an untrackable source forever (until coordinated
            # restart) while also rejecting later source owners immediately.
            record[1].request_release()

    def _discard_unsubmitted_locked(self, handle, record) -> None:
        """Undo an admission that lost the race to process quarantine."""
        _, guard, _ = record
        guard.unpin(handle.transfer_id)
        self.budget.release(handle.transfer_id)
        self._transfers.pop(handle.transfer_id, None)

    def complete(self, handle, success: bool) -> None:
        from sglang.srt.disaggregation.pvd.transfer_engine import TransferStatus

        with handle._lock:
            if handle.transport_state == TransportState.UNKNOWN:
                return
            with self._lock:
                record = self._transfers.get(handle.transfer_id)
            if record is None:
                return
            _, guard, byte_count = record
            if handle.transport_state not in (
                TransportState.TERMINAL_SUCCESS, TransportState.TERMINAL_FAILED
            ):
                handle.transport_state = (
                    TransportState.TERMINAL_SUCCESS if success
                    else TransportState.TERMINAL_FAILED
                )
            terminal_success = handle.transport_state == TransportState.TERMINAL_SUCCESS
            if terminal_success:
                handle.transferred_bytes = byte_count
            if handle.status == TransferStatus.PENDING:
                handle.status = (
                    TransferStatus.SUCCESS if terminal_success else TransferStatus.FAILED
                )
            if not terminal_success:
                handle.error = "Mooncake native transfer failed"

        # The callback can block and can race a caller retrying release_memory.
        # Do not retain the handle or manager lock while it runs.  A returning
        # unpin is not enough: another callback may still be in progress.
        unpin_outcome = guard.unpin(handle.transfer_id)
        if unpin_outcome == GuardUnpinOutcome.RELEASE_IN_PROGRESS:
            return

        # Failed deregistration keeps the record and its capacity until a
        # later terminal poll sees the callback finish successfully.
        with self._lock:
            if self._transfers.get(handle.transfer_id) is not record:
                return
            self.budget.release(handle.transfer_id)
            self._transfers.pop(handle.transfer_id, None)

    def request_cancel(self, handle) -> None:
        from sglang.srt.disaggregation.pvd.transfer_engine import TransferStatus

        with handle._lock:
            if handle.status == TransferStatus.PENDING:
                handle.status = TransferStatus.CANCELLED

    def snapshot(self) -> dict:
        result = self.budget.snapshot()
        with self._lock:
            result["tracked_transfers"] = len(self._transfers)
            result["unknown_transfers"] = sum(
                handle.transport_state == TransportState.UNKNOWN
                for handle, _, _ in self._transfers.values()
            )
            result["quarantined"] = self._quarantine_reason is not None
        return result
