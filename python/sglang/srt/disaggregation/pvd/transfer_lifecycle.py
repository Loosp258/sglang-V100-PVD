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

    def unpin(self, owner: str) -> None:
        with self._lock:
            self._owners.discard(owner)
            should_release = self._begin_release_locked()
        if should_release:
            self._run_release()

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
