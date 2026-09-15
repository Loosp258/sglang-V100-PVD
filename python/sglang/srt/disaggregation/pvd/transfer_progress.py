"""Bounded progress driving and worker-level admission state for PVD.

One driver per worker process. It advances native polling and the two control
protocols -- P upload sync and D receive fence -- a bounded number of steps per
tick, with backoff after a control failure, and it creates no per-request
background task. A request that fails a thousand times costs a thousand
failures' worth of records, not a thousand coroutines.

Nothing in this module releases a resource. It only drives the owners that can:
the upload manager and the Decode sessions. If a driver stops running, those
owners keep their pins, which is the safe direction.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Both roles must advertise this before a lifecycle authorization is issued.
PVD_TRANSFER_CAPABILITY = "pvd_transfer_lifecycle_v1"


class TransferProgress:
    """Bounded driver over one worker's outstanding PVD transfers."""

    def __init__(
        self,
        *,
        upload_manager: Any = None,
        decode_owner: Any = None,
        decode_refresher: Any = None,
        max_records_per_tick: int = 64,
        backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 30.0,
    ) -> None:
        if max_records_per_tick <= 0:
            raise ValueError("max_records_per_tick must be positive")
        if backoff_seconds <= 0 or max_backoff_seconds < backoff_seconds:
            raise ValueError("backoff bounds are invalid")
        self.upload_manager = upload_manager
        # The object that owns retained Decode closes (the KV manager) and the
        # refresher that knows how to drive them.
        self.decode_owner = decode_owner
        self.decode_refresher = decode_refresher
        self.max_records_per_tick = max_records_per_tick
        self.backoff_seconds = backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._next_control_at = 0.0
        self._ticks = 0
        self._control_ticks = 0
        self._skipped_for_backoff = 0

    # -- native polling -----------------------------------------------------

    def tick(self) -> Dict[str, Any]:
        """Advance native transport observation without any control traffic.

        Safe to call from the scheduler thread: it performs no awaits and never
        blocks on the network beyond one native status check per handle.
        """
        observed = 0
        unknown = 0
        if self.upload_manager is not None:
            records: List[Any] = []
            snapshot = self.upload_manager.snapshot()
            for entry in snapshot.get("records", [])[: self.max_records_per_tick]:
                record = self.upload_manager.get(entry["transfer_id"])
                if record is not None:
                    records.append(record)
            for record in records:
                state = self.upload_manager.observe(record)
                observed += 1
                if state is not None and getattr(state, "value", "") == "unknown":
                    unknown += 1
        with self._lock:
            self._ticks += 1
        return {"observed": observed, "unknown": unknown}

    # -- control plane ------------------------------------------------------

    def _control_allowed(self) -> bool:
        with self._lock:
            if time.monotonic() < self._next_control_at:
                self._skipped_for_backoff += 1
                return False
            return True

    def _record_control_result(self, failed: bool) -> None:
        with self._lock:
            if failed:
                self._consecutive_failures += 1
                delay = min(
                    self.backoff_seconds * (2 ** (self._consecutive_failures - 1)),
                    self.max_backoff_seconds,
                )
                self._next_control_at = time.monotonic() + delay
            else:
                self._consecutive_failures = 0
                self._next_control_at = 0.0

    async def tick_control(self) -> Dict[str, Any]:
        """Advance upload sync and Decode fencing by one bounded step each.

        Never raises: a control-plane failure leaves every record and pin in
        place and schedules the next attempt behind a backoff.
        """
        if not self._control_allowed():
            return {"skipped": True, "reason": "backoff"}
        uploads_outstanding = 0
        closes_outstanding = 0
        failed = False
        if self.upload_manager is not None:
            try:
                before = self.upload_manager.outstanding()
                await self.upload_manager.progress()
                uploads_outstanding = self.upload_manager.outstanding()
                # No record retired while work remained: treat as a failed step
                # so repeated unreachable-V ticks back off instead of spinning.
                failed = failed or (before > 0 and uploads_outstanding >= before)
            except Exception as exc:  # pragma: no cover - progress swallows
                logger.warning("PVD upload progress raised: %s", exc)
                failed = True
        if self.decode_refresher is not None:
            try:
                closes_outstanding = (
                    await self.decode_refresher.progress_pending_closes()
                )
                failed = failed or closes_outstanding > 0
            except Exception as exc:
                logger.warning("PVD decode close progress raised: %s", exc)
                failed = True
        self._record_control_result(failed)
        with self._lock:
            self._control_ticks += 1
        return {
            "skipped": False,
            "uploads_outstanding": uploads_outstanding,
            "decode_closes_outstanding": closes_outstanding,
            "backing_off": failed,
        }

    # -- reporting ----------------------------------------------------------

    def outstanding(self) -> int:
        total = 0
        if self.upload_manager is not None:
            total += self.upload_manager.outstanding()
        if self.decode_owner is not None:
            total += len(getattr(self.decode_owner, "pending_decode_closes", ()))
        return total

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            state = {
                "capability": PVD_TRANSFER_CAPABILITY,
                "ticks": self._ticks,
                "control_ticks": self._control_ticks,
                "consecutive_control_failures": self._consecutive_failures,
                "skipped_for_backoff": self._skipped_for_backoff,
                "backing_off": time.monotonic() < self._next_control_at,
            }
        state["uploads"] = (
            self.upload_manager.snapshot() if self.upload_manager is not None else None
        )
        state["decode_closes_outstanding"] = len(
            getattr(self.decode_owner, "pending_decode_closes", ())
        )
        return state


def require_capability(peer: Optional[Dict[str, Any]], role: str) -> None:
    """Refuse to participate with a peer that does not advertise lifecycle v1.

    A missing capability is not a reason to fall back to the legacy fence: the
    legacy path cannot produce transport-terminal proof, so it must not be used
    to satisfy a lifecycle authorization.
    """
    capabilities = (peer or {}).get("capabilities")
    if not isinstance(capabilities, (list, tuple)) or (
        PVD_TRANSFER_CAPABILITY not in capabilities
    ):
        raise RuntimeError(
            f"PVD {role} does not advertise {PVD_TRANSFER_CAPABILITY}; "
            "refusing to establish a transfer authorization"
        )
