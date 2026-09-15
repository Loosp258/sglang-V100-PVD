"""P-side ownership of in-flight uploads, independent of any request object.

A ``PVDKVSender`` lives as long as the scheduler keeps its request. Abort,
clear and queue removal can retire a sender while its RDMA WRITE is still
active on the wire, and after that the sender is never polled again. Upload
records, native handles, source staging ownership and the pending terminal
notification therefore belong here, to a per-worker manager, and not to the
sender.

This module deliberately provides no autonomous driver. ``progress()`` is a
single bounded step that callers invoke: the scheduler triggers it while a
request is live, and tests drive it explicitly after the sender is gone. The
bounded background driver is Task 7; until it exists, a worker that stops
calling ``progress()`` keeps its records and its pins rather than releasing
anything, which is the safe direction.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sglang.srt.disaggregation.pvd.protocol import KVEntryKey, WriteIdentity
from sglang.srt.disaggregation.pvd.transfer_engine import (
    TransferEngine,
    TransferHandle,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransportState

logger = logging.getLogger(__name__)


@dataclass
class UploadRecord:
    """One P -> V shard upload, owned by the manager until V acknowledges."""

    identity: WriteIdentity
    coordinator: Any
    engine: Optional[TransferEngine] = None
    handle: Optional[TransferHandle] = None
    # Set immediately before the native submit call, so a record whose handle
    # is missing afterwards is reported as UNKNOWN rather than NOT_SUBMITTED.
    submit_attempted: bool = False
    # Set when V asks for closure, or when P decides never to submit. Both
    # forbid a later submission under this identity.
    submission_forbidden: bool = False
    abandoned: bool = False
    reported_state: Optional[TransportState] = None
    terminal_acked: bool = False
    last_error: Optional[str] = None
    sync_attempts: int = 0
    rpc_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def transfer_id(self) -> str:
        return self.identity.transfer_id

    def snapshot(self) -> Dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "key": self.identity.key.to_dict(),
            "shard_rank": self.identity.shard_rank,
            "sender_epoch": self.identity.sender_epoch,
            "receiver_epoch": self.identity.receiver_epoch,
            "generation": self.identity.generation,
            "submit_attempted": self.submit_attempted,
            "submission_forbidden": self.submission_forbidden,
            "abandoned": self.abandoned,
            "transport_state": (
                self.handle.transport_state.value if self.handle is not None else None
            ),
            "reported_state": (
                self.reported_state.value if self.reported_state else None
            ),
            "terminal_acked": self.terminal_acked,
            "sync_attempts": self.sync_attempts,
            "last_error": self.last_error,
        }


class PVDUploadManager:
    """Per-worker registry of uploads that outlives individual senders."""

    def __init__(self) -> None:
        self._records: Dict[str, UploadRecord] = {}
        self._lock = threading.Lock()
        self._retired = 0
        self._tombstones: Dict[str, str] = {}

    # -- registry -----------------------------------------------------------

    def open(self, *, identity: WriteIdentity, coordinator: Any) -> UploadRecord:
        if not isinstance(identity, WriteIdentity):
            raise ValueError("identity must be a WriteIdentity")
        with self._lock:
            reason = self._tombstones.get(identity.transfer_id)
            if reason is not None:
                raise RuntimeError(f"upload {identity.transfer_id} is closed: {reason}")
            existing = self._records.get(identity.transfer_id)
            if existing is not None:
                if existing.identity != identity:
                    raise ValueError(
                        "upload transfer id already has a different identity"
                    )
                return existing
            record = UploadRecord(identity=identity, coordinator=coordinator)
            self._records[identity.transfer_id] = record
            return record

    def get(self, transfer_id: str) -> Optional[UploadRecord]:
        with self._lock:
            return self._records.get(transfer_id)

    def attach(
        self, record: UploadRecord, *, engine: TransferEngine, handle: TransferHandle
    ) -> None:
        """Take ownership of the native handle for an already-open record."""
        with self._lock:
            if self._records.get(record.transfer_id) is not record:
                raise RuntimeError("upload record is not owned by this manager")
            if record.handle is not None and record.handle is not handle:
                # Overwriting would drop the first handle, and with it the only
                # way to learn whether that write is still running.
                raise RuntimeError("upload record already owns a native handle")
            record.engine = engine
            record.handle = handle

    def claim_submission(self, record: UploadRecord) -> None:
        """Claim the single submission allowed for this identity.

        V's authorization is a one-shot gate, so P must not submit twice under
        one identity either: a second write would be untracked by the first
        record and invisible to the drain.
        """
        with self._lock:
            if self._records.get(record.transfer_id) is not record:
                raise RuntimeError("upload record is not owned by this manager")
            if record.submit_attempted:
                raise RuntimeError(
                    f"upload {record.transfer_id} has already been submitted"
                )
            if record.submission_forbidden or record.abandoned:
                raise RuntimeError(
                    f"upload {record.transfer_id} may no longer be submitted"
                )
            record.submit_attempted = True

    def abandon(self, transfer_id: str, reason: str) -> None:
        """Declare that P will make no further submission under this identity.

        This does not release anything on V. It only makes the record eligible
        to report a proven pre-native rejection, and blocks a late submit.
        """
        with self._lock:
            record = self._records.get(transfer_id)
        if record is None:
            return
        record.abandoned = True
        record.submission_forbidden = True
        if record.last_error is None:
            record.last_error = reason

    def abandon_key(self, key: KVEntryKey, reason: str) -> List[str]:
        with self._lock:
            transfer_ids = [
                transfer_id
                for transfer_id, record in self._records.items()
                if record.identity.key == key
            ]
        for transfer_id in transfer_ids:
            self.abandon(transfer_id, reason)
        return transfer_ids

    def outstanding(self) -> int:
        with self._lock:
            return len(self._records)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            records = list(self._records.values())
            retired = self._retired
            tombstones = len(self._tombstones)
        return {
            "outstanding": len(records),
            "retired": retired,
            "tombstones": tombstones,
            "records": [record.snapshot() for record in records],
        }

    # -- progress -----------------------------------------------------------

    def observe(self, record: UploadRecord) -> Optional[TransportState]:
        """Return this upload's transport state without releasing anything.

        ``None`` means "still preparing": P has neither submitted nor decided
        to stop, so there is nothing V may act on yet.
        """
        handle = record.handle
        if handle is None:
            if record.submit_attempted:
                # Entered native submission but produced no trackable handle.
                return TransportState.UNKNOWN
            if record.submission_forbidden or record.abandoned:
                return TransportState.NOT_SUBMITTED
            return None
        state = handle.transport_state
        if state.is_locally_safe_to_release or state is TransportState.UNKNOWN:
            return state
        if record.engine is None:
            return state
        try:
            record.engine.poll(handle)
        except Exception as exc:  # pragma: no cover - adapter maps most cases
            # A lost native status is not terminal evidence. Stop touching the
            # handle and keep every resource it owns.
            with handle._lock:
                handle.transport_state = TransportState.UNKNOWN
                handle.error = f"P native polling failed: {exc}"
            record.last_error = str(exc)
            logger.warning(
                "PVD upload %s retained after native poll failure: %s",
                record.transfer_id,
                exc,
            )
        return handle.transport_state

    async def progress_record(self, record: UploadRecord) -> Dict[str, Any]:
        """Advance one upload by one bounded step.

        Never raises for a control-plane failure: an unreachable V leaves the
        record and its pins exactly as they were, to be retried.
        """
        async with record.rpc_lock:
            if record.terminal_acked:
                return record.snapshot()
            state = self.observe(record)
            if state is None:
                return record.snapshot()
            closed = state.is_locally_safe_to_release
            record.sync_attempts += 1
            try:
                reply = await record.coordinator.sync_upload(
                    record.identity, state, closed
                )
            except Exception as exc:
                record.last_error = str(exc)
                logger.warning(
                    "PVD upload %s sync failed; retaining resources: %s",
                    record.transfer_id,
                    exc,
                )
                return record.snapshot()
            record.last_error = None
            record.reported_state = state
            if reply.get("close_requested") is True:
                record.submission_forbidden = True
            if closed and reply.get("terminal_ack") is True:
                record.terminal_acked = True
                self._retire(record, "terminal acknowledged by V")
            return record.snapshot()

    async def progress(self) -> Dict[str, Any]:
        """Drive every open record once. Bounded by the current record count."""
        with self._lock:
            records = list(self._records.values())
        for record in records:
            await self.progress_record(record)
        return self.snapshot()

    def _retire(self, record: UploadRecord, reason: str) -> None:
        with self._lock:
            if self._records.get(record.transfer_id) is record:
                del self._records[record.transfer_id]
                self._retired += 1
            # A retired identity must never be submitted again, even if a stale
            # caller still holds the lease that produced it.
            self._tombstones[record.transfer_id] = reason
