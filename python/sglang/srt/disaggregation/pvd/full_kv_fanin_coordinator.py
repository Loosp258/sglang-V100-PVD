"""Bounded, opt-in all-writer orchestration over existing V shard clients.

One record names one D destination. It pins the parent Entry count until all
writers ACK or fence. D-rank install agreement remains the receiver's obligation.
"""

import asyncio
import copy
import time
from dataclasses import dataclass, field
from typing import Mapping

from sglang.srt.disaggregation.pvd.coordinator import CoordinatorError
from sglang.srt.disaggregation.pvd.full_kv_fanin_plan import (
    canonical,
    validate_fanin_plan,
)
from sglang.srt.disaggregation.pvd.full_kv_fanin_proof import validate_fanin_proof
from sglang.srt.disaggregation.pvd.protocol import (
    PVD_TRANSFER_LIFECYCLE_PROTOCOL,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.request_state import EntryState


@dataclass
class _Group:
    plan: object
    manifest: dict
    identities: dict
    deadline: float
    state: str = "reserving"
    published: bool = False
    start_requested: bool = False
    cancelled: bool = False
    counted: bool = True
    error: str | None = None
    proofs: dict = field(default_factory=dict)
    shard_states: dict = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class FullKVFanInCoordinator:
    def __init__(self, parent, *, max_slices, max_records):
        if any(type(v) is not int or v <= 0 for v in (max_slices, max_records)):
            raise ValueError("positive fan-in coordinator bounds required")
        self.parent, self.max_slices, self.max_records = parent, max_slices, max_records
        self.records = {}
        self.closed_entries = set()

    @staticmethod
    def _token(delivery_id, destination_rank):
        if (
            not isinstance(delivery_id, str)
            or not delivery_id.strip()
            or type(destination_rank) is not int
            or destination_rank < 0
        ):
            raise CoordinatorError("exact fan-in delivery id and D rank required")
        return delivery_id, destination_rank

    async def _record(self, manifest, source_epochs, *, fencing=False):
        wire = copy.deepcopy(manifest)
        plan = validate_fanin_plan(wire, max_slices=self.max_slices)
        if not isinstance(source_epochs, Mapping) or set(source_epochs) != {
            str(r) for r in plan.writers
        }:
            raise CoordinatorError("complete V source epoch map required")
        if any(not isinstance(v, str) or not v.strip() for v in source_epochs.values()):
            raise CoordinatorError("non-empty V source epochs required")
        expected = {
            rank: WriteIdentity(
                protocol=PVD_TRANSFER_LIFECYCLE_PROTOCOL,
                sender_epoch=source_epochs[str(rank)],
                receiver_epoch=plan.destination.backend_metadata.get(
                    "pvd_receiver_epoch"
                ),
                transfer_id=f"{plan.delivery_id}:d{plan.destination.rank}:v{rank}",
                region_id=plan.destination.region_id,
                generation=plan.destination.backend_metadata.get("pvd_generation"),
                shard_rank=plan.destination.rank,
                key=plan.key,
            )
            for rank in plan.writers
        }
        token = self._token(plan.delivery_id, plan.destination.rank)
        async with self.parent._lock:
            previous = self.records.get(token)
            if previous is not None:
                if (
                    previous.plan.fingerprint != plan.fingerprint
                    or previous.identities != expected
                ):
                    raise CoordinatorError(
                        "fan-in delivery already has another plan/epoch"
                    )
                return previous
            if len(self.records) >= self.max_records:
                raise CoordinatorError("fan-in retained record capacity exhausted")
            entry = self.parent.entries.get(plan.key)
            if (
                entry is None
                or entry.manifest.layout.fingerprint != plan.storage.fingerprint
            ):
                raise CoordinatorError("fan-in Entry/layout is unknown or mismatched")
            if not fencing and (
                plan.key in self.closed_entries
                or entry.state not in (EntryState.P_WRITING, EntryState.STORED)
            ):
                raise CoordinatorError("fan-in Entry is not available")
            if not set(plan.writers).issubset(self.parent.shards):
                raise CoordinatorError("fan-in sources are outside this V group")
            record = _Group(
                plan,
                wire,
                expected,
                time.monotonic() + self.parent.delivery_timeout_secs,
            )
            self.records[token] = record
            entry.active_delivery_count += 1
            return record

    def _lookup(self, delivery_id, destination_rank):
        try:
            return self.records[self._token(delivery_id, destination_rank)]
        except KeyError as exc:
            raise CoordinatorError("unknown fan-in delivery") from exc

    def _snapshot(self, record):
        return {
            "delivery_id": record.plan.delivery_id,
            "destination_rank": record.plan.destination.rank,
            "plan_fingerprint": record.plan.fingerprint,
            "state": record.state,
            "write_identities": {
                str(r): i.to_dict() for r, i in record.identities.items()
            },
            "writer_proofs": {
                str(r): copy.deepcopy(p) for r, p in record.proofs.items()
            },
            "fenced": set(record.proofs) == set(record.identities),
            "entry_count_held": record.counted,
            "error": record.error,
        }

    def _proof(self, record, rank, proof):
        reported, terminal = validate_fanin_proof(
            proof,
            fingerprint=record.plan.fingerprint,
            identities=record.identities,
            byte_counts={
                r: sum(p.length for p in parts)
                for r, parts in record.plan.writers.items()
            },
        )
        if reported != rank:
            raise CoordinatorError("V reply substituted another source rank")
        if terminal is not None:
            old = record.proofs.get(rank)
            if old is not None and old != proof:
                raise CoordinatorError("V writer changed terminal proof")
            record.proofs[rank] = copy.deepcopy(proof)

    def _consume(self, record, rank, reply):
        identity = record.identities[rank]
        if (
            not isinstance(reply, Mapping)
            or reply.get("delivery_id") != identity.transfer_id
            or reply.get("entry_key") != record.plan.key.to_dict()
            or canonical(reply.get("destination"))
            != canonical(record.plan.destination.to_dict())
            or WriteIdentity.from_dict(reply.get("write_identity")) != identity
        ):
            raise CoordinatorError(
                "V reserved/submitted a different fan-in destination"
            )
        state = reply.get("state")
        if state not in {
            "waiting_source",
            "d_reserved",
            "v_writing",
            "delivered",
            "acked",
            "released",
        }:
            raise CoordinatorError(f"V fan-in delivery failed: {state}")
        self._proof(record, rank, reply.get("fanin_proof"))
        record.shard_states[rank] = state

    def _ready(self, record):
        return (
            not record.cancelled
            and set(record.proofs) == set(record.identities)
            and all(
                p["transport_state"] == "terminal_success"
                for p in record.proofs.values()
            )
            and all(
                record.shard_states.get(r) in {"delivered", "acked", "released"}
                for r in record.identities
            )
        )

    async def _uncount(self, record):
        async with self.parent._lock:
            if record.counted:
                entry = self.parent.entries[record.plan.key]
                if entry.active_delivery_count <= 0:
                    raise CoordinatorError("fan-in Entry ownership count was lost")
                entry.active_delivery_count -= 1
                record.counted = False

    async def _fence(self, record):
        if record.state in {"released", "cancelled"}:
            return self._snapshot(record)
        record.cancelled = True
        record.state = "cancelling"
        replies = await asyncio.gather(
            *(
                self.parent.shards[r].fence_fanin_delivery(
                    copy.deepcopy(record.manifest), identity
                )
                for r, identity in record.identities.items()
            ),
            return_exceptions=True,
        )
        confirmed = set()
        for rank, reply in zip(record.identities, replies):
            try:
                if isinstance(reply, BaseException):
                    raise CoordinatorError(str(reply))
                self._proof(record, rank, reply)
                if reply["fenced"]:
                    confirmed.add(rank)
            except Exception as exc:
                record.error = f"writer {rank} fence unconfirmed: {exc}"
        if confirmed == set(record.identities):
            record.state = "cancelled"
            await self._uncount(record)
        return self._snapshot(record)

    async def reserve(self, manifest, source_epochs):
        record = await self._record(manifest, source_epochs)
        async with record.lock:
            if record.cancelled:
                return await self._fence(record)
            if record.published:
                return self._snapshot(record)
            # Persist full expected identities BEFORE any shard sees the MR.
            record.published = True
            try:
                replies = await asyncio.gather(
                    *(
                        self.parent.shards[r].reserve_fanin_delivery(
                            copy.deepcopy(record.manifest),
                            expected_sender_epoch=i.sender_epoch,
                        )
                        for r, i in record.identities.items()
                    ),
                    return_exceptions=True,
                )
                for rank, reply in zip(record.identities, replies):
                    if isinstance(reply, BaseException):
                        raise CoordinatorError(str(reply))
                    self._consume(record, rank, reply)
            except BaseException as exc:
                record.cancelled = True
                record.state, record.error = "cancelling", f"reserve unconfirmed: {exc}"
                if not isinstance(exc, Exception):
                    raise
                return await self._fence(record)
            record.state = "delivered" if self._ready(record) else "reserved"
            return self._snapshot(record)

    async def _progress(self, record, *, start=False):
        record.start_requested = record.start_requested or start
        if record.cancelled:
            return await self._fence(record)
        if record.state in {"released", "delivered"}:
            return self._snapshot(record)
        if record.deadline <= time.monotonic():
            record.error = "fan-in delivery deadline expired"
            return await self._fence(record)
        async with self.parent._lock:
            state = self.parent.entries[record.plan.key].state
        if state != EntryState.STORED:
            if state == EntryState.P_WRITING:
                record.state = "waiting_source"
                return self._snapshot(record)
            record.error = "Entry stopped before fan-in completion"
            return await self._fence(record)
        submitting = record.start_requested and record.state != "writing"
        method = "start_delivery" if submitting else "poll_delivery"
        try:
            replies = await asyncio.gather(
                *(
                    getattr(self.parent.shards[r], method)(
                        record.plan.key, i.transfer_id
                    )
                    for r, i in record.identities.items()
                ),
                return_exceptions=True,
            )
            for rank, reply in zip(record.identities, replies):
                if isinstance(reply, BaseException):
                    raise CoordinatorError(str(reply))
                self._consume(record, rank, reply)
        except BaseException as exc:
            record.cancelled = True
            record.state, record.error = (
                "cancelling",
                f"writer progress unconfirmed: {exc}",
            )
            if not isinstance(exc, Exception):
                raise
            return await self._fence(record)
        record.state = (
            "delivered"
            if self._ready(record)
            else "writing"
            if submitting or record.state == "writing"
            else record.state
        )
        return self._snapshot(record)

    async def start(self, delivery_id, destination_rank):
        record = self._lookup(delivery_id, destination_rank)
        async with record.lock:
            return await self._progress(record, start=True)

    async def poll(self, delivery_id, destination_rank):
        record = self._lookup(delivery_id, destination_rank)
        async with record.lock:
            return await self._progress(record)

    async def ack(self, delivery_id, destination_rank):
        record = self._lookup(delivery_id, destination_rank)
        async with record.lock:
            if record.state == "released":
                return self._snapshot(record)
            if not self._ready(record) or record.state != "delivered":
                raise CoordinatorError("all V writers must deliver before fan-in ACK")
            replies = await asyncio.gather(
                *(
                    self.parent.shards[r].ack_delivery(record.plan.key, i.transfer_id)
                    for r, i in record.identities.items()
                ),
                return_exceptions=True,
            )
            for rank, reply in zip(record.identities, replies):
                if isinstance(reply, BaseException):
                    raise CoordinatorError(f"fan-in ACK unconfirmed: {reply}")
                self._consume(record, rank, reply)
                if reply["state"] != "released":
                    raise CoordinatorError("V fan-in ACK did not release delivery")
            record.state = "released"
            await self._uncount(record)
            return self._snapshot(record)

    async def fence(self, manifest, source_epochs):
        record = await self._record(manifest, source_epochs, fencing=True)
        async with record.lock:
            return await self._fence(record)

    async def cancel_entry(self, key):
        async with self.parent._lock:
            self.closed_entries.add(key)
        for record in tuple(self.records.values()):
            if record.plan.key == key and record.counted:
                async with record.lock:
                    await self._fence(record)

    async def reap(self, now):
        for record in tuple(self.records.values()):
            if not record.counted or record.lock.locked():
                continue
            async with record.lock:
                if record.deadline <= now or record.cancelled:
                    await self._fence(record)
                elif record.start_requested:
                    await self._progress(record)
