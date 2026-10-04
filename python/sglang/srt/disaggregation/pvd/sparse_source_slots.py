"""Bounded physical sending MRs with per-delivery owned sparse staging leases.

Only the store's terminal progress can return a lease after packing/native
completion. Timeout, cancellation and a returned submit call are not proofs.
The native adapter receives the exact physical RegisteredMemory, never an alias.
"""

from dataclasses import dataclass
from contextlib import nullcontext
import threading
import uuid

import torch
from sglang.srt.disaggregation.pvd.transfer_engine import MemorySlice, RegisteredMemory
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget, TransferCapacityError


class SparseSourceSlotUnknown(RuntimeError):
    pass


@dataclass
class _Slot:
    owner: str
    buffer: object = None
    registration: object = None
    registration_unknown: bool = False
    lease: object = None
    unknown: object = None
    retired: bool = False


class SparseSourceSlotLease:
    def __init__(self, pool, slot, owner, nbytes):
        self._pool, self._slot, self.owner = pool, slot, owner
        self._active = True
        self.buffer = slot.buffer[:nbytes]
        self.local = MemorySlice(slot.registration, 0, nbytes)

    def release_after_proof(self):
        """Internal store callback after local pack and adapter cleanup proof."""
        self._pool._return(self)

    def quarantine(self, reason):
        self._pool._quarantine(self, reason)


class SparseSourceSlotPool:
    def __init__(self, engine, budget, *, device, rank, rail, slots=2,
                 capacity_bytes=32768, endpoint='pvd-vector-sparse'):
        if (not isinstance(budget, TransferBudget) or type(slots) is not int
                or not 1 <= slots <= 64 or type(capacity_bytes) is not int
                or not 1 <= capacity_bytes <= 1 << 30 or type(rank) is not int or rank < 0
                or not isinstance(rail, str) or not rail or not endpoint):
            raise ValueError('bounded source pool configuration required')
        self.engine, self.budget = engine, budget
        self.device, self.rank, self.rail = torch.device(device), rank, rail
        self.slots, self.capacity_bytes, self.endpoint = slots, capacity_bytes, endpoint
        self._slots, self._lock = [], threading.RLock()
        self._prefix = 'v-source-slots:' + uuid.uuid4().hex
        self._allocate_calls = self._register_calls = self._release_calls = 0
        self._acquired = self._returned = 0
        self._closing = self._closed = False
        self._unknown = None

    def acquire(self, owner, nbytes, *, profile=None):
        if not isinstance(owner, str) or not owner or type(nbytes) is not int or not 0 < nbytes <= self.capacity_bytes:
            raise TransferCapacityError('sparse source exceeds physical slot capacity')
        with self._lock:
            if self._closing or self._closed or self._unknown:
                raise SparseSourceSlotUnknown('source pool is closing or quarantined')
            if any(s.lease is not None and s.lease.owner == owner for s in self._slots):
                raise ValueError('delivery already owns a source slot')
            idle = next((s for s in self._slots if not s.retired and s.lease is None), None)
            new = idle is None
            with profile.measure('allocate') if profile else nullcontext():
                if new:
                    if len(self._slots) >= self.slots:
                        raise TransferCapacityError('all sparse source slots are owned')
                    idle = _Slot(self._prefix + ':' + str(len(self._slots)))
                    self.budget.reserve(idle.owner, self.capacity_bytes, 0)
                    self._slots.append(idle)
                    try:
                        self._allocate_calls += 1
                        idle.buffer = torch.empty(self.capacity_bytes, dtype=torch.uint8, device=self.device)
                    except BaseException:
                        self._slots.remove(idle)
                        self.budget.release(idle.owner)
                        raise
            with profile.measure('register') if profile else nullcontext():
                if new:
                    idle.registration_unknown = True
                    self._register_calls += 1
                    try:
                        idle.registration = self.engine.register_memory(idle.buffer,
                            endpoint=self.endpoint, rank=self.rank, rail=self.rail,
                            metadata={'physical_source_generation': uuid.uuid4().hex})
                        reg = idle.registration
                        if (not isinstance(reg, RegisteredMemory) or reg.buffer is not idle.buffer
                                or reg.descriptor.address != idle.buffer.data_ptr()
                                or reg.descriptor.length != self.capacity_bytes
                                or reg.descriptor.rank != self.rank or reg.descriptor.rail != self.rail
                                or reg.descriptor.device != str(idle.buffer.device)):
                            raise ValueError('physical source registration identity mismatch')
                    except BaseException as exc:
                        idle.unknown = self._unknown = 'physical source registration outcome unknown'
                        raise SparseSourceSlotUnknown(self._unknown) from exc
                    idle.registration_unknown = False
            lease = SparseSourceSlotLease(self, idle, owner, nbytes)
            idle.lease = lease
            self._acquired += 1
            if profile:
                profile.record_source_slot_metrics(reused=not new,
                    slot_bytes=self.capacity_bytes, physical_allocate_calls=int(new),
                    physical_register_calls=int(new))
            return lease

    def _require(self, lease):
        if (not isinstance(lease, SparseSourceSlotLease) or lease._pool is not self
                or not lease._active or lease._slot.lease is not lease):
            raise ValueError('exact active source lease required')

    def _return(self, lease):
        with self._lock:
            self._require(lease)
            slot = lease._slot
            if slot.unknown or slot.registration_unknown:
                raise SparseSourceSlotUnknown('unknown source slot cannot be recycled')
            lease._active = False
            lease.buffer = lease.local = None
            slot.lease = None
            self._returned += 1

    def _quarantine(self, lease, reason):
        with self._lock:
            self._require(lease)
            if not isinstance(reason, str) or not reason:
                raise ValueError('source quarantine reason required')
            lease._slot.unknown = self._unknown = reason

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closing = True
            if self._unknown or any(s.lease is not None or s.registration_unknown for s in self._slots):
                raise SparseSourceSlotUnknown('retain live or unknown physical source owners')
            for slot in self._slots:
                if slot.retired:
                    continue
                self._release_calls += 1
                try:
                    self.engine.release_memory(slot.registration)
                except BaseException as exc:
                    slot.unknown = self._unknown = 'physical source unregister outcome unknown'
                    raise SparseSourceSlotUnknown(self._unknown) from exc
                slot.buffer = slot.registration = None
                slot.retired = True
                self.budget.release(slot.owner)
            self._closed = True

    def snapshot(self):
        with self._lock:
            live = [s for s in self._slots if not s.retired]
            return dict(enabled=True, slots=self.slots, capacity_bytes=self.capacity_bytes,
                physical_slots=len(live), physical_bytes=len(live) * self.capacity_bytes,
                physical_allocate_calls=self._allocate_calls, physical_register_calls=self._register_calls,
                physical_release_calls=self._release_calls, acquired_leases=self._acquired,
                returned_leases=self._returned, leased_slots=sum(s.lease is not None for s in live),
                unknown_slots=sum(bool(s.unknown or s.registration_unknown) for s in live),
                quarantine=self._unknown, closing=self._closing, closed=self._closed)
