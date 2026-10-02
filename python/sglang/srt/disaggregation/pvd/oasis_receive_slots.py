"""Bounded request-owned receive registrations for the Oasis experiment.

Only physical CUDA allocation/MR owners are shared across layer jobs. Each
delivery still gets a private exact-length descriptor, manifest, UUID generation
and receive record on its worker thread. The caller must prove remote closure
and finish every local reader before returning a lease. UNKNOWN leases are never
recycled. No finalizer retires native registrations.
"""

import dataclasses
import threading
import time
import uuid

import torch
from sglang.srt.disaggregation.pvd.protocol import (
    PVD_GENERATION_METADATA_KEY,
    PVD_RECEIVER_EPOCH_METADATA_KEY,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.sparse_delivery import (
    SPARSE_DELIVERY_KEY,
    SparseDeliveryManifest,
)
from sglang.srt.disaggregation.pvd.sparse_receiver import SparseReceiveError
from sglang.srt.disaggregation.pvd.transfer_engine import (
    RegisteredMemory,
    TransferEngine,
    descriptor_with_slice,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


@dataclasses.dataclass
class _PhysicalSlot:
    owner: str
    rank: int
    endpoint: str
    rail: str
    buffer: object = None
    registration: object = None
    registration_unknown: bool = False
    leased: bool = False
    lease: object = None
    unknown: object = None
    retired: bool = False


class OasisReceiveSlotLease:
    """One bounded logical write destination; never an engine MR owner.

``registration`` is an alias used for exact descriptor observation and local
ordering only. Passing it to ``engine.release_memory`` is forbidden: only the
pool owns, and eventually unregisters, the physical RegisteredMemory object.
"""

    def __init__(self, pool, slot, manifest, identity, *, allocate_seconds,
                 register_seconds, physical_register_calls):
        self._pool, self._slot = pool, slot
        self._active = True
        self.allocate_seconds = allocate_seconds
        self.register_seconds = register_seconds
        self.physical_register_calls = physical_register_calls
        self.buffer = slot.buffer[:manifest.nbytes]
        descriptor = descriptor_with_slice(slot.registration, offset=0,
                                           length=manifest.nbytes)
        metadata = dict(descriptor.backend_metadata)
        metadata.update({
            PVD_RECEIVER_EPOCH_METADATA_KEY: identity.receiver_epoch,
            PVD_GENERATION_METADATA_KEY: identity.generation,
            SPARSE_DELIVERY_KEY: manifest.to_dict(),
        })
        descriptor = dataclasses.replace(descriptor, backend_metadata=metadata)
        self.registration = RegisteredMemory(descriptor, self.buffer)
        self.identity = dataclasses.replace(identity, region_id=descriptor.region_id)
        self.identity.validate_destination(descriptor)

    def release_after_proof(self):
        """Return after the receive record proves remote and local completion.

This internal call is not evidence of a WRITE fence. Its caller retains the
normal record's exact identity, terminal byte proof, ACK/fence, GPUDirect
ordering and local copy completion checks. An unpublished record may return
after proving that no writer was authorized and no local reader started.
"""
        self._pool._return_lease(self)

    def quarantine(self, reason):
        """Keep the physical owner, exact alias and budget for process recovery."""
        self._pool._quarantine_lease(self, reason)


class OasisReceiveSlotPool:
    """Thread-independent per-rank slots, bounded across both job executors.

Bootstrap and Decode have different ThreadPoolExecutors. Binding registrations
to thread IDs would double physical owners. Slots instead remain request-owned;
each lease is exclusive until its worker-local record retires with full proof.

The physical maximum extent is charged for the entire MR lifetime, including
idle slots. Caller receive records separately charge (0 bytes, 1 transfer slot)
for each logical write, preserving the existing live-write admission bound.
"""

    def __init__(self, engine, budget, *, device, receiver_epoch, slots_per_rank,
                 capacity_bytes, ranks=(0, 1)):
        device = torch.device(device)
        if (not isinstance(engine, TransferEngine)
                or not isinstance(budget, TransferBudget)
                or device.type != "cuda" or device.index is None
                or not isinstance(receiver_epoch, str) or not receiver_epoch.strip()
                or type(slots_per_rank) is not int or not 1 <= slots_per_rank <= 4
                or type(capacity_bytes) is not int
                or not 0 < capacity_bytes <= 2 * 2048 * 512
                or ranks != (0, 1)):
            raise SparseReceiveError("explicit bounded two-rank CUDA receive pool required")
        self.engine, self.budget = engine, budget
        self.device, self.receiver_epoch = device, receiver_epoch
        self.slots_per_rank, self.capacity_bytes = slots_per_rank, capacity_bytes
        self.ranks = ranks
        self._owner = "oasis-receive-pool:" + uuid.uuid4().hex
        self._lock = threading.RLock()
        self._slots = []
        self._generations, self._delivery_ids = set(), set()
        self._quarantine_reason = None
        self._closing = self._closed = False
        self._register_calls = self._registered = 0
        self._release_calls = self._released = 0
        self._acquired = self._returned = 0

    def _allocate_buffer(self):
        return torch.empty(self.capacity_bytes, dtype=torch.uint8, device=self.device)

    def _validate_request(self, manifest, identity, endpoint, rail, device, ordering):
        if (not isinstance(manifest, SparseDeliveryManifest)
                or not isinstance(identity, WriteIdentity)
                or identity.receiver_epoch != self.receiver_epoch
                or identity.region_id != "pending-registration"
                or identity.shard_rank not in self.ranks
                or torch.device(device) != self.device
                or any(not isinstance(s, str) or not s.strip() for s in (endpoint, rail))
                or not callable(getattr(ordering, "prepare", None))):
            raise SparseReceiveError("matching private receive identity, device and route required")
        specs = manifest.specs
        start_head = identity.shard_rank * 2
        if (manifest.dtype != "torch.float16" or manifest.head_dim != 128
                or manifest.nbytes > self.capacity_bytes
                or specs[0].entry_transfer_id != identity.key.transfer_id
                or len(specs) > 2 or len({s.layer for s in specs}) != 1
                or any(not start_head <= s.kv_head < start_head + 2 for s in specs)):
            raise SparseReceiveError("sparse extent or heads exceed this Oasis rank slot")

    def _validate_registration(self, slot):
        registration, buffer = slot.registration, slot.buffer
        if not isinstance(registration, RegisteredMemory):
            raise SparseReceiveError("physical registration object required")
        descriptor = registration.descriptor
        if (descriptor.rank != slot.rank or descriptor.rail != slot.rail
                or descriptor.endpoint != slot.endpoint
                or descriptor.device != str(self.device)
                or descriptor.address != buffer.data_ptr()
                or descriptor.length != self.capacity_bytes
                or registration.buffer.data_ptr() != buffer.data_ptr()
                or registration.buffer.device != self.device
                or registration.buffer.numel() * registration.buffer.element_size()
                != self.capacity_bytes
                or descriptor.backend_metadata.get(PVD_RECEIVER_EPOCH_METADATA_KEY)
                != self.receiver_epoch
                or any(other is not slot and other.registration is not None
                       and other.registration.descriptor.region_id == descriptor.region_id
                       for other in self._slots)):
            raise SparseReceiveError("physical receive registration differs from owned slot")

    def acquire(self, manifest, identity, *, endpoint, rail, device, ordering):
        self._validate_request(manifest, identity, endpoint, rail, device, ordering)
        with self._lock:
            if self._closing or self._closed or self._quarantine_reason is not None:
                raise SparseReceiveError("receive slot pool is closed or quarantined")
            if (identity.generation in self._generations
                    or identity.transfer_id in self._delivery_ids):
                raise SparseReceiveError("receive slot write identity was already used")
            rank_slots = [s for s in self._slots
                          if s.rank == identity.shard_rank and not s.retired]
            if any(s.rail != rail or s.endpoint != endpoint for s in rank_slots):
                raise SparseReceiveError("receive slot rank cannot change endpoint or rail")
            slot = next((s for s in rank_slots if not s.leased and s.unknown is None), None)
            allocate_seconds = register_seconds = 0.0
            physical_register_calls = 0
            if slot is None:
                if len(rank_slots) >= self.slots_per_rank:
                    raise SparseReceiveError("all bounded receive rank slots are busy")
                slot = _PhysicalSlot(f"{self._owner}:r{identity.shard_rank}:s{len(rank_slots)}",
                                     identity.shard_rank, endpoint, rail)
                self.budget.reserve(slot.owner, self.capacity_bytes, 0)
                self._slots.append(slot)  # Own BEFORE allocation/native registration.
                started = time.perf_counter()
                try:
                    slot.buffer = self._allocate_buffer()
                except BaseException:
                    self._slots.remove(slot)
                    self.budget.release(slot.owner)
                    raise
                finally:
                    allocate_seconds = time.perf_counter() - started
                try:
                    # Establish SYNC_MEMOPS once per physical allocation before MR publication.
                    ordering.prepare(slot.buffer)
                except BaseException:
                    slot.buffer = None
                    self._slots.remove(slot)
                    self.budget.release(slot.owner)
                    raise
                slot.registration_unknown = True
                started = time.perf_counter()
                self._register_calls += 1
                physical_register_calls = 1
                try:
                    slot.registration = self.engine.register_memory(
                        slot.buffer, endpoint=endpoint, rank=identity.shard_rank,
                        rail=rail, metadata={
                            PVD_RECEIVER_EPOCH_METADATA_KEY: self.receiver_epoch,
                            PVD_GENERATION_METADATA_KEY: "physical:" + uuid.uuid4().hex,
                        })
                    self._validate_registration(slot)
                except BaseException:
                    slot.unknown = "physical receive registration outcome unknown"
                    self._quarantine_reason = slot.unknown
                    raise
                finally:
                    register_seconds = time.perf_counter() - started
                slot.registration_unknown = False
                self._registered += 1
            try:
                lease = OasisReceiveSlotLease(self, slot, manifest, identity,
                    allocate_seconds=allocate_seconds, register_seconds=register_seconds,
                    physical_register_calls=physical_register_calls)
            except BaseException:
                # No descriptor has escaped and the owned physical MR remains idle.
                raise
            slot.leased, slot.lease = True, lease
            self._generations.add(identity.generation)
            self._delivery_ids.add(identity.transfer_id)
            self._acquired += 1
            return lease

    def _require_lease(self, lease):
        if (not isinstance(lease, OasisReceiveSlotLease) or lease._pool is not self
                or not lease._active or lease._slot.lease is not lease
                or not lease._slot.leased):
            raise SparseReceiveError("exact live receive slot lease required")

    def _return_lease(self, lease):
        with self._lock:
            self._require_lease(lease)
            slot = lease._slot
            if slot.unknown is not None or slot.registration_unknown:
                raise SparseReceiveError("unknown receive slot cannot be recycled")
            lease._active = False
            lease.buffer = lease.registration = None
            slot.leased, slot.lease = False, None
            self._returned += 1

    def _quarantine_lease(self, lease, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise SparseReceiveError("explicit receive quarantine reason required")
        with self._lock:
            self._require_lease(lease)
            lease._slot.unknown = reason
            self._quarantine_reason = reason

    def close(self):
        """Retire exact physical owners only after joined workers and safe leases.

The caller must join both LayerLookahead executors before this method. Native
unregister failure is sticky: inventory and its physical byte charge remain,
and a second close cannot guess whether the first native call completed.
"""
        with self._lock:
            if self._closed:
                return
            if self._quarantine_reason is not None or any(
                    s.leased or s.registration_unknown or s.unknown is not None
                    for s in self._slots if not s.retired):
                raise SparseReceiveError("retain live or unknown receive pool owners")
            self._closing = True
            for slot in self._slots:
                if slot.retired:
                    continue
                self._release_calls += 1
                try:
                    # Never pass a logical alias to the engine's identity-checked API.
                    self.engine.release_memory(slot.registration)
                except BaseException:
                    slot.unknown = "physical receive unregister outcome unknown"
                    self._quarantine_reason = slot.unknown
                    raise
                slot.registration = slot.buffer = None
                slot.retired = True
                self.budget.release(slot.owner)
                self._released += 1
            self._closed = True

    def snapshot(self):
        with self._lock:
            live = [s for s in self._slots if not s.retired]
            return dict(slots_per_rank=self.slots_per_rank,
                capacity_bytes=self.capacity_bytes, physical_slots=len(live),
                physical_register_calls=self._register_calls,
                physical_registrations=self._registered,
                physical_release_calls=self._release_calls,
                physical_releases=self._released,
                leased_slots=sum(s.leased for s in live),
                unknown_slots=sum(s.unknown is not None or s.registration_unknown for s in live),
                physical_bytes=len(live) * self.capacity_bytes,
                acquired_leases=self._acquired, returned_leases=self._returned,
                quarantine=self._quarantine_reason,
                closing=self._closing, closed=self._closed)
