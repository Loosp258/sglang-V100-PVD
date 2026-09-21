"""Owned CPU destinations for the sparse shard Delivery protocol.

The registry must outlive its HTTP tasks. Unknown registration or write state
retains the tensor and budget; cancellation is never a memory fence. The first
consumer is CPUInstallGroup, not GPU attention or the production Scheduler.
Trusted shard-control responses are lifecycle evidence, not authentication of
an untrusted network peer. No finalizer releases possibly-live registrations.
"""

import asyncio
import threading
import uuid

import torch
from sglang.srt.disaggregation.pvd.protocol import (
    PVD_GENERATION_METADATA_KEY,
    PVD_RECEIVER_EPOCH_METADATA_KEY,
    PVD_TRANSFER_LIFECYCLE_PROTOCOL,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.sparse_delivery import (
    SPARSE_DELIVERY_KEY,
    SparseDeliveryManifest,
)
from sglang.srt.disaggregation.pvd.sparse_install import CPUInstallGroup
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


class SparseReceiveError(ValueError):
    pass


class SparseReceiveRegistry:
    """Single event-loop/thread owner, including synchronous CPU installation.

    prepare() inserts an owner BEFORE registration. Failed registrations are
    deliberately retained, and visible through snapshot(), even if prepare()
    raises and the caller never receives the record. Native recovery requires
    external proof; there is deliberately no force-free method.
    """

    def __init__(self, engine, budget, *, receiver_epoch):
        if not isinstance(budget, TransferBudget):
            raise SparseReceiveError("an explicit receive budget is required")
        if not isinstance(receiver_epoch, str) or not receiver_epoch.strip():
            raise SparseReceiveError("explicit receiver epoch required")
        self.engine, self.budget, self.receiver_epoch = engine, budget, receiver_epoch
        self._thread = threading.get_ident()
        self._records = {}

    def _owner(self):
        if threading.get_ident() != self._thread:
            raise SparseReceiveError("receiver must run on its owner thread")

    def prepare(
        self,
        manifest,
        *,
        key,
        rank,
        rail,
        endpoint,
        sender_epoch,
        client,
        owner_scope=None,
    ):
        self._owner()
        if not isinstance(manifest, SparseDeliveryManifest):
            raise SparseReceiveError("explicit sparse manifest required")
        if owner_scope is not None and (
            not isinstance(owner_scope, str) or not owner_scope.strip()
        ):
            raise SparseReceiveError("owner scope must be a nonempty string")
        delivery_id, generation = uuid.uuid4().hex, uuid.uuid4().hex
        # Validate identity before charging or allocating. The actual region is
        # filled only after registration; never learned from a response.
        identity = WriteIdentity(
            PVD_TRANSFER_LIFECYCLE_PROTOCOL,
            sender_epoch,
            self.receiver_epoch,
            delivery_id,
            "pending-registration",
            generation,
            rank,
            key,
        )
        if manifest.specs[0].entry_transfer_id != key.transfer_id:
            raise SparseReceiveError("manifest and Entry identity differ")
        if any(not isinstance(s, str) or not s.strip() for s in (rail, endpoint)):
            raise SparseReceiveError("explicit endpoint and rail required")
        record = SparseReceiveRecord(self, manifest, identity, client)
        record._scope = owner_scope
        self.budget.reserve(record.owner, manifest.nbytes, 1)
        self._records[delivery_id] = record
        try:
            record._buffer = torch.empty(manifest.nbytes, dtype=torch.uint8)
        except BaseException:
            self.budget.release(record.owner)
            del self._records[delivery_id]
            raise
        record._registration_unknown = True
        # A raising register_memory may have entered native code. Preserve its
        # backing storage rather than guessing that registration never happened.
        record._registration = self.engine.register_memory(
            record._buffer,
            endpoint=endpoint,
            rank=rank,
            rail=rail,
            metadata={
                PVD_RECEIVER_EPOCH_METADATA_KEY: self.receiver_epoch,
                PVD_GENERATION_METADATA_KEY: generation,
                SPARSE_DELIVERY_KEY: manifest.to_dict(),
            },
        )
        descriptor = record._registration.descriptor
        record.identity = WriteIdentity(
            **{**identity.__dict__, "region_id": descriptor.region_id}
        )
        record.identity.validate_destination(descriptor)
        if descriptor.length != manifest.nbytes:
            raise SparseReceiveError("registration extent differs from manifest")
        record._registration_unknown = False
        return record

    def snapshot(self, *, owner_scope=None):
        self._owner()
        return {
            key: record.snapshot()
            for key, record in self._records.items()
            if owner_scope is None or record._scope == owner_scope
        }

    async def close(self, *, owner_scope=None):
        """Bounded by each client's RPC timeout; unresolved owners stay here."""
        self._owner()
        errors = {}
        for key, record in tuple(self._records.items()):
            if owner_scope is not None and record._scope != owner_scope:
                continue
            try:
                if not await record.close():
                    errors[key] = "remote write or registration is not fenced"
            except Exception as exc:  # noqa: BLE001 -- retain owner and report cleanup failure
                errors[key] = str(exc)
        return errors


class SparseReceiveRecord:
    def __init__(self, registry, manifest, identity, client):
        self._registry, self.manifest, self.identity = registry, manifest, identity
        self._client = client
        self.owner = f"d-sparse:{identity.transfer_id}"
        self._buffer = self._registration = None
        self._registration_unknown = False
        self._published = self._safe = self._ready = False
        self._closing = self._closed = self._acknowledged = False
        self._group = self._receipt = None
        self._installed = False
        self._lock = asyncio.Lock()

    def _live(self):
        self._registry._owner()
        if self._closed or self._closing or self._registration_unknown:
            raise SparseReceiveError("receive destination is closed or quarantined")

    def _fenced(self, proof):
        if not isinstance(proof, dict) or type(proof.get("fenced")) is not bool:
            raise SparseReceiveError("explicit write fence proof required")
        identity = WriteIdentity.from_dict(
            {key: value for key, value in proof.items() if key != "fenced"}
        )
        if identity != self.identity:
            raise SparseReceiveError("write fence identity mismatch")
        return proof["fenced"]

    def _observe(self, reply):
        if (
            reply.get("delivery_id") != self.identity.transfer_id
            or reply.get("entry_key") != self.identity.key.to_dict()
            or reply.get("destination") != self._registration.descriptor.to_dict()
            or reply.get("sparse_fingerprint") != self.manifest.fingerprint
            or WriteIdentity.from_dict(reply.get("write_identity")) != self.identity
        ):
            raise SparseReceiveError(
                "delivery identity, destination or manifest mismatch"
            )
        safe = self._fenced(reply.get("write_fence"))
        ready = reply.get("state") in ("delivered", "acked", "released")
        if ready and (
            not safe
            or reply.get("transport_state") != "terminal_success"
            or type(reply.get("transferred_bytes")) is not int
            or reply["transferred_bytes"] != self.manifest.nbytes
        ):
            raise SparseReceiveError("delivery lacks exact successful terminal proof")
        self._safe |= safe
        self._ready |= ready
        if reply.get("state") in ("failed", "cancelled", "expired"):
            raise SparseReceiveError(f"sparse delivery {reply['state']}")
        return self._ready

    async def start(self):
        async with self._lock:
            self._live()
            if self._published:
                raise SparseReceiveError(
                    "destination already published; poll or fence it"
                )
            self._published = True  # BEFORE the first await, including lost replies
            reply = await self._client.reserve_delivery(
                self.identity.key,
                self.identity.transfer_id,
                self._registration.descriptor,
            )
            self._observe(reply)
            return self._observe(
                await self._client.start_delivery(
                    self.identity.key, self.identity.transfer_id
                )
            )

    async def poll(self):
        async with self._lock:
            self._live()
            if not self._published:
                raise SparseReceiveError("destination has not been published")
            return self._observe(
                await self._client.poll_delivery(
                    self.identity.key, self.identity.transfer_id
                )
            )

    def stage(self, group, epoch):
        self._live()
        if self._lock.locked() or not self._ready:
            raise SparseReceiveError("wait for successful delivery before staging")
        if self._receipt is not None:
            raise SparseReceiveError("destination already staged")
        if (
            not isinstance(group, CPUInstallGroup)
            or self.manifest.dtype != "torch.float32"
        ):
            raise SparseReceiveError("first installer supports CPU FP32 banks only")
        # Synchronous CPU copies; no await or exposed view escapes this scope.
        receipt = group.stage(
            epoch, self.identity.shard_rank, self.manifest.payload_views(self._buffer)
        )
        self._group, self._receipt = group, receipt
        return receipt

    async def ack(self):
        async with self._lock:
            self._live()
            self.confirm_install()
            reply = await self._client.ack_delivery(
                self.identity.key, self.identity.transfer_id
            )
            self._observe(reply)
            if reply.get("state") != "released":
                raise SparseReceiveError("delivery ACK was not confirmed")
            self._acknowledged = True

    def confirm_install(self):
        """Latch the exact CPU completion before a later round can replace it."""
        self._live()
        if not self._installed:
            if self._group is None or not self._group.installation_complete(
                self._receipt
            ):
                raise SparseReceiveError("all ranks must install before delivery ACK")
            self._installed = True

    async def close(self):
        async with self._lock:
            self._registry._owner()
            if self._closed:
                return True
            self._closing = True
            if self._registration_unknown:
                return False
            # An unacknowledged successful transfer also needs business closure
            # on V, otherwise its Entry retains an active Delivery until TTL.
            if self._published and (not self._safe or not self._acknowledged):
                self._safe = self._fenced(
                    await self._client.fence_delivery(self.identity)
                )
                if not self._safe:
                    return False
            # No concurrent CPU reader: stage() is synchronous on this owner
            # thread and rejects while this RPC/cleanup scope is held.
            self._registry.engine.release_memory(self._registration)
            self._registration = self._buffer = None
            self._registry.budget.release(self.owner)
            self._closed = True
            self._group = self._receipt = None
            self._registry._records.pop(self.identity.transfer_id, None)
            return True

    def snapshot(self):
        self._registry._owner()
        return {
            "published": self._published,
            "fenced": self._safe,
            "ready": self._ready,
            "staged": self._receipt is not None,
            "installed": self._installed,
            "acknowledged": self._acknowledged,
            "closing": self._closing,
            "closed": self._closed,
            "registration_unknown": self._registration_unknown,
        }
