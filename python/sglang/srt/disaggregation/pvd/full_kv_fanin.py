"""D-side multi-writer receive lifetime; not yet the production wire adapter.

The caller pins before publishing, adopts the authenticated coordinator's whole
authorization set, and reports identity-complete sender fences on the owner
thread. These reports are protocol evidence, not locally observed RDMA fences.
"""

import copy
import hashlib
import json
import threading
import uuid
from dataclasses import asdict
from typing import Mapping

from sglang.srt.disaggregation.pvd.protocol import (
    PVD_GENERATION_METADATA_KEY,
    PVD_RECEIVER_EPOCH_METADATA_KEY,
    PVD_TRANSFER_LIFECYCLE_PROTOCOL,
    KVEntryKey,
    ProtocolValidationError,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.sharding import (
    packed_fanin_transfer_slices,
    source_shard_intersections,
)
from sglang.srt.disaggregation.pvd.transfer_engine import RegisteredMemory
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransportState,
)

FULL_KV_FANIN_PROTOCOL = "pvd-full-kv-fanin-v1"


class FullKVFanInReceiver:
    """One D MR, several V writers, one pin until EVERY writer is fenced.

    Frozen relative slice plans bind the receiver's expected byte coverage.
    Production sender-side range enforcement must implement the same plan;
    this class alone cannot constrain an RDMA-capable remote peer.
    """

    def __init__(
        self,
        *,
        key,
        delivery_id,
        registration,
        guard,
        storage,
        compute,
        token_count,
        max_slices,
    ):
        if (
            not isinstance(key, KVEntryKey)
            or not isinstance(delivery_id, str)
            or not delivery_id.strip()
            or not isinstance(registration, RegisteredMemory)
            or not isinstance(guard, ResourceGuard)
            or guard.value is not registration
        ):
            raise ProtocolValidationError(
                "exact registered MR owner and delivery required"
            )
        if (
            type(token_count) is not int
            or token_count <= 0
            or type(max_slices) is not int
            or max_slices <= 0
        ):
            raise ProtocolValidationError("positive token and slice bounds required")
        descriptor = copy.deepcopy(registration.descriptor)
        storage, compute = copy.deepcopy(storage), copy.deepcopy(compute)
        parts = source_shard_intersections(storage, compute, descriptor.rank)
        length = sum(compute.extra["component_bytes_per_token"]) * token_count
        if (
            type(descriptor.length) is not int
            or descriptor.length != length
            or not registration.buffer.is_contiguous()
            or registration.buffer.data_ptr() != descriptor.address
            or registration.buffer.numel() * registration.buffer.element_size()
            != length
        ):
            raise ProtocolValidationError(
                "fan-in destination length differs from layout"
            )
        if (
            len(parts) * len(compute.extra["component_bytes_per_token"]) * token_count
            > max_slices
        ):
            raise ProtocolValidationError("fan-in slice count exceeds configured bound")
        plans = packed_fanin_transfer_slices(
            storage, compute, compute_rank=descriptor.rank, token_count=token_count
        )
        # Validate the receiver-owned identity now, before taking any ownership.
        identity = dict(
            protocol=PVD_TRANSFER_LIFECYCLE_PROTOCOL,
            sender_epoch="pending-authorization",
            receiver_epoch=descriptor.backend_metadata.get(
                PVD_RECEIVER_EPOCH_METADATA_KEY
            ),
            transfer_id=delivery_id,
            region_id=descriptor.region_id,
            generation=descriptor.backend_metadata.get(PVD_GENERATION_METADATA_KEY),
            shard_rank=descriptor.rank,
            key=key,
        )
        WriteIdentity(**identity).validate_destination(descriptor)
        self._descriptor, self._key, self._delivery = descriptor, key, delivery_id
        self._registration_id = id(registration)
        self._plans = {rank: tuple(parts) for rank, parts in plans.items()}
        self._bytes = {
            rank: sum(p.length for p in parts) for rank, parts in plans.items()
        }
        manifest = {
            "protocol": FULL_KV_FANIN_PROTOCOL,
            "key": key.to_dict(),
            "delivery_id": delivery_id,
            "destination": descriptor.to_dict(),
            "storage_layout": storage.to_dict(),
            "compute_layout": compute.to_dict(),
            "token_count": token_count,
            "writers": {
                str(rank): [asdict(part) for part in parts]
                for rank, parts in self._plans.items()
            },
        }
        self._fingerprint = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self._manifest = copy.deepcopy(manifest)
        self._owner_thread = threading.get_ident()
        self._guard, self._pin = guard, "full-kv-fanin:" + uuid.uuid4().hex
        self._published = self._cancelled = self._closed = False
        self._identities, self._proofs = {}, {}
        guard.pin(self._pin)

    def _check(self):
        if threading.get_ident() != self._owner_thread or self._closed:
            raise ProtocolValidationError(
                "fan-in receiver is closed or on another thread"
            )
        registration = self._guard.value
        if (
            not isinstance(registration, RegisteredMemory)
            or id(registration) != self._registration_id
            or registration.descriptor != self._descriptor
            or registration.buffer.data_ptr() != self._descriptor.address
        ):
            raise ProtocolValidationError("fan-in registered destination changed")

    def publish(self):
        """Call before any descriptor leaves D, including a failing RPC send."""
        self._check()
        if self._cancelled or self._published:
            raise ProtocolValidationError("fan-in publication is one-shot")
        self._published = True
        return {**copy.deepcopy(self._manifest), "plan_fingerprint": self._fingerprint}

    def adopt(self, identities: Mapping[int, WriteIdentity]):
        """Complete V-rank-keyed set; WriteIdentity.shard_rank remains D rank.

        Sender epochs come from the authenticated reserve response. Each sender
        must enforce the transfer ID's source rank and this plan fingerprint.
        No dict-by-D-rank conversion is permitted: it would lose peer writers.
        """
        self._check()
        if not self._published or not isinstance(identities, Mapping):
            raise ProtocolValidationError("publish before adopting writer identities")
        if any(type(rank) is not int for rank in identities) or set(identities) != set(
            self._plans
        ):
            raise ProtocolValidationError("complete V writer identity set required")
        adopted = dict(identities)
        for rank, identity in adopted.items():
            if not isinstance(identity, WriteIdentity):
                raise ProtocolValidationError("typed writer identity required")
            identity.validate_destination(self._descriptor)
            if (
                identity.key != self._key
                or identity.transfer_id
                != f"{self._delivery}:d{self._descriptor.rank}:v{rank}"
            ):
                raise ProtocolValidationError("writer does not belong to this fan-in")
        if self._identities and adopted != self._identities:
            raise ProtocolValidationError("cannot replace adopted writer identities")
        self._identities = adopted

    def observe(self, reply):
        """Accept only exact sender closure proof; timeout/cancel is not one."""
        self._check()
        fields = {
            "protocol",
            "plan_fingerprint",
            "source_rank",
            "identity",
            "fenced",
            "transport_state",
            "transferred_bytes",
        }
        if not isinstance(reply, Mapping) or set(reply) != fields:
            raise ProtocolValidationError("complete fan-in closure reply required")
        rank = reply["source_rank"]
        if type(rank) is not int or rank not in self._identities:
            raise ProtocolValidationError("unknown or unadopted V writer")
        identity = WriteIdentity.from_dict(reply["identity"])
        if (
            reply["protocol"] != FULL_KV_FANIN_PROTOCOL
            or reply["plan_fingerprint"] != self._fingerprint
            or identity != self._identities[rank]
            or type(reply["fenced"]) is not bool
        ):
            raise ProtocolValidationError("fan-in closure identity/plan mismatch")
        try:
            state = TransportState(reply["transport_state"])
        except (ValueError, TypeError) as exc:
            raise ProtocolValidationError("unknown transport state") from exc
        transferred = reply["transferred_bytes"]
        if type(transferred) is not int or not 0 <= transferred <= self._bytes[rank]:
            raise ProtocolValidationError("invalid writer transferred byte count")
        if not reply["fenced"]:
            return False
        if not state.is_locally_safe_to_release:
            raise ProtocolValidationError("fenced reply lacks terminal transport proof")
        if (state == TransportState.NOT_SUBMITTED and transferred != 0) or (
            state == TransportState.TERMINAL_SUCCESS
            and transferred != self._bytes[rank]
        ):
            raise ProtocolValidationError("terminal state contradicts byte coverage")
        proof = (state, transferred)
        if rank in self._proofs and self._proofs[rank] != proof:
            raise ProtocolValidationError("writer terminal proof changed")
        self._proofs[rank] = proof
        return True

    @property
    def ready(self):
        """Network completion only; D must still unpack/fence/install its banks."""
        self._check()
        return (
            not self._cancelled
            and set(self._proofs) == set(self._plans)
            and all(
                state == TransportState.TERMINAL_SUCCESS
                for state, _ in self._proofs.values()
            )
        )

    def cancel(self):
        self._check()
        self._cancelled = True  # Never unpin or invent NOT_SUBMITTED.

    def close(self):
        """Release this network pin; local consumers must retain their own pins."""
        self._check()
        if self._published and set(self._proofs) != set(self._plans):
            raise ProtocolValidationError("all possible V writers must be fenced")
        self._closed = True
        self._guard.unpin(self._pin)
        self._guard = None
