"""P- and D-side lifecycle adapters for the PVD control/data planes."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import torch
from sglang.srt.disaggregation.pvd.client import PVDCoordinatorClient
from sglang.srt.disaggregation.pvd.protocol import (
    FirstTokenMetadata,
    KVEntryKey,
    KVEntryManifest,
    KVLayoutSignature,
    KVShardManifest,
    RemoteRegionDescriptor,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.transfer_engine import (
    MemorySlice,
    RegisteredMemory,
    TransferEngine,
    TransferHandle,
    TransferStatus,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransportState
from sglang.srt.disaggregation.pvd.upload_manager import PVDUploadManager


class PVDDataPlaneError(RuntimeError):
    pass


@dataclass(frozen=True)
class PVDEntryLease:
    manifest: KVEntryManifest
    target_regions: Dict[int, RemoteRegionDescriptor]
    # One complete write identity per V storage shard rank. Empty for legacy
    # (non-lifecycle) entries; a lifecycle entry always has one per shard.
    upload_identities: Dict[int, WriteIdentity] = field(default_factory=dict)


@dataclass
class PVDDecodeShardLease:
    tensor: torch.Tensor
    registration: RegisteredMemory


class PVDPrefillRuntime:
    """Creates immutable V entries and publishes one complete P KV shard per rank."""

    def __init__(
        self,
        *,
        model_instance_id: str,
        coordinator: PVDCoordinatorClient,
        transfer_engine: TransferEngine,
        upload_manager: Optional[PVDUploadManager] = None,
        worker_epoch: Optional[str] = None,
        poll_interval_seconds: float = 0.002,
        upload_sync_attempts: int = 3,
    ) -> None:
        if (upload_manager is None) != (worker_epoch is None):
            raise ValueError(
                "PVD upload lifecycle requires both an upload manager and a "
                "worker epoch, or neither"
            )
        self.model_instance_id = model_instance_id
        self.coordinator = coordinator
        self.transfer_engine = transfer_engine
        self.upload_manager = upload_manager
        self.worker_epoch = worker_epoch
        self.poll_interval_seconds = poll_interval_seconds
        if upload_sync_attempts < 1:
            raise ValueError("upload_sync_attempts must be at least 1")
        self.upload_sync_attempts = upload_sync_attempts

    @property
    def lifecycle_enabled(self) -> bool:
        return self.upload_manager is not None and self.worker_epoch is not None

    async def create_entry(
        self,
        *,
        req_id: str,
        transfer_id: Optional[str],
        layout: KVLayoutSignature,
        prompt_token_count: int,
        shards: Mapping[int, KVShardManifest],
    ) -> PVDEntryLease:
        key = KVEntryKey(
            model_instance_id=self.model_instance_id,
            req_id=req_id,
            transfer_id=transfer_id or uuid.uuid4().hex,
        )
        manifest = KVEntryManifest(
            key=key,
            layout=layout,
            prompt_token_count=prompt_token_count,
            shards=[shards[rank] for rank in range(layout.tp_size)],
        )
        result = await self.coordinator.create_entry(
            manifest, uploader_epoch=self.worker_epoch
        )
        targets = {
            int(rank): RemoteRegionDescriptor.from_dict(value)
            for rank, value in result["target_regions"].items()
        }
        identities: Dict[int, WriteIdentity] = {}
        for rank, value in (result.get("upload_identities") or {}).items():
            identity = WriteIdentity.from_dict(value)
            identities[int(rank)] = identity
        if self.lifecycle_enabled:
            # V pinned its destination pages before returning the descriptor and
            # the coordinator has already consumed each begin gate. Take
            # ownership of every identity here, before anything can fail: a
            # request aborted between create_entry and publish_shard must still
            # have a record that can report a terminal state, or V would hold
            # those pages with nothing left to ask.
            opened = [
                self.upload_manager.open(
                    identity=identity, coordinator=self.coordinator
                )
                for identity in identities.values()
            ]
            try:
                # V must have bound every shard to this process incarnation
                # before P writes anything. A partial or foreign binding is not
                # usable.
                if set(identities) != set(targets):
                    raise PVDDataPlaneError(
                        "V did not publish an upload identity for every shard"
                    )
                for rank, identity in identities.items():
                    if identity.sender_epoch != self.worker_epoch:
                        raise PVDDataPlaneError(
                            f"V bound shard {rank} to a different uploader epoch"
                        )
                    if identity.key != key:
                        raise PVDDataPlaneError(
                            f"V bound shard {rank} to a different entry key"
                        )
                    identity.validate_destination(targets[rank])
                if len({i.transfer_id for i in identities.values()}) != len(identities):
                    raise PVDDataPlaneError("V published duplicate upload transfer ids")
            except Exception:
                # Nothing was submitted, and nothing will be. Marking the
                # records lets the next progress step report that proven
                # pre-native rejection so V can reclaim; it does not release
                # anything here.
                for record in opened:
                    self.upload_manager.abandon(
                        record.transfer_id, "P rejected the entry lease"
                    )
                raise
        elif identities:
            raise PVDDataPlaneError(
                "V published upload identities for a non-lifecycle entry"
            )
        return PVDEntryLease(
            manifest=manifest,
            target_regions=targets,
            upload_identities=identities,
        )

    async def publish_shard(
        self,
        *,
        lease: PVDEntryLease,
        rank: int,
        local: MemorySlice,
        first_token: Optional[FirstTokenMetadata] = None,
        deadline: Optional[float] = None,
    ) -> Dict[str, Any]:
        shard = lease.manifest.shard(rank)
        if local.length != shard.expected_bytes:
            raise PVDDataPlaneError(
                f"rank {rank} local KV has {local.length} bytes; "
                f"manifest requires {shard.expected_bytes}"
            )
        if rank != 0 and first_token is not None:
            raise PVDDataPlaneError("only P rank 0 may publish first-token metadata")

        identity = lease.upload_identities.get(rank)
        record = None
        if self.lifecycle_enabled:
            if identity is None:
                raise PVDDataPlaneError(
                    f"lifecycle upload for shard {rank} has no write identity"
                )
            record = self.upload_manager.open(
                identity=identity, coordinator=self.coordinator
            )
            try:
                # Claiming is atomic with the forbidden/already-submitted
                # checks, so a concurrent abort cannot let a second write out.
                self.upload_manager.claim_submission(record)
            except RuntimeError as exc:
                raise PVDDataPlaneError(
                    f"shard {rank} upload cannot be submitted: {exc}"
                ) from exc

        handle = self.transfer_engine.submit_put(local, lease.target_regions[rank])
        if record is not None:
            # Ownership of the native handle moves to the manager immediately,
            # so an aborted or cleared sender cannot strand it.
            self.upload_manager.attach(
                record, engine=self.transfer_engine, handle=handle
            )

        status = await self._drain_to_transport_terminal(handle, deadline=deadline)

        acknowledged = True
        if record is not None:
            # Report before committing, so a successful commit is always
            # accompanied by a matching closed terminal on V.
            acknowledged = await self._report_terminal(record, deadline=deadline)

        if (
            status != TransferStatus.SUCCESS
            or handle.transport_state != TransportState.TERMINAL_SUCCESS
        ):
            reason = handle.error or (f"{status.value}/{handle.transport_state.value}")
            # Business cancellation only. The manager keeps this record, its
            # handle and its source pin until a native terminal is observed;
            # a late success after this point permits reclamation on V but
            # never republishes the request.
            await self.coordinator.cancel_entry(lease.manifest.key, reason)
            raise PVDDataPlaneError(f"P rank {rank} -> V rank {rank} failed: {reason}")
        if not acknowledged:
            # The WRITE succeeded but V has not confirmed a closed terminal, so
            # committing here would publish KV that V still treats as an
            # unconfirmed upload. Fail the request; the manager keeps the
            # record and keeps retrying the report.
            raise PVDDataPlaneError(
                f"P rank {rank} upload terminal was not acknowledged by V rank "
                f"{rank}; the upload record is retained for retry"
            )
        return await self.coordinator.commit_shard(
            lease.manifest.key,
            rank,
            handle.transferred_bytes,
            first_token=first_token,
        )

    async def _report_terminal(
        self, record, *, deadline: Optional[float] = None
    ) -> bool:
        """Push this upload's terminal state to V, bounded by attempts/deadline.

        Returning False never means the transfer is finished on V's side: the
        record, its handle and its pins stay with the manager, which retries on
        every later progress step. The bounded autonomous retry driver is
        Task 7; this method only covers the window where a request is live.
        """
        for attempt in range(self.upload_sync_attempts):
            await self.upload_manager.progress_record(record)
            if record.terminal_acked:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            if attempt + 1 < self.upload_sync_attempts:
                await asyncio.sleep(self.poll_interval_seconds)
        return record.terminal_acked

    async def _drain_to_transport_terminal(
        self, handle: TransferHandle, *, deadline: Optional[float] = None
    ) -> TransferStatus:
        """Poll until the native transfer stops, or the business deadline ends.

        A first PENDING poll means the write is still in flight, not that it
        failed. Returning at the deadline records a business failure only: the
        handle, its source pin and its manager record all stay, and the drain
        continues through the manager's progress steps.
        """
        while True:
            status = self.transfer_engine.poll(handle)
            state = handle.transport_state
            if state.is_locally_safe_to_release or state is TransportState.UNKNOWN:
                return status
            if deadline is not None and time.monotonic() >= deadline:
                return status
            await asyncio.sleep(self.poll_interval_seconds)

    async def publish_tensor_shard(
        self,
        *,
        lease: PVDEntryLease,
        rank: int,
        tensor: torch.Tensor,
        endpoint: str,
        rail: str,
        first_token: Optional[FirstTokenMetadata] = None,
        deadline: Optional[float] = None,
    ) -> Dict[str, Any]:
        registration = self.transfer_engine.register_memory(
            tensor, endpoint=endpoint, rank=rank, rail=rail
        )
        try:
            return await self.publish_shard(
                lease=lease,
                rank=rank,
                local=MemorySlice(
                    registration, 0, tensor.numel() * tensor.element_size()
                ),
                first_token=first_token,
                deadline=deadline,
            )
        finally:
            # request_release only: the adapter defers the real deregistration
            # until every transfer pin on this registration is gone.
            self.transfer_engine.release_memory(registration)


class PVDDecodeRuntime:
    """Reserves D buffers, asks V to deliver, then ACKs independently."""

    def __init__(
        self,
        *,
        coordinator: PVDCoordinatorClient,
        transfer_engine: Optional[TransferEngine] = None,
    ) -> None:
        self.coordinator = coordinator
        self.transfer_engine = transfer_engine

    def prepare_shard(
        self,
        *,
        expected_bytes: int,
        device: str,
        endpoint: str,
        rank: int,
        rail: str,
    ) -> PVDDecodeShardLease:
        if self.transfer_engine is None:
            raise PVDDataPlaneError("decode TransferEngine is not configured")
        tensor = torch.empty(expected_bytes, dtype=torch.uint8, device=device)
        registration = self.transfer_engine.register_memory(
            tensor, endpoint=endpoint, rank=rank, rail=rail
        )
        return PVDDecodeShardLease(tensor=tensor, registration=registration)

    def release_shard(self, lease: PVDDecodeShardLease) -> None:
        if self.transfer_engine is None:
            raise PVDDataPlaneError("decode TransferEngine is not configured")
        self.transfer_engine.release_memory(lease.registration)

    async def select_entry(self, key: KVEntryKey) -> Dict[str, Any]:
        result = await self.coordinator.select([key])
        matches = result.get("results", [])
        if len(matches) != 1:
            raise PVDDataPlaneError("V selector returned an invalid result count")
        return matches[0]

    async def deliver(
        self,
        *,
        key: KVEntryKey,
        destinations: Mapping[int, RemoteRegionDescriptor],
        delivery_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        delivery_id = delivery_id or uuid.uuid4().hex
        await self.coordinator.reserve_delivery(key, delivery_id, destinations)
        result = await self.coordinator.start_delivery(delivery_id)
        if result.get("state") != "delivered":
            raise PVDDataPlaneError(
                f"V delivery {delivery_id} did not complete: {result}"
            )
        return result

    async def ack(self, delivery_id: str) -> Dict[str, Any]:
        return await self.coordinator.ack_delivery(delivery_id)
