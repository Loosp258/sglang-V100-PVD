"""P- and D-side lifecycle adapters for the PVD control/data planes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
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
)
from sglang.srt.disaggregation.pvd.transfer_engine import (
    MemorySlice,
    RegisteredMemory,
    TransferEngine,
    TransferStatus,
)


class PVDDataPlaneError(RuntimeError):
    pass


@dataclass(frozen=True)
class PVDEntryLease:
    manifest: KVEntryManifest
    target_regions: Dict[int, RemoteRegionDescriptor]


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
    ) -> None:
        self.model_instance_id = model_instance_id
        self.coordinator = coordinator
        self.transfer_engine = transfer_engine

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
        result = await self.coordinator.create_entry(manifest)
        targets = {
            int(rank): RemoteRegionDescriptor.from_dict(value)
            for rank, value in result["target_regions"].items()
        }
        return PVDEntryLease(manifest=manifest, target_regions=targets)

    async def publish_shard(
        self,
        *,
        lease: PVDEntryLease,
        rank: int,
        local: MemorySlice,
        first_token: Optional[FirstTokenMetadata] = None,
    ) -> Dict[str, Any]:
        shard = lease.manifest.shard(rank)
        if local.length != shard.expected_bytes:
            raise PVDDataPlaneError(
                f"rank {rank} local KV has {local.length} bytes; "
                f"manifest requires {shard.expected_bytes}"
            )
        if rank != 0 and first_token is not None:
            raise PVDDataPlaneError("only P rank 0 may publish first-token metadata")
        handle = self.transfer_engine.submit_put(local, lease.target_regions[rank])
        status = self.transfer_engine.poll(handle)
        if status != TransferStatus.SUCCESS:
            reason = handle.error or status.value
            await self.coordinator.cancel_entry(lease.manifest.key, reason)
            raise PVDDataPlaneError(f"P rank {rank} -> V rank {rank} failed: {reason}")
        return await self.coordinator.commit_shard(
            lease.manifest.key,
            rank,
            handle.transferred_bytes,
            first_token=first_token,
        )

    async def publish_tensor_shard(
        self,
        *,
        lease: PVDEntryLease,
        rank: int,
        tensor: torch.Tensor,
        endpoint: str,
        rail: str,
        first_token: Optional[FirstTokenMetadata] = None,
    ) -> Dict[str, Any]:
        registration = self.transfer_engine.register_memory(
            tensor, endpoint=endpoint, rank=rank, rail=rail
        )
        try:
            return await self.publish_shard(
                lease=lease,
                rank=rank,
                local=MemorySlice(registration, 0, tensor.numel() * tensor.element_size()),
                first_token=first_token,
            )
        finally:
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
