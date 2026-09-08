"""PVD TransferEngine adapter over SGLang's shared Mooncake implementation."""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Dict, Mapping, Optional

import torch
from sglang.srt.disaggregation.pvd.protocol import RemoteRegionDescriptor
from sglang.srt.disaggregation.pvd.transfer_engine import (
    MemorySlice,
    RegisteredMemory,
    TransferEngine,
    TransferHandle,
    TransferStatus,
)
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)

logger = logging.getLogger(__name__)


class MooncakePVDTransferEngine(TransferEngine):
    """Synchronous Mooncake writes behind the role-neutral PVD API."""

    name = "mooncake"

    def __init__(self, *, hostname: str, gpu_id: int, rail: str) -> None:
        self.rail = rail
        self._engine = MooncakeTransferEngine(
            hostname=hostname,
            gpu_id=gpu_id,
            ib_device=rail,
            require_fresh_metadata=True,
        )
        self._registrations: Dict[str, RegisteredMemory] = {}
        self._handles: Dict[str, TransferHandle] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_existing(
        cls, engine: MooncakeTransferEngine, *, rail: str
    ) -> "MooncakePVDTransferEngine":
        engine.require_pvd_metadata_policy()
        adapter = cls.__new__(cls)
        adapter.rail = rail
        adapter._engine = engine
        adapter._registrations = {}
        adapter._handles = {}
        adapter._lock = threading.Lock()
        return adapter

    def register_memory(
        self,
        buffer: torch.Tensor,
        *,
        endpoint: str,
        rank: int,
        rail: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> RegisteredMemory:
        if not buffer.is_cuda:
            raise RuntimeError("Mooncake PVD only registers CUDA memory")
        if not buffer.is_contiguous():
            raise ValueError("registered buffers must be contiguous")
        if rail != self.rail:
            raise RuntimeError(
                f"engine rail is {self.rail}, registration requested {rail}"
            )
        length = buffer.numel() * buffer.element_size()
        ptr = int(buffer.data_ptr())
        ret = self._engine.engine.register_memory(ptr, length)
        if ret != 0:
            raise RuntimeError(
                f"Mooncake GPU memory registration failed on {rail} with code {ret}"
            )
        region_id = uuid.uuid4().hex
        descriptor = RemoteRegionDescriptor(
            endpoint=self._engine.get_session_id(),
            region_id=region_id,
            address=ptr,
            length=length,
            device=str(buffer.device),
            rank=rank,
            rail=rail,
            backend_metadata={"transport": "mooncake", **dict(metadata or {})},
        )
        registration = RegisteredMemory(descriptor=descriptor, buffer=buffer)
        with self._lock:
            self._registrations[region_id] = registration
        logger.debug(
            "PVD MR registered: session=%s region=%s rank=%s rail=%s address=%#x bytes=%s",
            descriptor.endpoint,
            region_id,
            rank,
            rail,
            ptr,
            length,
        )
        return registration

    def release_memory(self, registration: RegisteredMemory) -> None:
        with self._lock:
            existing = self._registrations.pop(registration.descriptor.region_id, None)
        if existing is not None:
            ret = self._engine.engine.unregister_memory(
                existing.descriptor.address
                - int(existing.descriptor.backend_metadata.get("base_offset", 0))
            )
            if ret != 0:
                raise RuntimeError(f"Mooncake memory deregistration failed: {ret}")
            logger.debug(
                "PVD MR unregistered: session=%s region=%s rail=%s address=%#x",
                existing.descriptor.endpoint,
                existing.descriptor.region_id,
                self.rail,
                existing.descriptor.address,
            )

    def submit_put(
        self,
        local: MemorySlice,
        remote: RemoteRegionDescriptor,
        *,
        remote_offset: int = 0,
    ) -> TransferHandle:
        handle = TransferHandle(transfer_id=uuid.uuid4().hex)
        if remote.rail != self.rail:
            handle.status = TransferStatus.FAILED
            handle.error = f"rank-local rail mismatch: {self.rail} -> {remote.rail}"
            return handle
        if remote_offset < 0 or remote_offset + local.length > remote.length:
            handle.status = TransferStatus.FAILED
            handle.error = "PUT exceeds bounded remote region"
            return handle

        # Packing and heterogeneous head staging use CUDA kernels. GPUDirect
        # RDMA is not ordered behind PyTorch's current CUDA stream, so make the
        # source bytes visible before Mooncake starts reading GPU memory.
        try:
            torch.cuda.synchronize(local.registration.buffer.device)
        except Exception as exc:
            handle.status = TransferStatus.FAILED
            handle.error = f"CUDA source synchronization failed: {exc}"
            return handle

        local_address = local.registration.descriptor.address + local.offset
        remote_address = remote.address + remote_offset
        # region_id is a PVD identity, NOT a Mooncake rkey. Fresh descriptor
        # resolution happens inside classic Mooncake on every remote lookup,
        # enabled before its first native import. No 'seen region' shortcut.
        self._engine.require_pvd_metadata_policy()
        logger.debug(
            "PVD PUT submit: transfer=%s peer=%s region=%s rail=%s address=%#x "
            "bytes=%s metadata_policy=fresh",
            handle.transfer_id,
            remote.endpoint,
            remote.region_id,
            self.rail,
            remote_address,
            local.length,
        )
        ret = self._engine.transfer_sync(
            remote.endpoint, local_address, remote_address, local.length
        )
        if ret < 0:
            handle.status = TransferStatus.FAILED
            handle.error = f"Mooncake transfer_sync returned {ret}"
        else:
            handle.status = TransferStatus.SUCCESS
            handle.transferred_bytes = local.length
        logger.debug(
            "PVD PUT return: transfer=%s peer=%s region=%s result=%s",
            handle.transfer_id,
            remote.endpoint,
            remote.region_id,
            ret,
        )
        with self._lock:
            self._handles[handle.transfer_id] = handle
        return handle

    def poll(self, handle: TransferHandle) -> TransferStatus:
        return handle.status

    def abort(self, handle: TransferHandle) -> None:
        if handle.status == TransferStatus.PENDING:
            handle.status = TransferStatus.CANCELLED

    def health(self) -> Dict[str, Any]:
        with self._lock:
            registrations = len(self._registrations)
        return {
            "backend": self.name,
            "healthy": True,
            "rail": self.rail,
            "session_id": self._engine.get_session_id(),
            "registered_regions": registrations,
            "metadata_policy": "fresh",
            "metadata_policy_verification": "version-pinned-pre-init",
            "mooncake_version": self._engine.pvd_metadata_version,
        }
