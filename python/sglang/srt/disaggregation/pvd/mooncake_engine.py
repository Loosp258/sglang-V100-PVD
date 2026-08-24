"""PVD TransferEngine adapter over SGLang's shared Mooncake implementation."""

from __future__ import annotations

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


class MooncakePVDTransferEngine(TransferEngine):
    """Synchronous Mooncake writes behind the role-neutral PVD API."""

    name = "mooncake"

    def __init__(self, *, hostname: str, gpu_id: int, rail: str) -> None:
        self.rail = rail
        self._engine = MooncakeTransferEngine(
            hostname=hostname, gpu_id=gpu_id, ib_device=rail
        )
        self._registrations: Dict[str, RegisteredMemory] = {}
        self._handles: Dict[str, TransferHandle] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_existing(
        cls, engine: MooncakeTransferEngine, *, rail: str
    ) -> "MooncakePVDTransferEngine":
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
            raise RuntimeError(f"engine rail is {self.rail}, registration requested {rail}")
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
        return registration

    def release_memory(self, registration: RegisteredMemory) -> None:
        with self._lock:
            existing = self._registrations.pop(
                registration.descriptor.region_id, None
            )
        if existing is not None:
            ret = self._engine.engine.unregister_memory(
                existing.descriptor.address
                - int(existing.descriptor.backend_metadata.get("base_offset", 0))
            )
            if ret != 0:
                raise RuntimeError(f"Mooncake memory deregistration failed: {ret}")

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

        local_address = local.registration.descriptor.address + local.offset
        remote_address = remote.address + remote_offset
        ret = self._engine.transfer_sync(
            remote.endpoint, local_address, remote_address, local.length
        )
        if ret < 0:
            handle.status = TransferStatus.FAILED
            handle.error = f"Mooncake transfer_sync returned {ret}"
        else:
            handle.status = TransferStatus.SUCCESS
            handle.transferred_bytes = local.length
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
        }
