"""Role-neutral memory registration and one-sided PUT abstraction for PVD."""

from __future__ import annotations

import abc
import dataclasses
import enum
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import torch

from sglang.srt.disaggregation.pvd.protocol import RemoteRegionDescriptor


class TransferStatus(str, enum.Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RegisteredMemory:
    descriptor: RemoteRegionDescriptor
    buffer: torch.Tensor = field(repr=False)


@dataclass(frozen=True)
class MemorySlice:
    registration: RegisteredMemory
    offset: int
    length: int

    def __post_init__(self) -> None:
        if self.offset < 0 or self.length <= 0:
            raise ValueError("memory slice offset/length is invalid")
        if self.offset + self.length > self.registration.descriptor.length:
            raise ValueError("memory slice exceeds registered region")


@dataclass
class TransferHandle:
    transfer_id: str
    status: TransferStatus = TransferStatus.PENDING
    transferred_bytes: int = 0
    error: Optional[str] = None
    backend_handle: Any = None


class TransferEngine(abc.ABC):
    """Minimal interface shared by P, V and D.

    The existing PD sender/receiver classes remain adapters above their current
    engines. PVD uses this interface directly so the V role can receive and send
    without inheriting Prefill/Decode semantics.
    """

    name: str

    @abc.abstractmethod
    def register_memory(
        self,
        buffer: torch.Tensor,
        *,
        endpoint: str,
        rank: int,
        rail: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> RegisteredMemory: ...

    @abc.abstractmethod
    def release_memory(self, registration: RegisteredMemory) -> None: ...

    @abc.abstractmethod
    def submit_put(
        self,
        local: MemorySlice,
        remote: RemoteRegionDescriptor,
        *,
        remote_offset: int = 0,
    ) -> TransferHandle: ...

    @abc.abstractmethod
    def poll(self, handle: TransferHandle) -> TransferStatus: ...

    @abc.abstractmethod
    def abort(self, handle: TransferHandle) -> None: ...

    def health(self) -> Dict[str, Any]:
        return {"backend": self.name, "healthy": True}


class FakeTransferEngine(TransferEngine):
    """Process-local deterministic transport used by unit and state tests."""

    name = "fake"
    _regions: Dict[str, RegisteredMemory] = {}
    _regions_lock = threading.Lock()

    def __init__(self) -> None:
        self._handles: Dict[str, TransferHandle] = {}
        self._lock = threading.Lock()
        self.total_put_bytes = 0

    @staticmethod
    def _byte_view(buffer: torch.Tensor) -> torch.Tensor:
        if not buffer.is_contiguous():
            raise ValueError("registered buffers must be contiguous")
        return buffer.view(torch.uint8).reshape(-1)

    def register_memory(
        self,
        buffer: torch.Tensor,
        *,
        endpoint: str,
        rank: int,
        rail: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> RegisteredMemory:
        byte_view = self._byte_view(buffer)
        region_id = uuid.uuid4().hex
        descriptor = RemoteRegionDescriptor(
            endpoint=endpoint,
            region_id=region_id,
            address=int(buffer.data_ptr()),
            length=byte_view.numel(),
            device=str(buffer.device),
            rank=rank,
            rail=rail,
            backend_metadata=dict(metadata or {}),
        )
        registration = RegisteredMemory(descriptor=descriptor, buffer=buffer)
        with self._regions_lock:
            self._regions[region_id] = registration
        return registration

    def release_memory(self, registration: RegisteredMemory) -> None:
        with self._regions_lock:
            self._regions.pop(registration.descriptor.region_id, None)

    def submit_put(
        self,
        local: MemorySlice,
        remote: RemoteRegionDescriptor,
        *,
        remote_offset: int = 0,
    ) -> TransferHandle:
        handle = TransferHandle(transfer_id=uuid.uuid4().hex)
        with self._lock:
            self._handles[handle.transfer_id] = handle

        try:
            with self._regions_lock:
                remote_registration = self._regions.get(remote.region_id)
            if remote_registration is None:
                raise RuntimeError(f"remote region {remote.region_id} is not registered")
            if remote_offset < 0 or remote_offset + local.length > remote.length:
                raise ValueError("PUT exceeds remote region")

            absolute_remote_offset = int(
                remote.backend_metadata.get("base_offset", 0)
            ) + remote_offset
            source = self._byte_view(local.registration.buffer)[
                local.offset : local.offset + local.length
            ]
            destination = self._byte_view(remote_registration.buffer)[
                absolute_remote_offset : absolute_remote_offset + local.length
            ]
            destination.copy_(source, non_blocking=False)
            handle.transferred_bytes = local.length
            handle.status = TransferStatus.SUCCESS
            self.total_put_bytes += local.length
        except Exception as exc:
            handle.status = TransferStatus.FAILED
            handle.error = str(exc)
        return handle

    def poll(self, handle: TransferHandle) -> TransferStatus:
        return handle.status

    def abort(self, handle: TransferHandle) -> None:
        if handle.status == TransferStatus.PENDING:
            handle.status = TransferStatus.CANCELLED

    def health(self) -> Dict[str, Any]:
        with self._regions_lock:
            region_count = len(self._regions)
        return {
            "backend": self.name,
            "healthy": True,
            "registered_regions": region_count,
            "total_put_bytes": self.total_put_bytes,
        }


def descriptor_with_slice(
    registration: RegisteredMemory, *, offset: int, length: int
) -> RemoteRegionDescriptor:
    """Create a bounded descriptor without exposing a larger pool registration."""
    base = registration.descriptor
    if offset < 0 or length <= 0 or offset + length > base.length:
        raise ValueError("descriptor slice exceeds registered region")
    metadata = dict(base.backend_metadata)
    metadata["base_region_id"] = base.region_id
    metadata["base_offset"] = offset
    return dataclasses.replace(
        base,
        address=base.address + offset,
        length=length,
        backend_metadata=metadata,
    )
