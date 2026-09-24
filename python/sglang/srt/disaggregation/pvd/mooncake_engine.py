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
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferBudget,
    TransferLifecycleManager,
    TransportState,
)
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)

logger = logging.getLogger(__name__)
_manager_lock = threading.Lock()
_uncertain_lock = threading.Lock()
# Native registration may have taken effect even when its Python call fails.
# Keep both the engine and CUDA storage alive until process exit in that case.
_uncertain_native_registrations: list[tuple[Any, torch.Tensor, int, str]] = []


class MooncakePVDTransferEngine(TransferEngine):
    """Track native asynchronous writes until their source memory is safe."""

    name = "mooncake"

    def __init__(
        self, *, hostname: str, gpu_id: int, rail: str,
        budget: Optional[TransferBudget] = None,
    ) -> None:
        self.rail = rail
        self._engine = MooncakeTransferEngine(
            hostname=hostname,
            gpu_id=gpu_id,
            ib_device=rail,
            require_fresh_metadata=True,
        )
        self._initialize(budget)

    @classmethod
    def from_existing(
        cls, engine: MooncakeTransferEngine, *, rail: str,
        budget: Optional[TransferBudget] = None,
    ) -> "MooncakePVDTransferEngine":
        engine.require_pvd_metadata_policy()
        adapter = cls.__new__(cls)
        adapter.rail = rail
        adapter._engine = engine
        adapter._initialize(budget)
        return adapter

    def _initialize(self, budget: Optional[TransferBudget]) -> None:
        for name in ("transfer_submit_write", "transfer_check_status"):
            if not callable(getattr(self._engine.engine, name, None)):
                raise RuntimeError(f"Mooncake PVD requires native async API {name}")
        with _manager_lock:
            manager = getattr(self._engine, "_pvd_lifecycle_manager", None)
            if manager is None:
                if budget is None:
                    raise ValueError("PVD requires an explicit transfer budget")
                manager = TransferLifecycleManager(budget)
                self._engine._pvd_lifecycle_manager = manager
            elif budget is not None and manager.budget is not budget:
                raise ValueError("shared Mooncake engine already has a different budget")
            self.lifecycle_manager = manager
        self._registrations: Dict[str, RegisteredMemory] = {}
        self._guards: Dict[str, ResourceGuard] = {}
        self._lock = threading.Lock()
        self._registration_unknown_reason: Optional[str] = None

    def _quarantine_registration(
        self, buffer: torch.Tensor, ptr: int, reason: str
    ) -> None:
        # Called with self._lock held. Never recycle possibly registered storage.
        with _uncertain_lock:
            _uncertain_native_registrations.append((self._engine, buffer, ptr, reason))
            self._registration_unknown_reason = reason
            self._engine._pvd_registration_unknown_reason = reason

    def _clear_quarantined_registration(self, buffer: torch.Tensor, ptr: int) -> None:
        # A later successful native unregister is the only recovery proof.
        with _uncertain_lock:
            _uncertain_native_registrations[:] = [
                record
                for record in _uncertain_native_registrations
                if not (
                    record[0] is self._engine
                    and record[1] is buffer
                    and record[2] == ptr
                )
            ]
            remaining = next(
                (
                    record[3]
                    for record in _uncertain_native_registrations
                    if record[0] is self._engine
                ),
                None,
            )
            self._registration_unknown_reason = remaining
            self._engine._pvd_registration_unknown_reason = remaining

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
        with self._lock:
            transport_state = self.lifecycle_manager.snapshot()
            if transport_state["quarantined"]:
                raise RuntimeError(
                    "PVD native transport is quarantined: "
                    f"{transport_state['quarantine_reason']}"
                )
            unknown_reason = getattr(
                self._engine, "_pvd_registration_unknown_reason", None
            )
            if unknown_reason is not None:
                raise RuntimeError(
                    "Mooncake registration state is unknown: "
                    f"{unknown_reason}"
                )
            try:
                ret = self._engine.engine.register_memory(ptr, length)
            except Exception as exc:
                self._quarantine_registration(
                    buffer, ptr, f"native register raised: {exc}"
                )
                raise RuntimeError("Mooncake native registration outcome is unknown") from exc
            if ret != 0:
                self._quarantine_registration(
                    buffer, ptr, f"native register returned {ret}"
                )
                raise RuntimeError(
                    f"Mooncake GPU memory registration failed on {rail} with code {ret}; "
                    "registration outcome is unknown"
                )
            try:
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
                guard = ResourceGuard(
                    registration, lambda: self._unregister(registration)
                )
            except Exception as exc:
                try:
                    rollback_result = self._engine.engine.unregister_memory(ptr)
                    if rollback_result != 0:
                        raise RuntimeError(
                            f"native rollback returned {rollback_result}"
                        )
                except Exception as rollback_exc:
                    self._quarantine_registration(
                        buffer,
                        ptr,
                        f"descriptor construction failed ({exc}); "
                        f"native rollback failed ({rollback_exc})",
                    )
                    raise RuntimeError(
                        "Mooncake registration rollback failed; CUDA storage retained"
                    ) from exc
                raise
            try:
                self._registrations[region_id] = registration
                self._guards[region_id] = guard
            except Exception as exc:
                self._registrations.pop(region_id, None)
                self._guards.pop(region_id, None)
                try:
                    rollback_result = self._engine.engine.unregister_memory(ptr)
                    if rollback_result != 0:
                        raise RuntimeError(
                            f"native rollback returned {rollback_result}"
                        )
                except Exception as rollback_exc:
                    self._quarantine_registration(
                        buffer,
                        ptr,
                        f"registration publication failed ({exc}); "
                        f"native rollback failed ({rollback_exc})",
                    )
                    raise RuntimeError(
                        "Mooncake registration rollback failed; CUDA storage retained"
                    ) from exc
                raise
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
            existing = self._registrations.get(registration.descriptor.region_id)
            guard = self._guards.get(registration.descriptor.region_id)
            if existing is not None and existing is not registration:
                raise ValueError("registration identity does not belong to this adapter")
        if guard is not None:
            guard.request_release()

    def _unregister(self, existing: RegisteredMemory) -> None:
        # Keep the registration and tensor until native deregistration succeeds.
        # ResourceGuard serializes callbacks and rejects all new transfer pins.
        if existing is not None:
            ptr = existing.descriptor.address - int(
                existing.descriptor.backend_metadata.get("base_offset", 0)
            )
            try:
                ret = self._engine.engine.unregister_memory(ptr)
                if ret != 0:
                    raise RuntimeError(f"native unregister returned {ret}")
            except Exception as exc:
                with self._lock:
                    self._quarantine_registration(
                        existing.buffer,
                        ptr,
                        f"native unregister outcome is unknown: {exc}",
                    )
                raise RuntimeError("Mooncake memory deregistration failed") from exc
            with self._lock:
                self._clear_quarantined_registration(existing.buffer, ptr)
                self._registrations.pop(existing.descriptor.region_id, None)
                self._guards.pop(existing.descriptor.region_id, None)
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

        try:
            with self._lock:
                unknown_reason = getattr(
                    self._engine, "_pvd_registration_unknown_reason", None
                )
                if unknown_reason is not None:
                    raise RuntimeError(
                        "Mooncake registration state is unknown: "
                        f"{unknown_reason}"
                    )
                registration = self._registrations.get(local.registration.descriptor.region_id)
                guard = self._guards.get(local.registration.descriptor.region_id)
            if registration is not local.registration or guard is None:
                raise ValueError("source registration does not belong to this adapter")
            if registration.descriptor.rail != self.rail:
                raise ValueError("source registration rail mismatch")
            if local.offset < 0 or local.length <= 0 or local.offset + local.length > registration.descriptor.length:
                raise ValueError("PUT exceeds bounded source region")
            self._engine.require_pvd_metadata_policy()
            # GPUDirect reads are not ordered behind PyTorch packing kernels.
            torch.cuda.synchronize(local.registration.buffer.device)
            self.lifecycle_manager.attach(handle, guard, local.length)
        except Exception as exc:
            handle.status = TransferStatus.FAILED
            handle.error = f"PVD source preparation failed: {exc}"
            return handle

        local_address = local.registration.descriptor.address + local.offset
        remote_address = remote.address + remote_offset
        # region_id is a PVD identity, NOT a Mooncake rkey. Fresh descriptor
        # resolution happens inside classic Mooncake on every remote lookup,
        # enabled before its first native import. No 'seen region' shortcut.
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
        self.lifecycle_manager.submit_native(
            handle,
            lambda: self._engine.engine.transfer_submit_write(
                remote.endpoint, local_address, remote_address, local.length
            ),
        )
        return handle

    def poll(self, handle: TransferHandle) -> TransferStatus:
        terminal_success = None
        with handle._lock:
            if handle.transport_state in (TransportState.IN_FLIGHT, TransportState.DRAINING):
                try:
                    result = self._engine.engine.transfer_check_status(handle.backend_handle)
                except Exception as exc:
                    self.lifecycle_manager.mark_unknown(handle, f"native poll raised: {exc}")
                else:
                    if result in (1, -1):
                        terminal_success = result == 1
                        handle.transport_state = (
                            TransportState.TERMINAL_SUCCESS
                            if terminal_success
                            else TransportState.TERMINAL_FAILED
                        )
                    elif result == -2:
                        handle.transport_state = TransportState.DRAINING
                    elif result != 0:
                        self.lifecycle_manager.mark_unknown(handle, f"unexpected native status {result}")
            elif handle.transport_state in (TransportState.TERMINAL_SUCCESS, TransportState.TERMINAL_FAILED):
                terminal_success = handle.transport_state == TransportState.TERMINAL_SUCCESS
        if terminal_success is not None:
            self._complete(handle, terminal_success)
        return handle.status

    def _complete(self, handle: TransferHandle, success: bool) -> None:
        try:
            self.lifecycle_manager.complete(handle, success)
        except Exception as exc:
            # Native terminal state is definitive even if local cleanup fails.
            handle.error = f"PVD source release failed: {exc}"
            # Do not retain the exception object in a LogRecord: its traceback
            # can retain the registration and CUDA tensor after a later retry.
            logger.warning("PVD source retained after cleanup failure: %s", str(exc))

    def abort(self, handle: TransferHandle) -> None:
        self.lifecycle_manager.request_cancel(handle)

    def cleanup_complete(self, handle: TransferHandle) -> bool:
        return (
            handle.transport_state.is_locally_safe_to_release
            and self.lifecycle_manager.cleanup_complete(handle)
        )

    def health(self) -> Dict[str, Any]:
        lifecycle = self.lifecycle_manager.snapshot()
        with self._lock:
            registrations = len(self._registrations)
            unknown_reason = getattr(
                self._engine, "_pvd_registration_unknown_reason", None
            )
        try:
            session_id = self._engine.get_session_id()
        except Exception as exc:
            session_id = None
            unknown_reason = unknown_reason or f"Mooncake session ID unavailable: {exc}"
        return {
            "backend": self.name,
            "healthy": unknown_reason is None and not lifecycle["quarantined"],
            "registration_unknown_reason": unknown_reason,
            "rail": self.rail,
            "session_id": session_id,
            "registered_regions": registrations,
            "metadata_policy": "fresh",
            "metadata_policy_verification": "version-pinned-pre-init",
            "mooncake_version": self._engine.pvd_metadata_version,
            "lifecycle": lifecycle,
        }
