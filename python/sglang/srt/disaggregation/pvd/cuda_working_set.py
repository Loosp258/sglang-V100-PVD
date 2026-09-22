"""Opt-in CUDA Prompt banks with blocking, fail-closed completion ownership.

Not a production attention backend, RDMA visibility fence or TP installer. The
caller must prove receive completion/visibility BEFORE stage(). Generated KV is
never owned here. CUDA code is present but requires separate hardware acceptance.
"""

import threading
import uuid

import torch
from sglang.srt.disaggregation.pvd.sparse_payload import (
    SparseKVPayload,
    SparsePayloadError,
)
from sglang.srt.disaggregation.pvd.sparse_working_set import _SparseWorkingSetCore
from sglang.srt.disaggregation.pvd.transfer_engine import RegisteredMemory
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferBudget,
)


class CUDASparseWorkingSet(_SparseWorkingSetCore):
    """Single-owner CUDA bank, NOT a subclass accepted by CPUInstallGroup.

    stage() requires a source guard covering all payload bytes. Read scopes must
    encompass all enqueueing of consuming kernels. Both scope exit and staging
    drain the concrete device. Unknown completion is terminal quarantine: there
    is intentionally no force-free or automatic retry that invents a fence.
    """

    _budget_namespace = "pvd-cuda-working-set"

    def __init__(self, *, device, dtype, budget, **kwargs):
        selected = torch.device(device)
        if selected.type != "cuda" or selected.index is None:
            raise SparsePayloadError("an explicit indexed CUDA device is required")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise SparsePayloadError("CUDA bank requires an explicit floating KV dtype")
        if not isinstance(budget, TransferBudget):
            raise SparsePayloadError("an explicit bank budget is required")
        if not torch.cuda.is_available():
            raise SparsePayloadError("CUDA bank requested but CUDA is unavailable")
        self.device, self.dtype = selected, dtype
        self._thread = threading.get_ident()
        self._quarantine = None
        self._retained_stage = None
        self._retained_source_guard = None
        super().__init__(budget=budget, **kwargs)

    def _check_policy(self):
        if threading.get_ident() != self._thread:
            raise SparsePayloadError("CUDA bank must run on its owner thread")
        if self._quarantine is not None:
            raise SparsePayloadError(f"CUDA bank quarantined: {self._quarantine}")

    def _check_tensor(self, tensor):
        if (
            tensor.device != self.device
            or tensor.dtype != self.dtype
            or not tensor.is_contiguous()
        ):
            raise SparsePayloadError(
                "CUDA bank payload device/dtype/contiguity mismatch"
            )

    def _synchronize(self):
        torch.cuda.synchronize(self.device)

    def _drain_stage(self, source, copies, owner):
        try:
            self._synchronize()
        except BaseException:
            self._quarantine = "staging completion unknown"
            self._retained_stage = (source, copies, owner)
            raise

    def _drain_reader(self):
        try:
            self._synchronize()
        except BaseException:
            self._quarantine = "reader completion unknown"
            # The base read scope deliberately does not decrement its reader.
            raise

    def stage(self, payloads, *, source_guard):
        self._open()
        if not isinstance(source_guard, ResourceGuard):
            raise SparsePayloadError("a pinned source ResourceGuard is required")
        owner = f"cuda-bank-source:{uuid.uuid4().hex}"
        source_guard.pin(owner)
        try:
            backing = source_guard.value
            if isinstance(backing, RegisteredMemory):
                backing = backing.buffer
            if (
                not isinstance(backing, torch.Tensor)
                or backing.device != self.device
                or not backing.is_contiguous()
            ):
                raise SparsePayloadError(
                    "source guard must own contiguous tensor storage"
                )
            payloads = tuple(payloads)
            begin = backing.data_ptr()
            end = begin + backing.numel() * backing.element_size()
            storage = backing.untyped_storage().data_ptr()
            for payload in payloads:
                if not isinstance(payload, SparseKVPayload):
                    raise SparsePayloadError("verified scoped sparse payload required")
                tensor = payload.tensor
                self._check_tensor(tensor)
                if (
                    tensor.untyped_storage().data_ptr() != storage
                    or tensor.data_ptr() < begin
                    or tensor.data_ptr() + payload.nbytes > end
                ):
                    raise SparsePayloadError(
                        "source guard does not cover payload storage"
                    )
            super().stage(payloads)
        finally:
            if self._quarantine is None:
                try:
                    source_guard.unpin(owner)
                except BaseException:
                    self._quarantine = "source release outcome unknown"
                    self._retained_source_guard = (source_guard, owner)
                    raise
            else:
                self._retained_source_guard = (source_guard, owner)

    def snapshot(self):
        if threading.get_ident() != self._thread:
            raise SparsePayloadError("CUDA bank must run on its owner thread")
        return {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "closed": self._closed,
            "quarantine": self._quarantine,
            "readers": self._readers,
            "current_boundary": None
            if self._current is None
            else self._current.boundary,
            "next_boundary": None if self._next is None else self._next.boundary,
            "retained_stage": self._retained_stage is not None,
            "source_guard_held": self._retained_source_guard is not None,
            "completion_policy": "device_synchronize",
        }
