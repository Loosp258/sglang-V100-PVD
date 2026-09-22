"""Conservative Linux GPUDirect receive ordering, not a remote WRITE fence.

See NVIDIA GPUDirect RDMA 11.4, Synchronization and Memory Ordering: enable
SYNC_MEMOPS per allocation, wait for the NIC WRITE to complete with respect to
the CPU, THEN issue the consuming CUDA operation/synchronization from the CPU.
Never run a concurrent kernel against an RDMA-written destination.
"""

import ctypes
import sys

import torch
from sglang.srt.disaggregation.pvd.sparse_receiver import SparseReceiveError

_SYNC_MEMOPS = 6  # CU_POINTER_ATTRIBUTE_SYNC_MEMOPS (CUDA driver ABI)


class _PointerAttributes:
    def __init__(self):
        if sys.platform != "linux" or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise SparseReceiveError("GPUDirect receiver requires 64-bit Linux CUDA")
        self.library = ctypes.CDLL("libcuda.so.1")
        for name in ("cuPointerSetAttribute", "cuPointerGetAttribute"):
            function = getattr(self.library, name)
            function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
            function.restype = ctypes.c_int

    def enable_sync_memops(self, address):
        if type(address) is not int or not 0 < address < 2**64:
            raise SparseReceiveError("invalid CUDA receive pointer")
        flag = ctypes.c_uint(1)
        result = self.library.cuPointerSetAttribute(
            ctypes.byref(flag), _SYNC_MEMOPS, address
        )
        if result != 0:
            raise SparseReceiveError(
                f"cuPointerSetAttribute(SYNC_MEMOPS) failed: {result}"
            )
        flag.value = 0
        result = self.library.cuPointerGetAttribute(
            ctypes.byref(flag), _SYNC_MEMOPS, address
        )
        if result != 0 or flag.value != 1:
            raise SparseReceiveError(
                f"CUDA receive SYNC_MEMOPS verification failed: {result}, {flag.value}"
            )


class CUDAReceiveOrdering:
    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise SparseReceiveError("explicit indexed CUDA receive device required")
        self._attributes = None

    def prepare(self, buffer):
        if (
            buffer.device != self.device
            or not buffer.is_contiguous()
            or buffer.numel() == 0
        ):
            raise SparseReceiveError("invalid CUDA receive allocation")
        with torch.cuda.device(self.device):
            if self._attributes is None:
                self._attributes = _PointerAttributes()
            self._attributes.enable_sync_memops(int(buffer.data_ptr()))

    def after_remote_write(self, registration):
        """Only call AFTER exact successful remote terminal proof on this CPU.

        SYNC_MEMOPS was established before registration/publication. This is
        the CPU-initiated local ordering half, not permission to ignore pending
        NIC writes. No reader/kernel has seen the private receive allocation.
        """
        buffer = registration.buffer
        if buffer.device != self.device:
            raise SparseReceiveError("receive ordering device mismatch")
        torch.cuda.synchronize(self.device)
