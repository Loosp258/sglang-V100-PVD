"""HTTP sparse Delivery sink bound to an owned TP1 CUDA install runtime.

The transport may be Mooncake; real HTTP with fake bytes is not RDMA evidence.
CPU policy checks remain separate rather than treating a GPU record as CPU.
"""

from sglang.srt.disaggregation.pvd.cpu_sparse_delivery import (
    CPUReceiveRoute,
    _SparseDeliveryCore,
)
from sglang.srt.disaggregation.pvd.cuda_runtime_group import CUDARuntimeInstallGroup
from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import CUDASparseReceiveRegistry
from sglang.srt.disaggregation.pvd.sparse_receiver import SparseReceiveError


class CUDAReceiveRoute(CPUReceiveRoute):
    """Chosen V identity plus concrete D endpoint/rail (no guessed HCA names)."""


class CUDASparseDelivery(_SparseDeliveryCore):
    _route_type = CUDAReceiveRoute
    _namespace = "cuda-sparse-delivery"

    def _live(self):
        super()._live()
        # A waiting Delivery may be the only active owner turn. Enforce the
        # runtime's absolute deadline while polling; never turn that timeout
        # into a native WRITE completion or destination-release permission.
        self.group.progress()
        self.group.coordinator._live()

    def _validate_binding(self, group, registry):
        if not isinstance(group, CUDARuntimeInstallGroup) or not isinstance(
            registry, CUDASparseReceiveRegistry
        ):
            raise SparseReceiveError(
                "explicit CUDA runtime group and receive registry required"
            )
        if any(
            item["device"] != str(registry.device)
            for item in group.describe_banks().values()
        ):
            raise SparseReceiveError("CUDA bank and destination device differ")

    def _manifest_dtype(self, rank):
        return self._metadata[rank]["dtype"]

    def _stage_record(self, record, epoch):
        return self.group.stage_received(record, epoch)
