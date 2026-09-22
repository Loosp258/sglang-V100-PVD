"""TP1 CUDA request: prediction/search -> HTTP Delivery -> runtime install.

Initial complete Prompt must already be installed independently of the index.
This explicit component does not register itself with the serving Scheduler.
"""

import torch
from sglang.srt.disaggregation.pvd.cpu_prefetch_request import _PrefetchRequestCore
from sglang.srt.disaggregation.pvd.cuda_probe_search import (
    CUDAPredictionPipeline,
    CUDAProbeSearchSession,
)
from sglang.srt.disaggregation.pvd.cuda_runtime_group import CUDARuntimeInstallGroup
from sglang.srt.disaggregation.pvd.cuda_sparse_delivery import CUDASparseDelivery
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError


class CUDAPrefetchRequest(_PrefetchRequestCore):
    def __init__(self, group, pipeline, *, copy_budget, max_head_dim, **kwargs):
        if not isinstance(pipeline, CUDAPredictionPipeline):
            raise TypeError("explicit CUDA prediction pipeline required")
        self._query_device = torch.device(pipeline.probe.device)
        self._copy_budget, self._max_head_dim = copy_budget, max_head_dim
        super().__init__(group, pipeline, **kwargs)

    def _validate_group(self, group):
        if not isinstance(group, CUDARuntimeInstallGroup):
            raise TypeError("an owned CUDA runtime install group is required")
        if any(
            torch.device(meta["device"]) != self._query_device
            for meta in group.describe_banks().values()
        ):
            raise ValueError("probe and CUDA working-set devices differ")

    def _validate_delivery(self, delivery, group):
        if not isinstance(delivery, CUDASparseDelivery) or delivery.group is not group:
            raise ValueError("owned CUDA Delivery sink for this exact group required")

    def _create_session(self, *args, **kwargs):
        return CUDAProbeSearchSession(
            *args,
            **kwargs,
            device=self._query_device,
            copy_budget=self._copy_budget,
            max_head_dim=self._max_head_dim,
        )

    def _live(self):
        # Check the absolute deadline on entry and after HTTP returns. A stopped
        # runtime never makes an old logical selection installable again.
        self.group.progress()
        super()._live()

    def _require_query_drained(self):
        if (
            self._session._copy_unknown
            or self.pipeline._quarantined
            or self.pipeline.probe._quarantined
            or self.pipeline.provider.degraded
        ):
            raise InstallProtocolError("CUDA query ownership remains quarantined")
