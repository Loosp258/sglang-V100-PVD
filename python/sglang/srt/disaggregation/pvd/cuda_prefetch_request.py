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
from sglang.srt.disaggregation.pvd.cuda_sparse_delivery import (
    CUDASparseDelivery,
    CUDASparseFanInDelivery,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError


class CUDAPrefetchRequest(_PrefetchRequestCore):
    def __init__(self, group, pipeline, *, copy_budget, max_head_dim, **kwargs):
        if not isinstance(pipeline, CUDAPredictionPipeline):
            raise TypeError("explicit CUDA prediction pipeline required")
        self._query_device = torch.device(pipeline.probe.device)
        self._copy_budget, self._max_head_dim = copy_budget, max_head_dim
        super().__init__(group, pipeline, **kwargs)
        if isinstance(self.delivery, CUDASparseFanInDelivery):
            self._validate_fanin_routes()

    def _validate_fanin_routes(self):
        routing = self.delivery.routing
        if (
            set(self._routes) != {routing.compute_rank}
            or self.mapping.total_kv_heads != routing.total_kv_heads
            or self._metadata[routing.compute_rank]["groups"] != set(routing.groups)
        ):
            raise ValueError("fan-in search routes differ from the selected D bank")
        routes = self._routes[routing.compute_rank]
        expected = {
            (layer, query_head)
            for layer, kv_head in routing.groups
            for query_head in self.mapping.query_heads_for(kv_head)
        }
        observed = {(route.identity.layer, route.query_head) for route in routes}
        if (
            len(routes) != len(expected)
            or observed != expected
            or any(
                route.identity.kv_head != self.mapping.kv_head_for(route.query_head)
                or route.identity.entry_transfer_id != routing.entry_transfer_id
                or route.identity.vector_space != routing.vector_space
                or route.identity.positional_encoding != ROPE_APPLIED
                or route.scope != routing.scope
                for route in routes
            )
        ):
            raise ValueError("fan-in must search every Q head on its selected V route")

    def _validate_search_clients(self, clients):
        if isinstance(self.delivery, CUDASparseFanInDelivery):
            routing = self.delivery.routing
            if clients.get(routing.compute_rank) is not routing:
                raise ValueError(
                    "fan-in search must use the Delivery's selected V route"
                )

    def _validate_group(self, group):
        if not isinstance(group, CUDARuntimeInstallGroup):
            raise TypeError("an owned CUDA runtime install group is required")
        if any(
            torch.device(meta["device"]) != self._query_device
            for meta in group.describe_banks().values()
        ):
            raise ValueError("probe and CUDA working-set devices differ")

    def _validate_delivery(self, delivery, group):
        if (
            not isinstance(delivery, (CUDASparseDelivery, CUDASparseFanInDelivery))
            or delivery.group is not group
        ):
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
