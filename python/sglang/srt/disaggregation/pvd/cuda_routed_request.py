"""Assemble one selected V group into a bounded TP1 CUDA refresh request.

This is a request factory, not a Scheduler startup switch. The caller supplies
an installed or explicitly pending D bank, the concrete draft/target pipeline
and Gateway-selected V routes. Client sessions outlive every
remote destination and are closed only after the controller drains.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from sglang.srt.disaggregation.pvd.client import (
    PVDSelectedShardRoute,
    PVDSelectedShardRoutes,
)
from sglang.srt.disaggregation.pvd.control_server import HttpShardClient
from sglang.srt.disaggregation.pvd.cuda_prefetch_request import CUDAPrefetchRequest
from sglang.srt.disaggregation.pvd.cuda_probe_search import CUDAPredictionPipeline
from sglang.srt.disaggregation.pvd.cuda_runtime_group import CUDARuntimeInstallGroup
from sglang.srt.disaggregation.pvd.cuda_sparse_delivery import (
    CUDAReceiveRoute,
    CUDASparseFanInDelivery,
)
from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import CUDASparseReceiveRegistry
from sglang.srt.disaggregation.pvd.multi_rail_receive import RailMappedReceiveEngine
from sglang.srt.disaggregation.pvd.probe_search import ProbeSearchRoute
from sglang.srt.disaggregation.pvd.prompt_index import SearchRequestIdentity
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED, QueryHeadMapping
from sglang.srt.disaggregation.pvd.protocol import KVEntryManifest, KVLayoutSignature
from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient
from sglang.srt.disaggregation.pvd.search_routing import RoutedShardSearchClient
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

_PARTIAL_ASSEMBLY_QUARANTINE = []
_UNSTARTED_ASSEMBLY_QUARANTINE = []


class CUDARoutedPrefetchRequest(CUDAPrefetchRequest):
    """The request owns both HTTP client sets after native owners have drained."""

    def __init__(self, *args, owned_clients, **kwargs):
        super().__init__(*args, **kwargs)
        self._owned_clients = tuple(owned_clients)

    def close(self):
        raise InstallProtocolError(
            "routed HTTP owners require aclose() after remote destinations drain"
        )

    async def aclose(self):
        await super().aclose()
        await self.delivery.routing.close()
        for client in self._owned_clients:
            await client.close()


@dataclass(frozen=True)
class CUDARoutedRequestAssembly:
    controller: CUDARoutedPrefetchRequest
    clients: object  # {D compute rank: routed V search client}, for the driver

    async def discard_unstarted(self):
        """Retire an assembly that the refresh driver never claimed.

        A registered request must instead use the driver's ordered retirement;
        in particular this path must never close a published destination.
        """
        controller = self.controller
        delivery = controller.delivery
        state = delivery.snapshot()
        coordinator = controller.group.coordinator.snapshot()
        if (
            getattr(controller, "_refresh_driver_claimed", False)
            or controller._closed
            or controller._active is not None
            or controller._ready is not None
            or controller._tasks
            or controller._session._pending is not None
            or controller._session._prepared is not None
            or controller._session._ready is not None
            or coordinator["state"] != "idle"
            or state["pending_rounds"]
            or state["retained_destinations"]
            or state["ack_tasks"]
            or any(client._session is not None for client in controller._owned_clients)
        ):
            raise InstallProtocolError(
                "only an unclaimed, idle CUDA assembly can be discarded"
            )
        try:
            await controller.aclose()
        except BaseException:
            # A failed close cannot prove that readers, native registrations or
            # HTTP owners were retired. Retain every owner for explicit recovery.
            _UNSTARTED_ASSEMBLY_QUARANTINE.append(self)
            raise


def assemble_routed_cuda_request(
    selected: PVDSelectedShardRoutes,
    *,
    request_id: str,
    compute_layout: KVLayoutSignature,
    compute_rank: int,
    group,
    registry,
    pipeline: CUDAPredictionPipeline,
    head_mapping: QueryHeadMapping,
    vector_space: str,
    metric: str,
    top_k: int,
    max_union_tokens: int,
    max_head_dim: int,
    copy_budget: TransferBudget,
    aggregate_budget: TransferBudget,
    d_endpoint: str,
    d_rail: str,
    poll_interval_seconds: float,
    initial_import_pending: bool = False,
    d_rails: Mapping[int, str] | None = None,
    d_endpoints: Mapping[int, str] | None = None,
) -> CUDARoutedRequestAssembly:
    """Fail closed on mismatched Entry/layout/selected shard metadata.

    The returned controller may be registered with CUDARefreshDriver. That
    driver must call controller.aclose() on retirement; no URL is closed while
    a write or its fence may still need the control client.
    """
    if (
        not isinstance(selected, PVDSelectedShardRoutes)
        or not isinstance(selected.manifest, KVEntryManifest)
        or not isinstance(selected.shards, tuple)
        or any(
            not isinstance(route, PVDSelectedShardRoute) for route in selected.shards
        )
        or not isinstance(compute_layout, KVLayoutSignature)
        or not isinstance(request_id, str)
        or not request_id.strip()
        or type(compute_rank) is not int
        or not isinstance(group, CUDARuntimeInstallGroup)
        or not isinstance(registry, CUDASparseReceiveRegistry)
        or compute_rank not in group._banks
        or not isinstance(pipeline, CUDAPredictionPipeline)
        or not isinstance(head_mapping, QueryHeadMapping)
        or not isinstance(copy_budget, TransferBudget)
        or not isinstance(aggregate_budget, TransferBudget)
        or not isinstance(vector_space, str)
        or not vector_space.strip()
        or not isinstance(d_endpoint, str)
        or not d_endpoint.strip()
        or not isinstance(d_rail, str)
        or not d_rail.strip()
        or type(top_k) is not int
        or not 1 <= top_k <= min(512, selected.manifest.prompt_token_count)
        or type(max_union_tokens) is not int
        or not 0 < max_union_tokens <= group._banks[compute_rank].max_union_tokens
        or head_mapping.total_kv_heads != compute_layout.total_kv_heads
        or pipeline.probe_config.target_model_id != vector_space
        or pipeline.probe_config.head_start != 0
        or pipeline.probe_config.head_count != head_mapping.num_query_heads
        or pipeline.probe_config.layers != tuple(range(compute_layout.num_layers))
        or group.coordinator.identity[0] != request_id
        or group.coordinator.identity[2] != selected.manifest.key.transfer_id
        or group._banks[compute_rank].identity[:3] != group.coordinator.identity
        or group._banks[compute_rank].identity[3] != compute_layout.fingerprint
        or len(selected.shards) != 2
        or tuple(route.rank for route in selected.shards) != (0, 1)
        or any(
            route.rail != selected.manifest.shard(route.rank).rail
            for route in selected.shards
        )
    ):
        raise ValueError("exact selected Entry, model and D routing required")
    receive_rails = (
        {route.rank: d_rail for route in selected.shards}
        if d_rails is None
        else dict(d_rails)
    )
    if d_rails is None and any(route.rail != d_rail for route in selected.shards):
        raise ValueError(
            "single-rail D engine cannot receive from V shards on different rails"
        )
    if (
        set(receive_rails) != {route.rank for route in selected.shards}
        or any(
            type(rank) is not int or not isinstance(rail, str) or not rail.strip()
            for rank, rail in receive_rails.items()
        )
        or any(route.rail != receive_rails[route.rank] for route in selected.shards)
    ):
        raise ValueError("each V source needs its matching explicit D rail")
    if isinstance(registry.engine, RailMappedReceiveEngine):
        if any(
            not registry.engine.supports_rail(rail) for rail in receive_rails.values()
        ):
            raise ValueError("D receive engine lacks a selected rail adapter")
        native_sessions = {
            rank: registry.engine.adapters[rail].health().get("session_id")
            for rank, rail in receive_rails.items()
        }
        if d_endpoints is None and all(native_sessions.values()):
            receive_endpoints = native_sessions
        elif d_endpoints is not None:
            receive_endpoints = dict(d_endpoints)
        else:
            raise ValueError("explicit D endpoint per V source required")
        if any(
            native_sessions[rank] is not None
            and receive_endpoints.get(rank) != native_sessions[rank]
            for rank in receive_rails
        ):
            raise ValueError("D endpoint differs from its native rail session")
    elif (
        set(receive_rails.values()) != {d_rail}
        or getattr(registry.engine, "rail", d_rail) != d_rail
    ):
        raise ValueError(
            "single-rail D engine cannot receive from V shards on different rails"
        )
    else:
        receive_endpoints = (
            {route.rank: d_endpoint for route in selected.shards}
            if d_endpoints is None
            else dict(d_endpoints)
        )
        if set(receive_endpoints.values()) != {d_endpoint}:
            raise ValueError("single-rail D engine requires one exact D endpoint")
    if set(receive_endpoints) != set(receive_rails) or any(
        not isinstance(endpoint, str) or not endpoint.strip()
        for endpoint in receive_endpoints.values()
    ):
        raise ValueError("one explicit D endpoint per selected V source required")

    search_clients, control_clients = {}, {}
    routing = delivery = None
    try:
        for route in selected.shards:
            search_clients[route.rank] = PVDShardSearchClient(route.url)
            control_clients[route.rank] = HttpShardClient(route.rank, route.url)
        routing = RoutedShardSearchClient(
            storage_layout=selected.manifest.layout,
            compute_layout=compute_layout,
            compute_rank=compute_rank,
            entry_transfer_id=selected.manifest.key.transfer_id,
            prompt_tokens=selected.manifest.prompt_token_count,
            vector_space=vector_space,
            metric=metric,
            clients=search_clients,
            layers=tuple(range(compute_layout.num_layers)),
        )
        routes = {
            route.rank: CUDAReceiveRoute(
                control_clients[route.rank],
                route.sender_epoch,
                receive_endpoints[route.rank],
                receive_rails[route.rank],
            )
            for route in selected.shards
            if route.rank in routing.clients
        }
        delivery = CUDASparseFanInDelivery(
            group,
            registry,
            routing,
            key=selected.manifest.key,
            routes=routes,
            aggregate_budget=aggregate_budget,
            poll_interval_seconds=poll_interval_seconds,
        )
        rank_routes = tuple(
            ProbeSearchRoute(
                query_head,
                SearchRequestIdentity(
                    vector_space,
                    ROPE_APPLIED,
                    selected.manifest.key.transfer_id,
                    layer,
                    kv_head,
                ),
                routing.scope,
                top_k,
            )
            for layer, kv_head in sorted(routing.groups)
            for query_head in head_mapping.query_heads_for(kv_head)
        )
        controller = CUDARoutedPrefetchRequest(
            group,
            pipeline,
            copy_budget=copy_budget,
            max_head_dim=max_head_dim,
            head_mapping=head_mapping,
            rank_routes={compute_rank: rank_routes},
            max_union_tokens=max_union_tokens,
            initial_import_pending=initial_import_pending,
            delivery=delivery,
            owned_clients=tuple(search_clients.values())
            + tuple(control_clients.values()),
        )
        return CUDARoutedRequestAssembly(
            controller, MappingProxyType({compute_rank: routing})
        )
    except BaseException:
        # Nothing in this constructor publishes a destination or starts an
        # HTTP request. All clients are therefore lazy and have no session.
        # The caller still owns the bank/group and closes it separately.
        clients = tuple(search_clients.values()) + tuple(control_clients.values())
        try:
            if delivery is not None:
                state = delivery.snapshot()
                if state["pending_rounds"] or state["retained_destinations"]:
                    raise RuntimeError("partial CUDA assembly has live destinations")
                delivery.cancel()
            if routing is not None:
                routing._closed = True
            if any(client._session is not None for client in clients):
                raise RuntimeError("partial CUDA assembly unexpectedly opened HTTP")
            for client in clients:
                client._closed = True
        except BaseException:
            _PARTIAL_ASSEMBLY_QUARANTINE.append(
                (group, registry, routing, delivery, clients)
            )
            raise
        raise
