"""HTTP sparse Delivery sink bound to an owned TP1 CUDA install runtime.

The transport may be Mooncake; real HTTP with fake bytes is not RDMA evidence.
CPU policy checks remain separate rather than treating a GPU record as CPU.
"""

import asyncio
import math
import uuid

from sglang.srt.disaggregation.pvd.cpu_sparse_delivery import (
    CPUReceiveRoute,
    _SparseDeliveryCore,
)
from sglang.srt.disaggregation.pvd.cuda_runtime_group import CUDARuntimeInstallGroup
from sglang.srt.disaggregation.pvd.cuda_sparse_fanin import CUDASparseFanInStage
from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import CUDASparseReceiveRegistry
from sglang.srt.disaggregation.pvd.search_routing import RoutedShardSearchClient
from sglang.srt.disaggregation.pvd.sparse_receiver import SparseReceiveError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


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


class CUDASparseFanInDelivery(_SparseDeliveryCore):
    """One D bank, all selected V sources, one stage and post-install ACKs."""

    _namespace = "cuda-sparse-fanin-delivery"

    def __init__(
        self,
        group,
        registry,
        routing,
        *,
        key,
        routes,
        aggregate_budget,
        poll_interval_seconds,
    ):
        if (
            not isinstance(group, CUDARuntimeInstallGroup)
            or not isinstance(registry, CUDASparseReceiveRegistry)
            or not isinstance(routing, RoutedShardSearchClient)
            or not isinstance(aggregate_budget, TransferBudget)
            or key.transfer_id != routing.entry_transfer_id
            or group.coordinator.identity[2] != key.transfer_id
            or set(routes) != set(routing.clients)
            or any(
                type(rank) is not int
                or not isinstance(route, CUDAReceiveRoute)
                or route.client.rank != rank
                or route.client.base_url != routing._endpoints[rank]
                for rank, route in routes.items()
            )
            or type(poll_interval_seconds) not in (int, float)
            or not math.isfinite(poll_interval_seconds)
            or poll_interval_seconds <= 0
        ):
            raise SparseReceiveError(
                "exact V routes, D bank and aggregate budget required"
            )
        # This constructor also checks bank/layout/group/placement before any
        # native registration can be attempted by stage().
        CUDASparseFanInStage(group, registry, routing, aggregate_budget)
        registry._owner()
        self.group, self.registry, self.routing = group, registry, routing
        self.key, self.aggregate_budget = key, aggregate_budget
        self._routes, self._metadata = dict(routes), group.describe_banks()
        self._interval = float(poll_interval_seconds)
        self._rounds, self._stages, self._tasks, self._errors = {}, {}, {}, {}
        self._closed = False
        self._scope = self._namespace + ":" + uuid.uuid4().hex

    def _live(self):
        super()._live()
        self.group.progress()
        self.group.coordinator._live()

    async def _wait_source(self, record):
        ready = await record.start()
        while not ready:
            self._live()
            await asyncio.sleep(self._interval)
            ready = await record.poll()

    async def stage(self, epoch, rank, specs):
        self._live()
        self.group._check_stage(epoch, rank)
        if rank != self.routing.compute_rank or epoch in self._rounds:
            raise SparseReceiveError("foreign or replayed sparse fan-in epoch")
        plans = self.routing.partition_specs(tuple(specs))
        first = plans[0].decode_specs[0]
        if (
            first.request_id,
            first.incarnation,
            first.entry_transfer_id,
            first.operation_id,
            first.target_tokens,
        ) != (
            epoch.request_id,
            epoch.incarnation,
            epoch.entry_transfer_id,
            epoch.operation_id,
            epoch.target_tokens,
        ):
            raise SparseReceiveError("source selection differs from install epoch")
        records = self._rounds[epoch] = {}
        for plan in plans:
            source = plan.storage_rank
            route = self._routes[source]
            record = self.registry.prepare(
                plan.manifest,
                key=self.key,
                rank=source,
                rail=route.rail,
                endpoint=route.endpoint,
                sender_epoch=route.sender_epoch,
                client=route.client,
                owner_scope=self._scope,
            )
            records[source] = record
        stage = self._stages[epoch] = CUDASparseFanInStage(
            self.group, self.registry, self.routing, self.aggregate_budget
        )
        await asyncio.gather(
            *(self._wait_source(record) for record in records.values())
        )
        self._live()
        self.group.coordinator._match(epoch)
        return stage.stage(epoch, plans, records)

    async def _ack_round(self, epoch):
        errors = await super()._ack_round(epoch)
        if not errors:
            self._stages.pop(epoch, None)
        return errors

    async def close(self):
        errors = await super().close()
        for epoch in tuple(self._stages):
            if epoch not in self._rounds:
                self._stages.pop(epoch, None)
        return errors
