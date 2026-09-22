"""Selected V routes assemble one owned CUDA-policy request, never Scheduler auto-mode."""

import asyncio
from dataclasses import replace

import pytest
import torch
from sglang.srt.disaggregation.pvd.client import (
    PVDSelectedShardRoute,
    PVDSelectedShardRoutes,
)
from sglang.srt.disaggregation.pvd.coordinator import CoordinatorError
from sglang.srt.disaggregation.pvd.cuda_routed_request import (
    assemble_routed_cuda_request,
)
from sglang.srt.disaggregation.pvd.cuda_sparse_fanin import CUDASparseFanInStage
from sglang.srt.disaggregation.pvd.multi_rail_receive import RailMappedReceiveEngine
from sglang.srt.disaggregation.pvd.prediction import (
    ProbeConfig,
    QueryVectors,
    snapshot_committed,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine
from test_pvd_cuda_probe_search import bridge
from test_pvd_cuda_sparse_fanin import two_source
from test_pvd_prompt_index import SPACE


class FakeSessionEngine(FakeTransferEngine):
    def __init__(self, rail, session_id):
        super().__init__()
        self.rail, self.session_id = rail, session_id

    def health(self):
        return {**super().health(), "session_id": self.session_id}


def test_discovered_shards_assemble_exact_request_and_own_clients(monkeypatch):
    async def run():
        async with two_source(
            monkeypatch,
            prepare_records=False,
            begin_refresh=False,
            single_rail=True,
        ) as c:
            selected = PVDSelectedShardRoutes(
                c.manifest,
                tuple(
                    PVDSelectedShardRoute(
                        rank,
                        str(c.servers[rank].make_url("")),
                        c.stores[rank].worker_epoch,
                        c.stores[rank].rail,
                    )
                    for rank in (0, 1)
                ),
            )
            _, _, _, pipeline, _, copy_budget, _ = bridge(monkeypatch)
            pipeline.probe.head_count = 8
            pipeline.probe.config = pipeline.probe_config = ProbeConfig(
                SPACE, tuple(range(c.compute.num_layers)), head_count=8
            )
            original_describe = c.group.describe_banks
            monkeypatch.setattr(
                c.group,
                "describe_banks",
                lambda: {
                    rank: {**meta, "device": "cuda:0"}
                    for rank, meta in original_describe().items()
                },
            )
            kwargs = {
                "compute_layout": c.compute,
                "compute_rank": 0,
                "group": c.group,
                "registry": c.registry,
                "pipeline": pipeline,
                "head_mapping": QueryHeadMapping(8, 4),
                "vector_space": SPACE,
                "metric": "l2",
                "top_k": 1,
                "max_union_tokens": 2,
                "max_head_dim": 8,
                "copy_budget": copy_budget,
                "aggregate_budget": c.aggregate_budget,
                "d_endpoint": "D",
                "d_rail": "mlx5_0",
                "poll_interval_seconds": 0.001,
            }
            with pytest.raises(ValueError, match="exact selected Entry"):
                assemble_routed_cuda_request(
                    PVDSelectedShardRoutes(
                        c.manifest,
                        (selected.shards[0], selected.shards[0]),
                    ),
                    **kwargs,
                )
            with pytest.raises(ValueError, match="exact selected Entry"):
                assemble_routed_cuda_request(
                    selected, **{**kwargs, "head_mapping": QueryHeadMapping(16, 8)}
                )
            mixed = PVDSelectedShardRoutes(
                replace(
                    c.manifest,
                    shards=[
                        c.manifest.shards[0],
                        replace(c.manifest.shards[1], rail="mlx5_1"),
                    ],
                ),
                (
                    selected.shards[0],
                    replace(selected.shards[1], rail="mlx5_1"),
                ),
            )
            with pytest.raises(ValueError, match="single-rail"):
                assemble_routed_cuda_request(mixed, **kwargs)
            with pytest.raises(ValueError, match="single-rail"):
                assemble_routed_cuda_request(
                    mixed,
                    **{
                        **kwargs,
                        "d_rails": {0: "mlx5_0", 1: "mlx5_1"},
                    },
                )
            with pytest.raises(ValueError, match="single-rail"):
                assemble_routed_cuda_request(
                    selected,
                    **{
                        **kwargs,
                        "d_rail": "mlx5_1",
                        "d_rails": {0: "mlx5_0", 1: "mlx5_0"},
                    },
                )
            assert not c.registry.snapshot()
            assembly = assemble_routed_cuda_request(selected, **kwargs)
            controller = assembly.controller
            assert assembly.clients[0] is controller.delivery.routing
            assert len(controller._routes[0]) == c.compute.num_layers * 8
            assert set(controller.delivery._routes) == {0, 1}
            assert {route.rail for route in controller.delivery._routes.values()} == {
                "mlx5_0"
            }
            assert all(
                route.sender_epoch == selected.shards[rank].sender_epoch
                for rank, route in controller.delivery._routes.items()
            )
            with pytest.raises(InstallProtocolError, match="aclose"):
                controller.close()
            await controller.aclose()
            assert all(client._closed for client in controller._owned_clients)
            with pytest.raises(CoordinatorError, match="closed"):
                await controller.delivery._routes[0].client.health()
            assert not c.registry.snapshot()

    asyncio.run(run())


@pytest.mark.parametrize("single_rail", (True, False))
def test_factory_controller_runs_two_v_search_delivery_and_install(
    monkeypatch, single_rail
):
    async def run():
        async with two_source(
            monkeypatch,
            prepare_records=False,
            begin_refresh=False,
            single_rail=single_rail,
        ) as c:
            if not single_rail:
                c.registry.engine = RailMappedReceiveEngine(
                    {
                        rail: FakeSessionEngine(rail, f"D{rank}")
                        for rank, rail in enumerate(("mlx5_0", "mlx5_1"))
                    }
                )
            monkeypatch.setattr(CUDASparseFanInStage, "_synchronize", lambda self: None)
            monkeypatch.setattr(
                CUDASparseFanInStage,
                "_allocate",
                lambda self, size: torch.empty(size, dtype=torch.uint8),
            )
            _, _, _, pipeline, _, copy_budget, _ = bridge(monkeypatch)
            pipeline.probe.head_count = 8
            pipeline.probe.config = pipeline.probe_config = ProbeConfig(
                SPACE, tuple(range(c.compute.num_layers)), head_count=8
            )

            def capture(prefix, positions):
                vector = c.pool.k_buffer[0][3, 0].to(torch.float32)
                return tuple(
                    QueryVectors(
                        vector_space=SPACE,
                        version="target-q",
                        layer=layer,
                        head_start=0,
                        head_count=8,
                        positions=tuple(positions),
                        valid_length=len(positions),
                        vectors=vector.repeat(len(positions), 8, 1),
                        prefix_version=prefix.version,
                        positional_encoding="rope_applied",
                        request_id=prefix.request_id,
                    )
                    for layer in pipeline.probe_config.layers
                )

            pipeline.probe.capture = lambda prefix, prediction: capture(
                prefix,
                range(len(prefix.tokens), len(prefix.tokens) + len(prediction.tokens)),
            )
            pipeline.probe.capture_committed = capture
            original_describe = c.group.describe_banks
            monkeypatch.setattr(
                c.group,
                "describe_banks",
                lambda: {
                    rank: {**meta, "device": "cuda:0"}
                    for rank, meta in original_describe().items()
                },
            )
            selected = PVDSelectedShardRoutes(
                c.manifest,
                tuple(
                    PVDSelectedShardRoute(
                        rank,
                        str(c.servers[rank].make_url("")),
                        c.stores[rank].worker_epoch,
                        c.stores[rank].rail,
                    )
                    for rank in (0, 1)
                ),
            )
            kwargs = {
                "compute_layout": c.compute,
                "compute_rank": 0,
                "group": c.group,
                "registry": c.registry,
                "pipeline": pipeline,
                "head_mapping": QueryHeadMapping(8, 4),
                "vector_space": SPACE,
                "metric": "l2",
                "top_k": 1,
                "max_union_tokens": 2,
                "max_head_dim": 8,
                "copy_budget": copy_budget,
                "aggregate_budget": c.aggregate_budget,
                "d_endpoint": "D",
                "d_rail": "mlx5_0",
                "d_rails": (
                    None
                    if single_rail
                    else {rank: c.stores[rank].rail for rank in (0, 1)}
                ),
                "poll_interval_seconds": 0.001,
            }
            if not single_rail:
                with pytest.raises(ValueError, match="native rail session"):
                    assemble_routed_cuda_request(
                        selected,
                        **{
                            **kwargs,
                            "d_endpoints": {0: "D0", 1: "wrong-session"},
                        },
                    )
            assembly = assemble_routed_cuda_request(selected, **kwargs)
            controller = assembly.controller
            if not single_rail:
                assert {
                    rank: route.endpoint
                    for rank, route in controller.delivery._routes.items()
                } == {0: "D0", 1: "D1"}
            monkeypatch.setattr(
                controller._session, "_query_device", lambda t: t.device.type == "cpu"
            )
            sink = controller.delivery

            async def finish_remote():
                while not sink._rounds:
                    await asyncio.sleep(0.001)
                records = next(iter(sink._rounds.values()))
                for rank in (0, 1):
                    while rank not in records:
                        await asyncio.sleep(0.001)
                    record = records[rank]
                    while (
                        record.identity.transfer_id
                        not in c.stores[rank].entries[c.manifest.key].deliveries
                    ):
                        await asyncio.sleep(0.001)
                    delivery = (
                        c.stores[rank]
                        .entries[c.manifest.key]
                        .deliveries[record.identity.transfer_id]
                    )
                    while delivery.transfer_handle is None:
                        await asyncio.sleep(0.001)
                    c.engine.finish(delivery.transfer_handle)
                    c.stores[rank].progress_transfers()

            task = asyncio.create_task(finish_remote())
            try:
                epoch = await asyncio.wait_for(
                    controller.refresh(
                        snapshot_committed("request", [1] * 12, 3, "prefix3"),
                        query_positions=(12,),
                        clients=assembly.clients,
                    ),
                    5,
                )
                await task
                assert epoch.target_tokens == 4
                assert controller.try_install({0: 4})
                from test_pvd_cpu_sparse_delivery import wait_acks

                await wait_acks(sink)
                assert controller.can_decode(4)
                with c.bank.read() as groups:
                    assert set(groups) == set(sink.routing.groups)
                assert c.registry.budget.snapshot()["used_staging_bytes"] == 0
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await controller.aclose()
            assert all(client._closed for client in controller._owned_clients)

    asyncio.run(run())
