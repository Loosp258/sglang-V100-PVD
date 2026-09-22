"""Two real V stores/HTTP deliveries and one D bank; CPU CUDA-policy doubles."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from aiohttp.test_utils import TestServer
from sglang.srt.disaggregation.pvd.control_server import (
    HttpShardClient,
    create_shard_app,
)
from sglang.srt.disaggregation.pvd.cuda_prefetch_request import CUDAPrefetchRequest
from sglang.srt.disaggregation.pvd.cuda_runtime_group import CUDARuntimeInstallGroup
from sglang.srt.disaggregation.pvd.cuda_sparse_delivery import (
    CUDAReceiveRoute,
    CUDASparseFanInDelivery,
)
from sglang.srt.disaggregation.pvd.cuda_sparse_fanin import CUDASparseFanInStage
from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import CUDASparseReceiveRegistry
from sglang.srt.disaggregation.pvd.cuda_working_set import CUDASparseWorkingSet
from sglang.srt.disaggregation.pvd.prediction import (
    ProbeConfig,
    QueryVectors,
    snapshot_committed,
)
from sglang.srt.disaggregation.pvd.probe_search import ProbeSearchRoute
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient
from sglang.srt.disaggregation.pvd.search_routing import RoutedShardSearchClient
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
from sglang.srt.disaggregation.pvd.sparse_receiver import SparseReceiveError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferBudget,
    TransferCapacityError,
)
from sglang.srt.disaggregation.pvd.vector_store import VectorKVStore
from test_pvd_cuda_probe_search import bridge
from test_pvd_prompt_index import SPACE, build_entry, ident, manager
from test_pvd_prompt_vectors import pack_shard
from test_pvd_search_routing import layout
from test_pvd_vector_lifecycle import DelayedTransferEngine


@asynccontextmanager
async def two_source(monkeypatch, *, prepare_records=True, begin_refresh=True):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    pool, storage, manifest, _, _ = build_entry()
    compute = layout(storage, 1)
    engine = DelayedTransferEngine()
    engine.lifecycle_manager = SimpleNamespace(budget=TransferBudget(1 << 20, 32))
    stores, indexes, events = {}, {}, []
    async with AsyncExitStack() as stack:
        servers, clients = {}, {}
        for rank in (0, 1):
            index = indexes[rank] = manager(budget=TransferBudget(1 << 20, 32))
            shard = manifest.shards[rank]
            store = stores[rank] = VectorKVStore(
                rank=rank,
                world_size=2,
                rail=shard.rail,
                device="cpu",
                total_pages=16,
                page_bytes=shard.expected_bytes // shard.page_count,
                endpoint=f"v{rank}",
                transfer_engine=engine,
                allow_cpu_for_tests=True,
                prompt_index=index,
            )
            entry = store.create_entry(manifest)
            store.begin_p_write(manifest.key)
            packed, _, _ = pack_shard(pool, storage, rank=rank, prompt_tokens=8)
            offset = entry.allocation.start_page * store.page_bytes
            store.pool[offset : offset + shard.expected_bytes] = packed.tensor
            store.commit_p_write(manifest.key, shard.expected_bytes)
            assert store.progress_prompt_indexes()["built"] == 1
            server = servers[rank] = await stack.enter_async_context(
                TestServer(create_shard_app(store))
            )
            client = clients[rank] = HttpShardClient(rank, str(server.make_url("")))
            stack.push_async_callback(client.close)
        search_clients = {
            rank: PVDShardSearchClient(str(server.make_url("")))
            for rank, server in servers.items()
        }
        for client in search_clients.values():
            stack.push_async_callback(client.close)
        routing = RoutedShardSearchClient(
            storage_layout=storage,
            compute_layout=compute,
            compute_rank=0,
            entry_transfer_id=manifest.key.transfer_id,
            prompt_tokens=8,
            vector_space=SPACE,
            metric="l2",
            clients=search_clients,
        )
        bank_budget = TransferBudget(1 << 20, 4)
        bank = CUDASparseWorkingSet(
            device="cuda:0",
            dtype=torch.float16,
            budget=bank_budget,
            request_id="request",
            incarnation="inc",
            entry_transfer_id=manifest.key.transfer_id,
            layout_fingerprint=compute.fingerprint,
            expected_groups=tuple(routing.groups),
            prompt_tokens=8,
            head_dim=compute.head_dim,
            max_union_tokens=8,
        )
        bank.device = torch.device("cpu")
        monkeypatch.setattr(bank, "_synchronize", lambda: events.append("bank_sync"))
        group = CUDARuntimeInstallGroup(
            {0: bank},
            interval=4,
            lead_tokens=1,
            peer_epochs={0: "D-rank0"},
            timeout_seconds=10,
            max_pending_events=8,
            max_pending_bytes=65536,
        )
        initial = group.begin(0)
        full_specs = tuple(
            SparseKVSpec(
                request_id="request",
                incarnation="inc",
                operation_id=initial.operation_id,
                target_tokens=0,
                entry_transfer_id=manifest.key.transfer_id,
                index_version="initial",
                id_mapping_version="initial",
                layout_fingerprint=compute.fingerprint,
                layer=layer,
                kv_head=head,
                token_ids=tuple(range(8)),
            )
            for layer, head in routing.groups
        )
        full_manifest = SparseDeliveryManifest(
            full_specs, "torch.float16", compute.head_dim
        )
        backing = torch.cat(
            [
                torch.stack(
                    (
                        pool.k_buffer[s.layer][:8, s.kv_head],
                        pool.v_buffer[s.layer][:8, s.kv_head],
                    )
                )
                .reshape(-1)
                .view(torch.uint8)
                for s in full_specs
            ]
        )
        guard = ResourceGuard(backing, lambda: None)
        group.stage(
            initial, 0, full_manifest.payload_views(backing), source_guard=guard
        )
        guard.request_release()
        assert group.try_install(initial, {0: 0})
        epoch = group.begin(3) if begin_refresh else None
        specs = (
            tuple(
                replace(
                    spec,
                    operation_id=epoch.operation_id,
                    target_tokens=4,
                    token_ids=(1, 3),
                    index_version=(
                        indexes[routing.groups[spec.layer, spec.kv_head]]
                        .gate_for(manifest.key.transfer_id)
                        .descriptor.index_version
                    ),
                    id_mapping_version=(
                        indexes[routing.groups[spec.layer, spec.kv_head]]
                        .gate_for(manifest.key.transfer_id)
                        .descriptor.id_mapping_version
                    ),
                )
                for spec in full_specs
            )
            if epoch is not None
            else ()
        )
        plans = routing.partition_specs(specs) if specs else ()
        registry = CUDASparseReceiveRegistry(
            engine, TransferBudget(1 << 20, 8), receiver_epoch="D", device="cuda:0"
        )
        registry.device = torch.device("cpu")
        registry.ordering = SimpleNamespace(
            prepare=lambda _: events.append("register_order"),
            after_remote_write=lambda _: events.append("remote_visible"),
        )
        records = (
            {
                plan.storage_rank: registry.prepare(
                    plan.manifest,
                    key=manifest.key,
                    rank=plan.storage_rank,
                    rail=stores[plan.storage_rank].rail,
                    endpoint="D",
                    sender_epoch=stores[plan.storage_rank].worker_epoch,
                    client=clients[plan.storage_rank],
                )
                for plan in plans
            }
            if prepare_records
            else {}
        )
        aggregate_budget = TransferBudget(1 << 20, 2)
        stage = CUDASparseFanInStage(group, registry, routing, aggregate_budget)
        stage._allocate = lambda size: torch.empty(size, dtype=torch.uint8)
        stage._synchronize = lambda: events.append("aggregate_sync")
        case = SimpleNamespace(**locals())
        try:
            yield case
        finally:
            for store in stores.values():
                for delivery in store.entries[manifest.key].deliveries.values():
                    handle = delivery.transfer_handle
                    if handle and not handle.transport_state.is_locally_safe_to_release:
                        engine.finish(handle)
                store.progress_transfers()
            await registry.close()
            group.close()
            for store in stores.values():
                store.close()


def finish_one(case, rank):
    record = case.records[rank]
    delivery = (
        case.stores[rank]
        .entries[case.manifest.key]
        .deliveries[record.identity.transfer_id]
    )
    case.engine.finish(delivery.transfer_handle)
    case.stores[rank].progress_transfers()


def test_two_source_completion_stage_install_then_two_acks(monkeypatch):
    async def run():
        async with two_source(monkeypatch) as c:
            assert all(
                not await_record
                for await_record in (
                    await c.records[0].start(),
                    await c.records[1].start(),
                )
            )
            with pytest.raises(SparseReceiveError, match="terminal proof"):
                c.stage.stage(c.epoch, c.plans, c.records)
            assert c.aggregate_budget.snapshot()["used_staging_bytes"] == 0
            finish_one(c, 0)
            assert await c.records[0].poll()
            with pytest.raises(SparseReceiveError, match="terminal proof"):
                c.stage.stage(c.epoch, c.plans, c.records)
            finish_one(c, 1)
            assert await c.records[1].poll()
            receipt = c.stage.stage(c.epoch, c.plans, c.records)
            assert receipt.rank == 0 and receipt.epoch == c.epoch
            assert c.events.count("remote_visible") == 2
            assert c.aggregate_budget.snapshot()["used_staging_bytes"] == 0
            assert not c.stage.snapshot()["installed"]
            with pytest.raises(SparseReceiveError, match="all ranks"):
                await c.records[0].ack()
            assert c.group.try_install(c.epoch, {0: 4})
            assert c.stage.snapshot()["installed"]
            with c.bank.read() as groups:
                for (layer, head), (spec, tensor) in groups.items():
                    assert spec.token_ids == (1, 3)
                    torch.testing.assert_close(
                        tensor[0], c.pool.k_buffer[layer][[1, 3], head]
                    )
                    torch.testing.assert_close(
                        tensor[1], c.pool.v_buffer[layer][[1, 3], head]
                    )
            for record in c.records.values():
                await record.ack()
                assert await record.close()
            assert not c.registry.snapshot()
            assert c.registry.budget.snapshot()["used_staging_bytes"] == 0

    asyncio.run(run())


@pytest.mark.parametrize("fault", ["manifest", "endpoint", "unsafe", "wrong_bank"])
def test_malformed_source_plan_never_allocates_or_stages(monkeypatch, fault):
    async def run():
        async with two_source(monkeypatch) as c:
            for rank in (0, 1):
                await c.records[rank].start()
                finish_one(c, rank)
                assert await c.records[rank].poll()
            if fault == "manifest":
                plans = (
                    replace(c.plans[0], decode_specs=c.plans[1].decode_specs),
                    c.plans[1],
                )
            else:
                plans = c.plans
            if fault == "endpoint":
                c.records[1]._client.base_url = "http://wrong-v"
            elif fault == "unsafe":
                c.records[1]._safe = False
            elif fault == "wrong_bank":
                c.records[1]._buffer = torch.zeros_like(c.records[1]._buffer)[:1]
            with pytest.raises((SparseReceiveError, ValueError)):
                c.stage.stage(c.epoch, plans, c.records)
            assert c.aggregate_budget.snapshot()["used_staging_bytes"] == 0
            assert c.bank._next is None

    asyncio.run(run())


def test_unknown_local_completion_keeps_every_source_and_aggregate(monkeypatch):
    async def run():
        async with two_source(monkeypatch) as c:
            for rank in (0, 1):
                await c.records[rank].start()
                finish_one(c, rank)
                assert await c.records[rank].poll()

            def fail_sync():
                raise RuntimeError("device completion unknown")

            c.stage._synchronize = fail_sync
            with pytest.raises(RuntimeError, match="device completion unknown"):
                c.stage.stage(c.epoch, c.plans, c.records)
            assert c.stage.snapshot()["unknown"]
            assert c.aggregate_budget.snapshot()["used_staging_bytes"] > 0
            assert all(r._local_unknown is not None for r in c.records.values())
            assert all(r._source_guard.value is not None for r in c.records.values())
            assert all(
                r.identity.region_id not in c.engine.released
                for r in c.records.values()
            )
            assert c.bank._next is None

    asyncio.run(run())


def test_capacity_refusal_does_not_claim_sources_or_submit_copy(monkeypatch):
    async def run():
        async with two_source(monkeypatch) as c:
            for rank in (0, 1):
                await c.records[rank].start()
                finish_one(c, rank)
                assert await c.records[rank].poll()
            c.stage.budget = TransferBudget(1, 1)
            with pytest.raises(TransferCapacityError):
                c.stage.stage(c.epoch, c.plans, c.records)
            assert not c.stage._used
            assert all(
                getattr(r, "_fanin_stage", None) is None for r in c.records.values()
            )
            assert c.events.count("remote_visible") == 0
            c.stage.budget = c.aggregate_budget
            assert c.stage.stage(c.epoch, c.plans, c.records).epoch == c.epoch

    asyncio.run(run())


def test_request_delivery_drives_two_sources_then_acks_after_install(monkeypatch):
    async def run():
        async with two_source(monkeypatch, prepare_records=False) as c:
            monkeypatch.setattr(
                CUDASparseFanInStage,
                "_synchronize",
                lambda self: c.events.append("aggregate_sync"),
            )
            monkeypatch.setattr(
                CUDASparseFanInStage,
                "_allocate",
                lambda self, size: torch.empty(size, dtype=torch.uint8),
            )
            routes = {
                rank: CUDAReceiveRoute(
                    c.clients[rank],
                    c.stores[rank].worker_epoch,
                    "D",
                    c.stores[rank].rail,
                )
                for rank in (0, 1)
            }
            sink = CUDASparseFanInDelivery(
                c.group,
                c.registry,
                c.routing,
                key=c.manifest.key,
                routes=routes,
                aggregate_budget=c.aggregate_budget,
                poll_interval_seconds=0.001,
            )
            stage = next(iter(sink._stages.values()), None)
            assert stage is None

            async def finish_remote():
                while len(sink._rounds.get(c.epoch, {})) != 2:
                    await asyncio.sleep(0.001)
                for rank in (0, 1):
                    while not c.stores[rank].entries[c.manifest.key].deliveries:
                        await asyncio.sleep(0.001)
                    record = sink._rounds[c.epoch][rank]
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
                receipt = await asyncio.wait_for(
                    sink.stage(c.epoch, 0, c.specs), timeout=5
                )
                await task
                assert receipt.epoch == c.epoch
                assert set(sink._rounds[c.epoch]) == {0, 1}
                sink.require_installable(c.epoch)
                with pytest.raises(SparseReceiveError, match="all ranks"):
                    sink._rounds[c.epoch][0].confirm_install()
                assert c.group.try_install(c.epoch, {0: 4})
                sink.installed(c.epoch)
                from test_pvd_cpu_sparse_delivery import wait_acks

                await wait_acks(sink)
                assert sink.snapshot()["retained_destinations"] == 0
                assert c.aggregate_budget.snapshot()["used_staging_bytes"] == 0
                assert c.registry.budget.snapshot()["used_staging_bytes"] == 0
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                assert await sink.close() == {}

    asyncio.run(run())


def test_cancelled_fanin_keeps_both_live_remote_destinations(monkeypatch):
    async def run():
        async with two_source(monkeypatch, prepare_records=False) as c:
            routes = {
                rank: CUDAReceiveRoute(
                    c.clients[rank],
                    c.stores[rank].worker_epoch,
                    "D",
                    c.stores[rank].rail,
                )
                for rank in (0, 1)
            }
            sink = CUDASparseFanInDelivery(
                c.group,
                c.registry,
                c.routing,
                key=c.manifest.key,
                routes=routes,
                aggregate_budget=c.aggregate_budget,
                poll_interval_seconds=0.001,
            )
            task = asyncio.create_task(sink.stage(c.epoch, 0, c.specs))
            for _ in range(1000):
                if len(sink._rounds.get(c.epoch, {})) == 2 and all(
                    c.stores[rank]
                    .entries[c.manifest.key]
                    .deliveries.get(sink._rounds[c.epoch][rank].identity.transfer_id)
                    is not None
                    and c.stores[rank]
                    .entries[c.manifest.key]
                    .deliveries[sink._rounds[c.epoch][rank].identity.transfer_id]
                    .transfer_handle
                    is not None
                    for rank in (0, 1)
                ):
                    break
                await asyncio.sleep(0.001)
            else:
                pytest.fail("both source writes did not start")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            errors = await sink.close()
            assert len(errors) == 2
            assert len(c.registry.snapshot()) == 2
            assert c.registry.budget.snapshot()["used_staging_bytes"] > 0
            assert not c.engine.released
            for store in c.stores.values():
                delivery = next(iter(store.entries[c.manifest.key].deliveries.values()))
                c.engine.finish(delivery.transfer_handle)
                store.progress_transfers()
            assert await sink.close() == {}
            assert not c.registry.snapshot()
            assert c.registry.budget.snapshot()["used_staging_bytes"] == 0

    asyncio.run(run())


def test_request_controller_predicts_searches_both_v_and_installs(monkeypatch):
    async def run():
        async with two_source(
            monkeypatch, prepare_records=False, begin_refresh=False
        ) as c:
            monkeypatch.setattr(CUDASparseFanInStage, "_synchronize", lambda self: None)
            monkeypatch.setattr(
                CUDASparseFanInStage,
                "_allocate",
                lambda self, size: torch.empty(size, dtype=torch.uint8),
            )
            route_config = ProbeConfig(
                SPACE, tuple(range(c.storage.num_layers)), head_count=8
            )
            _, _, _, pipeline, _, copy_budget, _ = bridge(monkeypatch)
            pipeline.probe.head_count = 8
            pipeline.probe.config = pipeline.probe_config = route_config

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
                    for layer in route_config.layers
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
            routes = {
                rank: CUDAReceiveRoute(
                    c.clients[rank],
                    c.stores[rank].worker_epoch,
                    "D",
                    c.stores[rank].rail,
                )
                for rank in (0, 1)
            }
            sink = CUDASparseFanInDelivery(
                c.group,
                c.registry,
                c.routing,
                key=c.manifest.key,
                routes=routes,
                aggregate_budget=c.aggregate_budget,
                poll_interval_seconds=0.001,
            )
            route_list = tuple(
                ProbeSearchRoute(
                    qhead,
                    ident(c.manifest.key.transfer_id, layer, qhead // 2),
                    c.routing.scope,
                    1,
                )
                for layer in route_config.layers
                for qhead in range(8)
            )
            request = CUDAPrefetchRequest(
                c.group,
                pipeline,
                copy_budget=copy_budget,
                max_head_dim=8,
                head_mapping=QueryHeadMapping(8, 4),
                rank_routes={0: route_list},
                max_union_tokens=2,
                delivery=sink,
            )
            monkeypatch.setattr(
                request._session, "_query_device", lambda t: t.device.type == "cpu"
            )
            prefix = snapshot_committed("request", [1] * 12, 3, "prefix3")

            async def finish_remote():
                while not sink._rounds:
                    await asyncio.sleep(0.001)
                epoch = next(iter(sink._rounds))
                for rank in (0, 1):
                    while rank not in sink._rounds[epoch]:
                        await asyncio.sleep(0.001)
                    record = sink._rounds[epoch][rank]
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
                    request.refresh(
                        prefix,
                        query_positions=(len(prefix.tokens),),
                        clients={0: c.routing},
                    ),
                    5,
                )
                await task
                assert epoch.target_tokens == 4
                assert request.pending_install_boundary == 4
                assert not request.can_decode(4)
                assert request.try_install({0: 4})
                from test_pvd_cpu_sparse_delivery import wait_acks

                await wait_acks(sink)
                assert request.can_decode(4)
                with c.bank.read() as groups:
                    assert set(groups) == set(c.routing.groups)
                    assert all(len(spec.token_ids) <= 2 for spec, _ in groups.values())
                assert c.aggregate_budget.snapshot()["used_staging_bytes"] == 0
                assert c.registry.budget.snapshot()["used_staging_bytes"] == 0
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await request.aclose()
                await c.routing.close()

    asyncio.run(run())
