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
from sglang.srt.disaggregation.pvd.cuda_runtime_group import CUDARuntimeInstallGroup
from sglang.srt.disaggregation.pvd.cuda_sparse_fanin import CUDASparseFanInStage
from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import CUDASparseReceiveRegistry
from sglang.srt.disaggregation.pvd.cuda_working_set import CUDASparseWorkingSet
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
from test_pvd_prompt_index import SPACE, build_entry, manager
from test_pvd_prompt_vectors import pack_shard
from test_pvd_search_routing import layout
from test_pvd_vector_lifecycle import DelayedTransferEngine


@asynccontextmanager
async def two_source(monkeypatch):
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
        routing = RoutedShardSearchClient(
            storage_layout=storage,
            compute_layout=compute,
            compute_rank=0,
            entry_transfer_id=manifest.key.transfer_id,
            prompt_tokens=8,
            vector_space=SPACE,
            metric="l2",
            clients={
                rank: PVDShardSearchClient(str(server.make_url("")))
                for rank, server in servers.items()
            },
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
        epoch = group.begin(3)
        specs = tuple(
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
        plans = routing.partition_specs(specs)
        registry = CUDASparseReceiveRegistry(
            engine, TransferBudget(1 << 20, 8), receiver_epoch="D", device="cuda:0"
        )
        registry.device = torch.device("cpu")
        registry.ordering = SimpleNamespace(
            prepare=lambda _: events.append("register_order"),
            after_remote_write=lambda _: events.append("remote_visible"),
        )
        records = {
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
