"""Pressure eviction keeps an Entry reusable until physical V space is needed."""

import asyncio
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestServer
from sglang.srt.disaggregation.pvd.index_search import BruteForceIndexBackend
from sglang.srt.disaggregation.pvd.control_server import (
    HttpShardClient,
    create_shard_app,
)
from sglang.srt.disaggregation.pvd.request_state import EntryState
from sglang.srt.disaggregation.pvd.vector_store import ResourceExhaustedError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd3 import make_ready_entry, make_vector


def test_pressure_evicts_oldest_unleased_entry_on_both_shards():
    async def run():
        engine, stores, coordinator = make_vector()
        try:
            keys = [
                await make_ready_entry(coordinator, engine, f"pressure-{i}")
                for i in range(8)
            ]
            assert all(store.allocator.available_pages == 0 for store in stores)
            newest = await make_ready_entry(coordinator, engine, "pressure-new")
            assert coordinator.entries[keys[0]].state == EntryState.RELEASED
            assert coordinator.entries[newest].state == EntryState.STORED
            assert all(store.allocator.available_pages == 0 for store in stores)
            assert (
                coordinator.metrics.snapshot()["counters"][
                    "coordinator_pressure_evictions"
                ]
                == 1
            )
        finally:
            for store in stores:
                store.close()

    asyncio.run(run())


def test_pressure_does_not_evict_a_consumer_lease_or_admit_without_room():
    async def run():
        engine, stores, coordinator = make_vector()
        try:
            keys = [
                await make_ready_entry(coordinator, engine, f"leased-{i}")
                for i in range(8)
            ]
            await coordinator.renew_consumer(keys[0], "decode-0")
            newest = await make_ready_entry(coordinator, engine, "leased-new")
            assert coordinator.entries[keys[0]].state == EntryState.STORED
            assert coordinator.entries[keys[1]].state == EntryState.RELEASED
            await coordinator.renew_consumer(newest, "decode-new")
            for key in keys[2:]:
                await coordinator.renew_consumer(key, f"decode-{key.req_id}")
            with pytest.raises(ResourceExhaustedError, match="no safely evictable"):
                await make_ready_entry(coordinator, engine, "leased-overflow")
            assert all(store.allocator.available_pages == 0 for store in stores)
        finally:
            for store in stores:
                store.close()

    asyncio.run(run())


def test_shard_capacity_api_is_compact_and_matches_physical_allocator():
    async def run():
        engine, stores, coordinator = make_vector()
        try:
            store = stores[1]
            async with TestServer(create_shard_app(store)) as server:
                client = HttpShardClient(1, str(server.make_url("")))
                try:
                    report = await client.capacity()
                    assert report == store.capacity_snapshot()
                    assert set(report) == {
                        "rank",
                        "worker_epoch",
                        "ready",
                        "total_pages",
                        "available_pages",
                        "largest_contiguous_free_pages",
                    }
                    assert report["largest_contiguous_free_pages"] == 16
                    assert "entries" not in report
                    key = await make_ready_entry(coordinator, engine, "capacity-post")
                    manifest = coordinator.entries[key].manifest
                    planned = await client.capacity(manifest)
                    assert planned == store.capacity_snapshot(manifest)
                    assert planned["index_admission_room"] is None
                finally:
                    await client.close()
        finally:
            for store in stores:
                store.close()

    asyncio.run(run())


def test_index_budget_preflight_counts_extraction_index_and_peak_scratch():
    async def run():
        engine, stores, coordinator = make_vector()
        try:
            key = await make_ready_entry(coordinator, engine, "index-estimate")
            store = stores[0]
            manifest = coordinator.entries[key].manifest
            store.prompt_index = SimpleNamespace(
                budget=TransferBudget(100, 1),
                backend=BruteForceIndexBackend(device="cpu"),
                metric="ip",
            )
            report = store.capacity_snapshot(manifest)
            assert report["index_required_bytes"] > 100
            assert report["index_available_bytes"] == 100
            assert report["index_admission_room"] is False
            store.prompt_index.budget = TransferBudget(1000, 1)
            assert store.capacity_snapshot(manifest)["index_admission_room"] is True
        finally:
            for store in stores:
                store.prompt_index = None
                store.close()

    asyncio.run(run())


def test_index_only_pressure_eviction_preserves_dense_admission():
    async def run():
        engine, stores, coordinator = make_vector()
        try:
            originals = [store.capacity_snapshot for store in stores]
            for store, original in zip(stores, originals, strict=True):

                def limited(manifest=None, *, store=store, original=original):
                    report = original(manifest)
                    if manifest is not None:
                        stored = sum(
                            entry.state.value == "stored"
                            for entry in store.entries.values()
                        )
                        report["index_admission_room"] = stored < 2
                    return report

                store.capacity_snapshot = limited
            oldest = await make_ready_entry(coordinator, engine, "index-oldest")
            await make_ready_entry(coordinator, engine, "index-second")
            assert all(store.allocator.available_pages == 12 for store in stores)
            newest = await make_ready_entry(coordinator, engine, "index-new")
            assert coordinator.entries[oldest].state == EntryState.RELEASED
            assert coordinator.entries[newest].state == EntryState.STORED
            assert all(store.allocator.available_pages == 12 for store in stores)
            await coordinator.renew_consumer(newest, "decode")
            remaining = next(
                key
                for key, entry in coordinator.entries.items()
                if entry.state == EntryState.STORED and key != newest
            )
            await coordinator.renew_consumer(remaining, "decode")
            dense = await make_ready_entry(
                coordinator, engine, "dense-even-if-index-full"
            )
            assert coordinator.entries[dense].state == EntryState.STORED
            assert (
                coordinator.metrics.snapshot()["counters"][
                    "coordinator_index_capacity_deferred"
                ]
                == 1
            )
        finally:
            for store in stores:
                store.close()

    asyncio.run(run())


def test_post_upload_index_pressure_reclaims_idle_entry_for_active_consumer():
    async def run():
        engine, stores, coordinator = make_vector()
        try:
            old = await make_ready_entry(coordinator, engine, "index-idle-old")
            active = await make_ready_entry(coordinator, engine, "index-active")
            await coordinator.renew_consumer(active, "decode-active")
            for store in stores:
                original = store.capacity_snapshot

                def limited(manifest=None, *, store=store, original=original):
                    report = original(manifest)
                    if manifest is not None:
                        report["index_build_pending"] = manifest.key == active
                        report["index_admission_room"] = not any(
                            entry.state.value == "stored" and entry.key == old
                            for entry in store.entries.values()
                        )
                    return report

                store.capacity_snapshot = limited
            assert await coordinator.relieve_index_pressure() == 1
            assert coordinator.entries[old].state == EntryState.RELEASED
            assert coordinator.entries[active].state == EntryState.STORED
            assert await coordinator.relieve_index_pressure() == 0
        finally:
            for store in stores:
                store.close()

    asyncio.run(run())


def test_index_pressure_does_not_reclaim_without_pending_active_search():
    async def run():
        engine, stores, coordinator = make_vector()
        try:
            old = await make_ready_entry(coordinator, engine, "idle-no-index")
            active = await make_ready_entry(coordinator, engine, "active-ready-index")
            await coordinator.renew_consumer(active, "decode-active")
            for store in stores:
                original = store.capacity_snapshot

                def ready(manifest=None, *, original=original):
                    report = original(manifest)
                    if manifest is not None:
                        report["index_build_pending"] = False
                        report["index_admission_room"] = False
                    return report

                store.capacity_snapshot = ready
            assert await coordinator.relieve_index_pressure() == 0
            assert coordinator.entries[old].state == EntryState.STORED
        finally:
            for store in stores:
                store.close()

    asyncio.run(run())
