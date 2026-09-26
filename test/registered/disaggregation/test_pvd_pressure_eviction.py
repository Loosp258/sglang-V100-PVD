"""Pressure eviction keeps an Entry reusable until physical V space is needed."""

import asyncio

import pytest
from aiohttp.test_utils import TestServer
from sglang.srt.disaggregation.pvd.control_server import (
    HttpShardClient,
    create_shard_app,
)
from sglang.srt.disaggregation.pvd.request_state import EntryState
from sglang.srt.disaggregation.pvd.vector_store import ResourceExhaustedError
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
        _, stores, _coordinator = make_vector()
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
                finally:
                    await client.close()
        finally:
            for store in stores:
                store.close()

    asyncio.run(run())
