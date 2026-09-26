"""Historical coordinator IDs remain fenced but leave maintenance scans."""

import asyncio
from dataclasses import replace

import pytest
import torch
from sglang.srt.disaggregation.pvd.coordinator import CoordinatorError
from sglang.srt.disaggregation.pvd.protocol import KVEntryKey
from sglang.srt.disaggregation.pvd.vector_store import (
    EntryConflictError,
    ResourceExhaustedError,
)
from test_pvd3 import make_ready_entry, make_vector


def test_released_entry_tombstone_is_not_rescanned_by_coordinator_reaper():
    async def run():
        engine, stores, coordinator = make_vector()
        key = await make_ready_entry(coordinator, engine, "released")
        await coordinator.release_entry(key)
        assert key in coordinator.entries
        await coordinator.reap_expired()
        assert key not in coordinator._maintenance_entries

        class UnscannableLeases(dict):
            def items(self):
                pytest.fail("released Entry tombstone scanned again")

        entry = coordinator.entries[key]
        original = entry.consumer_leases
        entry.consumer_leases = UnscannableLeases()
        try:
            assert await coordinator.reap_expired() == {
                "entries": 0,
                "deliveries": 0,
            }
        finally:
            entry.consumer_leases = original
        assert coordinator.entries[key] is entry
        assert (await coordinator.health())["record_capacity"][
            "maintenance_entries"
        ] == 0
        for store in stores:
            store.close()

    asyncio.run(run())


def test_terminal_delivery_tombstone_is_not_rescanned_by_coordinator_reaper():
    async def run():
        engine, stores, coordinator = make_vector()
        key = await make_ready_entry(coordinator, engine, "delivery")
        destinations = {
            rank: engine.register_memory(
                torch.zeros(32, dtype=torch.uint8),
                endpoint=f"decode-{rank}",
                rank=rank,
                rail=f"mlx5_{rank}",
            ).descriptor
            for rank in range(2)
        }
        delivery = await coordinator.reserve_delivery(
            key=key, delivery_id="terminal", destinations=destinations
        )
        await coordinator.cancel_delivery("terminal", "test cleanup")
        await coordinator.reap_expired()
        assert "terminal" in coordinator.deliveries
        assert "terminal" not in coordinator._maintenance_deliveries

        class UnscannableState:
            def __hash__(self):
                pytest.fail("terminal Delivery tombstone scanned again")

        original = delivery.state
        delivery.state = UnscannableState()
        try:
            assert await coordinator.reap_expired() == {
                "entries": 0,
                "deliveries": 0,
            }
        finally:
            delivery.state = original
        assert coordinator.deliveries["terminal"] is delivery
        for store in stores:
            store.close()

    asyncio.run(run())


def test_failed_entry_create_reports_original_error_and_unconfirmed_cleanup():
    async def run():
        engine, stores, coordinator = make_vector()
        try:
            seed = await make_ready_entry(coordinator, engine, "seed")
            manifest = replace(
                coordinator.entries[seed].manifest,
                key=KVEntryKey("model", "failed-create", "failed-create"),
            )
            shard0, shard1 = coordinator.shards[0], coordinator.shards[1]
            original_create = shard0.create_entry
            original_cancel = shard1.cancel_entry

            async def fail_create(*args, **kwargs):
                raise ValueError("original allocation failure")

            async def fail_cancel(*args, **kwargs):
                raise OSError("cleanup RPC unavailable")

            shard0.create_entry = fail_create
            shard1.cancel_entry = fail_cancel
            with pytest.raises(CoordinatorError) as caught:
                await coordinator.create_entry(manifest)
            assert "original allocation failure" in str(caught.value)
            assert "cleanup RPC unavailable" in str(caught.value)
            assert isinstance(caught.value.__cause__, ValueError)
            assert manifest.key in coordinator._pending_entry_cancellations

            shard0.create_entry = original_create
            shard1.cancel_entry = original_cancel
            await coordinator.reap_expired()
            assert manifest.key not in coordinator._pending_entry_cancellations
        finally:
            for store in stores:
                store.close()

    asyncio.run(run())


def test_absent_shard_cancellation_fences_late_create_and_is_bounded():
    async def run():
        engine, stores, coordinator = make_vector()
        try:
            seed = await make_ready_entry(coordinator, engine, "seed")
            shard = stores[0]
            shard._max_absent_entry_cancellations = 1
            key = KVEntryKey("model", "never-allocated", "never-allocated")
            manifest = replace(coordinator.entries[seed].manifest, key=key)
            shard.cancel_entry(key, "peer allocation failed")
            shard.cancel_entry(key, "retry")
            assert shard.snapshot()["absent_entry_cancellations"] == 1
            with pytest.raises(EntryConflictError, match="cancelled before allocation"):
                shard.create_entry(manifest)
            another = KVEntryKey("model", "another", "another")
            with pytest.raises(ResourceExhaustedError, match="fence capacity"):
                shard.cancel_entry(another, "capacity")
            assert key not in shard.entries
            assert another not in shard._absent_entry_cancellations
        finally:
            for store in stores:
                store.close()

    asyncio.run(run())
