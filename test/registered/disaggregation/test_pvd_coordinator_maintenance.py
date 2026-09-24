"""Historical coordinator IDs remain fenced but leave maintenance scans."""

import asyncio

import pytest
import torch
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
