"""Fail-closed V coordinator history bounds within one worker epoch."""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.coordinator import (
    CoordinatorError,
    VectorCoordinator,
)
from sglang.srt.disaggregation.pvd.protocol import (
    PVD_TRANSFER_LIFECYCLE_PROTOCOL,
    KVEntryKey,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.vector_store import ResourceExhaustedError
from test_pvd3 import make_ready_entry, make_vector


def bare_coordinator(**bounds):
    return VectorCoordinator(
        [SimpleNamespace(rank=0), SimpleNamespace(rank=1)], **bounds
    )


def test_router_admission_capacity_is_atomic_and_existing_ids_can_renew():
    async def run():
        coordinator = bare_coordinator(max_admissions=1)
        first = {
            "pvd_transfer_id": "transfer-1",
            "pvd_delivery_id": "delivery-1",
            "pvd_vector_group_id": "group-1",
        }
        assert await coordinator.admit_request(first) == {"accepted": ["transfer-1"]}
        assert await coordinator.admit_request(first) == {"accepted": ["transfer-1"]}
        with pytest.raises(ResourceExhaustedError, match="admission capacity"):
            await coordinator.admit_request(
                {
                    "pvd_transfer_id": ["transfer-1", "transfer-2"],
                    "pvd_delivery_id": ["delivery-1", "delivery-2"],
                    "pvd_vector_group_id": ["group-1", "group-1"],
                }
            )
        assert list(coordinator.admissions) == ["transfer-1"]

    asyncio.run(run())


def test_unknown_retrieval_fence_capacity_keeps_existing_tombstone():
    async def run():
        coordinator = bare_coordinator(max_unknown_retrieval_fences=1)
        identity = WriteIdentity(
            protocol=PVD_TRANSFER_LIFECYCLE_PROTOCOL,
            sender_epoch="sender",
            receiver_epoch="receiver",
            transfer_id="subdelivery",
            region_id="region",
            generation="generation",
            shard_rank=0,
            key=KVEntryKey.new("model", "request"),
        ).to_dict()
        for _ in range(2):
            with pytest.raises(CoordinatorError, match="unknown delivery"):
                await coordinator.fence_retrieval("missing-one", [identity])
        with pytest.raises(ResourceExhaustedError, match="fence capacity"):
            await coordinator.fence_retrieval("missing-two", [identity])
        assert coordinator._unknown_retrieval_fences == {"missing-one"}
        assert coordinator._fenced_retrievals == {"missing-one"}

    asyncio.run(run())


def test_entry_record_capacity_refuses_new_key_before_shard_allocation():
    async def run():
        engine, stores, coordinator = make_vector()
        coordinator._max_entry_records = 1
        key = await make_ready_entry(coordinator, engine, "first")
        await coordinator.release_entry(key)
        before = [store.allocator.available_pages for store in stores]
        assert (
            await coordinator.create_entry(coordinator.entries[key].manifest)
            is coordinator.entries[key]
        )
        with pytest.raises(ResourceExhaustedError, match="Entry record capacity"):
            await make_ready_entry(coordinator, engine, "second")
        assert [store.allocator.available_pages for store in stores] == before
        assert len(coordinator.entries) == 1
        for store in stores:
            store.close()

    asyncio.run(run())


def test_delivery_record_capacity_refuses_new_id_before_shard_reservation():
    async def run():
        engine, stores, coordinator = make_vector()
        coordinator._max_delivery_records = 1
        key = await make_ready_entry(coordinator, engine, "first")
        destinations = {
            rank: engine.register_memory(
                torch.zeros(32, dtype=torch.uint8),
                endpoint=f"decode-{rank}",
                rank=rank,
                rail=f"mlx5_{rank}",
            ).descriptor
            for rank in range(2)
        }
        first = await coordinator.reserve_delivery(
            key=key, delivery_id="first", destinations=destinations
        )
        assert (
            await coordinator.reserve_delivery(
                key=key, delivery_id="first", destinations=destinations
            )
            is first
        )
        with pytest.raises(ResourceExhaustedError, match="Delivery record capacity"):
            await coordinator.reserve_delivery(
                key=key, delivery_id="second", destinations=destinations
            )
        assert len(coordinator.deliveries) == 1
        assert all(len(store.entries[key].deliveries) == 1 for store in stores)
        await coordinator.cancel_delivery("first", "test complete")
        for store in stores:
            store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "name",
    [
        "max_entry_records",
        "max_delivery_records",
        "max_admissions",
        "max_unknown_retrieval_fences",
    ],
)
@pytest.mark.parametrize("bound", [0, -1, True, 1.5])
def test_coordinator_history_bounds_are_positive_exact_integers(name, bound):
    with pytest.raises(ValueError, match=name):
        bare_coordinator(**{name: bound})
