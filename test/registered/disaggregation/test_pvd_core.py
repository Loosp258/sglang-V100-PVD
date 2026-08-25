import asyncio
import time
import unittest

import torch

from sglang.srt.disaggregation.pvd.coordinator import (
    CoordinatorError,
    LocalShardClient,
    VectorCoordinator,
)
from sglang.srt.disaggregation.pvd.kv_packer import (
    pack_full_prompt_kv,
    unpack_full_prompt_kv,
)
from sglang.srt.disaggregation.pvd.protocol import (
    FirstTokenMetadata,
    KVEntryKey,
    KVEntryManifest,
    KVLayoutSignature,
    KVShardManifest,
)
from sglang.srt.disaggregation.pvd.preflight import (
    PVDPreflightError,
    validate_rank_rail_names,
)
from sglang.srt.disaggregation.pvd.request_state import (
    DeliveryState,
    EntryState,
)
from sglang.srt.disaggregation.pvd.transfer_engine import (
    FakeTransferEngine,
    MemorySlice,
    TransferStatus,
)
from sglang.srt.disaggregation.pvd.vector_store import VectorKVStore


PAGE_BYTES = 16
ENTRY_BYTES = 32


def make_manifest(req_id: str = "req-1") -> KVEntryManifest:
    layout = KVLayoutSignature(
        model_id="test-model",
        model_revision="revision",
        kv_dtype="float16",
        page_size=4,
        num_layers=4,
        kv_heads_per_rank=2,
        head_dim=8,
        tp_size=2,
        pp_size=1,
        tensor_layout="flat-test-layout",
    )
    return KVEntryManifest(
        key=KVEntryKey.new("model-instance", req_id),
        layout=layout,
        prompt_token_count=5,
        shards=[
            KVShardManifest(
                rank=rank,
                rail=f"mlx5_{rank}",
                expected_bytes=ENTRY_BYTES,
                page_count=2,
                last_page_valid_tokens=1,
                layer_start=0,
                layer_end=4,
            )
            for rank in (0, 1)
        ],
    )


def make_store(rank: int, engine: FakeTransferEngine) -> VectorKVStore:
    return VectorKVStore(
        rank=rank,
        world_size=2,
        rail=f"mlx5_{rank}",
        device="cpu",
        total_pages=8,
        page_bytes=PAGE_BYTES,
        endpoint=f"v{rank}",
        transfer_engine=engine,
        allow_cpu_for_tests=True,
    )


def put_tensor(
    engine: FakeTransferEngine, tensor: torch.Tensor, target, rank: int
):
    registration = engine.register_memory(
        tensor, endpoint=f"source-{rank}", rank=rank, rail=f"mlx5_{rank}"
    )
    handle = engine.submit_put(
        MemorySlice(registration, offset=0, length=tensor.numel()), target
    )
    assert engine.poll(handle) == TransferStatus.SUCCESS
    return registration


def test_rank_rail_modes_are_explicit_and_bounded():
    assert validate_rank_rail_names(["mlx5_0", "mlx5_1"]) == "dual-rail"
    assert (
        validate_rank_rail_names(["mlx5_0", "mlx5_0"])
        == "single-rail-debug"
    )
    with unittest.TestCase().assertRaises(PVDPreflightError):
        validate_rank_rail_names(["mlx5_1", "mlx5_1"])


def test_full_prompt_entry_can_feed_multiple_deliveries():
    engine = FakeTransferEngine()
    store = make_store(0, engine)
    manifest = make_manifest()
    entry = store.create_entry(manifest)
    store.begin_p_write(manifest.key)

    prompt_kv = torch.arange(ENTRY_BYTES, dtype=torch.uint8)
    p_registration = put_tensor(engine, prompt_kv, entry.target_region, rank=0)
    store.commit_p_write(manifest.key, ENTRY_BYTES)

    d_registrations = []
    d_buffers = []
    try:
        for index in range(2):
            d_buffer = torch.zeros(ENTRY_BYTES, dtype=torch.uint8)
            d_registration = engine.register_memory(
                d_buffer,
                endpoint=f"d0-{index}",
                rank=0,
                rail="mlx5_0",
            )
            d_buffers.append(d_buffer)
            d_registrations.append(d_registration)
            delivery_id = f"delivery-{index}"
            store.reserve_delivery(
                manifest.key, delivery_id, d_registration.descriptor
            )
            delivery = store.start_delivery(manifest.key, delivery_id)
            assert delivery.state == DeliveryState.DELIVERED

        assert store.entries[manifest.key].active_delivery_count == 2
        assert all(torch.equal(buffer, prompt_kv) for buffer in d_buffers)

        for index in range(2):
            store.ack_delivery(manifest.key, f"delivery-{index}")
        assert store.entries[manifest.key].active_delivery_count == 0

        store.release_entry(manifest.key)
        assert store.entries[manifest.key].resources_released
    finally:
        engine.release_memory(p_registration)
        for registration in d_registrations:
            engine.release_memory(registration)
        store.close()


def test_bounded_descriptor_uses_pool_base_offset():
    engine = FakeTransferEngine()
    store = make_store(0, engine)
    first = make_manifest("first")
    second = make_manifest("second")
    first_entry = store.create_entry(first)
    second_entry = store.create_entry(second)
    assert second_entry.target_region.backend_metadata["base_offset"] > 0

    source = torch.full((ENTRY_BYTES,), 77, dtype=torch.uint8)
    registration = put_tensor(engine, source, second_entry.target_region, rank=0)
    try:
        offset = int(second_entry.target_region.backend_metadata["base_offset"])
        assert torch.count_nonzero(store.pool[:offset]) == 0
        assert torch.equal(store.pool[offset : offset + ENTRY_BYTES], source)
        assert first_entry.target_region.region_id == second_entry.target_region.region_id
    finally:
        engine.release_memory(registration)
        store.close()


def test_full_prompt_packer_preserves_page_and_component_order():
    class Pool:
        def __init__(self, fill: bool):
            self.k_buffer = [
                torch.arange(16, dtype=torch.float16).reshape(8, 1, 2)
                if fill
                else torch.zeros((8, 1, 2), dtype=torch.float16)
            ]
            self.v_buffer = [
                torch.arange(100, 116, dtype=torch.float16).reshape(8, 1, 2)
                if fill
                else torch.zeros((8, 1, 2), dtype=torch.float16)
            ]

    source = Pool(fill=True)
    destination = Pool(fill=False)
    packed = pack_full_prompt_kv(source, [1, 3], page_size=2)
    unpack_full_prompt_kv(packed.tensor, destination, [0, 2], page_size=2)

    for source_component, destination_component in zip(
        source.k_buffer + source.v_buffer,
        destination.k_buffer + destination.v_buffer,
        strict=True,
    ):
        assert torch.equal(destination_component[0:2], source_component[2:4])
        assert torch.equal(destination_component[4:6], source_component[6:8])
        assert torch.count_nonzero(destination_component[2:4]) == 0
        assert torch.count_nonzero(destination_component[6:8]) == 0


def test_coordinator_requires_rank0_first_token_and_returns_two_targets():
    async def scenario():
        engine = FakeTransferEngine()
        stores = [make_store(rank, engine) for rank in (0, 1)]
        coordinator = VectorCoordinator([LocalShardClient(store) for store in stores])
        manifest = make_manifest()
        try:
            entry = await coordinator.create_entry(manifest)
            assert entry.state == EntryState.P_WRITING
            assert sorted(entry.target_regions) == [0, 1]
            with unittest.TestCase().assertRaisesRegex(CoordinatorError, 'rank 0'):
                await coordinator.commit_shard(
                    manifest.key,
                    1,
                    ENTRY_BYTES,
                    FirstTokenMetadata(output_token_id=10),
                )

            await coordinator.commit_shard(manifest.key, 1, ENTRY_BYTES)
            entry = await coordinator.commit_shard(
                manifest.key,
                0,
                ENTRY_BYTES,
                FirstTokenMetadata(output_token_id=10),
            )
            assert entry.state == EntryState.STORED
            assert entry.first_token.output_token_id == 10
        finally:
            for store in stores:
                store.close()

    asyncio.run(scenario())


def test_coordinator_create_and_delivery_are_idempotent_under_rank_races():
    async def scenario():
        engine = FakeTransferEngine()
        stores = [make_store(rank, engine) for rank in (0, 1)]
        coordinator = VectorCoordinator([LocalShardClient(store) for store in stores])
        manifest = make_manifest("concurrent")
        registrations = []
        try:
            first, second = await asyncio.gather(
                coordinator.create_entry(manifest),
                coordinator.create_entry(manifest),
            )
            assert first.target_regions == second.target_regions
            await coordinator.commit_shard(
                manifest.key,
                0,
                ENTRY_BYTES,
                FirstTokenMetadata(output_token_id=11),
            )
            await coordinator.commit_shard(manifest.key, 1, ENTRY_BYTES)

            destinations = {}
            for rank in (0, 1):
                tensor = torch.zeros(ENTRY_BYTES, dtype=torch.uint8)
                registration = engine.register_memory(
                    tensor,
                    endpoint=f"d{rank}",
                    rank=rank,
                    rail=f"mlx5_{rank}",
                )
                registrations.append(registration)
                destinations[rank] = registration.descriptor
            await asyncio.gather(
                coordinator.reserve_delivery(
                    key=manifest.key,
                    delivery_id="shared-delivery",
                    destinations=destinations,
                ),
                coordinator.reserve_delivery(
                    key=manifest.key,
                    delivery_id="shared-delivery",
                    destinations=destinations,
                ),
            )
            deliveries = await asyncio.gather(
                coordinator.start_delivery("shared-delivery"),
                coordinator.start_delivery("shared-delivery"),
            )
            assert all(item.state == DeliveryState.DELIVERED for item in deliveries)
            await asyncio.gather(
                coordinator.ack_delivery("shared-delivery"),
                coordinator.ack_delivery("shared-delivery"),
            )
            assert coordinator.entries[manifest.key].active_delivery_count == 0
        finally:
            for registration in registrations:
                engine.release_memory(registration)
            for store in stores:
                store.close()

    asyncio.run(scenario())


def test_coordinator_owns_entry_ttl():
    async def scenario():
        engine = FakeTransferEngine()
        stores = [make_store(rank, engine) for rank in (0, 1)]
        coordinator = VectorCoordinator(
            [LocalShardClient(store) for store in stores], entry_ttl_secs=0.01
        )
        manifest = make_manifest("ttl")
        try:
            await coordinator.create_entry(manifest)
            await coordinator.commit_shard(
                manifest.key,
                0,
                ENTRY_BYTES,
                FirstTokenMetadata(output_token_id=12),
            )
            await coordinator.commit_shard(manifest.key, 1, ENTRY_BYTES)
            result = await coordinator.reap_expired(time.monotonic() + 1.0)
            assert result["entries"] == 1
            assert coordinator.entries[manifest.key].state == EntryState.RELEASED
            assert all(store.entries[manifest.key].resources_released for store in stores)
        finally:
            for store in stores:
                store.close()

    asyncio.run(scenario())


if __name__ == '__main__':
    suite = unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_rank_rail_modes_are_explicit_and_bounded,
            test_full_prompt_entry_can_feed_multiple_deliveries,
            test_bounded_descriptor_uses_pool_base_offset,
            test_full_prompt_packer_preserves_page_and_component_order,
            test_coordinator_requires_rank0_first_token_and_returns_two_targets,
            test_coordinator_create_and_delivery_are_idempotent_under_rank_races,
            test_coordinator_owns_entry_ttl,
        )
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
