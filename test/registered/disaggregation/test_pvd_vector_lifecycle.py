"""Real V store/allocator with controllable asynchronous transport boundaries."""

import asyncio
import dataclasses
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from sglang.srt.disaggregation.pvd.request_state import DeliveryState
from sglang.srt.disaggregation.pvd.transfer_engine import (
    FakeTransferEngine,
    TransferHandle,
    TransferStatus,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransportState
from test_pvd_core import ENTRY_BYTES, make_manifest, make_store


class DelayedTransferEngine(FakeTransferEngine):
    def __init__(self):
        super().__init__()
        self.pending = {}
        self.released = []

    def submit_put(self, local, remote, *, remote_offset=0):
        handle = TransferHandle(
            uuid.uuid4().hex, transport_state=TransportState.IN_FLIGHT
        )
        self.pending[handle.transfer_id] = (local, remote, remote_offset)
        return handle

    def finish(self, handle, success=True):
        local, remote, offset = self.pending.pop(handle.transfer_id)
        if success:
            assert local.registration.descriptor.region_id in self._regions
            result = super().submit_put(local, remote, remote_offset=offset)
            assert result.status == TransferStatus.SUCCESS
            handle.transferred_bytes = result.transferred_bytes
        handle.transport_state = (
            TransportState.TERMINAL_SUCCESS
            if success
            else TransportState.TERMINAL_FAILED
        )
        if handle.status == TransferStatus.PENDING:
            handle.status = TransferStatus.SUCCESS if success else TransferStatus.FAILED

    def release_memory(self, registration):
        self.released.append(registration.descriptor.region_id)
        super().release_memory(registration)


def ready_store(engine=None):
    engine = engine or DelayedTransferEngine()
    store = make_store(0, engine)
    manifest = make_manifest()
    entry = store.create_entry(manifest)
    store.begin_p_write(manifest.key)
    store.pool[:ENTRY_BYTES] = torch.arange(ENTRY_BYTES, dtype=torch.uint8)
    store.commit_p_write(manifest.key, ENTRY_BYTES)
    return engine, store, entry


def reserve(engine, store, entry, name="delivery", *, heterogeneous=False):
    metadata = {
        "pvd_receiver_epoch": "decode-worker-epoch",
        "pvd_generation": uuid.uuid4().hex,
    }
    size = ENTRY_BYTES
    if heterogeneous:
        layout = dataclasses.replace(
            entry.layout,
            tp_size=4,
            kv_heads_per_rank=1,
            extra={
                **entry.layout.extra,
                "component_bytes_per_token": [2],
                "component_token_shapes": [[1, 1]],
            },
        )
        metadata["pvd_layout"] = layout.to_dict()
        size //= 2
    target = engine.register_memory(
        torch.zeros(size, dtype=torch.uint8),
        endpoint="D",
        rank=0,
        rail="mlx5_0",
        metadata=metadata,
    )
    return store.reserve_delivery(entry.key, name, target.descriptor), target


@pytest.mark.parametrize("success", [True, False])
def test_cancel_keeps_entry_pages_until_real_transport_terminal(success):
    engine, store, entry = ready_store()
    delivery, _ = reserve(engine, store, entry)
    before = store.allocator.available_pages
    store.start_delivery(entry.key, delivery.delivery_id)
    store.cancel_entry(entry.key, "timeout")
    assert store.allocator.available_pages == before
    assert not entry.resources_released
    assert store.fence_write(delivery.authorization.identity)["fenced"] is False
    engine.finish(delivery.transfer_handle, success)
    store.progress_transfers()
    assert store.fence_write(delivery.authorization.identity)["fenced"] is True
    assert entry.resources_released
    assert delivery.state == DeliveryState.CANCELLED
    store.close()


def test_heterogeneous_staging_survives_cancel_and_late_completion():
    engine, store, entry = ready_store()
    delivery, _ = reserve(engine, store, entry, heterogeneous=True)
    store.start_delivery(entry.key, delivery.delivery_id)
    source = engine.pending[delivery.transfer_handle.transfer_id][0].registration
    store.cancel_entry(entry.key, "cancel")
    assert source.descriptor.region_id not in engine.released
    engine.finish(delivery.transfer_handle)
    store.progress_transfers()
    assert engine.released.count(source.descriptor.region_id) == 1
    assert entry.resources_released
    store.close()


def test_two_deliveries_pin_the_same_entry_independently():
    engine, store, entry = ready_store()
    first, _ = reserve(engine, store, entry, "first")
    second, _ = reserve(engine, store, entry, "second")
    store.start_delivery(entry.key, "first")
    store.start_delivery(entry.key, "second")
    store.cancel_entry(entry.key, "cancel both")
    engine.finish(first.transfer_handle)
    store.progress_transfers()
    assert not entry.resources_released
    engine.finish(second.transfer_handle)
    store.progress_transfers()
    assert entry.resources_released
    assert entry.active_delivery_count == 0
    store.close()


@pytest.mark.parametrize("action", ["cancel", "ttl", "close"])
def test_unconfirmed_upload_keeps_pages_and_pool_registered(action):
    engine = DelayedTransferEngine()
    store = make_store(0, engine)
    entry = store.create_entry(make_manifest())
    store.begin_p_write(entry.key)
    if action == "cancel":
        store.cancel_entry(entry.key, "upload timeout")
    elif action == "ttl":
        store.reap_expired(time.monotonic() + 1000)
    else:
        store.close()
    assert not entry.resources_released
    assert store.allocator.allocated_pages == entry.allocation.page_count
    assert store.registration.descriptor.region_id not in engine.released


def test_fence_during_native_submit_does_not_wait_for_store_lock_or_free_source():
    engine, store, entry = ready_store()
    delivery, _ = reserve(engine, store, entry)
    entered, resume = threading.Event(), threading.Event()
    original = engine.submit_put

    def blocked(*args, **kwargs):
        entered.set()
        assert resume.wait(5)
        return original(*args, **kwargs)

    engine.submit_put = blocked
    with ThreadPoolExecutor(max_workers=2) as executor:
        submitting = executor.submit(
            store.start_delivery, entry.key, delivery.delivery_id
        )
        assert entered.wait(5)
        try:
            fencing = executor.submit(
                store.fence_write, delivery.authorization.identity
            )
            assert fencing.result(timeout=1)["fenced"] is False
            assert not entry.resources_released
        finally:
            resume.set()
        submitting.result(timeout=5)
    engine.finish(delivery.transfer_handle)
    store.progress_transfers()
    assert store.fence_write(delivery.authorization.identity)["fenced"] is True
    store.close()


@pytest.mark.parametrize("failure", ["submit", "poll"])
def test_lost_native_tracking_retains_resources_and_isolates_admission(failure):
    engine, store, entry = ready_store()
    delivery, _ = reserve(engine, store, entry, heterogeneous=True)

    def fail(*args, **kwargs):
        raise RuntimeError("native tracking lost")

    if failure == "submit":
        engine.submit_put = fail
    else:
        engine.poll = fail
    store.start_delivery(entry.key, delivery.delivery_id)
    store.cancel_entry(entry.key, "abort")
    assert not entry.resources_released
    assert store.fence_write(delivery.authorization.identity)["fenced"] is False
    assert store.snapshot()["isolated_reason"]
    with pytest.raises(Exception, match="isolated"):
        store.create_entry(make_manifest("new-request"))
    assert delivery.staging_guard.value is not None


def test_wrong_epoch_fence_does_not_cancel_current_write():
    engine, store, entry = ready_store()
    delivery, _ = reserve(engine, store, entry)
    store.start_delivery(entry.key, delivery.delivery_id)
    stale = dataclasses.replace(
        delivery.authorization.identity, sender_epoch="old-process"
    )
    with pytest.raises(Exception, match="identity"):
        store.fence_write(stale)
    assert delivery.state == DeliveryState.V_WRITING
    engine.finish(delivery.transfer_handle)
    store.progress_transfers()
    assert delivery.state == DeliveryState.DELIVERED
    store.close()


def test_closed_unsubmitted_authorization_cannot_start_or_reuse_changed_destination():
    engine, store, entry = ready_store()
    delivery, target = reserve(engine, store, entry)
    assert store.fence_write(delivery.authorization.identity)["fenced"] is True
    with pytest.raises(Exception, match="fenced"):
        store.start_delivery(entry.key, delivery.delivery_id)
    with pytest.raises(Exception, match="fenced"):
        store.reserve_delivery(
            entry.key,
            delivery.delivery_id,
            dataclasses.replace(
                target.descriptor, address=target.descriptor.address + 1
            ),
        )
    assert not engine.pending
    store.close()


def test_close_with_inflight_write_retains_pool_until_terminal_and_prevents_new_work():
    engine, store, entry = ready_store()
    delivery, _ = reserve(engine, store, entry)
    store.start_delivery(entry.key, delivery.delivery_id)
    store.close()
    assert store.registration.descriptor.region_id not in engine.released
    with pytest.raises(Exception, match="closed"):
        store.create_entry(make_manifest("late"))
    engine.finish(delivery.transfer_handle)
    store.progress_transfers()
    store.close()
    assert engine.released.count(store.registration.descriptor.region_id) == 1


def test_cleanup_failure_is_retried_without_double_freeing_pages():
    engine, store, entry = ready_store()
    delivery, _ = reserve(engine, store, entry, heterogeneous=True)
    store.start_delivery(entry.key, delivery.delivery_id)
    source = engine.pending[delivery.transfer_handle.transfer_id][0].registration
    original = engine.release_memory
    failed = []

    def release(registration):
        if registration is source and not failed:
            failed.append(True)
            raise RuntimeError("transient unregister failure")
        original(registration)

    engine.release_memory = release
    store.cancel_entry(entry.key, "cancel")
    engine.finish(delivery.transfer_handle)
    store.progress_transfers()
    assert source.descriptor.region_id in engine._regions
    assert (entry.key, delivery.delivery_id) in store._active_progress
    store.progress_transfers()
    assert source.descriptor.region_id not in engine._regions
    assert (entry.key, delivery.delivery_id) not in store._active_progress
    assert store.allocator.allocated_pages == 0
    store.close()


def test_terminal_delivery_keeps_replay_but_stops_background_and_direct_poll():
    class CountingEngine(DelayedTransferEngine):
        def __init__(self):
            super().__init__()
            self.poll_count = 0

        def poll(self, handle):
            self.poll_count += 1
            return super().poll(handle)

    engine, store, entry = ready_store(CountingEngine())
    delivery, _ = reserve(engine, store, entry)
    store.start_delivery(entry.key, delivery.delivery_id)
    assert (entry.key, delivery.delivery_id) in store._active_progress
    engine.finish(delivery.transfer_handle)
    store.progress_transfers()
    assert delivery.state == DeliveryState.DELIVERED
    assert (entry.key, delivery.delivery_id) not in store._active_progress
    settled_polls = engine.poll_count
    for _ in range(20):
        store.progress_transfers()
        assert (
            store.poll_delivery(entry.key, delivery.delivery_id).state
            == DeliveryState.DELIVERED
        )
    assert engine.poll_count == settled_polls
    store.ack_delivery(entry.key, delivery.delivery_id)
    assert (
        store.poll_delivery(entry.key, delivery.delivery_id).state
        == DeliveryState.RELEASED
    )
    store.close()


def test_unsettled_native_cleanup_keeps_terminal_delivery_pollable():
    class CleanupEngine(DelayedTransferEngine):
        def __init__(self):
            super().__init__()
            self.cleanup_ready = False
            self.poll_count = 0

        def poll(self, handle):
            self.poll_count += 1
            return super().poll(handle)

        def cleanup_complete(self, handle):
            return (
                self.cleanup_ready
                and handle.transport_state.is_locally_safe_to_release
            )

    engine, store, entry = ready_store(CleanupEngine())
    delivery, _ = reserve(engine, store, entry)
    store.start_delivery(entry.key, delivery.delivery_id)
    engine.finish(delivery.transfer_handle)
    store.progress_transfers()
    assert delivery.state == DeliveryState.DELIVERED
    assert (entry.key, delivery.delivery_id) in store._active_progress
    previous_polls = engine.poll_count
    store.progress_transfers()
    assert engine.poll_count > previous_polls
    engine.cleanup_ready = True
    store.progress_transfers()
    assert (entry.key, delivery.delivery_id) not in store._active_progress
    store.close()


def test_unknown_delivery_remains_active_and_unfenced():
    engine, store, entry = ready_store()
    delivery, _ = reserve(engine, store, entry)
    store.start_delivery(entry.key, delivery.delivery_id)
    delivery.transfer_handle.transport_state = TransportState.UNKNOWN
    store.cancel_delivery(entry.key, delivery.delivery_id, "unknown native outcome")
    store.progress_transfers()
    assert (entry.key, delivery.delivery_id) in store._active_progress
    assert store.fence_write(delivery.authorization.identity)["fenced"] is False
    assert not delivery.progress_settled


@pytest.mark.parametrize("cancelled", [True, False])
def test_real_http_poll_and_identity_fence_across_two_v_shards(cancelled):
    async def scenario():
        from aiohttp.test_utils import TestServer
        from sglang.srt.disaggregation.pvd.client import PVDCoordinatorClient
        from sglang.srt.disaggregation.pvd.control_server import (
            HttpShardClient,
            create_coordinator_app,
            create_shard_app,
        )
        from sglang.srt.disaggregation.pvd.coordinator import (
            LocalShardClient,
            VectorCoordinator,
        )
        from sglang.srt.disaggregation.pvd.protocol import FirstTokenMetadata

        engine = DelayedTransferEngine()
        stores = [make_store(rank, engine) for rank in range(2)]
        async with TestServer(create_shard_app(stores[1])) as shard_server:
            remote = HttpShardClient(1, str(shard_server.make_url("")))
            coordinator = VectorCoordinator([LocalShardClient(stores[0]), remote])
            manifest = make_manifest("http-delayed")
            await coordinator.create_entry(manifest)
            for rank, store in enumerate(stores):
                store.pool[:ENTRY_BYTES] = rank + 1
                await coordinator.commit_shard(
                    manifest.key,
                    rank,
                    ENTRY_BYTES,
                    FirstTokenMetadata(output_token_id=2) if rank == 0 else None,
                )
            destinations = {
                rank: engine.register_memory(
                    torch.zeros(ENTRY_BYTES, dtype=torch.uint8),
                    endpoint=f"D{rank}",
                    rank=rank,
                    rail=f"mlx5_{rank}",
                    metadata={
                        "pvd_receiver_epoch": f"decode-epoch-{rank}",
                        "pvd_generation": "generation-1",
                    },
                ).descriptor
                for rank in range(2)
            }
            async with TestServer(create_coordinator_app(coordinator)) as server:
                client = PVDCoordinatorClient(str(server.make_url("")))
                try:
                    reserved = await client.reserve_delivery(
                        manifest.key, "http-delivery", destinations
                    )
                    identities = list(reserved["write_identities"].values())
                    assert len(identities) == 2
                    started = await client.start_delivery("http-delivery")
                    assert started["state"] == "v_writing"
                    assert (await client.poll_delivery("http-delivery"))[
                        "state"
                    ] == "v_writing"
                    if cancelled:
                        assert (
                            await client.fence_retrieval("http-delivery", identities)
                        )["fenced"] is False
                    for rank, store in enumerate(stores):
                        delivery = store.entries[manifest.key].deliveries[
                            f"http-delivery:d{rank}"
                        ]
                        engine.finish(delivery.transfer_handle)
                    result = await client.poll_delivery("http-delivery")
                    if not cancelled:
                        assert result["state"] == "delivered"
                        for rank, destination in destinations.items():
                            assert torch.all(
                                engine._regions[destination.region_id].buffer
                                == rank + 1
                            )
                        await client.ack_delivery("http-delivery")
                    assert (await client.fence_retrieval("http-delivery", identities))[
                        "fenced"
                    ] is True
                    assert all(
                        store.entries[manifest.key].active_delivery_count == 0
                        for store in stores
                    )
                finally:
                    await client.close()
            await remote.close()
        for store in stores:
            store.close()

    asyncio.run(scenario())


def test_expired_upload_cannot_be_resurrected_by_late_begin_or_commit():
    engine = DelayedTransferEngine()
    store = make_store(0, engine)
    entry = store.create_entry(make_manifest())
    store.reap_expired(time.monotonic() + 1000)
    with pytest.raises(Exception, match="release"):
        store.begin_p_write(entry.key)
    with pytest.raises(Exception, match="release"):
        store.commit_p_write(entry.key, ENTRY_BYTES)
    assert entry.upload_pending
    assert not entry.resources_released


def test_coordinator_retrieval_does_not_hold_fence_lock_across_submit():
    async def scenario():
        from sglang.srt.disaggregation.pvd.coordinator import (
            LocalShardClient,
            VectorCoordinator,
        )
        from sglang.srt.disaggregation.pvd.protocol import FirstTokenMetadata

        engine = DelayedTransferEngine()
        stores = [make_store(rank, engine) for rank in range(2)]
        coordinator = VectorCoordinator([LocalShardClient(store) for store in stores])
        manifest = make_manifest("blocked-submit")
        await coordinator.create_entry(manifest)
        for rank in range(2):
            await coordinator.commit_shard(
                manifest.key,
                rank,
                ENTRY_BYTES,
                FirstTokenMetadata(output_token_id=2) if rank == 0 else None,
            )
        destinations = {
            rank: engine.register_memory(
                torch.zeros(ENTRY_BYTES, dtype=torch.uint8),
                endpoint=f"D{rank}",
                rank=rank,
                rail=f"mlx5_{rank}",
                metadata={"pvd_receiver_epoch": f"D-{rank}", "pvd_generation": "g"},
            ).descriptor
            for rank in range(2)
        }
        entered, resume = threading.Event(), threading.Event()
        original = engine.submit_put

        def blocked(*args, **kwargs):
            entered.set()
            assert resume.wait(5)
            return original(*args, **kwargs)

        engine.submit_put = blocked
        request = {
            "key": manifest.key.to_dict(),
            "sequence_id": manifest.key.req_id,
            "delivery_id": "blocked",
            "selection": "full_prompt",
            "destinations": {
                str(rank): d.to_dict() for rank, d in destinations.items()
            },
        }
        task = asyncio.create_task(coordinator.retrieve([request]))
        assert await asyncio.to_thread(entered.wait, 5)
        try:
            identities = [
                v.to_dict()
                for v in coordinator.deliveries["blocked"].write_identities.values()
            ]
            reply = await asyncio.wait_for(
                coordinator.fence_retrieval("blocked", identities), timeout=1
            )
            assert reply["fenced"] is False
        finally:
            resume.set()
            await task
        for rank, store in enumerate(stores):
            delivery = store.entries[manifest.key].deliveries[f"blocked:d{rank}"]
            if delivery.transfer_handle is not None:
                engine.finish(delivery.transfer_handle)
            store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("sync_fails", [True, False])
def test_cancel_before_submit_waits_for_local_gpu_packing_fence(
    monkeypatch, sync_fails
):
    # Exercise CUDA ordering branches with a controlled boundary, not real GPU
    # execution. The underlying registration and allocation remain real CPU data.
    from types import SimpleNamespace

    engine, store, entry = ready_store()
    delivery, _ = reserve(engine, store, entry)
    store.pool = SimpleNamespace(is_cuda=True, device="test-cuda-device")
    entered, resume = threading.Event(), threading.Event()

    def synchronize(device):
        assert device == "test-cuda-device"
        entered.set()
        assert resume.wait(5)
        if sync_fails:
            raise RuntimeError("GPU completion unknown")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    with ThreadPoolExecutor(max_workers=1) as executor:
        submitting = executor.submit(
            store.start_delivery, entry.key, delivery.delivery_id
        )
        assert entered.wait(5)
        try:
            store.cancel_entry(entry.key, "cancel during packing")
            assert not entry.resources_released
            assert store.fence_write(delivery.authorization.identity)["fenced"] is False
        finally:
            resume.set()
        submitting.result(timeout=5)
    assert not engine.pending
    assert entry.resources_released is (not sync_fails)
    assert store.fence_write(delivery.authorization.identity)["fenced"] is (
        not sync_fails
    )
    if sync_fails:
        assert store.snapshot()["isolated_reason"]


def test_adapter_local_rejection_after_gate_begin_is_safe_failed_delivery():
    engine, store, entry = ready_store()
    delivery, _ = reserve(engine, store, entry)

    def rejected(*args, **kwargs):
        return TransferHandle(
            uuid.uuid4().hex,
            status=TransferStatus.FAILED,
            error="capacity rejection",
            transport_state=TransportState.NOT_SUBMITTED,
        )

    engine.submit_put = rejected
    store.start_delivery(entry.key, delivery.delivery_id)
    assert delivery.state == DeliveryState.FAILED
    assert store.fence_write(delivery.authorization.identity)["fenced"] is True
    store.release_entry(entry.key)
    assert entry.resources_released
    store.close()
