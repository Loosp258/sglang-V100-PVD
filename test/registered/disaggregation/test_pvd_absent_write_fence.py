"""Lost reservations need a closed sender gate, not an inference from absence."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from sglang.srt.disaggregation.pvd.protocol import KVEntryKey
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransportState
from sglang.srt.disaggregation.pvd.vector_store import (
    EntryConflictError,
    EntryNotFoundError,
    ResourceExhaustedError,
    VectorKVStore,
)
from test_pvd_sparse_receiver import receiving


def test_absent_reservation_fence_releases_receiver_and_refuses_late_reserve(
    monkeypatch,
):
    async def run():
        async with receiving() as c:
            reserve = c.client.reserve_delivery

            async def lost(*_):
                raise TimeoutError("reserve not received yet")

            monkeypatch.setattr(c.client, "reserve_delivery", lost)
            with pytest.raises(TimeoutError):
                await c.record.start()
            identity = c.record.identity
            destination = c.record._registration.descriptor
            assert c.registry.budget.snapshot()["used_staging_bytes"] > 0
            assert await c.registry.close() == {}
            assert c.registry.budget.snapshot()["used_staging_bytes"] == 0
            assert c.record.snapshot()["closed"]
            assert not c.record.snapshot()["ready"]
            assert identity.region_id in c.engine.released
            with pytest.raises(RuntimeError, match="fenced"):
                await reserve(identity.key, identity.transfer_id, destination)
            assert not c.store.entries[identity.key].deliveries

    asyncio.run(run())


@pytest.mark.parametrize(
    "field", ["receiver_epoch", "region_id", "generation", "shard_rank"]
)
def test_absent_proof_is_identity_complete_and_idempotent(field):
    async def run():
        async with receiving() as c:
            identity = c.record.identity
            proof = c.store.fence_write(identity)
            assert proof == {**identity.to_dict(), "fenced": True}
            assert c.store.fence_write(identity) == proof
            changed = replace(
                identity, **{field: 3 if field == "shard_rank" else "other"}
            )
            with pytest.raises(EntryConflictError, match="identity mismatch"):
                c.store.fence_write(changed)
            assert c.store.snapshot()["absent_write_fences"] == 1
            assert c.store.entries[identity.key].active_delivery_count == 0

    asyncio.run(run())


def test_wrong_worker_epoch_and_unknown_entry_do_not_create_proof():
    async def run():
        async with receiving() as c:
            with pytest.raises(EntryConflictError, match="identity mismatch"):
                c.store.fence_write(replace(c.record.identity, sender_epoch="old-V"))
            with pytest.raises(EntryNotFoundError):
                c.store.fence_write(
                    replace(c.record.identity, key=KVEntryKey.new("model", "unknown"))
                )
            with pytest.raises(EntryConflictError, match="complete write identity"):
                c.store.fence_write(c.record.identity.to_dict())
            assert c.store.snapshot()["absent_write_fences"] == 0
            assert not c.store._fenced_deliveries

    asyncio.run(run())


def test_absent_fence_bound_never_evicts_prior_proofs():
    async def run():
        async with receiving() as c:
            c.store._max_absent_write_fences = 1
            identity = c.record.identity
            proof = c.store.fence_write(identity)
            other = replace(identity, transfer_id="other-delivery")
            with pytest.raises(ResourceExhaustedError, match="capacity exceeded"):
                c.store.fence_write(other)
            assert (other.key, other.transfer_id) not in c.store._fenced_deliveries
            assert c.store.fence_write(identity) == proof  # retry at capacity
            c.store.reap_expired(now=float("inf"))
            assert c.store.fence_write(identity) == proof  # survives Entry release
            with pytest.raises(EntryConflictError, match="fenced"):
                c.store.reserve_delivery(
                    identity.key,
                    identity.transfer_id,
                    c.record._registration.descriptor,
                )
            with pytest.raises(EntryConflictError, match="fenced"):
                c.store.start_delivery(identity.key, identity.transfer_id)
            assert c.store.snapshot()["absent_write_fences"] == 1

    asyncio.run(run())


def test_absent_fence_reply_loss_retries_without_leaking_receiver(monkeypatch):
    async def run():
        async with receiving() as c:
            original = c.client.fence_delivery

            async def lost_reserve(*_):
                raise TimeoutError("lost reserve")

            async def lost_fence(identity):
                await original(identity)  # server closed the gate, reply lost
                raise TimeoutError("lost fence reply")

            monkeypatch.setattr(c.client, "reserve_delivery", lost_reserve)
            monkeypatch.setattr(c.client, "fence_delivery", lost_fence)
            with pytest.raises(TimeoutError):
                await c.record.start()
            assert await c.registry.close()
            assert c.registry.budget.snapshot()["used_staging_bytes"] > 0
            assert c.record.identity.region_id not in c.engine.released
            assert c.store.snapshot()["absent_write_fences"] == 1
            monkeypatch.setattr(c.client, "fence_delivery", original)
            assert await c.registry.close() == {}
            assert c.registry.budget.snapshot()["used_staging_bytes"] == 0

    asyncio.run(run())


@pytest.mark.parametrize("first", ["reserve", "fence"])
def test_reserve_and_fence_share_an_atomic_gate(first, monkeypatch):
    async def run():
        async with receiving() as c:
            identity = c.record.identity
            descriptor = c.record._registration.descriptor
            entered, release, contender = (
                threading.Event(),
                threading.Event(),
                threading.Event(),
            )
            original_entry = c.store._entry

            def blocked_entry(key):
                if not entered.is_set():
                    entered.set()  # both callers must hold store._lock here
                    assert release.wait(5), "test did not release first caller"
                return original_entry(key)

            def reserve():
                try:
                    return c.store.reserve_delivery(
                        identity.key, identity.transfer_id, descriptor
                    )
                except EntryConflictError as exc:
                    assert "fenced" in str(exc)
                    return "refused"

            operations = {
                "reserve": reserve,
                "fence": lambda: c.store.fence_write(identity),
            }
            second = "fence" if first == "reserve" else "reserve"

            def competing():
                contender.set()
                return operations[second]()

            monkeypatch.setattr(c.store, "_entry", blocked_entry)
            with ThreadPoolExecutor(max_workers=2) as executor:
                a = executor.submit(operations[first])
                try:
                    assert entered.wait(5)
                    b = executor.submit(competing)
                    assert contender.wait(5)
                finally:
                    release.set()
                results = {first: a.result(timeout=5), second: b.result(timeout=5)}
            assert results["fence"] == {**identity.to_dict(), "fenced": True}
            if first == "fence":
                assert results["reserve"] == "refused"
                assert not c.store.entries[identity.key].deliveries
            else:
                assert results["reserve"].local_terminal == TransportState.NOT_SUBMITTED
                assert c.store.snapshot()["absent_write_fences"] == 0
            with pytest.raises(EntryConflictError, match="fenced"):
                c.store.start_delivery(identity.key, identity.transfer_id)

    asyncio.run(run())


def test_existing_unknown_write_is_never_reclassified_as_absent():
    async def run():
        async with receiving() as c:
            await c.record.start()
            delivery = c.store.entries[c.entry.key].deliveries[
                c.record.identity.transfer_id
            ]
            delivery.transfer_handle.transport_state = TransportState.UNKNOWN
            assert not c.store.fence_write(c.record.identity)["fenced"]
            assert c.store.snapshot()["absent_write_fences"] == 0
            assert c.record.identity.region_id not in c.engine.released
            assert not await c.record.close()

    asyncio.run(run())


@pytest.mark.parametrize("bound", [0, -1, True, 1.5])
def test_absent_fence_capacity_validated_before_allocating(bound):
    with pytest.raises(ValueError, match="max_absent_write_fences"):
        VectorKVStore(
            rank=0,
            world_size=2,
            rail="rail",
            device="cpu",
            total_pages=1,
            page_bytes=1,
            endpoint="V",
            transfer_engine=None,  # invalid bound must fail before registration
            allow_cpu_for_tests=True,
            max_absent_write_fences=bound,
        )


def test_http_capacity_refusal_retains_d_destination(monkeypatch):
    async def run():
        async with receiving() as c:
            c.store._max_absent_write_fences = 1
            c.store.fence_write(replace(c.record.identity, transfer_id="earlier"))

            async def lost(*_):
                raise TimeoutError("lost reserve")

            monkeypatch.setattr(c.client, "reserve_delivery", lost)
            with pytest.raises(TimeoutError):
                await c.record.start()
            errors = await c.registry.close()
            assert "capacity exceeded" in errors[c.record.identity.transfer_id]
            assert c.record._buffer is not None
            assert (
                c.registry.budget.snapshot()["used_staging_bytes"] == c.manifest.nbytes
            )
            assert c.record.identity.region_id not in c.engine.released
            assert not c.record.snapshot()["fenced"]

    asyncio.run(run())


def test_fence_rank_is_destination_rank_not_source_rank():
    async def run():
        async with receiving() as c:
            identity = replace(c.record.identity, shard_rank=3)
            assert c.store.rank == 0
            assert c.store.fence_write(identity) == {
                **identity.to_dict(),
                "fenced": True,
            }
            # The closed gate covers ALL destinations for this Delivery id.
            with pytest.raises(EntryConflictError, match="fenced"):
                c.store.reserve_delivery(
                    identity.key,
                    identity.transfer_id,
                    c.record._registration.descriptor,
                )

    asyncio.run(run())
