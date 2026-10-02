"""Real HTTP/FP16 bytes with declared CPU CUDA-policy stubs; not RDMA proof."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from aiohttp.test_utils import TestServer
from sglang.srt.disaggregation.pvd.control_server import HttpShardClient, create_shard_app
from sglang.srt.disaggregation.pvd.coordinator import CoordinatorError
from sglang.srt.disaggregation.pvd.oasis_receive_slots import OasisReceiveSlotPool
from sglang.srt.disaggregation.pvd.oasis_transport import OasisCUDAReceiveRegistry
from sglang.srt.disaggregation.pvd.protocol import KVEntryKey, KVEntryManifest
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
from sglang.srt.disaggregation.pvd.sparse_receiver import SparseReceiveError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget, TransportState
from test_pvd_prompt_index import ident, make_store, manager
from test_pvd_prompt_vectors import FakePool, pack_shard, storage_layout


@asynccontextmanager
async def slot_case(monkeypatch, *, combine=False):
    # Actual target dtype/head width and two heads in one layer. Only the
    # local CUDA ordering/copy API policy is stubbed; tensors/copies/HTTP/store
    # validation remain real. No production CPU fallback is introduced.
    source = FakePool(layers=2, heads=4, dim=128, dtype=torch.float16)
    layout = storage_layout(source)
    packed, shard, _ = pack_shard(source, layout, rank=0, prompt_tokens=8)
    entry = KVEntryManifest(
        KVEntryKey.new("model", "slot-prompt"), layout, 8,
        [pack_shard(source, layout, rank=r, prompt_tokens=8)[1] for r in (0, 1)],
    )
    index = manager(budget=TransferBudget(1 << 20, 32))
    store = make_store(entry, index=index)
    engine = store.transfer_engine
    engine.lifecycle_manager = SimpleNamespace(budget=TransferBudget(1 << 20, 32))
    registrations = [store.registration]
    released_objects = []
    register, release = engine.register_memory, engine.release_memory

    def checked_register(*args, **kwargs):
        original = register(*args, **kwargs)
        registrations.append(original)
        return original

    def checked_release(registration):
        assert any(registration is original for original in registrations), (
            "logical receive alias passed to native unregister"
        )
        released_objects.append(registration)
        return release(registration)

    monkeypatch.setattr(engine, "register_memory", checked_register)
    monkeypatch.setattr(engine, "release_memory", checked_release)
    stored = store.create_entry(entry)
    store.begin_p_write(entry.key)
    offset = stored.allocation.start_page * store.page_bytes
    store.pool[offset : offset + shard.expected_bytes] = packed.tensor
    store.commit_p_write(entry.key, shard.expected_bytes)
    store.progress_prompt_indexes()
    result = index.search(ident(entry.key.transfer_id),
                          queries=torch.ones(1, 128), top_k=1)
    spec = SparseKVSpec("req", "inc", "op0", 1, entry.key.transfer_id,
                       result.index_version, result.id_mapping_version,
                       layout.fingerprint, 0, 0, (3, 0, 7))
    manifest = SparseDeliveryManifest((spec, replace(spec, kv_head=1, token_ids=(1, 4))),
                                     "torch.float16", 128)
    budget = TransferBudget(1 << 20, 16)
    pool = OasisReceiveSlotPool(engine, budget, device="cuda:0", receiver_epoch="D-slots",
                               slots_per_rank=2, capacity_bytes=4096)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    registry = OasisCUDAReceiveRegistry(engine, budget, receiver_epoch="D-slots",
        device="cuda:0", combine_reserve_start=combine, receive_pool=pool)
    # Declare the CPU policy explicitly after real constructor compatibility.
    pool.device = registry.device = torch.device("cpu")
    events = []
    registry.ordering = SimpleNamespace(
        prepare=lambda buffer: events.append(("prepare", buffer.data_ptr())),
        after_remote_write=lambda registration: events.append(
            ("remote_order", registration.descriptor.region_id)),
    )
    empty_like = torch.empty_like

    def cpu_empty_like(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty_like(*args, **kwargs)

    monkeypatch.setattr(torch, "empty_like", cpu_empty_like)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(
        synchronize=lambda: events.append(("copy_fence", str(device)))))
    async with TestServer(create_shard_app(store)) as server:
        client = HttpShardClient(0, str(server.make_url("")))
        case = SimpleNamespace(source=source, entry=entry, store=store, engine=engine,
            manifest=manifest, pool=pool, registry=registry, client=client,
            budget=budget, events=events, registrations=registrations,
            released_objects=released_objects)
        try:
            yield case
        finally:
            # Controlled fake native completions are real terminal evidence for
            # these CPU copies; sticky local UNKNOWN remains quarantined.
            for d in store.entries[entry.key].deliveries.values():
                handle = d.transfer_handle
                if handle and not handle.transport_state.is_locally_safe_to_release:
                    engine.finish(handle)
            store.progress_transfers()
            await registry.close()
            try:
                pool.close()
            except SparseReceiveError:
                assert pool.snapshot()["leased_slots"] or pool.snapshot()["unknown_slots"]
            store.close()
            await client.close()


def prepare(c, *, operation="op0", counts=None):
    manifest = replace(c.manifest, specs=tuple(
        replace(s, operation_id=operation,
                token_ids=s.token_ids if counts is None else tuple(range(counts[i])))
        for i, s in enumerate(c.manifest.specs)))
    return c.registry.prepare(manifest, key=c.entry.key, rank=0, rail=c.store.rail,
                             endpoint="D", sender_epoch=c.store.worker_epoch,
                             client=c.client)


def writer(c, record):
    return c.store.entries[c.entry.key].deliveries[record.identity.transfer_id]


async def ready(c, record):
    assert not await record.start()
    c.engine.finish(writer(c, record).transfer_handle)
    assert await record.poll()


@pytest.mark.parametrize("combine", [False, True])
def test_ordered_installed_ack_close_returns_exact_lease_and_reuses_generation(monkeypatch, combine):
    async def run():
        async with slot_case(monkeypatch, combine=combine) as c:
            previous = None
            for operation, counts in (("op0", None), ("op1", (1, 1))):
                r = prepare(c, operation=operation, counts=counts)
                lease = r._slot_lease
                physical = c.pool._slots[0].registration
                assert r._registration is lease.registration and r._registration is not physical
                assert r._registration.descriptor.length == r.manifest.nbytes
                assert c.pool.snapshot()["leased_slots"] == 1
                assert c.budget.snapshot()["used_staging_bytes"] == 4096
                assert c.budget.snapshot()["used_inflight"] == 1
                if previous is not None:
                    assert r.identity.region_id == previous.region_id
                    assert r.identity.generation != previous.generation
                    assert r.identity.transfer_id != previous.transfer_id
                    with pytest.raises(ValueError, match="generation"):
                        previous.validate_destination(r._registration.descriptor)
                await ready(c, r)
                cache = {0: {}, 1: {}}
                r.copy_to_cache(cache)
                assert r.snapshot()["local_ordering_complete"] and r.snapshot()["installed"]
                assert not r._cache_copy_owners
                for s in r.manifest.specs:
                    for token in s.token_ids:
                        expected = torch.stack((c.source.k_buffer[s.layer][token, s.kv_head],
                                                c.source.v_buffer[s.layer][token, s.kv_head]))
                        torch.testing.assert_close(cache[s.kv_head][token], expected, rtol=0, atol=0)
                await r.ack()
                assert await r.close()
                assert r._slot_lease is r._registration is r._buffer is None
                assert lease.buffer is lease.registration is None
                assert c.pool.snapshot()["leased_slots"] == 0
                assert c.pool.snapshot()["physical_slots"] == 1
                assert c.budget.snapshot()["used_staging_bytes"] == 4096
                assert c.budget.snapshot()["used_inflight"] == 0
                assert not any(physical is released for released in c.released_objects)
                previous = r.identity
            assert c.pool.snapshot()["physical_register_calls"] == 1
            c.pool.close()
            assert sum(physical is released for released in c.released_objects) == 1
            assert c.budget.snapshot()["used_staging_bytes"] == 0
            assert not c.registry.snapshot()

    asyncio.run(run())


def test_unpublished_close_returns_idle_lease_without_fence_or_unregister(monkeypatch):
    async def run():
        async with slot_case(monkeypatch) as c:
            r = prepare(c)
            physical = c.pool._slots[0].registration

            async def never_published(*args):
                raise AssertionError("unpublished slot needs no remote RPC")

            monkeypatch.setattr(c.client, "fence_delivery", never_published)
            assert await r.close()
            assert not c.store.entries[c.entry.key].deliveries
            assert c.pool.snapshot()["returned_leases"] == 1
            assert not any(physical is released for released in c.released_objects)
            c.pool.close()
            assert sum(physical is released for released in c.released_objects) == 1

    asyncio.run(run())


@pytest.mark.parametrize("combine", [False, True])
def test_lost_start_reply_retains_slot_until_late_write_fenced_and_no_cache_copy(monkeypatch, combine):
    async def run():
        async with slot_case(monkeypatch, combine=combine) as c:
            r = prepare(c)
            method = "reserve_and_start_delivery" if combine else "start_delivery"
            start = getattr(c.client, method)

            async def lost(*args):
                await start(*args)
                raise TimeoutError("lost start reply after native submission")

            monkeypatch.setattr(c.client, method, lost)
            with pytest.raises(TimeoutError):
                await r.start()
            assert not r.snapshot()["source_started"]
            assert not await r.close()
            assert c.pool.snapshot()["leased_slots"] == 1
            assert c.budget.snapshot()["used_inflight"] == 1
            c.engine.finish(writer(c, r).transfer_handle)
            assert await r.close()
            assert not r.snapshot()["installed"]
            assert c.pool.snapshot()["returned_leases"] == 1
            assert c.budget.snapshot()["used_inflight"] == 0

    asyncio.run(run())


def test_old_delayed_reserve_start_cannot_write_into_reused_physical_slot(monkeypatch):
    async def run():
        async with slot_case(monkeypatch) as c:
            old = prepare(c)
            old_destination = old._registration.descriptor
            reserve = c.client.reserve_delivery

            async def queued_reserve(*args):
                raise TimeoutError("reservation queued before reaching V")

            monkeypatch.setattr(c.client, "reserve_delivery", queued_reserve)
            with pytest.raises(TimeoutError):
                await old.start()
            assert old.snapshot()["published"]
            # close must collect an exact absent-write tombstone before reuse;
            # the queued reserve below is then forbidden to authorize a write.
            assert await old.close()
            assert old.snapshot()["fenced"]
            monkeypatch.setattr(c.client, "reserve_delivery", reserve)
            current = prepare(c, operation="op1", counts=(1, 2))
            assert current.identity.region_id == old.identity.region_id
            assert current.identity.generation != old.identity.generation
            for operation in (
                lambda: c.client.reserve_delivery(old.identity.key, old.identity.transfer_id, old_destination),
                lambda: c.client.start_delivery(old.identity.key, old.identity.transfer_id),
                lambda: c.client.reserve_and_start_delivery(old.identity.key, old.identity.transfer_id, old_destination),
            ):
                with pytest.raises(CoordinatorError, match="fenced"):
                    await operation()
            assert not c.engine.pending
            await ready(c, current)
            assert c.engine.total_put_bytes == current.manifest.nbytes
            assert await current.close()  # terminal success then cancellation/fence
            assert not current.snapshot()["installed"]
            assert c.pool.snapshot()["returned_leases"] == 2

    asyncio.run(run())


def test_successful_remote_write_cancelled_before_ack_returns_only_after_fence(monkeypatch):
    async def run():
        async with slot_case(monkeypatch) as c:
            r = prepare(c)
            await ready(c, r)
            calls = []
            fence = c.client.fence_delivery

            async def checked_fence(identity):
                assert c.pool.snapshot()["leased_slots"] == 1
                calls.append(identity)
                return await fence(identity)

            monkeypatch.setattr(c.client, "fence_delivery", checked_fence)
            assert r.snapshot()["fenced"] and not r.snapshot()["acknowledged"]
            assert await r.close()
            assert calls == [r.identity]
            assert not r.snapshot()["installed"] and not r.snapshot()["staged"]
            assert r._buffer is r._registration is None
            assert c.pool.snapshot()["leased_slots"] == 0

    asyncio.run(run())


def test_unknown_remote_write_keeps_physical_and_logical_charge(monkeypatch):
    async def run():
        async with slot_case(monkeypatch) as c:
            r = prepare(c)
            await r.start()
            writer(c, r).transfer_handle.transport_state = TransportState.UNKNOWN
            assert not await r.close()
            assert c.pool.snapshot()["leased_slots"] == 1
            assert c.pool.snapshot()["returned_leases"] == 0
            assert c.budget.snapshot()["used_staging_bytes"] == 4096
            assert c.budget.snapshot()["used_inflight"] == 1
            with pytest.raises(SparseReceiveError, match="live or unknown"):
                c.pool.close()
            assert r._slot_lease is not None and r._buffer is not None

    asyncio.run(run())


def test_ordering_unknown_stays_quarantined_after_successful_stream_drain(monkeypatch):
    async def run():
        async with slot_case(monkeypatch) as c:
            r = prepare(c)
            await ready(c, r)

            def fail_ordering(registration):
                raise RuntimeError("GPUDirect ordering outcome unknown")

            c.registry.ordering.after_remote_write = fail_ordering
            with pytest.raises(RuntimeError, match="ordering outcome unknown"):
                r.copy_to_cache({0: {}, 1: {}})
            assert any(event[0] == "copy_fence" for event in c.events)
            assert r.snapshot()["local_completion_unknown"] == "CUDA receive ordering unknown"
            assert not await r.close()
            assert c.pool.snapshot()["leased_slots"] == c.pool.snapshot()["unknown_slots"] == 1
            assert c.pool.snapshot()["returned_leases"] == 0
            with pytest.raises(SparseReceiveError, match="live or unknown"):
                c.pool.close()
            assert c.budget.snapshot()["used_staging_bytes"] == 4096
            assert c.budget.snapshot()["used_inflight"] == 1

    asyncio.run(run())
