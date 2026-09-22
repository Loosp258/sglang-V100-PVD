"""Real V store/HTTP/allocator with fake payload; not native RDMA evidence."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch
from aiohttp.test_utils import TestClient, TestServer
from sglang.srt.disaggregation.pvd.control_server import (
    HttpShardClient,
    create_shard_app,
)
from sglang.srt.disaggregation.pvd.full_kv_fanin import FullKVFanInReceiver
from sglang.srt.disaggregation.pvd.full_kv_fanin_plan import plan_fingerprint
from sglang.srt.disaggregation.pvd.protocol import (
    PVD_TRANSFER_LIFECYCLE_PROTOCOL,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.request_state import DeliveryState
from sglang.srt.disaggregation.pvd.server import (
    _create_store,
    _validate_args,
    build_parser,
)
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
)
from sglang.srt.disaggregation.pvd.vector_store import EntryConflictError, VectorKVStore
from test_pvd_core import ENTRY_BYTES, PAGE_BYTES, make_manifest
from test_pvd_fanin_writer import DelayedEngine


@contextmanager
def setup(*, engine=None, enabled=True, committed=True):
    engine = engine or FakeTransferEngine()
    original = make_manifest()
    manifest = replace(
        original, shards=[replace(s, rail="mlx5_7") for s in original.shards]
    )
    key = manifest.key
    stores = [
        VectorKVStore(
            rank=rank,
            world_size=2,
            rail="mlx5_7",
            device="cpu",
            total_pages=8,
            page_bytes=PAGE_BYTES,
            endpoint=f"V{rank}",
            transfer_engine=engine,
            allow_cpu_for_tests=True,
            full_kv_fanin_max_slices=64 if enabled else None,
            full_kv_fanin_max_inflight=2 if enabled else None,
        )
        for rank in (0, 1)
    ]
    entries, raw = [], []
    for rank, store in enumerate(stores):
        entries.append(store.create_entry(manifest))
        store.begin_p_write(key)
        data = (torch.arange(ENTRY_BYTES) + rank * 40).to(torch.uint8)
        store.pool[:ENTRY_BYTES] = data
        raw.append(data)
        if committed:
            store.commit_p_write(key, ENTRY_BYTES)
    storage = manifest.layout
    compute = replace(
        storage,
        tp_size=1,
        kv_heads_per_rank=4,
        extra={
            **storage.extra,
            "component_bytes_per_token": [8],
            "component_token_shapes": [[4, 1]],
        },
    )
    target = engine.register_memory(
        torch.zeros(ENTRY_BYTES * 2, dtype=torch.uint8),
        endpoint="D",
        rank=0,
        rail="mlx5_7",
        metadata={"pvd_receiver_epoch": "D-process", "pvd_generation": "allocation"},
    )
    guard = ResourceGuard(target, lambda: engine.release_memory(target))
    receiver = FullKVFanInReceiver(
        key=key,
        delivery_id="fanin",
        registration=target,
        guard=guard,
        storage=storage,
        compute=compute,
        token_count=8,
        max_slices=64,
    )
    wire = receiver.publish()
    c = NS(**locals())
    try:
        yield c
    finally:
        # All transport here is process-local fake. Unknown cases deliberately
        # retain production guards; explicit test cleanup is not a recovery API.
        for store in stores:
            store.cancel_entry(key, "test cleanup")
        if isinstance(engine, DelayedEngine):
            for handle, *_ in engine.submitted:
                if not handle.transport_state.is_locally_safe_to_release:
                    engine.complete(handle)
        for store in stores:
            store.progress_transfers()
            store.close()
            engine.release_memory(store.registration)
        engine.release_memory(target)


def identity(c, rank):
    return WriteIdentity(
        protocol=PVD_TRANSFER_LIFECYCLE_PROTOCOL,
        sender_epoch=c.stores[rank].worker_epoch,
        receiver_epoch="D-process",
        transfer_id=f"fanin:d0:v{rank}",
        region_id=c.target.descriptor.region_id,
        generation="allocation",
        shard_rank=0,
        key=c.key,
    )


def test_real_store_two_source_delivery_reuses_entry_after_ack():
    with setup() as c:
        ds = [s.reserve_fanin_delivery(c.wire) for s in c.stores]
        c.receiver.adopt({r: d.authorization.identity for r, d in enumerate(ds)})
        c.guard.request_release()
        for store, d in zip(c.stores, ds):
            store.start_delivery(c.key, d.delivery_id)
            for _ in range(10):
                if d.state == DeliveryState.DELIVERED:
                    break
                store.poll_delivery(c.key, d.delivery_id)
            assert d.state == DeliveryState.DELIVERED
            c.receiver.observe(d.to_dict()["fanin_proof"])
            assert store.reserve_fanin_delivery(c.wire) is d
            store.ack_delivery(c.key, d.delivery_id)
            assert store.entries[c.key].active_delivery_count == 0
            assert not store.entries[c.key].resources_released
        expected = torch.cat([r.reshape(8, 4) for r in c.raw], dim=1).reshape(-1)
        assert c.receiver.ready and torch.equal(c.target.buffer, expected)
        c.receiver.close()
        for store in c.stores:
            store.release_entry(c.key)
            assert store.entries[c.key].resources_released


def test_partial_native_writes_delay_actual_allocator_reuse():
    with setup(engine=DelayedEngine()) as c:
        s = c.stores[0]
        d = s.reserve_fanin_delivery(c.wire)
        s.start_delivery(c.key, d.delivery_id)
        before = s.allocator.available_pages
        s.cancel_entry(c.key, "client cancelled")
        assert (
            s.allocator.available_pages == before
            and not c.entries[0].resources_released
        )
        assert not s.fence_fanin_delivery(c.wire, d.authorization.identity)["fenced"]
        for h, *_ in c.engine.submitted:
            c.engine.complete(h)
        proof = s.fence_fanin_delivery(c.wire, d.authorization.identity)
        assert proof["fenced"] and proof["transport_state"] == "terminal_failed"
        assert c.entries[0].resources_released and s.allocator.available_pages == 8


def test_absent_fence_atomically_prevents_late_reservation():
    with setup() as c:
        s = c.stores[0]
        proof = s.fence_fanin_delivery(c.wire, identity(c, 0))
        assert proof["fenced"] and proof["transport_state"] == "not_submitted"
        with pytest.raises(EntryConflictError, match="fenced"):
            s.reserve_fanin_delivery(c.wire)
        assert not c.entries[0].deliveries


def test_concurrent_reserve_has_one_writer_and_one_allocation_pin():
    with setup() as c, ThreadPoolExecutor(4) as pool:
        s, entry = c.stores[0], c.entries[0]
        ds = list(pool.map(lambda _: s.reserve_fanin_delivery(c.wire), range(8)))
        assert all(d is ds[0] for d in ds)
        assert (
            entry.active_delivery_count == 1
            and len(entry.allocation_guard._owners) == 1
        )


def test_waiting_source_never_sends_before_p_upload_commit():
    with setup(committed=False) as c:
        s = c.stores[0]
        d = s.reserve_fanin_delivery(c.wire)
        assert d.state == DeliveryState.WAITING_SOURCE
        s.start_delivery(c.key, d.delivery_id)
        s.progress_transfers()
        assert c.engine.total_put_bytes == 0
        s.commit_p_write(c.key, ENTRY_BYTES)
        s.start_delivery(c.key, d.delivery_id)
        assert c.engine.total_put_bytes > 0
        # Complete the peer's controlled upload for fixture cleanup.
        c.stores[1].commit_p_write(c.key, ENTRY_BYTES)


def test_unknown_isolates_store_and_retains_source_pages():
    class Raises(FakeTransferEngine):
        def submit_put(self, *a, **kw):
            raise RuntimeError("native submit ambiguous")

    with setup(engine=Raises()) as c:
        s = c.stores[0]
        d = s.reserve_fanin_delivery(c.wire)
        s.start_delivery(c.key, d.delivery_id)
        assert s._isolated_reason and d.state == DeliveryState.FAILED
        s.cancel_entry(c.key, "stop")
        assert not c.entries[0].resources_released
        assert not s.fence_fanin_delivery(c.wire, d.authorization.identity)["fenced"]
        with pytest.raises(EntryConflictError, match="isolated"):
            s.reserve_fanin_delivery(c.wire)


def test_same_id_cannot_switch_plan_or_legacy_protocol():
    with setup() as c:
        s = c.stores[0]
        d = s.reserve_fanin_delivery(c.wire)
        changed = {
            **c.wire,
            "destination": {**c.wire["destination"], "endpoint": "different-D"},
        }
        changed.pop("plan_fingerprint")
        changed["plan_fingerprint"] = plan_fingerprint(changed)
        with pytest.raises(EntryConflictError, match="different fan-in plan"):
            s.reserve_fanin_delivery(changed)
        with pytest.raises(EntryConflictError, match="plan mismatch"):
            s.fence_fanin_delivery(changed, d.authorization.identity)
        with pytest.raises(EntryConflictError, match="belongs to full-KV"):
            s.reserve_delivery(c.key, d.delivery_id, d.destination)
        assert not d.authorization.fence(d.authorization.identity)["fenced"]


def test_shard_http_reserve_start_poll_ack_and_fence():
    async def run(c):
        server = TestServer(create_shard_app(c.stores[0]))
        await server.start_server()
        client = HttpShardClient(0, str(server.make_url("")))
        try:
            r = await client.reserve_fanin_delivery(c.wire)
            await client.start_delivery(c.key, r["delivery_id"])
            for _ in range(10):
                r = await client.poll_delivery(c.key, r["delivery_id"])
                if r["state"] == "delivered":
                    break
            assert r["state"] == "delivered" and r["fanin_proof"]["fenced"]
            await client.ack_delivery(c.key, r["delivery_id"])
            proof = await client.fence_fanin_delivery(
                c.wire, WriteIdentity.from_dict(r["write_identity"])
            )
            assert proof["fenced"] and proof["source_rank"] == 0
        finally:
            await client.close()
            await server.close()

    with setup() as c:
        asyncio.run(run(c))


def test_disabled_route_refuses_without_reserving():
    async def run(c):
        async with TestClient(TestServer(create_shard_app(c.stores[0]))) as client:
            response = await client.post(
                "/internal/v1/fanin/reserve", json={"manifest": c.wire}
            )
            assert response.status == 409
            assert "disabled" in (await response.json())["error"]
            assert c.entries[0].active_delivery_count == 0

    with setup(enabled=False) as c:
        asyncio.run(run(c))


def test_launcher_requires_both_bounds_before_constructing_store():
    parser = build_parser()
    flags = [
        "--advertise-host",
        "127.0.0.1",
        "--transfer-staging-budget-bytes",
        "4096",
        "--transfer-max-inflight",
        "4",
        "--total-pages",
        "8",
        "--page-bytes",
        "16",
        "--transfer-backend",
        "fake",
        "--allow-fake-transport",
        "--allow-cpu-for-tests",
        "--no-strict-rdma-preflight",
        "--rails",
        "mlx5_7,mlx5_7",
        "--full-kv-fanin-max-slices",
        "64",
    ]
    args = parser.parse_args(flags)
    with pytest.raises(ValueError, match="both full-KV"):
        _validate_args(args)
    args = parser.parse_args(flags + ["--full-kv-fanin-max-inflight", "2"])
    rails = _validate_args(args)
    store, _ = _create_store(args, rank=0, local_rank=0, rails=rails)
    try:
        assert store._fanin_max_slices == 64 and store._fanin_max_inflight == 2
    finally:
        store.close()


@pytest.mark.parametrize("cause", ["timeout", "shutdown"])
def test_timeout_and_shutdown_keep_pending_native_sources(cause):
    with setup(engine=DelayedEngine()) as c:
        s = c.stores[0]
        d = s.reserve_fanin_delivery(c.wire)
        s.start_delivery(c.key, d.delivery_id)
        if cause == "timeout":
            d.deadline = 0
            s.reap_expired()
            s.release_entry(c.key)
        else:
            s.close()
        assert not c.entries[0].resources_released
        assert not d.to_dict()["fanin_proof"]["fenced"]
        for h, *_ in c.engine.submitted:
            c.engine.complete(h)
        s.progress_transfers()
        assert c.entries[0].resources_released


def test_absent_fence_refuses_foreign_epoch_without_blocking_valid_reserve():
    with setup() as c:
        s = c.stores[0]
        with pytest.raises(EntryConflictError, match="sender epoch"):
            s.fence_fanin_delivery(
                c.wire, replace(identity(c, 0), sender_epoch="old-process")
            )
        assert not s._absent_write_fences and not s._fenced_deliveries
        assert s.reserve_fanin_delivery(c.wire).state == DeliveryState.D_RESERVED


def test_two_deliveries_share_one_entry_but_cancel_independently():
    with setup(engine=DelayedEngine()) as c:
        s, entry = c.stores[0], c.entries[0]
        first = s.reserve_fanin_delivery(c.wire)
        wire = {**c.wire, "delivery_id": "second"}
        wire.pop("plan_fingerprint")
        wire["plan_fingerprint"] = plan_fingerprint(wire)
        second = s.reserve_fanin_delivery(wire)
        assert entry.active_delivery_count == 2
        s.start_delivery(c.key, first.delivery_id)
        s.start_delivery(c.key, second.delivery_id)
        s.cancel_delivery(c.key, first.delivery_id, "cancel first")
        assert entry.active_delivery_count == 1
        assert second.state == DeliveryState.V_WRITING
        assert not second.authorization.fence(second.authorization.identity)["fenced"]
        assert not entry.resources_released
