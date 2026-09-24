"""Real localhost shard HTTP + owned CPU buffers; delayed copies, not RDMA."""

import asyncio
import copy
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from aiohttp.test_utils import TestServer
from sglang.srt.disaggregation.pvd.control_server import (
    HttpShardClient,
    create_shard_app,
)
from sglang.srt.disaggregation.pvd.protocol import KVEntryKey, KVEntryManifest
from sglang.srt.disaggregation.pvd.sparse_install import CPUInstallGroup
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVPayload
from sglang.srt.disaggregation.pvd.sparse_receiver import (
    SparseReceiveError,
    SparseReceiveRegistry,
)
from sglang.srt.disaggregation.pvd.sparse_working_set import CPUSparseWorkingSet
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_prompt_index import make_store, manager
from test_pvd_prompt_vectors import FakePool, pack_shard, storage_layout
from test_pvd_sparse_delivery import manifest_for


@asynccontextmanager
async def receiving():
    pool = FakePool(dtype=torch.float32)
    layout = storage_layout(pool)
    packed, shard, _ = pack_shard(pool, layout, rank=0, prompt_tokens=8)
    entry = KVEntryManifest(
        key=KVEntryKey.new("model", "prompt"),
        layout=layout,
        prompt_token_count=8,
        shards=[pack_shard(pool, layout, rank=r, prompt_tokens=8)[1] for r in (0, 1)],
    )
    index = manager(budget=TransferBudget(1 << 20, 32))
    store = make_store(entry, index=index)
    engine = store.transfer_engine
    engine.lifecycle_manager = SimpleNamespace(budget=TransferBudget(65536, 32))
    stored = store.create_entry(entry)
    store.begin_p_write(entry.key)
    offset = stored.allocation.start_page * store.page_bytes
    store.pool[offset : offset + shard.expected_bytes] = packed.tensor
    store.commit_p_write(entry.key, shard.expected_bytes)
    store.progress_prompt_indexes()
    manifest = manifest_for(index, entry.key, layout)
    bank_budget = TransferBudget(65536, 32)
    bank = CPUSparseWorkingSet(
        request_id="consumer",
        incarnation="inc",
        entry_transfer_id=entry.key.transfer_id,
        layout_fingerprint=layout.fingerprint,
        expected_groups=((0, 0), (1, 0)),
        prompt_tokens=8,
        head_dim=layout.head_dim,
        max_union_tokens=8,
        budget=bank_budget,
    )
    group = CPUInstallGroup({0: bank}, interval=4, lead_tokens=1)
    initial = group.begin(0)
    group.stage(
        initial,
        0,
        tuple(
            SparseKVPayload(
                replace(
                    s,
                    operation_id=initial.operation_id,
                    target_tokens=0,
                    token_ids=tuple(range(8)),
                ),
                torch.stack(
                    (pool.k_buffer[s.layer][:8, 0], pool.v_buffer[s.layer][:8, 0])
                ),
            )
            for s in manifest.specs
        ),
    )
    assert group.try_install(initial, {0: 0})
    epoch = group.begin(3)
    manifest = replace(
        manifest,
        specs=tuple(
            replace(s, operation_id=epoch.operation_id) for s in manifest.specs
        ),
    )
    registry = SparseReceiveRegistry(
        engine, TransferBudget(65536, 32), receiver_epoch="D-inc"
    )
    async with TestServer(create_shard_app(store)) as server:
        client = HttpShardClient(0, str(server.make_url("")))
        record = registry.prepare(
            manifest,
            key=entry.key,
            rank=0,
            rail=store.rail,
            endpoint="D",
            sender_epoch=store.worker_epoch,
            client=client,
        )
        case = SimpleNamespace(
            store=store,
            engine=engine,
            entry=entry,
            pool=pool,
            manifest=manifest,
            group=group,
            epoch=epoch,
            registry=registry,
            record=record,
            client=client,
        )
        try:
            yield case
        finally:
            # Test teardown drains controlled transport, never guesses terminal.
            for delivery in store.entries[entry.key].deliveries.values():
                handle = delivery.transfer_handle
                if handle and not handle.transport_state.is_locally_safe_to_release:
                    engine.finish(handle)
            store.progress_transfers()
            await registry.close()
            group.close()
            store.close()
            await client.close()
            assert bank_budget.snapshot()["used_staging_bytes"] == 0


def finish(case):
    delivery = case.store.entries[case.entry.key].deliveries[
        case.record.identity.transfer_id
    ]
    case.engine.finish(delivery.transfer_handle)
    return delivery


def test_http_delivery_install_ack_and_free_are_distinct():
    async def run():
        async with receiving() as c:
            r = c.record
            assert not r.snapshot()["source_started"]
            assert not await r.start()
            assert r.snapshot()["source_started"]
            with pytest.raises(SparseReceiveError, match="successful delivery"):
                r.stage(c.group, c.epoch)
            assert not await r.poll()
            finish(c)
            assert await r.poll()
            r.stage(c.group, c.epoch)
            with pytest.raises(SparseReceiveError, match="all ranks must install"):
                await r.ack()
            assert not c.group.coordinator.can_decode(4)
            assert c.group.try_install(c.epoch, {0: 4})
            await r.ack()
            assert r.snapshot()["acknowledged"]
            assert await r.close()
            assert c.registry.snapshot() == {}
            assert c.registry.budget.snapshot()["used_staging_bytes"] == 0
            # Installed banks own independent copies, not freed receive views.
            with c.group.read(0, 4) as groups:
                for (layer, head), (spec, data) in groups.items():
                    expected = torch.stack(
                        (
                            c.pool.k_buffer[layer][list(spec.token_ids), head],
                            c.pool.v_buffer[layer][list(spec.token_ids), head],
                        )
                    )
                    torch.testing.assert_close(data, expected, rtol=0, atol=0)

    asyncio.run(run())


def test_cancel_and_late_write_hold_destination_until_exact_fence():
    async def run():
        async with receiving() as c:
            await c.record.start()
            region = c.record.identity.region_id
            assert not await c.record.close()
            assert region not in c.engine.released
            assert (
                c.registry.budget.snapshot()["used_staging_bytes"] == c.manifest.nbytes
            )
            finish(c)  # abort does not prevent this late native copy
            assert await c.record.close()
            assert c.engine.released.count(region) == 1
            assert not c.group.coordinator.can_decode(4)

    asyncio.run(run())


@pytest.mark.parametrize(
    "field",
    [
        "sender_epoch",
        "receiver_epoch",
        "transfer_id",
        "region_id",
        "generation",
        "shard_rank",
        "key",
        "sparse_fingerprint",
        "destination",
        "transferred_bytes",
        "transport_state",
        "fenced",
    ],
)
def test_wrong_completion_cannot_make_buffer_readable(field, monkeypatch):
    async def run():
        async with receiving() as c:
            await c.record.start()
            finish(c)
            good = await c.client.poll_delivery(
                c.entry.key, c.record.identity.transfer_id
            )
            bad = copy.deepcopy(good)
            if field in c.record.identity.__dict__:
                bad["write_fence"][field] = 7 if field == "shard_rank" else "wrong"
            elif field == "fenced":
                bad["write_fence"][field] = False
            else:
                bad[field] = 1 if field == "transferred_bytes" else "wrong"
            original = c.client.poll_delivery

            async def reply(*_):
                return bad

            monkeypatch.setattr(c.client, "poll_delivery", reply)
            with pytest.raises(ValueError):
                await c.record.poll()
            assert not c.record.snapshot()["ready"]
            with pytest.raises(SparseReceiveError, match="successful delivery"):
                c.record.stage(c.group, c.epoch)
            monkeypatch.setattr(c.client, "poll_delivery", original)
            assert await c.record.poll()

    asyncio.run(run())


@pytest.mark.parametrize("method", ["reserve_delivery", "start_delivery"])
def test_lost_response_keeps_published_destination_owned(method, monkeypatch):
    async def run():
        async with receiving() as c:
            original = getattr(c.client, method)

            async def lost(*args):
                await original(*args)
                raise TimeoutError("lost response")

            monkeypatch.setattr(c.client, method, lost)
            with pytest.raises(TimeoutError):
                await c.record.start()
            assert c.record.snapshot()["published"]
            assert not c.record.snapshot()["source_started"]
            assert (
                c.registry.budget.snapshot()["used_staging_bytes"] == c.manifest.nbytes
            )
            if method == "start_delivery":
                assert not await c.record.close()
                finish(c)
            assert await c.record.close()

    asyncio.run(run())


def test_lost_ack_can_retry_without_reinstalling(monkeypatch):
    async def run():
        async with receiving() as c:
            await c.record.start()
            finish(c)
            await c.record.poll()
            c.record.stage(c.group, c.epoch)
            assert c.group.try_install(c.epoch, {0: 4})
            original = c.client.ack_delivery

            async def lost(*args):
                await original(*args)
                raise TimeoutError("lost ACK")

            monkeypatch.setattr(c.client, "ack_delivery", lost)
            with pytest.raises(TimeoutError):
                await c.record.ack()
            assert c.record.snapshot()["installed"]
            c.group.begin(7)  # retry uses a latched exact completion, not new epoch
            monkeypatch.setattr(c.client, "ack_delivery", original)
            await c.record.ack()
            assert c.record.snapshot()["acknowledged"]

    asyncio.run(run())


def test_unregistration_failure_retains_tensor_and_budget_for_retry(monkeypatch):
    async def run():
        async with receiving() as c:
            original = c.engine.release_memory

            def fail(_):
                raise RuntimeError("unregister failed")

            monkeypatch.setattr(c.engine, "release_memory", fail)
            with pytest.raises(RuntimeError, match="unregister failed"):
                await c.record.close()
            assert c.registry.snapshot()
            assert c.record._buffer is not None
            assert (
                c.registry.budget.snapshot()["used_staging_bytes"] == c.manifest.nbytes
            )
            monkeypatch.setattr(c.engine, "release_memory", original)
            assert await c.record.close()

    asyncio.run(run())


def test_registration_failure_remains_visible_and_charged(monkeypatch):
    async def run():
        async with receiving() as c:

            def fail(*_, **__):
                raise RuntimeError("unknown registration")

            monkeypatch.setattr(c.engine, "register_memory", fail)
            with pytest.raises(RuntimeError, match="unknown registration"):
                c.registry.prepare(
                    c.manifest,
                    key=c.entry.key,
                    rank=0,
                    rail=c.store.rail,
                    endpoint="D",
                    sender_epoch=c.store.worker_epoch,
                    client=c.client,
                )
            assert len(c.registry.snapshot()) == 2
            assert (
                sum(s["registration_unknown"] for s in c.registry.snapshot().values())
                == 1
            )
            errors = await c.registry.close()
            assert len(errors) == 1
            assert (
                c.registry.budget.snapshot()["used_staging_bytes"] == c.manifest.nbytes
            )
            # Unknown MR is intentionally held until process teardown, not freed.

    asyncio.run(run())


def test_absence_without_a_successful_fence_reply_keeps_buffer_owned(monkeypatch):
    async def run():
        async with receiving() as c:

            async def lost(*_):
                raise TimeoutError("submission outcome unknown")

            monkeypatch.setattr(c.client, "reserve_delivery", lost)
            monkeypatch.setattr(c.client, "fence_delivery", lost)
            with pytest.raises(TimeoutError):
                await c.record.start()
            errors = await c.registry.close()  # absence/timeout is still not proof
            assert c.record.identity.transfer_id in errors
            assert c.record._buffer is not None
            assert not c.record.snapshot()["fenced"]
            assert (
                c.registry.budget.snapshot()["used_staging_bytes"] == c.manifest.nbytes
            )

    asyncio.run(run())


def test_foreign_install_epoch_cannot_stage_or_ack():
    async def run():
        async with receiving() as c:
            await c.record.start()
            finish(c)
            await c.record.poll()
            with pytest.raises(ValueError, match="stale or foreign"):
                c.record.stage(c.group, replace(c.epoch, operation_id="old-operation"))
            with pytest.raises(SparseReceiveError, match="all ranks must install"):
                await c.record.ack()
            assert not c.record.snapshot()["staged"]
            c.record.stage(c.group, c.epoch)
            with pytest.raises(SparseReceiveError, match="already staged"):
                c.record.stage(c.group, c.epoch)

    asyncio.run(run())
