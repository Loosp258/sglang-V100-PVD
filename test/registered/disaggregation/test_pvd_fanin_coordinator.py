"""Real coordinator, stores and fake transport; no native/GPU evidence."""

import asyncio
import copy
import time
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer
from sglang.srt.disaggregation.pvd.control_server import create_coordinator_app
from sglang.srt.disaggregation.pvd.coordinator import (
    CoordinatorError,
    EntryRecord,
    LocalShardClient,
    VectorCoordinator,
)
from sglang.srt.disaggregation.pvd.full_kv_fanin_plan import plan_fingerprint
from sglang.srt.disaggregation.pvd.protocol import WriteIdentity
from sglang.srt.disaggregation.pvd.request_state import EntryState
from sglang.srt.disaggregation.pvd.server import _fanin_coordinator_args
from test_pvd_fanin_store import setup
from test_pvd_fanin_writer import DelayedEngine


def group(c, *, enabled=True, max_records=8):
    parent = VectorCoordinator(
        [LocalShardClient(s) for s in c.stores],
        full_kv_fanin_max_slices=64 if enabled else None,
        full_kv_fanin_max_records=max_records if enabled else None,
    )
    parent.entries[c.key] = EntryRecord(
        c.manifest,
        EntryState.STORED if c.committed else EntryState.P_WRITING,
        time.monotonic(),
        time.monotonic() + 1000,
    )
    epochs = {str(s.rank): s.worker_epoch for s in c.stores}
    return parent, parent.fanin, epochs


async def delivered(actor):
    result = await actor.start("fanin", 0)
    for _ in range(16):
        if result["state"] == "delivered":
            return result
        result = await actor.poll("fanin", 0)
    raise AssertionError(result)


def test_all_writers_then_ack_and_real_receiver():
    async def run():
        with setup() as c:
            parent, actor, epochs = group(c)
            reserved = await actor.reserve(c.wire, epochs)
            c.receiver.adopt(
                {
                    int(r): WriteIdentity.from_dict(i)
                    for r, i in reserved["write_identities"].items()
                }
            )
            assert reserved["state"] == "reserved"
            assert not reserved["fenced"]
            assert parent.entries[c.key].active_delivery_count == 1
            with pytest.raises(CoordinatorError, match="active deliveries"):
                await parent.release_entry(c.key)
            result = await delivered(actor)
            for proof in result["writer_proofs"].values():
                c.receiver.observe(proof)
            assert c.receiver.ready
            assert result["entry_count_held"]
            result = await actor.ack("fanin", 0)
            assert result["state"] == "released"
            assert not result["entry_count_held"]
            assert await actor.ack("fanin", 0) == result
            assert await actor.fence(c.wire, epochs) == result
            assert parent.entries[c.key].active_delivery_count == 0
            await parent.release_entry(c.key)
            c.receiver.close()

    asyncio.run(run())


def test_slow_writer_keeps_mr_and_entry_until_cancel_drains():
    async def run():
        with setup(engine=DelayedEngine()) as c:
            parent, actor, epochs = group(c)
            await actor.reserve(c.wire, epochs)
            await actor.start("fanin", 0)
            for handle, _, remote, _ in c.engine.submitted:
                if ":v0:" in handle.transfer_id:
                    c.engine.complete(handle)
            with pytest.raises(CoordinatorError, match="all V writers"):
                await actor.ack("fanin", 0)
            result = await actor.fence(c.wire, epochs)
            assert not result["fenced"] and result["entry_count_held"]
            assert parent.entries[c.key].active_delivery_count == 1
            for handle, *_ in c.engine.submitted:
                if not handle.transport_state.is_locally_safe_to_release:
                    c.engine.complete(handle)
            result = await actor.poll("fanin", 0)
            assert result["fenced"] and not result["entry_count_held"]
            assert result["state"] == "cancelled"

    asyncio.run(run())


@pytest.mark.parametrize("after_reserve", [False, True])
def test_lost_reserve_response_is_fenced_on_every_source(after_reserve):
    async def run():
        with setup() as c:
            parent, actor, epochs = group(c)
            original = parent.shards[1].reserve_fanin_delivery

            async def lost(*args, **kwargs):
                if after_reserve:
                    await original(*args, **kwargs)
                raise TimeoutError("response lost")

            parent.shards[1].reserve_fanin_delivery = lost
            result = await actor.reserve(c.wire, epochs)
            assert result["state"] == "cancelled" and result["fenced"]
            assert len(result["writer_proofs"]) == 2
            assert not result["entry_count_held"]
            with pytest.raises(Exception, match="fenc|terminal|released"):
                await original(c.wire, expected_sender_epoch=epochs["1"])
            assert await actor.reserve(c.wire, epochs) == result

    asyncio.run(run())


def test_wrong_epoch_never_reserves_and_never_manufactures_closure():
    async def run():
        with setup() as c:
            parent, actor, epochs = group(c)
            epochs["1"] = "old-V-process"
            result = await actor.reserve(c.wire, epochs)
            assert result["state"] == "cancelling"
            assert not result["fenced"] and result["entry_count_held"]
            assert c.entries[1].active_delivery_count == 0
            assert parent.entries[c.key].active_delivery_count == 1
            changed = {**epochs, "1": c.stores[1].worker_epoch}
            with pytest.raises(CoordinatorError, match="another plan/epoch"):
                await actor.fence(c.wire, changed)

    asyncio.run(run())


def test_fence_before_reserve_and_retained_tombstone_capacity():
    async def run():
        with setup() as c:
            _, actor, epochs = group(c, max_records=1)
            result = await actor.fence(c.wire, epochs)
            assert result["state"] == "cancelled"
            assert await actor.reserve(c.wire, epochs) == result
            wire = copy.deepcopy(c.wire)
            wire["delivery_id"] = "another"
            wire.pop("plan_fingerprint")
            wire["plan_fingerprint"] = plan_fingerprint(wire)
            with pytest.raises(CoordinatorError, match="capacity"):
                await actor.reserve(wire, epochs)

    asyncio.run(run())


def test_start_waits_for_source_then_reaper_submits():
    async def run():
        with setup(committed=False) as c:
            parent, actor, epochs = group(c)
            await actor.reserve(c.wire, epochs)
            assert (await actor.start("fanin", 0))["state"] == "waiting_source"
            for store in c.stores:
                store.commit_p_write(c.key, 32)
            parent.entries[c.key].state = EntryState.STORED
            for _ in range(16):
                await parent.reap_expired()
            assert (await actor.poll("fanin", 0))["state"] == "delivered"

    asyncio.run(run())


def test_cancel_entry_closes_new_reservations_and_reaper_drains():
    async def run():
        with setup(engine=DelayedEngine()) as c:
            parent, actor, epochs = group(c)
            await actor.reserve(c.wire, epochs)
            await actor.start("fanin", 0)
            await parent.cancel_entry(c.key, "abort")
            assert parent.entries[c.key].active_delivery_count == 1
            for handle, *_ in c.engine.submitted:
                c.engine.complete(handle)
            await parent.reap_expired()
            assert parent.entries[c.key].active_delivery_count == 0
            assert (await actor.poll("fanin", 0))["state"] == "cancelled"

    asyncio.run(run())


def test_partial_ack_failure_retries_without_double_refund():
    async def run():
        with setup() as c:
            parent, actor, epochs = group(c)
            await actor.reserve(c.wire, epochs)
            await delivered(actor)
            original = parent.shards[1].ack_delivery

            async def fail(*args):
                await original(*args)
                raise TimeoutError("ACK response lost")

            parent.shards[1].ack_delivery = fail
            with pytest.raises(CoordinatorError, match="ACK unconfirmed"):
                await actor.ack("fanin", 0)
            assert parent.entries[c.key].active_delivery_count == 1
            parent.shards[1].ack_delivery = original
            assert (await actor.ack("fanin", 0))["state"] == "released"
            assert parent.entries[c.key].active_delivery_count == 0

    asyncio.run(run())


@pytest.mark.parametrize("corruption", ["rank", "epoch", "bytes", "plan", "missing"])
def test_invalid_fence_reply_does_not_release_entry(corruption):
    async def run():
        with setup() as c:
            parent, actor, epochs = group(c)
            await actor.reserve(c.wire, epochs)
            original = parent.shards[1].fence_fanin_delivery

            async def bad(*args):
                proof = copy.deepcopy(await original(*args))
                if corruption == "rank":
                    proof["source_rank"] = 0
                elif corruption == "epoch":
                    proof["identity"]["sender_epoch"] = "other"
                elif corruption == "bytes":
                    proof["transferred_bytes"] = 1
                elif corruption == "plan":
                    proof["plan_fingerprint"] = "other"
                else:
                    proof.pop("fenced")
                return proof

            parent.shards[1].fence_fanin_delivery = bad
            result = await actor.fence(c.wire, epochs)
            assert not result["fenced"] and result["entry_count_held"]
            assert parent.entries[c.key].active_delivery_count == 1
            parent.shards[1].fence_fanin_delivery = original
            assert not (await actor.poll("fanin", 0))["entry_count_held"]

    asyncio.run(run())


def test_concurrent_reserve_and_epoch_map_validation():
    async def run():
        with setup() as c:
            parent, actor, epochs = group(c)
            for bad in (
                {"0": epochs["0"]},
                {0: epochs["0"], 1: epochs["1"]},
                {**epochs, "1": ""},
            ):
                with pytest.raises(CoordinatorError, match="epoch"):
                    await actor.reserve(c.wire, bad)
            results = await asyncio.gather(
                *(actor.reserve(c.wire, epochs) for _ in range(4))
            )
            assert all(r == results[0] for r in results)
            assert parent.entries[c.key].active_delivery_count == 1
            assert all(e.active_delivery_count == 1 for e in c.entries)

    asyncio.run(run())


@pytest.mark.parametrize("enabled", [False, True])
def test_global_http_is_opt_in_and_keeps_exact_rank_type(enabled):
    async def run():
        with setup() as c:
            parent, _, epochs = group(c, enabled=enabled)
            async with TestClient(TestServer(create_coordinator_app(parent))) as client:
                response = await client.post(
                    "/v1/fanin/reserve",
                    json={"manifest": c.wire, "source_epochs": epochs},
                )
                body = await response.json()
                if not enabled:
                    assert response.status >= 400 and "disabled" in str(body)
                    return
                assert response.status == 200 and body["state"] == "reserved"
                response = await client.post(
                    "/v1/fanin/start",
                    json={"delivery_id": "fanin", "destination_rank": "0"},
                )
                assert response.status >= 400 and "D rank" in str(await response.json())
                response = await client.post(
                    "/v1/fanin/fence",
                    json={"manifest": c.wire, "source_epochs": epochs},
                )
                assert response.status == 200 and (await response.json())["fenced"]
                response = await client.get("/health")
                assert (await response.json())["full_kv_fanin"]["retained_records"] == 1

    asyncio.run(run())


def test_launcher_bounds_do_not_silently_enable_coordinator():
    args = SimpleNamespace(full_kv_fanin_max_slices=64)
    assert _fanin_coordinator_args(args) == {
        "full_kv_fanin_max_slices": None,
        "full_kv_fanin_max_records": None,
    }
    args.full_kv_fanin_max_records = 8
    assert _fanin_coordinator_args(args) == {
        "full_kv_fanin_max_slices": 64,
        "full_kv_fanin_max_records": 8,
    }


def test_cancelled_reserve_task_retains_record_until_reaper_fence():
    async def run():
        with setup() as c:
            parent, actor, epochs = group(c)
            seen = asyncio.Event()
            original = parent.shards[1].reserve_fanin_delivery

            async def blocked(*args, **kwargs):
                await original(*args, **kwargs)
                seen.set()
                await asyncio.Event().wait()

            parent.shards[1].reserve_fanin_delivery = blocked
            task = asyncio.create_task(actor.reserve(c.wire, epochs))
            await seen.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert parent.entries[c.key].active_delivery_count == 1
            await parent.reap_expired()
            result = await actor.poll("fanin", 0)
            assert result["state"] == "cancelled" and result["fenced"]
            assert parent.entries[c.key].active_delivery_count == 0

    asyncio.run(run())


def test_deadline_fences_before_any_submission():
    async def run():
        with setup(engine=DelayedEngine()) as c:
            _, actor, epochs = group(c)
            await actor.reserve(c.wire, epochs)
            actor.records[("fanin", 0)].deadline = time.monotonic() - 1
            result = await actor.start("fanin", 0)
            assert result["state"] == "cancelled" and result["fenced"]
            assert not c.engine.submitted

    asyncio.run(run())
