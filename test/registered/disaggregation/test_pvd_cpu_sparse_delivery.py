"""Search -> actual shard HTTP Delivery -> CPU banks, without local pack_source."""

import asyncio

import pytest
from pvd_controlled_prefetch import ControlledFixture
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import (
    CPUDecodeLifecycle,
    TargetExecutionArbiter,
)
from sglang.srt.disaggregation.pvd.cpu_refresh_driver import CPURefreshDriver
from sglang.srt.disaggregation.pvd.request_state import DeliveryState
from sglang.srt.disaggregation.pvd.sparse_receiver import SparseReceiveError
from test_pvd_controlled_prefetch import components


async def refresh(fixture, clients, count=3):
    prefix = fixture.refresh_prefix(count)
    return await fixture.request.refresh(
        prefix, query_positions=(len(prefix.tokens) - int(count == 4),), clients=clients
    )


async def wait_acks(delivery):
    async def wait():
        while delivery.snapshot()["ack_tasks"]:
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 5)


def test_two_rounds_use_http_delivery_and_ack_without_local_packing():
    async def run():
        fixture = ControlledFixture(*components())
        try:
            async with fixture.clients(wire_delivery=True) as clients:
                delivery = fixture.request.delivery
                epochs = []
                for n in (3, 7):
                    epoch = await refresh(fixture, clients, n)
                    epochs.append(epoch)
                    assert not fixture.request.can_decode(n + 1)
                    assert all(
                        d.state == DeliveryState.DELIVERED
                        for store in fixture.stores.values()
                        for d in store.entries[fixture.manifest.key].deliveries.values()
                        if d.sparse_manifest.specs[0].operation_id == epoch.operation_id
                    )
                    assert fixture.request.try_install({0: n + 1, 1: n + 1})
                    await wait_acks(delivery)
                    assert delivery.snapshot()["pending_rounds"] == 0
                    assert delivery.registry.snapshot() == {}
                    assert delivery.snapshot()["errors"] == {}
                assert epochs[0].operation_id != epochs[1].operation_id
                assert fixture.packed_specs == []  # local packing path never used
                for store in fixture.stores.values():
                    records = store.entries[fixture.manifest.key].deliveries
                    assert len(records) == 2
                    assert all(
                        d.state == DeliveryState.RELEASED for d in records.values()
                    )
                    assert store.transfer_engine.total_put_bytes == sum(
                        d.sparse_manifest.nbytes for d in records.values()
                    )
        finally:
            fixture.close()

    asyncio.run(run())


def test_delivery_wait_keeps_clock_and_does_not_requery_when_new_request_arrives(
    monkeypatch,
):
    async def run():
        fixture = ControlledFixture(*components("old"))
        newcomer = None
        task = None
        try:
            async with fixture.clients(wire_delivery=True) as clients:
                sink = fixture.request.delivery
                control = sink._routes[1].client
                original = control.start_delivery
                entered, release = asyncio.Event(), asyncio.Event()

                async def delayed(*args):
                    result = await original(*args)
                    entered.set()
                    await release.wait()
                    return result

                monkeypatch.setattr(control, "start_delivery", delayed)
                task = asyncio.create_task(refresh(fixture, clients))
                await asyncio.wait_for(entered.wait(), 5)
                assert not task.done()
                state = fixture.group.coordinator.snapshot()
                assert fixture.request.can_decode(3)
                assert not fixture.request.can_decode(4)
                assert not fixture.request.try_install({0: 4, 1: 4})
                newcomer = ControlledFixture(*components("new"))
                assert fixture.group.coordinator.snapshot() == state
                release.set()
                await asyncio.wait_for(task, 5)
                assert fixture.request.try_install({0: 4, 1: 4})
                await wait_acks(sink)
                assert len(fixture.request.pipeline.provider.calls) == 1
        finally:
            if task and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if newcomer:
                newcomer.close()
            fixture.close()

    asyncio.run(run())


def test_lost_ack_is_visible_and_explicit_retry_does_not_repeat_install(monkeypatch):
    async def run():
        fixture = ControlledFixture(*components())
        try:
            async with fixture.clients(wire_delivery=True) as clients:
                sink = fixture.request.delivery
                control = sink._routes[0].client
                original = control.ack_delivery

                async def lost(*args):
                    await original(*args)
                    raise TimeoutError("lost ACK response")

                monkeypatch.setattr(control, "ack_delivery", lost)
                epoch = await refresh(fixture, clients)
                assert fixture.request.try_install({0: 4, 1: 4})
                await wait_acks(sink)
                assert (
                    sink.snapshot()["errors"][epoch.operation_id][0]
                    == "lost ACK response"
                )
                assert sink.snapshot()["retained_destinations"] == 1
                assert fixture.request.can_decode(4)
                monkeypatch.setattr(control, "ack_delivery", original)
                await sink.retry_acks(epoch)
                assert sink.snapshot()["retained_destinations"] == 0
                assert fixture.group.coordinator.snapshot()["round"] == 2
        finally:
            fixture.close()

    asyncio.run(run())


def test_stale_index_between_search_and_delivery_aborts_before_install(monkeypatch):
    async def run():
        fixture = ControlledFixture(*components())
        try:
            async with fixture.clients(wire_delivery=True) as clients:
                sink = fixture.request.delivery
                control = sink._routes[1].client
                original = control.start_delivery

                async def rebuild(*args):
                    store = fixture.stores[1]
                    index = store.prompt_index
                    index.close(fixture.manifest.key.transfer_id)
                    index.note_kv_readable(fixture.manifest.key.transfer_id)
                    store.progress_prompt_indexes()
                    return await original(*args)

                monkeypatch.setattr(control, "start_delivery", rebuild)
                with pytest.raises(SparseReceiveError, match="failed"):
                    await refresh(fixture, clients)
                assert not fixture.request.can_decode(4)
                assert fixture.group.coordinator.snapshot()["installed_tokens"] == 0
        finally:
            fixture.close()

    asyncio.run(run())


def test_no_local_source_may_silently_override_configured_delivery():
    async def run():
        fixture = ControlledFixture(*components())
        try:
            async with fixture.clients(wire_delivery=True) as clients:
                with pytest.raises(ValueError, match="exactly one"):
                    await fixture.request.refresh(
                        fixture.refresh_prefix(),
                        query_positions=(9,),
                        clients=clients,
                        pack_source=fixture.pack_source,
                    )
                assert not fixture.request.delivery.registry.snapshot()
        finally:
            fixture.close()

    asyncio.run(run())


def test_automatic_driver_installs_wire_delivery_without_source_callback():
    async def run():
        fixture = ControlledFixture(*components())
        arbiter = TargetExecutionArbiter()
        driver = CPURefreshDriver(arbiter)
        life = CPUDecodeLifecycle("r", fixture.prefix.tokens, 9, arbiter=arbiter)
        try:
            async with fixture.clients(wire_delivery=True) as clients:
                life.admit(fixture.request)
                driver.register(life, clients=clients, timeout_seconds=10)
                for i in range(3):
                    life.complete_decode(life.begin_decode(), 10 + i)
                task = driver.progress().launched[0].task
                await task
                assert life.committed_tokens == 3
                assert driver.progress().installed == ()
                life.complete_decode(life.begin_decode(), 13)
                assert driver.progress().installed == ("r",)
                await wait_acks(fixture.request.delivery)
                assert life.can_decode()
                assert fixture.packed_specs == []
                await driver.close()
        finally:
            await life.close()
            fixture.close()

    asyncio.run(run())


def test_scoped_close_does_not_release_another_requests_destination():
    async def run():
        fixture = ControlledFixture(*components())
        try:
            async with fixture.clients(wire_delivery=True) as clients:
                sink = fixture.request.delivery
                await refresh(fixture, clients)
                existing = next(iter(sink._rounds.values()))[0]
                route = sink._routes[0]
                unrelated = sink.registry.prepare(
                    existing.manifest,
                    key=fixture.manifest.key,
                    rank=0,
                    endpoint="other-D",
                    rail=route.rail,
                    sender_epoch=route.sender_epoch,
                    client=route.client,
                    owner_scope="other-request",
                )
                assert await sink.close() == {}
                assert unrelated.identity.transfer_id in sink.registry.snapshot()
                assert not unrelated.snapshot()["closed"]
                await unrelated.close()
        finally:
            fixture.close()

    asyncio.run(run())
