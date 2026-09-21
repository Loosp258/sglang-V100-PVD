"""Automatic refresh clocks over real local V HTTP and CPU install groups."""

import asyncio
from dataclasses import replace

import pytest
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError
from sglang.srt.disaggregation.pvd.cpu_refresh_driver import CPURefreshDriver
from test_pvd_controlled_prefetch import DelayedClient
from test_pvd_cpu_decode_lifecycle import advance, env


def register(driver, life, f, clients, timeout=30):
    driver.register(
        life, clients=clients, pack_source=f.pack_source, timeout_seconds=timeout
    )


def test_early_ready_installs_only_at_boundary_and_missed_window_uses_actual_q():
    async def run():
        with env() as (life, f, _):
            life.admit(f.request)
            driver = CPURefreshDriver(life.arbiter)
            async with f.clients() as clients:
                register(driver, life, f, clients)
                assert not driver.progress().launched
                advance(life, 3)
                report = driver.progress()
                launch = report.launched[0]
                assert launch.query_source == "predicted"
                assert launch.query_position == len(life.prompt) + 4
                assert not driver.progress().launched  # queued probe already owns lease
                await launch.task
                assert not driver.progress().installed  # result early, no clock jump
                assert life.committed_tokens == 3
                advance(life, 1)
                assert driver.progress().installed == (life.request_id,)
                calls = len(f.request.pipeline.provider.calls)
                advance(life, 4)  # deliberately miss next prefetch window
                launch = driver.progress().launched[0]
                assert launch.query_source == "committed"
                assert launch.query_position == len(life.snapshot().tokens) - 1
                await launch.task
                assert driver.progress().installed == (life.request_id,)
                assert len(f.request.pipeline.provider.calls) == calls
                assert life.committed_tokens == 8 and life.can_decode()
                await driver.close()
                assert not driver._records and not life.arbiter.busy

    asyncio.run(run())


def test_new_registration_does_not_reset_old_query_and_late_result_is_not_replaced():
    async def run():
        with env("old") as (old, f, _), env("new", old.arbiter) as (new, nf, _):
            old.admit(f.request)
            new.admit(nf.request)
            driver = CPURefreshDriver(old.arbiter)
            async with f.clients() as clients, nf.clients() as new_clients:
                delayed = DelayedClient(clients[1])
                register(driver, old, f, {0: clients[0], 1: delayed})
                advance(old, 3)
                launch = driver.progress().launched[0]
                await delayed.entered.wait()
                before = f.group.coordinator.snapshot()
                register(driver, new, nf, new_clients)
                assert f.group.coordinator.snapshot() == before
                advance(old, 1)
                for _ in range(3):
                    progress = driver.progress()
                    assert not progress.launched and not progress.installed
                assert old._refresh is launch.task and new.committed_tokens == 0
                delayed.release.set()
                await launch.task
                assert driver.progress().installed == ("old",)
                assert nf.request.pipeline.provider.calls == []
                await driver.close()

    asyncio.run(run())


def test_two_due_requests_capture_serially_but_search_can_be_in_flight_together():
    async def run():
        with env("a") as (a, f, _), env("b", a.arbiter) as (b, bf, _):
            a.admit(f.request)
            b.admit(bf.request)
            advance(a, 3)
            advance(b, 3)
            driver = CPURefreshDriver(a.arbiter)
            async with f.clients() as ca, bf.clients() as cb:
                da, db = DelayedClient(ca[1]), DelayedClient(cb[1])
                register(driver, a, f, {0: ca[0], 1: da})
                register(driver, b, bf, {0: cb[0], 1: db})
                first = driver.progress().launched[0]
                assert first.request_id == "a" and not driver.progress().launched
                await da.entered.wait()
                second = driver.progress().launched[0]
                assert second.request_id == "b"
                await db.entered.wait()
                assert not first.task.done() and not second.task.done()
                da.release.set()
                db.release.set()
                await asyncio.gather(first.task, second.task)
                advance(a, 1)
                advance(b, 1)
                assert driver.progress().installed == ("a", "b")
                await driver.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "mode", ["duplicate", "ranks", "timeout", "nan", "short_draft"]
)
def test_registration_refuses_ambiguous_or_unbounded_configuration(mode):
    async def run():
        with env() as (life, f, _):
            life.admit(f.request)
            driver = CPURefreshDriver(life.arbiter)
            async with f.clients() as clients:
                if mode == "duplicate":
                    register(driver, life, f, clients)
                if mode == "short_draft":
                    # Increase the required lead beyond the configured horizon.
                    f.request.pipeline.draft_config = replace(
                        f.request.pipeline.draft_config, predict_tokens=1
                    )
                    f.group.coordinator._clock.lead_tokens = 2
                with pytest.raises(LifecycleError):
                    register(
                        driver,
                        life,
                        f,
                        {} if mode == "ranks" else clients,
                        float("nan")
                        if mode == "nan"
                        else 0
                        if mode == "timeout"
                        else 30,
                    )
                await driver.close()

    asyncio.run(run())


def test_close_refuses_live_forward_and_timeout_drains_queued_capture():
    async def run():
        with env() as (life, f, now):
            life.admit(f.request)
            driver = CPURefreshDriver(life.arbiter)
            async with f.clients() as clients:
                register(driver, life, f, clients, timeout=1)
                permit = life.begin_decode()
                with pytest.raises(LifecycleError, match="must drain"):
                    await driver.close()
                assert life.arbiter.busy
                life.complete_decode(permit, 12)
                advance(life, 2)
                launch = driver.progress().launched[0]
                now[0] = 2
                assert driver.progress().aborted == (life.request_id,)
                assert life.arbiter.busy  # cancel-before-first-dispatch not drained
                await driver.close()
                assert launch.task.done() and not life.arbiter.busy

    asyncio.run(run())
