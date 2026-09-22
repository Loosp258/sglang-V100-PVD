"""Bounded owner-thread cleanup polling; pool callbacks here are explicit doubles."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError
from sglang.srt.disaggregation.pvd.cpu_refresh_driver import CPURefreshDriver
from sglang.srt.disaggregation.pvd.cpu_release_driver import CPUReleaseDriver
from test_pvd_cpu_decode_lifecycle import env
from test_pvd_cpu_request_release import binding


def setup(life, *, clock=lambda: 0, refresh=None, capacity=2, inflight=1):
    temporary, req, cache, release, calls = binding(life)
    req.pvd_cpu_kv_release = None  # unpublished test binding, no resource acquired
    driver = CPUReleaseDriver(
        temporary.executor,
        cache,
        max_requests=capacity,
        max_inflight=inflight,
        retry_seconds=1,
        refresh_driver=refresh,
        clock=clock,
    )
    driver._request_release = lambda r: r.owner.defer(r.req, cache, False, release)
    owner = driver.register(req, life)
    return driver, req, owner, calls


async def turns(driver, count=6):
    for _ in range(count):
        driver.poll()
        await asyncio.sleep(0)


def test_async_poll_retires_terminal_request_and_refresh_registration():
    async def run():
        with env() as (life, f, _):
            life.admit(f.request)
            refresh = CPURefreshDriver(life.arbiter)
            refresh.register(
                life,
                clients={0: None, 1: None},
                pack_source=f.pack_source,
                timeout_seconds=30,
            )
            driver, req, owner, calls = setup(life, refresh=refresh)
            assert driver.register is not None and owner.state == "attached"
            driver.poll()
            assert not calls
            life.terminate("client cancelled")
            await turns(driver)
            assert owner.state == "released" and len(calls) == 1
            assert req.req_pool_idx is None and not refresh.contains(life)
            assert not driver.snapshot()["requests"]
            driver.begin_shutdown()
            assert driver.snapshot()["drained"]
            driver.close_loop()  # never closes caller's running loop
            assert not asyncio.get_running_loop().is_closed()

    asyncio.run(run())


def test_sync_poll_pumps_one_owner_turn_without_a_background_thread():
    with env() as (life, f, _):
        life.admit(f.request)
        driver, req, owner, calls = setup(life)
        try:
            driver.begin_shutdown()
            for _ in range(8):
                driver.poll()
            assert driver.snapshot()["drained"] and req.req_pool_idx is None
            assert owner.state == "released" and len(calls) == 1
        finally:
            driver.close_loop()
        assert driver._loop.is_closed()
        with pytest.raises(LifecycleError, match="closing"):
            driver.register(req, life)


def test_pending_io_does_not_block_poll_or_another_live_request(monkeypatch):
    async def run():
        with env("a") as (a, f, _), env("b", a.arbiter) as (b, bf, _):
            a.admit(f.request)
            b.admit(bf.request)
            driver, _, owner, calls = setup(a)
            event = asyncio.Event()
            original = a.close

            async def slow_close():
                await event.wait()
                await original()

            monkeypatch.setattr(a, "close", slow_close)
            driver.begin_shutdown()
            await turns(driver)
            assert not calls and owner.state == "pending"
            assert b.can_decode()
            with pytest.raises(LifecycleError, match="must drain"):
                driver.close_loop()
            event.set()
            await turns(driver)
            assert driver.snapshot()["drained"]

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["error", "cancel"])
def test_drain_failure_keeps_capacity_and_retries_only_after_backoff(
    monkeypatch, failure
):
    async def run():
        with env() as (life, f, _):
            life.admit(f.request)
            now = [0]
            driver, req, owner, calls = setup(life, clock=lambda: now[0], capacity=1)
            original = life.close

            async def failed_close():
                if failure == "cancel":
                    raise asyncio.CancelledError()
                raise RuntimeError("unknown fence")

            monkeypatch.setattr(life, "close", failed_close)
            life.terminate("cancel")
            await turns(driver)
            record = driver.snapshot()["requests"][req.rid]
            assert record["error"] and not record["inflight"]
            assert owner.state == "pending" and req.req_pool_idx == 1 and not calls
            with pytest.raises(LifecycleError, match="full"):
                driver.register(req, life)
            monkeypatch.setattr(life, "close", original)
            await turns(driver)
            assert not calls
            now[0] = 1
            await turns(driver)
            assert owner.state == "released" and len(calls) == 1
            driver.begin_shutdown()
            driver.close_loop()

    asyncio.run(run())


def test_forward_lease_prevents_release_and_shutdown_does_not_cancel_it():
    async def run():
        with env() as (life, f, _):
            life.admit(f.request)
            driver, req, _, calls = setup(life)
            permit = life.begin_decode()
            driver.begin_shutdown()
            await turns(driver)
            assert life.arbiter.busy and not calls and req.req_pool_idx == 1
            life.complete_decode(permit, 17)
            await turns(driver)
            assert len(calls) == 1 and driver.snapshot()["drained"]

    asyncio.run(run())


def test_partial_release_failure_is_not_retried_by_poll():
    async def run():
        with env() as (life, f, _):
            life.admit(f.request)
            now = [0]
            driver, _, owner, calls = setup(life, clock=lambda: now[0])

            def bad_release(*args, **kwargs):
                calls.append("partial")
                raise RuntimeError("allocator partially mutated")

            driver._request_release = lambda r: r.owner.defer(
                r.req, driver.tree_cache, False, bad_release
            )
            driver.begin_shutdown()
            await turns(driver)
            now[0] = 100
            await turns(driver)
            assert calls == ["partial"] and owner.state == "quarantined"
            assert not driver.snapshot()["drained"]

    asyncio.run(run())


def test_wrong_loop_cannot_progress_owned_cleanup():
    with env() as (life, f, _):
        life.admit(f.request)
        driver, _, _, _ = setup(life)

        async def foreign_loop():
            with pytest.raises(LifecycleError, match="event loop"):
                driver.poll()

        asyncio.run(foreign_loop())
        driver.begin_shutdown()
        for _ in range(8):
            driver.poll()
        driver.close_loop()


def test_multiple_requests_respect_drain_concurrency_and_keep_admission(monkeypatch):
    async def run():
        with env("a") as (a, af, _), env("b", a.arbiter) as (b, bf, _):
            a.admit(af.request)
            b.admit(bf.request)
            driver, _, aowner, calls = setup(a, capacity=2, inflight=1)
            # Both explicit test requests bind distinct rows in the same pool.
            driver.executor._storage[b] = 2
            breq = NS(rid="b", req_pool_idx=2, finished=lambda: False)
            bowner = driver.register(breq, b)
            event = asyncio.Event()
            original = a.close

            async def slow_close():
                await event.wait()
                await original()

            def release(req, cache, *, is_insert):
                calls.append(req.rid)
                req.req_pool_idx = None

            driver._request_release = lambda r: r.owner.defer(
                r.req, driver.tree_cache, False, release
            )
            monkeypatch.setattr(a, "close", slow_close)
            driver.begin_shutdown()
            await turns(driver)
            assert aowner.state == bowner.state == "pending"
            assert (
                sum(r["inflight"] for r in driver.snapshot()["requests"].values()) == 1
            )
            assert not calls
            with pytest.raises(LifecycleError, match="closing"):
                driver.register(breq, b)
            event.set()
            await turns(driver, 12)
            assert calls == ["a", "b"] and driver.snapshot()["drained"]
            driver.close_loop()

    asyncio.run(run())


def scheduler_methods():
    # Execute the checkout's exact loop body with explicit Scheduler doubles.
    # This tests hook ordering, not a real serving Scheduler/model forward.
    path = Path("python/sglang/srt/disaggregation/decode.py")
    cls = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef)
        and n.name == "SchedulerDisaggregationDecodeMixin"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("poll_pvd_cpu_releases", "event_loop_normal_disagg_decode")
    ]
    for method in methods:
        method.decorator_list = []
    scope = {"Scheduler": object}
    exec(  # noqa: S102 -- trusted checkout source; no user-supplied code
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope
    )
    return scope


@pytest.mark.parametrize(
    "paused,batch,expected",
    [
        (True, None, ["inputs", "queue", "poll"]),
        (False, None, ["inputs", "queue", "poll", "idle", "poll"]),
        (False, object(), ["inputs", "queue", "poll", "forward", "result", "poll"]),
    ],
)
def test_actual_scheduler_loop_polls_when_paused_idle_or_after_result(
    paused, batch, expected
):
    methods, events = scheduler_methods(), []
    received = False

    class EndIteration(Exception):
        pass

    def receive():
        nonlocal received
        if received:
            raise EndIteration()
        received = True
        return []

    scheduler = NS(
        request_receiver=NS(recv_requests=receive),
        process_input_requests=lambda _: events.append("inputs"),
        process_decode_queue=lambda: events.append("queue"),
        poll_pvd_cpu_releases=lambda: events.append("poll"),
        _engine_paused=paused,
        get_next_disagg_decode_batch_to_run=lambda: batch,
        run_batch=lambda _: events.append("forward"),
        process_batch_result=lambda *args: events.append("result"),
        on_idle=lambda: events.append("idle"),
    )
    with pytest.raises(EndIteration):
        methods["event_loop_normal_disagg_decode"](scheduler)
    assert events == expected
    hook = methods["poll_pvd_cpu_releases"]
    assert hook(NS()) is None
    with pytest.raises(TypeError, match="CPU release driver"):
        hook(NS(pvd_cpu_release_driver=object()))


def test_actual_scheduler_poll_hook_drives_sync_retirement():
    with env() as (life, f, _):
        life.admit(f.request)
        driver, req, owner, calls = setup(life)
        scheduler = NS(pvd_cpu_release_driver=driver)
        hook = scheduler_methods()["poll_pvd_cpu_releases"]
        driver.begin_shutdown()
        for _ in range(8):
            hook(scheduler)
        assert driver.snapshot()["drained"] and len(calls) == 1
        assert owner.state == "released" and req.req_pool_idx is None
        driver.close_loop()


def test_actual_loop_does_not_check_idle_pools_or_sleep_while_release_is_pending():
    methods, events = scheduler_methods(), []
    received = 0

    class EndIteration(Exception):
        pass

    def receive():
        nonlocal received
        received += 1
        if received > 2:
            raise EndIteration()
        return []

    def poll():
        events.append("poll")
        return {"requests": {"held": {}} if received == 1 else {}}

    scheduler = NS(
        request_receiver=NS(recv_requests=receive),
        process_input_requests=lambda _: None,
        process_decode_queue=lambda: None,
        poll_pvd_cpu_releases=poll,
        _engine_paused=False,
        get_next_disagg_decode_batch_to_run=lambda: None,
        on_idle=lambda: events.append("idle"),
    )
    with pytest.raises(EndIteration):
        methods["event_loop_normal_disagg_decode"](scheduler)
    assert events == ["poll", "poll", "poll", "idle", "poll"]
