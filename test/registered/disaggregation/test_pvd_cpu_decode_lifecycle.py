"""Owner-thread dispatch contracts; real forwards run in controlled-decode smoke."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace

import pytest
from pvd_controlled_prefetch import ControlledFixture
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import (
    CPUDecodeLifecycle,
    LifecycleError,
    TargetExecutionArbiter,
)
from test_pvd_controlled_prefetch import DelayedClient, components


@contextmanager
def env(name="r", arbiter=None):
    f = ControlledFixture(*components(name))
    now = [0.0]
    lifecycle = CPUDecodeLifecycle(
        name,
        f.prefix.tokens,
        9,
        arbiter=arbiter or TargetExecutionArbiter(),
        clock=lambda: now[0],
    )
    try:
        yield lifecycle, f, now
    finally:
        assert lifecycle._permit is None and lifecycle._refresh_lease is None
        f.close()


def advance(lifecycle, count):
    for _ in range(count):
        permit = lifecycle.begin_decode()
        assert lifecycle.complete_decode(permit, 10 + lifecycle.committed_tokens)


def test_waiting_admission_and_actual_output_snapshot():
    with env() as (life, f, _):
        assert not life.can_decode()
        with pytest.raises(LifecycleError, match="wait or abort"):
            life.begin_decode()
        life.admit(f.request)
        assert life.committed_tokens == 0 and life.outputs == (9,)
        advance(life, 3)
        before = life.snapshot()
        assert before.tokens == f.prefix.tokens + (9, 10, 11, 12)
        assert before.committed_position == 3
        advance(life, 1)
        assert before.tokens != life.snapshot().tokens
        assert not life.can_decode()  # boundary, no refresh installed
        with pytest.raises(LifecycleError):
            life.begin_decode()


def test_controller_cannot_be_claimed_by_two_lifecycles():
    with env() as (life, f, _):
        life.admit(f.request)
        other = CPUDecodeLifecycle("r", f.prefix.tokens, 9, arbiter=life.arbiter)
        with pytest.raises(LifecycleError, match="already belongs"):
            other.admit(f.request)


@pytest.mark.parametrize("mutation", ["copy", "foreign", "replay"])
def test_completion_requires_the_actual_outstanding_permit(mutation):
    with env() as (life, f, _):
        life.admit(f.request)
        permit = life.begin_decode()
        bad = (
            replace(permit, incarnation="other")
            if mutation == "foreign"
            else replace(permit)
        )
        if mutation == "replay":
            life.complete_decode(permit, 10)
            bad = permit
        with pytest.raises(LifecycleError, match="stale or foreign"):
            life.complete_decode(bad, 11)
        if life._permit is not None:
            life.fail_decode(permit, "test cleanup")


def test_cancel_during_forward_retains_execution_until_completion_then_discards():
    with env() as (life, f, _):
        life.admit(f.request)
        permit = life.begin_decode()
        life.terminate("client cancellation")
        assert life.arbiter.busy
        with pytest.raises(LifecycleError, match="busy"):
            life.arbiter.acquire()
        assert not life.complete_decode(permit, 12)
        assert life.outputs == (9,) and not life.arbiter.busy


def test_failure_never_advances_output_and_eos_commits_only_its_last_token():
    with env() as (life, f, _):
        life.admit(f.request)
        permit = life.begin_decode()
        life.fail_decode(permit, "model failed")
        assert life.committed_tokens == 0 and life.state == "aborted"
    with env() as (life, f, _):
        life.admit(f.request)
        assert life.complete_decode(life.begin_decode(), 2, finished=True)
        assert life.committed_tokens == 1 and life.state == "finished"
        assert not life.can_decode()


def test_shared_target_execution_blocks_probe_decode_and_other_request():
    with env("old") as (old, f, _):
        old.admit(f.request)
        with env("new", old.arbiter) as (new, nf, _):
            new.admit(nf.request)
            permit = old.begin_decode()
            assert not new.can_decode()
            with pytest.raises(LifecycleError, match="between forwards"):
                old.snapshot()
            old.complete_decode(permit, 10)
            assert new.can_decode()
            new.terminate("new request cancelled")
            assert old.committed_tokens == 1 and old.can_decode()


def test_owner_thread_refuses_callbacks_mutating_counts():
    with env() as (life, f, _):
        life.admit(f.request)
        with (
            ThreadPoolExecutor(1) as executor,
            pytest.raises(LifecycleError, match="owner thread"),
        ):
            executor.submit(life.begin_decode).result()
        assert life.committed_tokens == 0


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan")])
def test_bad_refresh_timeout_refused_before_dispatch(timeout):
    async def run():
        with env() as (life, f, _):
            life.admit(f.request)
            advance(life, 3)
            with pytest.raises(LifecycleError, match="finite positive"):
                life.launch_refresh(
                    query_positions=(9,),
                    clients={},
                    pack_source=f.pack_source,
                    timeout_seconds=timeout,
                )
            assert not life.arbiter.busy and life.state == "running"
            await life.close()

    asyncio.run(run())


@pytest.mark.parametrize("boundary_start", [False, True])
def test_delayed_http_releases_target_but_waits_at_boundary(boundary_start):
    async def run():
        with env() as (life, f, _):
            life.admit(f.request)
            advance(life, 4 if boundary_start else 3)
            async with f.clients() as clients:
                delayed = DelayedClient(clients[1])
                prefix = life.snapshot()
                task = life.launch_refresh(
                    query_positions=(len(prefix.tokens) - int(boundary_start),),
                    clients={0: clients[0], 1: delayed},
                    pack_source=f.pack_source,
                    timeout_seconds=5,
                )
                try:
                    assert (
                        life.arbiter.busy and not life.can_decode()
                    )  # queued probe owns target
                    await asyncio.wait_for(delayed.entered.wait(), 5)
                    assert not life.arbiter.busy
                    if not boundary_start:
                        advance(life, 1)
                    assert not life.can_decode()
                    assert not life.try_install({0: 4, 1: 4})
                    delayed.release.set()
                    epoch = await task
                    assert epoch.target_tokens == 4
                    assert life.try_install({0: 4, 1: 4})
                    assert life.can_decode()
                    advance(life, 1)
                    assert life.committed_tokens == 5
                    assert len(f.request.pipeline.provider.calls) == int(
                        not boundary_start
                    )
                finally:
                    await life.close()

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["queued", "http", "ready"])
def test_timeout_or_cancel_drains_without_reusing_target_lease(phase):
    async def run():
        with env() as (life, f, now):
            life.admit(f.request)
            advance(life, 3)
            async with f.clients() as clients:
                delayed = DelayedClient(clients[1])
                task = life.launch_refresh(
                    query_positions=(len(life.snapshot().tokens),),
                    clients={0: clients[0], 1: delayed},
                    pack_source=f.pack_source,
                    timeout_seconds=5,
                )
                try:
                    if phase != "queued":
                        await asyncio.wait_for(delayed.entered.wait(), 5)
                    if phase == "ready":
                        delayed.release.set()
                        await task
                    now[0] = 5
                    assert life.poll() == "aborted"
                    assert life.reason == "refresh timeout"
                    assert not life.can_decode()
                    assert f.group.coordinator.snapshot()["installed_tokens"] == 0
                finally:
                    await life.close()
                assert not life.arbiter.busy

    asyncio.run(run())


def test_early_result_cannot_forge_rank_counts_to_install_ahead():
    async def run():
        with env() as (life, f, _):
            life.admit(f.request)
            advance(life, 3)
            async with f.clients() as clients:
                await life.launch_refresh(
                    query_positions=(len(life.snapshot().tokens),),
                    clients=clients,
                    pack_source=f.pack_source,
                    timeout_seconds=5,
                )
                with pytest.raises(LifecycleError, match="actually committed"):
                    life.try_install({0: 4, 1: 4})
                assert not life.try_install({0: 3, 1: 3})
                assert f.group.coordinator.snapshot()["installed_tokens"] == 0
                advance(life, 1)
                assert life.try_install({0: 4, 1: 4})
                await life.close()

    asyncio.run(run())


def test_close_refuses_live_decode_and_can_be_retried_after_completion():
    async def run():
        with env() as (life, f, _):
            life.admit(f.request)
            permit = life.begin_decode()
            with pytest.raises(LifecycleError, match="must drain"):
                await life.close()
            assert life.arbiter.busy
            assert not life.complete_decode(permit, 10)
            await life.close()
            assert not life.arbiter.busy

    asyncio.run(run())


def test_refresh_failure_before_capture_refunds_dispatch_lease():
    async def run():
        with env() as (life, f, _):
            life.admit(f.request)
            advance(life, 3)
            task = life.launch_refresh(
                query_positions=(0,),
                clients={0: object(), 1: object()},
                pack_source=f.pack_source,
                timeout_seconds=5,
            )
            await asyncio.gather(task, return_exceptions=True)
            assert life.poll() == "aborted"
            assert not life.arbiter.busy
            await life.close()

    asyncio.run(run())


def test_new_request_and_eos_do_not_reset_another_inflight_refresh():
    async def run():
        with env("old") as (life, f, _):
            life.admit(f.request)
            advance(life, 3)
            async with f.clients() as clients:
                delayed = DelayedClient(clients[1])
                task = life.launch_refresh(
                    query_positions=(len(life.snapshot().tokens),),
                    clients={0: clients[0], 1: delayed},
                    pack_source=f.pack_source,
                    timeout_seconds=5,
                )
                try:
                    await asyncio.wait_for(delayed.entered.wait(), 5)
                    before = f.group.coordinator.snapshot()
                    with env("new", life.arbiter) as (new, nf, _):
                        new.admit(nf.request)
                        assert new.complete_decode(new.begin_decode(), 2, finished=True)
                        await new.close()
                    assert f.group.coordinator.snapshot() == before
                    assert life.committed_tokens == 3
                    delayed.release.set()
                    await task
                    advance(life, 1)
                    assert life.try_install({0: 4, 1: 4})
                finally:
                    await life.close()

    asyncio.run(run())


def test_timeout_during_decode_discards_its_late_output_and_drains_both_owners():
    async def run():
        with env() as (life, f, now):
            life.admit(f.request)
            advance(life, 3)
            async with f.clients() as clients:
                delayed = DelayedClient(clients[1])
                life.launch_refresh(
                    query_positions=(len(life.snapshot().tokens),),
                    clients={0: clients[0], 1: delayed},
                    pack_source=f.pack_source,
                    timeout_seconds=5,
                )
                await asyncio.wait_for(delayed.entered.wait(), 5)
                permit = life.begin_decode()
                now[0] = 6
                assert not life.complete_decode(permit, 21)
                assert life.committed_tokens == 3
                assert not life.arbiter.busy and life.state == "aborted"
                await life.close()

    asyncio.run(run())


def test_eos_cancels_pending_refresh_without_leaking_a_dispatch_lease():
    async def run():
        with env() as (life, f, _):
            life.admit(f.request)
            advance(life, 3)
            async with f.clients() as clients:
                delayed = DelayedClient(clients[1])
                task = life.launch_refresh(
                    query_positions=(len(life.snapshot().tokens),),
                    clients={0: clients[0], 1: delayed},
                    pack_source=f.pack_source,
                    timeout_seconds=5,
                )
                await asyncio.wait_for(delayed.entered.wait(), 5)
                assert life.complete_decode(life.begin_decode(), 2, finished=True)
                assert life.state == "finished" and life.committed_tokens == 4
                await life.close()
                assert task.cancelled() and not life.arbiter.busy
                assert f.group.coordinator.snapshot()["installed_tokens"] == 0

    asyncio.run(run())
