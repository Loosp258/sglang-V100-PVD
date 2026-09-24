"""Receiver/driver handoff with real control-loop close, CPU banks, fake model/MR.

Controller execution and allocator return are doubles here. Their independent
tests cover forward/retirement; this file proves ordering and ownership only.
"""

import asyncio
import time
from concurrent.futures import Future
from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
from sglang.srt.disaggregation.pvd.conn import _AsyncControlLoop
from sglang.srt.disaggregation.pvd.cuda_prefetch_request import CUDAPrefetchRequest
from sglang.srt.disaggregation.pvd.cuda_refresh_driver import CUDARefreshDriver
from sglang.srt.disaggregation.pvd.cuda_request_release import CUDARequestRelease
from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeRefresher
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferCapacityError,
)
from test_pvd_cuda_received_prompt import install, received


def pump(driver, until):
    deadline = time.monotonic() + 5
    while not until():
        assert time.monotonic() < deadline, driver.snapshot()
        driver.poll()
        time.sleep(0.001)


@contextmanager
def bound(monkeypatch, *, claim=True, provisional=False, import_prompt=True):
    c = received(monkeypatch)
    if not provisional:
        install(c)
    driver = CUDARefreshDriver(c.arbiter, max_requests=2, max_prefix_tokens=32)
    controller = CUDAPrefetchRequest.__new__(CUDAPrefetchRequest)
    controller.group = c.group
    controller.pipeline = NS(
        _lock=c.c.lock,
        _quarantined=False,
        probe=NS(_quarantined=False),
        provider=NS(degraded=False),
        draft_config=NS(predict_tokens=2),
    )
    controller._session = NS(
        _copy_unknown=False, observe=lambda n: None, close=lambda: None
    )
    controller._active = controller._ready = None
    controller._tasks, controller._closed, controller.delivery = (), False, None
    controller._metadata, controller._routes = {0: {"prompt_tokens": 4}}, {0: ()}
    events, sparse_done, lease_done = [], Future(), Future()

    async def close_controller():
        events.append("sparse close started")
        await asyncio.wrap_future(sparse_done)
        c.group.close()
        events.append("sparse drained")

    controller.aclose = close_controller
    driver.register(
        c.request,
        controller,
        clients={0: object()},
        timeout_seconds=20,
        initial_import_pending=provisional,
        initial_session=c.session if provisional else None,
        pool_owner=c.c.owner if provisional else None,
    )
    c.allocator.device = "cpu"
    retirement = CUDARequestRelease(
        c.request,
        driver,
        c.cache,
        pool_owner=c.c.owner,
        release=lambda *a: None,
    )
    control = _AsyncControlLoop()
    c.manager.control = control
    c.manager.pending_decode_closes = []
    c.manager.close_bootstrap_gate = lambda req: c.gate.close()
    c.manager.gather_rank_objects = lambda value: [value]
    c.session.synchronize = lambda: events.append("full sync")
    c.session.receive_guard = ResourceGuard(
        object(), lambda: events.append("full MR released")
    )

    async def release_consumer(key, consumer_id):
        events.append("lease close started")
        await asyncio.wrap_future(lease_done)
        events.append("lease closed")

    c.session.client = NS(release_consumer=release_consumer)

    def retire():
        assert c.session._close_future.result() is True
        assert events[-1] == "lease closed"
        events.append("request pool returned")
        retirement.pool_owner.unpin(retirement._pin)
        retirement.state = "released"

    retirement.release_after_controller_close = retire
    if provisional and import_prompt:
        install(c)
    if claim:
        driver.claim_received_session(c.session)
    result = NS(**locals())
    try:
        yield result
    finally:
        # All payload/model/MR boundaries in this fixture are controlled CPU
        # doubles. Finish them to close test loops, NEVER a production recovery.
        for future in (sparse_done, lease_done):
            if not future.done():
                future.set_result(None)
        driver.begin_shutdown()
        pump(
            driver,
            lambda: (
                not driver._records
                or all(r.quarantined for r in driver._records.values())
            ),
        )
        if driver._records:
            assert not asyncio.all_tasks(driver._loop)
            driver._loop.close()  # Production close_loop refuses quarantine.
        else:
            driver.close_loop()
        control.loop.call_soon_threadsafe(control.loop.stop)
        control.thread.join(timeout=5)
        assert not control.thread.is_alive()
        control.loop.close()


def test_claim_disables_old_full_refresh_but_keeps_consumer_lease(monkeypatch):
    with bound(monkeypatch) as b:
        c = b.c
        c.request.output_ids.extend([6] * 4)
        assert not c.session.due()
        with pytest.raises(RuntimeError, match="ownership transferred"):
            c.session.prepare(c.session.pages)
        refresher = PVDDecodeRefresher(c.manager)
        assert refresher.refresh([c.request]) == []
        c.session.lease_error = "lease lost"
        # The CUDA driver owns this failure too; old refresh must submit nothing.
        assert refresher.refresh([c.request]) == []
        assert not c.session._closed and not b.events
        b.driver.poll()
        assert b.driver.snapshot()["requests"]["r"]["stopping"]


def test_provisional_registration_is_idle_until_receiver_import(monkeypatch):
    with bound(monkeypatch, claim=False, provisional=True, import_prompt=False) as b:
        record = b.driver._records["r"]
        assert record.provisional and record.full_session is None
        assert b.driver.snapshot()["requests"]["r"]["owns_provisional_source"]
        assert record.provisional_source is b.c.session
        assert b.c.session._cuda_refresh_driver is b.driver
        assert record.retirement is b.retirement
        assert not b.c.group.can_decode(0)
        for _ in range(3):
            b.driver.poll()
        assert not record.stopping and record.refresh is None
        assert not b.events
        install(b.c)
        b.driver.claim_received_session(b.c.session)
        assert not record.provisional and record.provisional_source is None
        assert record.full_session is b.c.session


def test_provisional_precopy_capacity_refusal_can_drain(monkeypatch):
    with bound(monkeypatch, claim=False, provisional=True, import_prompt=False) as b:
        b.c.budget.reserve("other", 65536, 0)
        with pytest.raises(TransferCapacityError):
            install(b.c)
        assert not b.c.importer._used and not b.c.importer._quarantined
        assert b.c.session._cuda_prompt_importer is None
        b.driver.cancel(b.c.request)
        b.sparse_done.set_result(None)
        b.lease_done.set_result(None)
        pump(b.driver, lambda: not b.driver._records)
        assert b.events[-1] == "request pool returned"


def test_provisional_successful_import_requires_exact_claim(monkeypatch):
    with bound(monkeypatch, claim=False, provisional=True) as b:
        original = b.c.importer._received_pool_owner
        b.c.importer._received_pool_owner = object()
        with pytest.raises(ValueError, match="release owner required"):
            b.driver.claim_received_session(b.c.session)
        assert b.driver._records["r"].provisional
        assert b.c.session._cuda_refresh_driver is b.driver
        b.c.importer._received_pool_owner = original
        b.driver.claim_received_session(b.c.session)
        assert not b.driver._records["r"].provisional


def test_preimport_cancel_drains_both_owners_before_pool_return(monkeypatch):
    with bound(monkeypatch, claim=False, provisional=True, import_prompt=False) as b:
        b.driver.cancel(b.c.request)
        pump(b.driver, lambda: "sparse close started" in b.events)
        assert b.retirement.state == "attached"
        assert b.c.session.receive_guard.value is not None
        b.sparse_done.set_result(None)
        pump(b.driver, lambda: "lease close started" in b.events)
        assert "request pool returned" not in b.events
        b.lease_done.set_result(None)
        pump(b.driver, lambda: not b.driver._records)
        assert b.events[-2:] == ["lease closed", "request pool returned"]


def test_unknown_preclaim_import_quarantines_provisional_owners(monkeypatch):
    with bound(monkeypatch, claim=False, provisional=True, import_prompt=False) as b:

        def unknown_completion():
            raise RuntimeError("GPU Prompt completion unknown")

        monkeypatch.setattr(b.c.importer, "_synchronize", unknown_completion)
        with pytest.raises(RuntimeError, match="initial Prompt completion unknown"):
            install(b.c)
        assert b.c.importer._quarantined and b.c.importer._receive_lease is not None
        b.driver.cancel(b.c.request, "initial Prompt completion unknown")
        record = b.driver._records["r"]
        assert record.quarantined and record.provisional
        assert record.provisional_source is b.c.session
        assert record.provisional_pool_owner is b.c.c.owner
        assert b.retirement.state == "attached"
        assert b.c.session.receive_guard.value is not None
        assert b.driver.arbiter.busy
        assert b.c.allocator.pvd_cuda_retirement_error
        with pytest.raises(ValueError, match="driver quarantined"):
            b.driver.poll()


def test_provisional_controller_cancel_failure_poison_pools(monkeypatch):
    with bound(monkeypatch, claim=False, provisional=True, import_prompt=False) as b:

        def fail_cancel(reason):
            raise RuntimeError("sparse cancel completion unknown")

        b.controller.cancel = fail_cancel
        with pytest.raises(RuntimeError, match="sparse cancel completion unknown"):
            b.driver.cancel(b.c.request)
        record = b.driver._records["r"]
        assert record.quarantined and record.provisional
        assert record.provisional_source is b.c.session
        assert b.retirement.state == "attached"
        assert b.c.session.receive_guard.value is not None
        assert b.c.allocator.pvd_cuda_retirement_error
        assert b.driver.arbiter.busy


@pytest.mark.parametrize("provisional", [False, True])
@pytest.mark.parametrize("finish", [False, True])
def test_original_release_intent_waits_for_sparse_then_full_session_close(
    monkeypatch, finish, provisional
):
    with bound(monkeypatch, provisional=provisional) as b:
        c = b.c
        refresher = PVDDecodeRefresher(c.manager)
        if finish:
            c.request.finished = lambda: True
            refresher.cleanup_finished()
        else:
            refresher.release_request(c.request)
        pump(b.driver, lambda: "sparse close started" in b.events)
        assert c.manager.decode_sessions[c.key] is c.session
        assert not c.session._closed and c.session.receive_guard.value is not None
        assert not c.gate.is_runnable
        b.sparse_done.set_result(None)
        pump(b.driver, lambda: "lease close started" in b.events)
        assert b.events == [
            "sparse close started",
            "sparse drained",
            "full sync",
            "full MR released",
            "lease close started",
        ]
        assert "r" in b.driver._records
        assert b.retirement.state == "attached"
        b.lease_done.set_result(None)
        pump(b.driver, lambda: not b.driver._records)
        assert b.events[-2:] == ["lease closed", "request pool returned"]
        assert c.key not in c.manager.decode_sessions
        assert c.session._cuda_refresh_driver is None


@pytest.mark.parametrize(
    "fault",
    ["missing-retirement", "other-pool-owner", "other-lock", "other-import", "busy"],
)
def test_claim_requires_exact_completed_assembly(monkeypatch, fault):
    with bound(monkeypatch, claim=False) as b:
        record = b.driver._records["r"]
        lease = None
        original = record.retirement
        if fault == "missing-retirement":
            record.retirement = None
        elif fault == "other-pool-owner":
            b.c.importer._received_pool_owner = object()
        elif fault == "other-lock":
            b.c.importer._lock = object()
        elif fault == "other-import":
            b.c.importer._received_receipt = object()
        else:
            lease = b.driver.arbiter.acquire()
        try:
            with pytest.raises(ValueError, match="release owner required"):
                b.driver.claim_received_session(b.c.session)
            assert record.full_session is None
            assert b.c.session._cuda_refresh_driver is None
        finally:
            if lease is not None:
                b.driver.arbiter.release(lease)
            # Unclaimed fixture: no source close is owed by the driver, and
            # this test's retirement double assumes a source was claimed.
            original.release_after_controller_close = lambda: original.pool_owner.unpin(
                original._pin
            )
            record.retirement = original


def test_duplicate_claim_is_refused_without_resetting_clock(monkeypatch):
    with bound(monkeypatch) as b:
        with pytest.raises(ValueError, match="release owner required"):
            b.driver.claim_received_session(b.c.session)
        assert b.c.session.clock.round == 1
        assert b.driver._records["r"].full_session is b.c.session


def test_sparse_close_failure_keeps_full_mr_lease_and_admission(monkeypatch):
    with bound(monkeypatch) as b:
        b.sparse_done.set_exception(RuntimeError("remote writer unknown"))
        b.driver.cancel(b.c.request)
        pump(b.driver, lambda: b.driver._records["r"].quarantined)
        assert not b.c.session._closed
        assert b.c.session.receive_guard.value is not None
        assert b.events == ["sparse close started"]
        assert b.c.manager.decode_sessions[b.c.key] is b.c.session
        assert b.retirement.state == "attached"
        assert b.driver.arbiter.busy
        assert b.driver.snapshot()["source_quarantine"]


def test_full_close_failure_does_not_return_request_pool(monkeypatch):
    with bound(monkeypatch) as b:

        def unknown():
            raise RuntimeError("full source completion unknown")

        b.c.session.synchronize = unknown
        b.sparse_done.set_result(None)
        b.driver.cancel(b.c.request)
        pump(b.driver, lambda: b.driver._records["r"].quarantined)
        assert "full MR released" not in b.events
        assert "request pool returned" not in b.events
        assert b.c.session.receive_guard.value is not None
        assert b.c.manager.decode_sessions[b.c.key] is b.c.session
        assert b.driver.arbiter.busy
        assert "quarantined" in b.c.allocator.pvd_cuda_retirement_error
        with pytest.raises(ValueError, match="driver quarantined"):
            b.driver.poll()


def test_replaced_session_registry_cannot_release_the_original_owner(monkeypatch):
    with bound(monkeypatch) as b:
        b.c.manager.decode_sessions[b.c.key] = object()
        b.driver.poll()
        assert b.driver._records["r"].stopping
        b.sparse_done.set_result(None)
        pump(b.driver, lambda: b.driver._records["r"].quarantined)
        assert b.c.session._close_future is None
        assert b.retirement.state == "attached"
        assert b.driver.arbiter.busy
