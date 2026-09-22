"""Owner-loop CUDA policies with CPU placement, real HTTP, fake payloads."""

import asyncio
import threading
import time
from array import array
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import (
    LifecycleError,
    TargetExecutionArbiter,
)
from sglang.srt.disaggregation.pvd.cuda_prefetch_request import CUDAPrefetchRequest
from sglang.srt.disaggregation.pvd.cuda_refresh_driver import CUDARefreshDriver
from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient
from test_pvd_cuda_prefetch_request import controller
from test_pvd_cuda_sparse_delivery import case, complete


def req():
    return SimpleNamespace(
        rid="consumer",
        origin_input_ids=array("q", [1] * 8),
        output_ids=array("q", [2]),
        req_pool_idx=1,
        is_retracted=False,
        finished=lambda: False,
    )


def finish_writes(c):
    for delivery in tuple(c.store.entries[c.entry.key].deliveries.values()):
        handle = delivery.transfer_handle
        if handle and not handle.transport_state.is_locally_safe_to_release:
            c.engine.finish(handle)
    c.store.progress_transfers()


def pump(driver, c, until, *, finish=True):
    deadline = time.monotonic() + 5
    while not until():
        assert time.monotonic() < deadline, driver.snapshot()
        if finish:
            finish_writes(c)
        driver.poll()
        time.sleep(0.001)


@contextmanager
def synchronous(monkeypatch, req_factory=req):
    driver = CUDARefreshDriver(
        TargetExecutionArbiter(), max_requests=2, max_prefix_tokens=64
    )
    ctx = case(monkeypatch)
    c = driver._loop.run_until_complete(ctx.__aenter__())
    driver._loop.run_until_complete(complete(c, 0, tuple(range(8))))
    control, captures, _ = controller(c, monkeypatch)
    client = PVDShardSearchClient(c.client.base_url)
    request = req_factory()
    driver.register(request, control, clients={0: client}, timeout_seconds=20)
    try:
        yield driver, c, control, request, captures
    finally:
        driver.begin_shutdown()
        pump(
            driver,
            c,
            lambda: (
                not driver._records
                or all(r.quarantined for r in driver._records.values())
            ),
        )
        driver._loop.run_until_complete(client.close())
        driver._loop.run_until_complete(ctx.__aexit__(None, None, None))
        if not driver._records:
            driver.close_loop()
        else:
            # CPU fault doubles only: assert no live async work, close fixture
            # loop directly. Production close_loop MUST retain quarantine.
            assert not asyncio.all_tasks(driver._loop)
            driver._loop.close()


def test_sync_owner_loop_runs_two_http_rounds_from_actual_req_counts(monkeypatch):
    with synchronous(monkeypatch) as (driver, c, control, request, captures):
        owner = threading.get_ident()
        capture_threads = []
        original = control.pipeline.probe.capture

        def capture(*args):
            capture_threads.append(threading.get_ident())
            return original(*args)

        monkeypatch.setattr(control.pipeline.probe, "capture", capture)
        for boundary in (4, 8):
            request.output_ids.extend([3] * (boundary - len(request.output_ids)))
            driver.poll()
            assert driver.arbiter.busy  # Queued capture owns the exact snapshot.
            record = driver._records[request.rid]
            pump(driver, c, lambda: record.ready)
            assert len(request.output_ids) - 1 == boundary - 1
            assert control.pending_install_boundary == boundary
            assert not control.can_decode(boundary)
            assert not driver.arbiter.busy  # HTTP wait holds no target lease.
            request.output_ids.append(
                4
            )  # Only the real result processor would write this.
            pump(driver, c, lambda: record.refresh is None)
            assert control.can_decode(boundary)
            assert (
                driver.snapshot()["requests"]["consumer"]["committed_tokens"]
                == boundary
            )
        # The probe captures both configured draft tokens; the query bridge
        # selects the boundary row (12 / 16) from those captured positions.
        assert captures == [(12, 13), (16, 17)]
        assert capture_threads == [owner, owner]


def test_missed_window_probes_current_committed_prefix_without_draft(monkeypatch):
    with synchronous(monkeypatch) as (driver, c, control, request, captures):

        def no_draft(*args, **kwargs):
            raise AssertionError("fallback must not run draft")

        monkeypatch.setattr(control.pipeline.provider, "branch", no_draft)
        request.output_ids.extend([3] * 4)
        driver.poll()
        pump(
            driver,
            c,
            lambda: control.group.coordinator.snapshot()["installed_tokens"] == 4,
        )
        assert captures == [(12,)]
        assert len(request.output_ids) == 5


@pytest.mark.parametrize("change", ["prefix", "slot", "output", "cross_boundary"])
def test_mutated_req_cannot_start_refresh_or_advance_clock(monkeypatch, change):
    with synchronous(monkeypatch) as (driver, c, control, request, captures):
        if change == "prefix":
            request.origin_input_ids[0] = 99
        elif change == "slot":
            request.req_pool_idx = 2
        elif change == "output":
            request.output_ids[0] = 99
        else:
            request.output_ids.extend([3] * 5)
        driver.poll()
        assert driver._records["consumer"].stopping
        assert captures == []
        assert control.group.coordinator.snapshot()["installed_tokens"] == 0


def test_poll_cannot_enter_inside_target_forward_or_from_another_thread(monkeypatch):
    with synchronous(monkeypatch) as (driver, c, control, request, captures):
        lease = driver.arbiter.acquire()
        with pytest.raises(LifecycleError, match="between target"):
            driver.poll()
        driver.arbiter.release(lease)
        errors = []

        def foreign():
            try:
                driver.poll()
            except LifecycleError as exc:
                errors.append(str(exc))

        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join()
        assert errors and "owner thread" in errors[0]
        assert captures == []


def test_cancel_before_capture_dispatch_returns_lease_without_predicting(monkeypatch):
    with synchronous(monkeypatch) as (driver, c, control, request, captures):
        request.output_ids.extend([3] * 3)
        driver.poll()
        assert driver.arbiter.busy
        driver.cancel(request)
        pump(driver, c, lambda: not driver._records)
        assert not driver.arbiter.busy and captures == []


def test_cleanup_unknown_retains_registration_capacity_and_refuses_loop_close(
    monkeypatch,
):
    with synchronous(monkeypatch) as (driver, c, control, request, captures):

        async def unknown():
            raise RuntimeError("native owners remain")

        monkeypatch.setattr(control, "aclose", unknown)
        driver.begin_shutdown()
        pump(driver, c, lambda: driver._records["consumer"].quarantined)
        with pytest.raises(LifecycleError, match="owners must drain"):
            driver.close_loop()
        record = driver._records["consumer"]
        task = record.close_task
        for _ in range(3):
            driver.poll()
        assert record.close_task is task  # No automatic unknown cleanup retry.
        assert record.req is request and record.controller is control


def test_probe_unknown_keeps_shared_arbiter_not_only_reentrant_lock(monkeypatch):
    with synchronous(monkeypatch) as (driver, c, control, request, captures):

        def unknown(*args):
            control.pipeline.probe._quarantined = True
            raise RuntimeError("probe native completion unknown")

        monkeypatch.setattr(control.pipeline.probe, "capture", unknown)
        request.output_ids.extend([3] * 3)
        driver.poll()
        pump(driver, c, lambda: driver._records["consumer"].quarantined)
        assert driver.arbiter.busy
        with pytest.raises(LifecycleError, match="busy"):
            driver.arbiter.acquire()


def test_registration_bounds_and_foreign_request_cancellation(monkeypatch):
    with synchronous(monkeypatch) as (driver, c, control, request, captures):
        with pytest.raises(LifecycleError, match="unique live"):
            driver.register(request, control, clients={0: object()}, timeout_seconds=1)
        with pytest.raises(LifecycleError, match="incarnation"):
            driver.cancel(req())
        assert driver.snapshot()["requests"]["consumer"]["committed_tokens"] == 0


def test_new_request_does_not_restart_old_prefetch_or_clock(monkeypatch):
    with synchronous(monkeypatch) as (driver, c, control, request, captures):
        request.output_ids.extend([3] * 3)
        driver.poll()
        first = driver._records["consumer"]
        task, deadline = first.refresh, first.deadline
        # Admission-only policy double; the first request still runs the real
        # HTTP/controller path. This peer performs no model or transport work.
        peer = object.__new__(CUDAPrefetchRequest)
        peer.pipeline = control.pipeline
        peer._session = SimpleNamespace(_copy_unknown=False)
        peer._active = peer._ready = None
        peer._tasks = ()
        peer._metadata = {0: {"prompt_tokens": 8}}
        peer._routes = {0: None}
        peer.group = SimpleNamespace(
            coordinator=SimpleNamespace(
                identity=("new", "new-inc", "new-entry"),
                _owner=lambda: None,
                snapshot=lambda: {
                    "state": "idle",
                    "installed_tokens": 0,
                    "next_boundary": 4,
                    "lead_tokens": 1,
                },
            )
        )
        peer.can_decode = lambda n: True
        peer._live = lambda: None
        peer.cancel = lambda reason: None

        async def close():
            return None

        peer.aclose = close
        second = req()
        second.rid, second.req_pool_idx = "new", 2
        driver.register(second, peer, clients={0: object()}, timeout_seconds=10)
        assert first.refresh is task and first.deadline == deadline
        assert len(first.outputs) - 1 == 3
        pump(driver, c, lambda: first.ready)
        assert len(captures) == 1
        assert driver._records["new"].refresh is None
        request.output_ids.append(4)
        pump(driver, c, lambda: first.refresh is None)
        assert driver.snapshot()["requests"]["new"]["committed_tokens"] == 0


def test_late_existing_prefetch_waits_at_boundary_without_requery(monkeypatch):
    with synchronous(monkeypatch) as (driver, c, control, request, captures):
        request.output_ids.extend([3] * 3)
        driver.poll()
        record = driver._records["consumer"]
        original = record.refresh
        pump(
            driver,
            c,
            lambda: bool(control.delivery.snapshot()["retained_destinations"]),
            finish=False,
        )
        request.output_ids.append(4)
        for _ in range(5):
            driver.poll()
        assert record.refresh is original and not record.ready
        assert len(captures) == 1
        assert not control.can_decode(4)
        pump(driver, c, lambda: record.refresh is None)
        assert control.can_decode(4) and len(captures) == 1


def test_queued_snapshot_is_revalidated_before_any_capture(monkeypatch):
    with synchronous(monkeypatch) as (driver, c, control, request, captures):
        request.output_ids.extend([3] * 3)
        driver.poll()
        request.output_ids.append(4)  # Illegal write while capture owns arbiter.
        pump(driver, c, lambda: not driver._records)
        assert captures == []
        assert not driver.arbiter.busy


def test_timeout_stops_refresh_without_changing_authoritative_outputs(monkeypatch):
    with synchronous(monkeypatch) as (driver, c, control, request, captures):
        now = [10.0]
        driver._clock = lambda: now[0]
        request.output_ids.extend([3] * 3)
        driver.poll()
        record = driver._records["consumer"]
        pump(
            driver,
            c,
            lambda: bool(control.delivery.snapshot()["retained_destinations"]),
            finish=False,
        )
        now[0] = record.deadline
        driver.poll()
        assert record.stopping and "timeout" in str(record.error)
        assert tuple(request.output_ids) == (2, 3, 3, 3)


def test_async_owner_loop_can_be_borrowed_without_being_closed(monkeypatch):
    async def run():
        async with case(monkeypatch) as c:
            await complete(c, 0, tuple(range(8)))
            control, _, _ = controller(c, monkeypatch)
            driver = CUDARefreshDriver(
                TargetExecutionArbiter(), max_requests=1, max_prefix_tokens=32
            )
            client = PVDShardSearchClient(c.client.base_url)
            try:
                driver.register(req(), control, clients={0: client}, timeout_seconds=2)
                assert not driver._owns_loop
                driver.begin_shutdown()
                for _ in range(50):
                    driver.poll()
                    if driver.snapshot()["drained"]:
                        break
                    await asyncio.sleep(0.001)
                assert driver.snapshot()["drained"]
                driver.close_loop()
                assert not asyncio.get_running_loop().is_closed()
            finally:
                await client.close()

    asyncio.run(run())


def test_real_req_array_fields_are_observed_without_token_writes(monkeypatch):
    try:
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams
    except Exception as exc:
        pytest.skip(f"real Req import unavailable: {type(exc).__name__}: {exc}")

    def real_req():
        sampling = SamplingParams(max_new_tokens=32, ignore_eos=True)
        sampling.normalize(None)
        value = Req("consumer", "", [1] * 8, sampling, vocab_size=32)
        value.output_ids.append(2)
        value.req_pool_idx = 1
        return value

    with synchronous(monkeypatch, real_req) as (driver, c, control, request, captures):
        assert isinstance(request.output_ids, array)
        request.output_ids.extend([3] * 3)
        driver.poll()
        pump(driver, c, lambda: driver._records["consumer"].ready)
        assert tuple(request.output_ids) == (2, 3, 3, 3)
        request.output_ids.append(4)
        pump(driver, c, lambda: driver._records["consumer"].refresh is None)
        assert control.can_decode(4)
        assert tuple(request.output_ids) == (2, 3, 3, 3, 4)
