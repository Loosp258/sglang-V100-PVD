"""Owner-polled selected-V lookup; no CUDA buffer or serving factory claimed."""

import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from sglang.srt.disaggregation.pvd.conn import PVDSelectedRouteBinding
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError
from sglang.srt.disaggregation.pvd.cuda_route_discovery import (
    CUDARouteDiscoveryQueue,
)


class Manager:
    def __init__(self):
        self.started = []
        self.runnable = {}

    def key_for(self, req):
        return req.pvd_transfer_id

    def vector_group_for(self, req):
        return req.pvd_vector_group_id

    def bootstrap_runnable(self, req):
        return self.runnable.get(req.rid, True)

    def start_selected_cuda_routes(self, req):
        future = Future()
        self.started.append((req, future))
        return future


def request(rid, *, entry=None):
    return SimpleNamespace(
        rid=rid,
        pvd_transfer_id=entry or rid,
        pvd_vector_group_id="chosen",
        pvd_delivery_id=rid + ":delivery",
        is_retracted=False,
        finished=lambda: False,
    )


def binding(manager, req):
    return PVDSelectedRouteBinding(
        manager,
        req,
        req.rid,
        manager.key_for(req),
        manager.vector_group_for(req),
        req.pvd_delivery_id,
        object(),
    )


def test_waiting_queue_starts_only_after_initial_prompt_and_is_bounded():
    manager = Manager()
    queue = CUDARouteDiscoveryQueue(manager, max_inflight=1)
    first, second = request("first"), request("second")
    manager.runnable["first"] = False
    assert queue.poll([first, second]) == []
    assert [req for req, _ in manager.started] == [second]
    assert queue.ready_for(first) is None
    assert queue.poll([first, second]) == []
    assert len(manager.started) == 1
    manager.started[0][1].set_result(binding(manager, second))
    assert queue.poll([first, second]) == []
    assert queue.ready_for(second).req is second
    assert queue.poll([first]) == []
    assert not queue.pending
    manager.runnable["first"] = True
    assert queue.poll([first]) == []
    assert [req for req, _ in manager.started] == [second, first]


def test_identity_change_cancels_old_result_and_never_publishes_it():
    manager = Manager()
    queue = CUDARouteDiscoveryQueue(manager, max_inflight=1)
    req = request("one")
    queue.poll([req])
    future = manager.started[0][1]
    assert future.set_running_or_notify_cancel()
    req.pvd_vector_group_id = "another"
    assert queue.poll([req]) == [(req, "selected V request identity changed")]
    assert queue.ready_for(req) is None
    assert not future.cancelled()  # Running HTTP is retained until it settles.
    future.set_result(binding(manager, req))
    # The caller removes the failed Req from waiting before its next poll.
    assert queue.poll([]) == []
    assert not queue.pending


def test_abandoned_lookup_blocks_same_rid_successor_until_terminal():
    manager = Manager()
    queue = CUDARouteDiscoveryQueue(manager, max_inflight=2)
    old = request("same", entry="old")
    queue.poll([old])
    future = manager.started[0][1]
    successor = request("same", entry="new")
    assert queue.poll([successor]) == []
    assert not future.cancelled() and not future.done()
    assert len(manager.started) == 1
    future.set_result(binding(manager, old))
    assert queue.poll([successor]) == []
    assert [req for req, _ in manager.started] == [old, successor]


def test_bad_result_and_http_failure_are_reported_once_without_same_poll_retry():
    manager = Manager()
    queue = CUDARouteDiscoveryQueue(manager, max_inflight=2)
    one, two = request("one"), request("two")
    queue.poll([one, two])
    manager.started[0][1].set_result(binding(manager, two))
    manager.started[1][1].set_exception(RuntimeError("V unavailable"))
    failures = queue.poll([one, two])
    assert [req for req, _ in failures] == [one, two]
    assert "identity" in failures[0][1]
    assert "V unavailable" in failures[1][1]
    assert len(manager.started) == 2


def test_close_retains_running_lookup_until_terminal_and_refuses_new_work():
    manager = Manager()
    queue = CUDARouteDiscoveryQueue(manager, max_inflight=1)
    req = request("one")
    queue.poll([req])
    future = manager.started[0][1]
    assert not queue.close() and queue.pending
    assert not future.cancelled() and not future.done()
    future.set_result(binding(manager, req))
    assert queue.close() and not queue.pending
    assert queue.poll([request("two")]) == []
    assert len(manager.started) == 1


def test_route_queue_refuses_cross_thread_poll_and_invalid_limit():
    manager = Manager()
    with pytest.raises(LifecycleError, match="positive"):
        CUDARouteDiscoveryQueue(manager, max_inflight=0)
    queue = CUDARouteDiscoveryQueue(manager, max_inflight=1)
    errors = []
    thread = threading.Thread(target=lambda: _poll_in_thread(queue, errors))
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert isinstance(errors[0], LifecycleError)


def _poll_in_thread(queue, errors):
    try:
        queue.poll(())
    except BaseException as exc:
        errors.append(exc)
