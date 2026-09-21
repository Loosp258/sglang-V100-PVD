"""Bounded callback inbox, owner progress, fixed deadlines and request stop scope."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from sglang.srt.disaggregation.pvd.rank_install_runtime import RankInstallRuntime
from sglang.srt.disaggregation.pvd.rank_install_wire import RankInstallMessage
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError
from test_pvd_rank_install_wire import event, exchange


def case(**overrides):
    x = exchange()
    time, sent, stopped = [10.0], [], []
    options = {
        "send": lambda rank, raw: sent.append((rank, raw)),
        "stop_peer": lambda rank, identity, reason: stopped.append(
            (rank, identity, reason)
        ),
        "max_pending_events": 16,
        "max_pending_bytes": 16384,
        "clock": lambda: time[0],
    }
    options.update(overrides)
    runtime = RankInstallRuntime(x, **options)
    return SimpleNamespace(r=runtime, x=x, time=time, sent=sent, stopped=stopped)


def post(c, kind, epoch, rank=0):
    return c.r.post(
        event(kind, epoch, rank).encode(), peer_rank=rank, peer_epoch=f"worker-{rank}"
    )


def prepared(c, epoch):
    for rank in (0, 1):
        assert post(c, "prepared", epoch, rank)
        assert post(c, "parked", epoch, rank)
    c.r.progress()
    assert [RankInstallMessage.decode(raw).kind for _, raw in c.sent] == [
        "install",
        "install",
    ]


def complete(c, count=0):
    epoch = c.r.begin(count, timeout_seconds=5)
    prepared(c, epoch)
    for rank in (0, 1):
        post(c, "applied", epoch, rank)
    c.r.progress()
    assert c.r.snapshot()["phase"] == "resuming"
    assert not c.r.can_decode(epoch.target_tokens)
    for rank in (0, 1):
        post(c, "resumed", epoch, rank)
    c.r.progress()
    assert c.r.snapshot()["phase"] == "idle"
    assert c.r.can_decode(epoch.target_tokens)
    return epoch


def test_drives_both_barriers_and_resume_without_resetting_independent_request():
    c = case()
    previous = complete(c)
    c.sent.clear()
    pending = c.r.begin(3, timeout_seconds=5)
    before = c.r.snapshot()
    other = case()
    complete(other)
    assert c.r.snapshot() == before
    assert c.r.can_decode(3) and not c.r.can_decode(4)
    for kind in ("prepared", "parked", "applied", "resumed"):
        post(c, kind, previous)
    c.r.progress()  # exact completed-round replays don't poison the next round
    assert not c.sent
    prepared(c, pending)
    assert not c.r.can_decode(4)


def test_callbacks_do_not_mutate_coordinator_and_poll_is_bounded():
    c = case()
    epoch = c.r.begin(0, timeout_seconds=5)
    with ThreadPoolExecutor(2) as threads:
        assert threads.submit(post, c, "prepared", epoch, 0).result()
        assert threads.submit(post, c, "prepared", epoch, 1).result()
        with pytest.raises(InstallProtocolError, match="owner thread"):
            threads.submit(c.r.progress).result()
    assert c.x.coordinator.snapshot()["prepared"] == ()
    c.r.progress(max_events=1)
    assert c.x.coordinator.snapshot()["prepared"] == (0,)
    assert c.r.snapshot()["pending_events"] == 1
    assert not c.r.can_decode(0)
    c.r.progress(max_events=1)
    assert c.r.snapshot()["pending_bytes"] == 0


@pytest.mark.parametrize("phase", ["preparing", "installing", "resuming"])
def test_fixed_deadline_includes_lost_resume_acks_and_stops_all_peers(phase):
    c = case()
    epoch = c.r.begin(0, timeout_seconds=5)
    if phase != "preparing":
        prepared(c, epoch)
    if phase == "resuming":
        for rank in (0, 1):
            post(c, "applied", epoch, rank)
        c.r.progress()
        post(c, "resumed", epoch, 0)
        c.r.progress()
    c.time[0] = 14
    if phase == "preparing":
        post(c, "prepared", epoch)
    c.r.progress()
    assert c.r.snapshot()["deadline"] == 15
    c.time[0] = 15
    assert not c.r.can_decode(0)
    c.r.progress()
    assert c.r.snapshot()["phase"] == "failed"
    assert {rank for rank, _, _ in c.stopped} == {0, 1}
    assert all(identity == ("r", "inc", "entry") for _, identity, _ in c.stopped)
    assert not c.r.snapshot()["resource_cleanup_proven"]
    with pytest.raises(InstallProtocolError, match="idle"):
        c.r.begin(0, timeout_seconds=5)


@pytest.mark.parametrize("bound", ["events", "bytes", "frame"])
def test_capacity_or_invalid_frame_latches_failure_without_discarding_safety(bound):
    c = (
        case(max_pending_events=1)
        if bound == "events"
        else case(max_pending_bytes=1)
        if bound == "bytes"
        else case()
    )
    epoch = c.r.begin(0, timeout_seconds=5)
    if bound == "events":
        assert post(c, "prepared", epoch)
    if bound == "frame":
        assert not c.r.post(b"", peer_rank=0, peer_epoch="worker-0")
    else:
        assert not post(c, "prepared", epoch)
    assert not c.r.can_decode(0)
    c.r.progress()
    assert c.r.snapshot()["pending_events"] == 0
    assert c.r.snapshot()["pending_bytes"] == 0
    assert len(c.stopped) == 2


def test_stale_channel_callback_cannot_fail_new_incarnation_but_bound_loss_can():
    c = case()
    complete(c)
    assert not c.r.peer_lost(peer_rank=0, peer_epoch="old-worker")
    assert not c.r.post(b"broken", peer_rank=True, peer_epoch="worker-1")
    assert c.r.can_decode(1)
    with ThreadPoolExecutor(1) as thread:
        assert thread.submit(c.r.peer_lost, peer_rank=0, peer_epoch="worker-0").result()
    assert not c.r.can_decode(1)
    c.r.progress()
    assert len(c.stopped) == 2


def test_send_exception_after_partial_publication_stops_even_unprepared_peers():
    sent = []

    def send(rank, raw):
        sent.append(rank)
        if rank == 1:
            raise RuntimeError("queued then lost response")

    c = case(send=send)
    epoch = c.r.begin(0, timeout_seconds=5)
    for rank in (0, 1):
        post(c, "prepared", epoch, rank)
        post(c, "parked", epoch, rank)
    c.r.progress()
    assert sent == [0, 1]
    assert c.r.snapshot()["reason"] == "rank-control send failed"
    assert len(c.stopped) == 2
    assert not c.r.can_decode(0)


def test_stop_enqueue_failure_remains_visible_and_retries_without_reopening():
    attempts = []

    def stop(rank, *_):
        attempts.append(rank)
        if rank == 1 and attempts.count(1) == 1:
            raise RuntimeError("stop queue busy")

    c = case(stop_peer=stop)
    c.r.cancel()  # no PREPARED received, still notify every bound peer
    assert c.r.snapshot()["stop_notifications_pending"] == (1,)
    c.r.progress()
    assert attempts == [0, 1, 1]
    assert c.r.snapshot()["stop_notifications_pending"] == ()
    assert not c.r.can_decode(0)


@pytest.mark.parametrize("raw", [b"{}", b"{broken"])
def test_malformed_bound_events_fail_request(raw):
    c = case()
    c.r.begin(0, timeout_seconds=5)
    assert c.r.post(raw, peer_rank=0, peer_epoch="worker-0")
    c.r.progress()
    assert c.r.snapshot()["reason"] == "invalid rank installation event"
    assert len(c.stopped) == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 9.0])
def test_bad_or_backwards_clock_closes_gate(value):
    c = case()
    c.r.begin(0, timeout_seconds=5)
    c.time[0] = value
    c.r.progress()
    assert c.r.snapshot()["reason"] == "rank-control clock failed"


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_timeout_must_be_explicit_positive_finite(timeout):
    c = case()
    with pytest.raises(InstallProtocolError, match="finite positive"):
        c.r.begin(0, timeout_seconds=timeout)
    assert c.x.coordinator.snapshot()["epoch"] is None


def test_pending_failure_is_processed_before_install_commands():
    c = case()
    epoch = c.r.begin(0, timeout_seconds=5)
    for rank in (0, 1):
        post(c, "prepared", epoch, rank)
        post(c, "parked", epoch, rank)
    post(c, "failed", epoch, 1)
    c.r.progress(max_events=4)
    assert not c.sent  # remaining event could be a failure, don't dispatch past it
    assert not c.r.can_decode(0)
    c.r.progress()
    assert c.r.snapshot()["reason"] == "rank reported installation failure"
    assert not c.sent and len(c.stopped) == 2


def test_reentrant_progress_from_send_cannot_advance_state_machine_twice():
    holder = {}
    c = case(send=lambda *_: holder["r"].progress())
    holder["r"] = c.r
    epoch = c.r.begin(0, timeout_seconds=5)
    for rank in (0, 1):
        post(c, "prepared", epoch, rank)
        post(c, "parked", epoch, rank)
    c.r.progress()
    assert c.r.snapshot()["reason"] == "rank-control send failed"
    assert not c.r.can_decode(0)


def test_stop_callback_reentry_does_not_recurse_or_repeat_notifications():
    holder, calls = {}, []

    def stop(rank, *_):
        calls.append(rank)
        holder["r"].cancel()

    c = case(stop_peer=stop)
    holder["r"] = c.r
    c.r.cancel()
    assert calls == [0, 1]
    assert c.r.snapshot()["stop_notifications_pending"] == ()


def test_clock_exception_fails_closed_and_notifies_all_peers():
    c = case()
    c.r.begin(0, timeout_seconds=5)

    def bad_clock():
        raise OSError("clock provider failed")

    c.r._clock = bad_clock
    c.r.progress()
    assert c.r.snapshot()["reason"] == "rank-control clock failed"
    assert len(c.stopped) == 2


def test_future_or_foreign_request_event_does_not_advance_current_round():
    c = case()
    epoch = c.r.begin(0, timeout_seconds=5)
    other = exchange().begin(0)  # different operation identity
    assert epoch != other
    post(c, "prepared", other)
    c.r.progress()
    assert c.x.coordinator.snapshot()["installed_tokens"] is None
    assert c.r.snapshot()["phase"] == "failed"


def test_full_resume_clears_deadline_before_later_idle_poll():
    c = case()
    complete(c)
    c.time[0] = 100
    c.r.progress()
    assert c.r.snapshot()["deadline"] is None
    assert c.r.can_decode(1)
    assert not c.stopped
