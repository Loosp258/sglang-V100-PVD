"""Execution tickets protect the install boundary, not native resource lifetimes."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError
from test_pvd_rank_install_runtime import case, complete, post


def ready():
    c = case()
    complete(c)
    c.sent.clear()
    return c


def test_single_dispatch_no_token_commit_and_exactly_once_retirement():
    c = ready()
    permit = c.r.begin_forward(0)
    assert permit.identity == ("r", "inc", "entry")
    assert permit.committed_tokens == 0
    assert not c.r.can_decode(0)
    with pytest.raises(InstallProtocolError, match="wait or abort"):
        c.r.begin_forward(0)
    with pytest.raises(InstallProtocolError, match="idle"):
        c.r.begin(3, timeout_seconds=5)
    assert c.r.finish_forward(permit, readers_drained=True, succeeded=True)
    assert c.r.can_decode(0)  # result owner, not this gate, advances the token count
    with pytest.raises(InstallProtocolError, match="stale or foreign"):
        c.r.finish_forward(permit, readers_drained=True, succeeded=True)


def test_prepared_and_parked_cannot_install_while_host_forward_still_owned():
    c = ready()
    epoch = c.r.begin(3, timeout_seconds=5)
    permit = c.r.begin_forward(3)
    for rank in (0, 1):
        post(c, "prepared", epoch, rank)
        post(c, "parked", epoch, rank)
    c.r.progress()
    assert c.r.snapshot()["phase"] == "preparing"
    assert not c.sent
    assert c.r.finish_forward(permit, readers_drained=True, succeeded=True)
    assert not c.sent  # completion drains callbacks before retiring its ticket
    c.r.progress()
    assert len(c.sent) == 2
    assert c.r.snapshot()["phase"] == "installing"


@pytest.mark.parametrize(
    "failure", ["cancel", "timeout", "lost_peer", "bad_event", "forward"]
)
def test_failure_retains_execution_until_drain_and_discards_late_output(failure):
    c = ready()
    epoch = c.r.begin(3, timeout_seconds=5)
    permit = c.r.begin_forward(3)
    if failure == "cancel":
        c.r.cancel()
    elif failure == "timeout":
        c.time[0] = 15
    elif failure == "lost_peer":
        c.r.peer_lost(peer_rank=0, peer_epoch="worker-0")
    elif failure == "bad_event":
        post(c, "applied", epoch)
    c.r.progress()
    assert c.r.snapshot()["forward_operation_id"] == permit.operation_id
    assert not c.r.can_decode(3)
    assert not c.r.finish_forward(
        permit, readers_drained=True, succeeded=failure != "forward"
    )
    assert c.r.snapshot()["forward_operation_id"] is None
    assert c.r.snapshot()["phase"] == "failed"
    assert {rank for rank, _, _ in c.stopped} == {0, 1}


@pytest.mark.parametrize(
    "change", ["copy", "other_request", "undrained", "nonbool", "thread"]
)
def test_wrong_completion_never_retires_or_reuses_execution(change):
    c = ready()
    permit = c.r.begin_forward(0)
    given, drained, succeeded = permit, True, True
    if change == "copy":
        given = replace(permit)
    elif change == "other_request":
        given = ready().r.begin_forward(0)
    elif change == "undrained":
        drained = False
    elif change == "nonbool":
        succeeded = 1
    with pytest.raises(InstallProtocolError):
        if change == "thread":
            with ThreadPoolExecutor(1) as executor:
                executor.submit(
                    c.r.finish_forward,
                    given,
                    readers_drained=drained,
                    succeeded=succeeded,
                ).result()
        else:
            c.r.finish_forward(given, readers_drained=drained, succeeded=succeeded)
    assert c.r.snapshot()["forward_operation_id"] == permit.operation_id
    assert c.r.finish_forward(permit, readers_drained=True, succeeded=True)


def test_queued_failure_is_processed_before_accepting_target_output():
    c = ready()
    epoch = c.r.begin(3, timeout_seconds=5)
    permit = c.r.begin_forward(3)
    post(c, "prepared", epoch)
    post(c, "failed", epoch)
    assert not c.r.finish_forward(permit, readers_drained=True, succeeded=True)
    assert c.r.snapshot()["phase"] == "failed"
    assert not c.sent
