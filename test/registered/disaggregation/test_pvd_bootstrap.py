"""Waiting-queue-triggered initial KV pull: request-local gating only.

These are CPU/logic tests. They prove ordering rules, not RDMA, GPU
visibility or real scheduler integration. Nothing here establishes that a
transfer happened or that memory is safe to reuse.
"""

import pytest
from sglang.srt.disaggregation.pvd.bootstrap import (
    BootstrapGate,
    BootstrapState,
    BootstrapTicket,
)

EPOCH = "d-worker-epoch"


def make_gate(delivery_id="req-1:bootstrap", prompt_tokens=5) -> BootstrapGate:
    return BootstrapGate(delivery_id, EPOCH, prompt_tokens)


def ready_gate() -> BootstrapGate:
    gate = make_gate()
    gate.enter_waiting_queue()
    gate.mark_source_ready()
    return gate


def installed_gate() -> BootstrapGate:
    gate = ready_gate()
    ticket = gate.begin()
    gate.mark_received(ticket)
    gate.mark_installed(ticket)
    return gate


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ("", EPOCH, 5),
        ("  ", EPOCH, 5),
        (None, EPOCH, 5),
        ("d", "", 5),
        ("d", None, 5),
        ("d", EPOCH, 0),
        ("d", EPOCH, -1),
        ("d", EPOCH, True),
        ("d", EPOCH, 1.0),
    ],
)
def test_invalid_identity_is_rejected(args):
    with pytest.raises(ValueError):
        BootstrapGate(*args)


def test_a_fresh_gate_is_queued_and_not_runnable():
    gate = make_gate()
    assert gate.state is BootstrapState.QUEUED
    assert not gate.is_runnable
    assert gate.ticket is None
    assert not gate.can_request()


# --------------------------------------------------------------------------
# The trigger is the final waiting queue, and only that
# --------------------------------------------------------------------------


def test_pull_requires_the_final_waiting_queue():
    gate = make_gate()
    gate.mark_source_ready()
    assert not gate.can_request()
    with pytest.raises(ValueError, match="final waiting queue"):
        gate.begin()
    assert gate.state is BootstrapState.QUEUED
    assert gate.ticket is None


def test_pull_requires_a_stored_entry_on_v():
    gate = make_gate()
    gate.enter_waiting_queue()
    assert not gate.can_request()
    with pytest.raises(ValueError, match="stored Entry"):
        gate.begin()
    assert gate.ticket is None


@pytest.mark.parametrize("source_first", [False, True])
def test_waiting_queue_and_kv_stored_may_arrive_in_either_order(source_first):
    gate = make_gate()
    if source_first:
        gate.mark_source_ready()
        gate.enter_waiting_queue()
    else:
        gate.enter_waiting_queue()
        gate.mark_source_ready()
    assert gate.can_request()
    ticket = gate.begin()
    assert ticket.delivery_id == "req-1:bootstrap"
    assert ticket.receiver_epoch == EPOCH
    assert ticket.prompt_tokens == 5
    assert gate.state is BootstrapState.AUTHORIZED


def test_entering_the_waiting_queue_twice_before_the_pull_is_harmless():
    gate = make_gate()
    gate.enter_waiting_queue()
    gate.enter_waiting_queue()
    gate.mark_source_ready()
    gate.mark_source_ready()
    assert gate.can_request()


def test_reentering_the_waiting_queue_after_the_pull_is_refused():
    gate = ready_gate()
    gate.begin()
    with pytest.raises(ValueError, match="already left the waiting-queue stage"):
        gate.enter_waiting_queue()


# --------------------------------------------------------------------------
# Exactly one pull per request
# --------------------------------------------------------------------------


def test_only_one_authorization_is_ever_issued():
    gate = ready_gate()
    gate.begin()
    with pytest.raises(ValueError, match="already been authorized"):
        gate.begin()


def test_a_duplicate_control_request_cannot_authorize_a_second_write():
    """Retries must be idempotent: reuse the first ticket, never mint another."""
    gate = ready_gate()
    first = gate.begin()
    for _ in range(3):
        with pytest.raises(ValueError):
            gate.begin()
        assert gate.ticket is first


def test_bootstrap_is_not_repeated_after_admission():
    gate = installed_gate()
    assert gate.handoff() == 0
    assert gate.handed_off
    with pytest.raises(ValueError, match="already been handed off"):
        gate.handoff()
    with pytest.raises(ValueError, match="already been authorized"):
        gate.begin()


# --------------------------------------------------------------------------
# Arrival is not readiness; readiness is not admission
# --------------------------------------------------------------------------


def test_received_is_not_runnable():
    gate = ready_gate()
    ticket = gate.begin()
    gate.mark_received(ticket)
    assert gate.state is BootstrapState.RECEIVED
    assert not gate.is_runnable
    with pytest.raises(ValueError, match="not installed"):
        gate.handoff()


def test_authorized_cannot_jump_straight_to_installed():
    gate = ready_gate()
    ticket = gate.begin()
    with pytest.raises(ValueError, match="has not been received"):
        gate.mark_installed(ticket)
    assert gate.state is BootstrapState.AUTHORIZED
    assert not gate.is_runnable


def test_only_installed_is_runnable():
    gate = make_gate()
    assert not gate.is_runnable
    gate.enter_waiting_queue()
    gate.mark_source_ready()
    assert not gate.is_runnable
    ticket = gate.begin()
    assert not gate.is_runnable
    gate.mark_received(ticket)
    assert not gate.is_runnable
    gate.mark_installed(ticket)
    assert gate.is_runnable


def test_duplicate_delivery_notifications_are_harmless():
    gate = ready_gate()
    ticket = gate.begin()
    gate.mark_received(ticket)
    gate.mark_received(ticket)
    assert gate.state is BootstrapState.RECEIVED
    gate.mark_installed(ticket)
    with pytest.raises(ValueError, match="has not been received"):
        gate.mark_installed(ticket)
    assert gate.is_runnable


def test_the_refresh_clock_starts_at_zero_committed_tokens():
    assert installed_gate().handoff() == 0


# --------------------------------------------------------------------------
# Stale and foreign completions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forged",
    [
        BootstrapTicket("other-req:bootstrap", EPOCH, 5),
        BootstrapTicket("req-1:bootstrap", "a-restarted-d-epoch", 5),
        BootstrapTicket("req-1:bootstrap", EPOCH, 6),
        "req-1:bootstrap",
        None,
    ],
)
def test_foreign_completions_are_refused(forged):
    gate = ready_gate()
    gate.begin()
    with pytest.raises(ValueError, match="stale or foreign"):
        gate.mark_received(forged)
    assert gate.state is BootstrapState.AUTHORIZED


def test_a_completion_before_any_authorization_is_refused():
    gate = ready_gate()
    with pytest.raises(ValueError, match="stale or foreign"):
        gate.mark_received(BootstrapTicket("req-1:bootstrap", EPOCH, 5))


# --------------------------------------------------------------------------
# Cancellation retains; it never releases
# --------------------------------------------------------------------------


def test_close_retains_the_pending_identity():
    gate = ready_gate()
    ticket = gate.begin()
    gate.close()
    assert gate.state is BootstrapState.CLOSED
    assert gate.ticket == ticket
    assert not gate.is_runnable


def test_a_late_write_after_cancellation_never_reopens_the_request():
    gate = ready_gate()
    ticket = gate.begin()
    gate.close()
    for call in (gate.mark_received, gate.mark_installed):
        with pytest.raises(ValueError, match="closed"):
            call(ticket)
    assert gate.state is BootstrapState.CLOSED
    assert not gate.is_runnable


@pytest.mark.parametrize(
    "call",
    [
        lambda g: g.enter_waiting_queue(),
        lambda g: g.mark_source_ready(),
        lambda g: g.begin(),
        lambda g: g.handoff(),
    ],
)
def test_a_closed_gate_refuses_every_transition(call):
    gate = ready_gate()
    gate.begin()
    gate.close()
    with pytest.raises(ValueError, match="closed"):
        call(gate)


def test_cancelling_an_installed_request_still_refuses_handoff():
    gate = installed_gate()
    gate.close()
    with pytest.raises(ValueError, match="closed"):
        gate.handoff()
    assert not gate.is_runnable


# --------------------------------------------------------------------------
# Request isolation
# --------------------------------------------------------------------------


def test_one_requests_bootstrap_does_not_touch_another():
    a = make_gate("req-a:bootstrap")
    b = make_gate("req-b:bootstrap")
    b.enter_waiting_queue()
    b.mark_source_ready()
    b_ticket = b.begin()

    a.enter_waiting_queue()
    a.mark_source_ready()
    a_ticket = a.begin()
    a.close()

    assert b.state is BootstrapState.AUTHORIZED
    assert b.ticket == b_ticket
    b.mark_received(b_ticket)
    b.mark_installed(b_ticket)
    assert b.is_runnable and b.handoff() == 0
    assert not a.is_runnable
    with pytest.raises(ValueError, match="stale or foreign"):
        b.mark_received(a_ticket)
