"""CPU-only timing tests; these do not demonstrate model or RDMA overlap."""

from dataclasses import replace

import pytest
from sglang.srt.disaggregation.pvd.prefetch import PrefetchClock, PrefetchState


def initialized(prefix="request-a", interval=16, lead=4):
    clock = PrefetchClock(prefix, interval, lead)
    initial = clock.begin(0)
    clock.mark_ready(initial)
    clock.commit_install(initial, 0)
    return clock


def test_initial_kv_blocks_first_forward_until_installed():
    clock = PrefetchClock("request-a", 16, 4)
    assert clock.requires_install(0)
    assert clock.needs_prefetch(0)
    ticket = clock.begin(0)
    assert ticket.target_tokens == ticket.prefix_tokens == 0
    assert not clock.can_install(ticket, 0)
    with pytest.raises(ValueError, match="not ready"):
        clock.commit_install(ticket, 0)
    clock.mark_ready(ticket)
    assert clock.requires_install(0)
    assert clock.installed_tokens is None
    clock.commit_install(ticket, 0)
    assert clock.boundary == 16
    assert not clock.requires_install(0)


@pytest.mark.parametrize("ready_at", [12, 14, 16])
def test_ready_does_not_advance_period_or_install_early(ready_at):
    clock = initialized()
    assert not clock.needs_prefetch(11)
    with pytest.raises(ValueError, match="lead window"):
        clock.begin(11)
    ticket = clock.begin(12)
    assert (ticket.prefix_tokens, ticket.target_tokens, ticket.round) == (12, 16, 1)
    assert clock.requires_install(ready_at) == (ready_at == 16)
    clock.mark_ready(ticket)
    clock.mark_ready(ticket)  # Matching duplicate notification is idempotent.
    assert clock.round == 1
    assert clock.installed_tokens == 0
    assert clock.boundary == 16
    assert not clock.needs_prefetch(ready_at)
    if ready_at < 16:
        assert not clock.can_install(ticket, ready_at)
        with pytest.raises(ValueError, match="not ready"):
            clock.commit_install(ticket, ready_at)
    assert clock.requires_install(16)
    clock.commit_install(ticket, 16)
    assert (clock.installed_tokens, clock.boundary, clock.round) == (16, 32, 2)
    assert not clock.needs_prefetch(27)
    assert clock.needs_prefetch(28)


def test_late_delivery_waits_without_silently_advancing_decode():
    clock = initialized()
    ticket = clock.begin(12)
    assert clock.requires_install(16)
    assert not clock.can_install(ticket, 16)
    with pytest.raises(ValueError, match="past an uninstalled"):
        clock.requires_install(17)
    assert clock.pending == ticket
    clock.mark_ready(ticket)
    clock.commit_install(ticket, 16)
    assert clock.boundary == 32


def test_late_start_can_run_synchronously_at_boundary():
    clock = initialized()
    ticket = clock.begin(16)
    assert ticket.target_tokens == 16
    clock.mark_ready(ticket)
    clock.commit_install(ticket, 16)
    assert clock.boundary == 32


@pytest.mark.parametrize("ready", [False, True])
def test_new_request_does_not_reset_or_cancel_existing_prefetch(ready):
    old = initialized("old")
    ticket = old.begin(12)
    if ready:
        old.mark_ready(ticket)
    before = (old.state, old.round, old.installed_tokens, old.pending, old.boundary)
    new = initialized("new")
    new.close()
    assert (
        old.state,
        old.round,
        old.installed_tokens,
        old.pending,
        old.boundary,
    ) == before
    old.mark_ready(ticket)
    old.commit_install(ticket, 16)
    assert old.boundary == 32


def test_batch_wait_set_is_only_due_requests_not_all_members():
    clocks = {
        "a": initialized("a"),
        "b": initialized("b"),
        "c": initialized("c"),
        "new": PrefetchClock("new", 16, 4),
    }
    counts = {"a": 10, "b": 16, "c": 3, "new": 0}
    due = [
        name for name, clock in clocks.items() if clock.requires_install(counts[name])
    ]
    assert due == ["b", "new"]
    for name in due:
        ticket = clocks[name].begin(counts[name])
        clocks[name].mark_ready(ticket)
        clocks[name].commit_install(ticket, counts[name])
    assert clocks["a"].round == clocks["c"].round == 1
    assert clocks["a"].needs_prefetch(12)
    assert not clocks["c"].needs_prefetch(4)


def test_foreign_stale_and_modified_tickets_cannot_install():
    clock = initialized()
    ticket = clock.begin(12)
    for wrong in (
        None,
        replace(ticket, delivery_id="other:prefetch:1"),
        replace(ticket, round=2),
        replace(ticket, prefix_tokens=11),
        replace(ticket, target_tokens=32),
    ):
        with pytest.raises(ValueError, match="stale or foreign"):
            clock.mark_ready(wrong)
    clock.mark_ready(ticket)
    clock.commit_install(ticket, 16)
    later = clock.begin(28)
    with pytest.raises(ValueError, match="stale or foreign"):
        clock.mark_ready(ticket)
    assert clock.pending == later
    assert clock.state == PrefetchState.IN_FLIGHT


def test_single_pending_and_regression_checks():
    clock = initialized()
    ticket = clock.begin(12)
    with pytest.raises(ValueError, match="already in flight"):
        clock.begin(12)
    assert not clock.requires_install(15)
    with pytest.raises(ValueError, match="regressed"):
        clock.requires_install(14)
    clock.mark_ready(ticket)
    with pytest.raises(ValueError, match="regressed"):
        clock.commit_install(ticket, 12)
    clock.commit_install(ticket, 16)


@pytest.mark.parametrize("ready", [False, True])
def test_close_keeps_identity_and_never_authorizes_buffer_reuse(ready):
    clock = initialized()
    ticket = clock.begin(12)
    if ready:
        clock.mark_ready(ticket)
    clock.close()
    clock.close()
    assert clock.pending == ticket
    assert clock.installed_tokens == 0
    for operation in (
        lambda: clock.mark_ready(ticket),
        lambda: clock.commit_install(ticket, 16),
        lambda: clock.needs_prefetch(12),
        lambda: clock.requires_install(16),
        lambda: clock.begin(16),
    ):
        with pytest.raises(ValueError, match="closed"):
            operation()


def test_zero_lead_is_synchronous_control_case():
    clock = initialized(lead=0)
    assert not clock.needs_prefetch(15)
    assert clock.needs_prefetch(16)


@pytest.mark.parametrize("value", [True, False, -1, 1.5, "1", None])
def test_counts_must_be_committed_nonnegative_integers(value):
    with pytest.raises(ValueError, match="non-negative integer"):
        initialized().needs_prefetch(value)


@pytest.mark.parametrize("interval", [0, -1, True, "16", 1.5, None])
def test_bad_interval(interval):
    with pytest.raises(ValueError, match="interval"):
        PrefetchClock("a", interval, 0)


@pytest.mark.parametrize("lead", [-1, 16, 17, True, "4", 1.5, None])
def test_bad_lead(lead):
    with pytest.raises(ValueError, match="lead_tokens"):
        PrefetchClock("a", 16, lead)


@pytest.mark.parametrize("prefix", [None, "", " ", 1])
def test_bad_prefix(prefix):
    with pytest.raises(ValueError, match="delivery_prefix"):
        PrefetchClock(prefix, 16, 4)


def test_existing_refresher_only_fetches_due_and_new_requests():
    """Real refresher/legacy clocks; fake delivery/session data operations.

    This is a selection/control regression, not a TP or transport test.
    """
    import asyncio
    from concurrent.futures import Future
    from types import SimpleNamespace

    import torch
    from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeRefresher
    from sglang.srt.disaggregation.pvd.protocol import KVEntryKey
    from sglang.srt.disaggregation.pvd.retrieval import RefreshClock

    calls, installed, released = [], [], []

    class Client:
        def __init__(self, group):
            self.group = group

        async def retrieve(self, sequences):
            calls.append((self.group, [s["sequence_id"] for s in sequences]))
            return {"results": [dict(s, state="delivered") for s in sequences]}

        async def ack_delivery(self, delivery_id):
            return {"delivery_id": delivery_id, "state": "released"}

    class Control:
        def submit(self, coroutine):
            future = Future()
            try:
                future.set_result(asyncio.run(coroutine))
            except Exception as exc:
                future.set_exception(exc)
            return future

    clients = {name: Client(name) for name in ("v0", "v1")}
    sessions = {}
    for index, (name, count) in enumerate((("a", 10), ("b", 16), ("c", 3), ("new", 0))):
        key = KVEntryKey("model", name, name)
        clock = RefreshClock(name, 16)
        if name != "new":
            clock.complete(clock.begin(0))
        req = SimpleNamespace(
            key=key,
            origin_input_ids=list(range(5)),
            req_pool_idx=index,
            group="v1" if name == "new" else "v0",
        )
        session = SimpleNamespace(
            req=req,
            key=key,
            clock=clock,
            decode_tokens=count,
            lease_error=None,
            client=clients[req.group],
            adopt_identities=lambda reply: None,
            unpack=lambda reply: installed.append(reply["sequence_id"]),
            release_refresh=lambda name=name: released.append(name),
        )
        session.due = lambda s=session: s.clock.due(s.decode_tokens)

        def prepare(pages, s=session):
            return {
                "sequence_id": s.key.req_id,
                "delivery_id": s.clock.begin(s.decode_tokens),
                "destination": {"fake_descriptor": True},
            }

        session.prepare = prepare
        sessions[key] = session
    manager = SimpleNamespace(
        decode_sessions=sessions,
        key_for=lambda req: req.key,
        vector_group_for=lambda req: req.group,
        page_size=4,
        tp_rank=0,
        gather_rank_objects=lambda payload: [payload],
        clients=clients,
        control=Control(),
        scheduler=SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(32).reshape(4, 8),
            )
        ),
    )
    refresher = PVDDecodeRefresher(manager)
    reqs = [session.req for session in sessions.values()]
    assert refresher.refresh(reqs) == []
    assert sorted(calls) == [("v0", ["b"]), ("v1", ["new"])]
    assert installed == released == ["b", "new"]
    assert [s.clock.last_tokens for s in sessions.values()] == [0, 16, 0, 0]
    assert [s.clock.round for s in sessions.values()] == [1, 2, 1, 1]
    assert refresher.refresh(reqs) == []
    assert len(calls) == 2
