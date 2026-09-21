"""Real rank runtimes, independent clocks, wait-all and single-writer decisions."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace

import pytest
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import TargetExecutionArbiter
from sglang.srt.disaggregation.pvd.cpu_rank_install import CPURankInstallParticipant
from sglang.srt.disaggregation.pvd.rank_batch_dispatch import (
    RankBatchDispatcher,
    RankBatchMember,
)
from sglang.srt.disaggregation.pvd.rank_install_runtime import RankInstallRuntime
from sglang.srt.disaggregation.pvd.rank_install_wire import RankInstallExchange
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError
from test_pvd_rank_install_runtime import complete, post
from test_pvd_sparse_install import cpu_group, payload, protocol


def request(name):
    c = SimpleNamespace(time=[10.0], sent=[], stopped=[])
    c.x = RankInstallExchange(
        protocol(request=name), peer_epochs={r: f"worker-{r}" for r in (0, 1)}
    )
    c.r = RankInstallRuntime(
        c.x,
        send=lambda rank, raw: c.sent.append((rank, raw)),
        stop_peer=lambda *args: c.stopped.append(args),
        max_pending_events=16,
        max_pending_bytes=16384,
        clock=lambda: c.time[0],
    )
    return c


def setup():
    a, b = request("a"), request("b")
    complete(a)
    complete(b)
    a.sent.clear()
    b.sent.clear()
    batch = RankBatchDispatcher(TargetExecutionArbiter(), max_requests=2)
    members = (RankBatchMember(a.r, 0), RankBatchMember(b.r, 0))
    return batch, a, b, members


def owned(batch, a, b):
    assert batch.arbiter.busy
    assert a.r.snapshot()["forward_operation_id"]
    assert b.r.snapshot()["forward_operation_id"]


def retired(batch, a, b):
    assert not batch.arbiter.busy
    assert a.r.snapshot()["forward_operation_id"] is None
    assert b.r.snapshot()["forward_operation_id"] is None


def test_wait_all_including_new_request_leaves_old_round_untouched():
    batch, a, _, _ = setup()
    a.r.begin(3, timeout_seconds=5)
    before = a.r.snapshot()
    newcomer = request("new")
    members = (RankBatchMember(a.r, 3), RankBatchMember(newcomer.r, 0))
    with pytest.raises(InstallProtocolError, match="as a whole"):
        batch.begin(members)
    assert a.r.snapshot() == before
    assert not batch.arbiter.busy
    complete(newcomer)
    ticket = batch.begin(members)
    assert tuple(p.committed_tokens for p in ticket.members) == (3, 0)
    assert a.r.snapshot()["deadline"] == before["deadline"]
    with batch.processing(ticket, readers_drained=True, succeeded=True) as decisions:
        assert all(d.accepted for d in decisions)
    assert a.r.can_decode(3)  # no private shadow token clock


def test_boundaries_and_every_rank_resume_are_wait_all_gates():
    batch, a, b, _ = setup()
    epoch = a.r.begin(3, timeout_seconds=5)
    members = (RankBatchMember(a.r, 4), RankBatchMember(b.r, 1))
    for rank in (0, 1):
        post(a, "prepared", epoch, rank)
        post(a, "parked", epoch, rank)
    a.r.progress()
    for rank in (0, 1):
        post(a, "applied", epoch, rank)
    a.r.progress()
    post(a, "resumed", epoch, 0)
    a.r.progress()
    with pytest.raises(InstallProtocolError, match="as a whole"):
        batch.begin(members)
    retired(batch, a, b)
    post(a, "resumed", epoch, 1)
    a.r.progress()
    ticket = batch.begin(members)
    with batch.processing(ticket, readers_drained=True, succeeded=True) as decisions:
        assert all(d.accepted for d in decisions)


def test_forward_and_result_scope_block_install_and_direct_single_completion():
    batch, a, b, _ = setup()
    epoch = a.r.begin(3, timeout_seconds=5)
    ticket = batch.begin((RankBatchMember(a.r, 3), RankBatchMember(b.r, 1)))
    for rank in (0, 1):
        post(a, "prepared", epoch, rank)
        post(a, "parked", epoch, rank)
    a.r.progress()
    assert not a.sent
    with pytest.raises(InstallProtocolError, match="batch-owned"):
        a.r.finish_forward(ticket.members[0], readers_drained=True, succeeded=True)
    with batch.processing(ticket, readers_drained=True, succeeded=True) as decisions:
        owned(batch, a, b)
        # Adversarial poll: even this cannot install during the result scope.
        a.r.progress()
        assert not a.sent
        assert [d.permit.identity[0] for d in decisions] == ["a", "b"]
        assert all(d.accepted for d in decisions)
        with pytest.raises(InstallProtocolError, match="reentered"):
            batch.begin(())
    retired(batch, a, b)
    a.r.progress()
    assert len(a.sent) == 2


@pytest.mark.parametrize("failure", ["cancel", "loss", "timeout", "event", "forward"])
def test_only_live_results_accepted_and_no_cancellation_releases_execution(failure):
    batch, a, b, _ = setup()
    epoch = a.r.begin(3, timeout_seconds=5)
    ticket = batch.begin((RankBatchMember(a.r, 3), RankBatchMember(b.r, 0)))
    if failure == "cancel":
        a.r.cancel()
    elif failure == "loss":
        a.r.peer_lost(peer_rank=0, peer_epoch="worker-0")
    elif failure == "timeout":
        a.time[0] = 15
    elif failure == "event":
        post(a, "applied", epoch)
    owned(batch, a, b)
    # Stands for the existing result writer, not a new production token writer.
    outputs = {"a": [], "b": []}
    with batch.processing(
        ticket, readers_drained=True, succeeded=failure != "forward"
    ) as decisions:
        assert [d.accepted for d in decisions] == [False, failure != "forward"]
        for decision, token in zip(decisions, (11, 22), strict=True):
            if decision.accepted:
                outputs[decision.permit.identity[0]].append(token)
    assert outputs == {"a": [], "b": [] if failure == "forward" else [22]}
    retired(batch, a, b)


@pytest.mark.parametrize("invalid", ["copy", "none", "drain", "status", "thread"])
def test_bad_completion_keeps_entire_ownership_and_can_retry(invalid):
    batch, a, b, members = setup()
    ticket = batch.begin(members)

    def attempt():
        given = (
            replace(ticket)
            if invalid == "copy"
            else None
            if invalid == "none"
            else ticket
        )
        with batch.processing(
            given,
            readers_drained=invalid != "drain",
            succeeded=1 if invalid == "status" else True,
        ):
            pytest.fail("invalid completion reached result writer")

    with pytest.raises((InstallProtocolError, ValueError)):
        if invalid == "thread":
            with ThreadPoolExecutor(1) as worker:
                worker.submit(attempt).result()
        else:
            attempt()
    owned(batch, a, b)
    with batch.processing(ticket, readers_drained=True, succeeded=True):
        pass
    retired(batch, a, b)
    with (
        pytest.raises(InstallProtocolError, match="stale or foreign"),
        batch.processing(ticket, readers_drained=True, succeeded=True),
    ):
        pass


def test_result_processor_exception_aborts_all_without_undoing_written_tokens():
    batch, a, b, members = setup()
    ticket = batch.begin(members)
    committed = []
    with (
        pytest.raises(RuntimeError, match="processor"),
        batch.processing(ticket, readers_drained=True, succeeded=True),
    ):
        committed.append(42)
        raise RuntimeError("processor failed after one write")
    assert committed == [42]
    retired(batch, a, b)
    assert a.r.snapshot()["phase"] == b.r.snapshot()["phase"] == "failed"


def test_second_acquisition_fault_rolls_back_only_undispatched_metadata(monkeypatch):
    batch, a, b, members = setup()
    begin = b.r.begin_forward

    def raced(count):
        b.r.peer_lost(peer_rank=0, peer_epoch="worker-0")
        return begin(count)

    monkeypatch.setattr(b.r, "begin_forward", raced)
    with pytest.raises(InstallProtocolError, match="wait or abort"):
        batch.begin(members)
    retired(batch, a, b)
    assert a.r.can_decode(0)
    assert a.r._forward_batch is None
    assert not a.stopped


@pytest.mark.parametrize(
    "invalid", ["empty", "list", "capacity", "duplicate", "same_id", "count", "workers"]
)
def test_invalid_membership_fails_before_any_acquisition(invalid):
    batch, a, b, members = setup()
    if invalid == "empty":
        members = ()
    elif invalid == "list":
        members = list(members)
    elif invalid == "capacity":
        members = members + (members[0],)
    elif invalid == "duplicate":
        members = (members[0], members[0])
    elif invalid == "same_id":
        members = (members[0], RankBatchMember(request("a").r, 0))
    elif invalid == "count":
        members = (members[0], RankBatchMember(b.r, True))
    elif invalid == "workers":
        b.x.peer_epochs = {0: "different-worker", 1: "worker-1"}
    with pytest.raises(InstallProtocolError):
        batch.begin(members)
    retired(batch, a, b)


def test_existing_shared_target_lease_blocks_admission_without_touching_requests():
    batch, a, b, members = setup()
    lease = batch.arbiter.acquire()
    with pytest.raises(InstallProtocolError, match="already owned"):
        batch.begin(members)
    assert a.r.can_decode(0) and b.r.can_decode(0)
    batch.arbiter.release(lease)
    retired(batch, a, b)


def test_all_member_ownership_checked_before_any_completion_is_retired():
    batch, a, b, members = setup()
    ticket = batch.begin(members)
    original = b.r._forward
    b.r._forward = replace(original)
    with (
        pytest.raises(InstallProtocolError, match="foreign"),
        batch.processing(ticket, readers_drained=True, succeeded=True),
    ):
        pytest.fail("foreign second member reached result writer")
    owned(batch, a, b)
    b.r._forward = original
    with batch.processing(ticket, readers_drained=True, succeeded=True):
        pass
    retired(batch, a, b)


def test_result_preparation_exception_aborts_all_drained_members(monkeypatch):
    batch, a, b, members = setup()
    ticket = batch.begin(members)

    def broken(*args, **kwargs):
        raise RuntimeError("preparation failure")

    monkeypatch.setattr(b.r, "_prepare_forward_result", broken)
    with (
        pytest.raises(RuntimeError, match="preparation failure"),
        batch.processing(ticket, readers_drained=True, succeeded=True),
    ):
        pytest.fail("partial decisions must not reach writer")
    retired(batch, a, b)
    assert a.r.snapshot()["phase"] == b.r.snapshot()["phase"] == "failed"


def test_single_forward_stop_callback_cannot_reenter_its_completion():
    c = request("single")
    complete(c)
    permit = c.r.begin_forward(0)
    calls = []

    def reenter(rank, *_):
        calls.append(rank)
        with pytest.raises(InstallProtocolError, match="foreign"):
            c.r.finish_forward(permit, readers_drained=True, succeeded=False)

    c.r._stop = reenter
    assert not c.r.finish_forward(permit, readers_drained=True, succeeded=False)
    assert calls == [0, 1]
    assert c.r.snapshot()["forward_operation_id"] is None


def test_real_cpu_banks_share_the_exact_runtime_epochs_and_wait_for_result_scope():
    """Actual bank readers, not a second unrelated coordinator or model forward."""
    cases = []
    for name in ("a", "b"):
        c = request(name)
        group, banks, c.budgets = cpu_group(request=name)
        # The runtime and participants below control these EXACT banks.
        c.r.exchange = RankInstallExchange(
            group.coordinator, peer_epochs={r: f"worker-{r}" for r in banks}
        )
        c.peers = {
            r: CPURankInstallParticipant(
                bank, rank=r, peer_epoch=f"worker-{r}", interval=4
            )
            for r, bank in banks.items()
        }
        cases.append(c)

    def enqueue(c, rank, raw):
        assert c.r.post(raw, peer_rank=rank, peer_epoch=f"worker-{rank}")

    def stage(c, epoch):
        for rank, peer in c.peers.items():
            packed = payload(epoch, rank)
            try:
                enqueue(c, rank, peer.stage(epoch, [packed]))
            finally:
                packed.close()

    def pump(c):
        for _ in range(3):
            c.r.progress()
            while c.sent:
                rank, raw = c.sent.pop(0)
                enqueue(c, rank, c.peers[rank].command(raw))

    try:
        for c in cases:
            epoch = c.r.begin(0, timeout_seconds=5)
            stage(c, epoch)
            for rank, peer in c.peers.items():
                enqueue(c, rank, peer.park(0))
            pump(c)
            assert c.r.can_decode(0)

        a, b = cases
        epoch = a.r.begin(3, timeout_seconds=5)
        batch = RankBatchDispatcher(TargetExecutionArbiter(), max_requests=2)
        ticket = batch.begin((RankBatchMember(a.r, 3), RankBatchMember(b.r, 0)))
        with ExitStack() as readers:
            for c, count in ((a, 3), (b, 0)):
                for peer in c.peers.values():
                    groups = readers.enter_context(peer.read(count))
                    assert next(iter(groups.values()))[0].target_tokens == 0
            stage(a, epoch)
            for peer in a.peers.values():
                assert peer.park(4) is None  # real CPU bank has live readers
            a.r.progress()
            assert not a.sent
        for rank, peer in a.peers.items():
            enqueue(a, rank, peer.park(4))
        a.r.progress()
        assert not a.sent  # readers drained, but host forward still owned
        with batch.processing(ticket, readers_drained=True, succeeded=True) as rows:
            assert all(row.accepted for row in rows)
            owned(batch, a, b)
            a.r.progress()
            assert not a.sent  # result processor still owns batch
        retired(batch, a, b)
        pump(a)
        for peer in a.peers.values():
            with peer.read(4) as groups:
                spec = next(iter(groups.values()))[0]
                assert spec.operation_id == epoch.operation_id
                assert spec.token_ids == (1, 3)
        # B was not refreshed just because it joined A's batch.
        for peer in b.peers.values():
            with peer.read(0) as groups:
                assert next(iter(groups.values()))[0].token_ids == (0, 1, 2, 3)
    finally:
        for c in cases:
            for peer in c.peers.values():
                peer.close()
            assert all(
                budget.snapshot()["used_staging_bytes"] == 0
                for budget in c.budgets.values()
            )
