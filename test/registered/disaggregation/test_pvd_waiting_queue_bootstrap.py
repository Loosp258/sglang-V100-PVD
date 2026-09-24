"""PVDKVManager's waiting-queue bootstrap trigger, run as real code.

The manager methods are invoked unbound against a minimal fake ``self``, so
the logic under test is the shipped implementation rather than a copy. The
refresher and the transport are fakes: nothing here proves that an RDMA WRITE
happened, that GPU memory is visible, or that the scheduler integration in
decode.py behaves correctly on real hardware.
"""

from collections import namedtuple
from types import SimpleNamespace

import pytest
from sglang.srt.disaggregation.pvd.bootstrap import BootstrapState
from sglang.srt.disaggregation.pvd.conn import PVDKVManager
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

PROMPT = [1, 2, 3, 4, 5]


class FakeRefresher:
    """Records which requests were asked for, and can fail chosen ones."""

    def __init__(self, failures=None):
        self.calls = []
        self.failures = failures or {}

    def refresh(self, reqs):
        self.calls.append(list(reqs))
        return [(r, self.failures[r.rid]) for r in reqs if r.rid in self.failures]

    bootstrap_pending = False

    def poll_bootstrap(self):
        return [], []

    def start_bootstrap(self, reqs):
        # Immediate fake completion for gate/admission unit tests. Real
        # asynchronous delivery and ACK are covered by the refresher tests.
        return list(reqs), self.refresh(reqs)


def make_req(rid="req-a"):
    return SimpleNamespace(rid=rid, origin_input_ids=list(PROMPT))


EntryKey = namedtuple("EntryKey", "transfer_id")


def make_manager(enabled=True, failures=None, staging_bytes=None, used=0, per_req=64):
    """A real PVDKVManager with only the fields these methods touch.

    __new__ without __init__ keeps the shipped method bodies and real attribute
    dispatch while avoiding the manager's GPU/transport construction.
    """
    mgr = PVDKVManager.__new__(PVDKVManager)
    mgr.waiting_queue_bootstrap = enabled
    mgr.bootstrap_gates = {}
    mgr.worker_epoch = "d-epoch"
    mgr.tp_rank = 0
    mgr.decode_refresher = FakeRefresher(failures)
    mgr.key_for = lambda req: EntryKey(f"entry-{req.rid}")
    mgr.gather_rank_objects = lambda value: [value]
    if staging_bytes is not None:
        mgr.transfer_budget = TransferBudget(
            staging_bytes=staging_bytes, max_inflight=16
        )
        if used:
            mgr.transfer_budget.reserve("someone-else", used, 0)
    mgr.local_shard_manifest = lambda tokens: SimpleNamespace(expected_bytes=per_req)
    return mgr


def test_waiting_queue_exchange_includes_rank_for_real_gather(monkeypatch):
    mgr = make_manager()
    mgr.tp_size = 1
    mgr.gloo_group = None
    mgr.gather_rank_objects = PVDKVManager.gather_rank_objects.__get__(mgr)

    def gather(out, local, *, group):
        assert group is None
        out[0] = local

    monkeypatch.setattr("torch.distributed.all_gather_object", gather)
    assert enter(mgr, []) == []


def open_gate(mgr, req):
    return mgr.open_bootstrap_gate(req)


def enter(mgr, reqs):
    return mgr.enter_waiting_queue(reqs)


def runnable(mgr, req):
    return mgr.bootstrap_runnable(req)


# --------------------------------------------------------------------------
# The flag is off by default and changes nothing
# --------------------------------------------------------------------------


def test_disabled_manager_opens_no_gate_and_never_blocks_admission():
    mgr = make_manager(enabled=False)
    req = make_req()
    assert open_gate(mgr, req) is None
    assert mgr.bootstrap_gates == {}
    assert runnable(mgr, req) is True
    assert enter(mgr, [req]) == []
    assert mgr.decode_refresher.calls == []


def test_a_request_without_a_gate_is_always_runnable():
    """Legacy PVD requests and ordinary PD are never gated by this."""
    mgr = make_manager(enabled=True)
    assert runnable(mgr, make_req("never-opened")) is True


# --------------------------------------------------------------------------
# The trigger is the waiting queue, and only that
# --------------------------------------------------------------------------


def test_a_gated_request_is_not_runnable_before_the_waiting_queue():
    mgr = make_manager()
    req = make_req()
    gate = open_gate(mgr, req)
    assert gate.state is BootstrapState.QUEUED
    assert runnable(mgr, req) is False
    assert mgr.decode_refresher.calls == []


def test_the_gate_identity_is_derived_from_the_entry_and_this_worker():
    mgr = make_manager()
    gate = open_gate(mgr, make_req("req-x"))
    assert gate.delivery_id == "entry-req-x:bootstrap"
    assert gate.receiver_epoch == "d-epoch"
    assert gate.prompt_tokens == len(PROMPT)


def test_opening_two_gates_for_one_request_is_refused():
    mgr = make_manager()
    req = make_req()
    open_gate(mgr, req)
    with pytest.raises(Exception, match="duplicate"):
        open_gate(mgr, req)


def test_reaching_the_waiting_queue_pulls_and_makes_the_request_runnable():
    mgr = make_manager()
    req = make_req()
    gate = open_gate(mgr, req)
    gate.mark_source_ready()

    assert enter(mgr, [req]) == []
    assert mgr.decode_refresher.calls == [[req]]
    assert gate.state is BootstrapState.INSTALLED
    assert gate.handed_off
    assert runnable(mgr, req) is True


def test_without_kv_stored_the_request_waits_instead_of_failing():
    """Not an error: V has not finished storing yet, so retry a later pass."""
    mgr = make_manager()
    req = make_req()
    gate = open_gate(mgr, req)

    assert enter(mgr, [req]) == []
    assert mgr.decode_refresher.calls == []
    assert gate.state is BootstrapState.QUEUED
    assert gate.in_waiting_queue
    assert runnable(mgr, req) is False

    # A later pass, once V reports KV_STORED, performs the pull.
    gate.mark_source_ready()
    assert enter(mgr, [req]) == []
    assert mgr.decode_refresher.calls == [[req]]
    assert runnable(mgr, req) is True


def test_the_pull_happens_exactly_once():
    mgr = make_manager()
    req = make_req()
    open_gate(mgr, req).mark_source_ready()
    enter(mgr, [req])
    for _ in range(3):
        assert enter(mgr, [req]) == []
    assert mgr.decode_refresher.calls == [[req]]


# --------------------------------------------------------------------------
# Failure does not install
# --------------------------------------------------------------------------


def test_a_failed_pull_is_reported_and_leaves_the_request_unrunnable():
    mgr = make_manager(failures={"req-a": "V retrieval failed"})
    req = make_req()
    gate = open_gate(mgr, req)
    gate.mark_source_ready()

    failures = enter(mgr, [req])
    assert failures == [(req, "V retrieval failed")]
    assert gate.state is BootstrapState.AUTHORIZED
    assert not gate.handed_off
    assert runnable(mgr, req) is False


def test_one_failed_pull_does_not_block_its_neighbours():
    mgr = make_manager(failures={"bad": "V retrieval failed"})
    good, bad = make_req("good"), make_req("bad")
    for req in (good, bad):
        open_gate(mgr, req).mark_source_ready()

    failures = enter(mgr, [good, bad])
    assert failures == [(bad, "V retrieval failed")]
    assert runnable(mgr, good) is True
    assert runnable(mgr, bad) is False


def test_all_ready_requests_are_pulled_in_one_collective():
    """TP ranks must agree on the set, so the pull is issued as one call."""
    mgr = make_manager()
    reqs = [make_req(f"req-{i}") for i in range(3)]
    for req in reqs:
        open_gate(mgr, req).mark_source_ready()
    assert enter(mgr, reqs) == []
    assert mgr.decode_refresher.calls == [reqs]


def test_only_ready_requests_join_the_collective():
    mgr = make_manager()
    ready, not_stored = make_req("ready"), make_req("not-stored")
    open_gate(mgr, ready).mark_source_ready()
    open_gate(mgr, not_stored)
    assert enter(mgr, [ready, not_stored]) == []
    assert mgr.decode_refresher.calls == [[ready]]
    assert runnable(mgr, ready) is True
    assert runnable(mgr, not_stored) is False


# --------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------


def test_closing_a_gate_keeps_the_request_unrunnable_and_stops_the_pull():
    mgr = make_manager()
    req = make_req()
    gate = open_gate(mgr, req)
    gate.mark_source_ready()
    mgr.close_bootstrap_gate(req)

    assert gate.state is BootstrapState.CLOSED
    # The gate is no longer registered, so this request is out of the feature's
    # scope; the abort path removes it from the queue separately.
    assert mgr.bootstrap_gates == {}
    assert enter(mgr, [req]) == []
    assert mgr.decode_refresher.calls == []


def test_closing_is_idempotent_and_safe_for_an_unknown_request():
    mgr = make_manager()
    mgr.close_bootstrap_gate(make_req("never-seen"))
    req = make_req()
    open_gate(mgr, req)
    mgr.close_bootstrap_gate(req)
    mgr.close_bootstrap_gate(req)


def test_a_closed_gate_mid_batch_is_reported_rather_than_installed():
    mgr = make_manager()
    a, b = make_req("a"), make_req("b")
    gate_a = open_gate(mgr, a)
    gate_a.mark_source_ready()
    open_gate(mgr, b).mark_source_ready()
    gate_a.close()  # closed without deregistering, e.g. a racing abort

    failures = enter(mgr, [a, b])
    assert [req for req, _ in failures] == [a]
    assert "closed" in failures[0][1]
    assert runnable(mgr, b) is True


# --------------------------------------------------------------------------
# Staging backpressure: a full budget defers, it never aborts
# --------------------------------------------------------------------------


def test_no_budget_means_no_staging_pre_check():
    mgr = make_manager()
    req = make_req()
    open_gate(mgr, req).mark_source_ready()
    assert enter(mgr, [req]) == []
    assert runnable(mgr, req) is True


def test_a_request_that_fits_the_staging_budget_is_pulled():
    mgr = make_manager(staging_bytes=128, per_req=64)
    req = make_req()
    open_gate(mgr, req).mark_source_ready()
    assert enter(mgr, [req]) == []
    assert mgr.decode_refresher.calls == [[req]]
    assert runnable(mgr, req) is True


def test_a_request_that_does_not_fit_is_deferred_not_aborted():
    mgr = make_manager(staging_bytes=128, used=100, per_req=64)
    req = make_req()
    gate = open_gate(mgr, req)
    gate.mark_source_ready()

    assert enter(mgr, [req]) == []  # no failure reported
    assert mgr.decode_refresher.calls == []  # nothing was pulled
    assert gate.state is BootstrapState.QUEUED
    assert gate.in_waiting_queue
    assert runnable(mgr, req) is False


def test_a_deferred_request_is_pulled_once_the_budget_frees_up():
    mgr = make_manager(staging_bytes=128, used=100, per_req=64)
    req = make_req()
    open_gate(mgr, req).mark_source_ready()
    enter(mgr, [req])

    mgr.transfer_budget.release("someone-else")
    assert enter(mgr, [req]) == []
    assert mgr.decode_refresher.calls == [[req]]
    assert runnable(mgr, req) is True


def test_headroom_is_charged_down_within_one_pass():
    """Two newcomers must not both be told there is room for one."""
    mgr = make_manager(staging_bytes=64, per_req=64)
    first, second = make_req("first"), make_req("second")
    for req in (first, second):
        open_gate(mgr, req).mark_source_ready()

    assert enter(mgr, [first, second]) == []
    assert [r.rid for r in mgr.decode_refresher.calls[0]] == ["first"]
    assert runnable(mgr, first) is True
    assert runnable(mgr, second) is False


def test_a_deferred_request_does_not_block_a_later_one_that_fits():
    mgr = make_manager(staging_bytes=64, per_req=64)
    big, small = make_req("big"), make_req("small")
    for req in (big, small):
        open_gate(mgr, req).mark_source_ready()
    mgr.local_shard_manifest = lambda tokens: SimpleNamespace(expected_bytes=64)

    enter(mgr, [big, small])
    assert runnable(mgr, big) is True
    assert runnable(mgr, small) is False
    assert enter(mgr, [small]) == []  # still deferred, still not an error


@pytest.mark.parametrize("blocked_by", ["budget", "source"])
def test_tp_ranks_agree_before_starting_a_wave_and_retry(blocked_by):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    managers = [make_manager(staging_bytes=128), make_manager(staging_bytes=128)]
    reqs = [make_req(), make_req()]
    barrier = threading.Barrier(2, timeout=5)
    slots = [None, None]
    for rank, (mgr, req) in enumerate(zip(managers, reqs)):

        def gather(value, rank=rank):
            slots[rank] = value
            barrier.wait()
            result = list(slots)
            barrier.wait()
            return result

        mgr.gather_rank_objects = gather
        gate = open_gate(mgr, req)
        if blocked_by != "source" or rank == 0:
            gate.mark_source_ready()
    if blocked_by == "budget":
        managers[1].transfer_budget.reserve("occupied", 128, 0)

    with ThreadPoolExecutor(max_workers=2) as pool:

        def step():
            futures = [pool.submit(enter, m, [r]) for m, r in zip(managers, reqs)]
            return [f.result(timeout=10) for f in futures]

        assert step() == [[], []]
        assert all(not m.decode_refresher.calls for m in managers)
        assert all(not runnable(m, r) for m, r in zip(managers, reqs))
        if blocked_by == "budget":
            managers[1].transfer_budget.release("occupied")
        else:
            managers[1].bootstrap_gate_for(reqs[1]).mark_source_ready()
        assert step() == [[], []]
        assert all(len(m.decode_refresher.calls) == 1 for m in managers)
        assert all(runnable(m, r) for m, r in zip(managers, reqs))


def test_pending_completion_installs_once_and_never_resurrects_cancelled_request():
    mgr = make_manager()
    live, cancelled = make_req("live"), make_req("cancelled")
    for req in (live, cancelled):
        gate = open_gate(mgr, req)
        gate.mark_source_ready()
        gate.enter_waiting_queue()
        gate.begin()
    mgr.close_bootstrap_gate(cancelled)
    results = [([live, cancelled], []), ([], [])]
    mgr.decode_refresher.poll_bootstrap = lambda: results.pop(0)
    assert enter(mgr, [live]) == []
    assert runnable(mgr, live)
    assert mgr.bootstrap_gate_for(cancelled) is None
    assert enter(mgr, [live]) == []
    assert not mgr.decode_refresher.calls
