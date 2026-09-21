"""Batch ownership/identity contracts, independent of GPU or serving imports."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch
from sglang.srt.disaggregation.pvd.cpu_batch_dispatch import (
    BatchTokenResult,
    CPUBatchDispatcher,
    batch_results_from_logits,
)
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import (
    LifecycleError,
    TargetExecutionArbiter,
)
from test_pvd_cpu_decode_lifecycle import advance, env


@contextmanager
def pair():
    arbiter = TargetExecutionArbiter()
    dispatch = CPUBatchDispatcher(arbiter)
    with env("a", arbiter) as (a, af, _), env("b", arbiter) as (b, bf, _):
        a.admit(af.request)
        b.admit(bf.request)
        try:
            yield dispatch, a, b
        finally:
            if dispatch._ticket is not None:
                dispatch.fail(dispatch._ticket, "test cleanup")


def results(ticket):
    return tuple(
        BatchTokenResult(m.request_id, m.permit, i + 20)
        for i, m in enumerate(ticket.members)
    )


def test_one_lease_multiple_members_and_reordered_results():
    with pair() as (d, a, b):
        ticket = d.begin([b, a])
        assert d.arbiter.busy
        assert a._decode_lease is b._decode_lease is None
        assert a._batch_owner is b._batch_owner is d
        committed = d.complete(ticket, reversed(results(ticket)))
        assert [r.request_id for r in committed] == ["b", "a"]
        assert a.outputs == (9, 21) and b.outputs == (9, 20)
        assert not d.arbiter.busy
        assert a._permit is b._permit is None
        second = d.begin([a, b])
        d.complete(second, results(second))
        assert a.outputs == (9, 21, 20) and b.outputs == (9, 20, 21)


@pytest.mark.parametrize(
    "mode",
    [
        "missing",
        "duplicate",
        "foreign",
        "copied_permit",
        "bad_token",
        "bad_finished",
        "untyped",
    ],
)
def test_entire_result_set_validated_before_any_output(mode):
    with pair() as (d, a, b):
        ticket = d.begin([a, b])
        r = list(results(ticket))
        if mode == "missing":
            r.pop()
        elif mode == "duplicate":
            r[1] = r[0]
        elif mode == "foreign":
            r[1] = replace(r[1], request_id="other")
        elif mode == "copied_permit":
            r[1] = replace(r[1], permit=replace(r[1].permit))
        elif mode == "bad_token":
            r[1] = replace(r[1], token=-1)
        elif mode == "bad_finished":
            r[1] = replace(r[1], finished=1)
        else:
            r[1] = object()
        with pytest.raises(LifecycleError):
            d.complete(ticket, r)
        assert a.committed_tokens == b.committed_tokens == 0
        assert d.arbiter.busy  # explicit fail required; no resource reuse


@pytest.mark.parametrize("member", ["a", "b", "both"])
def test_cancelled_members_discard_output_without_releasing_batch_early(member):
    with pair() as (d, a, b):
        ticket = d.begin([a, b])
        for life in (a, b):
            if member in (life.request_id, "both"):
                life.terminate("cancel or retract")
        assert d.arbiter.busy
        committed = d.complete(ticket, results(ticket))
        expected = {"a", "b"} - ({member} if member != "both" else {"a", "b"})
        assert {r.request_id for r in committed} == expected
        assert not d.arbiter.busy
        assert a.committed_tokens == int("a" in expected)
        assert b.committed_tokens == int("b" in expected)


def test_eos_and_failure_are_not_partial_commit_paths():
    with pair() as (d, a, b):
        ticket = d.begin([a, b])
        r = results(ticket)
        d.complete(ticket, [replace(r[0], finished=True), r[1]])
        assert a.state == "finished" and b.state == "running"
        assert a.committed_tokens == b.committed_tokens == 1
    with pair() as (d, a, b):
        ticket = d.begin([a, b])
        d.fail(ticket, "forward failed after writing one layer")
        assert a.state == b.state == "aborted"
        assert a.committed_tokens == b.committed_tokens == 0
        assert not d.arbiter.busy


def test_one_blocked_member_means_no_partial_dispatch():
    with pair() as (d, a, b):
        advance(b, 4)
        with pytest.raises(LifecycleError, match="no partial dispatch"):
            d.begin([a, b])
        assert not d.arbiter.busy
        assert a._permit is b._permit is None
        assert a.committed_tokens == 0 and b.committed_tokens == 4


@pytest.mark.parametrize("mode", ["empty", "duplicate", "untyped", "other_arbiter"])
def test_invalid_membership_refused_before_acquiring_lease(mode):
    with pair() as (d, a, b):
        members = {
            "empty": [],
            "duplicate": [a, a],
            "untyped": [a, object()],
            "other_arbiter": [a, b],
        }[mode]
        if mode == "other_arbiter":
            b.arbiter = TargetExecutionArbiter()
        with pytest.raises(LifecycleError):
            d.begin(members)
        assert not d.arbiter.busy and a._permit is None


def test_no_individual_completion_or_snapshot_of_batched_execution():
    with pair() as (d, a, b):
        ticket = d.begin([a, b])
        with pytest.raises(LifecycleError, match="batch-owned"):
            a.complete_decode(ticket.members[0].permit, 20)
        with pytest.raises(LifecycleError, match="batch-owned"):
            a.fail_decode(ticket.members[0].permit, "not your lease")
        with pytest.raises(LifecycleError, match="between forwards"):
            b.snapshot()
        assert d.arbiter.busy and a.committed_tokens == 0


def test_stale_batch_and_old_member_completion_cannot_change_new_round():
    with pair() as (d, a, b):
        old = d.begin([a, b])
        d.complete(old, results(old))
        new = d.begin([b, a])
        for invalid in (old, replace(new)):
            with pytest.raises(LifecycleError, match="stale or foreign batch"):
                d.complete(invalid, results(invalid))
        with pytest.raises(LifecycleError, match="member result"):
            d.complete(new, results(old))
        assert a.committed_tokens == b.committed_tokens == 1


def test_callback_thread_cannot_commit_batch():
    with pair() as (d, a, b):
        ticket = d.begin([a, b])
        with (
            ThreadPoolExecutor(1) as executor,
            pytest.raises(LifecycleError, match="owner thread"),
        ):
            executor.submit(d.complete, ticket, results(ticket)).result()
        assert a.committed_tokens == b.committed_tokens == 0


def test_full_logit_rows_not_only_last_row_and_no_order_inference():
    with pair() as (d, a, b):
        ticket = d.begin([b, a])
        logits = torch.tensor([[0.0, 8.0, 0.0], [0.0, 0.0, 9.0]])
        r = batch_results_from_logits(ticket, logits, finished=(False, True))
        assert [(x.request_id, x.token, x.finished) for x in r] == [
            ("b", 1, False),
            ("a", 2, True),
        ]
        d.complete(ticket, reversed(r))
        assert a.outputs[-1] == 2 and b.outputs[-1] == 1


@pytest.mark.parametrize(
    "logits",
    [
        torch.ones(3),
        torch.ones(1, 3),
        torch.ones(3, 3),
        torch.ones(2, 0),
        torch.ones(2, 3, dtype=torch.int64),
        torch.full((2, 3), float("nan")),
    ],
)
def test_missing_or_invalid_logits_rows_refused(logits):
    with pair() as (d, a, b):
        ticket = d.begin([a, b])
        with pytest.raises(LifecycleError, match="complete finite CPU logits"):
            batch_results_from_logits(ticket, logits, finished=(False, False))


@pytest.mark.parametrize("flags", [(), (False,), (False, 1)])
def test_explicit_finish_flag_per_member_required(flags):
    with pair() as (d, a, b):
        ticket = d.begin([a, b])
        with pytest.raises(LifecycleError, match="finish flag"):
            batch_results_from_logits(ticket, torch.ones(2, 3), finished=flags)


def test_destination_order_is_bound_by_dispatch_identity_not_caller_order():
    from sglang.srt.disaggregation.pvd.cpu_batch_forward import (
        CPUForwardDestination,
        bind_batch_destinations,
    )

    with pair() as (d, a, b):
        ticket = d.begin([b, a])
        ad, bd = CPUForwardDestination(a, 1, 12), CPUForwardDestination(b, 2, 13)
        assert bind_batch_destinations(d, ticket, [ad, bd]) == (bd, ad)


@pytest.mark.parametrize(
    "mode",
    [
        "missing",
        "duplicate",
        "aliased_slot",
        "aliased_row",
        "bad_slot",
        "bad_row",
        "cancelled",
        "untyped",
    ],
)
def test_bad_destination_mapping_refused_before_any_forward(mode):
    from sglang.srt.disaggregation.pvd.cpu_batch_forward import (
        CPUForwardDestination,
        bind_batch_destinations,
    )

    with pair() as (d, a, b):
        ticket = d.begin([a, b])
        dest = [CPUForwardDestination(a, 1, 12), CPUForwardDestination(b, 2, 13)]
        if mode == "missing":
            dest.pop()
        elif mode == "duplicate":
            dest[1] = dest[0]
        elif mode == "aliased_slot":
            dest[1] = replace(dest[1], slot=1)
        elif mode == "aliased_row":
            dest[1] = replace(dest[1], kv_row=12)
        elif mode == "bad_slot":
            dest[1] = replace(dest[1], slot=0)
        elif mode == "bad_row":
            dest[1] = replace(dest[1], kv_row=True)
        elif mode == "cancelled":
            b.terminate("cancelled before dispatch")
        else:
            dest[1] = object()
        with pytest.raises(LifecycleError):
            bind_batch_destinations(d, ticket, dest)
        assert a.committed_tokens == b.committed_tokens == 0
