"""Result bridge contracts; actual Req/processor exercised by strict CPU smoke."""

from array import array
from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
import torch
from sglang.srt.disaggregation.pvd.cpu_batch_forward import CPUBatchForwardExecutor
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError
from sglang.srt.disaggregation.pvd.cpu_schedule_bridge import CPUScheduleBridge
from test_pvd_cpu_batch_dispatch import pair


def req(life, slot):
    r = NS(
        rid=life.request_id,
        origin_input_ids=array("q", life.prompt),
        output_ids=array("q", life.outputs),
        req_pool_idx=slot,
        is_retracted=False,
        ended=False,
    )
    r.finished = lambda: r.ended
    return r


@contextmanager
def setup():
    with pair() as (dispatch, a, b):
        executor = CPUBatchForwardExecutor.__new__(CPUBatchForwardExecutor)
        executor.dispatcher = dispatch
        executor._storage = {a: 1, b: 2}
        executor._completed_operation = None
        batch = NS(
            reqs=[req(a, 1), req(b, 2)],
            device="cpu",
            enable_overlap=False,
            forward_mode=NS(is_decode=lambda: True),
            spec_algorithm=NS(is_none=lambda: True),
            is_spec_v2=False,
        )
        ticket = dispatch.begin([a, b])
        processor = NS(enable_overlap=False, enable_overlap_mlx=False)
        result = NS(next_token_ids=[31, 41], copy_done=None)
        yield executor, batch, ticket, processor, result, a, b


def append(bridge, batch, result):
    for r, t in zip(batch.reqs, result.next_token_ids, strict=True):
        if bridge.accepts(r):
            r.output_ids.append(int(t))


def test_single_authoritative_write_sampled_not_argmax_and_replay():
    with setup() as (e, batch, ticket, p, result, a, b):
        bridge = CPUScheduleBridge(e, batch, ticket)
        e._completed_operation = ticket.operation_id
        with bridge.processing(p, batch, result):
            append(bridge, batch, result)
            assert a.outputs == b.outputs == (9,)  # observer has not committed
        assert tuple(batch.reqs[0].output_ids) == a.outputs == (9, 31)
        assert tuple(batch.reqs[1].output_ids) == b.outputs == (9, 41)
        assert not e.dispatcher.arbiter.busy
        with (
            pytest.raises(LifecycleError, match="replayed"),
            bridge.processing(p, batch, result),
        ):
            pytest.fail("replay reached output writer")
        assert a.committed_tokens == b.committed_tokens == 1


@pytest.mark.parametrize("mode", ["retract", "cancel", "finished"])
def test_cancelled_row_is_not_committed_other_member_can_finish(mode):
    with setup() as (e, batch, ticket, p, result, a, b):
        bridge = CPUScheduleBridge(e, batch, ticket)
        e._completed_operation = ticket.operation_id
        if mode == "retract":
            batch.reqs[0].is_retracted = True
        elif mode == "finished":
            batch.reqs[0].ended = True
        else:
            a.terminate("cancel")
        with bridge.processing(p, batch, result):
            append(bridge, batch, result)
            batch.reqs[1].ended = True
            batch.reqs[1].req_pool_idx = None  # normal finish may release slot
        assert a.outputs == (9,) and a.state == "aborted"
        assert b.outputs == (9, 41) and b.state == "finished"


@pytest.mark.parametrize(
    "mode",
    [
        "order",
        "replace",
        "slot",
        "prompt",
        "outputs",
        "rid",
        "missing_row",
        "float",
        "bool",
        "nested",
        "negative",
        "copy_done",
        "overlap",
        "spec",
    ],
)
def test_preflight_refuses_before_any_writer_and_aborts(mode):
    with setup() as (e, batch, ticket, p, result, a, b):
        bridge = CPUScheduleBridge(e, batch, ticket)
        e._completed_operation = ticket.operation_id
        if mode == "order":
            batch.reqs.reverse()
        elif mode == "replace":
            batch.reqs[0] = req(a, 1)
        elif mode == "slot":
            batch.reqs[0].req_pool_idx = 2
        elif mode == "prompt":
            batch.reqs[0].origin_input_ids[0] += 1
        elif mode == "outputs":
            batch.reqs[0].output_ids.append(8)
        elif mode == "rid":
            batch.reqs[0].rid = "other"
        elif mode == "missing_row":
            result.next_token_ids.pop()
        elif mode == "float":
            result.next_token_ids = torch.tensor([1.0, 2.0])
        elif mode == "bool":
            result.next_token_ids[0] = True
        elif mode == "nested":
            result.next_token_ids = [[1], [2]]
        elif mode == "negative":
            result.next_token_ids[1] = -1
        elif mode == "copy_done":
            result.copy_done = object()
        elif mode == "overlap":
            p.enable_overlap = True
        else:
            batch.spec_algorithm = NS(is_none=lambda: False)
        with pytest.raises(LifecycleError), bridge.processing(p, batch, result):
            pytest.fail("invalid dispatch reached output writer")
        assert a.outputs == b.outputs == (9,)
        assert a.state == b.state == "aborted"
        assert not e.dispatcher.arbiter.busy


def test_early_callback_does_not_release_executing_lease():
    with setup() as (e, batch, ticket, p, result, a, _b):
        bridge = CPUScheduleBridge(e, batch, ticket)
        with (
            pytest.raises(LifecycleError, match="has not completed"),
            bridge.processing(p, batch, result),
        ):
            pytest.fail("early callback")
        assert e.dispatcher.arbiter.busy and a._permit is not None
        bridge.fail_after_execution("forward unwound")
        assert not e.dispatcher.arbiter.busy


@pytest.mark.parametrize("mode", ["missing", "double", "wrong", "raises"])
def test_bad_or_partial_authoritative_write_aborts_without_rollback(mode):
    with setup() as (e, batch, ticket, p, result, a, b):
        bridge = CPUScheduleBridge(e, batch, ticket)
        e._completed_operation = ticket.operation_id
        with (
            pytest.raises((LifecycleError, RuntimeError)),
            bridge.processing(p, batch, result),
        ):
            batch.reqs[0].output_ids.append(31)
            if mode == "double":
                append(bridge, batch, result)
            elif mode == "wrong":
                batch.reqs[1].output_ids.append(99)
            elif mode == "raises":
                raise RuntimeError("output streaming failed after first commit")
        assert tuple(batch.reqs[0].output_ids)[:2] == (9, 31)  # no rollback
        assert a.state == b.state == "aborted"
        assert a.outputs == b.outputs == (9,)  # no partial mirror advance
        assert not e.dispatcher.arbiter.busy


@pytest.mark.parametrize("mode", ["cuda", "slot", "prefix", "order", "duplicate"])
def test_attach_validates_before_modifying_batch(mode):
    with setup() as (e, batch, ticket, _p, _result, _a, _b):
        if mode == "cuda":
            batch.device = "cuda"
        elif mode == "slot":
            batch.reqs[0].req_pool_idx = 3
        elif mode == "prefix":
            batch.reqs[0].origin_input_ids.append(2)
        elif mode == "order":
            batch.reqs.reverse()
        else:
            batch.pvd_cpu_result_bridge = object()
        with pytest.raises(LifecycleError):
            CPUScheduleBridge(e, batch, ticket)
        assert e.dispatcher.arbiter.busy  # caller still owns failed attach


def test_cpu_tensor_results_and_no_second_timeout_poll_after_output_commit():
    with setup() as (e, batch, ticket, p, result, a, _b):
        bridge = CPUScheduleBridge(e, batch, ticket)
        e._completed_operation = ticket.operation_id
        result.next_token_ids = torch.tensor([31, 41], dtype=torch.int64)
        with bridge.processing(p, batch, result):
            append(bridge, batch, result)
            a.poll = lambda: pytest.fail("cannot uncommit authoritative Req token")
        assert a.outputs == (9, 31)
