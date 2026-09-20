"""The two decode.py hooks for the waiting-queue bootstrap, executed for real.

Both methods are extracted from the shipped source with ast and executed
against fake collaborators, so the logic under test is the code that runs in
production rather than a restatement of it. Only the CUDA/frontend objects the
methods touch are doubles.

This proves scheduling arithmetic and error handling. It proves nothing about
RDMA, GPU visibility, or the behaviour of a real multi-rank scheduler loop.
"""

import ast
import logging
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace

import pytest

DECODE_PATH = (
    Path(__file__).resolve().parents[3] / "python/sglang/srt/disaggregation/decode.py"
)


def load_method(name):
    """Execute one SchedulerDisaggregationDecodeMixin method in isolation."""
    tree = ast.parse(DECODE_PATH.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SchedulerDisaggregationDecodeMixin"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {
        "List": list,
        "Req": object,
        "logger": logging.getLogger(__name__),
        "HTTPStatus": HTTPStatus,
        "set_time_batch": lambda *a, **k: None,
        "ScheduleBatch": SimpleNamespace(
            init_new=lambda *a, **k: SimpleNamespace(
                reqs=list(a[0]),
                prepare_for_prebuilt=lambda: None,
                process_prebuilt=lambda *b, **c: None,
            )
        ),
    }
    # decode.py relies on `from __future__ import annotations`; keep it so the
    # `self: Scheduler` annotations stay unevaluated strings.
    future = ast.parse("from __future__ import annotations\n").body
    module = ast.fix_missing_locations(
        ast.Module(body=future + [method], type_ignores=[])
    )
    exec(compile(module, str(DECODE_PATH), "exec"), namespace)
    return namespace[name], namespace


def make_req(rid):
    return SimpleNamespace(
        rid=rid,
        return_logprob=False,
        kv_committed_len=None,
        finished_reason=None,
        init_next_round_input=lambda tree_cache: None,
    )


def make_scheduler(reqs, runnable, *, topology="pvd", enabled=True, capacity=8):
    manager = SimpleNamespace(
        waiting_queue_bootstrap=enabled,
        bootstrap_runnable=lambda req: runnable.get(req.rid, True),
    )
    return SimpleNamespace(
        waiting_queue=list(reqs),
        grammar_manager=SimpleNamespace(has_waiting_grammars=lambda: False),
        enable_priority_scheduling=False,
        running_batch=SimpleNamespace(batch_size=lambda: 0),
        req_to_token_pool=SimpleNamespace(size=capacity),
        max_running_requests=capacity,
        server_args=SimpleNamespace(
            disaggregation_topology=topology,
            disaggregation_decode_enable_radix_cache=False,
        ),
        disagg_decode_prealloc_queue=SimpleNamespace(kv_manager=manager),
        tree_cache=None,
        token_to_kv_pool_allocator=None,
        model_config=None,
        enable_overlap=False,
        spec_algorithm=None,
        future_map=None,
    )


def admit(scheduler):
    """Run the real batch builder; return (admitted rids, still-queued rids)."""
    method, _ = load_method("get_new_prebuilt_batch")
    batch = method(scheduler)
    admitted = [] if batch is None else [r.rid for r in batch.reqs]
    return admitted, [r.rid for r in scheduler.waiting_queue]


# --------------------------------------------------------------------------
# get_new_prebuilt_batch: the not-runnable skip
# --------------------------------------------------------------------------


def test_an_empty_queue_builds_no_batch():
    assert admit(make_scheduler([], {})) == ([], [])


@pytest.mark.parametrize("topology", ["pvd", "pd"])
def test_every_request_is_admitted_when_all_are_runnable(topology):
    reqs = [make_req(f"r{i}") for i in range(3)]
    admitted, queued = admit(make_scheduler(reqs, {}, topology=topology))
    assert admitted == ["r0", "r1", "r2"]
    assert queued == []


def test_pd_is_never_gated_even_if_a_manager_says_otherwise():
    """Ordinary PD must not consult the PVD bootstrap at all."""
    reqs = [make_req("r0"), make_req("r1")]
    admitted, queued = admit(make_scheduler(reqs, {"r0": False}, topology="pd"))
    assert admitted == ["r0", "r1"]
    assert queued == []


def test_the_flag_being_off_admits_a_not_runnable_request():
    reqs = [make_req("r0"), make_req("r1")]
    admitted, queued = admit(make_scheduler(reqs, {"r0": False}, enabled=False))
    assert admitted == ["r0", "r1"]
    assert queued == []


def test_a_not_runnable_request_stays_queued_and_is_not_a_barrier():
    reqs = [make_req("blocked"), make_req("ready")]
    admitted, queued = admit(make_scheduler(reqs, {"blocked": False}))
    assert admitted == ["ready"]
    assert queued == ["blocked"]


def test_a_skipped_request_does_not_consume_a_batch_slot():
    """The regression the admitted-counter refactor exists to prevent.

    With capacity 2 and the first request not runnable, position-based counting
    would have admitted only one request. Counting admissions fills both slots.
    """
    reqs = [make_req("blocked"), make_req("a"), make_req("b"), make_req("c")]
    admitted, queued = admit(make_scheduler(reqs, {"blocked": False}, capacity=2))
    assert admitted == ["a", "b"]
    assert queued == ["blocked", "c"]


def test_capacity_still_bounds_admissions():
    reqs = [make_req(f"r{i}") for i in range(5)]
    admitted, queued = admit(make_scheduler(reqs, {}, capacity=2))
    assert admitted == ["r0", "r1"]
    assert queued == ["r2", "r3", "r4"]


def test_queue_order_is_preserved_for_skipped_requests():
    reqs = [make_req("x"), make_req("y"), make_req("z")]
    admitted, queued = admit(make_scheduler(reqs, {"x": False, "z": False}, capacity=8))
    assert admitted == ["y"]
    assert queued == ["x", "z"]


def test_no_batch_is_built_when_every_request_is_still_pulling():
    reqs = [make_req("a"), make_req("b")]
    admitted, queued = admit(make_scheduler(reqs, {"a": False, "b": False}))
    assert admitted == []
    assert queued == ["a", "b"]


def test_a_full_running_batch_leaves_everyone_queued():
    scheduler = make_scheduler([make_req("r0")], {}, capacity=1)
    scheduler.running_batch = SimpleNamespace(batch_size=lambda: 1)
    assert admit(scheduler) == ([], ["r0"])


# --------------------------------------------------------------------------
# _pvd_enter_waiting_queue: trigger and failure handling
# --------------------------------------------------------------------------


def make_trigger_scheduler(reqs, failures=(), enabled=True):
    calls = {"entered": [], "closed": [], "streamed": [], "released": []}
    manager = SimpleNamespace(
        waiting_queue_bootstrap=enabled,
        enter_waiting_queue=lambda rs: (
            calls["entered"].append(list(rs)) or list(failures)
        ),
        close_bootstrap_gate=lambda req: calls["closed"].append(req.rid),
        decode_refresher=SimpleNamespace(release_request=lambda req: None),
    )
    scheduler = SimpleNamespace(
        waiting_queue=list(reqs),
        disagg_decode_prealloc_queue=SimpleNamespace(kv_manager=manager),
        output_streamer=SimpleNamespace(
            stream_output=lambda rs, lp: calls["streamed"].extend(r.rid for r in rs)
        ),
        tree_cache=None,
    )
    return scheduler, calls


def run_trigger(scheduler, reqs, monkeypatch):
    method, namespace = load_method("_pvd_enter_waiting_queue")
    aborted = []
    namespace["prepare_abort"] = lambda req, msg, status_code=None: aborted.append(
        (req.rid, msg)
    )
    namespace["release_kv_cache"] = lambda req, cache, is_insert: None
    method(scheduler, reqs)
    return aborted


def test_the_trigger_is_a_noop_when_the_flag_is_off(monkeypatch):
    reqs = [make_req("a")]
    scheduler, calls = make_trigger_scheduler(reqs, enabled=False)
    assert run_trigger(scheduler, reqs, monkeypatch) == []
    assert calls["entered"] == []


def test_the_trigger_hands_every_new_request_to_the_manager(monkeypatch):
    reqs = [make_req("a"), make_req("b")]
    scheduler, calls = make_trigger_scheduler(reqs)
    assert run_trigger(scheduler, reqs, monkeypatch) == []
    assert [r.rid for r in calls["entered"][0]] == ["a", "b"]
    assert scheduler.waiting_queue == reqs


def test_a_failed_pull_aborts_only_that_request_and_dequeues_it(monkeypatch):
    reqs = [make_req("good"), make_req("bad")]
    scheduler, calls = make_trigger_scheduler(
        reqs, failures=[(reqs[1], "V retrieval failed")]
    )
    aborted = run_trigger(scheduler, reqs, monkeypatch)

    assert aborted == [("bad", "PVD initial KV pull failed: V retrieval failed")]
    assert calls["closed"] == ["bad"]
    assert calls["streamed"] == ["bad"]
    assert [r.rid for r in scheduler.waiting_queue] == ["good"]


def test_a_failure_for_a_request_already_gone_from_the_queue_is_safe(monkeypatch):
    """Another path may have removed it first; removal must not raise."""
    queued = [make_req("still-here")]
    gone = make_req("gone")
    scheduler, calls = make_trigger_scheduler(
        queued, failures=[(gone, "V retrieval failed")]
    )
    run_trigger(scheduler, [gone], monkeypatch)
    assert [r.rid for r in scheduler.waiting_queue] == ["still-here"]
    assert calls["closed"] == ["gone"]


@pytest.mark.parametrize("queue_mode", ["normal", "non-polling", "retracted"])
def test_scheduler_retries_deferred_waiters_without_new_arrivals(queue_mode):
    from test_pvd_waiting_queue_bootstrap import make_manager, make_req, open_gate

    mgr = make_manager(staging_bytes=128, used=100, per_req=64)
    mgr.decode_refresher.cleanup_finished = lambda: None
    req = make_req()
    open_gate(mgr, req).mark_source_ready()
    scheduler = SimpleNamespace(
        waiting_queue=[req],
        server_args=SimpleNamespace(
            disaggregation_topology="pvd",
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_polling_interval=1,
        ),
        disagg_decode_prealloc_queue=SimpleNamespace(
            kv_manager=mgr,
            resume_retracted_reqs=lambda: [],
            retracted_queue=[object()] if queue_mode == "retracted" else [],
            pop_preallocated=lambda: ([], []),
        ),
        disagg_decode_transfer_queue=SimpleNamespace(
            extend=lambda reqs: None, pop_transferred=lambda: [],
        ),
        enable_hisparse=False,
        polling_count=0,
        polling_interval=100 if queue_mode == "non-polling" else 1,
        _pvd_enter_waiting_queue=lambda reqs: mgr.enter_waiting_queue(reqs),
    )
    process, _ = load_method("process_decode_queue")
    process(scheduler)
    assert not mgr.bootstrap_runnable(req)
    assert not mgr.decode_refresher.calls
    mgr.transfer_budget.release("someone-else")
    process(scheduler)
    assert mgr.bootstrap_runnable(req)
    assert mgr.decode_refresher.calls == [[req]]
