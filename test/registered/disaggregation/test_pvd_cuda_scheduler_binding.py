"""Real binding/Decode method bodies on CPU; model/transport remain doubles."""

from contextlib import contextmanager
from concurrent.futures import Future
from http import HTTPStatus
from types import SimpleNamespace as NS

import pytest
from sglang.srt.disaggregation.pvd import cuda_scheduler_binding as module
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError
from sglang.srt.disaggregation.pvd.conn import PVDSelectedRouteBinding
from sglang.srt.disaggregation.pvd.cuda_rank_batch import CUDARankBatchExecutor
from sglang.srt.disaggregation.pvd.cuda_route_discovery import CUDARouteDiscoveryQueue
from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeRefresher
from test_pvd_cpu_release_driver import scheduler_methods
from test_pvd_cuda_receiver_ownership import bound, pump
from test_pvd_cuda_request_release import source


def method(name, **namespace):
    return source(
        "python/sglang/srt/disaggregation/decode.py",
        name,
        "SchedulerDisaggregationDecodeMixin",
        namespace,
    )


class Batch:
    def __init__(self, reqs):
        self.reqs, self.batch_is_full, self.capacity = list(reqs), True, True

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, **kwargs):
        self.reqs = [r for r in self.reqs if not r.finished()]

    def check_decode_mem(self):
        return self.capacity


@contextmanager
def setup(monkeypatch, *, attach=True):
    with bound(monkeypatch) as b:
        c, events = b.c, []
        s = c.manager.scheduler
        args = s.server_args
        for key, value in dict(
            disaggregation_topology="pvd",
            disaggregation_mode="decode",
            disable_cuda_graph=True,
            dp_size=1,
            enable_dp_attention=False,
            speculative_algorithm=None,
            disaggregation_decode_enable_radix_cache=False,
            disaggregation_decode_enable_offload_kvcache=False,
            page_size=1,
        ).items():
            setattr(args, key, value)
        s.enable_overlap = s.enable_hisparse = False
        s.max_running_requests = 2
        s.tp_worker = NS(
            model_runner=NS(
                tp_size=1,
                pp_size=1,
                attn_cp_size=1,
                attn_backend=NS(consumer=c.c.consumer),
            )
        )
        s.disagg_decode_prealloc_queue = NS(kv_manager=c.manager)
        c.manager.waiting_queue_bootstrap = True
        c.manager.decode_refresher = PVDDecodeRefresher(c.manager)
        c.request.return_logprob = False

        def abort(req, message, status_code):
            assert status_code == HTTPStatus.SERVICE_UNAVAILABLE
            events.append(("abort", req.rid, message))
            req.finished = lambda: True

        def release(req, cache, is_insert):
            assert req is c.request and cache is c.cache and not is_insert
            assert b.retirement.state == "attached"
            events.append(("defer", req.rid))
            b.driver.cancel(req)

        abort_method = method(
            "_abort_pvd_cuda_requests",
            HTTPStatus=HTTPStatus,
            prepare_abort=abort,
            release_kv_cache=release,
        )
        s._abort_pvd_cuda_requests = lambda reqs, reason: abort_method(s, reqs, reason)
        s.output_streamer = NS(
            stream_output=lambda reqs, logprob: events.append(("stream", reqs[0].rid))
        )
        executor = CUDARankBatchExecutor(c.c.consumer, b.driver.arbiter, max_requests=2)
        monkeypatch.setattr(module, "_require_supported_pools", lambda _: None)
        binding = (
            module.CUDADecodeSchedulerBinding(
                s, b.driver, executor, pool_owner=c.c.owner
            )
            if attach
            else None
        )
        try:
            yield NS(**locals())
        finally:
            b.sparse_done.set_result(None) if not b.sparse_done.done() else None
            b.lease_done.set_result(None) if not b.lease_done.done() else None
            b.driver.begin_shutdown()
            pump(b.driver, lambda: not b.driver._records)
            if binding is not None and not binding.closed:
                binding.close()


def test_binding_is_explicit_and_owner_scoped(monkeypatch):
    assert module.binding_for(NS()) is None
    with pytest.raises(LifecycleError, match="foreign"):
        module.binding_for(NS(pvd_cuda_binding=object()))
    with setup(monkeypatch) as t:
        assert module.binding_for(t.s) is t.binding
        with pytest.raises(LifecycleError, match="foreign"):
            module.binding_for(NS(pvd_cuda_binding=t.binding))
        with pytest.raises(LifecycleError, match="exact TP1"):
            module.CUDADecodeSchedulerBinding(
                t.s, t.b.driver, t.executor, pool_owner=t.c.c.owner
            )


@pytest.mark.parametrize(
    "fault",
    [
        "overlap",
        "graphs",
        "dp",
        "tp",
        "backend",
        "kv",
        "manager",
        "bootstrap",
        "speculation",
        "cpu",
        "page",
        "quarantine",
    ],
)
def test_incompatible_binding_refused_before_install(monkeypatch, fault):
    with setup(monkeypatch, attach=False) as t, monkeypatch.context() as patch:
        target, key, value = {
            "overlap": (t.s, "enable_overlap", True),
            "graphs": (t.args, "disable_cuda_graph", False),
            "dp": (t.args, "dp_size", 2),
            "tp": (t.s.tp_worker.model_runner, "tp_size", 2),
            "backend": (t.s.tp_worker.model_runner, "attn_backend", object()),
            "kv": (t.c.manager, "kv_pool", object()),
            "manager": (t.c.manager, "scheduler", object()),
            "bootstrap": (t.c.manager, "waiting_queue_bootstrap", False),
            "speculation": (t.args, "speculative_algorithm", "EAGLE"),
            "cpu": (t.s, "pvd_cpu_release_driver", object()),
            "page": (t.args, "page_size", 16),
            "quarantine": (t.executor, "_quarantined", True),
        }[fault]
        patch.setattr(target, key, value, raising=False)
        with pytest.raises(LifecycleError, match="exact TP1"):
            module.CUDADecodeSchedulerBinding(
                t.s, t.b.driver, t.executor, pool_owner=t.c.c.owner
            )
        assert getattr(t.s, "pvd_cuda_binding", None) is None


def test_new_unclaimed_request_does_not_reset_peer(monkeypatch):
    with setup(monkeypatch) as t:
        record = t.b.driver._records["r"]
        before = (record.outputs, record.refresh, t.c.session.clock.round)
        assert not t.binding.waiting_ready(NS(rid="new"))
        assert t.binding.waiting_ready(t.c.request)
        assert before == (record.outputs, record.refresh, t.c.session.clock.round)
        with pytest.raises(LifecycleError, match="receiver-claimed"):
            t.binding.waiting_ready(NS(rid="r"))


def test_explicit_binding_polls_selected_v_without_admitting_unclaimed_req(monkeypatch):
    with setup(monkeypatch) as t:
        req = NS(
            rid="new",
            pvd_delivery_id="new:delivery",
            pvd_vector_group_id="chosen",
            is_retracted=False,
            finished=lambda: False,
        )
        future = Future()
        started = []
        t.c.manager.vector_group_for = lambda req: req.pvd_vector_group_id
        t.c.manager.bootstrap_runnable = lambda req: True
        t.c.manager.start_selected_cuda_routes = lambda req: (
            started.append(req) or future
        )
        t.binding.route_queue = CUDARouteDiscoveryQueue(t.c.manager, max_inflight=2)
        t.s.waiting_queue = [t.c.request, req]
        t.binding.poll()
        assert started == [req]
        assert t.binding.selected_routes_for(req) is None
        assert not t.binding.waiting_ready(req)
        selected = PVDSelectedRouteBinding(
            t.c.manager,
            req,
            req.rid,
            t.c.key,
            "chosen",
            req.pvd_delivery_id,
            object(),
        )
        future.set_result(selected)
        t.binding.poll()
        assert t.binding.selected_routes_for(req) is selected
        assert not t.binding.waiting_ready(req)
        t.s.waiting_queue.clear()
        t.binding.poll()
        assert not t.binding.route_queue.pending


def test_actual_selection_waits_before_decode_allocation(monkeypatch):
    with setup(monkeypatch) as t:
        batch = Batch([t.c.request])
        t.s.running_batch = batch
        t.s.get_new_prebuilt_batch = lambda: None
        t.s.dp_attn_adapter = NS(maybe_prepare_mlp_sync_batch=lambda b: b)
        t.s.update_running_batch = lambda b: t.events.append("allocate") or b
        select = method(
            "get_next_disagg_decode_batch_to_run",
            set_schedule_time_batch=lambda _: None,
        )
        ready = False
        t.b.controller.can_decode = lambda n: ready
        assert select(t.s) is None and not t.events
        ready = True
        assert select(t.s) is batch and t.events == ["allocate"]


def test_capacity_aborts_once_and_waits_for_both_closes(monkeypatch):
    with setup(monkeypatch) as t:
        batch = Batch([t.c.request])
        batch.capacity = False
        assert not t.binding.ready_to_prepare(batch)
        assert batch.is_empty() and t.s.waiting_queue == []
        assert [e[0] for e in t.events] == ["abort", "stream", "defer"]
        for _ in range(3):
            t.binding.poll()
        assert len(t.events) == 3 and t.binding.pending
        assert t.b.retirement.state == "attached"
        assert "request pool returned" not in t.b.events
        with pytest.raises(LifecycleError, match="drain first"):
            t.binding.close()


def test_stopped_record_emits_abort_before_poll_can_remove_it(monkeypatch):
    with setup(monkeypatch) as t:
        t.b.driver.cancel(t.c.request)
        original = t.b.driver.poll

        def observe():
            assert t.c.request.finished()
            assert t.events[0][0] == "abort"
            return original()

        monkeypatch.setattr(t.b.driver, "poll", observe)
        t.binding.poll()
        assert len(t.events) == 3


@pytest.mark.parametrize(
    "paused,has_batch", [(True, False), (False, False), (False, True)]
)
def test_actual_loop_polls_and_routes_result_once(monkeypatch, paused, has_batch):
    with setup(monkeypatch) as t:
        events, count = [], 0

        class End(Exception):
            pass

        def receive():
            nonlocal count
            count += 1
            if count > 1:
                raise End()
            return []

        t.s.request_receiver = NS(recv_requests=receive)
        t.s.process_input_requests = lambda _: None
        t.s.process_decode_queue = lambda: None
        t.s.poll_pvd_cpu_releases = lambda: None
        t.s._engine_paused = paused
        t.s.get_next_disagg_decode_batch_to_run = lambda: (
            object() if has_batch else None
        )
        t.s.on_idle = lambda: events.append("idle")
        t.s.run_batch = lambda _: pytest.fail("must use bridge")
        t.s.process_batch_result = lambda *_: pytest.fail("must not process twice")
        monkeypatch.setattr(t.binding, "poll", lambda: events.append("poll"))
        monkeypatch.setattr(t.binding, "run", lambda _: events.append("bridge"))
        with pytest.raises(End):
            scheduler_methods()["event_loop_normal_disagg_decode"](t.s)
        assert events == (
            ["poll"]
            if paused
            else ["poll"] + (["bridge"] if has_batch else []) + ["poll"]
        )


def test_waiting_source_body_skips_unclaimed_without_spending_capacity(monkeypatch):
    with setup(monkeypatch) as t:
        pending = NS(rid="pending")
        t.s.waiting_queue = [pending, t.c.request]
        t.s.grammar_manager = NS(has_waiting_grammars=lambda: False)
        t.s.enable_priority_scheduling = False
        t.s.running_batch = Batch([])
        t.c.c.req.size = 1
        t.s.max_running_requests = 1
        t.c.manager.bootstrap_runnable = lambda _: True
        t.c.request.init_next_round_input = lambda _: t.events.append("prepare")
        t.c.request.kv_committed_len = None
        t.s.token_to_kv_pool_allocator = t.c.allocator
        t.s.model_config = t.s.spec_algorithm = t.s.future_map = None
        built = NS(prepare_for_prebuilt=lambda: None, process_prebuilt=lambda *a: None)

        def init(reqs, *args):
            assert reqs == [t.c.request]
            return built

        result = method(
            "get_new_prebuilt_batch",
            ScheduleBatch=NS(init_new=init),
            set_time_batch=lambda *a: None,
        )(t.s)
        assert result is built and t.s.waiting_queue == [pending]
        assert t.events == ["prepare"]


def test_closed_binding_never_falls_back(monkeypatch):
    with setup(monkeypatch) as t:
        t.b.sparse_done.set_result(None)
        t.b.lease_done.set_result(None)
        t.b.driver.begin_shutdown()
        pump(t.b.driver, lambda: not t.b.driver._records)
        t.binding.close()
        assert t.s.pvd_cuda_binding is t.binding
        with pytest.raises(LifecycleError, match="changed or closed"):
            module.binding_for(t.s)


def test_binding_refuses_oversized_scheduler_before_admission(monkeypatch):
    with setup(monkeypatch, attach=False) as t:
        t.s.max_running_requests = 3
        with pytest.raises(LifecycleError, match="exact TP1"):
            module.CUDADecodeSchedulerBinding(
                t.s, t.b.driver, t.executor, pool_owner=t.c.c.owner
            )


def test_live_pool_replacement_and_graph_enable_are_refused(monkeypatch):
    with setup(monkeypatch) as t:
        for obj, field, replacement in [
            (t.c.manager, "kv_pool", object()),
            (t.args, "disable_cuda_graph", False),
        ]:
            with monkeypatch.context() as patch:
                patch.setattr(obj, field, replacement)
                with pytest.raises(LifecycleError, match="changed or closed"):
                    t.binding.waiting_ready(t.c.request)


def test_run_preserves_original_scheduler_result_wrapper(monkeypatch):
    with setup(monkeypatch) as t:
        batch, result = object(), object()
        t.s.batch_result_processor = object()
        t.s.run_batch = lambda b: result if b is batch else pytest.fail("batch")
        t.s.process_batch_result = object()

        class Bridge:
            def __init__(self, executor, driver, selected, *, pool_owner):
                assert executor is t.executor and driver is t.b.driver
                assert selected is batch and pool_owner is t.c.c.owner

            def run(self, *, forward, processor, result_handler):
                assert forward() is result
                assert processor is t.s.batch_result_processor
                assert result_handler is t.s.process_batch_result
                return result

        monkeypatch.setattr(module, "CUDAScheduleBridge", Bridge)
        assert t.binding.run(batch) is result
