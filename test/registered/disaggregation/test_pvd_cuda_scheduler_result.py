"""Run the shipped Scheduler result wrapper under CUDA bridge permits on CPU."""

from types import MethodType
from types import SimpleNamespace as NS

import pytest
from sglang.srt.disaggregation.pvd.cuda_schedule_bridge import CUDAScheduleBridge
from test_pvd_cuda_request_release import source
from test_pvd_cuda_schedule_bridge import case


def scheduler_wrapper(processor, driver, executor, events):
    def event(name):
        def record(*args, **kwargs):
            assert driver.arbiter.busy
            assert executor.dispatcher._ticket is not None
            events.append(name)

        return record

    scheduler = NS(
        batch_result_processor=processor,
        publish_load_snapshot=event("load"),
        enable_fpm=True,
        metrics_reporter=NS(
            log_batch_result_stats=event("stats"),
            _emit_forward_pass_metrics=event("forward metrics"),
            update_device_timer=event("device timer"),
        ),
        _maybe_clear_mm_inputs=event("multimodal cleanup"),
        maybe_send_health_check_signal=event("health"),
    )
    scheduler.process_batch_result = MethodType(
        source(
            "python/sglang/srt/managers/scheduler.py",
            "process_batch_result",
            "Scheduler",
        ),
        scheduler,
    )
    return scheduler


def test_original_scheduler_side_effects_run_once_inside_result_scope(monkeypatch):
    with case(monkeypatch) as (c, executor, driver, batch, processor, result, forward):
        events = []
        batch.forward_mode.is_extend = lambda: False
        scheduler = scheduler_wrapper(processor, driver, executor, events)
        normal = processor._process_batch_result_decode

        def process(b, r):
            events.append("tokens")
            return normal(b, r)

        processor._process_batch_result_decode = process
        bridge = CUDAScheduleBridge(executor, driver, batch, pool_owner=c.owner)
        bridge.run(
            forward=forward,
            processor=processor,
            result_handler=scheduler.process_batch_result,
        )
        assert events == [
            "load",
            "tokens",
            "stats",
            "forward metrics",
            "multimodal cleanup",
            "health",
            "device timer",
        ]
        assert [tuple(r.output_ids) for r in batch.reqs] == [(7, 8, 13), (7, 8, 17)]
        assert bridge.state == "completed"


@pytest.mark.parametrize(
    "fault", ["skip", "twice", "post-commit", "post-rewrite", "membership"]
)
def test_wrapper_cannot_skip_replay_or_rollback_committed_output(monkeypatch, fault):
    with case(monkeypatch) as (c, executor, driver, batch, processor, result, forward):
        events = []
        batch.forward_mode.is_extend = lambda: False
        scheduler = scheduler_wrapper(processor, driver, executor, events)
        if fault == "post-commit":

            def fail(*args):
                raise RuntimeError("metrics failure after commit")

            scheduler.metrics_reporter.log_batch_result_stats = fail

        def handler(b, r):
            if fault == "skip":
                return
            scheduler.process_batch_result(b, r)
            if fault == "twice":
                scheduler.process_batch_result(b, r)
            elif fault == "post-rewrite":
                b.reqs[0].output_ids[-1] = 99
            elif fault == "membership":
                b.reqs.reverse()

        bridge = CUDAScheduleBridge(executor, driver, batch, pool_owner=c.owner)
        with pytest.raises((ValueError, RuntimeError)):
            bridge.run(forward=forward, processor=processor, result_handler=handler)
        expected = [(7, 8), (7, 8)] if fault == "skip" else [(7, 8, 13), (7, 8, 17)]
        if fault == "post-rewrite":
            expected[0] = (7, 8, 99)
        elif fault == "membership":
            expected.reverse()
        assert [tuple(r.output_ids) for r in batch.reqs] == expected
        assert bridge.state == "failed"
        assert all(r.stopping for r in driver._records.values())


def test_async_scheduler_wrapper_is_refused_before_forward(monkeypatch):
    with case(monkeypatch) as (c, executor, driver, batch, processor, result, forward):

        async def handler(b, r):
            raise AssertionError("never dispatched")

        bridge = CUDAScheduleBridge(executor, driver, batch, pool_owner=c.owner)
        with pytest.raises(ValueError, match="synchronous"):
            bridge.run(
                forward=lambda: pytest.fail("forward started"),
                processor=processor,
                result_handler=handler,
            )
        assert bridge.state == "attached"
        assert all(tuple(r.output_ids) == (7, 8) for r in batch.reqs)
