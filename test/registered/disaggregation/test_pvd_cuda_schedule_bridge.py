"""CUDA batch result ownership: real CPU math and the source entrypoint hook."""

import ast
from array import array
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS

import pytest
import torch
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError
from sglang.srt.disaggregation.pvd.cuda_prefetch_request import CUDAPrefetchRequest
from sglang.srt.disaggregation.pvd.cuda_refresh_driver import CUDARefreshDriver
from sglang.srt.disaggregation.pvd.cuda_schedule_bridge import (
    CUDAScheduleBridge,
    _declared_device_matches_materialized,
)
from test_pvd_cuda_model_attention import run
from test_pvd_cuda_rank_batch import setup as rank_setup


def test_generic_cuda_batch_declaration_matches_verified_physical_pool():
    assert _declared_device_matches_materialized("cuda", torch.device("cuda:0"))
    assert _declared_device_matches_materialized("cuda:1", torch.device("cuda:1"))
    assert not _declared_device_matches_materialized("cuda:1", torch.device("cuda:0"))
    assert not _declared_device_matches_materialized("cpu", torch.device("cuda:0"))


def source_entrypoint():
    """Execute the real hook body without importing unrelated GPU frontend code."""
    root = Path(__file__).resolve().parents[3]
    path = (
        root
        / "python/sglang/srt/managers/scheduler_components/batch_result_processor.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SchedulerBatchResultProcessor"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "process_batch_result_decode"
    )
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {"ScheduleBatch": object, "GenerationBatchResult": object}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[fn.name]


@contextmanager
def case(monkeypatch, *, req_factory=None):
    c, executor, members, arbiter = rank_setup(monkeypatch)
    driver = CUDARefreshDriver(arbiter, max_requests=2, max_prefix_tokens=32)
    reqs = []
    for member in members:
        group = member.group
        # Controller policy double for registration only. Runtime, banks,
        # consumer tensor math, rank permits and result hook are real code.
        control = object.__new__(CUDAPrefetchRequest)
        control.group, control._closed, control.delivery = group, False, None
        control._active = control._ready = None
        control._tasks = ()
        control._metadata, control._routes = group.describe_banks(), {0: None}
        control._session = NS(
            _copy_unknown=False, observe=lambda n: None, close=lambda: None
        )
        control.pipeline = NS(
            _lock=c.consumer._lock,
            _quarantined=False,
            probe=NS(_quarantined=False),
            provider=NS(degraded=False),
            draft_config=NS(predict_tokens=1),
        )
        if req_factory:
            request = req_factory(group.coordinator.identity[0], member.slot)
        else:
            request = NS(
                rid=group.coordinator.identity[0],
                req_pool_idx=member.slot,
                origin_input_ids=array("q", [1, 2, 3, 4]),
                output_ids=array("q", [7]),
                is_retracted=False,
                ended=False,
            )
            request.finished = lambda r=request: r.ended
        driver.register(request, control, clients={0: object()}, timeout_seconds=10)
        request.output_ids.append(8)
        reqs.append(request)
    batch = NS(
        reqs=reqs,
        device=c.consumer.device,
        req_to_token_pool=c.req,
        enable_overlap=False,
        forward_mode=NS(is_decode=lambda: True),
        spec_algorithm=NS(is_none=lambda: True),
        is_spec_v2=False,
    )
    processor = NS(enable_overlap=False, enable_overlap_mlx=False)
    processor.process_batch_result_decode = MethodType(source_entrypoint(), processor)
    processor._process_batch_result_decode = lambda b, r: [
        req.output_ids.append(token)
        for req, token in zip(b.reqs, r.next_token_ids, strict=True)
    ]
    result = NS(copy_done=None, next_token_ids=[13, 17])

    def forward():
        run(c)
        return result

    try:
        yield c, executor, driver, batch, processor, result, forward
    finally:
        driver.begin_shutdown()
        for _ in range(20):
            driver.poll()
            if driver.snapshot()["drained"]:
                break
        assert driver.snapshot()["drained"]
        driver.close_loop()


def test_sampled_tokens_written_once_under_all_runtime_permits(monkeypatch):
    with case(monkeypatch) as (c, executor, driver, batch, processor, result, forward):
        bridge = CUDAScheduleBridge(executor, driver, batch, pool_owner=c.owner)
        normal = processor._process_batch_result_decode

        def process(b, r):
            assert driver.arbiter.busy and executor.dispatcher._ticket is not None
            assert all(
                s.registration.controller.group.runtime._forward for s in bridge.records
            )
            c.owner.request_release()
            assert not c.released
            return normal(b, r)

        processor._process_batch_result_decode = process
        bridge.run(forward=forward, processor=processor)
        assert bridge.state == "completed" and not driver.arbiter.busy
        assert [tuple(r.output_ids) for r in batch.reqs] == [(7, 8, 13), (7, 8, 17)]
        assert c.released == [True]
        with pytest.raises(LifecycleError, match="replayed"):
            bridge.run(forward=forward, processor=processor)
        with pytest.raises(LifecycleError, match="outside"):
            processor.process_batch_result_decode(batch, result)


def test_early_result_callback_cannot_retire_or_commit(monkeypatch):
    with case(monkeypatch) as (c, executor, driver, batch, processor, result, forward):
        bridge = CUDAScheduleBridge(executor, driver, batch, pool_owner=c.owner)
        with pytest.raises(LifecycleError, match="outside"):
            processor.process_batch_result_decode(batch, result)
        assert bridge.state == "attached"
        assert all(tuple(r.output_ids) == (7, 8) for r in batch.reqs)
        bridge.run(forward=forward, processor=processor)


def test_continuous_batch_reuse_keeps_old_results_rejected(monkeypatch):
    with case(monkeypatch) as (c, executor, driver, batch, processor, result, forward):
        first = CUDAScheduleBridge(executor, driver, batch, pool_owner=c.owner)
        first.run(forward=forward, processor=processor)
        second = CUDAScheduleBridge(executor, driver, batch, pool_owner=c.owner)
        assert batch.pvd_cuda_result_bridge is second
        with pytest.raises(LifecycleError, match="outside"):
            processor.process_batch_result_decode(batch, result)
        c.req.req_to_token[1, 6] = 14
        c.req.req_to_token[2, 6] = 15
        c.batch.positions = torch.tensor([6, 6])
        c.batch.seq_lens = torch.tensor([7, 7])
        c.batch.out_cache_loc = torch.tensor([14, 15])
        second.run(forward=forward, processor=processor)
        assert first.state == second.state == "completed"
        assert [tuple(r.output_ids) for r in batch.reqs] == [
            (7, 8, 13, 13),
            (7, 8, 17, 17),
        ]


def test_normal_unbound_result_path_is_unchanged():
    calls = []
    processor = NS(_process_batch_result_decode=lambda b, r: calls.append((b, r)))
    processor.process_batch_result_decode = MethodType(source_entrypoint(), processor)
    batch, result = NS(), object()
    processor.process_batch_result_decode(batch, result)
    assert calls == [(batch, result)]


@pytest.mark.parametrize(
    "fault", ["order", "output", "slot", "finish", "sampled_rows", "copy_event"]
)
def test_dispatch_changes_refuse_all_tokens(monkeypatch, fault):
    with case(monkeypatch) as (c, executor, driver, batch, processor, result, forward):
        bridge = CUDAScheduleBridge(executor, driver, batch, pool_owner=c.owner)
        original = tuple(batch.reqs)
        called = []
        processor._process_batch_result_decode = lambda *args: called.append(True)

        def changed():
            value = forward()
            if fault == "order":
                batch.reqs.reverse()
            elif fault == "output":
                batch.reqs[0].output_ids[0] = 99
            elif fault == "slot":
                batch.reqs[0].req_pool_idx = 2
            elif fault == "finish":
                batch.reqs[0].ended = True
            elif fault == "sampled_rows":
                result.next_token_ids = [13]
            else:
                result.copy_done = object()
            return value

        with pytest.raises(LifecycleError):
            bridge.run(forward=changed, processor=processor)
        assert not called and bridge.state == "failed"
        assert all(len(r.output_ids) == 2 for r in original)
        assert all(r.stopping for r in driver._records.values())
        assert not driver.arbiter.busy


@pytest.mark.parametrize("fault", ["partial", "extra_token", "rewrite_evidence"])
def test_bad_processor_never_rolls_back_or_replays_partial_writes(monkeypatch, fault):
    with case(monkeypatch) as (c, executor, driver, batch, processor, result, forward):
        bridge = CUDAScheduleBridge(executor, driver, batch, pool_owner=c.owner)

        def broken(b, r):
            b.reqs[0].output_ids.append(13)
            if fault == "partial":
                raise RuntimeError("partial commit")
            if fault == "extra_token":
                b.reqs[0].output_ids.append(99)
                b.reqs[1].output_ids.append(17)
            else:
                r.next_token_ids[1] = 99
                b.reqs[1].output_ids.append(99)

        processor._process_batch_result_decode = broken
        with pytest.raises((LifecycleError, RuntimeError)):
            bridge.run(forward=forward, processor=processor)
        assert (
            batch.reqs[0].output_ids[2] == 13
        )  # Never roll back authoritative writes.
        assert bridge.state == "failed"
        assert all(r.stopping for r in driver._records.values())
        with pytest.raises(LifecycleError, match="replayed"):
            bridge.run(forward=forward, processor=processor)


def test_real_req_and_result_processor_write_the_same_sampled_tokens(monkeypatch):
    try:
        from pvd_scheduled_result_smoke import make_processor
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    except Exception as exc:
        pytest.skip(
            f"real result processor import unavailable: {type(exc).__name__}: {exc}"
        )

    def real_req(rid, slot):
        params = SamplingParams(max_new_tokens=32, ignore_eos=True)
        params.normalize(None)
        request = Req(rid, "", [1, 2, 3, 4], params, vocab_size=32)
        request.req_pool_idx = slot
        request.output_ids.append(7)
        return request

    with case(monkeypatch, req_factory=real_req) as (c, executor, driver, b, _, _, _):
        batch = ScheduleBatch(
            reqs=b.reqs,
            device=c.consumer.device,
            req_to_token_pool=c.req,
            enable_overlap=False,
            forward_mode=ForwardMode.DECODE,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            return_logprob=False,
        )
        processor = make_processor()
        bridge = CUDAScheduleBridge(executor, driver, batch, pool_owner=c.owner)

        def forward():
            run(c)
            return GenerationBatchResult(next_token_ids=[13, 17])

        bridge.run(forward=forward, processor=processor)
        assert bridge.state == "completed"
        assert [tuple(r.output_ids) for r in batch.reqs] == [(7, 8, 13), (7, 8, 17)]
