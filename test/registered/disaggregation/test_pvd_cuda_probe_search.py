"""CUDA bridge policies using CPU tensors and substituted driver/RNG calls."""

import asyncio
import logging
import re
import threading
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.cuda_probe_search import (
    CUDAPredictionPipeline,
    CUDAProbeSearchSession,
)
from sglang.srt.disaggregation.pvd.cuda_target_probe import CUDALlamaTargetProbe
from sglang.srt.disaggregation.pvd.concurrent_prediction_worker import (
    PredictionCancelledError,
)
from sglang.srt.disaggregation.pvd.draft_hf import VocabularySignature
from sglang.srt.disaggregation.pvd.prediction import (
    DraftPrediction,
    PredictionConfigError,
    QueryVectors,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferBudget,
    TransferCapacityError,
)
from test_pvd_draft_forward import adapter
from test_pvd_draft_sglang import factory, provider
from test_pvd_probe_search import ScratchProbe, setup
from test_pvd_prompt_index import shard_client


def bridge(monkeypatch):
    _, store, old, window, original, _, route = setup()
    lock = threading.RLock()

    class PolicyProbe(ScratchProbe, CUDALlamaTargetProbe):
        # CPU capture substitution. Real CUDA probe lifecycle has separate tests.
        pass

    probe = PolicyProbe(original.probe.vector)
    probe.device, probe.config = "cuda:0", original.probe_config
    probe._execution_lock, probe._quarantined = lock, False
    made_provider = provider(factory(executor=adapter(transient_bytes_bound=2048)))
    pipeline = CUDAPredictionPipeline(
        made_provider,
        probe,
        made_provider.config,
        original.probe_config,
        execution_lock=lock,
    )
    budget = TransferBudget(4096, 1)
    session = CUDAProbeSearchSession(
        old.request_id,
        old.entry_transfer_id,
        device="cuda:0",
        copy_budget=budget,
        max_head_dim=8,
    )
    window = session.begin(window.prefix, target_tokens=4, query_positions=(5,))
    monkeypatch.setattr(session, "_query_device", lambda t: t.device.type == "cpu")
    calls = []
    actual_fork = torch.random.fork_rng

    @contextmanager
    def fork(*, devices, enabled):
        calls.append(tuple(devices))
        with actual_fork(devices=[], enabled=enabled):
            yield

    monkeypatch.setattr(torch.random, "fork_rng", fork)
    class FakeStream:
        def synchronize(self):
            calls.append("cuda:0")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: FakeStream())
    return store, session, window, pipeline, route, budget, calls


def prepare(session, window, pipeline, route):
    return session.prepare(
        window, pipeline, routes=(route,), head_mapping=QueryHeadMapping(1, 1)
    )


def test_pipeline_steps_release_same_gpu_for_formal_batch(monkeypatch):
    _, _, window, pipeline, _, _, _ = bridge(monkeypatch)
    prefix = window.prefix

    def draft_steps(prefix, max_tokens):
        assert max_tokens == 2
        yield "private-draft-forward"
        return DraftPrediction(prefix.request_id, prefix.version, (31, 32))

    def target_steps(prefix, prediction):
        assert prediction.tokens == (31, 32)
        yield "private-target-forward"
        positions = (len(prefix.tokens), len(prefix.tokens) + 1)
        return (
            QueryVectors(
                vector_space=pipeline.probe_config.target_model_id,
                version="private-target-q",
                layer=0,
                head_start=0,
                head_count=1,
                positions=positions,
                valid_length=2,
                vectors=torch.ones((2, 1, 8)),
                prefix_version=prefix.version,
                positional_encoding="rope_applied",
                request_id=prefix.request_id,
            ),
        )

    monkeypatch.setattr(pipeline.provider, "iter_predict", draft_steps)
    monkeypatch.setattr(pipeline.probe, "capture_steps", target_steps)

    def formal_worker_can_take_gpu():
        acquired = []

        def check():
            got = pipeline._lock.acquire(blocking=False)
            acquired.append(got)
            if got:
                pipeline._lock.release()

        thread = threading.Thread(target=check)
        thread.start()
        thread.join()
        return acquired == [True]

    steps = pipeline.iter_queries(prefix)
    assert next(steps) == "private-draft-forward"
    assert formal_worker_can_take_gpu()
    assert next(steps) == "private-target-forward"
    assert formal_worker_can_take_gpu()
    with pytest.raises(StopIteration) as finished:
        next(steps)
    assert finished.value.value[0].version == "private-target-q"
    assert not pipeline._scope_active
    assert not pipeline.provider._active


def test_cuda_prediction_logs_draft_probe_and_scope_entry_costs(monkeypatch, caplog):
    _, session, window, pipeline, route, _, _ = bridge(monkeypatch)
    with caplog.at_level(
        logging.INFO, logger="sglang.srt.disaggregation.pvd.cuda_probe_search"
    ):
        prepared = prepare(session, window, pipeline, route)
    assert prepared.queries
    messages = [
        record.message
        for record in caplog.records
        if record.name == "sglang.srt.disaggregation.pvd.cuda_probe_search"
    ]
    assert len(messages) == 2
    assert "PVD CUDA prediction stages:" in messages[0]
    assert "PVD CUDA prediction scope entered:" in messages[1]
    for field in ("draft_seconds", "probe_seconds", "enter_seconds"):
        match = next(
            (
                re.search(rf"\b{field}=([0-9]+\.[0-9]+)\b", message)
                for message in messages
                if field in message
            ),
            None,
        )
        assert match is not None
        assert float(match.group(1)) >= 0


def test_cuda_pipeline_requires_same_exact_draft_and_target_tokenizer(monkeypatch):
    _, _, _, pipeline, _, _, _ = bridge(monkeypatch)
    vocabulary = VocabularySignature(
        size=8,
        bos_token_id=1,
        eos_token_id=9,
        fingerprint="probe",
        allowed_ids=frozenset(range(8)) | {9},
        mapping_fingerprint="full-map",
    )
    pipeline.provider.vocabulary = vocabulary
    with pytest.raises(PredictionConfigError, match="same exact tokenizer mapping"):
        CUDAPredictionPipeline(
            pipeline.provider,
            pipeline.probe,
            pipeline.draft_config,
            pipeline.probe_config,
            execution_lock=pipeline._lock,
        )
    pipeline.probe.vocabulary = vocabulary
    CUDAPredictionPipeline(
        pipeline.provider,
        pipeline.probe,
        pipeline.draft_config,
        pipeline.probe_config,
        execution_lock=pipeline._lock,
    )


def test_cuda_policy_capture_to_actual_http_preserves_rows_identity_and_rng(
    monkeypatch,
):
    async def run():
        store, session, window, pipeline, route, budget, calls = bridge(monkeypatch)
        rng = torch.random.get_rng_state().clone()
        prepared = prepare(session, window, pipeline, route)
        assert torch.equal(rng, torch.random.get_rng_state())
        assert (0,) in calls and "cuda:0" in calls
        assert pipeline.probe.tensor is None
        assert budget.snapshot()["used_staging_bytes"] == 0
        async with shard_client(store) as http:
            client = PVDShardSearchClient(str(http.make_url("")))
            try:
                await session.search(prepared, client)
                selected = session.take_selection(window)
                assert selected.selections[0].token_ids == (3,)
                assert selected.queries[0].query_version == "probe-output-v1"
            finally:
                await client.close()

    asyncio.run(run())


def test_worker_capture_avoids_rng_fork_and_hands_off_bounded_cpu_queries(
    monkeypatch,
):
    _, _, window, pipeline, _, _, _ = bridge(monkeypatch)
    prefix = window.prefix
    events = []

    @contextmanager
    def private_branch():
        events.append("probe_enter")
        try:
            yield
        finally:
            events.append("probe_exit")

    monkeypatch.setattr(pipeline.probe, "branch", private_branch)
    pipeline.probe.device = "cpu"
    pipeline.probe.head_dim = 8
    monkeypatch.setattr(pipeline.probe, "_drain_private", lambda: events.append("fence"))

    @contextmanager
    def draft_branch():
        events.append("draft_enter")
        try:
            yield
        finally:
            events.append("draft_exit")

    monkeypatch.setattr(pipeline.provider, "branch", draft_branch)

    def iter_predict(prefix, max_tokens):
        assert max_tokens == pipeline.draft_config.predict_tokens
        events.append("draft_step")
        yield object()
        return DraftPrediction(prefix.request_id, prefix.version, (31, 32))

    monkeypatch.setattr(pipeline.provider, "iter_predict", iter_predict)

    def capture(prefix, prediction):
        assert not torch.is_grad_enabled()
        assert prediction.tokens == (31, 32)
        events.append("target_capture")
        return (
            QueryVectors(
                vector_space=pipeline.probe_config.target_model_id,
                version="worker-q",
                layer=0,
                head_start=0,
                head_count=1,
                positions=(4, 5),
                valid_length=2,
                vectors=torch.ones((2, 1, 8)),
                prefix_version=prefix.version,
                positional_encoding="rope_applied",
                request_id=prefix.request_id,
            ),
        )

    monkeypatch.setattr(pipeline.probe, "capture", capture)
    monkeypatch.setattr(
        torch.random,
        "fork_rng",
        lambda **kwargs: pytest.fail("worker capture must not fork RNG"),
    )
    # The worker path uses its private single-flight lock, not the scheduler
    # pipeline's ordinary scope lock.
    pipeline._lock.acquire()
    try:
        queries = pipeline.capture_for_worker(prefix, lambda: None)
    finally:
        pipeline._lock.release()
    assert queries[0].vectors.device.type == "cpu"
    assert pipeline._worker_cpu_queries
    assert events == [
        "draft_enter",
        "draft_step",
        "draft_exit",
        "probe_enter",
        "target_capture",
        "fence",
        "probe_exit",
    ]
    assert not pipeline._scope_active and not pipeline._worker_capture_lock.locked()


def test_worker_capture_cancellation_closes_draft_before_probe(monkeypatch):
    _, _, window, pipeline, _, _, _ = bridge(monkeypatch)
    events, checks = [], []

    @contextmanager
    def draft_branch():
        events.append("draft_enter")
        try:
            yield
        finally:
            events.append("draft_exit")

    monkeypatch.setattr(pipeline.provider, "branch", draft_branch)

    def iter_predict(prefix, max_tokens):
        try:
            yield object()
            return DraftPrediction(prefix.request_id, prefix.version, (31, 32))
        finally:
            events.append("draft_closed")

    monkeypatch.setattr(pipeline.provider, "iter_predict", iter_predict)

    def cancel_after_one_step():
        checks.append(True)
        if len(checks) == 3:
            raise PredictionCancelledError("cancelled at completed forward boundary")

    monkeypatch.setattr(
        pipeline.probe,
        "capture",
        lambda *args: pytest.fail("cancelled work must not start target capture"),
    )
    with pytest.raises(PredictionCancelledError, match="completed forward"):
        pipeline.capture_for_worker(window.prefix, cancel_after_one_step)
    assert events == ["draft_enter", "draft_closed", "draft_exit"]
    assert not pipeline._scope_active and not pipeline._worker_capture_lock.locked()


def test_committed_worker_capture_uses_cpu_handoff_without_rng_fork(monkeypatch):
    _, _, window, pipeline, _, _, _ = bridge(monkeypatch)
    pipeline.probe.device = "cpu"
    pipeline.probe.head_dim = 8
    monkeypatch.setattr(pipeline.probe, "_drain_private", lambda: None)

    @contextmanager
    def private_branch():
        yield

    monkeypatch.setattr(pipeline.probe, "branch", private_branch)

    def capture_committed(prefix, positions):
        assert positions == (2,)
        return (
            QueryVectors(
                vector_space=pipeline.probe_config.target_model_id,
                version="committed-worker-q",
                layer=0,
                head_start=0,
                head_count=1,
                positions=positions,
                valid_length=1,
                vectors=torch.ones((1, 1, 8)),
                prefix_version=prefix.version,
                positional_encoding="rope_applied",
                request_id=prefix.request_id,
            ),
        )

    monkeypatch.setattr(pipeline.probe, "capture_committed", capture_committed)
    monkeypatch.setattr(
        torch.random,
        "fork_rng",
        lambda **kwargs: pytest.fail("committed worker capture must not fork RNG"),
    )
    queries = pipeline.capture_committed_for_worker(window.prefix, (2,))
    assert queries[0].vectors.device.type == "cpu"
    assert pipeline._worker_cpu_queries


def test_cuda_session_accepts_only_worker_marked_cpu_query_rows(monkeypatch):
    _, session, window, pipeline, route, _, calls = bridge(monkeypatch)
    monkeypatch.setattr(
        session,
        "_query_device",
        CUDAProbeSearchSession._query_device.__get__(session),
    )
    query = QueryVectors(
        vector_space=pipeline.probe_config.target_model_id,
        version="worker-cpu-q",
        layer=0,
        head_start=0,
        head_count=1,
        positions=(5,),
        valid_length=1,
        vectors=torch.ones((1, 1, 8)),
        prefix_version=window.prefix.version,
        positional_encoding="rope_applied",
        request_id=window.prefix.request_id,
    )
    pipeline._worker_cpu_queries = True
    prepared = session.prepare_from_queries(
        window,
        pipeline,
        (query,),
        routes=(route,),
        head_mapping=QueryHeadMapping(1, 1),
    )
    assert prepared.queries[0].rows == ((1.0,) * 8,)
    assert "cuda:0" not in calls


def test_budget_refuses_before_draft_probe_or_copy(monkeypatch):
    _, session, window, pipeline, route, _, calls = bridge(monkeypatch)
    session.copy_budget = TransferBudget(1, 1)
    with pytest.raises(TransferCapacityError):
        prepare(session, window, pipeline, route)
    assert calls == [] and pipeline.probe.closed == 0


def test_route_dimension_bound_refuses_before_capture(monkeypatch):
    _, session, window, pipeline, route, budget, _ = bridge(monkeypatch)
    session.max_head_dim = 1
    with pytest.raises(ValueError, match="dimension exceeds"):
        prepare(session, window, pipeline, route)
    assert budget.snapshot()["reservations"] == 0 and pipeline.probe.closed == 0


def test_copy_unknown_retains_source_destination_and_reservation(monkeypatch):
    _, session, window, _, route, budget, _ = bridge(monkeypatch)
    source = torch.ones(2, 1, 8)
    quarantined = []
    pipeline = SimpleNamespace(
        probe=SimpleNamespace(quarantine_query_copy=lambda: quarantined.append(True))
    )

    class BrokenStream:
        def synchronize(self):
            raise RuntimeError("copy completion unknown")

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: BrokenStream()
    )
    with pytest.raises(RuntimeError, match="completion unknown"):
        with session._prepare_scope((route,), window):
            session._query_rows(source, (1,), 0, pipeline)
    assert quarantined == [True] and session._copy_unknown
    assert session._copy_retained[0] is source and len(session._copy_retained) == 2
    session.close()
    assert budget.snapshot()["used_staging_bytes"] == 64
    assert session._copy_retained[0] is source


def test_cuda_query_copy_batches_routed_heads_per_layer(monkeypatch):
    _, session, window, _, route, budget, calls = bridge(monkeypatch)
    routes = tuple(replace(route, query_head=head) for head in range(3))
    source = torch.arange(48, dtype=torch.float16).reshape(2, 3, 8)
    pipeline = SimpleNamespace(probe_config=SimpleNamespace(head_start=0))
    with session._prepare_scope(routes, window):
        for head in range(3):
            assert session._query_rows(source, (1,), head, pipeline) == (
                tuple(float(v) for v in source[1, head]),
            )
        assert budget.snapshot()["used_staging_bytes"] == 192
        assert calls.count("cuda:0") == 1
    assert budget.snapshot()["used_staging_bytes"] == 0
    assert not session._copy_retained


def test_cuda_query_copy_respects_nonzero_query_head_start(monkeypatch):
    _, session, window, _, route, _, calls = bridge(monkeypatch)
    routes = (replace(route, query_head=4), replace(route, query_head=5))
    source = torch.arange(32, dtype=torch.float32).reshape(2, 2, 8)
    pipeline = SimpleNamespace(probe_config=SimpleNamespace(head_start=4))
    with session._prepare_scope(routes, window):
        assert session._query_rows(source, (1,), 1, pipeline) == (
            tuple(float(v) for v in source[1, 1]),
        )
        assert session._query_rows(source, (1,), 0, pipeline) == (
            tuple(float(v) for v in source[1, 0]),
        )
        assert calls.count("cuda:0") == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_query_copy_pinned_batch_matches_source_and_budget(monkeypatch):
    original = torch.cuda.current_stream
    _, session, window, _, route, budget, _ = bridge(monkeypatch)
    routes = tuple(replace(route, query_head=head) for head in range(3))
    source = torch.arange(48, dtype=torch.float16, device="cuda:0").reshape(2, 3, 8)
    expected = source.cpu()
    pipeline = SimpleNamespace(probe_config=SimpleNamespace(head_start=0))
    synchronizations = []

    class RecordingStream:
        def __init__(self, stream):
            self.stream = stream

        def synchronize(self):
            synchronizations.append(torch.device("cuda:0"))
            self.stream.synchronize()

    def current_stream(device):
        return RecordingStream(original(device))

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    with session._prepare_scope(routes, window):
        for head in range(3):
            assert session._query_rows(source, (1,), head, pipeline) == (
                tuple(float(v) for v in expected[1, head]),
            )
        assert synchronizations == [torch.device("cuda:0")]
        assert session._copy_retained[1].is_pinned()
        assert budget.snapshot()["used_staging_bytes"] == 192
    assert budget.snapshot()["used_staging_bytes"] == 0


def test_nonfinite_query_releases_copy_budget_after_successful_fence(monkeypatch):
    _, session, window, pipeline, route, budget, _ = bridge(monkeypatch)
    pipeline.probe.vector = [float("nan")] * 8
    with pytest.raises(ValueError, match="finite"):
        prepare(session, window, pipeline, route)
    assert budget.snapshot()["reservations"] == 0 and not session._copy_retained


def test_pipeline_run_cannot_bypass_locked_scope(monkeypatch):
    _, _, window, pipeline, _, _, _ = bridge(monkeypatch)
    with pytest.raises(PredictionConfigError, match="branch scope"):
        pipeline.run(window.prefix)


@pytest.mark.parametrize("phase", ["enter", "exit"])
def test_rng_failure_quarantines_and_keeps_shared_lease(monkeypatch, phase):
    _, _, _, pipeline, _, _, _ = bridge(monkeypatch)

    @contextmanager
    def broken(**kwargs):
        if phase == "enter":
            raise RuntimeError("RNG unknown")
        yield
        raise RuntimeError("RNG unknown")

    monkeypatch.setattr(torch.random, "fork_rng", broken)
    with pytest.raises(RuntimeError, match="RNG unknown"):
        with pipeline._scope():
            pass
    assert pipeline._quarantined
    acquired = []

    def peer():
        success = pipeline._lock.acquire(blocking=False)
        acquired.append(success)
        if success:
            pipeline._lock.release()

    thread = threading.Thread(target=peer)
    thread.start()
    thread.join(timeout=5)
    assert acquired == [False]
    with pytest.raises(PredictionConfigError, match="quarantined"):
        with pipeline._scope():
            pass


def test_query_source_device_dtype_are_not_inferred_from_cast(monkeypatch):
    _, session, _, _, _, _, _ = bridge(monkeypatch)
    check = CUDAProbeSearchSession._query_device
    assert not check(session, torch.ones(2, 1, 8))
    assert check(
        session, SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.float16)
    )
    assert not check(
        session, SimpleNamespace(device=torch.device("cuda:1"), dtype=torch.float16)
    )
    assert not check(
        session, SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.int64)
    )


def test_failed_copy_is_fenced_before_refunding(monkeypatch):
    _, session, window, pipeline, route, budget, calls = bridge(monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("copy submission failed")

    monkeypatch.setattr(torch.Tensor, "copy_", fail)
    with pytest.raises(RuntimeError, match="copy submission"):
        prepare(session, window, pipeline, route)
    assert "cuda:0" in calls
    assert budget.snapshot()["reservations"] == 0
    assert not session._copy_retained and not session._copy_unknown


def test_constructor_requires_real_adapter_matching_configs_and_reentrant_lock(
    monkeypatch,
):
    _, _, _, pipeline, _, _, _ = bridge(monkeypatch)
    values = (
        pipeline.provider,
        pipeline.probe,
        pipeline.draft_config,
        pipeline.probe_config,
    )
    with pytest.raises(PredictionConfigError, match="reentrant"):
        CUDAPredictionPipeline(*values, execution_lock=threading.Lock())
    with pytest.raises(PredictionConfigError, match="exact CUDA probe"):
        CUDAPredictionPipeline(*values, execution_lock=threading.RLock())
    pipeline.provider.factory._executor = object()
    with pytest.raises(PredictionConfigError, match="concrete prediction-only"):
        CUDAPredictionPipeline(*values, execution_lock=pipeline._lock)


def test_real_probe_keeps_captured_queries_and_lease_on_copy_unknown(monkeypatch):
    from test_pvd_cuda_target_probe import environment

    case = environment(monkeypatch)
    probe = case.probe
    with probe.branch():
        probe.capture(case.prefix, case.prediction)
        probe.quarantine_query_copy()
    assert probe._quarantined and probe._state is not None
    assert probe._execution_held
    assert probe.budget.snapshot()["used_staging_bytes"] > 0
