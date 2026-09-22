"""CUDA bridge policies using CPU tensors and substituted driver/RNG calls."""

import asyncio
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.cuda_probe_search import (
    CUDAPredictionPipeline,
    CUDAProbeSearchSession,
)
from sglang.srt.disaggregation.pvd.cuda_target_probe import CUDALlamaTargetProbe
from sglang.srt.disaggregation.pvd.prediction import PredictionConfigError
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
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: calls.append(str(device))
    )
    return store, session, window, pipeline, route, budget, calls


def prepare(session, window, pipeline, route):
    return session.prepare(
        window, pipeline, routes=(route,), head_mapping=QueryHeadMapping(1, 1)
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

    def fail(device):
        assert device == torch.device("cuda:0")
        raise RuntimeError("copy completion unknown")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    with pytest.raises(RuntimeError, match="completion unknown"):
        with session._prepare_scope((route,), window):
            session._query_rows(source, (1,), 0, pipeline)
    assert quarantined == [True] and session._copy_unknown
    assert session._copy_retained[0] is source and len(session._copy_retained) == 2
    session.close()
    assert budget.snapshot()["used_staging_bytes"] == 64
    assert session._copy_retained[0] is source


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
