"""Owner identity and aggregate-budget tests for production CUDA composition."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from sglang.srt.disaggregation.pvd import cuda_serving_startup as startup
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError
from sglang.srt.disaggregation.pvd.cuda_serving_limits import CUDAServingLimits


def _setup(monkeypatch, *, wrong_lock=False, wrong_budget=False, fail_target=False):
    observed = {}
    model_config = NS(
        context_len=128,
        head_dim=128,
        get_total_num_kv_heads=lambda: 4,
    )
    runner = NS(
        model=NS(config=NS(num_attention_heads=28)),
        model_config=model_config,
        gpu_id=0,
        token_to_kv_pool_allocator=NS(
            get_kvcache=lambda: NS(
                get_key_buffer=lambda _: torch.empty(1, dtype=torch.float16)
            )
        ),
    )
    args = NS(
        pvd_cuda_predictive_serving=True,
        pvd_predictive_retrieval_config=True,
        pvd_kv_refresh_interval=16,
        pvd_draft_predict_tokens=4,
        pvd_retrieval_bank_budget_bytes=1_000_000,
        pvd_retrieval_scratch_budget_bytes=2_000_000,
        pvd_draft_scratch_budget_bytes=500_000,
        pvd_draft_persistent_budget_bytes=1_000_000,
        pvd_retrieval_top_k=2,
        pvd_retrieval_max_union_tokens=4,
        pvd_draft_model_path="/draft",
        pvd_draft_revision="revision",
        pvd_draft_mem_fraction_static=0.1,
        pvd_retrieval_vector_space="target-qwen",
        pvd_retrieval_metric="ip",
        max_total_tokens=128,
    )
    manager = object()
    scheduler = NS(
        server_args=args,
        tp_worker=NS(model_runner=runner),
        disagg_decode_prealloc_queue=NS(kv_manager=manager),
        max_running_requests=2,
        pvd_cuda_components=None,
    )
    limits = CUDAServingLimits(
        max_sequence_tokens=64,
        lead_tokens=3,
        attention_chunk_tokens=8,
        request_timeout_seconds=10.0,
        poll_interval_seconds=0.01,
        max_pending_events=4,
        max_pending_bytes=4096,
        draft_transient_bytes_bound=1024,
        probe_transient_bytes_bound=2048,
        target_scratch_max_reservations=16,
        bank_max_reservations=4,
    )

    def prediction_factory(target_runner, **kw):
        observed["prediction"] = kw
        return NS(
            target_scratch_budget=(
                object() if wrong_budget else kw["target_scratch_budget"]
            ),
            pipeline=NS(probe=NS(_execution_lock=kw["execution_lock"])),
        )

    def target_factory(target_scheduler, **kw):
        observed["target"] = kw
        if fail_target:
            raise RuntimeError("target installation failed")
        lock = object() if wrong_lock else kw["execution_lock"]
        return NS(
            execution_lock=lock,
            backend=NS(consumer=NS(_lock=lock)),
            workspace=NS(_budget=kw["target_scratch_budget"]),
            close_drained=lambda: None,
        )

    monkeypatch.setattr(startup, "build_cuda_prediction_startup", prediction_factory)
    monkeypatch.setattr(startup, "install_cuda_target_components", target_factory)
    return scheduler, limits, observed


def test_composition_shares_one_lock_and_target_scratch_budget(monkeypatch):
    scheduler, limits, observed = _setup(monkeypatch)
    installed = startup.install_cuda_predictive_serving(scheduler, limits)
    prediction, target = observed["prediction"], observed["target"]
    assert scheduler.pvd_cuda_components is installed
    assert installed.target_scratch_budget is prediction["target_scratch_budget"]
    assert installed.target_scratch_budget is target["target_scratch_budget"]
    assert installed.target.execution_lock is prediction["execution_lock"]
    assert prediction["max_prefix_tokens"] == 60
    assert target["max_prefix_tokens"] == 64
    admission = target["prepare_cuda_admission"](object())
    assert admission.pipeline is installed.prediction.pipeline
    assert admission.bank_budget is installed.bank_budget
    assert admission.staging_budget is installed.target_scratch_budget
    assert admission.copy_budget is installed.target_scratch_budget
    assert admission.aggregate_budget is installed.target_scratch_budget
    assert admission.execution_lock is installed.target.execution_lock
    assert installed.head_mapping.num_query_heads == 28
    assert installed.head_mapping.total_kv_heads == 4
    assert target["attention_impl"] == "online"
    assert target["max_sequence_tokens"] == 64
    assert target["num_query_heads"] == 28
    assert target["total_kv_heads"] == 4


def test_opt_in_sdpa_reaches_target_workspace_without_changing_prediction(monkeypatch):
    scheduler, limits, observed = _setup(monkeypatch)
    startup.install_cuda_predictive_serving(
        scheduler, replace(limits, attention_impl="sdpa_bounded")
    )
    assert observed["target"]["attention_impl"] == "sdpa_bounded"
    assert "attention_impl" not in observed["prediction"]


def test_opt_in_qwen_precompile_precedes_target_publication(monkeypatch):
    from sglang.srt.disaggregation.pvd import qwen_kernel_precompile
    from sglang.srt.models import qwen2

    scheduler, limits, observed = _setup(monkeypatch)

    class Qwen:
        config = NS(num_attention_heads=28)

    target_model, draft_model = Qwen(), Qwen()
    scheduler.tp_worker.model_runner.model = target_model
    monkeypatch.setattr(qwen2, "Qwen2ForCausalLM", Qwen)
    original_prediction = startup.build_cuda_prediction_startup
    original_target = startup.install_cuda_target_components
    events = []

    def prediction(*args, **kwargs):
        value = original_prediction(*args, **kwargs)
        value.draft_runner = NS(model=draft_model, token_to_kv_pool=object())
        return value

    def target(*args, **kwargs):
        assert events == ["target-jit", "draft-jit"]
        return original_target(*args, **kwargs)

    def precompile(model, pool, *, device):
        assert str(device) == "cuda:0"
        events.append("target-jit" if model is target_model else "draft-jit")

    monkeypatch.setenv("PVD_PRECOMPILE_QWEN_KERNELS", "1")
    monkeypatch.setattr(startup, "build_cuda_prediction_startup", prediction)
    monkeypatch.setattr(startup, "install_cuda_target_components", target)
    monkeypatch.setattr(
        qwen_kernel_precompile, "precompile_qwen_decode_kernels", precompile
    )
    installed = startup.install_cuda_predictive_serving(scheduler, limits)
    assert installed.prediction.draft_runner.model is draft_model
    assert observed["target"] is not None
    assert events == ["target-jit", "draft-jit"]


@pytest.mark.parametrize("wrong_lock,wrong_budget", [(True, False), (False, True)])
def test_composition_refuses_independent_execution_owners(
    monkeypatch, wrong_lock, wrong_budget
):
    scheduler, limits, _ = _setup(
        monkeypatch, wrong_lock=wrong_lock, wrong_budget=wrong_budget
    )
    before = len(startup._STARTUP_QUARANTINE)
    with pytest.raises(LifecycleError):
        startup.install_cuda_predictive_serving(scheduler, limits)
    assert len(startup._STARTUP_QUARANTINE) == before + 1
    assert scheduler.pvd_cuda_components is None


def test_failed_target_install_keeps_loaded_draft_for_process_exit(monkeypatch):
    scheduler, limits, _ = _setup(monkeypatch, fail_target=True)
    before = len(startup._STARTUP_QUARANTINE)
    with pytest.raises(RuntimeError, match="target installation failed"):
        startup.install_cuda_predictive_serving(scheduler, limits)
    assert len(startup._STARTUP_QUARANTINE) == before + 1
    assert scheduler.pvd_cuda_components is None


def test_disabled_mode_cannot_invoke_composition(monkeypatch):
    scheduler, limits, observed = _setup(monkeypatch)
    scheduler.server_args.pvd_cuda_predictive_serving = False
    with pytest.raises(LifecycleError, match="opt-in"):
        startup.install_cuda_predictive_serving(scheduler, limits)
    assert not observed


def test_disabled_startup_gate_has_no_loader_side_effects(monkeypatch):
    scheduler, _, _ = _setup(monkeypatch)
    scheduler.server_args.pvd_cuda_predictive_serving = False
    assert startup.maybe_install_cuda_predictive_serving(scheduler) is None


def test_enabled_startup_gate_loads_config_and_installs(monkeypatch):
    from sglang.srt.disaggregation.pvd import cuda_serving_limits

    scheduler, limits, _ = _setup(monkeypatch)
    scheduler.server_args.pvd_cuda_serving_config = "/explicit/limits.json"
    seen = []

    def load(path, *, refresh_interval, predict_tokens):
        seen.append((path, refresh_interval, predict_tokens))
        return limits

    monkeypatch.setattr(cuda_serving_limits, "load_cuda_serving_limits", load)
    monkeypatch.setattr(
        startup,
        "install_cuda_predictive_serving",
        lambda scheduler_arg, limits_arg: (scheduler_arg, limits_arg),
    )
    assert startup.maybe_install_cuda_predictive_serving(scheduler) == (
        scheduler,
        limits,
    )
    assert seen == [("/explicit/limits.json", 16, 4)]


def test_scheduler_installs_before_publication_and_only_when_opted_in():
    source = (
        Path(__file__).resolve().parents[3] / "python/sglang/srt/managers/scheduler.py"
    ).read_text(encoding="utf-8")
    before, hook, published = (
        source.index("self.init_batch_result_processor()"),
        source.index("maybe_install_cuda_predictive_serving(self)"),
        source.index("self.is_initializing = False"),
    )
    assert before < hook < published
    gate = source[before:hook]
    assert "if self.server_args.pvd_cuda_predictive_serving:" in gate
