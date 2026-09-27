"""Private-pool and completion policies with CPU tensors/interface doubles.

No CUDA model forward is claimed. Real CPU model regression runs separately.
"""

import json
import sys
import threading
import types
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd import draft_forward_adapter, target_probe
from sglang.srt.disaggregation.pvd.cuda_target_probe import (
    CUDALlamaTargetProbe,
    CUDAQwen2TargetProbe,
)
from sglang.srt.disaggregation.pvd.draft_hf import VocabularySignature
from sglang.srt.disaggregation.pvd.prediction import (
    CommittedPrefix,
    DraftPrediction,
    PredictionConfigError,
    ProbeConfig,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferBudget,
    TransferCapacityError,
)


def environment(
    monkeypatch,
    *,
    fail=None,
    dtype=torch.float16,
    architecture="LlamaForCausalLM",
    vocabulary=None,
    prefix_cache=False,
):
    events, tensors, holders, forward_shapes = [], [], {}, []
    lock = threading.Lock()
    budget = TransferBudget(65536, 1)
    config = ProbeConfig("target", (0, 1), head_start=1, head_count=2)

    def publish(name, **attributes):
        module = types.ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    class Model:
        quant_config = None
        config = SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=100,
        )

        def parameters(self):
            return [SimpleNamespace(device=torch.device("cuda:0"), dtype=dtype)]

        def model(self, ids, positions, batch):
            events.append("forward")
            forward_shapes.append(
                (tuple(positions.tolist()), getattr(batch, "pvd_compact_extend", False))
            )
            if hasattr(batch, "pvd_query_capture"):
                for layer in (0, 1):
                    q = torch.arange(ids.numel() * 12, dtype=torch.float32).reshape(
                        -1, 12
                    )
                    batch.pvd_query_capture.capture(layer, positions, q)
            if fail == "forward":
                raise RuntimeError("forward failed")

    model = Model()
    publish("sglang.srt.models.llama", LlamaForCausalLM=Model)
    publish("sglang.srt.models.qwen2", Qwen2ForCausalLM=Model)
    publish(
        "sglang.srt.compilation.piecewise_context_manager",
        get_forward_context=lambda: None,
    )

    @contextmanager
    def context(value):
        events.append("context_enter")
        try:
            yield
        finally:
            events.append("context_exit")

    publish(
        "sglang.srt.model_executor.forward_context",
        ForwardContext=SimpleNamespace,
        forward_context=context,
    )

    def remember():
        tensor = torch.ones(16)
        tensors.append(weakref.ref(tensor))
        return tensor

    class Requests:
        def __init__(self, size, length, device, saver):
            probe = holders["probe"]
            assert probe._private_state is not None or (
                prefix_cache
                and any(
                    record.owner is not None for record in probe._prefix_caches.values()
                )
            )
            assert budget.snapshot()["used_staging_bytes"] > 0
            assert device == "cuda:0"
            self.tensor = remember()
            self.req_to_token = torch.zeros((2, length), dtype=torch.int32)
            events.append("requests")

    class Pool:
        def __init__(self, size, page, actual_dtype, heads, dim, layers, device, saver):
            assert actual_dtype == dtype and device == "cuda:0"
            self.tensor = remember()
            events.append("pool")
            if fail == "pool":
                raise RuntimeError("partial pool constructor")

    class Allocator:
        def __init__(self, requests, kv):
            self.requests, self.kv = requests, kv

        def alloc_request(self):
            events.append("allocate_slot")
            if fail == "slot":
                raise RuntimeError("slot failure")
            return 1

        def alloc_kv(self, n):
            return list(range(1, n + 1))

        def write_mapping(self, *args):
            events.append("map")

        def clear_mapping(self, slot):
            events.append("clear")
            if fail == "cleanup":
                raise RuntimeError("cleanup failed")

        def free_kv(self, rows):
            events.append("free_kv")

        def free_request(self, slot):
            events.append("free_slot")

    class Backend:
        def __init__(self, runner):
            self.runner = runner

        def init_forward_metadata(self, batch):
            pass

    class Builder:
        def __init__(self, *args, **kwargs):
            assert kwargs["device"] == "cuda:0"
            assert kwargs["architecture"] == architecture

        def build_forward_batch(self, inputs):
            return SimpleNamespace(
                input_ids=torch.tensor(inputs.input_ids),
                positions=torch.tensor(inputs.positions),
            )

    publish(
        "sglang.srt.mem_cache.memory_pool",
        ReqToTokenPool=Requests,
        MHATokenToKVPool=Pool,
    )
    publish(
        "sglang.srt.mem_cache.allocator.token",
        TokenToKVPoolAllocator=lambda *args: SimpleNamespace(pool=args[3]),
    )
    publish(
        "sglang.srt.layers.attention.torch_native_backend",
        TorchNativeAttnBackend=Backend,
    )
    monkeypatch.setattr(draft_forward_adapter, "PrivatePoolAllocator", Allocator)
    monkeypatch.setattr(draft_forward_adapter, "DraftForwardAdapter", Builder)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device_context = []

    @contextmanager
    def select_device(device):
        assert device == "cuda:0"
        device_context.append(device)
        try:
            yield
        finally:
            device_context.pop()

    monkeypatch.setattr(torch.cuda, "device", select_device)
    runner = SimpleNamespace(
        model=model,
        device="cuda",
        gpu_id=0,
        tp_size=1,
        pp_size=1,
        attn_cp_size=1,
        server_args=SimpleNamespace(
            attention_backend="torch_native",
            enable_dp_attention=False,
            speculative_algorithm=None,
        ),
        model_config=SimpleNamespace(context_len=32, head_dim=3),
    )
    probe_type = (
        CUDAQwen2TargetProbe
        if architecture == "Qwen2ForCausalLM"
        else CUDALlamaTargetProbe
    )
    probe = probe_type(
        runner,
        config,
        device="cuda:0",
        execution_lock=lock,
        target_model_id="target",
        max_tokens=12,
        max_predict_tokens=2,
        transient_bytes_bound=1024,
        budget=budget,
        vocabulary=vocabulary,
        prefix_budget=TransferBudget(131072, 1) if prefix_cache else None,
    )
    holders["probe"] = probe
    drains = []

    def drain():
        assert device_context == ["cuda:0"]
        events.append("drain")
        drains.append(True)
        if fail == "drain" or (fail == "cleanup_drain" and len(drains) == 2):
            raise RuntimeError("completion unknown")

    monkeypatch.setattr(probe, "_drain_private", drain)
    prefix = CommittedPrefix("r", (1, 2, 3), 0, "version")
    prediction = DraftPrediction("r", "version", (4, 5))
    return SimpleNamespace(
        probe=probe,
        runner=runner,
        lock=lock,
        budget=budget,
        events=events,
        tensors=tensors,
        prefix=prefix,
        prediction=prediction,
        forward_shapes=forward_shapes,
    )


def test_exact_tokenizer_mapping_accepts_added_eos_but_rejects_padding_hole(
    monkeypatch,
):
    vocabulary = VocabularySignature(
        size=10,
        bos_token_id=1,
        eos_token_id=12,
        fingerprint="probe",
        allowed_ids=frozenset(range(10)) | {12},
        mapping_fingerprint="full-map",
    )
    c = environment(monkeypatch, vocabulary=vocabulary)
    with c.probe.branch():
        with pytest.raises(
            PredictionConfigError, match="outside the target vocabulary"
        ):
            c.probe.capture(c.prefix, DraftPrediction("r", "version", (11,)))
    assert "forward" not in c.events
    with c.probe.branch():
        c.probe.capture(
            CommittedPrefix("r", (1, 12, 3), 0, "version"),
            DraftPrediction("r", "version", (4, 5)),
        )
    assert "forward" in c.events


def test_probe_reuses_weights_but_fences_before_cleanup_and_refund(monkeypatch):
    c = environment(monkeypatch)
    original = c.budget.release

    def release(owner):
        assert all(ref() is None for ref in c.tensors)
        assert c.probe._private_state is None
        original(owner)

    monkeypatch.setattr(c.budget, "release", release)
    with c.probe.branch():
        result = c.probe.capture(c.prefix, c.prediction)
        assert c.lock.locked() and c.probe.model is c.runner.model
        assert result[0].positions == (3, 4)
        assert result[0].positional_encoding == "rope_applied"
        assert c.events[-5:] == ["drain", "clear", "free_kv", "free_slot", "drain"]
    assert not c.lock.locked()
    assert c.budget.snapshot()["used_staging_bytes"] == 0


def test_qwen2_probe_uses_private_pool_and_same_post_rope_contract(monkeypatch):
    c = environment(monkeypatch, architecture="Qwen2ForCausalLM")
    with c.probe.branch():
        result = c.probe.capture(c.prefix, c.prediction)
        assert result[0].positions == (3, 4)
        assert result[0].positional_encoding == "rope_applied"
    assert c.budget.snapshot()["used_staging_bytes"] == 0


def test_cached_target_prefix_prefill_bounds_each_forward(monkeypatch):
    monkeypatch.setattr(target_probe, "_TARGET_PREFIX_CHUNK_TOKENS", 4)
    c = environment(monkeypatch, prefix_cache=True)
    req = SimpleNamespace(rid="r")
    c.probe.register_cached_request(req)
    prefix = CommittedPrefix("r", tuple(range(1, 10)), 0, "version")
    prediction = DraftPrediction("r", "version", (10, 11))
    with c.probe.branch():
        result = c.probe.capture(prefix, prediction)
    assert c.forward_shapes == [
        ((0, 1, 2, 3), False),
        ((4, 5, 6, 7), True),
        ((8,), True),
        ((9, 10), True),
    ]
    assert result[0].positions == (9, 10)
    assert c.probe._prefix_caches["r"].tokens == prefix.tokens
    c.probe.retire_cached_request(req)
    assert not c.probe._prefix_caches


def test_cooperative_cached_probe_releases_target_between_prefix_forwards(monkeypatch):
    monkeypatch.setattr(target_probe, "_TARGET_PREFIX_CHUNK_TOKENS", 4)
    c = environment(monkeypatch, prefix_cache=True)
    req = SimpleNamespace(rid="r")
    c.probe.register_cached_request(req)
    prefix = CommittedPrefix("r", tuple(range(1, 10)), 0, "version")
    prediction = DraftPrediction("r", "version", (10, 11))
    steps = c.probe.capture_steps(prefix, prediction)
    for completed in range(1, 4):
        next(steps)
        assert len(c.forward_shapes) == completed
        assert not c.lock.locked()
        assert c.probe._active
        # The ordinary formal batch may use the same target between steps.
        assert c.lock.acquire(blocking=False)
        c.lock.release()
    with pytest.raises(StopIteration) as finished:
        next(steps)
    assert finished.value.value[0].positions == (9, 10)
    assert len(c.forward_shapes) == 4
    assert not c.lock.locked() and not c.probe._active
    c.probe.retire_cached_request(req)


def test_cooperative_probe_reuses_one_cache_reservation_across_requests(monkeypatch):
    c = environment(monkeypatch, prefix_cache=True)
    first = SimpleNamespace(rid="r")
    second = SimpleNamespace(rid="next")
    c.probe.register_cached_request(first)
    c.probe.register_cached_request(second)
    prefix = CommittedPrefix("r", tuple(range(1, 10)), 0, "first")
    with c.probe.branch():
        c.probe.capture(prefix, DraftPrediction("r", "first", (10, 11)))
    assert c.probe._prefix_caches["r"].owner is not None

    next_prefix = CommittedPrefix("next", tuple(range(1, 10)), 0, "second")
    list(
        c.probe.capture_steps(next_prefix, DraftPrediction("next", "second", (10, 11)))
    )
    assert c.probe._prefix_caches["r"].owner is None
    assert c.probe._prefix_caches["next"].tokens == next_prefix.tokens
    c.probe.retire_cached_request(first)
    c.probe.retire_cached_request(second)
    assert c.probe.prefix_budget.snapshot()["used_staging_bytes"] == 0


def test_cancelled_cooperative_probe_retires_partial_private_prefix(monkeypatch):
    monkeypatch.setattr(target_probe, "_TARGET_PREFIX_CHUNK_TOKENS", 4)
    c = environment(monkeypatch, prefix_cache=True)
    req = SimpleNamespace(rid="r")
    c.probe.register_cached_request(req)
    prefix = CommittedPrefix("r", tuple(range(1, 10)), 0, "version")
    steps = c.probe.capture_steps(prefix, DraftPrediction("r", "version", (10, 11)))
    next(steps)
    steps.close()
    assert not c.lock.locked() and not c.probe._active
    assert c.probe._prefix_caches["r"].tokens == ()
    assert c.budget.snapshot()["used_staging_bytes"] == 0
    c.probe.retire_cached_request(req)


def test_opt_in_probe_timeline_separates_setup_forward_and_retirement(
    monkeypatch, caplog
):
    monkeypatch.setenv("PVD_PROFILE_REFRESH_TIMELINE", "1")
    c = environment(monkeypatch)
    with caplog.at_level("INFO", logger="sglang.srt.disaggregation.pvd.target_probe"):
        with c.probe.branch():
            c.probe.capture(c.prefix, c.prediction)
    records = [
        row.message for row in caplog.records if "event=probe_stage" in row.message
    ]
    assert len(records) == 1
    assert "tokens=5" in records[0]
    assert all(
        f"{stage}_ms=" in records[0] for stage in ("setup", "forward_capture", "retire")
    )
    assert c.budget.snapshot()["used_staging_bytes"] == 0


@pytest.mark.parametrize("failure", ["pool", "slot", "forward"])
def test_failed_construction_or_forward_drops_traceback_storage_before_refund(
    monkeypatch, failure
):
    c = environment(monkeypatch, fail=failure)
    original = c.budget.release

    def release(owner):
        assert all(ref() is None for ref in c.tensors)
        original(owner)

    monkeypatch.setattr(c.budget, "release", release)
    with pytest.raises(RuntimeError), c.probe.branch():
        c.probe.capture(c.prefix, c.prediction)
    assert c.events.count("drain") == 2
    assert c.budget.snapshot()["used_staging_bytes"] == 0
    assert not c.lock.locked() and not c.probe._quarantined


@pytest.mark.parametrize("failure", ["drain", "cleanup", "cleanup_drain"])
def test_unknown_completion_or_cleanup_retains_private_state_budget_and_target_lock(
    monkeypatch, failure
):
    c = environment(monkeypatch, fail=failure)
    with pytest.raises(RuntimeError), c.probe.branch():
        c.probe.capture(c.prefix, c.prediction)
    assert c.probe._quarantined and c.probe._private_state is not None
    assert c.budget.snapshot()["used_staging_bytes"] == c.probe.reservation_bytes
    assert c.lock.locked() and c.probe._execution_held
    assert c.probe.snapshot()["target_execution_lease_retained"]
    assert c.probe.snapshot()["quarantined"]
    assert any(ref() is not None for ref in c.tensors)
    with pytest.raises(PredictionConfigError, match="quarantined"), c.probe.branch():
        pass


def test_busy_target_does_not_charge_or_allocate(monkeypatch):
    c = environment(monkeypatch)
    c.lock.acquire()
    with pytest.raises(PredictionConfigError, match="busy"), c.probe.branch():
        pass
    assert not c.events and c.budget.snapshot()["used_staging_bytes"] == 0
    c.lock.release()


def test_capacity_refusal_returns_target_lock(monkeypatch):
    c = environment(monkeypatch)
    c.probe.budget = TransferBudget(1, 1)
    with pytest.raises(TransferCapacityError), c.probe.branch():
        pass
    assert not c.events and not c.lock.locked()


def test_missed_window_uses_actual_prefix_without_draft_tokens(monkeypatch):
    c = environment(monkeypatch)
    with c.probe.branch():
        result = c.probe.capture_committed(c.prefix, (2,))
        assert all(q.positions == (2,) for q in result)
        assert all(q.prefix_version == "version" for q in result)
    assert c.budget.snapshot()["used_staging_bytes"] == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_private_pool_dtype_follows_validated_target_weights(monkeypatch, dtype):
    c = environment(monkeypatch, dtype=dtype)
    with c.probe.branch():
        c.probe.capture(c.prefix, c.prediction)
    assert c.probe.dtype == dtype


@pytest.mark.parametrize("device", ["cpu", "cuda", "meta"])
def test_no_guessed_device_or_cpu_fallback(device):
    with pytest.raises(PredictionConfigError, match="indexed CUDA"):
        CUDALlamaTargetProbe(None, None, device=device, execution_lock=threading.Lock())


def test_real_smoke_reports_blocked_without_cuda(monkeypatch, capsys):
    import run_pvd_cuda_probe_smoke

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert run_pvd_cuda_probe_smoke.main([]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked"
    assert not report["production_gpu_rdma_validated"]
    assert "evidence" not in report


def test_real_smoke_accepts_bounded_long_context_without_cuda(
    monkeypatch, capsys, tmp_path
):
    import run_pvd_cuda_probe_smoke

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert (
        run_pvd_cuda_probe_smoke.main(
            [
                "--model-path",
                str(checkpoint),
                "--context-length",
                "9216",
                "--max-total-tokens",
                "9216",
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["reason"] == (
        "requires a serving-capable Linux CUDA environment"
    )
    with pytest.raises(SystemExit):
        run_pvd_cuda_probe_smoke.main(
            ["--context-length", "20481", "--max-total-tokens", "20481"]
        )
