"""CPU fault injection for the opt-in CUDA draft startup transaction."""

from __future__ import annotations

import threading
from contextlib import contextmanager, nullcontext
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd import cuda_prediction_startup as startup
from sglang.srt.disaggregation.pvd.draft_sglang import DraftPlacement
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


class Qwen2ForCausalLM:
    def __init__(self):
        self.config = SimpleNamespace(
            vocab_size=2, num_hidden_layers=2, num_attention_heads=2
        )

    def parameters(self):
        return (SimpleNamespace(numel=lambda: 100),)


class _Tokenizer:
    vocab_size = 2
    bos_token_id = 0
    eos_token_id = 1
    all_special_ids = (0, 1)

    def encode(self, text):
        return [0]

    def get_vocab(self):
        return {"a": 0, "b": 1}


def _target():
    args = SimpleNamespace(
        model_path="/target",
        tokenizer_path=None,
        revision=None,
        attention_backend="torch_native",
        speculative_algorithm=None,
        max_total_tokens=32,
        pvd_draft_model_path=None,
        pvd_draft_mem_fraction_static=None,
        pvd_draft_device=None,
        pvd_draft_revision=None,
        device="cuda",
        disaggregation_mode="decode",
        pvd_predictive_retrieval_config=True,
        pvd_retrieval_top_k=4,
    )
    return SimpleNamespace(
        model=Qwen2ForCausalLM(),
        server_args=args,
        model_config=SimpleNamespace(context_len=32, head_dim=4),
        gpu_id=0,
        tp_size=1,
        pp_size=1,
        dist_port=12345,
        device="cuda",
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=object(),
    )


def _options(target, lock):
    return {
        "draft_model_path": "/draft",
        "draft_revision": "test-revision",
        "draft_mem_fraction_static": 0.1,
        "target_model_id": "qwen-test",
        "placement": DraftPlacement(
            gpu_id=0,
            tp_rank=0,
            scratch_budget_bytes=1024,
            persistent_budget_bytes=1024,
            max_concurrent_branches=1,
        ),
        "execution_lock": lock,
        "max_prefix_tokens": 8,
        "predict_tokens": 2,
        "draft_transient_bytes_bound": 16,
        "probe_transient_bytes_bound": 16,
        "target_scratch_budget": TransferBudget(1024, 1),
        "target_tokenizer": _Tokenizer(),
        "draft_tokenizer": _Tokenizer(),
    }


def _fake_cuda(monkeypatch):
    # server_args imports FLA code that asks torch.cuda for device properties.
    # Import it before replacing torch.cuda.device with the context-manager
    # double; PyTorch itself uses that name as an isinstance type.
    import_module("sglang.srt.server_args")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    cuda_state = [torch.tensor([1], dtype=torch.uint8)]
    monkeypatch.setattr(
        torch.cuda, "get_rng_state", lambda _device: cuda_state[0].clone()
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda value, _device: cuda_state.__setitem__(0, value.clone()),
    )

    @contextmanager
    def fork_rng(*, devices, enabled):
        cpu_before = torch.get_rng_state().clone()
        cuda_before = cuda_state[0].clone()
        try:
            yield
        finally:
            torch.set_rng_state(cpu_before)
            cuda_state[0] = cuda_before

    monkeypatch.setattr(torch.random, "fork_rng", fork_rng)


def _fake_global_args(monkeypatch, target):
    from sglang.srt import server_args

    current = [target.server_args]
    monkeypatch.setattr(server_args, "get_global_server_args", lambda: current[0])
    monkeypatch.setattr(
        server_args,
        "set_global_server_args_for_scheduler",
        lambda args: current.__setitem__(0, args),
    )
    return current


def _lock_is_free(lock):
    free = []

    def check():
        acquired = lock.acquire(blocking=False)
        free.append(acquired)
        if acquired:
            lock.release()

    thread = threading.Thread(target=check)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert free == [True]


def test_refuses_unsupported_budget_before_loading(monkeypatch):
    _fake_cuda(monkeypatch)
    target = _target()
    options = _options(target, threading.RLock())
    options["placement"] = DraftPlacement(
        gpu_id=0,
        tp_rank=0,
        scratch_budget_bytes=1024,
        persistent_budget_bytes=0,
        max_concurrent_branches=1,
    )
    with pytest.raises(ValueError, match="persistent budget"):
        startup.build_cuda_prediction_startup(target, **options)


def test_partial_draft_load_is_retained_and_state_restored(monkeypatch):
    _fake_cuda(monkeypatch)
    target = _target()
    global_args = _fake_global_args(monkeypatch, target)
    lock = threading.RLock()
    partial = object()
    before = torch.get_rng_state().clone()

    def loader(args, _target, retain):
        retain.append(partial)
        global_args[0] = args
        torch.manual_seed(123)
        raise RuntimeError("partial GPU allocation")

    with pytest.raises(RuntimeError, match="partial GPU allocation"):
        startup.build_cuda_prediction_startup(
            target, **_options(target, lock), runner_loader=loader
        )
    assert global_args[0] is target.server_args
    assert torch.equal(before, torch.get_rng_state())
    assert any(partial in record for record in startup._STARTUP_QUARANTINE)
    _lock_is_free(lock)


def test_occupied_target_lock_refuses_load(monkeypatch):
    _fake_cuda(monkeypatch)
    target = _target()
    _fake_global_args(monkeypatch, target)
    lock = threading.RLock()
    ready = threading.Event()
    release = threading.Event()

    def hold():
        with lock:
            ready.set()
            release.wait(timeout=2)

    thread = threading.Thread(target=hold)
    thread.start()
    ready.wait(timeout=2)
    try:
        with pytest.raises(ValueError, match="target execution is busy"):
            startup.build_cuda_prediction_startup(
                target,
                **_options(target, lock),
                runner_loader=lambda *_args: pytest.fail("loader reached"),
            )
    finally:
        release.set()
        thread.join(timeout=2)


def test_composes_one_private_draft_and_target_probe(monkeypatch):
    _fake_cuda(monkeypatch)
    target = _target()
    global_args = _fake_global_args(monkeypatch, target)
    lock = threading.RLock()
    loaded = []

    class FakeParameter:
        device = torch.device("cuda:0")
        dtype = torch.float16

        def numel(self):
            return 10

    class FakeBuffer:
        def __getitem__(self, index):
            assert index == 0
            return FakeParameter()

        def element_size(self):
            return 2

    def loader(args, _target, retain):
        assert args.model_path == "/draft"
        assert args.device == "cuda"
        assert args.speculative_algorithm is None
        assert args.disaggregation_mode == "null"
        assert args.pvd_predictive_retrieval_config is False
        assert args.pvd_retrieval_top_k is None
        global_args[0] = args
        model = Qwen2ForCausalLM()
        model.parameters = lambda: iter((FakeParameter(),))
        runner = SimpleNamespace(
            model=model,
            tp_size=1,
            pp_size=1,
            device="cuda",
            model_config=SimpleNamespace(),
            token_to_kv_pool=SimpleNamespace(
                k_buffer=[FakeBuffer()], v_buffer=[FakeBuffer()]
            ),
            req_to_token_pool=object(),
            token_to_kv_pool_allocator=object(),
        )
        retain.append(runner)
        loaded.append(runner)
        return runner

    class FakeProvider:
        def __init__(self, config, placement, factory, **kwargs):
            self.config = config
            self.placement = placement
            self.factory = factory
            self.pool_ownership = SimpleNamespace(storage_verified=True)
            assert kwargs["worker"].get_memory_pool()[0] is loaded[-1].req_to_token_pool
            assert (
                kwargs["target_worker"].get_memory_pool()[0] is target.req_to_token_pool
            )

    class FakeProbe:
        def __init__(self, runner, config, **kwargs):
            assert runner is target
            assert kwargs["execution_lock"] is lock
            assert kwargs["target_model_id"] == config.target_model_id
            assert kwargs["budget"] is options["target_scratch_budget"]
            self.config = config
            self.reservation_bytes = 100

    class FakePipeline:
        def __init__(self, provider, probe, draft_config, probe_config, **kwargs):
            assert provider.config is draft_config
            assert probe.config is probe_config
            assert kwargs["execution_lock"] is lock

    monkeypatch.setattr(
        startup,
        "measure_draft_retained_tensors",
        lambda _r: SimpleNamespace(total_bytes=100),
    )
    monkeypatch.setattr(
        startup, "DraftForwardAdapter", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(startup, "PrivatePoolAllocator", lambda *_args: object())
    monkeypatch.setattr(
        startup, "SGLangDraftRunnerFactory", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(startup, "SGLangDraftProvider", FakeProvider)
    monkeypatch.setattr(startup, "CUDAQwen2TargetProbe", FakeProbe)
    monkeypatch.setattr(startup, "CUDAPredictionPipeline", FakePipeline)

    options = _options(target, lock)
    result = startup.build_cuda_prediction_startup(
        target, **options, runner_loader=loader
    )
    assert result.draft_runner is loaded[0]
    assert result.draft_retained_bytes == 100
    assert result.target_model_id == "qwen-test"
    assert result.target_scratch_budget is options["target_scratch_budget"]
    assert global_args[0] is target.server_args
    assert target.server_args.pvd_predictive_retrieval_config is True
    _lock_is_free(lock)

    options["target_scratch_budget"] = TransferBudget(120, 1)
    with pytest.raises(ValueError, match="probe and query copy"):
        startup.build_cuda_prediction_startup(target, **options, runner_loader=loader)
    assert global_args[0] is target.server_args
    _lock_is_free(lock)
