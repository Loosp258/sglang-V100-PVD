"""Strict dual-model CUDA gate for a real Qwen2 target and independent draft.

This reuses the standalone target ModelRunner launcher, then loads a separately
configured Qwen2 draft ModelRunner with private request/KV pools on the same
GPU. It runs one two-token ``SGLangDraftProvider`` + ``CUDAPredictionPipeline``
branch and checks vocabulary compatibility, target-state canaries, pool
ownership and branch accounting. It is not production Scheduler activation,
RDMA/search/delivery validation, a latency claim, or a memory bound.

Both checkpoint paths and all memory budgets are explicit command-line inputs;
the script does not select or download model weights.
"""

import argparse
import copy
import os
import sys
import threading
from contextlib import contextmanager
from types import SimpleNamespace


def _runner_canaries(runner):
    """Bounded canaries for the resident target's model and mutable pools."""

    model_tensors = [
        *runner.model.parameters(),
        *runner.model.buffers(),
    ]
    weights = [
        (tensor, tensor.detach().reshape(-1)[:32].cpu().clone())
        for tensor in model_tensors
    ]
    kv = [
        (tensor, tensor[:1].detach().cpu().clone())
        for tensor in (
            runner.token_to_kv_pool.k_buffer + runner.token_to_kv_pool.v_buffer
        )
    ]
    req_map = runner.req_to_token_pool.req_to_token.detach().cpu().clone()
    free_pages = runner.token_to_kv_pool_allocator.free_pages.detach().cpu().clone()
    free_slots = tuple(runner.req_to_token_pool.free_slots)
    available_kv = runner.token_to_kv_pool_allocator.available_size()
    return {
        "model": weights,
        "kv": kv,
        "req_map": req_map,
        "free_pages": free_pages,
        "free_slots": free_slots,
        "available_kv": available_kv,
    }


def _assert_runner_unchanged(runner, before):
    import torch

    for current, expected in before["model"]:
        torch.testing.assert_close(
            current.detach().reshape(-1)[: expected.numel()].cpu(),
            expected,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
    for current, expected in before["kv"]:
        # The target pool has not been used yet, so its rows are intentionally
        # not initialized. Preserve and compare the snapshot without treating
        # unchanged NaN payloads as a mutation.
        torch.testing.assert_close(
            current[:1].cpu(), expected, rtol=0, atol=0, equal_nan=True
        )
    torch.testing.assert_close(
        runner.req_to_token_pool.req_to_token.cpu(),
        before["req_map"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        runner.token_to_kv_pool_allocator.free_pages.cpu(),
        before["free_pages"],
        rtol=0,
        atol=0,
    )
    assert tuple(runner.req_to_token_pool.free_slots) == before["free_slots"]
    assert runner.token_to_kv_pool_allocator.available_size() == before["available_kv"]


def _worker_view(runner):
    return SimpleNamespace(
        get_memory_pool=lambda: (
            runner.req_to_token_pool,
            runner.token_to_kv_pool_allocator,
        ),
        model_config=runner.model_config,
        device=runner.device,
    )


def _assert_rlock_released(lock):
    """Check an RLock from another thread (RLock is reentrant locally)."""
    acquired = threading.Event()

    def probe_lock():
        if lock.acquire(blocking=False):
            acquired.set()
            lock.release()

    thread = threading.Thread(target=probe_lock, name="pvd-rlock-check")
    thread.start()
    thread.join(timeout=5)
    if thread.is_alive() or not acquired.is_set():
        raise AssertionError("shared target execution RLock was not released")


def validate_dual_model(
    target_runner,
    *,
    checkpoint,
    draft_model_path,
    draft_revision,
    draft_mem_fraction_static,
    draft_scratch_budget_bytes,
    draft_persistent_budget_bytes,
    draft_transient_bytes_bound,
    probe_budget_bytes,
    probe_transient_bytes_bound,
    prefix_text,
):
    if not checkpoint:
        raise ValueError("a real target Qwen2 checkpoint is required")

    import torch
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.disaggregation.pvd.cuda_probe_search import (
        CUDAPredictionPipeline,
    )
    from sglang.srt.disaggregation.pvd.cuda_target_probe import (
        CUDAQwen2TargetProbe,
    )
    from sglang.srt.disaggregation.pvd.draft_forward_adapter import (
        DraftForwardAdapter,
        PrivatePoolAllocator,
    )
    from sglang.srt.disaggregation.pvd.draft_hf import VocabularySignature
    from sglang.srt.disaggregation.pvd.draft_memory import (
        measure_draft_retained_tensors,
    )
    from sglang.srt.disaggregation.pvd.draft_runner_sglang import (
        SGLangDraftRunnerFactory,
    )
    from sglang.srt.disaggregation.pvd.draft_sglang import (
        DraftCapabilities,
        DraftPlacement,
        SGLangDraftProvider,
    )
    from sglang.srt.disaggregation.pvd.prediction import (
        CommittedPrefix,
        DraftConfig,
        ProbeConfig,
    )
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import (
        get_global_server_args,
        set_global_server_args_for_scheduler,
    )
    from transformers import AutoTokenizer

    if type(target_runner.model).__name__ != "Qwen2ForCausalLM":
        raise ValueError("target checkpoint must load as Qwen2ForCausalLM")
    if target_runner.tp_size != 1 or target_runner.pp_size != 1:
        raise ValueError("target runner must be TP1/PP1")
    if target_runner.server_args.attention_backend != "torch_native":
        raise ValueError("target runner must use torch_native attention")

    target_args = target_runner.server_args
    device = torch.device("cuda:0")
    target_path = os.path.realpath(target_args.model_path)
    draft_path = os.path.realpath(draft_model_path)
    if target_path == draft_path:
        raise ValueError("target and draft must be separate checkpoint directories")

    # Fail vocabulary mismatch before allocating a second model on the GPU.
    target_tokenizer_path = target_args.tokenizer_path or target_path
    target_tokenizer = AutoTokenizer.from_pretrained(
        target_tokenizer_path,
        revision=target_args.revision,
        local_files_only=True,
    )
    draft_tokenizer = AutoTokenizer.from_pretrained(
        draft_path,
        revision=draft_revision,
        local_files_only=True,
    )
    target_vocabulary = VocabularySignature.from_tokenizer(target_tokenizer)
    draft_vocabulary = VocabularySignature.from_tokenizer(draft_tokenizer)
    if draft_vocabulary != target_vocabulary:
        raise ValueError(
            "draft and target VocabularySignature differ: "
            "compare tokenizer mappings, special IDs and probe fingerprints"
        )
    if not target_vocabulary.exact_mapping_available:
        raise ValueError("real-model gate requires an exact tokenizer ID mapping")

    prefix_tokens = tuple(
        int(token)
        for token in target_tokenizer.encode(prefix_text, add_special_tokens=True)
    )
    if not prefix_tokens:
        raise ValueError("--prefix-text must encode to at least one token")
    if any(not target_vocabulary.contains(token) for token in prefix_tokens):
        raise ValueError("encoded prefix contains an undeclared tokenizer ID")
    if len(prefix_tokens) + 2 > target_runner.model_config.context_len:
        raise ValueError("prefix plus two predictions exceeds target context length")
    if len(prefix_tokens) + 2 > target_runner.server_args.max_total_tokens:
        raise ValueError("prefix plus two predictions exceeds target KV capacity")

    target_parameter_count = sum(
        parameter.numel() for parameter in target_runner.model.parameters()
    )
    if not 6_000_000_000 <= target_parameter_count <= 10_000_000_000:
        raise ValueError(
            "target checkpoint must be 7B-class (6B..10B parameters); "
            f"loaded {target_parameter_count} parameters"
        )
    architecture = "Qwen2ForCausalLM"
    placement = DraftPlacement(
        gpu_id=0,
        tp_rank=0,
        scratch_budget_bytes=draft_scratch_budget_bytes,
        persistent_budget_bytes=draft_persistent_budget_bytes,
        max_concurrent_branches=1,
    )
    draft_source_args = copy.deepcopy(target_args)
    draft_source_args.pvd_draft_model_path = draft_path
    draft_source_args.pvd_draft_revision = draft_revision
    draft_source_args.pvd_draft_device = str(device)
    draft_source_args.pvd_draft_mem_fraction_static = draft_mem_fraction_static
    draft_source_args.pvd_draft_scratch_budget_bytes = draft_scratch_budget_bytes
    draft_source_args.pvd_draft_persistent_budget_bytes = draft_persistent_budget_bytes
    from sglang.srt.disaggregation.pvd.draft_sglang import build_draft_server_args

    draft_args = build_draft_server_args(draft_source_args, placement)
    draft_nccl_port = getattr(target_runner, "dist_port", None)
    if type(draft_nccl_port) is not int or draft_nccl_port <= 0:
        raise ValueError("target ModelRunner does not expose its distributed port")
    global_args = get_global_server_args()
    target_before = _runner_canaries(target_runner)
    target_cpu_rng_before_draft_load = torch.get_rng_state().clone()
    target_cuda_rng_before_draft_load = torch.cuda.get_rng_state(device).clone()
    torch.cuda.synchronize(device)
    target_allocated_before_draft = torch.cuda.memory_allocated(device)
    target_reserved_before_draft = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        # Runner initialization may consume RNG while constructing non-weight
        # state. Keep that separate from the measured prediction branch.
        with torch.random.fork_rng(devices=[device.index], enabled=True):
            draft_model_config = ModelConfig.from_server_args(draft_args)
            draft_runner = ModelRunner(
                draft_model_config,
                mem_fraction_static=draft_args.mem_fraction_static,
                gpu_id=0,
                tp_rank=0,
                tp_size=1,
                moe_ep_rank=0,
                moe_ep_size=1,
                pp_rank=0,
                pp_size=1,
                nccl_port=draft_nccl_port,
                server_args=draft_args,
                is_draft_worker=True,
            )
    finally:
        set_global_server_args_for_scheduler(global_args)
    torch.cuda.synchronize(device)
    if not torch.equal(
        target_cpu_rng_before_draft_load, torch.get_rng_state()
    ) or not torch.equal(
        target_cuda_rng_before_draft_load, torch.cuda.get_rng_state(device)
    ):
        raise AssertionError(
            "draft ModelRunner initialization changed target RNG state"
        )
    _assert_runner_unchanged(target_runner, target_before)
    if type(draft_runner.model).__name__ != architecture:
        raise ValueError("draft checkpoint must load as Qwen2ForCausalLM")
    if draft_runner.tp_size != 1 or draft_runner.pp_size != 1:
        raise ValueError("draft runner must be TP1/PP1")
    if draft_args.attention_backend != "torch_native":
        raise ValueError("draft runner must use torch_native attention")
    if any(
        parameter.device != device
        or parameter.dtype not in (torch.float16, torch.float32)
        for parameter in draft_runner.model.parameters()
    ):
        raise ValueError("draft parameters must be FP16/FP32 on cuda:0")
    draft_vocab_size = int(draft_runner.model.config.vocab_size)
    target_vocab_size = int(target_runner.model.config.vocab_size)
    required_vocab_size = max(target_vocabulary.allowed_ids) + 1
    if draft_vocab_size < required_vocab_size:
        raise ValueError(
            "draft model output vocabulary is smaller than its tokenizer: "
            f"{draft_vocab_size} < {required_vocab_size}"
        )
    if target_vocab_size < required_vocab_size:
        raise ValueError(
            "target model output vocabulary is smaller than its tokenizer: "
            f"{target_vocab_size} < {required_vocab_size}"
        )
    draft_parameter_count = sum(
        parameter.numel() for parameter in draft_runner.model.parameters()
    )
    if draft_parameter_count >= target_parameter_count:
        raise ValueError("independent draft checkpoint must be smaller than the target")

    target_identity = (
        f"qwen-target:{target_path}@{target_args.revision or 'local-unknown'}"
    )
    probe_config = ProbeConfig(
        target_identity,
        tuple(range(target_runner.model.config.num_hidden_layers)),
        head_start=0,
        head_count=target_runner.model.config.num_attention_heads,
    )
    execution_lock = threading.RLock()
    probe_budget = TransferBudget(probe_budget_bytes, 1)
    probe = CUDAQwen2TargetProbe(
        target_runner,
        probe_config,
        device=device,
        execution_lock=execution_lock,
        target_model_id=target_identity,
        max_tokens=len(prefix_tokens) + 2,
        max_predict_tokens=2,
        transient_bytes_bound=probe_transient_bytes_bound,
        budget=probe_budget,
        vocabulary=target_vocabulary,
    )

    kv_pool = draft_runner.token_to_kv_pool
    bytes_per_token = sum(
        tensor[0].numel() * tensor.element_size()
        for tensor in kv_pool.k_buffer + kv_pool.v_buffer
    )
    retained = measure_draft_retained_tensors(draft_runner)
    draft_adapter = DraftForwardAdapter(
        draft_runner,
        architecture=architecture,
        attention_backend="torch_native",
        bytes_per_token=bytes_per_token,
        device=device,
        transient_bytes_bound=draft_transient_bytes_bound,
    )
    allocator = PrivatePoolAllocator(
        draft_runner.req_to_token_pool,
        draft_runner.token_to_kv_pool_allocator,
    )
    factory = SGLangDraftRunnerFactory(
        draft_adapter,
        allocator,
        capabilities=DraftCapabilities(
            architectures=(architecture,),
            attention_backends=("torch_native",),
            max_prefix_tokens=len(prefix_tokens),
            max_predict_tokens=2,
        ),
        persistent_bytes=retained.total_bytes,
        max_tokens=2,
    )
    scratch_budget = TransferBudget(draft_scratch_budget_bytes, 1)
    persistent_budget = TransferBudget(draft_persistent_budget_bytes, 1)

    class ObservedDraftProvider(SGLangDraftProvider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_prediction = None
            self.max_scratch_used_during_branch = 0

        @contextmanager
        def branch(self):
            with super().branch():
                current = self.scratch_budget.snapshot()["used_staging_bytes"]
                self.max_scratch_used_during_branch = max(
                    self.max_scratch_used_during_branch, current
                )
                yield self

        def predict(self, prefix, max_tokens):
            self.last_prediction = super().predict(prefix, max_tokens)
            return self.last_prediction

    provider = ObservedDraftProvider(
        DraftConfig(
            model_name_or_path=draft_path,
            revision=draft_revision,
            device=str(device),
            dtype=str(next(draft_runner.model.parameters()).dtype).removeprefix(
                "torch."
            ),
            predict_tokens=2,
        ),
        placement,
        factory,
        worker=_worker_view(draft_runner),
        target_worker=_worker_view(target_runner),
        draft_vocabulary=draft_vocabulary,
        target_vocabulary=target_vocabulary,
        scratch_budget=scratch_budget,
        persistent_budget=persistent_budget,
    )
    if provider.pool_ownership is None or not provider.pool_ownership.storage_verified:
        raise AssertionError("draft pool storage was not positively verified private")
    if (
        provider.persistent_budget.snapshot()["used_staging_bytes"]
        != retained.total_bytes
    ):
        raise AssertionError(
            "draft persistent tensor charge differs from the measured floor"
        )

    pipeline = CUDAPredictionPipeline(
        provider,
        probe,
        provider.config,
        probe_config,
        execution_lock=execution_lock,
    )
    prefix = CommittedPrefix(
        "pvd-dual-model-smoke",
        prefix_tokens,
        committed_position=0,
        version="dual-model-prefix-v1",
    )
    draft_pool_capacity_before = (
        len(draft_runner.req_to_token_pool.free_slots),
        draft_runner.token_to_kv_pool_allocator.available_size(),
    )
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state(device).clone()
    target_allocated_with_draft = torch.cuda.memory_allocated(device)
    target_reserved_with_draft = torch.cuda.memory_reserved(device)

    query_evidence = []
    with pipeline.query_branch(prefix) as queries:
        prediction = provider.last_prediction
        if prediction is None or len(prediction.tokens) != 2:
            raise AssertionError("draft provider did not produce exactly two tokens")
        if any(not target_vocabulary.contains(token) for token in prediction.tokens):
            raise AssertionError(
                "draft emitted a padded/out-of-range id outside the shared tokenizer vocabulary"
            )
        if len(queries) != target_runner.model.config.num_hidden_layers:
            raise AssertionError("target Q was not captured for every model layer")
        for query in queries:
            expected_positions = tuple(
                range(len(prefix_tokens), len(prefix_tokens) + 2)
            )
            if (
                query.vector_space != target_identity
                or query.positional_encoding != "rope_applied"
                or query.positions != expected_positions
                or query.valid_length != 2
                or query.head_start != 0
                or query.head_count != target_runner.model.config.num_attention_heads
                or query.vectors.shape
                != (
                    2,
                    target_runner.model.config.num_attention_heads,
                    target_runner.model_config.head_dim,
                )
                or not torch.isfinite(query.vectors).all()
            ):
                raise AssertionError("target Q metadata or values are incompatible")
            query_evidence.append(
                {
                    "layer": query.layer,
                    "positions": query.positions,
                    "shape": tuple(query.vectors.shape),
                    "positional_encoding": query.positional_encoding,
                }
            )
        prediction_tokens = tuple(prediction.tokens)
        del queries

    torch.cuda.synchronize(device)
    target_cpu_rng_after = torch.get_rng_state()
    target_cuda_rng_after = torch.cuda.get_rng_state(device)
    if not torch.equal(cpu_rng_before, target_cpu_rng_after) or not torch.equal(
        cuda_rng_before, target_cuda_rng_after
    ):
        raise AssertionError("prediction branch changed CPU or CUDA RNG state")
    _assert_runner_unchanged(target_runner, target_before)
    draft_pool_capacity_after = (
        len(draft_runner.req_to_token_pool.free_slots),
        draft_runner.token_to_kv_pool_allocator.available_size(),
    )
    if draft_pool_capacity_after != draft_pool_capacity_before:
        raise AssertionError("draft request/KV pool capacity was not refunded")
    if provider.active_branches != 0 or provider.degraded:
        raise AssertionError("draft provider retained an active or degraded branch")
    if scratch_budget.snapshot()["used_staging_bytes"] != 0:
        raise AssertionError("draft branch scratch budget was not refunded")
    if probe_budget.snapshot()["used_staging_bytes"] != 0:
        raise AssertionError("target probe budget was not refunded")
    if provider.max_scratch_used_during_branch <= 0:
        raise AssertionError("draft branch did not reserve scratch before execution")
    if (
        provider.persistent_budget.snapshot()["used_staging_bytes"]
        != retained.total_bytes
    ):
        raise AssertionError("draft persistent charge changed during branch retirement")
    _assert_rlock_released(execution_lock)

    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "target_model_path": target_path,
        "target_revision": target_args.revision or "local-unknown",
        "draft_model_path": draft_path,
        "draft_revision": draft_revision or "local-unknown",
        "target_architecture": type(target_runner.model).__name__,
        "draft_architecture": type(draft_runner.model).__name__,
        "target_parameter_count": target_parameter_count,
        "draft_parameter_count": draft_parameter_count,
        "model_vocab_sizes": {
            "target_embedding": target_vocab_size,
            "draft_embedding": draft_vocab_size,
            "shared_tokenizer": target_vocabulary.size,
        },
        "target_and_draft_resident_concurrently": True,
        "vocabulary_signature": {
            "exact_match": target_vocabulary == draft_vocabulary,
            "size": target_vocabulary.size,
            "bos_token_id": target_vocabulary.bos_token_id,
            "eos_token_id": target_vocabulary.eos_token_id,
            "fingerprint": target_vocabulary.fingerprint,
            "mapping_fingerprint": target_vocabulary.mapping_fingerprint,
            "valid_token_id_count": len(target_vocabulary.allowed_ids),
        },
        "draft_prediction_tokens": prediction_tokens,
        "draft_forward_count": draft_adapter.forward_count,
        "target_q_layers": len(query_evidence),
        "target_q_positions": query_evidence[0]["positions"],
        "target_q_shape_per_layer": query_evidence[0]["shape"],
        "target_q_post_rope": all(
            item["positional_encoding"] == "rope_applied" for item in query_evidence
        ),
        "target_model_state_canaries_unchanged": True,
        "cpu_cuda_rng_unchanged": True,
        "draft_pool_storage_verified_private": provider.pool_ownership.storage_verified,
        "draft_pool_capacity_before_after": {
            "before": draft_pool_capacity_before,
            "after": draft_pool_capacity_after,
        },
        "draft_branch_scratch_reserved_bytes": provider.max_scratch_used_during_branch,
        "draft_branch_scratch_refunded": True,
        "draft_persistent_known_tensor_bytes": retained.total_bytes,
        "draft_persistent_budget_used_after_branch": provider.persistent_budget.snapshot()[
            "used_staging_bytes"
        ],
        "target_probe_budget_refunded": True,
        "target_allocated_bytes_before_draft": target_allocated_before_draft,
        "target_reserved_bytes_before_draft": target_reserved_before_draft,
        "combined_allocated_bytes_after_draft_load": target_allocated_with_draft,
        "combined_reserved_bytes_after_draft_load": target_reserved_with_draft,
        "combined_peak_allocated_bytes": peak_allocated,
        "combined_peak_reserved_bytes": peak_reserved,
        "cuda_peak_is_hard_memory_bound": False,
        "production_scheduler_activated": False,
        "rdma_validated": False,
        "v_search_or_sparse_delivery_validated": False,
        "latency_or_compute_network_overlap_validated": False,
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", required=True, help="Local target Qwen2 checkpoint"
    )
    parser.add_argument("--architecture", choices=("qwen2",), default="qwen2")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--max-total-tokens", type=int, default=128)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--draft-revision", default=None)
    parser.add_argument("--draft-mem-fraction-static", type=float, required=True)
    parser.add_argument("--draft-scratch-budget-bytes", type=int, required=True)
    parser.add_argument("--draft-persistent-budget-bytes", type=int, required=True)
    parser.add_argument("--draft-transient-bytes-bound", type=int, required=True)
    parser.add_argument("--probe-budget-bytes", type=int, required=True)
    parser.add_argument("--probe-transient-bytes-bound", type=int, required=True)
    parser.add_argument(
        "--prefix-text", default="The quick brown fox jumps over the lazy dog"
    )
    args, target_passthrough = parser.parse_known_args(argv)

    for label, model_path in (
        ("target", args.model_path),
        ("draft", args.draft_model_path),
    ):
        if not os.path.isabs(model_path) or not os.path.isfile(
            os.path.join(model_path, "config.json")
        ):
            flag = "--model-path" if label == "target" else "--draft-model-path"
            parser.error(f"{flag} must be an absolute local checkpoint directory")
    if os.path.realpath(args.model_path) == os.path.realpath(args.draft_model_path):
        parser.error("target and draft model paths must be different")
    if not 0 < args.draft_mem_fraction_static < 1:
        parser.error("--draft-mem-fraction-static must be between 0 and 1")
    for name in (
        "draft_scratch_budget_bytes",
        "draft_persistent_budget_bytes",
        "draft_transient_bytes_bound",
        "probe_budget_bytes",
        "probe_transient_bytes_bound",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    target_argv = [
        "--model-path",
        args.model_path,
        "--architecture",
        args.architecture,
        "--dtype",
        args.dtype,
        "--context-length",
        str(args.context_length),
        "--max-total-tokens",
        str(args.max_total_tokens),
        *target_passthrough,
    ]

    def validator(target_runner, *, checkpoint=False):
        return validate_dual_model(
            target_runner,
            checkpoint=checkpoint,
            draft_model_path=args.draft_model_path,
            draft_revision=args.draft_revision,
            draft_mem_fraction_static=args.draft_mem_fraction_static,
            draft_scratch_budget_bytes=args.draft_scratch_budget_bytes,
            draft_persistent_budget_bytes=args.draft_persistent_budget_bytes,
            draft_transient_bytes_bound=args.draft_transient_bytes_bound,
            probe_budget_bytes=args.probe_budget_bytes,
            probe_transient_bytes_bound=args.probe_transient_bytes_bound,
            prefix_text=args.prefix_text,
        )

    from run_pvd_cuda_probe_smoke import main as run

    return run(
        target_argv,
        validator=validator,
        schema="pvd-qwen-dual-model-draft-probe-v1",
    )


if __name__ == "__main__":
    raise SystemExit(main())
