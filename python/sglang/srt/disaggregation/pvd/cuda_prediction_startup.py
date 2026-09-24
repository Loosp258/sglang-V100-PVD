"""Opt-in construction of a private draft and target-Q prediction pipeline.

This module does not register a Scheduler hook. The caller must start it on the
target worker's main thread before serving requests, and must pass the same
RLock used by committed target forwards. A failed CUDA load has no reliable
general-purpose unload operation; its partially built runner is retained in a
process-lifetime quarantine instead of being silently reused or freed.
"""

from __future__ import annotations

import copy
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from sglang.srt.disaggregation.pvd.cuda_probe_search import CUDAPredictionPipeline
from sglang.srt.disaggregation.pvd.cuda_target_probe import (
    CUDALlamaTargetProbe,
    CUDAQwen2TargetProbe,
)
from sglang.srt.disaggregation.pvd.draft_forward_adapter import (
    DraftForwardAdapter,
    PrivatePoolAllocator,
)
from sglang.srt.disaggregation.pvd.draft_hf import VocabularySignature
from sglang.srt.disaggregation.pvd.draft_memory import measure_draft_retained_tensors
from sglang.srt.disaggregation.pvd.draft_runner_sglang import SGLangDraftRunnerFactory
from sglang.srt.disaggregation.pvd.draft_sglang import (
    DraftCapabilities,
    DraftPlacement,
    SGLangDraftProvider,
    build_draft_server_args,
)
from sglang.srt.disaggregation.pvd.prediction import (
    DraftConfig,
    PredictionConfigError,
    ProbeConfig,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

_RLOCK_TYPE = type(threading.RLock())
_STARTUP_QUARANTINE: list[tuple[Any, ...]] = []


@dataclass(frozen=True)
class CUDAPredictionStartup:
    """Strong ownership of the loaded draft and its charged private pools."""

    pipeline: CUDAPredictionPipeline
    draft_runner: Any
    draft_args: Any
    vocabulary: VocabularySignature
    target_model_id: str
    draft_retained_bytes: int
    target_scratch_budget: TransferBudget


def _worker_view(runner: Any) -> SimpleNamespace:
    return SimpleNamespace(
        get_memory_pool=lambda: (
            runner.req_to_token_pool,
            runner.token_to_kv_pool_allocator,
        ),
        model_config=runner.model_config,
        device=runner.device,
    )


def _load_tokenizer(path: str, revision: str | None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, revision=revision, local_files_only=True)


def _load_runner(draft_args: Any, target_runner: Any, retain: list[Any]):
    """Keep the object even if ModelRunner.__init__ raises after CUDA allocation."""
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.model_runner import ModelRunner

    model_config = ModelConfig.from_server_args(draft_args)
    runner = ModelRunner.__new__(ModelRunner)
    retain.append(runner)
    runner.__init__(
        model_config,
        mem_fraction_static=draft_args.mem_fraction_static,
        gpu_id=target_runner.gpu_id,
        tp_rank=0,
        tp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        pp_rank=0,
        pp_size=1,
        nccl_port=target_runner.dist_port,
        server_args=draft_args,
        is_draft_worker=True,
    )
    return runner


def build_cuda_prediction_startup(
    target_runner: Any,
    *,
    draft_model_path: str,
    draft_revision: str | None,
    draft_mem_fraction_static: float,
    target_model_id: str,
    placement: DraftPlacement,
    execution_lock: Any,
    max_prefix_tokens: int,
    predict_tokens: int,
    draft_transient_bytes_bound: int,
    probe_transient_bytes_bound: int,
    target_scratch_budget: TransferBudget,
    target_tokenizer: Any = None,
    draft_tokenizer: Any = None,
    tokenizer_loader: Callable[[str, str | None], Any] = _load_tokenizer,
    runner_loader: Callable[[Any, Any, list[Any]], Any] = _load_runner,
) -> CUDAPredictionStartup:
    """Load one independent TP1 draft and bind its private probe to the target.

    ``runner_loader`` is injectable for validation. An injected loader must
    append any partially allocated object to ``retain`` before it can fail.
    Returned runners and providers are never unloaded by this function.
    ``target_scratch_budget`` is the caller-owned aggregate budget. Its exact
    object is passed to the Q probe and returned to the caller for the query
    copy, Prompt importer, delivery and attention workspaces. Draft scratch
    and draft resident weights/pools use placement's separate budgets.
    """
    if not isinstance(execution_lock, _RLOCK_TYPE):
        raise PredictionConfigError("shared target execution RLock required")
    if (
        not isinstance(placement, DraftPlacement)
        or placement.max_concurrent_branches != 1
    ):
        raise PredictionConfigError("one explicit draft branch placement is required")
    if placement.persistent_budget_bytes <= 0:
        raise PredictionConfigError("positive draft persistent budget required")
    if not isinstance(target_scratch_budget, TransferBudget):
        raise PredictionConfigError("shared target scratch budget required")
    for name, value in (
        ("max_prefix_tokens", max_prefix_tokens),
        ("predict_tokens", predict_tokens),
    ):
        if type(value) is not int or value <= 0:
            raise PredictionConfigError(f"{name} must be a positive integer")
    for name, value in (
        ("draft_transient_bytes_bound", draft_transient_bytes_bound),
        ("probe_transient_bytes_bound", probe_transient_bytes_bound),
    ):
        if type(value) is not int or value < 0:
            raise PredictionConfigError(f"{name} must be a non-negative integer")
    if not target_model_id or not draft_model_path:
        raise PredictionConfigError("target identity and draft checkpoint are required")
    if not torch.cuda.is_available():
        raise PredictionConfigError("CUDA prediction requires an available GPU")
    device = torch.device(f"cuda:{placement.gpu_id}")
    if placement.gpu_id != target_runner.gpu_id or placement.tp_rank != 0:
        raise PredictionConfigError("draft and target must use the same TP1 GPU")
    if target_runner.tp_size != 1 or target_runner.pp_size != 1:
        raise PredictionConfigError("target must use TP1 and PP1")
    if target_runner.server_args.attention_backend != "torch_native":
        raise PredictionConfigError("target must use torch_native attention")
    if target_runner.server_args.speculative_algorithm is not None:
        raise PredictionConfigError("native speculative decoding is unsupported")
    if torch.device(f"cuda:{target_runner.gpu_id}") != device:
        raise PredictionConfigError("target CUDA device differs from draft placement")
    if (
        max_prefix_tokens + predict_tokens > target_runner.model_config.context_len
        or max_prefix_tokens + predict_tokens
        > target_runner.server_args.max_total_tokens
    ):
        raise PredictionConfigError("prediction exceeds target context or KV capacity")
    if type(predict_tokens) is int and predict_tokens > 16:
        raise PredictionConfigError("draft prediction exceeds supported bounded length")
    architecture = type(target_runner.model).__name__
    probes = {
        "Qwen2ForCausalLM": CUDAQwen2TargetProbe,
        "LlamaForCausalLM": CUDALlamaTargetProbe,
    }
    if architecture not in probes:
        raise PredictionConfigError("target architecture has no CUDA Q probe")
    target_args = target_runner.server_args
    if os.path.realpath(target_args.model_path) == os.path.realpath(draft_model_path):
        raise PredictionConfigError("draft and target checkpoints must differ")
    target_tokenizer = target_tokenizer or tokenizer_loader(
        target_args.tokenizer_path or target_args.model_path, target_args.revision
    )
    draft_tokenizer = draft_tokenizer or tokenizer_loader(
        draft_model_path, draft_revision
    )
    vocabulary = VocabularySignature.from_tokenizer(target_tokenizer)
    if (
        vocabulary != VocabularySignature.from_tokenizer(draft_tokenizer)
        or not vocabulary.exact_mapping_available
    ):
        raise PredictionConfigError("draft and target tokenizer mappings must match")
    if max(vocabulary.allowed_ids) >= target_runner.model.config.vocab_size:
        raise PredictionConfigError("target embedding does not cover tokenizer IDs")

    source_args = copy.deepcopy(target_args)
    source_args.pvd_draft_model_path = draft_model_path
    source_args.pvd_draft_revision = draft_revision
    source_args.pvd_draft_device = str(device)
    source_args.pvd_draft_mem_fraction_static = draft_mem_fraction_static
    draft_args = build_draft_server_args(source_args, placement)
    # ModelRunner temporarily installs its own ServerArgs globally. The draft
    # copy must not advertise the target's predictive retrieval configuration.
    for name, value in (
        ("pvd_predictive_retrieval_config", False),
        ("pvd_retrieval_vector_space", None),
        ("pvd_retrieval_metric", "ip"),
        ("pvd_retrieval_top_k", None),
        ("pvd_retrieval_max_union_tokens", None),
        ("pvd_retrieval_bank_budget_bytes", None),
        ("pvd_retrieval_scratch_budget_bytes", None),
    ):
        if hasattr(draft_args, name):
            setattr(draft_args, name, value)
    if type(target_runner.dist_port) is not int or target_runner.dist_port <= 0:
        raise PredictionConfigError("target distributed port is required")
    from sglang.srt.server_args import (
        get_global_server_args,
        set_global_server_args_for_scheduler,
    )

    global_args = get_global_server_args()
    if not execution_lock.acquire(blocking=False):
        raise PredictionConfigError("target execution is busy during draft startup")
    keep_lock = False
    retained: list[Any] = []
    try:
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone()
        try:
            with (
                torch.cuda.device(device),
                torch.random.fork_rng(devices=[device.index], enabled=True),
            ):
                draft_runner = runner_loader(draft_args, target_runner, retained)
                if not any(owner is draft_runner for owner in retained):
                    retained.append(draft_runner)
        finally:
            set_global_server_args_for_scheduler(global_args)
        torch.cuda.synchronize(device)
        if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(
            cuda_rng, torch.cuda.get_rng_state(device)
        ):
            raise PredictionConfigError("draft load changed target RNG state")
        if type(draft_runner.model).__name__ != architecture:
            raise PredictionConfigError("draft architecture differs from target")
        if draft_runner.tp_size != 1 or draft_runner.pp_size != 1:
            raise PredictionConfigError("draft must use TP1 and PP1")
        if any(
            p.device != device or p.dtype not in (torch.float16, torch.float32)
            for p in draft_runner.model.parameters()
        ):
            raise PredictionConfigError(
                "draft weights require FP16/FP32 on selected GPU"
            )
        if max(vocabulary.allowed_ids) >= draft_runner.model.config.vocab_size:
            raise PredictionConfigError("draft output cannot represent tokenizer IDs")
        if sum(p.numel() for p in draft_runner.model.parameters()) >= sum(
            p.numel() for p in target_runner.model.parameters()
        ):
            raise PredictionConfigError("prediction draft must be smaller than target")
        kv_pool = draft_runner.token_to_kv_pool
        bytes_per_token = sum(
            tensor[0].numel() * tensor.element_size()
            for tensor in kv_pool.k_buffer + kv_pool.v_buffer
        )
        retained_bytes = measure_draft_retained_tensors(draft_runner).total_bytes
        adapter = DraftForwardAdapter(
            draft_runner,
            architecture=architecture,
            attention_backend="torch_native",
            bytes_per_token=bytes_per_token,
            device=device,
            transient_bytes_bound=draft_transient_bytes_bound,
        )
        factory = SGLangDraftRunnerFactory(
            adapter,
            PrivatePoolAllocator(
                draft_runner.req_to_token_pool,
                draft_runner.token_to_kv_pool_allocator,
            ),
            capabilities=DraftCapabilities(
                architectures=(architecture,),
                attention_backends=("torch_native",),
                max_prefix_tokens=max_prefix_tokens,
                max_predict_tokens=predict_tokens,
            ),
            persistent_bytes=retained_bytes,
            max_tokens=predict_tokens,
        )
        draft_config = DraftConfig(
            model_name_or_path=draft_model_path,
            revision=draft_revision,
            device=str(device),
            dtype=str(next(draft_runner.model.parameters()).dtype).removeprefix(
                "torch."
            ),
            predict_tokens=predict_tokens,
        )
        provider = SGLangDraftProvider.__new__(SGLangDraftProvider)
        retained.append(provider)
        provider.__init__(
            draft_config,
            placement,
            factory,
            worker=_worker_view(draft_runner),
            target_worker=_worker_view(target_runner),
            draft_vocabulary=vocabulary,
            target_vocabulary=vocabulary,
            scratch_budget=TransferBudget(placement.scratch_budget_bytes, 1),
            persistent_budget=TransferBudget(placement.persistent_budget_bytes, 1),
        )
        if (
            provider.pool_ownership is None
            or not provider.pool_ownership.storage_verified
        ):
            raise PredictionConfigError("draft pool storage independence is unverified")
        probe_config = ProbeConfig(
            target_model_id,
            tuple(range(target_runner.model.config.num_hidden_layers)),
            head_start=0,
            head_count=target_runner.model.config.num_attention_heads,
        )
        probe_type = probes[architecture]
        probe = probe_type.__new__(probe_type)
        retained.append(probe)
        probe.__init__(
            target_runner,
            probe_config,
            device=device,
            execution_lock=execution_lock,
            target_model_id=target_model_id,
            max_tokens=max_prefix_tokens + predict_tokens,
            max_predict_tokens=predict_tokens,
            transient_bytes_bound=probe_transient_bytes_bound,
            budget=target_scratch_budget,
            vocabulary=vocabulary,
        )
        # A query copy overlaps the probe's retained Q. At least this known
        # pair must fit; importer/delivery/attention share the same budget and
        # may still apply further backpressure during serving.
        query_copy_bound = predict_tokens * target_runner.model_config.head_dim * 8
        if (
            probe.reservation_bytes + query_copy_bound
            > target_scratch_budget.snapshot()["staging_bytes"]
        ):
            raise PredictionConfigError(
                "shared target scratch budget cannot hold probe and query copy"
            )
        pipeline = CUDAPredictionPipeline(
            provider, probe, draft_config, probe_config, execution_lock=execution_lock
        )
        return CUDAPredictionStartup(
            pipeline,
            draft_runner,
            draft_args,
            vocabulary,
            target_model_id,
            retained_bytes,
            target_scratch_budget,
        )
    except BaseException:
        if retained:
            _STARTUP_QUARANTINE.append(tuple(retained))
        try:
            set_global_server_args_for_scheduler(global_args)
            if "cpu_rng" in locals():
                torch.set_rng_state(cpu_rng)
            if "cuda_rng" in locals():
                torch.cuda.set_rng_state(cuda_rng, device)
            torch.cuda.synchronize(device)
        except BaseException:
            keep_lock = True
            _STARTUP_QUARANTINE.append((execution_lock, *retained))
            raise
        raise
    finally:
        if not keep_lock:
            execution_lock.release()
