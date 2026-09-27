"""Compose the opt-in TP1 CUDA PVD Decode owners on the Scheduler thread.

This module is deliberately separate from ServerArgs parsing and the Scheduler
hook. It does not publish a partially loaded draft or silently fall back to
full-Prompt decode if a CUDA component fails to install.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import torch
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError
from sglang.srt.disaggregation.pvd.cuda_prediction_startup import (
    CUDAPredictionStartup,
    build_cuda_prediction_startup,
)
from sglang.srt.disaggregation.pvd.cuda_prompt_prewarm import CUDAPromptPrewarmer
from sglang.srt.disaggregation.pvd.cuda_route_discovery import (
    CUDARouteDiscoveryQueue,
)
from sglang.srt.disaggregation.pvd.cuda_target_startup import (
    CUDATargetServingComponents,
    install_cuda_target_components,
)
from sglang.srt.disaggregation.pvd.cuda_waiting_admission import (
    CUDAWaitingAdmissionResources,
)
from sglang.srt.disaggregation.pvd.draft_sglang import DraftPlacement
from sglang.srt.disaggregation.pvd.probe_lane_identity import checkpoint_identity
from sglang.srt.disaggregation.pvd.probe_lane_sidecar_process import (
    ProbeSidecarProcess,
    launch_probe_sidecar,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

_STARTUP_QUARANTINE = []


def maybe_install_cuda_predictive_serving(scheduler):
    """The only production startup gate; disabled mode has no side effects."""
    args = scheduler.server_args
    if not getattr(args, "pvd_cuda_predictive_serving", False):
        return None
    from sglang.srt.disaggregation.pvd.cuda_serving_limits import (
        load_cuda_serving_limits,
    )

    limits = load_cuda_serving_limits(
        args.pvd_cuda_serving_config,
        refresh_interval=args.pvd_kv_refresh_interval,
        predict_tokens=args.pvd_draft_predict_tokens,
    )
    return install_cuda_predictive_serving(scheduler, limits)


@dataclass(frozen=True)
class CUDAPredictiveServing:
    """Strong process-lifetime owners of the CUDA target and private draft."""

    target: CUDATargetServingComponents
    prediction: CUDAPredictionStartup
    route_queue: CUDARouteDiscoveryQueue
    bank_budget: TransferBudget
    target_scratch_budget: TransferBudget
    head_mapping: QueryHeadMapping
    sidecar: ProbeSidecarProcess | None = None
    prompt_prewarm: CUDAPromptPrewarmer | None = None

    def close_drained(self):
        """Best-effort orderly close; process exit still owns model-pool memory."""
        if self.target.close_drained() is False:
            return False
        if self.prompt_prewarm is not None:
            self.prompt_prewarm.close_drained()
        if self.sidecar is not None:
            self.sidecar.close()
        return True


def install_cuda_predictive_serving(scheduler, limits) -> CUDAPredictiveServing:
    """Load one draft and install one exact-pool sparse target binding.

    The caller must invoke this before requests are served. All target scratch
    users receive the same TransferBudget *object*, not independently sized
    budgets. The draft's weights, private pools and scratch have their own
    explicit budgets. Failed CUDA loading retains possibly-live owners until
    worker exit; it never resumes the legacy path in that process.
    """
    from sglang.srt.disaggregation.pvd.cuda_serving_limits import (
        CUDAServingLimits,
        ProbeSidecarLimits,
    )

    if not isinstance(limits, CUDAServingLimits):
        raise LifecycleError("validated explicit CUDA serving limits required")
    args = scheduler.server_args
    if not args.pvd_cuda_predictive_serving or not args.pvd_predictive_retrieval_config:
        raise LifecycleError("explicit CUDA predictive serving opt-in required")
    if getattr(scheduler, "pvd_cuda_components", None) is not None:
        raise LifecycleError("CUDA predictive serving already installed")
    runner = scheduler.tp_worker.model_runner
    manager = scheduler.disagg_decode_prealloc_queue.kv_manager
    if (
        type(scheduler.max_running_requests) is not int
        or scheduler.max_running_requests <= 0
    ):
        raise LifecycleError("positive Scheduler request bound required")
    if (
        type(limits.probe_prefix_cache_bytes) is not int
        or limits.probe_prefix_cache_bytes < 0
    ):
        raise LifecycleError("nonnegative target probe cache budget required")
    if (
        args.pvd_kv_refresh_interval <= limits.lead_tokens
        or args.pvd_draft_predict_tokens < limits.lead_tokens
        or limits.max_sequence_tokens <= args.pvd_draft_predict_tokens
        or limits.max_sequence_tokens
        > min(runner.model_config.context_len, args.max_total_tokens)
    ):
        raise LifecycleError(
            "prediction lead or prefix exceeds committed target bounds"
        )
    if any(
        type(value) is not int or value <= 0
        for value in (
            args.pvd_retrieval_bank_budget_bytes,
            args.pvd_retrieval_scratch_budget_bytes,
            args.pvd_draft_scratch_budget_bytes,
            args.pvd_draft_persistent_budget_bytes,
            args.pvd_retrieval_top_k,
            args.pvd_retrieval_max_union_tokens,
        )
    ):
        raise LifecycleError("explicit PVD retrieval and draft budgets required")
    model = runner.model.config
    head_mapping = QueryHeadMapping(
        model.num_attention_heads, runner.model_config.get_total_num_kv_heads()
    )
    device = torch.device(f"cuda:{runner.gpu_id}")
    kv_pool = runner.token_to_kv_pool_allocator.get_kvcache()
    kv_dtype = kv_pool.get_key_buffer(0).dtype
    lock = threading.RLock()
    bank_budget = TransferBudget(
        args.pvd_retrieval_bank_budget_bytes, limits.bank_max_reservations
    )
    scratch_budget = TransferBudget(
        args.pvd_retrieval_scratch_budget_bytes,
        limits.target_scratch_max_reservations,
    )
    prefix_budget = (
        TransferBudget(limits.probe_prefix_cache_bytes, scheduler.max_running_requests)
        if limits.probe_prefix_cache_bytes
        else None
    )
    placement = DraftPlacement(
        gpu_id=runner.gpu_id,
        tp_rank=0,
        scratch_budget_bytes=args.pvd_draft_scratch_budget_bytes,
        persistent_budget_bytes=args.pvd_draft_persistent_budget_bytes,
        max_concurrent_branches=1,
    )
    route_queue = CUDARouteDiscoveryQueue(
        manager, max_inflight=scheduler.max_running_requests
    )
    prediction = target = sidecar = None
    try:
        prediction = build_cuda_prediction_startup(
            runner,
            draft_model_path=args.pvd_draft_model_path,
            draft_revision=args.pvd_draft_revision,
            draft_mem_fraction_static=args.pvd_draft_mem_fraction_static,
            target_model_id=args.pvd_retrieval_vector_space,
            placement=placement,
            execution_lock=lock,
            max_prefix_tokens=(
                limits.max_sequence_tokens - args.pvd_draft_predict_tokens
            ),
            predict_tokens=args.pvd_draft_predict_tokens,
            draft_transient_bytes_bound=limits.draft_transient_bytes_bound,
            probe_transient_bytes_bound=limits.probe_transient_bytes_bound,
            target_scratch_budget=scratch_budget,
            prefix_budget=prefix_budget,
        )
        if prediction.target_scratch_budget is not scratch_budget:
            raise LifecycleError("prediction did not retain the shared target budget")

        if os.environ.get("PVD_PRECOMPILE_QWEN_KERNELS") == "1":
            from sglang.srt.disaggregation.pvd.qwen_kernel_precompile import (
                precompile_qwen_decode_kernels,
            )
            from sglang.srt.models.qwen2 import Qwen2ForCausalLM

            if (
                type(runner.model) is not Qwen2ForCausalLM
                or type(prediction.draft_runner.model) is not Qwen2ForCausalLM
            ):
                raise LifecycleError(
                    "PVD Qwen JIT precompile requires Qwen2 target and draft"
                )
            precompile_qwen_decode_kernels(runner.model, kv_pool, device=device)
            precompile_qwen_decode_kernels(
                prediction.draft_runner.model,
                prediction.draft_runner.token_to_kv_pool,
                device=device,
            )

        sidecar_config = limits.probe_sidecar
        if sidecar_config is not None:
            if not isinstance(sidecar_config, ProbeSidecarLimits):
                raise LifecycleError("validated probe sidecar limits required")
            parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if (
                parent_visible is None
                or not parent_visible.isdecimal()
                or parent_visible == sidecar_config.cuda_visible_devices
                or runner.gpu_id != 0
                or args.pvd_draft_predict_tokens not in (1, 2, 4, 8, 16, 32)
                or not os.path.isdir(args.model_path)
                or os.path.realpath(args.tokenizer_path or args.model_path)
                != os.path.realpath(args.model_path)
            ):
                raise LifecycleError(
                    "sidecar requires a distinct visible GPU, local target "
                    "checkpoint/tokenizer and supported prediction length"
                )
            checkpoint = checkpoint_identity(os.path.realpath(args.model_path))
            sidecar = launch_probe_sidecar(
                sidecar_config.script_path,
                [
                    "--model-path",
                    os.path.realpath(args.model_path),
                    "--draft-model-path",
                    os.path.realpath(args.pvd_draft_model_path),
                    "--context-length",
                    str(limits.max_sequence_tokens),
                    "--max-total-tokens",
                    str(limits.max_sequence_tokens),
                    "--predict-tokens",
                    str(args.pvd_draft_predict_tokens),
                    "--draft-mem-fraction-static",
                    str(args.pvd_draft_mem_fraction_static),
                    "--draft-scratch-budget-bytes",
                    str(args.pvd_draft_scratch_budget_bytes),
                    "--draft-persistent-budget-bytes",
                    str(args.pvd_draft_persistent_budget_bytes),
                    "--draft-prefix-cache-budget-bytes",
                    str(limits.draft_prefix_cache_bytes),
                    "--draft-transient-bytes-bound",
                    str(limits.draft_transient_bytes_bound),
                    "--probe-budget-bytes",
                    str(args.pvd_retrieval_scratch_budget_bytes),
                    "--probe-prefix-cache-budget-bytes",
                    str(limits.probe_prefix_cache_bytes),
                    "--probe-transient-bytes-bound",
                    str(limits.probe_transient_bytes_bound),
                    "--reply-budget-bytes",
                    str(sidecar_config.reply_budget_bytes),
                    "--max-connections",
                    str(min(scheduler.max_running_requests, 8)),
                ],
                checkpoint=checkpoint,
                target_model_id=args.pvd_retrieval_vector_space,
                reply_budget=TransferBudget(
                    sidecar_config.reply_budget_bytes, scheduler.max_running_requests
                ),
                startup_timeout=sidecar_config.startup_timeout_seconds,
                directory_parent=sidecar_config.directory_parent,
                cuda_visible_devices=sidecar_config.cuda_visible_devices,
                background_loop=manager.control.loop,
            )

        def prepare(_preflight):
            if sidecar is not None:
                sidecar.check_alive()
            return CUDAWaitingAdmissionResources(
                pipeline=prediction.pipeline,
                head_mapping=head_mapping,
                vector_space=args.pvd_retrieval_vector_space,
                metric=args.pvd_retrieval_metric,
                top_k=args.pvd_retrieval_top_k,
                max_union_tokens=args.pvd_retrieval_max_union_tokens,
                max_head_dim=runner.model_config.head_dim,
                bank_budget=bank_budget,
                staging_budget=scratch_budget,
                copy_budget=scratch_budget,
                aggregate_budget=scratch_budget,
                execution_lock=lock,
                lead_tokens=limits.lead_tokens,
                timeout_seconds=limits.request_timeout_seconds,
                max_pending_events=limits.max_pending_events,
                max_pending_bytes=limits.max_pending_bytes,
                poll_interval_seconds=limits.poll_interval_seconds,
                lane_client=sidecar.client if sidecar is not None else None,
                lane_checkpoint=sidecar.checkpoint if sidecar is not None else None,
                sidecar_prefix_prewarm=limits.sidecar_prefix_prewarm,
            )

        target = install_cuda_target_components(
            scheduler,
            device=device,
            dtype=kv_dtype,
            head_dim=runner.model_config.head_dim,
            chunk_tokens=limits.attention_chunk_tokens,
            attention_impl=limits.attention_impl,
            max_sequence_tokens=limits.max_sequence_tokens,
            total_kv_heads=head_mapping.total_kv_heads,
            num_query_heads=head_mapping.num_query_heads,
            target_scratch_budget=scratch_budget,
            max_batch_size=scheduler.max_running_requests,
            max_requests=scheduler.max_running_requests,
            max_prefix_tokens=limits.max_sequence_tokens,
            execution_lock=lock,
            route_queue=route_queue,
            prepare_cuda_admission=prepare,
        )
        if (
            target.execution_lock is not lock
            or target.backend.consumer._lock is not lock
            or prediction.pipeline.probe._execution_lock is not lock
            or target.workspace._budget is not scratch_budget
        ):
            raise LifecycleError(
                "CUDA target and prediction do not share execution owners"
            )
        installed = CUDAPredictiveServing(
            target,
            prediction,
            route_queue,
            bank_budget,
            scratch_budget,
            head_mapping,
            sidecar,
            (
                CUDAPromptPrewarmer(
                    target.driver,
                    lane_client=sidecar.client,
                    checkpoint=sidecar.checkpoint,
                    target_model_id=prediction.target_model_id,
                    probe_config=prediction.pipeline.probe_config,
                    head_dim=runner.model_config.head_dim,
                    timeout_seconds=limits.request_timeout_seconds,
                    max_prefix_tokens=min(limits.max_sequence_tokens, 16_384 - 1),
                )
                if sidecar is not None and limits.sidecar_prefix_prewarm
                else None
            ),
        )
        if installed.prompt_prewarm is not None:
            manager.cuda_prompt_prewarm = installed.prompt_prewarm
            target.driver.cuda_prompt_prewarm = installed.prompt_prewarm
        scheduler.pvd_cuda_components = installed
        return installed
    except BaseException:
        if sidecar is not None:
            try:
                sidecar.close()
            except BaseException:
                _STARTUP_QUARANTINE.append(
                    (scheduler, target, prediction, route_queue, sidecar)
                )
                raise LifecycleError("sidecar shutdown uncertain during startup")
        # A loaded draft cannot be safely unloaded from a running target
        # process. Any failure after that point is terminal for this worker.
        if prediction is not None or target is not None:
            _STARTUP_QUARANTINE.append((scheduler, target, prediction, route_queue))
        raise
