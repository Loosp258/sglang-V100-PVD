"""Explicit, fail-closed assembly of the TP1 CUDA target serving components.

This factory is not a Scheduler startup hook. The caller supplies concrete
limits and one shared target scratch budget. The Scheduler owns its model
pools for the lifetime of the worker process; closing this binding must never
claim to free those pools while the runner and cache still reference them.
"""

import threading
from dataclasses import dataclass

from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import (
    LifecycleError,
    TargetExecutionArbiter,
)
from sglang.srt.disaggregation.pvd.cuda_model_attention import (
    CUDAModelPools,
    make_cuda_sparse_backend,
)
from sglang.srt.disaggregation.pvd.cuda_rank_batch import CUDARankBatchExecutor
from sglang.srt.disaggregation.pvd.cuda_refresh_driver import CUDARefreshDriver
from sglang.srt.disaggregation.pvd.cuda_route_discovery import CUDARouteDiscoveryQueue
from sglang.srt.disaggregation.pvd.cuda_scheduler_binding import (
    CUDADecodeSchedulerBinding,
)
from sglang.srt.disaggregation.pvd.cuda_sparse_attention import (
    CUDASparseAttentionWorkspace,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferBudget,
)

# A failed cleanup or a constructor that publishes a binding before raising
# must retain all owners: dropping them could free live CUDA memory.
_STARTUP_QUARANTINE = []


def _refuse_in_process_pool_release():
    raise LifecycleError("Scheduler model pools are process-owned")


@dataclass(frozen=True)
class CUDATargetServingComponents:
    scheduler: object
    backend: object
    native_backend: object
    workspace: CUDASparseAttentionWorkspace
    arbiter: TargetExecutionArbiter
    execution_lock: object
    driver: CUDARefreshDriver
    executor: CUDARankBatchExecutor
    pool_owner: ResourceGuard
    binding: CUDADecodeSchedulerBinding

    def close_drained(self):
        """Close an idle binding; the exact model pools remain process-owned."""
        self.driver._owner()
        if (
            self.scheduler.pvd_cuda_binding is not self.binding
            or self.scheduler.tp_worker.model_runner.attn_backend is not self.backend
            or self.driver._records
            or self.executor._active
            or self.executor._quarantined
            or self.driver.arbiter.busy
            or self.workspace.snapshot()["quarantine"] is not None
            or self.workspace.snapshot()["active"]
        ):
            raise LifecycleError("CUDA target still active or ownership changed")
        try:
            self.driver.begin_shutdown()
            self.binding.close()
            self.workspace.close()
        except BaseException:
            _STARTUP_QUARANTINE.append(self)
            raise


def install_cuda_target_components(
    scheduler,
    *,
    device,
    dtype,
    head_dim,
    chunk_tokens,
    target_scratch_budget,
    max_batch_size,
    max_requests,
    max_prefix_tokens,
    execution_lock,
    arbiter=None,
    route_queue=None,
    prepare_cuda_admission=None,
):
    """Construct resources then publish the backend/binding as one startup step.

    All bounds, placement and budgets are explicit. The binding itself checks
    topology, exact ChunkCache pools and all unsupported SGLang serving modes.
    No request may be admitted while this owner-thread function is executing.
    """
    if (
        not isinstance(target_scratch_budget, TransferBudget)
        or any(
            type(value) is not int or value <= 0
            for value in (
                head_dim,
                chunk_tokens,
                max_batch_size,
                max_requests,
                max_prefix_tokens,
            )
        )
        or max_batch_size < max_requests
        or getattr(scheduler, "pvd_cuda_binding", None) is not None
    ):
        raise LifecycleError(
            "explicit CUDA target bounds and shared scratch budget required"
        )
    runner = scheduler.tp_worker.model_runner
    manager = scheduler.disagg_decode_prealloc_queue.kv_manager
    cache = scheduler.tree_cache
    req_pool = scheduler.req_to_token_pool
    kv_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    if (
        runner.req_to_token_pool is not req_pool
        or runner.token_to_kv_pool_allocator is not scheduler.token_to_kv_pool_allocator
        or manager.kv_pool is not kv_pool
        or cache.req_to_token_pool is not req_pool
        or cache.token_to_kv_pool_allocator.get_kvcache() is not kv_pool
        or manager.scheduler is not scheduler
    ):
        raise LifecycleError("target model, cache and receiver must share exact pools")

    native_backend = runner.attn_backend
    workspace = driver = backend = executor = pool_owner = None
    if arbiter is None:
        arbiter = TargetExecutionArbiter()
    if not isinstance(execution_lock, type(threading.RLock())) or not isinstance(
        arbiter, TargetExecutionArbiter
    ):
        raise LifecycleError(
            "explicit shared reentrant lock and target arbiter required"
        )
    arbiter.owner()
    if arbiter.busy:
        raise LifecycleError("target execution is busy during CUDA startup")
    if route_queue is not None and (
        not isinstance(route_queue, CUDARouteDiscoveryQueue)
        or route_queue.manager is not manager
        or route_queue._closed
        or route_queue.pending
    ):
        raise LifecycleError("selected-route queue must be idle and owned by this D")
    try:
        workspace = CUDASparseAttentionWorkspace(
            device=device,
            dtype=dtype,
            head_dim=head_dim,
            chunk_tokens=chunk_tokens,
            budget=target_scratch_budget,
        )
        backend = make_cuda_sparse_backend(
            runner,
            workspace=workspace,
            execution_lock=execution_lock,
            output_budget=target_scratch_budget,
            max_batch_size=max_batch_size,
        )
        if (
            backend.consumer.req_pool is not req_pool
            or backend.consumer.kv_pool is not kv_pool
            or backend.consumer._lock is not execution_lock
        ):
            raise LifecycleError("CUDA backend did not bind the exact target pools")
        driver = CUDARefreshDriver(
            arbiter, max_requests=max_requests, max_prefix_tokens=max_prefix_tokens
        )
        executor = CUDARankBatchExecutor(
            backend.consumer, arbiter, max_requests=max_requests
        )
        pool_owner = ResourceGuard(
            CUDAModelPools(req_pool, kv_pool), _refuse_in_process_pool_release
        )

        # The binding requires this identity. If its constructor fails before
        # publishing itself, the native backend is restored below.
        runner.attn_backend = backend
        binding = CUDADecodeSchedulerBinding(
            scheduler,
            driver,
            executor,
            pool_owner=pool_owner,
            route_queue=route_queue,
            prepare_cuda_admission=prepare_cuda_admission,
        )
        return CUDATargetServingComponents(
            scheduler,
            backend,
            native_backend,
            workspace,
            arbiter,
            execution_lock,
            driver,
            executor,
            pool_owner,
            binding,
        )
    except BaseException:
        published = getattr(scheduler, "pvd_cuda_binding", None) is not None
        owner_pinned = pool_owner is not None and bool(pool_owner._owners)
        driver_claimed_lock = (
            driver is not None and getattr(driver, "_execution_lock", None) is not None
        )
        if (
            published
            or (
                runner.attn_backend is not native_backend
                and runner.attn_backend is not backend
            )
            or owner_pinned
            or driver_claimed_lock
        ):
            _STARTUP_QUARANTINE.append(
                (
                    scheduler,
                    runner,
                    workspace,
                    driver,
                    backend,
                    executor,
                    pool_owner,
                    route_queue,
                )
            )
            raise
        try:
            runner.attn_backend = native_backend
            if driver is not None:
                driver.begin_shutdown()
                driver.close_loop()
            if workspace is not None:
                workspace.close()
        except BaseException:
            _STARTUP_QUARANTINE.append(
                (
                    scheduler,
                    runner,
                    workspace,
                    driver,
                    backend,
                    executor,
                    pool_owner,
                    route_queue,
                )
            )
            raise
        raise
