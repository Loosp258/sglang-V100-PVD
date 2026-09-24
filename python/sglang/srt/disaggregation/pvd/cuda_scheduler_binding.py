"""Explicit CUDA service binding for the ordinary disaggregated Decode loop.

Does not load a model or build request controllers. The startup/admission
factory must supply the installed backend and receiver-claimed requests.
"""

import uuid

from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError
from sglang.srt.disaggregation.pvd.cuda_model_attention import CUDAModelPools
from sglang.srt.disaggregation.pvd.cuda_rank_batch import (
    CUDABatchResultRefused,
    CUDARankBatchExecutor,
)
from sglang.srt.disaggregation.pvd.cuda_refresh_driver import CUDARefreshDriver
from sglang.srt.disaggregation.pvd.cuda_request_release import _require_supported_pools
from sglang.srt.disaggregation.pvd.cuda_route_discovery import (
    CUDARouteDiscoveryQueue,
)
from sglang.srt.disaggregation.pvd.cuda_schedule_bridge import CUDAScheduleBridge
from sglang.srt.disaggregation.pvd.cuda_waiting_admission import (
    CUDAWaitingAdmissionCoordinator,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import ResourceGuard


def binding_for(scheduler):
    binding = getattr(scheduler, "pvd_cuda_binding", None)
    if binding is not None:
        if (
            not isinstance(binding, CUDADecodeSchedulerBinding)
            or binding.scheduler is not scheduler
        ):
            raise LifecycleError("foreign CUDA Scheduler binding")
        binding._check()
    return binding


class CUDADecodeSchedulerBinding:
    def __init__(
        self,
        scheduler,
        driver,
        executor,
        *,
        pool_owner,
        route_queue=None,
        prepare_cuda_admission=None,
    ):
        if (
            not isinstance(driver, CUDARefreshDriver)
            or not isinstance(executor, CUDARankBatchExecutor)
            or not isinstance(pool_owner, ResourceGuard)
        ):
            raise LifecycleError(
                "explicit CUDA driver, executor and pool owner required"
            )
        driver._owner()
        args = scheduler.server_args
        runner = scheduler.tp_worker.model_runner
        manager = scheduler.disagg_decode_prealloc_queue.kv_manager
        pools = pool_owner.value
        cache = scheduler.tree_cache
        _require_supported_pools(cache)
        if (
            getattr(scheduler, "pvd_cuda_binding", None) is not None
            or getattr(scheduler, "pvd_cpu_release_driver", None) is not None
            or args.disaggregation_topology != "pvd"
            or args.disaggregation_mode != "decode"
            or scheduler.enable_overlap
            or not args.disable_cuda_graph
            or args.dp_size != 1
            or args.enable_dp_attention
            or args.speculative_algorithm is not None
            or args.disaggregation_decode_enable_radix_cache
            or args.disaggregation_decode_enable_offload_kvcache
            or scheduler.enable_hisparse
            or type(scheduler.max_running_requests) is not int
            or not 0
            < scheduler.max_running_requests
            <= min(
                driver.max_requests,
                executor.dispatcher.max_requests,
                executor.consumer._max_batch,
            )
            or args.page_size != 1
            or runner.tp_size != 1
            or runner.pp_size != 1
            or runner.attn_cp_size != 1
            or not manager.waiting_queue_bootstrap
            or manager.tp_size != 1
            or manager.tp_rank != 0
            or manager.scheduler is not scheduler
            or not isinstance(pools, CUDAModelPools)
            or pools.req_pool is not scheduler.req_to_token_pool
            or pools.kv_pool is not manager.kv_pool
            or pools.req_pool is not cache.req_to_token_pool
            or pools.kv_pool is not cache.token_to_kv_pool_allocator.get_kvcache()
            or executor.consumer.req_pool is not pools.req_pool
            or executor.consumer.kv_pool is not pools.kv_pool
            or getattr(runner.attn_backend, "consumer", None) is not executor.consumer
            or executor.dispatcher.arbiter is not driver.arbiter
            or driver.arbiter.busy
            or driver._closing
            or driver._source_quarantine is not None
            or executor._active
            or executor._quarantined
            or driver._execution_lock not in (None, executor.consumer._lock)
            or (
                prepare_cuda_admission is not None
                and (route_queue is None or not callable(prepare_cuda_admission))
            )
            or (
                route_queue is not None
                and (
                    not isinstance(route_queue, CUDARouteDiscoveryQueue)
                    or route_queue.manager is not manager
                    or route_queue.max_inflight > driver.max_requests
                    or route_queue._closed
                )
            )
        ):
            raise LifecycleError(
                "exact TP1 non-overlap CUDA backend/pools/bootstrap required"
            )
        admission = (
            CUDAWaitingAdmissionCoordinator(
                manager, driver, pool_owner, prepare_cuda_admission
            )
            if prepare_cuda_admission is not None
            else None
        )
        self.scheduler, self.driver, self.executor = scheduler, driver, executor
        self.route_queue = route_queue
        self.admission = admission
        self.pool_owner, self.manager, self.runner = pool_owner, manager, runner
        self.closed = False
        self._pin = "cuda-scheduler:" + uuid.uuid4().hex
        pool_owner.pin(self._pin)
        driver._execution_lock = executor.consumer._lock
        scheduler.pvd_cuda_binding = self

    def _check(self):
        self.driver._owner()
        if (
            self.closed
            or self.scheduler.pvd_cuda_binding is not self
            or self.pool_owner.value is None
            or self.runner is not self.scheduler.tp_worker.model_runner
            or getattr(self.runner.attn_backend, "consumer", None)
            is not self.executor.consumer
            or self.scheduler.req_to_token_pool is not self.executor.consumer.req_pool
            or self.scheduler.disagg_decode_prealloc_queue.kv_manager
            is not self.manager
            or self.manager.scheduler is not self.scheduler
            or self.manager.kv_pool is not self.executor.consumer.kv_pool
            or self.scheduler.tree_cache.req_to_token_pool
            is not self.executor.consumer.req_pool
            or self.scheduler.tree_cache.token_to_kv_pool_allocator.get_kvcache()
            is not self.executor.consumer.kv_pool
            or self.driver._execution_lock is not self.executor.consumer._lock
            or self.executor.dispatcher.arbiter is not self.driver.arbiter
            or self.scheduler.enable_overlap
            or not self.scheduler.server_args.disable_cuda_graph
            or (
                self.route_queue is not None
                and self.route_queue.manager is not self.manager
            )
        ):
            raise LifecycleError("CUDA Scheduler binding changed or closed")

    def _record(self, req):
        record = self.driver._records.get(req.rid)
        if record is None:
            return None
        if (
            record.req is not req
            or record.full_session is None
            or record.retirement is None
        ):
            raise LifecycleError("Scheduler requires receiver-claimed CUDA Req owners")
        if (
            record.full_session.manager is not self.manager
            or record.retirement.pool_owner is not self.pool_owner
        ):
            raise LifecycleError(
                "CUDA request belongs to another receiver or pool owner"
            )
        return record

    def waiting_ready(self, req):
        self._check()
        record = self._record(req)
        if record is None or record.stopping or record.quarantined or req.finished():
            return False
        n = self.driver._observe(record)
        if n != 0:
            raise LifecycleError(
                "initial CUDA waiting admission requires zero D tokens"
            )
        return record.controller.can_decode(0)

    def poll(self):
        self._check()
        if self.route_queue is not None:
            unclaimed = [
                req
                for req in self.scheduler.waiting_queue
                if self.driver._records.get(req.rid) is None
            ]
            for req, error in self.route_queue.poll(unclaimed):
                self.scheduler._abort_pvd_cuda_requests(
                    [req], f"selected V route discovery failed: {error}"
                )
            if self.admission is not None:
                for req in unclaimed:
                    if (
                        req.finished()
                        or self.driver.arbiter.busy
                        or len(self.driver._records) >= self.driver.max_requests
                    ):
                        continue
                    try:
                        selected = self.route_queue.ready_for(req)
                        if selected is not None:
                            self.admission.admit(req, selected)
                    except Exception as exc:  # noqa: BLE001 - request-level abort boundary
                        self.scheduler._abort_pvd_cuda_requests(
                            [req], f"CUDA waiting admission failed: {exc}"
                        )
        # Emit already-stopped requests before poll can retire their records.
        self._abort_stopped()
        self.driver.poll()
        self._abort_stopped()
        return self.driver.snapshot()

    def selected_routes_for(self, req):
        """Consume one validated binding without blocking the Scheduler."""
        self._check()
        if self.route_queue is None:
            return None
        return self.route_queue.ready_for(req)

    def _abort_stopped(self):
        failures = [
            r
            for r in self.driver._records.values()
            if r.stopping and not r.req.finished()
        ]
        for record in failures:
            self.scheduler._abort_pvd_cuda_requests(
                [record.req], str(record.error or "CUDA request stopped")
            )

    @property
    def pending(self):
        self._check()
        return bool(self.driver._records or self.scheduler.waiting_queue)

    def ready_to_prepare(self, batch):
        """Gate BEFORE prepare_for_decode allocates another generated KV row."""
        self._check()
        if self.driver.arbiter.busy:
            return False
        before = batch.batch_size()
        batch.filter_batch(v1_spec_info_filtered=True)
        if batch.batch_size() != before:
            batch.batch_is_full = False
        if batch.is_empty():
            return False
        for req in batch.reqs:
            record = self._record(req)
            if record is None:
                raise LifecycleError("unregistered request in CUDA running batch")
            if record.stopping or record.quarantined:
                return False
            if not record.controller.can_decode(self.driver._observe(record)):
                return False  # Wait-all: no peer allocation or partial forward.
        if not batch.check_decode_mem():
            # Native in-place retraction destroys deferred-release bookkeeping.
            # First explicit serving policy: terminate this batch, drain normally.
            self.scheduler._abort_pvd_cuda_requests(
                list(batch.reqs),
                "CUDA Decode KV capacity exhausted; asynchronous retraction is unsupported",
            )
            batch.filter_batch(v1_spec_info_filtered=True)
            return False
        return True

    def run(self, batch):
        self._check()
        bridge = CUDAScheduleBridge(
            self.executor, self.driver, batch, pool_owner=self.pool_owner
        )
        try:
            return bridge.run(
                forward=lambda: self.scheduler.run_batch(batch),
                processor=self.scheduler.batch_result_processor,
                result_handler=self.scheduler.process_batch_result,
            )
        except CUDABatchResultRefused as exc:
            # Only this pre-commit refusal can become a request-local abort.
            # Unknown GPU completion, a failed result hook, or uncertain stop
            # ownership must still fail closed at the worker boundary.
            if (
                bridge.state != "failed"
                or bridge._result_processing_started
                or self.executor._quarantined
                or self.executor._active
                or self.executor.dispatcher._ticket is not None
                or self.executor.consumer.snapshot()["quarantine"] is not None
                or len(bridge.records) != len(batch.reqs)
                or any(
                    saved.registration.req is not req
                    or saved.registration.req.rid != saved.request_id
                    or not saved.registration.stopping
                    or saved.registration.quarantined
                    or tuple(saved.registration.req.output_ids) != saved.outputs
                    for saved, req in zip(bridge.records, batch.reqs, strict=True)
                )
            ):
                raise
            self.scheduler._abort_pvd_cuda_requests(
                list(batch.reqs), f"rank result refused: {exc}"
            )
            batch.filter_batch(v1_spec_info_filtered=True)
            batch.batch_is_full = False
            return None

    def close(self):
        """Worker shutdown only; never fall back to the normal backend afterwards."""
        self._check()
        if (
            not self.driver._closing
            or self.driver._records
            or self.executor._active
            or self.executor._quarantined
        ):
            raise LifecycleError("CUDA controllers and batch executor must drain first")
        if self.route_queue is not None and not self.route_queue.close():
            raise LifecycleError("selected V route lookups must drain first")
        self.driver.close_loop()
        # Even a failing final pool callback must never re-enable scheduling.
        self.closed = True  # Leave a closed marker; no silent legacy fallback.
        self.pool_owner.unpin(self._pin)
