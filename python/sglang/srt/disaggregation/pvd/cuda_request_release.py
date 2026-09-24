"""Deferred CUDA ChunkCache retirement after controller and target drain.

The original cache function consumes Req bookkeeping through a deferred-free
view. Real KV indices are copied into the allocator and fenced BEFORE clearing
the mapping; the host request slot is freed only after the clear completes.
Any ambiguous mutation poisons both pools and retains the shared target lease.
"""

import uuid
from types import SimpleNamespace

import torch
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError
from sglang.srt.disaggregation.pvd.cuda_model_attention import CUDAModelPools
from sglang.srt.disaggregation.pvd.cuda_refresh_driver import CUDARefreshDriver
from sglang.srt.disaggregation.pvd.transfer_lifecycle import ResourceGuard


def _require_supported_pools(cache):
    from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.chunk_cache import ChunkCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

    if (
        type(cache) is not ChunkCache
        or type(cache.req_to_token_pool) not in (ReqToTokenPool, DecodeReqToTokenPool)
        or type(cache.token_to_kv_pool_allocator) is not TokenToKVPoolAllocator
        or cache.page_size != 1
    ):
        raise LifecycleError("CUDA retirement requires exact page-1 ChunkCache pools")


class _FreePlan:
    """No pool publication during the original Req bookkeeping operation."""

    def __init__(self, req, cache):
        self.req, self.cache = req, cache
        self.rows, self.slot_frees = [], 0
        self.req_to_token_pool = SimpleNamespace(
            req_to_token=cache.req_to_token_pool.req_to_token,
            free=self.free_slot,
        )
        self.token_to_kv_pool_allocator = SimpleNamespace(free=self.rows.append)

    def cache_finished_req(self, req, *, is_insert):
        # Constructor accepts only the exact ChunkCache implementation.
        type(self.cache).cache_finished_req(self, req, is_insert=is_insert)

    def free_slot(self, req):
        if req is not self.req or self.slot_frees:
            raise LifecycleError("foreign or repeated planned slot release")
        self.slot_frees += 1


class CUDARequestRelease:
    def __init__(self, req, driver, cache, *, pool_owner, release=None):
        if not isinstance(driver, CUDARefreshDriver) or not isinstance(
            pool_owner, ResourceGuard
        ):
            raise LifecycleError("explicit CUDA driver and actual pool owner required")
        driver._owner()
        _require_supported_pools(cache)
        record = driver._records.get(req.rid)
        pools = pool_owner.value
        allocator = cache.token_to_kv_pool_allocator
        if (
            record is None
            or record.req is not req
            or record.stopping
            or record.retirement is not None
            or req.req_pool_idx != record.slot
            or getattr(req, "pvd_cpu_kv_release", None) is not None
            or getattr(req, "pvd_cuda_kv_release", None) is not None
            or not isinstance(pools, CUDAModelPools)
            or pools.req_pool is not cache.req_to_token_pool
            or pools.kv_pool is not allocator.get_kvcache()
            or torch.device(allocator.device) != pools.req_pool.req_to_token.device
            or getattr(allocator, "pvd_cuda_retirement_error", None) is not None
            or getattr(pools.req_pool, "pvd_cuda_retirement_error", None) is not None
        ):
            raise LifecycleError(
                "exact live Req, registration and allocator owner required"
            )
        if release is None:
            from sglang.srt.mem_cache.common import _release_kv_cache_now

            release = _release_kv_cache_now
        if not callable(release):
            raise LifecycleError("original cache release callback required")
        self.req, self.driver, self.record, self.cache = req, driver, record, cache
        self.arbiter = driver.arbiter
        self.pool_owner, self._release = pool_owner, release
        self._pin = "cuda-request-retirement:" + uuid.uuid4().hex
        self.state, self.error = "attached", None
        self._plan = self._lease = None
        self._lock_held = False
        self._insert = True
        pool_owner.pin(self._pin)
        req.pvd_cuda_kv_release = record.retirement = self

    def _binding(self):
        if (
            self.req.pvd_cuda_kv_release is not self
            or self.driver._records.get(self.req.rid) is not self.record
            or self.record.req is not self.req
            or self.req.req_pool_idx != self.record.slot
            or self.req.rid != self.record.controller.group.coordinator.identity[0]
        ):
            raise LifecycleError("CUDA retirement identity or slot changed")

    def defer(self, req, cache, is_insert, release):
        self.arbiter.owner()
        if req is not self.req or req.pvd_cuda_kv_release is not self:
            raise LifecycleError("foreign CUDA release owner")
        if self.state == "released":
            return  # Do not inspect or clear a successor's reused slot.
        if self.state not in ("attached", "pending"):
            raise LifecycleError("CUDA retirement is active or quarantined")
        self._binding()
        if (
            cache is not self.cache
            or release is not self._release
            or type(is_insert) is not bool
        ):
            raise LifecycleError(
                "exact cache, release callback and boolean policy required"
            )
        self._insert = self._insert and is_insert
        self.state = "pending"
        self.driver._stop(self.record, "Scheduler requested CUDA KV retirement")

    def _synchronize(self):
        torch.cuda.synchronize(self.cache.req_to_token_pool.req_to_token.device)

    def release_after_controller_close(self):
        """Called by the driver after successfully awaiting this controller close."""
        self.arbiter.owner()
        if self.state == "released":
            return
        if self.state not in ("attached", "pending"):
            raise LifecycleError("CUDA retirement is active or quarantined")
        self._binding()
        record = self.record
        # A separate initial Prompt import or sparse receive may have lost
        # completion proof after this release owner was attached.  Closing the
        # controller alone does not make those model-pool rows safe to reuse.
        if (
            getattr(self.cache.req_to_token_pool, "pvd_cuda_retirement_error", None)
            is not None
            or getattr(
                self.cache.token_to_kv_pool_allocator,
                "pvd_cuda_retirement_error",
                None,
            )
            is not None
        ):
            raise LifecycleError(
                "CUDA model pools are poisoned; request KV retirement is unsafe"
            )
        if (
            not record.stopping
            or record.close_task is None
            or not record.close_task.done()
            or record.close_task.cancelled()
            or record.close_task.exception() is not None
            or self.driver.arbiter.busy
        ):
            raise LifecycleError("successful controller close and idle target required")
        pool, allocator = (
            self.cache.req_to_token_pool,
            self.cache.token_to_kv_pool_allocator,
        )
        if not allocator.is_not_in_free_group:
            raise LifecycleError(
                "CUDA retirement cannot run inside allocator free_group"
            )
        allocated = self.req.kv_allocated_len
        if (
            type(allocated) is not int
            or not 0 <= allocated <= pool.req_to_token.shape[1]
        ):
            raise LifecycleError("invalid allocated KV extent")
        self._lease = self.driver.arbiter.acquire()
        lock = self.driver._execution_lock
        if not lock.acquire(blocking=False):
            self.driver.arbiter.release(self._lease)
            self._lease = None
            raise LifecycleError("CUDA target lock is busy")
        self._lock_held, self.state = True, "releasing"
        try:
            self._synchronize()
            with torch.cuda.device(pool.req_to_token.device):
                self._plan = _FreePlan(self.req, self.cache)
                self._release(self.req, self._plan, is_insert=self._insert)
                if self._plan.slot_frees != 1 or self.req.req_pool_idx != record.slot:
                    raise LifecycleError(
                        "original release did not defer exactly one slot"
                    )
                rows = [v for t in self._plan.rows for v in t.tolist()]
                expected = pool.req_to_token[record.slot, :allocated].tolist()
                if (
                    rows != expected
                    or len(set(rows)) != len(rows)
                    or any(v <= 0 for v in rows)
                ):
                    raise LifecycleError(
                        "cache release plan does not cover exactly owned KV rows"
                    )
                for indices in self._plan.rows:
                    allocator.free(indices)
                self._synchronize()  # allocator must consume mapping views first
                pool.req_to_token[record.slot].zero_()
                self._synchronize()  # slot is NOT yet on the host free list
                pool.free(self.req)
                if self.req.req_pool_idx is not None:
                    raise LifecycleError("request slot was not retired")
            self.pool_owner.unpin(self._pin)
        except BaseException as exc:
            self.error, self.state = exc, "quarantined"
            reason = "PVD CUDA retirement unknown: " + str(exc)[:256]
            pool.pvd_cuda_retirement_error = allocator.pvd_cuda_retirement_error = (
                reason
            )
            # Keep actual plan/views, pool pin, shared arbiter and target lock.
            raise
        self._plan = None
        self.state = "released"
        self.driver.arbiter.release(self._lease)
        self._lease = None
        lock.release()
        self._lock_held = False
        # Successful tombstones must not keep the whole worker/other requests
        # alive through a finished Req. Replay only needs identity and arbiter.
        self.driver = self.record = self.cache = self.pool_owner = self._release = None
