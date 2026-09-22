"""Opt-in CPU request release at the real cache-release boundary.

Owner-thread progress is explicit; no serving mode or GPU/RDMA fence is added.
The synchronous Scheduler callback requests retirement, never waits on HTTP.
The async owner drains the lifecycle and invokes the original cache release
exactly once. A partially failed allocator release is quarantined, not retried.
"""

from sglang.srt.disaggregation.pvd.cpu_batch_forward import CPUBatchForwardExecutor
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError


class CPURequestRelease:
    def __init__(self, req, lifecycle, executor):
        if not isinstance(executor, CPUBatchForwardExecutor):
            raise LifecycleError("explicit CPU batch executor required")
        self.arbiter = executor.dispatcher.arbiter
        self.arbiter.owner()
        slot = executor._storage.get(lifecycle)
        if (
            lifecycle.arbiter is not self.arbiter
            or req.rid != lifecycle.request_id
            or type(slot) is not int
            or req.req_pool_idx != slot
            or getattr(req, "pvd_cpu_kv_release", None) is not None
            or executor.runner.req_to_token_pool.req_to_token.device.type != "cpu"
        ):
            raise LifecycleError("exact bound CPU request storage required")
        self.req, self.life, self.executor = req, lifecycle, executor
        self.slot, self.incarnation = slot, lifecycle.incarnation
        self.state = "attached"
        self._cache = self._release = None
        self._is_insert = True
        self._progressing = False
        req.pvd_cpu_kv_release = self

    def _binding(self, req):
        if req is not self.req or req.pvd_cpu_kv_release is not self:
            raise LifecycleError("foreign request release owner")
        if (
            req.rid != self.life.request_id
            or self.life.incarnation != self.incarnation
            or req.req_pool_idx != self.slot
            or self.executor._storage.get(self.life) != self.slot
        ):
            raise LifecycleError("request release storage or incarnation changed")

    def defer(self, req, tree_cache, is_insert, release):
        self.arbiter.owner()
        if req is not self.req or req.pvd_cpu_kv_release is not self:
            raise LifecycleError("foreign request release owner")
        if self.state == "released":
            return  # successful duplicate callback cannot touch a reused slot
        if self.state in ("releasing", "quarantined"):
            raise LifecycleError("ambiguous cache release is quarantined")
        self._binding(req)
        if (
            type(is_insert) is not bool
            or not callable(release)
            or not tree_cache.is_chunk_cache()
            or tree_cache.req_to_token_pool
            is not self.executor.runner.req_to_token_pool
            or tree_cache.token_to_kv_pool_allocator
            is not self.executor.runner.token_to_kv_pool_allocator
            or (self._cache is not None and self._cache is not tree_cache)
            or (self._release is not None and self._release is not release)
        ):
            raise LifecycleError(
                "exact cache, release function and insertion policy required"
            )
        self._cache, self._release = tree_cache, release
        self._is_insert = self._is_insert and is_insert
        self.state = "pending"
        # During the result callback, the real Req writer may already have
        # appended its final token. The bridge will still observe that write;
        # termination never rolls it back or releases an in-flight permit.
        self.life.terminate("Scheduler requested KV release", finished=req.finished())

    async def progress(self):
        self.arbiter.owner()
        if self._progressing:
            raise LifecycleError("request release progress is already in flight")
        if self.state == "released":
            return True
        if self.state in ("releasing", "quarantined"):
            raise LifecycleError("ambiguous cache release is quarantined")
        if self.state != "pending" or self.arbiter.busy:
            return False
        self._binding(self.req)
        self._progressing = True
        try:
            # A failed/ cancelled drain leaves the pending registration and
            # final pool allocation intact. Native ownership is handled by the
            # controller; task cancellation is NOT a completion proof.
            await self.life.close()
            if self.arbiter.busy:
                return False
            self._binding(self.req)  # identity can change across an await
            self.executor.unregister_storage(self.life)
            self.state = "releasing"
            try:
                self._release(self.req, self._cache, is_insert=self._is_insert)
                if self.req.req_pool_idx is not None:
                    raise LifecycleError(
                        "cache release did not retire the request slot"
                    )
            except BaseException:
                self.state = "quarantined"
                raise
            self.state = "released"
            self._cache = self._release = self.executor = self.life = None
            return True
        finally:
            self._progressing = False
