"""Bounded owner-thread retirement polling for the opt-in CPU Decode path.

No background thread and no native/GPU completion inference. A synchronous
Scheduler may own a private asyncio loop; an async caller uses its current
loop. poll() never awaits network completion. Failed drains retain admission
capacity, while partially failed releases remain quarantined.
"""

import asyncio
import math
import time
from dataclasses import dataclass

from sglang.srt.disaggregation.pvd.cpu_batch_forward import CPUBatchForwardExecutor
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError
from sglang.srt.disaggregation.pvd.cpu_request_release import CPURequestRelease


@dataclass
class _ReleaseRecord:
    req: object
    life: object
    owner: CPURequestRelease
    task: object = None
    retry_at: float = 0
    error: str | None = None


class CPUReleaseDriver:
    def __init__(
        self,
        executor,
        tree_cache,
        *,
        max_requests,
        max_inflight,
        retry_seconds,
        refresh_driver=None,
        clock=time.monotonic,
    ):
        if not isinstance(executor, CPUBatchForwardExecutor):
            raise LifecycleError("explicit CPU batch executor required")
        if (
            type(max_requests) is not int
            or max_requests <= 0
            or type(max_inflight) is not int
            or not 0 < max_inflight <= max_requests
            or type(retry_seconds) not in (float, int)
            or not math.isfinite(retry_seconds)
            or retry_seconds <= 0
        ):
            raise LifecycleError(
                "explicit positive bounded release capacities and retry interval required"
            )
        self.executor, self.tree_cache = executor, tree_cache
        self.arbiter = executor.dispatcher.arbiter
        self.arbiter.owner()
        if (
            getattr(executor, "release_driver", None) is not None
            or not tree_cache.is_chunk_cache()
            or tree_cache.req_to_token_pool is not executor.runner.req_to_token_pool
            or tree_cache.token_to_kv_pool_allocator
            is not executor.runner.token_to_kv_pool_allocator
            or executor.runner.req_to_token_pool.req_to_token.device.type != "cpu"
        ):
            raise LifecycleError(
                "one release driver for the exact CPU ChunkCache pools required"
            )
        if refresh_driver is not None and refresh_driver.arbiter is not self.arbiter:
            raise LifecycleError(
                "release and refresh drivers must share the target owner"
            )
        self.refresh_driver = refresh_driver
        self.max_requests, self.max_inflight = max_requests, max_inflight
        self.retry_seconds, self._clock = retry_seconds, clock
        self._records = {}
        self._closing = False
        self._pumping = False
        try:
            self._loop = asyncio.get_running_loop()
            self._owns_loop = False
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            self._owns_loop = True
        executor.release_driver = self

    def register(self, req, life):
        self.arbiter.owner()
        if self._closing or self._loop.is_closed():
            raise LifecycleError("release driver is closing")
        if req.rid in self._records or len(self._records) >= self.max_requests:
            raise LifecycleError(
                "release admission is full or request identity already owned"
            )
        owner = CPURequestRelease(req, life, self.executor)
        self._records[req.rid] = _ReleaseRecord(req, life, owner)
        return owner

    def _request_release(self, record):
        # Lazy import: keep pure ownership tests independent of serving imports.
        from sglang.srt.mem_cache.common import release_kv_cache

        release_kv_cache(record.req, self.tree_cache, is_insert=False)

    async def _progress(self, record):
        if self.refresh_driver is not None and self.refresh_driver.contains(
            record.life
        ):
            await self.refresh_driver.remove(record.life)
        return await record.owner.progress()

    def _defer(self, record, now):
        # A failed cache-binding check must stop this request, not its peers.
        # Never repair a changed binding or free an unknown owner's rows here.
        record.life.terminate(
            "CPU retirement requested", finished=record.req.finished()
        )
        try:
            self._request_release(record)
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
            record.error = (type(exc).__name__ + ": " + str(exc))[:512]
            record.retry_at = now + self.retry_seconds
        else:
            record.error = None

    def poll(self):
        self.arbiter.owner()
        if self._pumping:
            raise LifecycleError("release polling cannot reenter its owner loop")
        if self._loop.is_closed():
            return self.snapshot()
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if (not self._owns_loop and current is not self._loop) or (
            self._owns_loop and current is not None
        ):
            raise LifecycleError(
                "release polling must use its original owner event loop"
            )
        now = self._clock()
        for key, record in tuple(self._records.items()):
            record.life.poll()  # also retires a cancelled queued capture lease
            if record.task is not None and record.task.done():
                try:
                    done = record.task.result()
                except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
                    # Unknown drain outcome retains all ownership for retry.
                    record.error = (type(exc).__name__ + ": " + str(exc))[:512]
                    record.retry_at = now + self.retry_seconds
                else:
                    if done:
                        if record.owner.state != "released":
                            raise LifecycleError(
                                "release task claimed success without pool retirement"
                            )
                        del self._records[key]
                        continue
                    record.retry_at = now + self.retry_seconds
                record.task = None
            if (
                record.owner.state == "attached"
                and now >= record.retry_at
                and (
                    self._closing
                    or record.life.state in ("finished", "aborted")
                    or record.req.finished()
                    or getattr(record.req, "is_retracted", False)
                )
            ):
                self._defer(record, now)
        available = self.max_inflight - sum(
            record.task is not None for record in self._records.values()
        )
        if not self.arbiter.busy:
            for record in self._records.values():
                if available <= 0:
                    break
                if (
                    record.owner.state == "pending"
                    and record.task is None
                    and now >= record.retry_at
                ):
                    coroutine = self._progress(record)
                    try:
                        record.task = self._loop.create_task(coroutine)
                    except BaseException:
                        coroutine.close()
                        raise
                    record.error = None
                    available -= 1
        if self._owns_loop:
            self._pumping = True
            try:
                # Exactly one nonblocking turn; callbacks may suspend on I/O.
                self._loop.call_soon(self._loop.stop)
                self._loop.run_forever()
            finally:
                self._pumping = False
        return self.snapshot()

    def begin_shutdown(self):
        self.arbiter.owner()
        self._closing = True
        # Stop every lifecycle before the first cache callback can fail.
        for record in self._records.values():
            record.life.terminate(
                "CPU release driver shutting down", finished=record.req.finished()
            )
        now = self._clock()
        for record in self._records.values():
            if record.owner.state == "attached" and now >= record.retry_at:
                self._defer(record, now)

    def close_loop(self):
        """Only after every owned request and async task really drained."""
        self.arbiter.owner()
        if not self._closing or self._records:
            raise LifecycleError("release ownership must drain before loop close")
        if self._owns_loop and not self._loop.is_closed():
            if asyncio.all_tasks(self._loop):
                raise LifecycleError("other event-loop tasks still own resources")
            self._loop.close()

    def snapshot(self):
        self.arbiter.owner()
        return {
            "closing": self._closing,
            "drained": self._closing and not self._records,
            "requests": {
                key: {
                    "incarnation": r.life.incarnation,
                    "state": r.owner.state,
                    "inflight": r.task is not None,
                    "error": r.error,
                }
                for key, r in self._records.items()
            },
        }
