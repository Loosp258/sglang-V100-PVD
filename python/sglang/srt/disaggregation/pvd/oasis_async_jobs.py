"""Request-owned async layer jobs: two active jobs and two retiring owners.

One event-loop thread owns registrations, clients and CUDA submissions. Public
future cancellation never cancels an admitted owner. A failing close retains
the loop so unknown native registrations never lose their creating owner.
"""
import asyncio
from concurrent.futures import Future
import threading
import time


def async_tensor_bound(capacity):
    """Conservative queued Q plus four live tensor/wire scratch allowance.

    Prompt/cache/bank/history and physical MRs retain their existing separate
    charges. This allowance covers queued Q and local job temporaries.
    """
    if type(capacity) is not int or not 1 <= capacity <= 32:
        raise ValueError('bounded async bank capacity required')
    return 56*28*128*4 + 4*(5*28*128*4 + 3*4*capacity*512)


class BoundedAsyncLayerJobs:
    def __init__(self, *, active=2, cleanup=2, capacity=56):
        if (type(active) is not int or type(cleanup) is not int or type(capacity) is not int
                or active != 2 or cleanup != 2 or not active+cleanup <= capacity <= 56):
            raise ValueError('two active/two cleanup owners and bounded capacity required')
        self.loop = asyncio.new_event_loop()
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._drain_future = None
        self._stop_requested = False
        self._capacity, self._live = capacity, 0
        self._closed, self._stopped, self._errors = False, False, []
        self._tasks = set()
        self._active_count = self._cleanup_count = 0
        self._active_peak = self._cleanup_peak = 0
        self._ready = threading.Event()
        def owner():
            asyncio.set_event_loop(self.loop)
            self._active = asyncio.Semaphore(active)
            self._cleanup = asyncio.Semaphore(cleanup)
            self._ready.set()
            self.loop.run_forever()
            self.loop.close()
        self.thread = threading.Thread(target=owner, name='oasis-async-owner', daemon=True)
        self.thread.start()
        if not self._ready.wait(5): raise RuntimeError('async owner did not start')

    def submit(self, work, *, published, timeout):
        public = Future()
        with self._lock:
            if self._closed or self._errors: raise RuntimeError('async owner is closed or failed')
            if self._live >= self._capacity: raise RuntimeError('async owner capacity exhausted')
            self._live += 1

            async def run():
                acquired = retiring = ready = False
                started = None
                def publish(reply):
                    nonlocal ready
                    if ready: raise RuntimeError('duplicate async READY publication')
                    if time.monotonic() >= published+timeout:
                        raise TimeoutError('async layer expired before READY')
                    ready = True
                    if public.set_running_or_notify_cancel():
                        public.set_result((reply, started, time.monotonic()))
                async def begin_cleanup():
                    nonlocal acquired, retiring
                    if not ready or retiring: raise RuntimeError('invalid async cleanup transition')
                    await self._cleanup.acquire()
                    retiring, acquired = True, False
                    with self._lock:
                        self._active_count -= 1
                        self._cleanup_count += 1
                        self._cleanup_peak = max(self._cleanup_peak, self._cleanup_count)
                    self._active.release()
                try:
                    await self._active.acquire()
                    acquired = True
                    started = time.monotonic()
                    with self._lock:
                        self._active_count += 1
                        self._active_peak = max(self._active_peak, self._active_count)
                        failed = bool(self._errors)
                    if failed: raise RuntimeError('async request previously failed')
                    if started >= published+timeout: raise TimeoutError('async layer expired in queue')
                    await work(publish, begin_cleanup)
                    if not ready: raise RuntimeError('async job did not publish READY')
                except BaseException as exc:
                    with self._lock: self._errors.append(exc)
                    if not ready and public.set_running_or_notify_cancel(): public.set_exception(exc)
                finally:
                    with self._lock:
                        self._live -= 1
                        if acquired: self._active_count -= 1
                        if retiring: self._cleanup_count -= 1
                    if acquired: self._active.release()
                    if retiring: self._cleanup.release()

            def admit():
                task = self.loop.create_task(run())
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            try:
                self.loop.call_soon_threadsafe(admit)
            except BaseException:
                self._live -= 1
                raise
        return public

    def close(self, *, timeout=60):
        if threading.get_ident() == self.thread.ident: raise RuntimeError('owner cannot join itself')
        with self._close_lock:
            return self._close_owned(timeout)

    def _close_owned(self, timeout):
        with self._lock:
            if self._stopped: return
            self._closed = True
        async def drain():
            # All submit/admit calls precede this barrier on the same loop.
            while self._tasks:
                # gather(all-done) may complete without yielding (Python 3.14).
                # Do not spin waiting for queued done callbacks to prune the set.
                self._tasks.difference_update(t for t in tuple(self._tasks) if t.done())
                if self._tasks:
                    await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        if self._drain_future is None:
            self._drain_future = asyncio.run_coroutine_threadsafe(drain(), self.loop)
        self._drain_future.result(timeout=timeout)
        with self._lock:
            if self._live: raise RuntimeError('async jobs did not drain')
            if self._errors:
                raise RuntimeError('async owned layer failed; retain owner loop') from self._errors[0]
        if not self._stop_requested:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._stop_requested = True
        self.thread.join(timeout=timeout)
        if self.thread.is_alive(): raise RuntimeError('async owner did not stop')
        with self._lock: self._stopped = True

    def snapshot(self):
        with self._lock:
            return dict(capacity=self._capacity, live=self._live, active=self._active_count,
                cleanup=self._cleanup_count, active_peak=self._active_peak, cleanup_peak=self._cleanup_peak,
                failures=len(self._errors), closed=self._closed, stopped=self._stopped,
                owner_threads=1, owner_thread=self.thread.ident)
