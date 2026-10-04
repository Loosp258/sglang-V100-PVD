"""Bounded jobs publish READY before same-thread owner cleanup.

Publication never cancels the owned worker. Cleanup failures latch a request
error, and close joins all admitted work even if a public future was cancelled.
"""
from concurrent.futures import Future, ThreadPoolExecutor
import threading
import time


class OwnedReadyCleanup:
    def __init__(self, *, workers=2, capacity=56):
        if type(workers) is not int or type(capacity) is not int or not 1 <= workers <= capacity:
            raise ValueError("positive bounded worker/capacity required")
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="oasis-owned")
        self._lock = threading.Lock()
        self._capacity, self._live = capacity, 0
        self._closed, self._errors = False, []
        self._owned = set()

    def submit(self, work, *, published, timeout):
        public = Future()
        with self._lock:
            if self._closed or self._errors:
                raise RuntimeError("owned cleanup is closed or failed")
            if self._live >= self._capacity:
                raise RuntimeError("owned cleanup capacity exhausted")
            self._live += 1

            def run():
                started = time.monotonic()
                ready = False

                def publish(reply):
                    nonlocal ready
                    if ready:
                        raise RuntimeError("duplicate READY publication")
                    ready = True
                    # Consumer cancellation never abandons native owner cleanup.
                    if public.set_running_or_notify_cancel():
                        public.set_result((reply, started, time.monotonic()))

                try:
                    if started >= published + timeout:
                        raise TimeoutError("owned layer expired in queue")
                    work(publish)
                    if not ready:
                        raise RuntimeError("owned job did not publish READY")
                except BaseException as exc:
                    with self._lock:
                        self._errors.append(exc)
                    if not ready and public.set_running_or_notify_cancel():
                        public.set_exception(exc)
                finally:
                    with self._lock:
                        self._live -= 1

            try:
                owned = self._pool.submit(run)
            except BaseException:
                self._live -= 1
                raise
            self._owned.add(owned)
        # Register outside lock: an already-completed future calls inline.
        def retired(future):
            with self._lock:
                self._owned.discard(future)
        owned.add_done_callback(retired)
        return public

    def close(self):
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=False)
        with self._lock:
            if self._live or self._owned:
                raise RuntimeError("owned cleanup did not drain")
            if self._errors:
                raise RuntimeError("owned layer cleanup failed") from self._errors[0]

    def snapshot(self):
        with self._lock:
            return dict(capacity=self._capacity, live=self._live,
                        failures=len(self._errors), closed=self._closed)
