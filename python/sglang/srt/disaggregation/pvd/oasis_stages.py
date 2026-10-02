"""Bounded, joined layer stages with thread-affine persistent resources.

Future cancellation never abandons a published CUDA/native owner. A failed job
must run its explicit drainage hook before its future becomes terminal. Native
UNKNOWN retention remains the transport's responsibility.
"""
from concurrent.futures import Future
from dataclasses import dataclass, field
import queue
import threading
import time

from sglang.srt.disaggregation.pvd.oasis_pipeline import LayerReply


@dataclass
class _Job:
    key: object
    ticket: object
    payload: object
    published: float
    deadline: float
    future: Future
    cleanup: object
    queued: float
    phases: list = field(default_factory=list)


class BoundedLayerStages:
    names = ('search', 'delivery', 'install')

    def __init__(self, callbacks, *, initialize, retire, max_pending=56,
                 workers=(2, 2, 1)):
        if (len(callbacks) != 3 or not all(callable(f) for f in callbacks)
                or not callable(initialize) or not callable(retire)
                or type(max_pending) is not int or max_pending <= 0
                or len(workers) != 3 or any(type(n) is not int or n <= 0 for n in workers)):
            raise ValueError('three bounded stages and owned initialization/retirement required')
        self.callbacks, self.initialize, self.retire = callbacks, initialize, retire
        self.max_pending, self.worker_counts = max_pending, tuple(workers)
        self._lock, self._close_lock = threading.Lock(), threading.Lock()
        self._queues = [queue.Queue(max_pending) for _ in self.names]
        self._threads, self._pending, self._seen = [], {}, set()
        self._active = [0, 0, 0]
        self._peak = [0, 0, 0]
        self._accepted = self._completed = self._failed = self._peak_pending = 0
        self._retired = [0, 0, 0]
        self._retirement_errors, self._retained_states, self.trace = [], [], []
        self.closing = self.closed = False
        for phase, count in enumerate(workers):
            threads = [threading.Thread(target=self._worker, args=(phase, index),
                name=f'oasis-{self.names[phase]}-{index}') for index in range(count)]
            self._threads.append(threads)
            for thread in threads:
                thread.start()

    def submit(self, key, ticket, payload, *, published, timeout, cleanup):
        if timeout <= 0 or not callable(cleanup):
            raise ValueError('publication deadline and explicit failure drainage required')
        now = time.monotonic()
        with self._lock:
            if self.closing or self.closed:
                raise RuntimeError('layer stages admission closed')
            if key in self._seen or len(self._pending) >= self.max_pending:
                raise RuntimeError('duplicate staged job or bounded admission exceeded')
            future = Future()
            # This future owns already-published CUDA inputs, even while queued.
            future.set_running_or_notify_cancel()
            job = _Job(key, ticket, payload, published, published + timeout,
                       future, cleanup, now)
            self._pending[key] = job
            self._seen.add(key)
            self._accepted += 1
            self._peak_pending = max(self._peak_pending, len(self._pending))
            self._queues[0].put_nowait(job)
            return future

    def _finish(self, job, *, error=None, reply=None):
        completed = time.monotonic()
        if error is None and completed >= job.deadline:
            error = TimeoutError('layer publication deadline expired before READY')
        if error is not None:
            try:
                job.cleanup(job.payload)
            except BaseException as drainage:
                error = RuntimeError('staged job drainage unproven; retain owners')
                error.__cause__ = drainage
            completed = time.monotonic()
        record = dict(key=str(job.key), step=job.ticket.step, layer=job.ticket.layer,
            bootstrap=job.key[0] if isinstance(job.key, tuple) and type(job.key[0]) is bool else None,
            published=job.published, deadline=job.deadline, phases=job.phases,
            terminal=completed, failed=error is not None)
        with self._lock:
            self._pending.pop(job.key)
            self._completed += int(error is None)
            self._failed += int(error is not None)
            self.trace.append(record)
        if error is None:
            job.future.set_result((reply, job.phases[0]['start'], completed))
        else:
            job.future.set_exception(error)

    def _worker(self, phase, index):
        state = None
        try:
            while True:
                job = self._queues[phase].get()
                if job is None:
                    return
                started = time.monotonic()
                record = dict(stage=self.names[phase], worker=index,
                    thread=threading.get_ident(), queued=job.queued, start=started)
                job.phases.append(record)
                with self._lock:
                    self._active[phase] += 1
                    self._peak[phase] = max(self._peak[phase], self._active[phase])
                try:
                    if started >= job.deadline:
                        raise TimeoutError('layer publication deadline expired before stage')
                    if state is None:
                        state = self.initialize(self.names[phase], index)
                    value = self.callbacks[phase](job.payload, state)
                    ended = time.monotonic()
                    record['end'] = ended
                    if ended >= job.deadline:
                        raise TimeoutError('layer publication deadline expired during stage')
                    if phase == 2:
                        if not isinstance(value, LayerReply) or value.ticket != job.ticket:
                            raise RuntimeError('stale or foreign staged layer reply')
                        self._finish(job, reply=value)
                    else:
                        job.payload = value
                        job.queued = ended
                        self._queues[phase + 1].put_nowait(job)
                except BaseException as error:
                    record.setdefault('end', time.monotonic())
                    self._finish(job, error=error)
                finally:
                    with self._lock:
                        self._active[phase] -= 1
        finally:
            if state is not None:
                try:
                    self.retire(self.names[phase], state)
                    with self._lock:
                        self._retired[phase] += 1
                except BaseException as error:
                    with self._lock:
                        self._retained_states.append(state)
                        self._retirement_errors.append(error)

    def snapshot(self):
        with self._lock:
            return dict(closing=self.closing, closed=self.closed,
                max_pending=self.max_pending, pending=len(self._pending),
                peak_pending=self._peak_pending, accepted=self._accepted,
                completed=self._completed, failed=self._failed,
                workers=dict(zip(self.names, self.worker_counts)),
                peak_active=dict(zip(self.names, self._peak)),
                retired_workers=dict(zip(self.names, self._retired)),
                retained_states=len(self._retained_states),
                retirement_errors=len(self._retirement_errors))

    def close(self):
        with self._close_lock:
            with self._lock:
                if self.closed:
                    return
                if self._retirement_errors:
                    raise RuntimeError('retain undrained persistent stage owners')
                self.closing = True
            # Join upstream before closing downstream. Accepted owners always
            # advance to their terminal stage or explicit failure drainage.
            for queues, threads in zip(self._queues, self._threads):
                for _ in threads:
                    queues.put(None)
                for thread in threads:
                    thread.join()
            with self._lock:
                if self._pending or self._retirement_errors:
                    raise RuntimeError('retain undrained persistent stage owners')
                self.closed = True
