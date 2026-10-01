"""Bounded request-local, one-token, per-layer lookahead for experiments.

Transport and device ownership stay with the callback. A completed reply must
carry the exact ticket. Closing drains running callbacks before releasing their
captured input owners; this component does not install Scheduler KV banks.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from time import monotonic


@dataclass(frozen=True)
class LayerTicket:
    request_id: str
    incarnation: str
    step: int
    layer: int


@dataclass(frozen=True)
class LayerReply:
    ticket: LayerTicket
    value: object


def select_resident(candidates, resident, *, capacity, max_new):
    """Keep ranked resident hits; admit bounded misses; fill with old residents.

    Inputs and output are token IDs for one KV head. Resident order is newest
    selection first, so old unselected IDs are evicted last. No duplicate rows.
    """
    if capacity <= 0 or not 0 <= max_new <= capacity:
        raise ValueError("positive capacity and bounded max_new required")
    resident = tuple(resident)
    candidates = tuple(candidates)
    if len(set(resident)) != len(resident) or len(resident) > capacity:
        raise ValueError("unique bounded resident IDs required")
    if any(type(t) is not int or t < 0 for t in (*resident, *candidates)):
        raise ValueError("nonnegative integer token IDs required")
    old = set(resident)
    # Reserve room for every predicted/resident intersection before admitting
    # new IDs. Ranked misses must not evict a later ranked resident hit.
    hits = set(candidates) & old
    new_budget = min(max_new, capacity - len(hits))
    chosen, seen, admitted = [], set(), 0
    for token in candidates:
        if token in seen:
            continue
        if token not in old:
            if admitted == new_budget:
                continue
            admitted += 1
        chosen.append(token)
        seen.add(token)
        if len(chosen) == capacity:
            break
    for token in resident:
        if len(chosen) == capacity:
            break
        if token not in seen:
            chosen.append(token)
            seen.add(token)
    return tuple(chosen)


class LayerLookahead:
    """Publish (t+1, layer) during t; wait only when that layer is consumed."""

    def __init__(self, request_id, incarnation, *, layers, workers=2, timeout=60):
        if not request_id or not incarnation or layers <= 0 or workers <= 0 or timeout <= 0:
            raise ValueError("explicit request, incarnation and positive bounds required")
        self.request_id, self.incarnation = request_id, incarnation
        self.layers, self.timeout = layers, timeout
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="oasis-layer")
        self._lock = Lock()
        self._closed = False
        self._pending = {}
        self._published = [-1] * layers
        self._consumed = [-1] * layers
        self.trace = []

    def publish(self, step, layer, callback):
        if type(step) is not int or type(layer) is not int or step < 0 or not 0 <= layer < self.layers:
            raise ValueError("valid step/layer required")
        with self._lock:
            if self._closed:
                raise RuntimeError("lookahead closed")
            if step != self._published[layer] + 1 or layer in self._pending:
                raise RuntimeError("duplicate, gap or unconsumed layer prediction")
            ticket = LayerTicket(self.request_id, self.incarnation, step, layer)
            published = monotonic()

            def work():
                started = monotonic()
                reply = callback(ticket)
                if not isinstance(reply, LayerReply) or reply.ticket != ticket:
                    raise RuntimeError("stale or foreign layer reply")
                return reply, started, monotonic()

            future = self._pool.submit(work)
            self._pending[layer] = (ticket, future, published)
            self._published[layer] = step
            return ticket

    def consume(self, step, layer):
        with self._lock:
            if self._closed:
                raise RuntimeError("lookahead closed")
            item = self._pending.get(layer)
            if item is None or item[0].step != step or step != self._consumed[layer] + 1:
                raise RuntimeError("missing, replayed or out-of-order layer consume")
        ticket, future, published = item
        wait_start = monotonic()
        # The deadline starts at publication, including worker queueing.
        remaining = self.timeout - (wait_start - published)
        if remaining <= 0:
            raise TimeoutError("layer lookahead expired")
        reply, started, completed = future.result(timeout=remaining)
        consumed = monotonic()
        with self._lock:
            if self._closed or self._pending.get(layer) is not item:
                raise RuntimeError("layer ownership changed during consumption")
            del self._pending[layer]
            self._consumed[layer] = step
            self.trace.append({"step": step, "layer": layer, "published": published,
                "worker_start": started, "ready": completed, "consumed": consumed,
                "queue_seconds": started - published,
                "service_seconds": completed - started,
                "consumer_wait_seconds": consumed - wait_start,
                "ready_before_consume": completed <= wait_start})
        return reply.value

    def close(self):
        with self._lock:
            self._closed = True
        # Do not abandon callbacks holding queries, banks or device copy owners.
        self._pool.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            errors = [f.exception() for _, f, _ in self._pending.values()
                      if not f.cancelled() and f.exception() is not None]
            self._pending.clear()
        return tuple(errors)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
