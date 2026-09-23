"""Bounded, owner-polled discovery of the Gateway-selected V for CUDA Decode.

Only control-plane HTTP runs in the background. A discovered route never
allocates a D destination, installs a bank or makes a request runnable. The
serving factory must still assemble and claim the request explicitly.
"""

import threading
from concurrent.futures import Future
from dataclasses import dataclass

from sglang.srt.disaggregation.pvd.conn import PVDSelectedRouteBinding
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError


@dataclass
class _Record:
    req: object
    identity: tuple
    future: object
    binding: PVDSelectedRouteBinding | None = None
    abandoned: bool = False


class CUDARouteDiscoveryQueue:
    """One bounded route lookup per eligible waiting Req, never a queue barrier."""

    def __init__(self, manager, *, max_inflight):
        if type(max_inflight) is not int or max_inflight <= 0:
            raise LifecycleError("positive selected-route admission bound required")
        self.manager = manager
        self.max_inflight = max_inflight
        self._owner_thread = threading.get_ident()
        self._records = {}
        self._closed = False

    def _owner(self):
        if threading.get_ident() != self._owner_thread:
            raise LifecycleError("selected-route discovery requires its owner thread")

    def _identity(self, req):
        return (
            req.rid,
            self.manager.key_for(req),
            self.manager.vector_group_for(req),
            req.pvd_delivery_id,
        )

    @staticmethod
    def _finished(req):
        finished = getattr(req, "finished", None)
        return (callable(finished) and finished()) or getattr(
            req, "is_retracted", False
        )

    def _abandon(self, record):
        if not record.abandoned:
            record.abandoned = True
            # A run_coroutine_threadsafe Future can report cancelled before
            # its underlying HTTP coroutine has actually stopped. Do not
            # mistake cancellation for drain or reuse the bounded slot yet.

    def poll(self, waiting_reqs):
        """Return failures for the Scheduler to abort; never wait on HTTP."""
        self._owner()
        waiting = {id(req): req for req in waiting_reqs}
        failures, failed = [], set()
        for req_id, record in tuple(self._records.items()):
            live = waiting.get(req_id) is record.req and not self._finished(record.req)
            if live:
                try:
                    live = self._identity(record.req) == record.identity
                except Exception:
                    live = False
                if not live:
                    failures.append((record.req, "selected V request identity changed"))
                    failed.add(req_id)
            if not live or self._closed:
                self._abandon(record)
            if record.future is None:
                if record.abandoned:
                    self._records.pop(req_id)
                continue
            if not record.future.done():
                continue
            try:
                binding = record.future.result()
                if not record.abandoned and (
                    not isinstance(binding, PVDSelectedRouteBinding)
                    or binding.manager is not self.manager
                    or binding.req is not record.req
                    or (
                        binding.rid,
                        binding.key,
                        binding.group_id,
                        binding.delivery_id,
                    )
                    != record.identity
                ):
                    raise LifecycleError("selected V route result changed identity")
            except Exception as exc:
                if not record.abandoned:
                    failures.append((record.req, str(exc)))
                    failed.add(req_id)
                self._records.pop(req_id)
                continue
            if record.abandoned:
                self._records.pop(req_id)
            else:
                record.binding, record.future = binding, None

        if self._closed:
            return failures
        for req in waiting_reqs:
            req_id = id(req)
            if (
                req_id in self._records
                or req_id in failed
                or self._finished(req)
                or len(self._records) >= self.max_inflight
                or not self.manager.bootstrap_runnable(req)
            ):
                continue
            try:
                identity = self._identity(req)
                # A cancelled old HTTP call can still be running. Do not start
                # a successor with the same rid until that call terminates.
                if any(r.identity[0] == identity[0] for r in self._records.values()):
                    continue
                future = self.manager.start_selected_cuda_routes(req)
                if not isinstance(future, Future):
                    raise LifecycleError("selected V discovery must return a Future")
            except Exception as exc:
                failures.append((req, str(exc)))
                failed.add(req_id)
                continue
            self._records[req_id] = _Record(req, identity, future)
        return failures

    def ready_for(self, req):
        self._owner()
        record = self._records.get(id(req))
        if record is None or record.req is not req or record.abandoned:
            return None
        if self._identity(req) != record.identity:
            raise LifecycleError("selected V request identity changed")
        return record.binding

    def close(self):
        """Stop admission; caller polls until every HTTP lookup settles."""
        self._owner()
        self._closed = True
        self.poll(())
        return not self._records

    @property
    def pending(self):
        self._owner()
        return bool(self._records)
