"""Worker-owned private prefix caches for concurrent PVD prediction.

The cache keeps its own request marker, never a formal Decode Req or KV row.
Only the prediction worker may prepare or retire it. A different incarnation,
including reuse of the same request id, retires both private caches first.
"""

import threading
from types import SimpleNamespace

from .concurrent_prediction_worker import PredictionJob


class ConcurrentPrefixCacheOwner:
    def __init__(self, probe, provider):
        self.probe = probe
        self.provider = provider
        self._owner_thread_id = None
        self._identity = None
        self._request = None
        self._quarantined = False

    def _owner(self):
        thread_id = threading.get_ident()
        if self._owner_thread_id is None:
            self._owner_thread_id = thread_id
        elif self._owner_thread_id != thread_id:
            raise RuntimeError("private prefix caches belong to one worker thread")
        if self._quarantined:
            raise RuntimeError("private prefix cache owner is quarantined")

    def prepare(self, job: PredictionJob):
        self._owner()
        if not isinstance(job, PredictionJob):
            raise TypeError("immutable prediction job required")
        identity = (job.request_id, job.incarnation)
        if self._identity == identity:
            return
        if self._identity is not None:
            self.retire()

        # This marker is private to the worker. It cannot keep a formal Req or
        # its allocator mapping alive after Scheduler retirement.
        request = SimpleNamespace(rid=job.request_id)
        self._identity, self._request = identity, request
        try:
            if self.probe.prefix_budget is not None:
                self.probe.register_cached_request(request)
            if self.provider.factory.prefix_cache_enabled:
                self.provider.set_sidecar_cache_identity(*identity)
        except BaseException:
            # An allocation or fence may already have happened. The worker
            # quarantines callback failures; do not retry or drop ownership.
            self._quarantined = True
            raise

    def retire(self):
        self._owner()
        if self._identity is None:
            return
        try:
            if self.provider.factory.prefix_cache_enabled:
                self.provider.retire_sidecar_cache(if_identity=self._identity)
            if self.probe.prefix_budget is not None:
                self.probe.retire_cached_request(self._request)
        except BaseException:
            self._quarantined = True
            raise
        self._identity = self._request = None
