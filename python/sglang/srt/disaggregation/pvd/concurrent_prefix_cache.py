"""Worker-owned private prefix caches for concurrent PVD prediction.

The cache keeps its own request marker, never a formal Decode Req or KV row.
Only the prediction worker may prepare or retire it. A different incarnation,
including reuse of the same request id, retires both private caches first.
"""

import threading
from collections import OrderedDict
from types import SimpleNamespace

from .concurrent_prediction_worker import PredictionJob


class ConcurrentPrefixCacheOwner:
    def __init__(self, probe, provider, *, max_cached_requests=1):
        if type(max_cached_requests) is not int or max_cached_requests <= 0:
            raise ValueError("positive private target cache request bound required")
        self.probe = probe
        self.provider = provider
        self.max_cached_requests = max_cached_requests
        self._owner_thread_id = None
        self._draft_identity = None
        self._target_requests = OrderedDict()
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
        try:
            if self.probe.prefix_budget is not None:
                previous = self._target_requests.get(job.request_id)
                if previous is not None and previous[0] != identity:
                    self._retire_target(job.request_id)
                    previous = None
                if previous is None:
                    if len(self._target_requests) >= self.max_cached_requests:
                        self._retire_target(next(iter(self._target_requests)))
                    # This marker cannot keep a formal Req or its allocator
                    # mapping alive after Scheduler retirement.
                    request = SimpleNamespace(rid=job.request_id)
                    self._target_requests[job.request_id] = (identity, request)
                    self.probe.register_cached_request(request)
                self._target_requests.move_to_end(job.request_id)
            if (
                self.provider.factory.prefix_cache_enabled
                and self._draft_identity != identity
            ):
                if self._draft_identity is not None:
                    self.provider.retire_sidecar_cache(
                        if_identity=self._draft_identity
                    )
                self._draft_identity = identity
                self.provider.set_sidecar_cache_identity(*identity)
        except BaseException:
            # An allocation or fence may already have happened. The worker
            # quarantines callback failures; do not retry or drop ownership.
            self._quarantined = True
            raise

    def _retire_target(self, request_id):
        _, request = self._target_requests[request_id]
        self.probe.retire_cached_request(request)
        del self._target_requests[request_id]

    def retire(self):
        self._owner()
        try:
            if self._draft_identity is not None:
                self.provider.retire_sidecar_cache(
                    if_identity=self._draft_identity
                )
                self._draft_identity = None
            for request_id in tuple(self._target_requests):
                self._retire_target(request_id)
        except BaseException:
            self._quarantined = True
            raise
