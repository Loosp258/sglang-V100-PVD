"""Explicit caller-owned search workspace for immutable four-head IP graphs.

Experimental API: caller charges ``retained_bytes`` and consumes/copies returned
views before the next call. It does not alter serving defaults or search width.
"""

import importlib
import threading
import time

import numpy as np
import torch

from .index_search import IndexCompletionUnknown, IndexSearchError


class FilteredSearchWorkspace:
    def __init__(self, backend, index, bitsets, *, num_queries, top_k):
        self.backend, self.index = backend, index
        self.rows, self.dim = index.count, index.dim
        self.num_queries, self.top_k = num_queries, top_k
        self.lock = threading.Lock()
        self.closed = False
        self.unknown = None
        self.pending_queries = None
        self.pending_host_queries = None
        with backend._lock:
            self._check_owner()
            if index.metric != "ip" or len(bitsets) != 4:
                raise IndexSearchError("workspace requires four-head IP search")
            backend.search_footprint(index.count, index.dim, num_queries, top_k)
            for bits in bitsets:
                if (not isinstance(bits, torch.Tensor) or bits.dtype != torch.uint32
                        or bits.device != backend.device or bits.ndim != 1
                        or bits.numel() != (index.count + 31) // 32
                        or not bits.is_contiguous()):
                    raise IndexSearchError("invalid search workspace filter")
            # Own immutable filter snapshots, not mutable caller aliases.
            self.bitsets = tuple(bits.clone() for bits in bitsets)
            self.neighbors = torch.empty((4, num_queries, top_k),
                device=backend.device, dtype=torch.uint32)
            self.scores = torch.empty((4, num_queries, top_k),
                device=backend.device, dtype=torch.float32)
            runtime = backend.runtime
            self.params = runtime.cagra.SearchParams(itopk_size=backend.itopk_size)
            self.itopk_size = backend.itopk_size
            filters = getattr(runtime, "filters", None)
            if filters is None:
                filters = importlib.import_module("cuvs.neighbors.filters")
            self.filters = tuple(filters.from_bitset(runtime.cp.from_dlpack(b))
                                 for b in self.bitsets)
            self.retained_bytes = sum(t.numel() * t.element_size()
                for t in (*self.bitsets, self.neighbors, self.scores))

    def _check_owner(self):
        self.backend._check()
        owner = self.index.handle
        if (self.closed or self.backend._owners.get(id(owner)) is not owner
                or owner.disposed or self.index.count != owner.rows
                or self.index.count != self.rows):
            raise IndexSearchError("closed, stale or disposed search workspace")

    def _check_request(self, queries, heads, *, device):
        self._check_owner()
        if (not isinstance(heads, tuple) or not 1 <= len(heads) <= 4
                or any(type(head) is not int or not 0 <= head < 4 for head in heads)
                or len(set(heads)) != len(heads)):
            raise IndexSearchError("unique native head subset required")
        if self.itopk_size != self.backend.itopk_size:
            raise IndexSearchError("search width changed after workspace creation")
        if (not isinstance(queries, torch.Tensor)
                or queries.shape != (len(heads), self.num_queries, self.dim)
                or queries.dtype != torch.float32
                or queries.device != device
                or not queries.is_contiguous()):
            raise IndexSearchError("invalid batched query shape/device/dtype")

    def search(self, queries, *, heads=(0, 1, 2, 3), timings=None):
        started = time.perf_counter() if timings is not None else 0.0
        with self.lock, self.backend._lock:
            if timings is not None:
                timings['workspace_lock'] = time.perf_counter() - started
            self._check_request(queries, heads, device=self.backend.device)
            rows, scores, _ = self._search_locked(queries, heads, timings)
            return rows, scores

    def search_host(self, queries, *, heads=(0, 1, 2, 3), timings=None):
        """Validate an owned CPU snapshot, then search its identical device copy.

        This entry point offers no externally supplied validation proof or
        unchecked option. The snapshot is private until native completion;
        caller mutations cannot change the values whose finiteness was proved.
        The returned device queries keep float32 score restoration unchanged.
        """
        started = time.perf_counter() if timings is not None else 0.0
        with self.lock, self.backend._lock:
            if timings is not None:
                timings['workspace_lock'] = time.perf_counter() - started
            self._check_request(queries, heads, device=torch.device('cpu'))
            started = time.perf_counter() if timings is not None else 0.0
            snapshot = queries.detach().clone()
            if timings is not None:
                timings['host_snapshot'] = time.perf_counter() - started
                started = time.perf_counter()
            # This reduction reads the exact owned float32 bytes copied below;
            # no CUDA reduction or GPU-to-host scalar synchronization is needed.
            if not bool(np.isfinite(snapshot.numpy()).all()):
                raise IndexSearchError("CAGRA vectors/queries must be finite")
            if timings is not None:
                timings['finite_proof'] = time.perf_counter() - started
            return self._search_locked(snapshot, heads, timings, host_snapshot=snapshot)

    def _search_locked(self, queries, heads, timings, *, host_snapshot=None):
        runtime, owner = self.backend.runtime, self.index.handle
        self.pending_host_queries = host_snapshot
        self.pending_queries = queries if host_snapshot is None else None
        try:
            if host_snapshot is not None:
                started = time.perf_counter() if timings is not None else 0.0
                # Default synchronous copy also handles pinned CPU snapshots;
                # sources remain owned until the conservative native fence.
                queries = host_snapshot.to(self.backend.device).contiguous()
                self.pending_queries = queries
                self.backend._matrix(queries.view(-1, self.dim), check_finite=False)
                if timings is not None:
                    timings['query_place'] = time.perf_counter() - started
            else:
                # Ordinary device callers still prove finite Q on the device.
                started = time.perf_counter() if timings is not None else 0.0
                self.backend._matrix(queries.view(-1, self.dim))
                if timings is not None:
                    timings['finite_proof'] = time.perf_counter() - started
            started = time.perf_counter() if timings is not None else 0.0
            with runtime.scope(owner):
                if timings is not None:
                    timings['native_scope'] = time.perf_counter() - started
                    started = time.perf_counter()
                for position, head in enumerate(heads):
                    runtime.cagra.search(self.params, owner.native,
                        runtime.cp.from_dlpack(queries[position]), self.top_k,
                        neighbors=runtime.cp.from_dlpack(self.neighbors[head]),
                        distances=runtime.cp.from_dlpack(self.scores[head]),
                        filter=self.filters[head], resources=owner.resources)
                if timings is not None:
                    timings['native_submit'] = time.perf_counter() - started
                    started = time.perf_counter()
                # Preserve the validated device-wide completion contract.
                runtime.synchronize()
                if timings is not None:
                    timings['native_completion'] = time.perf_counter() - started
        except BaseException as exc:
            try:
                runtime.synchronize()
            except BaseException as cleanup:
                self.unknown = str(cleanup)
                self.backend._unknown = str(cleanup)
                # Native work/copies may still reference every workspace buffer,
                # the private host source and the corresponding device queries.
                retained = getattr(self.backend, "_search_quarantine", [])
                retained.append(self)
                self.backend._search_quarantine = retained
                raise IndexCompletionUnknown(str(cleanup)) from exc
            self.pending_queries = self.pending_host_queries = None
            raise
        self.pending_queries = self.pending_host_queries = None
        return self.neighbors, self.scores, queries

    def close(self):
        with self.lock, self.backend._lock:
            if self.unknown or self.backend._unknown:
                raise IndexCompletionUnknown("search workspace completion is unknown")
            self.closed = True
            self.filters = self.bitsets = ()
            self.neighbors = self.scores = None
