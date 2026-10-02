"""Explicit caller-owned search workspace for immutable four-head IP graphs.

Experimental API: caller charges ``retained_bytes`` and consumes/copies returned
views before the next call. It does not alter serving defaults or search width.
"""

import importlib
import threading

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

    def search(self, queries, *, heads=(0, 1, 2, 3)):
        with self.lock, self.backend._lock:
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
                    or queries.device != self.backend.device
                    or not queries.is_contiguous()):
                raise IndexSearchError("invalid batched query shape/device/dtype")
            # One finite-Q proof instead of one host wait per filtered call.
            self.backend._matrix(queries.view(-1, self.dim))
            runtime, owner = self.backend.runtime, self.index.handle
            self.pending_queries = queries
            try:
                with runtime.scope(owner):
                    for position, head in enumerate(heads):
                        runtime.cagra.search(self.params, owner.native,
                            runtime.cp.from_dlpack(queries[position]), self.top_k,
                            neighbors=runtime.cp.from_dlpack(self.neighbors[head]),
                            distances=runtime.cp.from_dlpack(self.scores[head]),
                            filter=self.filters[head], resources=owner.resources)
                    # Preserve the validated device-wide completion contract.
                    runtime.synchronize()
            except BaseException as exc:
                try:
                    runtime.synchronize()
                except BaseException as cleanup:
                    self.unknown = str(cleanup)
                    self.backend._unknown = str(cleanup)
                    # Native work may still reference all workspace buffers.
                    retained = getattr(self.backend, "_search_quarantine", [])
                    retained.append(self)
                    self.backend._search_quarantine = retained
                    raise IndexCompletionUnknown(str(cleanup)) from exc
                self.pending_queries = None
                raise
            self.pending_queries = None
            return self.neighbors, self.scores

    def close(self):
        with self.lock, self.backend._lock:
            if self.unknown or self.backend._unknown:
                raise IndexCompletionUnknown("search workspace completion is unknown")
            self.closed = True
            self.filters = self.bitsets = ()
            self.neighbors = self.scores = None
