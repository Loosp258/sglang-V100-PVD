"""Experimental batched exact KNN edge maintenance for immutable Prompt K.

This is a new adjacency-update algorithm, not native ``cagra.extend``. Native
CAGRA continues to own the searchable index and execute filtered searches.
All tensor storage is reserved before creation using the final Prompt length.
"""

import threading
import time

import torch
from sglang.srt.disaggregation.pvd.cagra_backend import CagraIndexBackend
from sglang.srt.disaggregation.pvd.index_search import (
    BuiltIndex,
    IndexCompletionUnknown,
    IndexSearchError,
)

_CAPTURE_LOCK = threading.Lock()


class KVGraphBuffers:
    degree = 16

    @staticmethod
    def retained_bytes(capacity_rows, dim, *, tile_rows=128):
        heads, n, k = 4, capacity_rows // 4, 16
        # Head-major and chunk-major K; exact neighbors/scores; native and
        # mapped edges; row-ID table; tiled GEMM and selection workspaces.
        return (
            capacity_rows * (dim * 8 + k * (8 + 4 + 4 + 8) + 8)
            + heads * tile_rows * n * 4
            + heads * tile_rows * (k * (4 + 8) * 3 + k * 8)
            + (capacity_rows + n) * 8
        )

    def __init__(
        self,
        graphs,
        capacity_rows,
        dim,
        device,
        *,
        tile_rows=128,
        routing_edges=0,
        small_tail_max_rows=0,
        small_tail_prune=False,
        fused_core_block=0,
        fused_core_rows=1,
        fused_selection=False,
        cuda_graph_min_rows=0,
        fused_prepare=False,
        reuse_scores=False,
        fused_edge_write=False,
        new_top16=False,
        prepared_tail=False,
        ahead_capture=False,
        profile_gpu=False,
        stream_completion=False,
        initial_rows=0,
    ):
        self.graphs, self.capacity, self.dim = graphs, capacity_rows // 4, dim
        self.heads, self.tile, self.count = graphs * 4, tile_rows, 0
        self.routing_edges = routing_edges
        self.small_tail_max_rows = small_tail_max_rows
        self.small_tail_prune = small_tail_prune
        self.fused_core_block = fused_core_block
        self.fused_core_rows = fused_core_rows
        self.fused_selection = fused_selection
        self.cuda_graph_min_rows = cuda_graph_min_rows
        self.fused_prepare = fused_prepare
        self.reuse_scores = reuse_scores
        self.fused_edge_write = fused_edge_write
        self.new_top16 = new_top16
        self.prepared_tail = prepared_tail
        self.ahead_capture = ahead_capture
        self.profile_gpu = profile_gpu
        # One event per owned batch, reused under the backend lock. Prime its
        # CUDA handle during the initial build, before final KV can arrive.
        self._completion_event = torch.cuda.Event() if stream_completion else None
        self.last_reused_scores = False
        self.last_replayed = False
        self.last_gpu_seconds = 0.0
        self.last_update_gpu_seconds = 0.0
        self.last_completion_wait_seconds = 0.0
        self.ahead_capture_seconds = 0.0
        self._graph_range = None
        self.last_prepare_batched = False
        self.capture_seconds = 0.0
        self._cuda_graph = None
        self._capture_stream = None
        self._capture_warm = False
        if cuda_graph_min_rows or ahead_capture:
            self._capture_stream = torch.cuda.Stream(device=device)
        if fused_core_block and (dim != 128 or torch.device(device).type != "cuda"):
            raise IndexSearchError("KV fused core requires CUDA and head dimension 128")
        h, n, d, r, k = self.heads, self.capacity, dim, tile_rows, self.degree

        def empty(shape, dtype=torch.float32):
            return torch.empty(shape, device=device, dtype=dtype)

        self.head_data = empty((h, n, d))
        self.native_data = empty((graphs, 4 * n, d))
        self.neighbors = empty((h, n, k), torch.int64)
        self.top_scores = empty((h, n, k))
        self.native_graph = empty((graphs, 4 * n, k), torch.int32)
        self.mapped = empty((h, n, k), torch.int64)
        self.ids = empty((h, n), torch.int64)
        self.work = empty((h, r, n))
        self.cross_scores = empty((h, n, r)) if reuse_scores else None
        self.new_top16_workspace = (
            empty((h, r, (n + 255) // 256, 16), torch.uint64) if new_top16 else None
        )
        self.tail_old = initial_rows
        self.tail_raw = self.tail_centered = self.tail_groups = None
        if prepared_tail and 0 < initial_rows < n:
            self.tail_raw = empty((h, n - initial_rows, d))
            self.tail_centered = empty((h, n - initial_rows, d))
            self.tail_groups = tuple(
                self.tail_centered.view(graphs, 4 * (n - initial_rows), d).unbind(0)
            )
        self._gpu_start = self._gpu_end = self._update_start = None
        if profile_gpu:
            self._gpu_start = torch.cuda.Event(enable_timing=True)
            self._gpu_end = torch.cuda.Event(enable_timing=True)
            self._update_start = torch.cuda.Event(enable_timing=True)
        self.tmp_scores = empty((h, r, k))
        self.tmp_ids = empty((h, r, k), torch.int64)
        self.merge_scores = empty((h, r, 2 * k))
        self.merge_ids = empty((h, r, 2 * k), torch.int64)
        self.selected = empty((h, r, k), torch.int64)
        self.row_ids = torch.arange(4 * n, device=device, dtype=torch.int64)
        self.tokens = torch.arange(n, device=device, dtype=torch.int64)
        if fused_selection:
            from sglang.srt.disaggregation.pvd.cagra_kv_select_write import prewarm

            prewarm(device, routing_edges=routing_edges, max_rows=n)
        if new_top16:
            from sglang.srt.disaggregation.pvd.cagra_kv_new_top16 import prewarm

            prewarm(device, max_rows=n)

    def append(self, items):
        """Compute only new-new and old-new scores; reuse old-old top-16."""
        if len(items) != self.graphs or not items:
            raise IndexSearchError("KV graph update requires the complete graph batch")
        delta = len(items[0]) // 4
        old, total = self.count, self.count + delta
        if delta <= 0 or total > self.capacity or total <= self.degree:
            raise IndexSearchError(
                "KV graph update exceeds capacity or has too few rows"
            )
        for data in items:
            if (
                data.shape != (4 * delta, self.dim)
                or data.dtype != torch.float32
                or data.device != self.head_data.device
                or not data.is_contiguous()
            ):
                raise IndexSearchError("KV graph batch has mismatched tensors")
        joined = None
        if self.profile_gpu:
            self._update_start.record(torch.cuda.current_stream(self.head_data.device))
        if self.fused_prepare:
            from sglang.srt.disaggregation.pvd.cagra_kv_prepare import (
                contiguous_batch_view,
                prepare,
            )

            joined = contiguous_batch_view(items)
        self.last_prepare_batched = joined is not None
        if joined is not None:
            prepare(self, joined, old, delta)
        else:
            for group, data in enumerate(items):
                begin, end = 4 * group, 4 * (group + 1)
                self.head_data[begin:end, old:total].copy_(
                    data.view(4, delta, self.dim)
                )
                self.native_data[group, 4 * old : 4 * total].copy_(data)
                self.ids[begin:end, old:total].copy_(
                    self.row_ids[4 * old : 4 * total].view(4, delta)
                )
        if self.profile_gpu:
            self._gpu_start.record(torch.cuda.current_stream(self.head_data.device))
        self.last_replayed = self._graph_range == (old, total)
        if self.last_replayed:
            with torch.cuda.device(self.head_data.device):
                self._cuda_graph.replay()
            self.capture_seconds = 0.0
            self.last_reused_scores = bool(self.reuse_scores and delta <= self.tile)
        elif self.cuda_graph_min_rows and delta >= self.cuda_graph_min_rows:
            self._capture_compute(old, total)
        else:
            self.capture_seconds = 0.0
            self._compute(old, total)
        if self.profile_gpu:
            self._gpu_end.record(torch.cuda.current_stream(self.head_data.device))
        self.count = total

    def prime_update_completion(self):
        if self._completion_event is not None:
            self._completion_event.record(
                torch.cuda.current_stream(self.head_data.device)
            )

    def wait_update_completion(self):
        # Fixed-view extension submits only Torch/Triton operations. append()
        # joins any capture side stream onto this current stream. Native import
        # and view replacement are drained separately during initial build.
        with torch.cuda.device(self.head_data.device):
            self._completion_event.record(
                torch.cuda.current_stream(self.head_data.device)
            )
            self._completion_event.synchronize()

    def _capture_compute(self, old, total, *, replay=True):
        # Capture only computation over owned preallocated buffers. Input PUT
        # proofs, validation and native index publication remain outside.
        device = self.head_data.device
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(device):
            current = torch.cuda.current_stream(device)
            side = self._capture_stream
            side.wait_stream(current)
            started = time.perf_counter()
            with _CAPTURE_LOCK, torch.cuda.stream(side):
                if not self._capture_warm:
                    rows = min(self.tile, total - old)
                    torch.bmm(
                        self.head_data[:, old : old + rows],
                        self.head_data[:, :total].transpose(1, 2),
                        out=self.work[:, :rows, :total],
                    )
                    self._capture_warm = True
                side.synchronize()
                # Retain the graph even if recording or a later native action
                # fails. The backend quarantines the complete batch on unknown.
                pool = self._cuda_graph.pool() if self._cuda_graph is not None else None
                prior = self._cuda_graph
                self._cuda_graph = graph
                graph.capture_begin(pool=pool, capture_error_mode="thread_local")
                try:
                    self._compute(old, total)
                finally:
                    graph.capture_end()
                # Prior work was drained by the preceding backend operation;
                # its pool can be reused. Keep the prior owner until recording
                # is finished, including the exception path.
                del prior
            self.capture_seconds = time.perf_counter() - started
            self._graph_range = (old, total)
            if replay:
                graph.replay()
            current.wait_stream(side)

    def prepare_final_capture(self):
        if self.ahead_capture and 0 < self.count < self.capacity:
            # Warm cuBLAS only on owned initialized storage. Recording does
            # not execute the future update or read the Entry's unread pages.
            self.head_data[:, self.count :].zero_()
            self._capture_compute(self.count, self.capacity, replay=False)
            self.ahead_capture_seconds = self.capture_seconds
            self.capture_seconds = 0.0

    def read_gpu_times(self):
        # Caller has already proved completion through its normal fence.
        if self.profile_gpu:
            self.last_gpu_seconds = self._gpu_start.elapsed_time(self._gpu_end) / 1000
            self.last_update_gpu_seconds = (
                self._update_start.elapsed_time(self._gpu_end) / 1000
            )

    def _compute(self, old, total):
        delta, k = total - old, self.degree
        self.last_reused_scores = False
        if (
            self.reuse_scores
            and old
            and delta <= self.tile
            and self.small_tail_max_rows >= delta
        ):
            self._compute_reusing_scores(old, total)
            self.last_reused_scores = True
            return
        if self.fused_core_block:
            from sglang.srt.disaggregation.pvd.cagra_kv_fused_core import update

            update(self, old, total)
            return
        if self.fused_selection:
            from sglang.srt.disaggregation.pvd.cagra_kv_select_write import select_write
        small_tail = bool(
            old and delta <= self.small_tail_max_rows and self.head_data.is_cuda
        )
        if small_tail:
            from sglang.srt.disaggregation.pvd.cagra_kv_small_tail import (
                merge_neighbors,
                write_graph,
            )
        # Reinterpret the existing score workspace when the comparison width
        # is small. This increases the row batch without allocating more memory.
        old_tile = (
            min(old, self.tile * self.capacity // delta)
            if small_tail or self.fused_selection
            else self.tile
        )
        # Existing rows: only a new point can displace an old exact neighbor.
        for begin in range(0, old, old_tile or self.tile):
            end = min(old, begin + (old_tile or self.tile))
            r, take = end - begin, min(k, delta)
            scores = (
                self.work.view(self.heads, -1)[:, : r * delta].view(
                    self.heads, r, delta
                )
                if small_tail or self.fused_selection
                else self.work[:, :r, :delta]
            )
            torch.bmm(
                self.head_data[:, begin:end],
                self.head_data[:, old:total].transpose(1, 2),
                out=scores,
            )
            if self.fused_selection:
                select_write(self, scores, begin, old, total, existing=True)
                continue
            if small_tail:
                merge_neighbors(self, scores, begin, old, delta)
                continue
            torch.topk(
                scores,
                take,
                dim=2,
                out=(self.tmp_scores[:, :r, :take], self.tmp_ids[:, :r, :take]),
            )
            self.tmp_ids[:, :r, :take].add_(old)
            merge = self.merge_scores[:, :r, : k + take]
            merge[:, :, :k].copy_(self.top_scores[:, begin:end])
            merge[:, :, k:].copy_(self.tmp_scores[:, :r, :take])
            candidates = self.merge_ids[:, :r, : k + take]
            candidates[:, :, :k].copy_(self.neighbors[:, begin:end])
            candidates[:, :, k:].copy_(self.tmp_ids[:, :r, :take])
            torch.topk(
                merge,
                k,
                dim=2,
                out=(self.top_scores[:, begin:end], self.selected[:, :r]),
            )
            torch.gather(
                candidates, 2, self.selected[:, :r], out=self.neighbors[:, begin:end]
            )
        # New rows: compare against the complete proven prefix, including peers.
        for begin in range(old, total, self.tile):
            end = min(total, begin + self.tile)
            r = end - begin
            scores = self.work[:, :r, :total]
            torch.bmm(
                self.head_data[:, begin:end],
                self.head_data[:, :total].transpose(1, 2),
                out=scores,
            )
            if self.fused_selection:
                select_write(self, scores, begin, old, total, existing=False)
                continue
            scores.scatter_(
                2,
                self.tokens[begin:end].view(1, r, 1).expand(self.heads, -1, -1),
                -float("inf"),
            )
            torch.topk(
                scores,
                k,
                dim=2,
                out=(self.top_scores[:, begin:end], self.neighbors[:, begin:end]),
            )
        if self.fused_selection:
            return
        if small_tail:
            write_graph(self, total)
            return
        torch.gather(
            self.ids[:, :total],
            1,
            self.neighbors[:, :total].view(self.heads, total * k),
            out=self.mapped[:, :total].view(self.heads, total * k),
        )
        if self.routing_edges:
            # Keep the exact top-16 separately. The searchable degree-16 graph
            # replaces its last two edges with a bidirectional token-ID ring,
            # ensuring each head remains strongly connected after every append.
            for edge, offset in enumerate((-1, 1)):
                target = (self.tokens[:total] + offset) % total
                self.mapped[:, :total, k - 2 + edge].copy_(self.ids[:, target])
        for group in range(self.graphs):
            section = slice(4 * group, 4 * (group + 1))
            self.native_graph[group].index_copy_(
                0,
                self.ids[section, :total].reshape(-1),
                self.mapped[section, :total].reshape(-1, k).to(torch.int32),
            )

    def _compute_reusing_scores(self, old, total):
        from sglang.srt.disaggregation.pvd.cagra_kv_small_tail import (
            merge_neighbors,
            write_graph,
        )

        delta = total - old
        scores = self.work[:, :delta, :total]
        torch.bmm(
            self.head_data[:, old:total],
            self.head_data[:, :total].transpose(1, 2),
            out=scores,
        )
        cross = self.cross_scores[:, :old, :delta]
        cross.copy_(scores[:, :, :old].transpose(1, 2))
        if self.fused_edge_write:
            from sglang.srt.disaggregation.pvd.cagra_kv_merge_edges import merge_edges

            merge_edges(self, cross, old, delta)
        else:
            merge_neighbors(self, cross, 0, old, delta)
        if self.new_top16:
            from sglang.srt.disaggregation.pvd.cagra_kv_new_top16 import select

            select(self, scores, old)
        else:
            scores.scatter_(
                2,
                self.tokens[old:total].view(1, delta, 1).expand(self.heads, -1, -1),
                -float("inf"),
            )
            torch.topk(
                scores,
                self.degree,
                dim=2,
                out=(self.top_scores[:, old:total], self.neighbors[:, old:total]),
            )
        if self.fused_edge_write:
            from sglang.srt.disaggregation.pvd.cagra_kv_merge_edges import write_new

            write_new(self, old, total)
        else:
            write_graph(self, total)


class CagraKVUpdateBackend(CagraIndexBackend):
    """Default-off exact edge updates; keep every head's IDs within its graph."""

    batched_kv_graph = True

    @property
    def shared_footprint(self):
        # Reserve backend-wide cuBLAS handle/workspace headroom for the worker
        # lifetime, alongside the existing bounded native parent reservation.
        return super().shared_footprint + 64 * 1024 * 1024

    def __init__(
        self,
        *,
        routing_edges=0,
        small_tail_max_rows=0,
        small_tail_prune=False,
        fused_core_block=0,
        fused_core_rows=1,
        fused_selection=False,
        cuda_graph_min_rows=0,
        fused_prepare=False,
        reuse_scores=False,
        fused_edge_write=False,
        new_top16=False,
        prepared_tail=False,
        ahead_capture=False,
        fixed_native_views=False,
        stream_completion=False,
        profile_gpu=False,
        **kwargs,
    ):
        if routing_edges not in (0, 2):
            raise ValueError("KV routing edges must be 0 or 2")
        self.routing_edges = routing_edges
        if type(small_tail_max_rows) is not int or not 0 <= small_tail_max_rows <= 512:
            raise ValueError("KV small-tail threshold must be between 0 and 512")
        self.small_tail_max_rows = small_tail_max_rows
        if type(small_tail_prune) is not bool or (
            small_tail_prune and not small_tail_max_rows
        ):
            raise ValueError(
                "KV small-tail pruning requires an enabled small-tail threshold"
            )
        self.small_tail_prune = small_tail_prune
        if type(fused_core_block) is not int or fused_core_block not in (
            0,
            32,
            64,
            128,
        ):
            raise ValueError("KV fused-core block must be 0, 32, 64 or 128")
        self.fused_core_block = fused_core_block
        if type(fused_core_rows) is not int or fused_core_rows not in (1, 16, 32):
            raise ValueError("KV fused-core rows must be 1, 16 or 32")
        self.fused_core_rows = fused_core_rows
        if type(fused_selection) is not bool:
            raise ValueError("KV fused selection must be boolean")
        if type(cuda_graph_min_rows) is not int or cuda_graph_min_rows < 0:
            raise ValueError("KV CUDA Graph threshold must be a nonnegative integer")
        if cuda_graph_min_rows and not fused_selection:
            raise ValueError("KV CUDA Graph capture requires fused selection")
        if fused_selection and fused_core_block:
            raise ValueError("KV fused selection and fused core are mutually exclusive")
        self.fused_selection = fused_selection
        self.cuda_graph_min_rows = cuda_graph_min_rows
        if type(fused_prepare) is not bool:
            raise ValueError("KV fused preparation must be boolean")
        self.fused_prepare = fused_prepare
        for name, value in (
            ("reuse_scores", reuse_scores),
            ("fused_edge_write", fused_edge_write),
            ("new_top16", new_top16),
            ("prepared_tail", prepared_tail),
            ("ahead_capture", ahead_capture),
            ("fixed_native_views", fixed_native_views),
            ("stream_completion", stream_completion),
            ("profile_gpu", profile_gpu),
        ):
            if type(value) is not bool:
                raise ValueError(f"KV {name} must be boolean")
            setattr(self, name, value)
        if ahead_capture and cuda_graph_min_rows:
            raise ValueError(
                "ahead capture and arrival-time capture are mutually exclusive"
            )
        if fused_edge_write and (not reuse_scores or not small_tail_max_rows):
            raise ValueError("selective edge writes require reused small-tail scores")
        if new_top16 and (not reuse_scores or not small_tail_max_rows):
            raise ValueError("new Top-16 requires reused small-tail scores")
        if stream_completion and not fixed_native_views:
            raise ValueError("stream completion requires fixed native views")
        super().__init__(**kwargs)
        if (
            reuse_scores
            or prepared_tail
            or ahead_capture
            or profile_gpu
            or stream_completion
        ) and self.device.type != "cuda":
            raise ValueError("planned KV tail options require CUDA")
        if self.exact_head_groups != 4 or self.graph_degree != 16:
            raise ValueError("KV edge update requires exact four-head degree-16 CAGRA")
        if self.fused_prepare:
            if self.device.type != "cuda":
                raise ValueError("KV fused preparation requires CUDA")
            from sglang.srt.disaggregation.pvd.cagra_kv_prepare import prewarm

            prewarm(self.device)
        if self.fused_selection:
            if self.device.type != "cuda":
                raise ValueError("KV fused selection requires CUDA")
            from sglang.srt.disaggregation.pvd.cagra_kv_select_write import prewarm

            prewarm(self.device, routing_edges=self.routing_edges)
        if self.fused_core_block:
            if self.device.type != "cuda":
                raise ValueError("KV fused core requires CUDA")
            from sglang.srt.disaggregation.pvd.cagra_kv_fused_core import prewarm

            prewarm(
                self.device,
                candidates=self.fused_core_block,
                rows=self.fused_core_rows,
                routing_edges=self.routing_edges,
            )
        validate = getattr(self.runtime, "require_graph_view_adapter", None)
        if callable(validate):
            validate()
        if self.small_tail_max_rows and self.device.type == "cuda":
            from sglang.srt.disaggregation.pvd.cagra_kv_small_tail import prewarm

            prewarm(
                self.device,
                max_rows=self.small_tail_max_rows,
                routing_edges=self.routing_edges,
                prune=self.small_tail_prune,
            )
        if self.fused_edge_write:
            from sglang.srt.disaggregation.pvd.cagra_kv_merge_edges import prewarm

            prewarm(
                self.device,
                prune=self.small_tail_prune,
                routing_edges=self.routing_edges,
            )
        if self.new_top16:
            from sglang.srt.disaggregation.pvd.cagra_kv_new_top16 import prewarm

            prewarm(self.device)

    def build_footprint(self, rows, dim, *, metric):
        return (
            super().build_footprint(rows, dim, metric=metric)
            + KVGraphBuffers.retained_bytes(rows, dim)
            + (8 * 1024 * 1024 if self.cuda_graph_min_rows else 0)
            + (rows * dim * 8 if self.prepared_tail else 0)
            + (rows * 128 * 4 if self.reuse_scores else 0)
            + (4 * 128 * ((rows // 4 + 255) // 256) * 16 * 8 if self.new_top16 else 0)
            + (8 * 1024 * 1024 if self.ahead_capture else 0)
            + (4096 if self.stream_completion else 0)
        )

    def build_scratch_footprint(self, rows, dim, *, metric):
        # A fused finite check spans up to fourteen adjacent groups. Charge
        # its full temporary masks/abs tensors, not just one group's flags.
        groups = 14 if self.fused_prepare else 1
        return groups * super().build_scratch_footprint(rows, dim, metric=metric) + 4096

    def build(self, *args, **kwargs):
        raise IndexSearchError("KV edge update requires build_many and final capacity")

    def extend(self, *args, **kwargs):
        raise IndexSearchError("KV edge update requires the complete extend_many batch")

    def _validate_batch(self, items):
        if not items or len(items) > 14:
            raise IndexSearchError("KV graph batches must contain 1 to 14 graphs")
        shape = items[0].shape
        for data in items:
            self._matrix(data, check_finite=False)
            if data.shape != shape or len(data) % 4 or data.shape[1] % 4:
                raise IndexSearchError(
                    "KV graph batch requires equal four-head aligned matrices"
                )
        joined = None
        if self.fused_prepare:
            from sglang.srt.disaggregation.pvd.cagra_kv_prepare import (
                contiguous_batch_view,
            )

            joined = contiguous_batch_view(items)
        finite = (
            torch.isfinite(joined).all()
            if joined is not None
            else torch.stack([torch.isfinite(data).all() for data in items]).all()
        )
        if not bool(finite):
            raise IndexSearchError("CAGRA vectors/queries must be finite")

    def build_many(self, items, *, vector_space, metric, capacity_rows):
        items = tuple(items)
        with self._lock:
            self._check()
            self._validate_batch(items)
            if self.fused_core_block and items[0].shape[1] != 128:
                raise IndexSearchError("KV fused core requires head dimension 128")
            if len(items[0]) // 4 <= 16:
                raise IndexSearchError("each KV head needs more than degree-16 rows")
            if (
                metric != "ip"
                or not isinstance(vector_space, str)
                or not vector_space.strip()
            ):
                raise IndexSearchError(
                    "KV graph update requires explicit IP vector space"
                )
            if (
                type(capacity_rows) is not int
                or capacity_rows < len(items[0])
                or capacity_rows % 4
                or capacity_rows >= 2**31
            ):
                raise IndexSearchError("invalid KV graph final capacity")
            self._shape(len(items[0]), items[0].shape[1], metric)
            batch = KVGraphBuffers(
                len(items),
                capacity_rows,
                items[0].shape[1],
                self.device,
                routing_edges=self.routing_edges,
                small_tail_max_rows=self.small_tail_max_rows,
                small_tail_prune=self.small_tail_prune,
                fused_core_block=self.fused_core_block,
                fused_core_rows=self.fused_core_rows,
                fused_selection=self.fused_selection,
                cuda_graph_min_rows=self.cuda_graph_min_rows,
                fused_prepare=self.fused_prepare,
                reuse_scores=self.reuse_scores,
                fused_edge_write=self.fused_edge_write,
                new_top16=self.new_top16,
                prepared_tail=self.prepared_tail,
                ahead_capture=self.ahead_capture,
                profile_gpu=self.profile_gpu,
                stream_completion=self.stream_completion,
                initial_rows=len(items[0]) // 4,
            )
            owners = [self.runtime.create(self.cap) for _ in items]
            for group, owner in enumerate(owners):
                owner.auxiliary = (batch, group)
                self._owners[id(owner)] = owner
            try:
                batch.append(items)
                for group, owner in enumerate(owners):
                    self.runtime.import_owned_graph(
                        owner,
                        batch.native_data[group, : 4 * batch.count],
                        batch.native_graph[group, : 4 * batch.count],
                    )
                self.runtime.synchronize()
                batch.read_gpu_times()
                if self.fixed_native_views:
                    for group, owner in enumerate(owners):
                        self.runtime.replace_owned_graph_views(
                            owner, batch.native_data[group], batch.native_graph[group]
                        )
                batch.prepare_final_capture()
                batch.prime_update_completion()
                if self.fixed_native_views or self.ahead_capture:
                    self.runtime.synchronize()
            except BaseException as exc:
                self._unknown = str(exc)
                raise IndexCompletionUnknown(
                    f"KV graph build state is unknown: {exc}"
                ) from exc
            for owner in owners:
                owner.rows = len(items[0])
            return [
                BuiltIndex(vector_space, metric, items[0].shape[1], owner.rows, owner)
                for owner in owners
            ]

    def extend_many(self, items):
        items = tuple(items)
        with self._lock:
            self._check()
            self._validate_batch(tuple(data for _, data in items))
            seen, batch = set(), None
            for group, (index, data) in enumerate(items):
                owner = index.handle
                if self._owners.get(id(owner)) is not owner or owner.disposed:
                    raise IndexSearchError("unknown or disposed KV graph index")
                if index.count != owner.rows:
                    raise IndexSearchError("stale KV graph count after update")
                if id(owner) in seen:
                    raise IndexSearchError("KV graph update requires distinct indexes")
                seen.add(id(owner))
                current, position = owner.auxiliary
                batch = current if batch is None else batch
                if (
                    current is not batch
                    or position != group
                    or index.dim != data.shape[1]
                ):
                    raise IndexSearchError(
                        "KV graph update requires its ordered complete batch"
                    )
            if (
                len(items) != batch.graphs
                or batch.count + len(items[0][1]) // 4 > batch.capacity
            ):
                raise IndexSearchError(
                    "KV graph batch exceeds capacity or is incomplete"
                )
            try:
                batch.append(tuple(data for _, data in items))
                if not self.fixed_native_views:
                    for group, (index, _) in enumerate(items):
                        self.runtime.replace_owned_graph_views(
                            index.handle,
                            batch.native_data[group, : 4 * batch.count],
                            batch.native_graph[group, : 4 * batch.count],
                        )
                fence_started = time.perf_counter() if self.profile_gpu else 0.0
                if self.stream_completion:
                    batch.wait_update_completion()
                else:
                    self.runtime.synchronize()
                if self.profile_gpu:
                    batch.last_completion_wait_seconds = (
                        time.perf_counter() - fence_started
                    )
                batch.read_gpu_times()
            except BaseException as exc:
                self._unknown = str(exc)
                raise IndexCompletionUnknown(
                    f"KV graph update state is unknown: {exc}"
                ) from exc
            result = []
            for index, _ in items:
                index.handle.rows = 4 * batch.count
                result.append(
                    BuiltIndex(
                        index.vector_space,
                        index.metric,
                        index.dim,
                        index.handle.rows,
                        index.handle,
                    )
                )
            return result
