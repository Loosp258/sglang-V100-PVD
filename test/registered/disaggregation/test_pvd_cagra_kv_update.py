"""Exact edge maintenance, append-only native IDs and batch failure gates."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.cagra_backend import _NativeIndex
from sglang.srt.disaggregation.pvd.cagra_kv_update import (
    CagraKVUpdateBackend,
    KVGraphBuffers,
)
from sglang.srt.disaggregation.pvd.index_search import (
    IndexCompletionUnknown,
    IndexSearchError,
)


@pytest.mark.parametrize("chunks", [(32, 3, 12, 17), (48, 16), (64,)])
def test_edges_match_full_exact_knn_after_every_arrival(chunks):
    generator = torch.Generator().manual_seed(17)
    data = torch.randn((8, 64, 8), generator=generator)
    buffers = KVGraphBuffers(2, 4 * 64, 8, "cpu", tile_rows=19)
    old = 0
    pointers = [
        value.data_ptr()
        for value in vars(buffers).values()
        if isinstance(value, torch.Tensor)
    ]
    for delta in chunks:
        total = old + delta
        items = [
            data[4 * group : 4 * (group + 1), old:total].reshape(-1, 8).contiguous()
            for group in range(2)
        ]
        buffers.append(items)
        scores = data[:, :total] @ data[:, :total].transpose(1, 2)
        scores.diagonal(dim1=1, dim2=2).fill_(-float("inf"))
        values, neighbors = scores.topk(16, dim=2)
        assert torch.equal(buffers.neighbors[:, :total], neighbors)
        torch.testing.assert_close(buffers.top_scores[:, :total], values)
        for head in range(8):
            group = head // 4
            rows = buffers.ids[head, :total]
            graph = buffers.native_graph[group, rows].long()
            assert torch.equal(graph, buffers.ids[head, :total][neighbors[head]])
            assert torch.equal(buffers.native_data[group, rows], data[head, :total])
        assert pointers == [
            value.data_ptr()
            for value in vars(buffers).values()
            if isinstance(value, torch.Tensor)
        ]
        old = total
    actual_bytes = sum(
        value.numel() * value.element_size()
        for value in vars(buffers).values()
        if isinstance(value, torch.Tensor)
    )
    assert actual_bytes <= 2 * KVGraphBuffers.retained_bytes(256, 8, tile_rows=19)


class Runtime:
    device = torch.device("cpu")
    supports_extend = True

    def create(self, cap):
        return _NativeIndex(cap)

    def import_owned_graph(self, owner, data, graph):
        owner.native = SimpleNamespace(trained=True, dim=data.shape[1])
        owner.vectors, owner.graph = data, graph

    def replace_owned_graph_views(self, owner, data, graph):
        self.view_calls = getattr(self, "view_calls", 0) + 1
        if getattr(self, "fail", False):
            raise RuntimeError("native view update has unknown completion")
        owner.vectors, owner.graph = data, graph

    def search(self, owner, queries, *, top_k, itopk_size, bitset=None):
        scores = queries @ owner.vectors.T
        if bitset is not None:
            ids = torch.arange(len(owner.vectors))
            allowed = (bitset.long()[ids // 32] >> (ids % 32)) & 1
            scores[:, allowed == 0] = -float("inf")
        values, ids = scores.topk(top_k, dim=1)
        return ids.to(torch.uint32), values

    def extend(self, *args):
        raise AssertionError("KV edge update must not call native extend")

    def synchronize(self):
        self.sync_calls = getattr(self, "sync_calls", 0) + 1

    def dispose(self, owner):
        owner.native = owner.vectors = owner.graph = owner.auxiliary = None
        owner.disposed = True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA selective adjacency")
@pytest.mark.parametrize(
    "chunks", [(32, 3, 12, 17), (2048, 111), (2112, 47), (1792, 256, 111), (64, 129)]
)
@pytest.mark.parametrize("ring", [0, 2])
@pytest.mark.parametrize("prune", [False, True])
@pytest.mark.parametrize("ties", [False, True])
def test_selective_merge_edges_preserves_every_graph_byte(chunks, ring, prune, ties):
    from sglang.srt.disaggregation.pvd.cagra_kv_merge_edges import prewarm

    prewarm("cuda:0", prune=prune, routing_edges=ring)
    total = sum(chunks)
    data = torch.randn(8, total, 8, device="cuda:0")
    if ties:
        data.fill_(0)
        data[:, ::2].fill_(-0.0)
    options = dict(
        small_tail_max_rows=512,
        small_tail_prune=prune,
        routing_edges=ring,
        reuse_scores=True,
    )
    reference = KVGraphBuffers(2, total * 4, 8, "cuda:0", **options)
    candidate = KVGraphBuffers(
        2, total * 4, 8, "cuda:0", fused_edge_write=True, ahead_capture=True, **options
    )
    old = 0
    for step, delta in enumerate(chunks):
        end = old + delta
        items = [
            data[4 * g : 4 * (g + 1), old:end].reshape(-1, 8).contiguous()
            for g in range(2)
        ]
        reference.append(items)
        candidate.append(items)
        torch.cuda.synchronize()
        assert torch.equal(
            reference.native_graph[:, : 4 * end], candidate.native_graph[:, : 4 * end]
        )
        assert torch.equal(reference.mapped[:, :end], candidate.mapped[:, :end])
        assert torch.equal(reference.neighbors[:, :end], candidate.neighbors[:, :end])
        assert torch.equal(reference.top_scores[:, :end], candidate.top_scores[:, :end])
        if step == 0:
            candidate.prepare_final_capture()
        old = end


def test_fixed_views_keep_logical_proven_count_and_bind_only_once():
    runtime = Runtime()
    backend = CagraKVUpdateBackend(
        device="cpu",
        _runtime=runtime,
        native_bytes_per_index=4096,
        itopk_size=64,
        exact_head_groups=4,
        graph_degree=16,
        intermediate_degree=16,
        fixed_native_views=True,
    )
    data = torch.randn(4, 64, 8)
    indexes = backend.build_many(
        [data[:, :48].reshape(-1, 8).contiguous()],
        vector_space="test",
        metric="ip",
        capacity_rows=256,
    )
    assert indexes[0].count == indexes[0].handle.rows == 192
    assert len(indexes[0].handle.vectors) == 256
    indexes = backend.extend_many(
        [(indexes[0], data[:, 48:].reshape(-1, 8).contiguous())]
    )
    assert indexes[0].count == 256
    assert runtime.view_calls == 1
    backend.dispose(indexes[0])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA capture and reused GEMM"
)
@pytest.mark.parametrize("chunks", [(2048, 111), (2112, 47), (1792, 256, 111)])
@pytest.mark.parametrize("reuse", [False, True])
def test_ahead_capture_preserves_prefix_and_replays_only_matching_final_range(
    chunks, reuse
):
    total = sum(chunks)
    data = torch.randn(8, total, 128, device="cuda")
    options = dict(small_tail_max_rows=512, routing_edges=2, reuse_scores=reuse)
    reference = KVGraphBuffers(2, 4 * total, 128, "cuda", **options)
    captured = KVGraphBuffers(
        2,
        4 * total,
        128,
        "cuda",
        ahead_capture=True,
        profile_gpu=True,
        prepared_tail=True,
        initial_rows=chunks[0],
        **options,
    )
    old = 0
    for step, delta in enumerate(chunks):
        end = old + delta
        items = [
            data[4 * g : 4 * (g + 1), old:end].reshape(-1, 128).contiguous()
            for g in range(2)
        ]
        reference.append(items)
        captured.append(items)
        torch.cuda.synchronize()
        captured.read_gpu_times()
        assert captured.last_update_gpu_seconds > 0
        assert captured.last_replayed == (step == 1 and len(chunks) == 2)
        assert torch.equal(
            reference.native_graph[:, : 4 * end], captured.native_graph[:, : 4 * end]
        )
        torch.testing.assert_close(
            reference.top_scores[:, :end], captured.top_scores[:, :end], rtol=0, atol=0
        )
        if step == 0:
            prefix = captured.neighbors[:, :end].clone()
            captured.prepare_final_capture()
            torch.cuda.synchronize()
            assert captured.count == end
            assert torch.equal(prefix, captured.neighbors[:, :end])
        old = end
    # Compare against exhaustive exact scores, allowing float-rounding ties.
    scores = data @ data.transpose(1, 2)
    scores.diagonal(dim1=1, dim2=2).fill_(-float("inf"))
    torch.testing.assert_close(
        captured.top_scores, scores.topk(16, dim=2).values, atol=3e-5, rtol=1e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA small-tail kernels")
@pytest.mark.parametrize("routing_edges", [0, 2])
@pytest.mark.parametrize("prune", [False, True])
@pytest.mark.parametrize(
    "chunks",
    [(32, 1, 3, 12, 16), (48, 16), (1792, 367), (2048, 111), (1536, 512), (64, 513)],
)
def test_small_tail_preserves_exact_cache_mapping_and_ring(
    chunks, routing_edges, prune
):
    total = sum(chunks)
    data = torch.randn(8, total, 8, device="cuda")
    reference = KVGraphBuffers(2, 4 * total, 8, "cuda", routing_edges=routing_edges)
    optimized = KVGraphBuffers(
        2,
        4 * total,
        8,
        "cuda",
        routing_edges=routing_edges,
        small_tail_max_rows=512,
        small_tail_prune=prune,
    )
    pointers = {
        name: value.data_ptr()
        for name, value in vars(optimized).items()
        if isinstance(value, torch.Tensor)
    }
    old = 0
    for delta in chunks:
        end = old + delta
        items = [
            data[4 * g : 4 * (g + 1), old:end].reshape(-1, 8).contiguous()
            for g in range(2)
        ]
        reference.append(items)
        optimized.append(items)
        torch.cuda.synchronize()
        assert pointers == {
            name: value.data_ptr()
            for name, value in vars(optimized).items()
            if isinstance(value, torch.Tensor)
        }
        torch.testing.assert_close(
            optimized.top_scores[:, :end], reference.top_scores[:, :end]
        )
        # Floating-point ties may choose different valid exact neighbors. Check
        # each selected neighbor's score, uniqueness, head mapping and ring.
        for head in range(8):
            selected = optimized.neighbors[head, :end]
            expected = optimized.top_scores[head, :end]
            scores = (data[head, :end, None] * data[head, selected]).sum(dim=-1)
            torch.testing.assert_close(scores, expected, atol=1e-5, rtol=1e-5)
            assert bool((selected.sort(dim=-1).values.diff(dim=-1) > 0).all())
            mapped = optimized.ids[head, :end][selected]
            if routing_edges:
                mapped[:, -2] = optimized.ids[head, :end].roll(1)
                mapped[:, -1] = optimized.ids[head, :end].roll(-1)
            graph = optimized.native_graph[head // 4, optimized.ids[head, :end]].long()
            assert torch.equal(graph, mapped)
        old = end


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pruning total order")
def test_pruned_merge_preserves_ties_signed_zero_and_changed_rows():
    from sglang.srt.disaggregation.pvd.cagra_kv_small_tail import merge_neighbors

    full = KVGraphBuffers(1, 4 * 19, 8, "cuda")
    pruned = KVGraphBuffers(1, 4 * 19, 8, "cuda", small_tail_prune=True)
    prior = (
        torch.arange(16, 0, -1, dtype=torch.float32, device="cuda")
        .expand(4, 4, 16)
        .clone()
    )
    prior[:, 3] = -0.0
    prior[:, 2] = 1.0
    scores = torch.zeros(4, 4, 3, device="cuda")
    scores[:, 1] = 100.0
    scores[:, 2] = 1.0
    for buffers in (full, pruned):
        buffers.top_scores[:, :4].copy_(prior)
        buffers.neighbors[:, :4].copy_(torch.arange(16, device="cuda"))
        buffers.neighbors[:, 2].copy_(torch.arange(15, -1, -1, device="cuda"))
        merge_neighbors(buffers, scores, 0, 16, 3)
    torch.cuda.synchronize()
    assert torch.equal(full.neighbors[:, :4], pruned.neighbors[:, :4])
    assert torch.equal(
        full.top_scores[:, :4].view(torch.int32),
        pruned.top_scores[:, :4].view(torch.int32),
    )


@pytest.mark.parametrize("threshold", [-1, 513, True])
def test_small_tail_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="small-tail threshold"):
        CagraKVUpdateBackend(small_tail_max_rows=threshold)


@pytest.mark.parametrize("block", [-1, 16, 256, True])
def test_fused_core_rejects_invalid_block(block):
    with pytest.raises(ValueError, match="fused-core block"):
        CagraKVUpdateBackend(fused_core_block=block)


@pytest.mark.parametrize("threshold", [-1, True, 1.0])
def test_cuda_graph_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="CUDA Graph threshold"):
        CagraKVUpdateBackend(fused_selection=True, cuda_graph_min_rows=threshold)


def test_cuda_graph_requires_selection_and_selection_requires_cuda():
    with pytest.raises(ValueError, match="requires fused selection"):
        CagraKVUpdateBackend(cuda_graph_min_rows=1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        CagraKVUpdateBackend(fused_selection=True, fused_core_block=32)
    with pytest.raises(ValueError, match="requires CUDA"):
        CagraKVUpdateBackend(
            device="cpu",
            fused_selection=True,
            _runtime=Runtime(),
            exact_head_groups=4,
            graph_degree=16,
            native_bytes_per_index=4096,
            intermediate_degree=16,
            itopk_size=32,
        )


def test_contiguous_batch_view_checks_exact_span_offsets_and_order():
    pytest.importorskip("triton")
    from sglang.srt.disaggregation.pvd.cagra_kv_prepare import contiguous_batch_view

    backing = torch.full((8 + 2 * 128 * 8 + 8,), float("nan"))
    source = backing[8:-8].view(256, 8)
    source.fill_(1)
    items = source.split(128)
    joined = contiguous_batch_view(items)
    assert torch.equal(joined, source)
    assert joined.data_ptr() == items[0].data_ptr()
    assert bool(torch.isfinite(joined).all())
    assert contiguous_batch_view(items[::-1]) is None
    assert contiguous_batch_view((items[0], items[0])) is None
    assert contiguous_batch_view((items[0], items[1].clone())) is None
    assert contiguous_batch_view((items[0], items[1][:64])) is None


def test_fused_prepare_cli_defaults_off_and_requires_kv_backend():
    from sglang.srt.disaggregation.pvd.server import _validate_args
    from test_pvd_cagra_backend import args

    assert args().prompt_index_cagra_fused_prepare is False
    with pytest.raises(ValueError, match="fused preparation requires"):
        _validate_args(args("--prompt-index-cagra-fused-prepare"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA batch preparation")
def test_batched_preparation_equals_per_group_for_every_arrival_and_fallback():
    torch.manual_seed(31)
    data = torch.randn(8, 2159, 8, device="cuda")
    optimized = KVGraphBuffers(
        2, 4 * 2159, 8, "cuda", routing_edges=2, fused_prepare=True
    )
    reference = KVGraphBuffers(2, 4 * 2159, 8, "cuda", routing_edges=2)
    old = 0
    for total in (1792, 2048, 2159):
        chunk = data[:, old:total].reshape(-1, 8).contiguous()
        items = chunk.split(4 * (total - old))
        if old == 2048:
            items = [item.clone() for item in items]
        reference.append(items)
        optimized.append(items)
        torch.cuda.synchronize()
        assert optimized.last_prepare_batched == (old != 2048)
        for name in (
            "head_data",
            "native_data",
            "ids",
            "neighbors",
            "top_scores",
            "native_graph",
        ):
            left, right = getattr(optimized, name), getattr(reference, name)
            # Compare initialized storage only; unused capacity is unspecified.
            if name in ("native_data", "native_graph"):
                assert torch.equal(left[:, : 4 * total], right[:, : 4 * total])
            else:
                assert torch.equal(left[:, :total], right[:, :total])
        old = total


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA batch finite proof")
def test_batched_finite_proof_rejects_last_group_before_graph_mutation():
    runtime = Runtime()
    runtime.device = torch.device("cuda:0")
    runtime.synchronize = torch.cuda.synchronize
    owner = CagraKVUpdateBackend(
        device="cuda:0",
        native_bytes_per_index=4096,
        graph_degree=16,
        intermediate_degree=16,
        itopk_size=32,
        exact_head_groups=4,
        _runtime=runtime,
        fused_prepare=True,
    )
    from sglang.srt.disaggregation.pvd.cagra_backend import CagraIndexBackend

    one_group = CagraIndexBackend.build_scratch_footprint(owner, 256, 8, metric="ip")
    assert owner.build_scratch_footprint(256, 8, metric="ip") >= 14 * one_group
    source = torch.randn(256, 8, device="cuda:0")
    indexes = owner.build_many(
        source.split(128), vector_space="test", metric="ip", capacity_rows=256
    )
    batch = indexes[0].handle.auxiliary[0]
    previous = batch.native_graph[:, :128].clone()
    tail = torch.randn(128, 8, device="cuda:0")
    tail[-1, -1] = float("nan")
    with pytest.raises(IndexSearchError, match="finite"):
        owner.extend_many(list(zip(indexes, tail.split(64))))
    assert batch.count == 32
    assert torch.equal(batch.native_graph[:, :128], previous)
    assert all(index.count == 128 for index in indexes)
    for index in indexes:
        owner.dispose(index)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices")
def test_fused_prepare_guards_buffer_device():
    with torch.cuda.device(0):
        values = torch.randn(4, 63, 8, device="cuda:1")
        buffers = KVGraphBuffers(1, 4 * 63, 8, "cuda:1", fused_prepare=True)
        for begin, end in ((0, 48), (48, 63)):
            buffers.append([values[:, begin:end].reshape(-1, 8).contiguous()])
            torch.cuda.synchronize(1)
            assert buffers.last_prepare_batched
            assert torch.cuda.current_device() == 0
        assert torch.equal(buffers.head_data, values)
        for head in range(4):
            assert torch.equal(buffers.native_data[0, buffers.ids[head]], values[head])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA selection/graph")
@pytest.mark.parametrize("capture_threshold", [0, 1, 256])
@pytest.mark.parametrize("chunks", [(32, 1, 3, 12, 16), (1792, 367), (2048, 111)])
def test_fused_selection_and_first_capture_preserve_cache_and_graph(
    chunks, capture_threshold
):
    total = sum(chunks)
    torch.manual_seed(29)
    data = torch.randn(4, total, 8, device="cuda")
    buffers = KVGraphBuffers(
        1,
        4 * total,
        8,
        "cuda",
        routing_edges=2,
        fused_selection=True,
        cuda_graph_min_rows=capture_threshold,
    )
    pointers = {
        name: value.data_ptr()
        for name, value in vars(buffers).items()
        if isinstance(value, torch.Tensor)
    }
    old = 0
    for delta in chunks:
        end = old + delta
        buffers.append([data[:, old:end].reshape(-1, 8).contiguous()])
        torch.cuda.synchronize()
        scores = data[:, :end] @ data[:, :end].transpose(1, 2)
        scores.diagonal(dim1=1, dim2=2).fill_(-float("inf"))
        expected = scores.topk(16, dim=2).values
        selected = buffers.neighbors[:, :end]
        torch.testing.assert_close(
            scores.gather(2, selected), expected, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(buffers.top_scores[:, :end], expected)
        assert bool((selected.sort(dim=2).values.diff(dim=2) > 0).all())
        for head in range(4):
            mapped = buffers.ids[head, :end][selected[head]]
            mapped[:, -2] = buffers.ids[head, :end].roll(1)
            mapped[:, -1] = buffers.ids[head, :end].roll(-1)
            assert torch.equal(
                buffers.native_graph[0, buffers.ids[head, :end]].long(), mapped
            )
        assert pointers == {
            name: value.data_ptr()
            for name, value in vars(buffers).items()
            if isinstance(value, torch.Tensor)
        }
        assert bool(buffers.capture_seconds) == bool(
            capture_threshold and delta >= capture_threshold
        )
        old = end


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices")
def test_cuda_graph_selection_guards_device_and_retains_graph():
    with torch.cuda.device(0):
        values = torch.randn(4, 63, 8, device="cuda:1")
        buffers = KVGraphBuffers(
            1, 4 * 63, 8, "cuda:1", fused_selection=True, cuda_graph_min_rows=1
        )
        for begin, end in ((0, 48), (48, 63)):
            buffers.append([values[:, begin:end].reshape(-1, 8).contiguous()])
            torch.cuda.synchronize(1)
            assert buffers._cuda_graph is not None
            assert torch.cuda.current_device() == 0
        scores = values @ values.transpose(1, 2)
        scores.diagonal(dim1=1, dim2=2).fill_(-float("inf"))
        torch.testing.assert_close(buffers.top_scores, scores.topk(16, dim=2).values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fused core")
@pytest.mark.parametrize("rows", [1, 16, 32])
@pytest.mark.parametrize("block", [32, 64, 128])
def test_fused_core_cache_native_ids_ring_and_no_reallocation(rows, block):
    torch.manual_seed(27)
    data = torch.randn(4, 113, 128, device="cuda")
    buffers = KVGraphBuffers(
        1,
        4 * 113,
        128,
        "cuda",
        routing_edges=2,
        fused_core_block=block,
        fused_core_rows=rows,
    )
    pointers = {
        name: value.data_ptr()
        for name, value in vars(buffers).items()
        if isinstance(value, torch.Tensor)
    }
    old = 0
    for total in (63, 96, 113):
        buffers.append([data[:, old:total].reshape(-1, 128).contiguous()])
        scores = data[:, :total] @ data[:, :total].transpose(1, 2)
        scores.diagonal(dim1=1, dim2=2).fill_(-float("inf"))
        wanted = scores.topk(16, dim=2).values
        neighbors = buffers.neighbors[:, :total]
        torch.testing.assert_close(
            scores.gather(2, neighbors), wanted, atol=1e-4, rtol=1e-5
        )
        torch.testing.assert_close(
            buffers.top_scores[:, :total], wanted, atol=1e-4, rtol=1e-5
        )
        assert bool((neighbors.sort(dim=2).values.diff(dim=2) > 0).all())
        for head in range(4):
            mapped = buffers.ids[head, :total][neighbors[head]]
            mapped[:, -2] = buffers.ids[head, :total].roll(1)
            mapped[:, -1] = buffers.ids[head, :total].roll(-1)
            assert torch.equal(
                buffers.native_graph[0, buffers.ids[head, :total]].long(), mapped
            )
        assert pointers == {
            name: value.data_ptr()
            for name, value in vars(buffers).items()
            if isinstance(value, torch.Tensor)
        }
        old = total


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices")
def test_fused_core_launch_guards_tensor_device():
    with torch.cuda.device(0):
        values = torch.randn(4, 63, 128, device="cuda:1")
        buffers = KVGraphBuffers(
            1, 4 * 63, 128, "cuda:1", fused_core_block=32, fused_core_rows=16
        )
        buffers.append([values.reshape(-1, 128)])
        torch.cuda.synchronize(1)
        assert torch.cuda.current_device() == 0
        scores = values @ values.transpose(1, 2)
        scores.diagonal(dim1=1, dim2=2).fill_(-float("inf"))
        torch.testing.assert_close(
            buffers.top_scores, scores.topk(16, dim=2).values, atol=1e-4, rtol=1e-5
        )


def test_small_tail_cli_defaults_off_and_requires_kv_edge_update():
    from sglang.srt.disaggregation.pvd.server import _validate_args
    from test_pvd_cagra_backend import args

    assert args().prompt_index_cagra_small_tail_max_rows == 0
    with pytest.raises(ValueError, match="small-tail kernels require"):
        _validate_args(args("--prompt-index-cagra-small-tail-max-rows", "512"))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices")
def test_small_tail_launch_uses_buffer_device_and_restores_thread_device():
    with torch.cuda.device(0):
        values = torch.randn(4, 64, 8, device="cuda:1")
        buffers = KVGraphBuffers(
            1, 256, 8, "cuda:1", routing_edges=2, small_tail_max_rows=512
        )
        for begin, end in ((0, 48), (48, 64)):
            buffers.append([values[:, begin:end].reshape(-1, 8).contiguous()])
        torch.cuda.synchronize(1)
        assert torch.cuda.current_device() == 0
        scores = values @ values.transpose(1, 2)
        scores.diagonal(dim1=1, dim2=2).fill_(-float("inf"))
        torch.testing.assert_close(buffers.top_scores, scores.topk(16, dim=2).values)


def test_single_graph_api_rejects_silent_native_fallback():
    value = backend()
    with pytest.raises(IndexSearchError, match="build_many"):
        value.build(torch.zeros((128, 8)))
    with pytest.raises(IndexSearchError, match="extend_many"):
        value.extend(None, torch.zeros((128, 8)))


def backend():
    return CagraKVUpdateBackend(
        device="cpu",
        native_bytes_per_index=4096,
        graph_degree=16,
        intermediate_degree=16,
        itopk_size=32,
        exact_head_groups=4,
        _runtime=Runtime(),
    )


def test_invalid_or_partial_batch_is_rejected_before_mutating_any_graph():
    owner = backend()
    old = owner.build_many(
        [torch.randn(128, 8) for _ in range(2)],
        vector_space="test",
        metric="ip",
        capacity_rows=256,
    )
    with pytest.raises(IndexSearchError, match="incomplete"):
        owner.extend_many([(old[0], torch.randn(64, 8))])
    bad = torch.ones((64, 8))
    bad[0, 0] = float("nan")
    with pytest.raises(IndexSearchError, match="finite"):
        owner.extend_many([(old[0], torch.randn(64, 8)), (old[1], bad)])
    assert old[0].handle.auxiliary[0].count == 32
    new = owner.extend_many([(index, torch.randn(64, 8)) for index in old])
    assert [index.count for index in new] == [192, 192]
    with pytest.raises(IndexSearchError, match="stale"):
        owner.extend_many([(index, torch.randn(64, 8)) for index in old])
    for index in new:
        owner.dispose(index)
    assert not owner._owners


def test_failure_retains_storage_and_does_not_publish_new_counts():
    owner = backend()
    old = owner.build_many(
        [torch.randn(128, 8) for _ in range(2)],
        vector_space="test",
        metric="ip",
        capacity_rows=256,
    )
    owner.runtime.fail = True
    with pytest.raises(IndexCompletionUnknown, match="unknown completion"):
        owner.extend_many([(index, torch.randn(64, 8)) for index in old])
    assert [index.handle.rows for index in old] == [128, 128]
    assert len(owner._owners) == 2
    assert all(index.handle.auxiliary is not None for index in old)
    with pytest.raises(IndexCompletionUnknown):
        owner.synchronize()


def test_routing_ring_preserves_append_ids_and_connects_every_token():
    buffers = KVGraphBuffers(1, 4 * 64, 8, "cpu", routing_edges=2)
    for delta in (32, 16, 16):
        buffers.append([torch.randn(4 * delta, 8)])
        total = buffers.count
        for head in range(4):
            rows = buffers.ids[head, :total]
            edges = buffers.native_graph[0, rows].long()
            assert torch.equal(edges[:, -2], rows.roll(1))
            assert torch.equal(edges[:, -1], rows.roll(-1))
            assert set(edges.flatten().tolist()) <= set(rows.tolist())


@pytest.mark.parametrize("fixed_views", [False, True])
def test_manager_reserves_final_capacity_and_publishes_only_after_partial_final_page(
    fixed_views,
):
    from sglang.srt.disaggregation.pvd.index_lifecycle import IndexState
    from sglang.srt.disaggregation.pvd.prompt_index import (
        PromptIndexManager,
        SearchRequestIdentity,
    )
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
    from test_pvd_prompt_vectors import FakePool, pack_shard, storage_layout

    pool = FakePool(layers=2)
    layout = storage_layout(pool)
    packed, manifest, _ = pack_shard(pool, layout, rank=0, prompt_tokens=63)
    owner = backend()
    owner.fixed_native_views = fixed_views
    manager = PromptIndexManager(
        vector_space="test",
        backend=owner,
        group_heads=4,
        budget=TransferBudget(128 * 1024 * 1024, 1),
    )
    gate = manager.open("entry")
    for pages in (8, 12):
        result = manager.progress_chunked(
            "entry",
            packed.tensor,
            layout=layout,
            manifest=manifest,
            complete_pages=pages,
            stored=False,
        )
        assert result in ("built_prefix", "extended")
        assert gate.state is IndexState.ABSENT
        index = next(iter(manager._entries["entry"].indexes.values()))
        assert index.handle.auxiliary[0].capacity == 63
        assert owner.runtime.sync_calls == (1 if pages == 8 else 2) + int(fixed_views)
        with pytest.raises(Exception, match="not ready"):
            manager.search(
                SearchRequestIdentity(
                    vector_space="test",
                    positional_encoding="rope_applied",
                    entry_transfer_id="entry",
                    layer=0,
                    kv_head=0,
                ),
                queries=torch.ones(1, layout.head_dim),
                top_k=1,
            )
    assert (
        manager.progress_chunked(
            "entry",
            packed.tensor,
            layout=layout,
            manifest=manifest,
            complete_pages=manifest.page_count,
            stored=False,
        )
        == "extended"
    )
    assert gate.state is IndexState.ABSENT
    manager.note_kv_readable("entry")
    assert (
        manager.progress_chunked(
            "entry",
            packed.tensor,
            layout=layout,
            manifest=manifest,
            complete_pages=manifest.page_count,
            stored=True,
        )
        == "ready"
    )
    assert owner.runtime.sync_calls == 3 + int(fixed_views)
    query = pool.k_buffer[0][62, 0].float().view(1, 8)
    result = manager.search(
        SearchRequestIdentity(
            vector_space="test",
            positional_encoding="rope_applied",
            entry_transfer_id="entry",
            layer=0,
            kv_head=0,
        ),
        queries=query,
        top_k=1,
    )
    expected = int((pool.k_buffer[0][:63, 0].float() @ query.flatten()).argmax())
    assert result.selection.token_ids[0] == expected
    assert (
        len({id(value.mapping) for value in manager._entries["entry"].vectors.values()})
        == 1
    )
    manager.close("entry")
    assert not owner._owners
