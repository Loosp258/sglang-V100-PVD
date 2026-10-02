"""Ownership/validation checks; fake runtime is not GPU performance evidence."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from test_pvd_cagra_backend import Runtime, backend, build
from sglang.srt.disaggregation.pvd.cagra_search_batch import FilteredSearchWorkspace
from sglang.srt.disaggregation.pvd.index_search import IndexCompletionUnknown, IndexSearchError


class BatchRuntime(Runtime):
    def __init__(self):
        super().__init__()
        self.cp = SimpleNamespace(from_dlpack=lambda x: x)
        self.filters = SimpleNamespace(from_bitset=lambda x: x)
        self.cagra = SimpleNamespace(SearchParams=lambda **kw: kw, search=self.submit)

    def scope(self, owner):
        return nullcontext()

    def submit(self, params, index, query, k, *, neighbors, distances, **kw):
        self.events.append(("submit",))
        if self.search_error:
            raise self.search_error
        neighbors.copy_(torch.arange(k).repeat(len(query), 1).to(torch.uint32))
        distances.fill_(1)


def workspace():
    rt = BatchRuntime()
    b = backend(rt)
    index = build(b)
    ws = FilteredSearchWorkspace(b, index,
        [torch.tensor([255], dtype=torch.uint32) for _ in range(4)],
        num_queries=2, top_k=2)
    rt.events.clear()
    return b, ws, torch.ones(4, 2, 3)


def test_one_fence_and_reused_outputs():
    b, ws, q = workspace()
    rows, scores = ws.search(q)
    assert b.runtime.events == [("submit",)] * 4 + [("sync",)]
    assert rows.tolist() == [[[0, 1], [0, 1]]] * 4
    assert ws.retained_bytes == 4 * 4 + 4 * 2 * 2 * 8
    assert ws.search(q)[0].data_ptr() == rows.data_ptr()
    ws.close()
    with pytest.raises(IndexSearchError):
        ws.search(q)


def test_filters_are_owned_snapshots_and_invalid_filters_are_refused():
    b, ws, q = workspace()
    bits = [torch.tensor([255], dtype=torch.uint32) for _ in range(4)]
    other = FilteredSearchWorkspace(b, ws.index, bits, num_queries=2, top_k=2)
    bits[0].zero_()
    assert other.bitsets[0].item() == 255
    bits[0] = torch.tensor([255], dtype=torch.int64)
    with pytest.raises(IndexSearchError):
        FilteredSearchWorkspace(b, ws.index, bits, num_queries=2, top_k=2)


@pytest.mark.parametrize("change", ["nonfinite", "shape", "width", "stale", "disposed"])
def test_reject_before_native(change):
    b, ws, q = workspace()
    if change == "nonfinite":
        q[0, 0, 0] = float("nan")
    elif change == "shape":
        q = q[:3]
    elif change == "width":
        b.itopk_size = 3
    elif change == "stale":
        ws.index.handle.rows += 1
    else:
        b.dispose(ws.index)
        b.runtime.events.clear()
    with pytest.raises(IndexSearchError):
        ws.search(q)
    assert b.runtime.events == ([("sync",)] if change == "nonfinite" else [])


def test_native_error_drains_before_release():
    b, ws, q = workspace()
    b.runtime.search_error = ValueError("native submit")
    with pytest.raises(ValueError):
        ws.search(q)
    assert b.runtime.events == [("submit",), ("sync",)]
    assert ws.pending_queries is None
    ws.close()


def test_unknown_completion_retains_every_owner():
    b, ws, q = workspace()
    b.runtime.sync_error = ValueError("unknown completion")
    with pytest.raises(IndexCompletionUnknown):
        ws.search(q)
    assert b._search_quarantine == [ws]
    assert ws.pending_queries is q
    assert ws.neighbors is not None and ws.bitsets
    with pytest.raises(IndexCompletionUnknown):
        ws.close()
    with pytest.raises(IndexCompletionUnknown):
        b.dispose(ws.index)


def test_subset_submits_only_received_queries_with_one_fence():
    b, ws, q = workspace()
    rows, _ = ws.search(q[:2].contiguous(), heads=(2, 3))
    assert b.runtime.events == [("submit",), ("submit",), ("sync",)]
    assert rows[2:].tolist() == [[[0, 1], [0, 1]]] * 2
    assert ws.pending_queries is None
    ws.close()


@pytest.mark.parametrize("heads", [(), (0, 0), (0, 4), (True, 1), [0, 1]])
def test_invalid_subset_cannot_submit_native_work(heads):
    b, ws, q = workspace()
    with pytest.raises(IndexSearchError):
        ws.search(q[:2], heads=heads)
    assert b.runtime.events == []
    ws.close()


@pytest.mark.parametrize("heads", [(0, 1, 2, 3), (2, 0), (3,)])
def test_host_snapshot_preserves_head_order_and_device_query_values(heads):
    b, ws, q = workspace()
    q = torch.stack([torch.full((2, 3), float(head + 1)) for head in heads])
    seen = []
    submit = b.runtime.submit

    def capture(params, index, query, k, **kwargs):
        seen.append(query.clone())
        submit(params, index, query, k, **kwargs)
        kwargs['distances'].fill_(float(query[0, 0]))

    b.runtime.cagra.search = capture
    timings = {}
    rows, scores, device_queries = ws.search_host(q, heads=heads, timings=timings)
    assert b.runtime.events == [("submit",)] * len(heads) + [("sync",)]
    assert torch.equal(device_queries, q)
    assert device_queries.data_ptr() != q.data_ptr()
    assert all(torch.equal(query, q[position]) for position, query in enumerate(seen))
    for position, head in enumerate(heads):
        assert rows[head].tolist() == [[0, 1], [0, 1]]
        assert scores[head].tolist() == [[float(head + 1)] * 2] * 2
    assert all(timings[name] >= 0 for name in
               ('host_snapshot', 'finite_proof', 'query_place', 'native_completion'))
    assert ws.pending_queries is ws.pending_host_queries is None
    ws.close()


def test_host_finite_proof_has_no_gpu_reduction_or_public_unchecked_switch(monkeypatch):
    b, ws, q = workspace()
    calls = []
    matrix = b._matrix

    def checked_matrix(tensor, *, check_finite=True):
        calls.append(check_finite)
        matrix(tensor, check_finite=check_finite)

    monkeypatch.setattr(b, '_matrix', checked_matrix)
    ws.search(q)
    assert calls == [True]
    calls.clear()

    def unexpected_gpu_reduction(tensor):
        raise AssertionError("CPU snapshot proof must not call torch.isfinite")

    monkeypatch.setattr(torch, 'isfinite', unexpected_gpu_reduction)
    ws.search_host(q)
    assert calls == [False]
    with pytest.raises(TypeError):
        ws.search_host(q, check_finite=False)
    with pytest.raises(TypeError):
        ws.search(q, check_finite=False)
    ws.close()


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_host_nonfinite_rejects_before_copy_and_native(value, monkeypatch):
    b, ws, q = workspace()
    q[0, 0, 0] = value

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("non-finite host queries must be refused before copy")

    monkeypatch.setattr(torch.Tensor, 'to', unexpected_copy)
    with pytest.raises(IndexSearchError, match='finite'):
        ws.search_host(q)
    assert b.runtime.events == []
    assert ws.pending_queries is ws.pending_host_queries is None
    ws.close()


@pytest.mark.parametrize('change', ['shape', 'dtype', 'noncontiguous', 'device', 'nontensor'])
def test_invalid_host_queries_reject_before_native(change):
    b, ws, q = workspace()
    if change == 'shape':
        q = q[:3]
    elif change == 'dtype':
        q = q.double()
    elif change == 'noncontiguous':
        q = torch.ones(4, 2, 6)[..., ::2]
    elif change == 'device':
        q = torch.empty(4, 2, 3, device='meta')
    else:
        q = q.tolist()
    with pytest.raises(IndexSearchError):
        ws.search_host(q)
    assert b.runtime.events == []
    assert ws.pending_queries is ws.pending_host_queries is None
    ws.close()


@pytest.mark.parametrize('heads', [(), (0, 0), (0, 4), (True, 1), [0, 1]])
def test_invalid_host_head_subset_cannot_submit_native_work(heads):
    b, ws, q = workspace()
    with pytest.raises(IndexSearchError):
        ws.search_host(q[:2], heads=heads)
    assert b.runtime.events == []
    ws.close()


def test_host_snapshot_cannot_alias_mutation_after_proof():
    b, ws, q = workspace()
    q.requires_grad_(True)
    seen = []
    submit = b.runtime.submit

    def scope(owner):
        # Simulate the caller changing its alias after validation, before the
        # native calls. The private snapshot is the sole source of submitted Q.
        with torch.no_grad():
            q.fill_(float('nan'))
        return nullcontext()

    def capture(params, index, query, k, **kwargs):
        seen.append(query.clone())
        submit(params, index, query, k, **kwargs)

    b.runtime.scope = scope
    b.runtime.cagra.search = capture
    _, _, device_queries = ws.search_host(q)
    assert torch.isnan(q).all()
    assert torch.equal(device_queries, torch.ones_like(device_queries))
    assert not device_queries.requires_grad
    assert device_queries.data_ptr() != q.data_ptr()
    assert all(torch.equal(query, torch.ones_like(query)) for query in seen)
    ws.close()


def test_host_native_error_drains_both_sources_before_release():
    b, ws, q = workspace()
    b.runtime.search_error = ValueError('native submit')
    with pytest.raises(ValueError, match='native submit'):
        ws.search_host(q)
    assert b.runtime.events == [("submit",), ("sync",)]
    assert ws.pending_queries is ws.pending_host_queries is None
    ws.close()


def test_host_unknown_completion_retains_private_snapshot_device_and_outputs():
    b, ws, q = workspace()
    b.runtime.sync_error = ValueError('unknown completion')
    with pytest.raises(IndexCompletionUnknown):
        ws.search_host(q)
    assert b._search_quarantine == [ws]
    assert ws.pending_host_queries is not q
    assert ws.pending_host_queries.data_ptr() != q.data_ptr()
    assert torch.equal(ws.pending_host_queries, q)
    assert torch.equal(ws.pending_queries, q)
    q.zero_()
    assert torch.equal(ws.pending_host_queries, torch.ones_like(q))
    assert ws.neighbors is not None and ws.bitsets
    with pytest.raises(IndexCompletionUnknown):
        ws.close()
    with pytest.raises(IndexCompletionUnknown):
        b.dispose(ws.index)


@pytest.mark.parametrize('unknown', [False, True])
def test_host_copy_failure_drains_or_retains_source(monkeypatch, unknown):
    b, ws, q = workspace()
    if unknown:
        b.runtime.sync_error = ValueError('unknown completion')

    def failed_copy(*args, **kwargs):
        raise ValueError('device copy failed')

    monkeypatch.setattr(torch.Tensor, 'to', failed_copy)
    expected = IndexCompletionUnknown if unknown else ValueError
    with pytest.raises(expected):
        ws.search_host(q)
    assert b.runtime.events == [("sync",)]
    assert ws.pending_queries is None
    if unknown:
        assert b._search_quarantine == [ws]
        assert ws.pending_host_queries is not q
        assert torch.equal(ws.pending_host_queries, q)
        with pytest.raises(IndexCompletionUnknown):
            ws.close()
    else:
        assert ws.pending_host_queries is None
        ws.close()
