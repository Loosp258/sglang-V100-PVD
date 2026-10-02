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
    assert b.runtime.events == []


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
