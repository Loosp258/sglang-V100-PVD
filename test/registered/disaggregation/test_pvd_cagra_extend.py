"""Incremental CAGRA ownership and failure gates, using a runtime double."""

import pytest
import torch
from sglang.srt.disaggregation.pvd.cagra_backend import (
    CagraAutoIndexBackend,
    CagraIndexBackend,
    _NativeIndex,
)
from sglang.srt.disaggregation.pvd.index_search import (
    IndexCompletionUnknown,
    IndexSearchError,
)


class Runtime:
    device = torch.device("cpu")
    supports_extend = True

    def __init__(self):
        self.extends = 0
        self.fail_extend = False

    def create(self, cap):
        return _NativeIndex(cap)

    def build(self, owner, vectors, **kwargs):
        owner.native = object()
        owner.vectors = vectors

    def extend(self, owner, additional_vectors):
        self.extends += 1
        if self.fail_extend:
            raise RuntimeError("native extension failed after submission")
        owner.vectors = (owner.vectors, additional_vectors)

    def synchronize(self):
        pass

    def search(self, owner, queries, *, top_k, itopk_size):
        return (
            torch.arange(top_k).to(torch.uint32).repeat(len(queries), 1),
            torch.ones((len(queries), top_k)),
        )

    def dispose(self, owner):
        owner.disposed = True


def backend(runtime):
    return CagraIndexBackend(
        device="cpu",
        native_bytes_per_index=4096,
        graph_degree=2,
        intermediate_degree=4,
        itopk_size=8,
        _runtime=runtime,
    )


def test_extended_index_replaces_the_old_count_view():
    runtime = Runtime()
    owner = backend(runtime)
    original = owner.build(torch.ones((32, 8)), vector_space="prompt", metric="ip")
    extended = owner.extend(original, torch.ones((16, 8)))
    assert extended.count == 48
    assert runtime.extends == 1
    with pytest.raises(IndexSearchError, match="stale"):
        owner.search(original, torch.ones((1, 8)), top_k=2)
    rows, _ = owner.search(extended, torch.ones((1, 8)), top_k=2)
    assert rows.shape == (1, 2)
    owner.dispose(extended)


def test_rejected_input_does_not_submit_an_extension():
    runtime = Runtime()
    owner = backend(runtime)
    original = owner.build(torch.ones((32, 8)), vector_space="prompt", metric="ip")
    with pytest.raises(IndexSearchError, match="dimension"):
        owner.extend(original, torch.ones((16, 4)))
    assert runtime.extends == 0
    assert original.count == original.handle.rows == 32
    owner.dispose(original)


def test_native_extension_failure_quarantines_and_retains_owner():
    runtime = Runtime()
    runtime.fail_extend = True
    owner = backend(runtime)
    original = owner.build(torch.ones((32, 8)), vector_space="prompt", metric="ip")
    with pytest.raises(IndexCompletionUnknown, match="state is unknown"):
        owner.extend(original, torch.ones((16, 8)))
    assert owner._owners[id(original.handle)] is original.handle
    with pytest.raises(IndexCompletionUnknown):
        owner.search(original, torch.ones((1, 8)), top_k=2)


def test_auto_backend_tracks_only_latest_provisional_index():
    runtime = Runtime()
    owner = CagraAutoIndexBackend(backend(runtime))
    original = owner.build(torch.ones((32, 8)), vector_space="prompt", metric="ip")
    extended = owner.extend(original, torch.ones((16, 8)))
    assert owner.supports_extend
    assert extended.count == 48
    with pytest.raises(IndexSearchError, match="unknown"):
        owner.search(original, torch.ones((1, 8)), top_k=2)
    assert owner.search(extended, torch.ones((1, 8)), top_k=2)[0].shape == (1, 2)
    owner.dispose(extended)


def test_auto_backend_refuses_short_exact_graph_extension():
    runtime = Runtime()
    owner = CagraAutoIndexBackend(backend(runtime))
    original = owner.build(torch.ones((4, 8)), vector_space="prompt", metric="ip")
    with pytest.raises(IndexSearchError, match="short-Prompt"):
        owner.extend(original, torch.ones((8, 8)))
    assert runtime.extends == 0
    owner.dispose(original)
