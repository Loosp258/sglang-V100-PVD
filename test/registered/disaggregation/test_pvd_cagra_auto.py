"""Explicit short-Prompt exact fallback; CPU doubles are not CUDA evidence."""

import pytest
import torch
from sglang.srt.disaggregation.pvd.cagra_backend import CagraAutoIndexBackend
from sglang.srt.disaggregation.pvd.index_search import (
    IndexCompletionUnknown,
    IndexSearchError,
)
from test_pvd_cagra_backend import Runtime, args, backend
from test_pvd_prompt_index import budgeted, ident, stored_entry


def test_short_prompt_uses_exact_device_path_with_actual_footprints():
    native = backend()
    auto = CagraAutoIndexBackend(native)
    vectors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert auto.device == native.device
    assert auto.build_footprint(2, 3, metric="ip") == 2 * 3 * 4
    assert auto.build_scratch_footprint(2, 3, metric="ip") > 0
    assert auto.search_footprint(2, 3, 1, 1) > 0
    index = auto.build(vectors, vector_space="model", metric="ip")
    assert index.handle.data_ptr() != vectors.data_ptr()
    assert native.runtime.events == [], "a short prompt must not build a CAGRA graph"
    rows, scores = auto.search(index, vectors[:1], top_k=1)
    assert rows.tolist() == [[0]] and scores.tolist() == [[1.0]]
    auto.dispose(index)
    assert not auto._built
    with pytest.raises(IndexSearchError, match="disposed"):
        auto.search(index, vectors[:1], top_k=1)


def test_long_prompt_keeps_the_native_cap_and_native_lifecycle():
    native = backend()
    auto = CagraAutoIndexBackend(native)
    assert auto.build_footprint(8, 3, metric="ip") == native.cap
    assert auto.search_footprint(8, 3, 1, 2) == native.search_footprint(
        8, 3, 1, 2
    )
    index = auto.build(torch.ones((8, 3)), vector_space="model", metric="ip")
    assert native.runtime.events[0] == ("create", native.cap)
    auto.search(index, torch.ones((1, 3)), top_k=2)
    auto.dispose(index)
    assert native.runtime.events.count(("dispose",)) == 1
    assert not auto._built and not native._owners


def test_exact_cutoff_is_independent_of_graph_intermediate_degree():
    native = backend(intermediate_degree=2)
    auto = CagraAutoIndexBackend(native, exact_max_rows=4)
    assert auto.exact_max_rows == 4
    assert auto.build_path(4) == "exact"
    assert auto.build_path(5) == "cagra"
    assert auto.build_footprint(4, 3, metric="ip") == 4 * 3 * 4
    assert auto.build_footprint(5, 3, metric="ip") == native.cap
    exact = auto.build(torch.ones((4, 3)), vector_space="model", metric="ip")
    assert native.runtime.events == []
    auto.dispose(exact)
    graph = auto.build(torch.ones((5, 3)), vector_space="model", metric="ip")
    assert native.runtime.events[0] == ("create", native.cap)
    auto.dispose(graph)


def test_exact_cutoff_cannot_leave_unsupported_native_row_gap():
    with pytest.raises(ValueError, match="at least"):
        CagraAutoIndexBackend(backend(intermediate_degree=4), exact_max_rows=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA device unavailable")
def test_short_prompt_exact_fallback_stays_on_the_gpu():
    runtime = Runtime()
    runtime.device = torch.device("cuda:0")
    auto = CagraAutoIndexBackend(backend(runtime, device="cuda:0"))
    vectors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda:0"
    )
    index = auto.build(vectors, vector_space="model", metric="ip")
    try:
        rows, scores = auto.search(index, vectors[:1], top_k=1)
        assert index.handle.device == rows.device == scores.device == auto.device
        assert rows.tolist() == [[0]] and scores.tolist() == [[1.0]]
        assert not any(event[0] == "create" for event in runtime.events)
    finally:
        auto.dispose(index)


def test_real_store_short_entry_is_searchable_and_refunds_exact_bytes():
    native = backend(intermediate_degree=16)
    auto = CagraAutoIndexBackend(native)
    manager, budget = budgeted(auto)
    store, manifest, _, _ = stored_entry(manager, prompt_tokens=2)
    assert store.progress_prompt_indexes()["built"] == 1
    record = manager._entries[manifest.key.transfer_id]
    expected = sum(
        vector.vectors.numel() * vector.vectors.element_size() +
        record.indexes[key].handle.numel() * record.indexes[key].handle.element_size()
        for key, vector in record.vectors.items()
    )
    assert budget.snapshot()["used_staging_bytes"] == expected
    (layer, head), vector = next(iter(record.vectors.items()))
    result = manager.search(
        ident(manifest.key.transfer_id, layer, head),
        queries=vector.vectors[:1].clone(),
        top_k=1,
    )
    assert len(result.selection.token_ids) == 1
    assert not any(event[0] == "create" for event in native.runtime.events)
    store.release_entry(manifest.key)
    assert budget.snapshot()["used_staging_bytes"] == 0


def test_native_unknown_poison_refuses_short_exact_fallback():
    runtime = Runtime()
    runtime.build_error = RuntimeError("native build failed")
    runtime.cleanup_error = RuntimeError("native cleanup unknown")
    auto = CagraAutoIndexBackend(backend(runtime))
    with pytest.raises(IndexCompletionUnknown):
        auto.build(torch.ones((8, 3)), vector_space="model", metric="ip")
    with pytest.raises(IndexCompletionUnknown):
        auto.build(torch.ones((2, 3)), vector_space="model", metric="ip")


def test_explicit_auto_cli_requires_native_bounds_and_rank_device(monkeypatch):
    from sglang.srt.disaggregation.pvd import cagra_backend, server

    config = args(
        "--prompt-index-backend", "cagra-auto",
        "--prompt-index-vector-space", "model",
        "--prompt-index-budget-bytes", "1048576",
    )
    with pytest.raises(ValueError, match="CAGRA"):
        server._validate_args(config)
    config.prompt_index_cagra_native_bytes = 4096
    server._validate_args(config)
    with pytest.raises(ValueError, match="actual device"):
        server._build_prompt_index(config)

    class DeviceRuntime(Runtime):
        def __init__(self, device):
            super().__init__()
            self.device = torch.device(device)

    monkeypatch.setattr(cagra_backend, "CagraNativeRuntime", DeviceRuntime)
    manager = server._build_prompt_index(config, device="cuda:0")
    assert manager.backend.name == "cagra_auto"
    assert manager.backend.device == torch.device("cuda:0")


def test_explicit_auto_cli_exact_cutoff(monkeypatch):
    from sglang.srt.disaggregation.pvd import cagra_backend, server

    config = args(
        "--prompt-index-backend",
        "cagra-auto",
        "--prompt-index-vector-space",
        "model",
        "--prompt-index-budget-bytes",
        "1048576",
        "--prompt-index-cagra-native-bytes",
        "4096",
        "--prompt-index-cagra-graph-degree",
        "2",
        "--prompt-index-cagra-intermediate-degree",
        "4",
        "--prompt-index-exact-max-rows",
        "32",
    )
    server._validate_args(config)

    class DeviceRuntime(Runtime):
        def __init__(self, device):
            super().__init__()
            self.device = torch.device(device)

    monkeypatch.setattr(cagra_backend, "CagraNativeRuntime", DeviceRuntime)
    manager = server._build_prompt_index(config, device="cuda:0")
    assert manager.backend.exact_max_rows == 32
    config.prompt_index_exact_max_rows = 3
    with pytest.raises(ValueError, match="exact-max-rows"):
        server._validate_args(config)
    config.prompt_index_exact_max_rows = 32
    config.prompt_index_backend = "cagra"
    with pytest.raises(ValueError, match="cagra-auto"):
        server._validate_args(config)
