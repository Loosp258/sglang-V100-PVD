"""Explicit short-Prompt exact fallback; CPU doubles are not CUDA evidence."""

import pytest
import torch
from sglang.srt.disaggregation.pvd.cagra_backend import CagraAutoIndexBackend
from sglang.srt.disaggregation.pvd.index_search import (
    IndexCompletionUnknown,
    IndexSearchError,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferBudget,
    TransferCapacityError,
)
from test_pvd_cagra_backend import Runtime, args, backend
from test_pvd_prompt_index import SPACE, budgeted, ident, manager, shard_client, stored_entry


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


def test_auto_groups_only_live_uniform_exact_indexes():
    native = backend(intermediate_degree=2)
    auto = CagraAutoIndexBackend(native)
    first = auto.build(torch.eye(2), vector_space="model", metric="ip")
    second = auto.build(torch.eye(2), vector_space="model", metric="ip")
    indexes = (first, second)
    assert auto.supports_grouped_exact(indexes, num_queries=1, top_k=1)
    assert auto.grouped_exact_footprint(indexes, num_queries=1, top_k=1) > 0
    result = auto.search_grouped_exact(
        indexes, (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])), top_k=1
    )
    assert [rows.tolist() for rows, _ in result] == [[[0]], [[1]]]
    graph = auto.build(torch.ones(3, 2), vector_space="model", metric="ip")
    assert not auto.supports_grouped_exact((first, graph), num_queries=1, top_k=1)
    auto.dispose(graph)
    auto.dispose(first)
    with pytest.raises(IndexSearchError, match="disposed"):
        auto.supports_grouped_exact(indexes, num_queries=1, top_k=1)
    auto.dispose(second)


def test_manager_search_many_groups_exact_indexes_under_one_budget(monkeypatch):
    auto = CagraAutoIndexBackend(backend(intermediate_degree=16))
    budget = TransferBudget(staging_bytes=1 << 20, max_inflight=4)
    index = manager(backend=auto, metric="ip", budget=budget)
    store, manifest, _, _ = stored_entry(index, prompt_tokens=2)
    assert store.progress_prompt_indexes()["built"] == 1
    record = index._entries[manifest.key.transfer_id]
    keys = sorted(record.vectors)[:2]
    requests = tuple(
        (
            ident(manifest.key.transfer_id, *key),
            record.vectors[key].vectors[:1].clone(),
            1,
        )
        for key in keys
    )
    expected = tuple(
        index.search(identity, queries=query, top_k=top_k)
        for identity, query, top_k in requests
    )
    before = budget.snapshot()["used_staging_bytes"]

    def scalar_must_not_run(*args, **kwargs):
        raise AssertionError("uniform exact batch must use grouped search")

    monkeypatch.setattr(auto, "search", scalar_must_not_run)
    results = index.search_many(requests)
    for actual, reference in zip(results, expected):
        assert actual.selection.token_ids == reference.selection.token_ids
        assert actual.selection.page_ids == reference.selection.page_ids
        assert actual.selection.id_mapping_version == reference.selection.id_mapping_version
        assert actual.validated == reference.validated
        assert actual.selection.scores == pytest.approx(
            reference.selection.scores, abs=1e-5
        )
    assert budget.snapshot()["used_staging_bytes"] == before
    store.release_entry(manifest.key)
    assert budget.snapshot()["used_staging_bytes"] == 0


def test_manager_batch_rejects_a_stale_pin_before_any_search(monkeypatch):
    auto = CagraAutoIndexBackend(backend(intermediate_degree=16))
    index = manager(backend=auto, metric="ip")
    store, manifest, _, _ = stored_entry(index, prompt_tokens=2)
    store.progress_prompt_indexes()
    record = index._entries[manifest.key.transfer_id]
    keys = sorted(record.vectors)[:2]
    requests = [
        (ident(manifest.key.transfer_id, *key), record.vectors[key].vectors[:1], 1)
        for key in keys
    ]
    requests[1] = (
        ident(manifest.key.transfer_id, *keys[1], expected_index_version="stale"),
        requests[1][1],
        1,
    )

    def any_search_is_wrong(*args, **kwargs):
        raise AssertionError("invalid batch must not search")

    monkeypatch.setattr(auto, "search_grouped_exact", any_search_is_wrong)
    monkeypatch.setattr(auto, "search", any_search_is_wrong)
    with pytest.raises(ValueError, match="rebuilt"):
        index.search_many(requests)
    assert record.users == 0
    store.release_entry(manifest.key)


def test_manager_batch_close_waits_for_grouped_reader_and_refunds(monkeypatch):
    import threading

    auto = CagraAutoIndexBackend(backend(intermediate_degree=16))
    budget = TransferBudget(staging_bytes=1 << 20, max_inflight=4)
    index = manager(backend=auto, metric="ip", budget=budget)
    store, manifest, _, _ = stored_entry(index, prompt_tokens=2)
    store.progress_prompt_indexes()
    record = index._entries[manifest.key.transfer_id]
    requests = tuple(
        (ident(manifest.key.transfer_id, *key), vector.vectors[:1].clone(), 1)
        for key, vector in sorted(record.vectors.items())[:2]
    )
    entered, continue_search = threading.Event(), threading.Event()
    original = auto.search_grouped_exact

    def paused(*args, **kwargs):
        entered.set()
        assert continue_search.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(auto, "search_grouped_exact", paused)
    output = []
    worker = threading.Thread(target=lambda: output.append(index.search_many(requests)))
    worker.start()
    assert entered.wait(5)
    try:
        assert record.users == 1
        store.release_entry(manifest.key)
        assert budget.snapshot()["used_staging_bytes"] > 0
    finally:
        continue_search.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert len(output[0]) == 2
    assert budget.snapshot()["used_staging_bytes"] == 0


def test_manager_batch_unknown_completion_retains_all_owners():
    runtime = Runtime()
    auto = CagraAutoIndexBackend(backend(runtime, intermediate_degree=16))
    budget = TransferBudget(staging_bytes=1 << 20, max_inflight=4)
    index = manager(backend=auto, metric="ip", budget=budget)
    store, manifest, _, _ = stored_entry(index, prompt_tokens=2)
    store.progress_prompt_indexes()
    record = index._entries[manifest.key.transfer_id]
    requests = tuple(
        (ident(manifest.key.transfer_id, *key), vector.vectors[:1].clone(), 1)
        for key, vector in sorted(record.vectors.items())[:2]
    )
    runtime.sync_error = RuntimeError("completion is unknown")
    with pytest.raises(IndexCompletionUnknown, match="completion is unknown"):
        index.search_many(requests)
    assert index.quarantined
    assert record.users == 1
    assert index._retained_operations
    assert budget.snapshot()["used_staging_bytes"] > 0


def test_manager_batch_capacity_refuses_before_grouped_backend(monkeypatch):
    auto = CagraAutoIndexBackend(backend(intermediate_degree=16))
    budget = TransferBudget(staging_bytes=1 << 20, max_inflight=4)
    index = manager(backend=auto, metric="ip", budget=budget)
    store, manifest, _, _ = stored_entry(index, prompt_tokens=2)
    store.progress_prompt_indexes()
    record = index._entries[manifest.key.transfer_id]
    keys = sorted(record.vectors)[:2]
    requests = tuple(
        (ident(manifest.key.transfer_id, *key), record.vectors[key].vectors[:1], 1)
        for key in keys
    )
    scratch = auto.grouped_exact_footprint(
        tuple(record.indexes[key] for key in keys), num_queries=1, top_k=1
    )
    before = budget.snapshot()["used_staging_bytes"]
    budget.reserve("competing-index", (1 << 20) - before - scratch + 1, 0)
    at_capacity = budget.snapshot()["used_staging_bytes"]

    def backend_must_not_run(*args, **kwargs):
        raise AssertionError("no backend call before reservation")

    monkeypatch.setattr(auto, "search_grouped_exact", backend_must_not_run)
    with pytest.raises(TransferCapacityError):
        index.search_many(requests)
    assert budget.snapshot()["used_staging_bytes"] == at_capacity
    assert record.users == 0
    budget.release("competing-index")
    store.release_entry(manifest.key)
    assert budget.snapshot()["used_staging_bytes"] == 0


def test_http_batch_opt_in_uses_grouped_exact_and_preserves_identity(
    monkeypatch, caplog
):
    import asyncio
    import logging

    async def scenario():
        auto = CagraAutoIndexBackend(backend(intermediate_degree=16))
        budget = TransferBudget(staging_bytes=1 << 20, max_inflight=4)
        index = manager(backend=auto, metric="ip", budget=budget)
        store, manifest, _, _ = stored_entry(index, prompt_tokens=2)
        store.progress_prompt_indexes()
        record = index._entries[manifest.key.transfer_id]
        descriptor = record.gate.descriptor
        items = []
        for n, (key, vector) in enumerate(sorted(record.vectors.items())[:2]):
            items.append(
                {
                    "search_protocol": "pvd.search.v1",
                    "search_id": f"grouped-{n}",
                    "transfer_id": manifest.key.transfer_id,
                    "vector_space": SPACE,
                    "positional_encoding": ROPE_APPLIED,
                    "expected_index_version": descriptor.index_version,
                    "expected_id_mapping_version": descriptor.id_mapping_version,
                    "layer": key[0],
                    "kv_head": key[1],
                    "queries": vector.vectors[:1].tolist(),
                    "top_k": 1,
                }
            )
        before = budget.snapshot()["used_staging_bytes"]
        monkeypatch.setenv("PVD_GROUPED_EXACT_SEARCH", "1")
        monkeypatch.setenv("PVD_PROFILE_V_SEARCH", "1")
        with caplog.at_level(
            logging.INFO, logger="sglang.srt.disaggregation.pvd.control_server"
        ):
            async with shard_client(store) as http:
                reply = await http.post(
                    "/internal/v1/indexes/search-batch",
                    json={
                        "batch_protocol": "pvd.search.batch.v1",
                        "batch_id": "grouped-batch",
                        "items": items,
                    },
                )
                assert reply.status == 200, await reply.text()
                body = await reply.json()
        assert body["batch_id"] == "grouped-batch"
        assert [row["search_id"] for row in body["results"]] == [
            "grouped-0",
            "grouped-1",
        ]
        assert all(row["token_ids"] == [0] for row in body["results"])
        assert all(row["index_version"] == descriptor.index_version for row in body["results"])
        assert all("index_version" in row["validated"] for row in body["results"])
        assert any("path=grouped_exact" in row.message for row in caplog.records)
        assert budget.snapshot()["used_staging_bytes"] == before
        store.release_entry(manifest.key)
        assert budget.snapshot()["used_staging_bytes"] == 0

    asyncio.run(scenario())


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
