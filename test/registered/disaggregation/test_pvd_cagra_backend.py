"""CAGRA policy/ownership tests. Runtime doubles are NOT native GPU evidence."""

import ctypes
import os
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.cagra_backend import (
    CagraIndexBackend,
    CagraNativeRuntime,
    _NativeIndex,
)
from sglang.srt.disaggregation.pvd.index_search import (
    IndexCompletionUnknown,
    IndexSearchError,
)
from test_pvd_prompt_index import budgeted, ident, stored_entry


class Runtime:
    device = torch.device("cpu")

    def __init__(self):
        self.events = []
        self.build_error = None
        self.search_error = None
        self.cleanup_error = None
        self.sync_error = None
        self.raw_scores = None

    def create(self, cap):
        self.events.append(("create", cap))
        return _NativeIndex(cap)

    def build(self, owner, vectors, **kwargs):
        self.events.append(("build", kwargs))
        owner.vectors = vectors
        owner.native = object()
        if self.build_error:
            raise self.build_error

    def synchronize(self):
        self.events.append(("sync",))
        if self.sync_error:
            raise self.sync_error

    def search(self, owner, queries, *, top_k, itopk_size):
        self.events.append(("search", top_k, itopk_size))
        if self.search_error:
            raise self.search_error
        rows = torch.arange(top_k).repeat(len(queries), 1).to(torch.uint32)
        scores = torch.ones((len(queries), top_k)) * 25
        if self.raw_scores is not None:
            scores.fill_(self.raw_scores)
        return rows, scores

    def dispose(self, owner):
        self.events.append(("dispose",))
        if self.cleanup_error:
            raise self.cleanup_error
        owner.native = owner.vectors = None
        owner.disposed = True


def backend(runtime=None, **kwargs):
    settings = dict(
        device="cpu",
        native_bytes_per_index=4096,
        graph_degree=1,
        intermediate_degree=2,
        itopk_size=4,
        _runtime=runtime if runtime is not None else Runtime(),
    )
    settings.update(kwargs)
    return CagraIndexBackend(**settings)


def build(b, metric="ip"):
    return b.build(
        torch.arange(24, dtype=torch.float32).reshape(8, 3),
        vector_space="model",
        metric=metric,
    )


@pytest.mark.parametrize("metric,score", [("ip", 25), ("l2", -5)])
def test_score_contract_and_lifetime(metric, score):
    b = backend()
    index = build(b, metric)
    rows, scores = b.search(index, torch.ones((2, 3)), top_k=2)
    assert rows.dtype == torch.uint32
    assert rows.tolist() == [[0, 1], [0, 1]]
    assert scores.tolist() == [[score, score], [score, score]]
    assert b.runtime.events[0] == ("create", b.cap)
    assert b.runtime.events[1][1] == dict(
        metric=metric, graph_degree=1, intermediate_degree=2
    )
    b.dispose(index)
    b.dispose(index)
    assert b.runtime.events.count(("dispose",)) == 1
    assert not b._owners
    with pytest.raises(IndexSearchError, match="disposed"):
        b.search(index, torch.ones((1, 3)), top_k=1)


@pytest.mark.parametrize("score", [-1, float("-inf")])
def test_negative_squared_distance_is_not_hidden(score):
    runtime = Runtime()
    runtime.raw_scores = score
    b = backend(runtime)
    index = build(b, "l2")
    with pytest.raises(IndexSearchError, match="negative squared"):
        b.search(index, torch.ones((1, 3)), top_k=1)
    assert runtime.events[-1] == ("sync",)
    b.dispose(index)


@pytest.mark.parametrize(
    "vectors",
    [
        torch.zeros((8, 3), dtype=torch.float16),
        torch.zeros(8),
        torch.zeros((3, 8)).T,
        torch.full((8, 3), float("nan")),
        torch.zeros((2, 3)),
    ],
)
def test_bad_vectors_fail_before_native_allocation(vectors):
    b = backend()
    with pytest.raises(IndexSearchError):
        b.build(vectors, vector_space="model", metric="ip")
    assert b.runtime.events == []


@pytest.mark.parametrize("top_k", [True, 0, 5, 99])
def test_topk_refused_before_search(top_k):
    b = backend()
    index = build(b)
    with pytest.raises(IndexSearchError, match="bounds"):
        b.search(index, torch.ones((1, 3)), top_k=top_k)
    assert not any(e[0] == "search" for e in b.runtime.events)
    b.dispose(index)


def test_footprints_reserve_native_cap_and_application_temporaries():
    b = backend()
    assert b.build_footprint(8, 3, metric="ip") == 4096
    assert b.build_scratch_footprint(8, 3, metric="ip") >= 8 * 3 * 8
    assert b.search_footprint(8, 3, 2, 4) >= 2 * 4 * 9 + 2 * 3 * 8


def test_known_build_failure_disposes_before_retry():
    runtime = Runtime()
    runtime.build_error = RuntimeError("build failed")
    b = backend(runtime)
    with pytest.raises(RuntimeError, match="build failed"):
        build(b)
    assert runtime.events[-2:] == [("sync",), ("dispose",)]
    assert not b._owners
    runtime.build_error = None
    b.dispose(build(b))


@pytest.mark.parametrize("failure", ["sync", "cleanup"])
def test_unknown_build_keeps_owner_and_refuses_future_work(failure):
    runtime = Runtime()
    runtime.build_error = RuntimeError("build failed")
    setattr(runtime, failure + "_error", RuntimeError("not drained"))
    b = backend(runtime)
    with pytest.raises(IndexCompletionUnknown, match="not drained"):
        build(b)
    assert len(b._owners) == 1
    assert next(iter(b._owners.values())).vectors is not None
    runtime.sync_error = runtime.cleanup_error = None
    with pytest.raises(IndexCompletionUnknown):
        b.synchronize()
    with pytest.raises(IndexCompletionUnknown):
        build(b)


def test_search_exception_is_fenced_and_known_error_is_recoverable():
    runtime = Runtime()
    b = backend(runtime)
    index = build(b)
    runtime.search_error = ValueError("search failed")
    with pytest.raises(ValueError, match="search failed"):
        b.search(index, torch.ones((1, 3)), top_k=1)
    assert runtime.events[-1] == ("sync",)
    runtime.search_error = None
    b.search(index, torch.ones((1, 3)), top_k=1)
    runtime.sync_error = RuntimeError("not complete")
    with pytest.raises(IndexCompletionUnknown):
        b.search(index, torch.ones((1, 3)), top_k=1)
    assert len(b._owners) == 1


def test_real_store_charges_every_head_and_refunds_after_disposal():
    b = backend()
    manager, budget = budgeted(b)
    store, manifest, _, _ = stored_entry(manager)
    assert store.progress_prompt_indexes()["built"] == 1
    record = manager._entries[manifest.key.transfer_id]
    count = len(record.indexes)
    vectors = sum(
        item.vectors.numel() * item.vectors.element_size()
        for item in record.vectors.values()
    )
    assert budget.snapshot()["used_staging_bytes"] == vectors + count * b.cap
    (layer, head), item = next(iter(record.vectors.items()))
    result = manager.search(
        ident(manifest.key.transfer_id, layer, head),
        queries=item.vectors[:1].clone(),
        top_k=2,
    )
    assert len(result.selection.token_ids) == 2
    assert budget.snapshot()["used_staging_bytes"] == vectors + count * b.cap
    store.release_entry(manifest.key)
    assert budget.snapshot()["used_staging_bytes"] == 0
    assert b.runtime.events.count(("dispose",)) == count


def test_real_manager_keeps_charge_after_unknown_build():
    runtime = Runtime()
    runtime.build_error = RuntimeError("build failed")
    runtime.cleanup_error = RuntimeError("native retained")
    b = backend(runtime)
    manager, budget = budgeted(b)
    store, manifest, _, _ = stored_entry(manager)
    assert store.progress_prompt_indexes()["failed"] == 1
    charged = budget.snapshot()["used_staging_bytes"]
    assert charged >= b.cap
    manager.close(manifest.key.transfer_id)
    assert budget.snapshot()["used_staging_bytes"] == charged
    assert len(b._owners) == 1


def bridge(*, seen=256, free_status=1, exceed=False):
    """Execute the actual bridge verifier with a fake C API, not cuVS."""
    runtime = object.__new__(CagraNativeRuntime)
    limit = SimpleNamespace(used=0, get_allocation_limit=lambda: 4096)
    limit.get_allocated_bytes = lambda: limit.used
    owner = _NativeIndex(limit, SimpleNamespace(get_c_obj=lambda: 7, sync=lambda: None))
    calls = []

    def alloc(handle, pointer, size):
        assert handle == 7
        calls.append(("alloc", size))
        if size > 4096 and not exceed:
            return 0
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = 123
        limit.used = seen
        return 1

    def free(handle, pointer, size):
        calls.append(("free", size))
        if free_status == 1:
            limit.used = 0
        return free_status

    runtime.alloc, runtime.free = alloc, free
    return runtime, owner, calls


def test_actual_bridge_policy_probes_shared_resource_then_capacity():
    runtime, owner, calls = bridge()
    runtime._verify_allocator_bridge(owner)
    assert calls == [("alloc", 256), ("free", 256), ("alloc", 4352)]
    assert not owner.probe_allocations


def native_runtime_double(monkeypatch):
    """Exercise runtime methods, replacing only GPU/library boundaries."""
    runtime = object.__new__(CagraNativeRuntime)
    runtime.device = torch.device("cpu")
    runtime.lock = threading.RLock()
    events = []
    mr = SimpleNamespace(current="original")
    mr.get_current_device_resource = lambda: mr.current
    mr.set_current_device_resource = lambda value: setattr(mr, "current", value)
    runtime.mr = mr
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda _: SimpleNamespace(cuda_stream=1234)
    )
    runtime.synchronize = lambda: events.append("sync")
    runtime.Resources = lambda **kwargs: events.append(kwargs) or object()
    runtime._verify_allocator_bridge = lambda owner: events.append("bridge")
    runtime.cp = SimpleNamespace(from_dlpack=lambda tensor: tensor)
    runtime.cagra = SimpleNamespace(
        IndexParams=lambda **kwargs: kwargs,
        SearchParams=lambda **kwargs: kwargs,
    )
    owner = _NativeIndex(SimpleNamespace(get_allocated_bytes=lambda: 0))
    return runtime, owner, events


def test_native_call_arguments_outputs_stream_and_resource_scope(monkeypatch):
    runtime, owner, events = native_runtime_double(monkeypatch)
    vectors = torch.ones((8, 3))

    def native_build(params, dataset, **kwargs):
        assert runtime.mr.current is owner.limit
        assert params == dict(
            metric="inner_product",
            graph_degree=1,
            intermediate_graph_degree=2,
            build_algo="ivf_pq",
        )
        assert dataset is vectors
        assert kwargs == {"resources": owner.resources}
        return SimpleNamespace(trained=True, dim=3)

    def native_search(params, index, queries, k, **kwargs):
        assert runtime.mr.current is owner.limit
        assert index is owner.native
        assert params == {"itopk_size": 4}
        assert k == 2 and queries.shape == (1, 3)
        assert kwargs["resources"] is owner.resources
        assert kwargs["neighbors"].dtype == torch.uint32
        assert kwargs["distances"].dtype == torch.float32
        kwargs["neighbors"].fill_(1)
        kwargs["distances"].fill_(9)

    runtime.cagra.build = native_build
    runtime.cagra.search = native_search
    runtime.build(owner, vectors, metric="ip", graph_degree=1, intermediate_degree=2)
    assert events[:2] == [{"stream": 1234}, "bridge"]
    assert runtime.mr.current == "original"
    rows, scores = runtime.search(owner, torch.ones((1, 3)), top_k=2, itopk_size=4)
    assert rows.tolist() == [[1, 1]] and scores.tolist() == [[9, 9]]
    assert runtime.mr.current == "original"
    runtime.dispose(owner)
    assert owner.disposed and owner.vectors is None and owner.native is None
    assert runtime.mr.current == "original"


def test_native_scope_restored_after_exception_and_leaks_not_refunded(monkeypatch):
    runtime, owner, events = native_runtime_double(monkeypatch)
    with pytest.raises(RuntimeError, match="in scope"):
        with runtime.scope(owner):
            assert runtime.mr.current is owner.limit
            raise RuntimeError("in scope")
    assert runtime.mr.current == "original"
    owner.vectors = torch.ones(8)
    owner.limit.get_allocated_bytes = lambda: 256
    with pytest.raises(IndexCompletionUnknown, match="allocations alive"):
        runtime.dispose(owner)
    assert not owner.disposed and owner.vectors is not None
    assert events == ["sync", "sync"]
    assert runtime.mr.current == "original"


def test_ambiguous_probe_allocation_keeps_resources(monkeypatch):
    runtime, owner, events = native_runtime_double(monkeypatch)
    owner.resources = object()
    owner.probe_allocations.append((ctypes.c_void_p(123), 256))
    with pytest.raises(IndexCompletionUnknown, match="probe remains unresolved"):
        runtime.dispose(owner)
    assert owner.resources is not None and not owner.disposed


def test_unknown_search_is_sticky_even_after_successful_sync():
    runtime = Runtime()
    b = backend(runtime)
    index = build(b)
    runtime.search_error = IndexCompletionUnknown("unproved native state")
    with pytest.raises(IndexCompletionUnknown):
        b.search(index, torch.ones((1, 3)), top_k=1)
    runtime.search_error = None
    with pytest.raises(IndexCompletionUnknown, match="unproved native state"):
        b.synchronize()
    assert len(b._owners) == 1


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"seen": 0}, "do not share"),
        ({"free_status": 0}, "could not free"),
        ({"exceed": True}, "bypassed"),
    ],
)
def test_allocator_mismatch_and_failed_free_are_unknown(kwargs, reason):
    runtime, owner, calls = bridge(**kwargs)
    with pytest.raises(IndexCompletionUnknown, match=reason):
        runtime._verify_allocator_bridge(owner)
    if "seen" not in kwargs:
        assert owner.probe_allocations  # Keep ambiguous pointers with their owner.
    else:
        assert ("alloc", 4352) not in calls  # Never allocate big until sharing proved.


def args(*extra):
    from sglang.srt.disaggregation.pvd.server import build_parser

    return build_parser().parse_args(
        [
            "--advertise-host",
            "v",
            "--transfer-staging-budget-bytes",
            "1048576",
            "--transfer-max-inflight",
            "4",
            "--total-pages",
            "8",
            "--page-bytes",
            "32",
            *extra,
        ]
    )


def test_launcher_exact_remains_default():
    from sglang.srt.disaggregation.pvd.server import _build_prompt_index

    assert _build_prompt_index(args()) is None
    manager = _build_prompt_index(
        args(
            "--prompt-index-vector-space",
            "model",
            "--prompt-index-budget-bytes",
            "1048576",
        )
    )
    assert manager.backend.device == torch.device("cpu")
    assert manager.backend.name != "cagra"


@pytest.mark.parametrize("missing", ["space", "budget", "cap", "cpu"])
def test_launcher_native_is_explicit(missing):
    from sglang.srt.disaggregation.pvd.server import _validate_args

    settings = {
        "space": ["--prompt-index-vector-space", "model"],
        "budget": ["--prompt-index-budget-bytes", "1048576"],
        "cap": ["--prompt-index-cagra-native-bytes", "4096"],
    }
    argv = ["--prompt-index-backend", "cagra"]
    for name, values in settings.items():
        if name != missing:
            argv.extend(values)
    if missing == "cpu":
        argv.append("--allow-cpu-for-tests")
    with pytest.raises(ValueError, match="CAGRA"):
        _validate_args(args(*argv))


def test_factory_passes_actual_rank_device_and_native_bounds(monkeypatch):
    from sglang.srt.disaggregation.pvd import cagra_backend, server

    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return backend()

    monkeypatch.setattr(cagra_backend, "CagraIndexBackend", factory)
    config = args(
        "--prompt-index-backend",
        "cagra",
        "--prompt-index-vector-space",
        "model",
        "--prompt-index-budget-bytes",
        "1048576",
        "--prompt-index-cagra-native-bytes",
        "8192",
    )
    with pytest.raises(ValueError, match="actual device"):
        server._build_prompt_index(config)
    manager = server._build_prompt_index(config, device="cuda:3")
    assert seen == dict(
        device="cuda:3",
        native_bytes_per_index=8192,
        graph_degree=64,
        intermediate_degree=128,
        itopk_size=512,
    )
    assert manager.budget.snapshot()["staging_bytes"] == 1048576


@pytest.mark.skipif(
    os.getenv("PVD_RUN_CAGRA_NATIVE") != "1",
    reason="opt-in real cuVS/CUDA validation; not executed in CPU suite",
)
@pytest.mark.parametrize("metric", ["ip", "l2"])
def test_native_build_search_scores_and_dispose(metric):
    """No importorskip once requested: missing CUDA/cuVS is an acceptance failure."""
    b = CagraIndexBackend(
        device="cuda:0",
        native_bytes_per_index=256 << 20,
        graph_degree=16,
        intermediate_degree=32,
        itopk_size=64,
    )
    generator = torch.Generator(device="cpu").manual_seed(79)
    cpu_vectors = torch.randn((2048, 32), generator=generator)
    cpu_queries = torch.randn((4, 32), generator=generator)
    vectors, queries = cpu_vectors.cuda(), cpu_queries.cuda()
    index = b.build(vectors, vector_space="native-smoke", metric=metric)
    try:
        rows, scores = b.search(index, queries, top_k=8)
        selected = cpu_vectors[rows.cpu().long()]
        expected = (
            (selected * cpu_queries[:, None]).sum(-1)
            if metric == "ip"
            else -(selected - cpu_queries[:, None]).square().sum(-1).sqrt()
        )
        torch.testing.assert_close(scores.cpu(), expected, rtol=2e-4, atol=2e-4)
        assert all(len(set(row)) == 8 for row in rows.cpu().tolist())
        assert bool((scores[:, :-1] >= scores[:, 1:]).all())
    finally:
        b.dispose(index)
    assert not b._owners
    assert index.handle.limit.get_allocated_bytes() == 0
