"""Exact retrieval backend, logical selection, and the merge policy.

CPU only, no cuVS, no GPU. These tests fix the contract a CAGRA backend must
satisfy and give the exact ground truth its recall will be measured against.
They establish nothing about CAGRA itself or about V100S support.
"""

import pytest
import torch
import torch.utils._python_dispatch
import torch.utils._pytree
from sglang.srt.disaggregation.pvd.index_search import (
    BruteForceIndexBackend,
    BuiltIndex,
    IdMapping,
    IndexSearchError,
    merge_selections,
    resolve_device,
    same_device,
    select,
)
from torch.multiprocessing.reductions import StorageWeakRef as _StorageWeakRef

SPACE = "target/model-8b"


def backend():
    return BruteForceIndexBackend()


def test_grouped_small_ip_search_matches_individual_searches():
    candidate = backend()
    rng = torch.Generator().manual_seed(47)
    indexes = tuple(
        candidate.build(
            torch.randn(17, 8, generator=rng), vector_space=SPACE, metric="ip"
        )
        for _ in range(24)
    )
    queries = tuple(torch.randn(7, 8, generator=rng) for _ in indexes)
    assert candidate.grouped_search_footprint(
        indexes=indexes, num_queries=7, top_k=4
    ) > 0
    grouped = candidate.search_grouped(indexes, queries, top_k=4)
    assert len(grouped) == len(indexes)
    for index, query, (rows, scores) in zip(indexes, queries, grouped):
        expected_rows, expected_scores = candidate.search(index, query, top_k=4)
        assert torch.equal(rows, expected_rows)
        torch.testing.assert_close(scores, expected_scores, atol=1e-5, rtol=1e-5)


def test_grouped_exact_keeps_lower_row_first_on_equal_scores():
    candidate = backend()
    vectors = torch.ones(4, 3)
    indexes = tuple(
        candidate.build(vectors, vector_space=SPACE, metric="ip") for _ in range(2)
    )
    results = candidate.search_grouped(
        indexes, (torch.ones(1, 3), torch.ones(1, 3)), top_k=3
    )
    assert [rows.tolist() for rows, _ in results] == [[[0, 1, 2]]] * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_grouped_small_ip_search_matches_individual_cuda():
    candidate = BruteForceIndexBackend(device="cuda:0")
    rng = torch.Generator().manual_seed(47)
    indexes = tuple(
        candidate.build(
            torch.randn(17, 128, generator=rng).to("cuda:0"),
            vector_space=SPACE,
            metric="ip",
        )
        for _ in range(24)
    )
    queries = tuple(
        torch.randn(7, 128, generator=rng).to("cuda:0") for _ in indexes
    )
    grouped = candidate.search_grouped(indexes, queries, top_k=4)
    for index, query, (rows, scores) in zip(indexes, queries, grouped):
        expected_rows, expected_scores = candidate.search(index, query, top_k=4)
        assert torch.equal(rows, expected_rows)
        torch.testing.assert_close(scores, expected_scores, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("variant", ["l2", "rows", "query_count", "nonfinite"])
def test_grouped_exact_refuses_incompatible_inputs(variant):
    candidate = backend()
    metric = "l2" if variant == "l2" else "ip"
    second_rows = 5 if variant == "rows" else 4
    indexes = (
        candidate.build(torch.eye(4), vector_space=SPACE, metric=metric),
        candidate.build(
            torch.ones(second_rows, 4), vector_space=SPACE, metric=metric
        ),
    )
    first = torch.ones(2, 4)
    second = torch.ones(1 if variant == "query_count" else 2, 4)
    if variant == "nonfinite":
        second[0, 0] = float("nan")
    with pytest.raises(IndexSearchError):
        candidate.search_grouped(indexes, (first, second), top_k=1)


def build(vectors=None, metric="ip"):
    vectors = torch.eye(4) if vectors is None else vectors
    return backend().build(vectors, vector_space=SPACE, metric=metric)


def mapping(tokens=(0, 1, 2, 3), page_size=2, version="map-v1"):
    return IdMapping(version, tokens, page_size)


# --------------------------------------------------------------------------
# Id mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"version": ""},
        {"token_ids": ()},
        {"token_ids": [0, 1]},
        {"token_ids": (0, -1)},
        {"token_ids": (0, True)},
        {"page_size": 0},
        {"page_size": True},
    ],
)
def test_an_incoherent_id_mapping_is_refused(kwargs):
    base = dict(version="map-v1", token_ids=(0, 1), page_size=2)
    base.update(kwargs)
    with pytest.raises(IndexSearchError):
        IdMapping(**base)


def test_rows_map_to_tokens_and_pages():
    m = mapping(tokens=(10, 11, 12, 13), page_size=4)
    assert m.token_of(2) == 12
    assert m.page_of(2) == 3
    assert len(m) == 4


@pytest.mark.parametrize("row", [-1, 4, True, 1.0])
def test_a_row_outside_the_mapping_is_refused(row):
    with pytest.raises(IndexSearchError):
        mapping().token_of(row)


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


def test_a_built_index_records_its_space_metric_and_shape():
    index = build(torch.randn(16, 8))
    assert (index.vector_space, index.metric, index.dim, index.count) == (
        SPACE,
        "ip",
        8,
        16,
    )


def test_an_unknown_metric_is_refused():
    with pytest.raises(IndexSearchError, match="metric must be one of"):
        backend().build(torch.eye(4), vector_space=SPACE, metric="cosine")


@pytest.mark.parametrize("bad", ["", "  ", None, 7])
def test_a_vector_space_is_required(bad):
    with pytest.raises(IndexSearchError, match="vector_space"):
        backend().build(torch.eye(4), vector_space=bad, metric="ip")


@pytest.mark.parametrize(
    "vectors",
    [
        torch.empty(0, 4),
        torch.empty(4, 0),
        torch.ones(4),
        torch.ones(2, 2, 2),
        torch.tensor([[float("nan"), 1.0], [0.0, 1.0]]),
        torch.tensor([[float("inf"), 1.0], [0.0, 1.0]]),
    ],
)
def test_unusable_vectors_are_refused(vectors):
    with pytest.raises(IndexSearchError):
        backend().build(vectors, vector_space=SPACE, metric="ip")


def test_a_built_index_does_not_alias_the_caller_s_tensor():
    """Zeroing the source row after build must not change the stored score."""
    vectors = torch.eye(4)
    index = build(vectors)
    vectors[0, 0] = 0.0
    rows, scores = backend().search(index, torch.eye(4)[:1], top_k=1)
    assert int(rows[0, 0]) == 0
    assert pytest.approx(float(scores[0, 0])) == 1.0


# --------------------------------------------------------------------------
# Searching
# --------------------------------------------------------------------------


def test_inner_product_finds_the_nearest_basis_vector():
    index = build()
    rows, scores = backend().search(index, torch.eye(4)[2:3], top_k=1)
    assert int(rows[0, 0]) == 2
    assert pytest.approx(float(scores[0, 0])) == 1.0


def test_l2_finds_the_closest_vector():
    vectors = torch.tensor([[0.0, 0.0], [10.0, 10.0], [1.0, 1.0]])
    index = build(vectors, metric="l2")
    rows, _ = backend().search(index, torch.tensor([[1.1, 1.1]]), top_k=1)
    assert int(rows[0, 0]) == 2


def test_results_are_ordered_best_first():
    vectors = torch.tensor([[1.0, 0.0], [0.9, 0.0], [0.1, 0.0]])
    index = build(vectors)
    rows, scores = backend().search(index, torch.tensor([[1.0, 0.0]]), top_k=3)
    assert rows[0].tolist() == [0, 1, 2]
    assert scores[0].tolist() == sorted(scores[0].tolist(), reverse=True)


def test_ties_break_on_the_lower_row_deterministically():
    index = build(torch.ones(4, 2))
    for _ in range(5):
        rows, _ = backend().search(index, torch.ones(1, 2), top_k=4)
        assert rows[0].tolist() == [0, 1, 2, 3]


def test_several_queries_are_answered_independently():
    index = build()
    rows, _ = backend().search(index, torch.eye(4)[[3, 1]], top_k=1)
    assert rows[:, 0].tolist() == [3, 1]


def test_a_query_of_the_wrong_dimension_is_refused():
    with pytest.raises(IndexSearchError, match="queries have dim"):
        backend().search(build(), torch.ones(1, 7), top_k=1)


def test_top_k_beyond_the_index_is_refused():
    with pytest.raises(IndexSearchError, match="exceeds the 4 indexed"):
        backend().search(build(), torch.eye(4)[:1], top_k=5)


@pytest.mark.parametrize("bad", [0, -1, True, 1.0])
def test_an_invalid_top_k_is_refused(bad):
    with pytest.raises(IndexSearchError, match="top_k"):
        backend().search(build(), torch.eye(4)[:1], top_k=bad)


def test_a_non_index_is_refused():
    with pytest.raises(IndexSearchError, match="built index is required"):
        backend().search({"count": 4}, torch.eye(4)[:1], top_k=1)


# --------------------------------------------------------------------------
# Selection is logical, never addresses
# --------------------------------------------------------------------------


def test_a_selection_returns_tokens_and_pages_not_rows():
    index = build()
    selection = select(
        backend(),
        index,
        torch.eye(4)[2:3],
        layer=1,
        mapping=mapping((5, 6, 7, 8), 4),
        top_k=1,
    )
    assert selection.token_ids == (7,)
    assert selection.page_ids == (1,)
    assert selection.layer == 1
    assert selection.id_mapping_version == "map-v1"
    assert selection.metric == "ip"


def test_one_token_chosen_by_two_queries_is_selected_once_at_its_best_score():
    index = build()
    selection = select(
        backend(), index, torch.eye(4)[[1, 1]], layer=0, mapping=mapping(), top_k=1
    )
    assert selection.token_ids == (1,)
    assert len(selection.scores) == 1


def test_a_selection_is_ordered_by_score():
    vectors = torch.tensor([[1.0, 0.0], [0.5, 0.0], [0.1, 0.0]])
    index = build(vectors)
    selection = select(
        backend(),
        index,
        torch.tensor([[1.0, 0.0]]),
        layer=0,
        mapping=mapping((0, 1, 2), 2),
        top_k=3,
    )
    assert selection.token_ids == (0, 1, 2)
    assert list(selection.scores) == sorted(selection.scores, reverse=True)


def test_pages_are_deduplicated_and_sorted():
    index = build()
    selection = select(
        backend(),
        index,
        torch.eye(4),
        layer=0,
        mapping=mapping((0, 1, 2, 3), 2),
        top_k=1,
    )
    assert selection.page_ids == (0, 1)


def test_a_mapping_that_does_not_cover_the_index_is_refused():
    with pytest.raises(IndexSearchError, match="id mapping covers"):
        select(
            backend(),
            build(),
            torch.eye(4)[:1],
            layer=0,
            mapping=mapping((0, 1)),
            top_k=1,
        )


def test_a_mapping_is_mandatory():
    with pytest.raises(IndexSearchError, match="id mapping is required"):
        select(backend(), build(), torch.eye(4)[:1], layer=0, mapping=None, top_k=1)


@pytest.mark.parametrize("layer", [-1, True, 1.0])
def test_an_invalid_layer_is_refused(layer):
    with pytest.raises(IndexSearchError, match="layer"):
        select(
            backend(),
            build(),
            torch.eye(4)[:1],
            layer=layer,
            mapping=mapping(),
            top_k=1,
        )


# --------------------------------------------------------------------------
# Merging requires a named policy
# --------------------------------------------------------------------------


def make_selection(layer, tokens, version="map-v1"):
    index = build()
    base = select(
        backend(),
        index,
        torch.eye(4)[:1],
        layer=layer,
        mapping=mapping(version=version),
        top_k=1,
    )
    return type(base)(
        layer=layer,
        token_ids=tuple(tokens),
        page_ids=tuple(sorted({t // 2 for t in tokens})),
        scores=tuple(float(len(tokens) - i) for i in range(len(tokens))),
        metric="ip",
        id_mapping_version=version,
    )


@pytest.mark.parametrize("policy", ["", "global_top_k", "best", None, 7])
def test_an_unnamed_or_unknown_merge_policy_is_refused(policy):
    selections = [make_selection(0, (1, 2))]
    with pytest.raises(IndexSearchError, match="merge policy must be one of"):
        merge_selections(selections, policy=policy)


def test_per_layer_keeps_each_layer_s_own_choice():
    merged = merge_selections(
        [make_selection(0, (1, 2)), make_selection(1, (2, 3))], policy="per_layer"
    )
    assert merged == {0: (1, 2), 1: (2, 3)}


def test_union_gives_every_layer_the_combined_tokens():
    merged = merge_selections(
        [make_selection(0, (1, 2)), make_selection(1, (2, 3))], policy="union"
    )
    assert merged == {0: (1, 2, 3), 1: (1, 2, 3)}


def test_intersection_keeps_only_what_every_layer_chose():
    merged = merge_selections(
        [make_selection(0, (1, 2)), make_selection(1, (2, 3))], policy="intersection"
    )
    assert merged == {0: (2,), 1: (2,)}


def test_merging_across_different_id_mappings_is_refused():
    with pytest.raises(IndexSearchError, match="different id mappings"):
        merge_selections(
            [make_selection(0, (1,)), make_selection(1, (2,), version="map-v2")],
            policy="union",
        )


def test_a_duplicate_layer_is_refused():
    with pytest.raises(IndexSearchError, match="duplicate layer"):
        merge_selections(
            [make_selection(0, (1,)), make_selection(0, (2,))], policy="union"
        )


def test_merging_nothing_is_refused():
    with pytest.raises(IndexSearchError, match="nothing to merge"):
        merge_selections([], policy="union")


# --------------------------------------------------------------------------
# The backend is swappable
# --------------------------------------------------------------------------


def test_select_works_through_any_backend_implementing_the_contract():
    """What a CAGRA backend has to satisfy: same call, same result shape."""

    class AlwaysFirstBackend(BruteForceIndexBackend):
        name = "always_first"

        def search(self, index, queries, *, top_k):
            rows = torch.zeros(queries.shape[0], top_k, dtype=torch.long)
            return rows, torch.ones(queries.shape[0], top_k)

    index = BuiltIndex(SPACE, "ip", 4, 4, torch.eye(4))
    selection = select(
        AlwaysFirstBackend(),
        index,
        torch.eye(4)[1:2],
        layer=0,
        mapping=mapping(),
        top_k=1,
    )
    assert selection.token_ids == (0,)


# --------------------------------------------------------------------------
# Device policy
#
# The exact backend declares where it lives instead of inheriting whatever
# device the vectors happened to be on. Without that, a V worker whose KV
# pool is on cuda:0 builds a CUDA index, the HTTP route hands it a CPU query,
# and the mismatch surfaces as a runtime error on a GPU box only -- never in
# a CPU test run. A dtype cast is not a device move, so the two are always
# stated separately.
# --------------------------------------------------------------------------


def test_the_exact_backend_declares_its_device_rather_than_inferring_one():
    assert backend().device == torch.device("cpu")
    assert BruteForceIndexBackend(device="cpu").device == torch.device("cpu")


def test_a_built_index_lands_on_the_backends_device_not_the_vectors():
    built = backend().build(torch.randn(6, 4), vector_space=SPACE, metric="ip")
    assert built.handle.device == torch.device("cpu")
    assert built.handle.dtype is torch.float32


def test_a_dtype_cast_is_not_a_device_move():
    """The assumption this policy exists to refuse, written down."""
    vectors = torch.randn(6, 4, dtype=torch.float64)
    assert vectors.to(torch.float32).device == vectors.device
    # So the backend names the device separately; it does not rely on the cast.
    built = backend().build(vectors, vector_space=SPACE, metric="ip")
    assert built.handle.device == backend().device


def test_the_index_never_aliases_the_vectors_it_was_built_from():
    vectors = torch.randn(6, 4)
    built = backend().build(vectors, vector_space=SPACE, metric="ip")
    assert built.handle.data_ptr() != vectors.data_ptr()
    vectors.zero_()
    assert built.handle.abs().sum() > 0


def test_a_query_is_placed_on_the_backends_device_before_searching():
    b = backend()
    vectors = torch.eye(4)
    built = b.build(vectors, vector_space=SPACE, metric="ip")
    rows, scores = b.search(built, torch.eye(4), top_k=1)
    assert rows.tolist() == [[0], [1], [2], [3]]
    assert scores.device == b.device


def test_an_index_from_another_device_policy_is_refused_not_guessed():
    """A handle built under a different policy must not be searched blind."""

    class Elsewhere(BruteForceIndexBackend):
        @property
        def device(self):
            return torch.device("meta")

    built = backend().build(torch.randn(6, 4), vector_space=SPACE, metric="ip")
    with pytest.raises(IndexSearchError, match="device policy"):
        Elsewhere().search(built, torch.randn(2, 4), top_k=1)


def test_the_backend_bounds_what_a_build_and_a_search_cost():
    b = backend()
    # A built index is float32 over rows x dim, and the declaration says so.
    built = b.build(torch.randn(16, 8), vector_space=SPACE, metric="ip")
    assert b.build_footprint(16, 8, metric="ip") == built.handle.numel() * 4
    # Stable direct-difference L2 needs no retained squared norms.
    built_l2 = b.build(torch.randn(16, 8), vector_space=SPACE, metric="l2")
    assert b.build_footprint(16, 8, metric="l2") == sum(
        t.numel() * t.element_size() for t in built_l2.retained_tensors()
    )
    assert b.build_footprint(16, 8, metric="l2") == b.build_footprint(
        16, 8, metric="ip"
    )
    # A build's transient peak is declared separately from what it retains.
    assert b.build_scratch_footprint(16, 8, metric="l2") > 0
    # Search scratch grows with the query count but is capped in the index:
    # past one chunk, adding rows changes nothing, which is what "bounded"
    # has to mean for a budget to be a budget.
    small = b.search_footprint(16, 8, 1, 1)
    assert b.search_footprint(16, 8, 4, 1) > small
    wide = b.search_footprint(b.chunk_rows * 8, 8, 1, 1)
    wider = b.search_footprint(b.chunk_rows * 64, 8, 1, 1)
    assert wide == wider
    for bad in ((0, 8, 1, 1), (16, 8, 0, 1), (16, 8, 1, 0)):
        with pytest.raises(IndexSearchError):
            b.search_footprint(*bad)
    with pytest.raises(IndexSearchError):
        b.build_footprint(16, 8, metric="cosine")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device in this environment"
)
def test_a_cuda_query_is_accepted_by_the_cpu_backend():
    """UNVERIFIED without GPU hardware: skipped on every CPU-only run.

    This is the case the HTTP route cannot produce but a future in-process
    probe can: Q living on the device while the exact index lives on the
    host. The policy says the backend places it, so this must not raise.
    """
    b = backend()
    built = b.build(torch.eye(4), vector_space=SPACE, metric="ip")
    rows, _ = b.search(built, torch.eye(4, device="cuda"), top_k=1)
    assert rows.tolist() == [[0], [1], [2], [3]]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device in this environment"
)
def test_a_cuda_backend_keeps_its_index_and_queries_on_the_device():
    """UNVERIFIED without GPU hardware: skipped on every CPU-only run."""
    b = BruteForceIndexBackend(device="cuda")
    built = b.build(torch.eye(4), vector_space=SPACE, metric="ip")
    assert built.handle.device.type == "cuda"
    rows, scores = b.search(built, torch.eye(4), top_k=1)
    assert scores.device.type == "cuda"
    assert rows.tolist() == [[0], [1], [2], [3]]


def test_a_backend_that_declares_another_device_builds_and_searches_there():
    """The declaration is followed, not the incoming tensor's device.

    'meta' stands in for cuda:N so this runs on a CPU-only box: it is a real
    non-CPU device as far as placement is concerned, which is the property
    under test. It says nothing about CUDA kernels.
    """

    class Elsewhere(BruteForceIndexBackend):
        @property
        def device(self):
            return torch.device("meta")

    b = Elsewhere()
    # The vectors arrive on the host, as extraction from a CPU pool leaves
    # them. A build that relied on the dtype cast would leave them there.
    built = b.build(torch.randn(6, 4), vector_space=SPACE, metric="ip")
    assert built.handle.device == torch.device("meta")
    rows, scores = b.search(built, torch.randn(2, 4), top_k=1)
    assert rows.device == torch.device("meta")
    assert scores.device == torch.device("meta")


# --------------------------------------------------------------------------
# Measured allocation behaviour
#
# The footprint methods are claims about memory, so they are checked against
# memory, not against the fact that they were called. What is measured here
# is *tensor storage bytes newly allocated inside the call* -- the CPU
# allocator's arena, Python object overhead and any library workspace are
# outside this instrument and are not claimed to be bounded by it.
# --------------------------------------------------------------------------


class _PeakTensorBytes(torch.utils._python_dispatch.TorchDispatchMode):
    """Peak live bytes of tensor storages allocated while active.

    Keyed on the storage, so views and aliases count once. Frees are seen by
    sweeping expired storage weak references before each op, which CPython's
    refcounting makes prompt. Storages passed to ``exclude`` are pre-seeded
    at zero: a slice of an index is the first time this mode *sees* that
    index's storage, and counting retained memory as freshly allocated would
    make every search look index-proportional.
    """

    def __init__(self, exclude=()):
        self._live = {}
        self.bytes = 0
        self.peak = 0
        for item in exclude:
            if isinstance(item, torch.Tensor):
                ref = _StorageWeakRef(item.untyped_storage())
                self._live[ref.cdata] = (ref, 0)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for key in [k for k, (ref, _) in self._live.items() if ref.expired()]:
            self.bytes -= self._live.pop(key)[1]
        for item in torch.utils._pytree.tree_flatten(out)[0]:
            if isinstance(item, torch.Tensor):
                try:
                    storage = item.untyped_storage()
                except Exception:  # pragma: no cover - exotic tensors
                    continue
                ref = _StorageWeakRef(storage)
                if ref.cdata not in self._live:
                    nbytes = storage.nbytes()
                    self._live[ref.cdata] = (ref, nbytes)
                    self.bytes += nbytes
        self.peak = max(self.peak, self.bytes)
        return out


def peak_tensor_bytes(fn, exclude=()):
    tracker = _PeakTensorBytes(exclude)
    with tracker:
        fn()
    return tracker.peak


def test_the_measurement_instrument_itself_is_sound():
    """A bound checked with a broken ruler is not a bound."""
    kept = []
    assert peak_tensor_bytes(lambda: kept.append(torch.empty(1000, 1000))) >= 4_000_000

    def transient():
        for _ in range(5):
            torch.empty(1000, 1000)

    # Five allocations, one live at a time: a peak, not a total.
    assert peak_tensor_bytes(transient) < 8_000_000
    existing = torch.empty(1000, 1000)
    assert peak_tensor_bytes(lambda: existing[0:10], exclude=(existing,)) == 0


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_a_search_stays_inside_its_declared_scratch(metric):
    """The exact number that motivated chunking: 16,908 declared against
    541,184 actually allocated, because torch.cdist and a full-width sort
    both scale with the index."""
    b = backend()
    rows, dim, nq, k = 1024, 128, 1, 1
    built = b.build(torch.randn(rows, dim), vector_space=SPACE, metric=metric)
    q = torch.randn(nq, dim)
    b.search(built, q, top_k=k)  # warm up lazily-initialised kernels
    declared = b.search_footprint(rows, dim, nq, k)
    peak = peak_tensor_bytes(
        lambda: b.search(built, q, top_k=k),
        exclude=(built.handle, built.aux, q),
    )
    assert peak <= declared, f"{metric}: allocated {peak} against {declared} declared"


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_search_scratch_does_not_grow_with_the_index(metric):
    """The property that makes the declaration a bound at all.

    A formula that merely happens to be large enough for one index size is
    not a bound; scratch must stop growing once the index exceeds one pass.
    """
    b = backend()
    dim, nq, k = 128, 1, 1
    peaks = []
    for rows in (b.chunk_rows * 2, b.chunk_rows * 16):
        built = b.build(torch.randn(rows, dim), vector_space=SPACE, metric=metric)
        q = torch.randn(nq, dim)
        b.search(built, q, top_k=k)
        peaks.append(
            peak_tensor_bytes(
                lambda: b.search(built, q, top_k=k),
                exclude=(built.handle, built.aux, q),
            )
        )
    small, large = peaks
    # An eightfold index for the same query: the same working set.
    assert large <= small * 1.1, f"{metric}: scratch grew {small} -> {large}"
    assert b.search_footprint(b.chunk_rows * 2, dim, nq, k) == b.search_footprint(
        b.chunk_rows * 16, dim, nq, k
    )


@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("rows,dim,nq,k", [(1024, 128, 64, 512), (200, 32, 4, 8)])
def test_search_stays_inside_its_bound_across_shapes(metric, rows, dim, nq, k):
    b = backend()
    built = b.build(torch.randn(rows, dim), vector_space=SPACE, metric=metric)
    q = torch.randn(nq, dim)
    b.search(built, q, top_k=k)
    declared = b.search_footprint(rows, dim, nq, k)
    peak = peak_tensor_bytes(
        lambda: b.search(built, q, top_k=k),
        exclude=(built.handle, built.aux, q),
    )
    assert peak <= declared


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_a_build_stays_inside_retained_plus_declared_scratch(metric):
    """Retained bytes are not peak build bytes; both are declared."""
    b = backend()
    rows, dim = 1024, 128
    vectors = torch.randn(rows, dim)
    b.build(vectors, vector_space=SPACE, metric=metric)
    declared = b.build_footprint(rows, dim, metric=metric) + b.build_scratch_footprint(
        rows, dim, metric=metric
    )
    peak = peak_tensor_bytes(
        lambda: b.build(vectors, vector_space=SPACE, metric=metric),
        exclude=(vectors,),
    )
    assert peak <= declared, f"{metric}: allocated {peak} against {declared} declared"


def test_chunking_changes_no_result():
    """Bounding memory must not quietly change what is retrieved."""
    vectors = torch.randn(300, 16)
    queries = torch.randn(5, 16)
    for metric in ("l2", "ip"):
        whole = BruteForceIndexBackend(chunk_rows=10_000)
        split = BruteForceIndexBackend(chunk_rows=7)
        a = whole.search(
            whole.build(vectors, vector_space=SPACE, metric=metric), queries, top_k=9
        )
        c = split.search(
            split.build(vectors, vector_space=SPACE, metric=metric), queries, top_k=9
        )
        assert a[0].tolist() == c[0].tolist()
        assert torch.allclose(a[1], c[1], atol=1e-5)


def test_ties_still_resolve_to_the_lower_row_across_chunk_boundaries():
    """The carried-forward merge must keep the old tie-break."""
    vectors = torch.ones(40, 4)  # every row identical: all scores tie
    b = BruteForceIndexBackend(chunk_rows=7)
    built = b.build(vectors, vector_space=SPACE, metric="ip")
    rows, _ = b.search(built, torch.ones(1, 4), top_k=5)
    assert rows.tolist() == [[0, 1, 2, 3, 4]]


def test_an_l2_index_needs_no_auxiliary_norm_cache():
    b = backend()
    built = b.build(torch.randn(6, 4), vector_space=SPACE, metric="l2")
    stripped = BuiltIndex(
        vector_space=built.vector_space,
        metric="l2",
        dim=built.dim,
        count=built.count,
        handle=built.handle,
    )
    assert built.aux is None
    queries = built.handle[2:3]
    rows, scores = b.search(stripped, queries, top_k=1)
    assert rows.tolist() == [[2]]
    assert scores.tolist() == [[0.0]]


# --------------------------------------------------------------------------
# Device aliases
#
# "cpu" and "cpu:0" are the same device; so are "cuda" and the cuda:N this
# process would use. Different GPU indices are NOT, and that must stay hard.
# --------------------------------------------------------------------------


def test_cpu_aliases_are_the_same_execution_device():
    assert resolve_device("cpu:0") == torch.device("cpu")
    assert resolve_device(torch.device("cpu", 0)) == torch.device("cpu")
    assert same_device(torch.device("cpu"), torch.device("cpu", 0))


def test_a_backend_configured_as_cpu0_can_search_what_it_built():
    """The whole round trip, because build and search disagreed before."""
    b = BruteForceIndexBackend(device="cpu:0")
    assert b.device == torch.device("cpu")
    built = b.build(torch.eye(4), vector_space=SPACE, metric="ip")
    assert built.handle.device == torch.device("cpu")
    rows, _ = b.search(built, torch.eye(4), top_k=1)
    assert rows.tolist() == [[0], [1], [2], [3]]


def test_a_cpu_index_that_does_not_exist_is_refused():
    with pytest.raises(IndexSearchError, match="no CPU 1"):
        resolve_device("cpu:1")
    with pytest.raises(IndexSearchError, match="no CPU 1"):
        BruteForceIndexBackend(device="cpu:1")


def test_different_gpu_indices_are_never_treated_as_aliases():
    """Normalising spellings must not soften the check that matters.

    Constructed as device objects so this runs without CUDA: reading an
    index off the wrong GPU is the failure a device check exists to catch.
    """
    assert not same_device(torch.device("cuda", 0), torch.device("cuda", 1))
    assert not same_device(torch.device("cpu"), torch.device("cuda", 0))
    assert not same_device(torch.device("meta"), torch.device("cpu"))


def test_unqualified_cuda_is_refused_when_there_is_no_cuda():
    if torch.cuda.is_available():  # pragma: no cover - covered by the GPU test
        pytest.skip("CUDA is available; resolution is exercised elsewhere")
    with pytest.raises(IndexSearchError, match="no CUDA device"):
        resolve_device("cuda")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device in this environment"
)
def test_unqualified_cuda_resolves_to_the_current_device():
    """UNVERIFIED without GPU hardware: skipped on every CPU-only run."""
    resolved = resolve_device("cuda")
    assert resolved == torch.device("cuda", torch.cuda.current_device())
    b = BruteForceIndexBackend(device="cuda")
    assert b.device == resolved
    built = b.build(torch.eye(4), vector_space=SPACE, metric="ip")
    # The tensor reports cuda:N; the configured spelling was bare "cuda".
    assert built.handle.device == resolved
    rows, _ = b.search(built, torch.eye(4), top_k=1)
    assert rows.tolist() == [[0], [1], [2], [3]]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device in this environment"
)
def test_an_explicitly_indexed_cuda_device_round_trips():
    """UNVERIFIED without GPU hardware: skipped on every CPU-only run."""
    index = torch.cuda.current_device()
    b = BruteForceIndexBackend(device=f"cuda:{index}")
    built = b.build(torch.eye(4), vector_space=SPACE, metric="l2")
    assert built.handle.device == torch.device("cuda", index)
    rows, _ = b.search(built, torch.eye(4), top_k=1)
    assert rows.tolist() == [[0], [1], [2], [3]]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_an_index_from_another_gpu_is_still_refused():
    """UNVERIFIED without two GPUs: skipped on every CPU-only run."""
    other = BruteForceIndexBackend(device="cuda:1")
    built = other.build(torch.eye(4), vector_space=SPACE, metric="ip")
    with pytest.raises(IndexSearchError, match="device policy"):
        BruteForceIndexBackend(device="cuda:0").search(built, torch.eye(4), top_k=1)


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_build_scratch_does_not_grow_with_the_index(metric):
    """Build temporaries must be bounded too, not just search ones.

    Validating and squaring the whole index at once are both
    index-proportional. Measured against retained bytes rather than against
    the total, because the retained copy is charged separately and would
    otherwise mask a temporary that does scale.
    """
    b = backend()
    dim = 128
    scratches = []
    for rows in (b.chunk_rows * 2, b.chunk_rows * 16):
        vectors = torch.randn(rows, dim)
        b.build(vectors, vector_space=SPACE, metric=metric)
        peak = peak_tensor_bytes(
            lambda: b.build(vectors, vector_space=SPACE, metric=metric),
            exclude=(vectors,),
        )
        retained = b.build_footprint(rows, dim, metric=metric)
        declared = b.build_scratch_footprint(rows, dim, metric=metric)
        assert peak <= retained + declared
        scratches.append(peak - retained)
    small, large = scratches
    # Eight times the index, the same transient working set.
    assert large <= max(small, 0) * 1.1 + 4096, f"{metric}: {small} -> {large}"
    assert b.build_scratch_footprint(
        b.chunk_rows * 2, dim, metric=metric
    ) == b.build_scratch_footprint(b.chunk_rows * 16, dim, metric=metric)


def test_a_backend_reporting_an_unresolved_device_is_still_matched():
    """A spelling is not a device.

    A subclass may return an unnormalised spelling from ``device``; the
    tensors it built report the canonical one. Comparing the two directly
    would refuse an index the backend had just produced itself.
    """

    class SpelledOut(BruteForceIndexBackend):
        @property
        def device(self):
            return torch.device("cpu", 0)

    b = SpelledOut()
    built = b.build(torch.eye(4), vector_space=SPACE, metric="ip")
    assert built.handle.device == torch.device("cpu")
    assert b.device != built.handle.device  # the spellings genuinely differ
    rows, _ = b.search(built, torch.eye(4), top_k=1)
    assert rows.tolist() == [[0], [1], [2], [3]]
