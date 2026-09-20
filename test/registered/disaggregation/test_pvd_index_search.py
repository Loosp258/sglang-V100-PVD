"""Exact retrieval backend, logical selection, and the merge policy.

CPU only, no cuVS, no GPU. These tests fix the contract a CAGRA backend must
satisfy and give the exact ground truth its recall will be measured against.
They establish nothing about CAGRA itself or about V100S support.
"""

import pytest
import torch
from sglang.srt.disaggregation.pvd.index_search import (
    BruteForceIndexBackend,
    BuiltIndex,
    IdMapping,
    IndexSearchError,
    merge_selections,
    select,
)

SPACE = "target/model-8b"


def backend():
    return BruteForceIndexBackend()


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
