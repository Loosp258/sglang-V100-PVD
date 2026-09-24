"""User-confirmed GQA union semantics; never rank scores across heads/layers."""

from dataclasses import replace

import pytest
from sglang.srt.disaggregation.pvd.prediction import CommittedPrefix
from sglang.srt.disaggregation.pvd.probe_search import (
    PreparedProbeQuery,
    ProbeSearchRoute,
    ProbeSelection,
    ProbeWindow,
)
from sglang.srt.disaggregation.pvd.prompt_index import SearchRequestIdentity
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED, QueryHeadMapping
from sglang.srt.disaggregation.pvd.search_client import SearchScope, ShardSearchResult
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from sglang.srt.disaggregation.pvd.sparse_union import union_query_head_selections


def fixture():
    window = ProbeWindow(
        "inc", "op", "entry", CommittedPrefix("r", (1, 2), 0, "v"), 4, (2,)
    )
    queries, results = [], []
    for layer in (0, 1):
        for head in range(4):
            ident = SearchRequestIdentity(
                "target", ROPE_APPLIED, "entry", layer, head // 2
            )
            route = ProbeSearchRoute(head, ident, SearchScope(8, 2, 1, "ip"), 2)
            queries.append(PreparedProbeQuery(route, "qv", ((1.0,),)))
            tokens = (layer, head + 2)
            results.append(
                ShardSearchResult(
                    ident,
                    f"index-{head // 2}",
                    f"map-{head // 2}",
                    tokens,
                    tuple(sorted({t // 2 for t in tokens})),
                    (1e6 * (head + 1), -1e6),
                    "ip",
                    (),
                )
            )
    return ProbeSelection(window, tuple(queries), tuple(results))


def plan(value, limit=8):
    return union_query_head_selections(
        value,
        mapping=QueryHeadMapping(4, 2),
        layout_fingerprint="layout",
        max_union_tokens=limit,
    )


def test_union_is_per_layer_kv_head_and_ignores_cross_head_score_scales():
    result = plan(fixture())
    assert {(s.layer, s.kv_head): s.token_ids for s in result} == {
        (0, 0): (0, 2, 3),
        (0, 1): (0, 4, 5),
        (1, 0): (1, 2, 3),
        (1, 1): (1, 4, 5),
    }
    assert all(s.request_id == "r" and s.operation_id == "op" for s in result)


def test_over_capacity_is_refused_not_truncated():
    with pytest.raises(SparsePayloadError, match="capacity exceeded"):
        plan(fixture(), limit=2)


@pytest.mark.parametrize("bad", [0, True, None, 1.5])
def test_union_limit_is_required_and_integral(bad):
    with pytest.raises(SparsePayloadError, match="explicitly positive"):
        plan(fixture(), limit=bad)


def test_missing_query_head_is_refused():
    value = fixture()
    with pytest.raises(SparsePayloadError, match="every Q head"):
        plan(replace(value, queries=value.queries[1:], selections=value.selections[1:]))


@pytest.mark.parametrize(
    "field,new", [("index_version", "stale"), ("id_mapping_version", "stale")]
)
def test_mixed_index_versions_cannot_be_unioned(field, new):
    value = fixture()
    rows = list(value.selections)
    rows[1] = replace(rows[1], **{field: new})
    with pytest.raises(SparsePayloadError, match="mixed versions"):
        plan(replace(value, selections=tuple(rows)))


def test_duplicate_query_head_is_refused():
    value = fixture()
    with pytest.raises(SparsePayloadError, match="duplicate Q head"):
        plan(
            replace(
                value,
                queries=value.queries + (value.queries[0],),
                selections=value.selections + (value.selections[0],),
            )
        )


def grouped_fixture():
    value = fixture()
    members = tuple((i, i + 1) for i in range(0, len(value.queries), 2))
    results = []
    for indices in members:
        first = value.selections[indices[0]]
        tokens = tuple(
            sorted({t for i in indices for t in value.selections[i].token_ids})
        )
        results.append(
            replace(
                first,
                token_ids=tokens,
                page_ids=tuple(sorted({t // 2 for t in tokens})),
                scores=(1.0,) * len(tokens),
            )
        )
    return replace(value, selections=tuple(results), query_groups=members)


def test_grouped_provenance_has_identical_union_and_capacity_semantics():
    assert plan(grouped_fixture()) == plan(fixture())
    with pytest.raises(SparsePayloadError, match="capacity exceeded"):
        plan(grouped_fixture(), limit=2)


@pytest.mark.parametrize(
    "members",
    [
        ((0,), (2, 3), (4, 5), (6, 7)),
        ((0, 0, 1), (2, 3), (4, 5), (6, 7)),
        ((0, True), (2, 3), (4, 5), (6, 7)),
        ((0, 1), (), (2, 3, 4, 5, 6, 7)),
    ],
)
def test_grouped_provenance_must_cover_every_query_once(members):
    with pytest.raises(SparsePayloadError, match="provenance"):
        plan(replace(grouped_fixture(), query_groups=members))


def test_grouped_provenance_cannot_merge_different_kv_heads():
    value = grouped_fixture()
    with pytest.raises(SparsePayloadError, match="incompatible grouped"):
        plan(replace(value, query_groups=((0, 2), (1, 3), (4, 5), (6, 7))))


def test_grouped_provenance_still_rejects_duplicate_query_head():
    value = grouped_fixture()
    queries = list(value.queries)
    queries[1] = replace(queries[1], route=replace(queries[1].route, query_head=0))
    with pytest.raises(SparsePayloadError, match="duplicate Q head"):
        plan(replace(value, queries=tuple(queries)))
