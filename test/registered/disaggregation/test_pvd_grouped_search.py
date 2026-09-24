"""Grouped HTTP retains GQA provenance, atomic versions and exact unions."""

import asyncio
from dataclasses import replace

import pytest
from sglang.srt.disaggregation.pvd.probe_search import (
    StaleProbeSearch,
    _group_search_queries,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient
from sglang.srt.disaggregation.pvd.sparse_union import union_query_head_selections
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_probe_search import setup
from test_pvd_prompt_index import shard_client


def prepare_heads(heads, positions=(5,)):
    index, store, session, window, pipeline, probe, route = setup(heads=heads)
    probe.budget = TransferBudget(max(4096, 64 * heads), 1)
    if positions != window.query_positions:
        session.invalidate()
        window = session.begin(
            window.prefix, target_tokens=4, query_positions=positions
        )
    prepared = session.prepare(
        window,
        pipeline,
        routes=tuple(replace(route, query_head=head) for head in range(heads)),
        head_mapping=QueryHeadMapping(heads, 1),
    )
    return index, store, session, window, prepared


@pytest.mark.parametrize(
    "field,value",
    [
        ("entry_transfer_id", "other-entry"),
        ("vector_space", "other-space"),
        ("positional_encoding", "none"),
        ("layer", 1),
        ("kv_head", 1),
        ("expected_index_version", "pinned-index"),
        ("expected_id_mapping_version", "pinned-map"),
    ],
)
def test_identity_fields_cannot_coalesce(field, value):
    *_, prepared = prepare_heads(2)
    first, second = prepared.queries
    second = replace(
        second,
        route=replace(
            second.route, identity=replace(second.route.identity, **{field: value})
        ),
    )
    assert _group_search_queries((first, second), (0, 0)) == ((0,), (1,))


@pytest.mark.parametrize("difference", ["source", "scope", "top_k", "query_version"])
def test_other_grouping_keys_are_explicit(difference):
    *_, prepared = prepare_heads(2)
    first, second = prepared.queries
    sources = (0, 1) if difference == "source" else (0, 0)
    if difference == "scope":
        second = replace(
            second,
            route=replace(second.route, scope=replace(second.route.scope, page_size=2)),
        )
    elif difference == "top_k":
        second = replace(second, route=replace(second.route, top_k=2))
    elif difference == "query_version":
        second = replace(second, query_version="another-query-version")
    assert _group_search_queries((first, second), sources) == ((0,), (1,))


def test_multi_position_chunks_preserve_every_head_and_row():
    *_, prepared = prepare_heads(65, (4, 5))
    groups = _group_search_queries(prepared.queries, (0,) * 65)
    assert tuple(map(len, groups)) == (32, 32, 1)
    assert tuple(index for group in groups for index in group) == tuple(range(65))
    assert [sum(len(prepared.queries[i].rows) for i in group) for group in groups] == [
        64,
        64,
        2,
    ]


def test_qwen_28_layers_28_q_heads_become_112_gqa_searches():
    *_, prepared = prepare_heads(784)
    queries = tuple(
        replace(
            query,
            route=replace(
                query.route,
                query_head=i % 28,
                identity=replace(
                    query.route.identity, layer=i // 28, kv_head=(i % 28) // 7
                ),
            ),
        )
        for i, query in enumerate(prepared.queries)
    )
    sources = tuple(query.route.identity.kv_head // 2 for query in queries)
    groups = _group_search_queries(queries, sources)
    assert len(groups) == 112
    assert all(len(group) == 7 for group in groups)
    assert sorted(i for group in groups for i in group) == list(range(784))


def test_grouped_http_matches_independent_exact_top_k_union():
    async def run():
        index, store, session, window, pipeline, probe, route = setup(heads=7)
        vectors = index._entries[window.entry_transfer_id].vectors[(0, 0)].vectors
        original_capture = probe.capture

        def capture(prefix, prediction):
            queries = original_capture(prefix, prediction)
            for head in range(7):
                probe.tensor[1, head].copy_(vectors[head % len(vectors)])
            return queries

        probe.capture = capture
        prepared = session.prepare(
            window,
            pipeline,
            routes=tuple(replace(route, query_head=head, top_k=2) for head in range(7)),
            head_mapping=QueryHeadMapping(7, 1),
        )

        class RecordingClient(PVDShardSearchClient):
            def __init__(self, url):
                super().__init__(url)
                self.row_counts = []

            async def search(self, *args, **kwargs):
                self.row_counts.append(len(kwargs["queries"]))
                return await super().search(*args, **kwargs)

        async with shard_client(store) as http:
            client = RecordingClient(str(http.make_url("")))
            try:
                expected = set()
                for query in prepared.queries:
                    result = await client.search(
                        query.route.identity,
                        queries=query.rows,
                        top_k=2,
                        scope=route.scope,
                    )
                    expected.update(result.token_ids)
                client.row_counts.clear()
                await session.search(prepared, client)
                selection = session.take_selection(window)
                assert client.row_counts == [7]
                assert selection.query_groups == (tuple(range(7)),)
                assert set(selection.selections[0].token_ids) == expected
                specs = union_query_head_selections(
                    selection,
                    mapping=QueryHeadMapping(7, 1),
                    layout_fingerprint="layout",
                    max_union_tokens=route.scope.prompt_tokens,
                )
                assert specs[0].token_ids == tuple(sorted(expected))
            finally:
                await client.close()

    asyncio.run(run())


def test_chunked_http_pins_versions_and_publishes_once():
    async def run():
        _, store, session, window, prepared = prepare_heads(65)

        class RecordingClient(PVDShardSearchClient):
            def __init__(self, url):
                super().__init__(url)
                self.calls = []

            async def search(self, identity, **kwargs):
                assert session._ready is None
                self.calls.append((identity, len(kwargs["queries"])))
                return await super().search(identity, **kwargs)

        async with shard_client(store) as http:
            client = RecordingClient(str(http.make_url("")))
            try:
                await session.search(prepared, client)
                result = session.take_selection(window)
                assert [count for _, count in client.calls] == [64, 1]
                assert client.calls[0][0].expected_index_version is None
                assert (
                    client.calls[1][0].expected_index_version
                    == result.selections[0].index_version
                )
                assert (
                    client.calls[1][0].expected_id_mapping_version
                    == result.selections[0].id_mapping_version
                )
                assert len(result.query_groups) == 2
            finally:
                await client.close()

    asyncio.run(run())


def test_cancel_second_chunk_never_publishes_partial_union():
    async def run():
        _, store, session, window, prepared = prepare_heads(65)
        entered = asyncio.Event()

        class DelayedClient(PVDShardSearchClient):
            calls = 0

            async def search(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    entered.set()
                    await asyncio.Future()
                return await super().search(*args, **kwargs)

        async with shard_client(store) as http:
            client = DelayedClient(str(http.make_url("")))
            try:
                task = asyncio.create_task(session.search(prepared, client))
                await asyncio.wait_for(entered.wait(), 3)
                assert session._ready is None
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                with pytest.raises(StaleProbeSearch):
                    session.take_selection(window)
                assert session._ready is None
            finally:
                await client.close()

    asyncio.run(run())
