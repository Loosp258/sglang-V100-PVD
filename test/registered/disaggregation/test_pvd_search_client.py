"""Single-shard D search protocol; CPU tensors and local HTTP only."""

import asyncio
from dataclasses import replace

import pytest
import torch
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
from sglang.srt.disaggregation.pvd.index_search import BruteForceIndexBackend
from sglang.srt.disaggregation.pvd.search_client import (
    PVDShardSearchClient,
    SearchRefused,
    SearchReplyError,
    SearchScope,
    SearchTransportError,
    ShardSearchError,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_prompt_index import ident, manager, shard_client, stored_entry


def fixture():
    index = manager()
    store, manifest, _, layout = stored_entry(index)
    store.progress_prompt_indexes()
    identity = ident(manifest.key.transfer_id)
    record = index._entries[identity.entry_transfer_id]
    query = record.vectors[(identity.layer, identity.kv_head)].vectors[3:4].tolist()
    scope = SearchScope(
        manifest.prompt_token_count, layout.page_size, layout.head_dim, "l2"
    )
    return index, store, identity, query, scope


def test_roundtrip_pins_bounds_and_session_ownership():
    async def run():
        _, store, identity, query, scope = fixture()
        async with shard_client(store) as http, ClientSession() as session:
            client = PVDShardSearchClient(str(http.make_url("")), session=session)
            result = await client.search(identity, queries=query, top_k=1, scope=scope)
            assert result.token_ids == (3,)
            assert result.page_ids == (0,)
            assert "index_version" not in result.validated
            pinned = replace(
                identity,
                expected_index_version=result.index_version,
                expected_id_mapping_version=result.id_mapping_version,
            )
            again = await client.search(pinned, queries=query, top_k=1, scope=scope)
            assert "index_version" in again.validated
            await client.close()
            assert not session.closed
            with pytest.raises(ShardSearchError, match="closed"):
                await client.search(identity, queries=query, top_k=1, scope=scope)

    asyncio.run(run())


def test_pinned_batch_roundtrip_preserves_each_head_identity_and_order():
    async def run():
        index, store, identity, query, scope = fixture()
        other = replace(identity, kv_head=1)
        record = index._entries[identity.entry_transfer_id]
        other_query = record.vectors[(0, 1)].vectors[3:4].tolist()
        async with shard_client(store) as http:
            client = PVDShardSearchClient(str(http.make_url("")))
            try:
                first = await client.search(
                    identity, queries=query, top_k=1, scope=scope
                )
                pins = dict(
                    expected_index_version=first.index_version,
                    expected_id_mapping_version=first.id_mapping_version,
                )
                results = await client.search_many(
                    (
                        (replace(identity, **pins), query, 1, scope),
                        (replace(other, **pins), other_query, 1, scope),
                    )
                )
                assert tuple(result.identity.kv_head for result in results) == (0, 1)
                assert all(result.token_ids == (3,) for result in results)
                assert all("index_version" in result.validated for result in results)
            finally:
                await client.close()

    asyncio.run(run())


def test_batch_refuses_unpinned_request_before_network():
    async def run():
        _, _, identity, query, scope = fixture()
        client = PVDShardSearchClient("http://127.0.0.1:1")
        with pytest.raises(ValueError, match="both version pins"):
            await client.search_many(((identity, query, 1, scope),))
        assert client._session is None
        await client.close()

    asyncio.run(run())


def test_batch_rejects_one_swapped_result_without_returning_partial_selection():
    async def run():
        _, _, identity, query, scope = fixture()
        pinned = replace(
            identity, expected_index_version="v1", expected_id_mapping_version="m1"
        )

        async def answer(request):
            payload = await request.json()
            replies = []
            for item in payload["items"]:
                reply = valid_body(item)
                reply["validated"] += ["index_version", "id_mapping_version"]
                replies.append(reply)
            replies[1]["search_id"] = replies[0]["search_id"]
            return web.json_response(
                {
                    "batch_protocol": payload["batch_protocol"],
                    "batch_id": payload["batch_id"],
                    "results": replies,
                }
            )

        app = web.Application()
        app.router.add_post("/internal/v1/indexes/search-batch", answer)
        async with TestServer(app) as server:
            client = PVDShardSearchClient(str(server.make_url("")))
            try:
                with pytest.raises(SearchReplyError, match="search_id"):
                    await client.search_many(
                        ((pinned, query, 1, scope), (pinned, query, 1, scope))
                    )
            finally:
                await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["mixed_pin", "too_many_items", "too_many_rows"])
def test_batch_server_rejects_unbounded_or_mixed_work_before_index_search(
    monkeypatch, failure
):
    async def run():
        index, store, identity, query, _ = fixture()
        calls = []

        def forbidden_search(*args, **kwargs):
            calls.append(True)
            raise AssertionError("invalid batch reached the index")

        monkeypatch.setattr(index, "search", forbidden_search)
        item = {
            "search_protocol": "pvd.search.v1",
            "search_id": "item-0",
            "transfer_id": identity.entry_transfer_id,
            "vector_space": identity.vector_space,
            "positional_encoding": identity.positional_encoding,
            "layer": 0,
            "kv_head": 0,
            "expected_index_version": "version-1",
            "expected_id_mapping_version": "mapping-1",
            "queries": query,
            "top_k": 1,
        }
        if failure == "mixed_pin":
            items = [
                item,
                {**item, "search_id": "item-1", "expected_index_version": "other"},
            ]
        elif failure == "too_many_items":
            items = [{**item, "search_id": f"item-{i}"} for i in range(33)]
        else:
            items = [
                {**item, "search_id": f"item-{i}", "queries": query * 64}
                for i in range(9)
            ]
        async with shard_client(store) as http:
            response = await http.post(
                "/internal/v1/indexes/search-batch",
                json={
                    "batch_protocol": "pvd.search.batch.v1",
                    "batch_id": "batch",
                    "items": items,
                },
            )
            assert response.status == 400
            assert (await response.json())["error"].startswith("invalid request:")
        assert not calls

    asyncio.run(run())


@pytest.mark.parametrize("condition", ["not_ready", "capacity", "stale", "failed"])
def test_refusals_are_explicit_and_not_retried(condition):
    async def run():
        index, store, identity, query, scope = fixture()
        if condition == "not_ready":
            index.close(identity.entry_transfer_id)
            index.note_kv_readable(identity.entry_transfer_id)
        elif condition == "capacity":
            index.budget = TransferBudget(staging_bytes=1, max_inflight=1)
        elif condition == "failed":
            index.close(identity.entry_transfer_id)
            index.note_kv_readable(identity.entry_transfer_id)
            gate = index.gate_for(identity.entry_transfer_id)
            for _ in range(gate.max_build_attempts):
                gate.begin_build()
                gate.mark_failed("permanent failure")
        else:
            identity = replace(identity, expected_index_version="stale")
        async with shard_client(store) as http:
            client = PVDShardSearchClient(str(http.make_url("")))
            try:
                with pytest.raises(SearchRefused) as caught:
                    await client.search(identity, queries=query, top_k=1, scope=scope)
                assert caught.value.retryable is (
                    condition in ("not_ready", "capacity")
                )
                if condition == "capacity":
                    assert caught.value.status == 507
            finally:
                await client.close()

    asyncio.run(run())


def valid_body(request):
    return {
        **{
            k: request[k]
            for k in (
                "search_protocol",
                "search_id",
                "transfer_id",
                "vector_space",
                "positional_encoding",
                "layer",
                "kv_head",
            )
        },
        "index_version": "v1",
        "id_mapping_version": "m1",
        "metric": "l2",
        "token_ids": [3],
        "page_ids": [0],
        "scores": [0.0],
        "validated": [
            "vector_space",
            "entry_transfer_id",
            "positional_encoding",
            "layer",
            "kv_head",
        ],
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("search_id", "old-request"),
        ("search_protocol", "old-protocol"),
        ("transfer_id", "other-entry"),
        ("vector_space", "other-model"),
        ("positional_encoding", "none"),
        ("layer", 1),
        ("kv_head", True),
        ("index_version", ""),
        ("id_mapping_version", None),
        ("metric", "ip"),
        ("token_ids", [8]),
        ("token_ids", [-1]),
        ("token_ids", [True]),
        ("token_ids", [3, 3]),
        ("token_ids", []),
        ("token_ids", [2, 3]),
        ("page_ids", [1]),
        ("page_ids", [False]),
        ("scores", [float("nan")]),
        ("scores", [1.0]),
        ("scores", []),
        ("validated", []),
        ("validated", ["index_version"]),
    ],
)
def test_corrupt_or_stale_reply_is_rejected(field, value):
    async def run():
        _, _, identity, query, scope = fixture()
        calls = []

        async def answer(request):
            payload = await request.json()
            calls.append(payload)
            body = valid_body(payload)
            body[field] = value
            return web.json_response(body)

        app = web.Application()
        app.router.add_post("/internal/v1/indexes/search", answer)
        async with TestServer(app) as server:
            client = PVDShardSearchClient(str(server.make_url("")))
            try:
                with pytest.raises(SearchReplyError):
                    await client.search(identity, queries=query, top_k=1, scope=scope)
                assert len(calls) == 1
                assert "expected_index_version" not in calls[0]
            finally:
                await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["timeout", "oversize", "non_json", "cancel"])
def test_transport_and_reply_limits(mode):
    async def run():
        _, _, identity, query, scope = fixture()
        unblock = asyncio.Event()
        started = asyncio.Event()

        async def answer(request):
            started.set()
            if mode in ("timeout", "cancel"):
                await unblock.wait()
            if mode == "non_json":
                return web.Response(text="not JSON")
            return web.Response(text="x" * 4096)

        app = web.Application()
        app.router.add_post("/internal/v1/indexes/search", answer)
        async with TestServer(app) as server:
            client = PVDShardSearchClient(
                str(server.make_url("")), timeout_seconds=0.1, max_response_bytes=1024
            )
            try:
                if mode == "cancel":
                    task = asyncio.create_task(
                        client.search(identity, queries=query, top_k=1, scope=scope)
                    )
                    await asyncio.wait_for(started.wait(), timeout=1)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    return
                error = SearchTransportError if mode == "timeout" else SearchReplyError
                with pytest.raises(error):
                    await client.search(identity, queries=query, top_k=1, scope=scope)
            finally:
                unblock.set()
                await client.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "field,value",
    [
        ("layer", 0.5),
        ("layer", False),
        ("kv_head", "0"),
        ("kv_head", True),
        ("queries", [[float("inf")]]),
        ("queries", [[True]]),
        ("queries", [[1e100]]),
        ("queries", [[10**1000]]),
    ],
)
def test_http_identity_and_values_are_not_coerced(field, value):
    async def run():
        _, store, identity, query, _ = fixture()
        payload = {
            "transfer_id": identity.entry_transfer_id,
            "vector_space": identity.vector_space,
            "positional_encoding": identity.positional_encoding,
            "layer": 0,
            "kv_head": 0,
            "queries": query,
            "top_k": 1,
        }
        payload[field] = value
        async with shard_client(store) as http:
            response = await http.post("/internal/v1/indexes/search", json=payload)
            assert response.status == 400
            error = (await response.json())["error"]
            if field == "queries":
                assert "finite float32 numbers" in error
            else:
                assert f"{field} must be a non-negative integer" in error

    asyncio.run(run())


@pytest.mark.parametrize("chunk_rows", [1, 2, 16])
def test_l2_reference_does_not_lose_nearby_large_vectors(chunk_rows):
    backend = BruteForceIndexBackend(chunk_rows=chunk_rows)
    vectors = torch.tensor([[10000.0, 10000.0], [10001.0, 10000.0]])
    index = backend.build(vectors, vector_space="test", metric="l2")
    rows, scores = backend.search(index, vectors[1:], top_k=2)
    assert rows.tolist() == [[1, 0]]
    assert scores.tolist() == [[0.0, -1.0]]


@pytest.mark.parametrize(
    "queries,top_k,error",
    [
        ([], 1, "1..64"),
        ([[0.0]] * 65, 1, "1..64"),
        ([[0.0]], 1, "dimensions"),
        ([[True] * 8], 1, "finite numbers"),
        ([[float("nan")] * 8], 1, "finite numbers"),
        ([[0.0] * 8], True, "positive integer"),
        ([[0.0] * 8], 9, "limit"),
    ],
)
def test_invalid_input_never_creates_a_session(queries, top_k, error):
    async def run():
        client = PVDShardSearchClient("http://127.0.0.1:1")
        scope = SearchScope(8, 4, 8, "l2")
        try:
            with pytest.raises(ValueError, match=error):
                await client.search(
                    ident("entry"), queries=queries, top_k=top_k, scope=scope
                )
            assert client._session is None
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("version", ["index_version", "id_mapping_version"])
def test_reply_cannot_claim_a_pin_check_and_return_another_version(version):
    async def run():
        _, _, identity, query, scope = fixture()
        identity = replace(identity, **{f"expected_{version}": "caller-version"})

        async def answer(request):
            payload = await request.json()
            assert payload[f"expected_{version}"] == "caller-version"
            body = valid_body(payload)
            body["validated"].append(version)
            return web.json_response(body)

        app = web.Application()
        app.router.add_post("/internal/v1/indexes/search", answer)
        async with TestServer(app) as server:
            client = PVDShardSearchClient(str(server.make_url("")))
            try:
                with pytest.raises(SearchReplyError, match=f"{version} mismatch"):
                    await client.search(identity, queries=query, top_k=1, scope=scope)
            finally:
                await client.close()

    asyncio.run(run())
