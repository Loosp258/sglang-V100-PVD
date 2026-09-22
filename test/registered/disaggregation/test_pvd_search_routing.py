"""Actual two-V HTTP indexes, one-D queries; no model/GPU/RDMA evidence."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace

import pytest
from sglang.srt.disaggregation.pvd.prediction import (
    DraftConfig,
    FakeDraftProvider,
    PredictionPipeline,
    ProbeConfig,
    snapshot_committed,
)
from sglang.srt.disaggregation.pvd.probe_search import (
    ProbeSearchRoute,
    ProbeSearchSession,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.search_client import (
    PVDShardSearchClient,
    ShardSearchError,
)
from sglang.srt.disaggregation.pvd.search_routing import RoutedShardSearchClient
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
from sglang.srt.disaggregation.pvd.sparse_union import union_query_head_selections
from test_pvd_probe_search import ScratchProbe
from test_pvd_prompt_index import (
    SPACE,
    build_entry,
    ident,
    make_store,
    manager,
    shard_client,
)
from test_pvd_prompt_vectors import pack_shard


def layout(storage, tp):
    heads = storage.total_kv_heads // tp
    return replace(
        storage,
        tp_size=tp,
        kv_heads_per_rank=heads,
        extra={
            **storage.extra,
            "component_bytes_per_token": [
                n // storage.kv_heads_per_rank * heads
                for n in storage.extra["component_bytes_per_token"]
            ],
            "component_token_shapes": [
                [heads, *shape[1:]] for shape in storage.extra["component_token_shapes"]
            ],
        },
    )


def router(storage, compute, clients, **kwargs):
    options = dict(
        storage_layout=storage,
        compute_layout=compute,
        compute_rank=0,
        entry_transfer_id="entry",
        prompt_tokens=8,
        vector_space=SPACE,
        metric="l2",
        clients=clients,
    )
    options.update(kwargs)
    return RoutedShardSearchClient(**options)


@pytest.mark.parametrize(
    "tp,rank,sources",
    [
        (1, 0, {0, 1}),
        (2, 0, {0}),
        (2, 1, {1}),
        (4, 0, {0}),
        (4, 1, {0}),
        (4, 2, {1}),
        (4, 3, {1}),
    ],
)
def test_compute_rank_is_not_storage_rank(tp, rank, sources):
    _, storage, _, _, _ = build_entry()
    clients = {s: PVDShardSearchClient(f"http://v{s}") for s in sources}
    r = router(storage, layout(storage, tp), clients, compute_rank=rank)
    assert set(r.groups.values()) == sources
    heads = storage.total_kv_heads // tp
    assert set(r.groups) == {
        (layer, head)
        for layer in range(storage.num_layers)
        for head in range(rank * heads, (rank + 1) * heads)
    }
    for (layer_id, head), source in r.groups.items():
        assert r.version_scope(ident("entry", layer_id, head)) == source


@pytest.mark.parametrize(
    "fault", ["missing", "extra", "same_url", "wrong_model", "bool_rank", "pp"]
)
def test_incompatible_route_configuration_refused(fault):
    _, storage, _, _, _ = build_entry()
    compute = layout(storage, 1)
    clients = {s: PVDShardSearchClient(f"http://v{s}") for s in (0, 1)}
    kwargs = {}
    if fault == "missing":
        clients.pop(1)
    elif fault == "extra":
        clients[2] = PVDShardSearchClient("http://v2")
    elif fault == "same_url":
        clients[1] = PVDShardSearchClient("http://v0")
    elif fault == "wrong_model":
        compute = replace(compute, model_id="other")
    elif fault == "bool_rank":
        kwargs["compute_rank"] = False
    else:
        storage, compute = replace(storage, pp_size=2), replace(compute, pp_size=2)
    with pytest.raises(ValueError):
        router(storage, compute, clients, **kwargs)


@pytest.mark.parametrize(
    "fault", ["entry", "space", "rope", "layer", "head", "scope", "endpoint"]
)
def test_request_scope_refused_before_http(fault):
    async def run():
        _, storage, _, _, _ = build_entry()
        clients = {s: PVDShardSearchClient(f"http://v{s}") for s in (0, 1)}
        r = router(storage, layout(storage, 1), clients)
        identity, scope = ident("entry"), r.scope
        changes = {
            "entry": dict(entry_transfer_id="another"),
            "space": dict(vector_space="draft"),
            "rope": dict(positional_encoding="none"),
            "layer": dict(layer=99),
            "head": dict(kv_head=99),
        }
        if fault in changes:
            identity = replace(identity, **changes[fault])
        elif fault == "scope":
            scope = replace(scope, prompt_tokens=7)
        else:
            clients[0].base_url = "http://unselected"
        with pytest.raises(ValueError):
            await r.search(
                identity, queries=[[1.0] * scope.head_dim], top_k=1, scope=scope
            )
        assert all(c._session is None for c in clients.values())

    asyncio.run(run())


def selection_plan():
    _, storage, _, _, _ = build_entry()
    clients = {s: PVDShardSearchClient(f"http://v{s}") for s in (0, 1)}
    r = router(storage, layout(storage, 1), clients)
    specs = tuple(
        SparseKVSpec(
            request_id="request",
            incarnation="incarnation",
            operation_id="round",
            target_tokens=8,
            entry_transfer_id="entry",
            index_version=f"index-{source}",
            id_mapping_version=f"map-{source}",
            layout_fingerprint=r.compute_fingerprint,
            layer=layer,
            kv_head=head,
            token_ids=(1, 3),
        )
        for (layer, head), source in r.groups.items()
    )
    return r, specs


def test_source_manifests_keep_distinct_versions_and_both_layouts():
    r, specs = selection_plan()
    plans = r.partition_specs(specs)
    assert len(plans) == 2
    assert {s for p in plans for s in p.decode_specs} == set(specs)
    assert r.compute_fingerprint != r.storage_fingerprint
    for plan in plans:
        for wire, local in zip(plan.manifest.specs, plan.decode_specs, strict=True):
            assert wire.layout_fingerprint == r.storage_fingerprint
            assert local.layout_fingerprint == r.compute_fingerprint
            assert wire == replace(local, layout_fingerprint=r.storage_fingerprint)
            assert wire.index_version == f"index-{plan.storage_rank}"
            assert wire.kv_head // 2 == plan.storage_rank


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "entry", "layout", "tokens", "epoch", "version"]
)
def test_incomplete_or_mixed_source_plan_is_refused(fault):
    r, specs = selection_plan()
    if fault == "missing":
        specs = specs[:-1]
    elif fault == "duplicate":
        specs = specs + (specs[0],)
    else:
        changes = dict(
            entry=dict(entry_transfer_id="other"),
            layout=dict(layout_fingerprint="other"),
            tokens=dict(token_ids=(99,)),
            epoch=dict(operation_id="other"),
            version=dict(index_version="changed-within-source"),
        )[fault]
        specs = (replace(specs[0], **changes), *specs[1:])
    with pytest.raises(ValueError):
        r.partition_specs(specs)


class RecordingClient(PVDShardSearchClient):
    def __init__(self, *args, corrupt=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []
        self.corrupt = corrupt

    async def search(self, identity, **kwargs):
        self.calls.append(identity)
        reply = await super().search(identity, **kwargs)
        if self.corrupt and len(self.calls) == 2:
            return replace(reply, index_version="changed-mid-window")
        return reply


@pytest.mark.parametrize("corrupt", [False, True])
def test_actual_http_two_source_versions_and_gqa_union(corrupt):
    async def run():
        pool, storage, manifest, _, _ = build_entry()
        indexes, stores = {}, {}
        for rank in (0, 1):
            index = indexes[rank] = manager()
            store = stores[rank] = make_store(manifest, index=index, rank=rank)
            packed, shard, _ = pack_shard(pool, storage, rank=rank, prompt_tokens=8)
            entry = store.create_entry(manifest)
            store.begin_p_write(manifest.key)
            offset = entry.allocation.start_page * store.page_bytes
            store.pool[offset : offset + shard.expected_bytes] = packed.tensor
            store.commit_p_write(manifest.key, shard.expected_bytes)
            assert store.progress_prompt_indexes()["built"] == 1
        versions = {
            rank: index.gate_for(manifest.key.transfer_id).descriptor.index_version
            for rank, index in indexes.items()
        }
        assert versions[0] != versions[1]
        async with AsyncExitStack() as stack:
            http = {
                rank: await stack.enter_async_context(shard_client(store))
                for rank, store in stores.items()
            }
            clients = {
                rank: RecordingClient(
                    str(server.make_url("")), corrupt=corrupt and rank == 1
                )
                for rank, server in http.items()
            }
            for client in clients.values():
                stack.push_async_callback(client.close)
            r = router(
                storage,
                layout(storage, 1),
                clients,
                entry_transfer_id=manifest.key.transfer_id,
            )
            vector = (
                indexes[0]
                ._entries[manifest.key.transfer_id]
                .vectors[0, 0]
                .vectors[3]
                .tolist()
            )
            config = DraftConfig("configurable/draft", predict_tokens=2)
            pipeline = PredictionPipeline(
                FakeDraftProvider(config, tokens=(31, 32)),
                ScratchProbe(vector, head_count=8),
                config,
                ProbeConfig(SPACE, (0,), head_count=8),
            )
            session = ProbeSearchSession("request", manifest.key.transfer_id)
            prefix = snapshot_committed("request", [10, 11, 12, 13], 2, "prefix")
            window = session.begin(prefix, target_tokens=4, query_positions=(5,))
            # Alternate sources to catch accidentally resetting version pins.
            routes = tuple(
                ProbeSearchRoute(
                    q, ident(manifest.key.transfer_id, 0, q // 2), r.scope, 1
                )
                for q in (0, 4, 1, 5, 2, 6, 3, 7)
            )
            prepared = session.prepare(
                window, pipeline, routes=routes, head_mapping=QueryHeadMapping(8, 4)
            )
            if corrupt:
                with pytest.raises(ValueError, match="index changed"):
                    await session.search(prepared, r)
                with pytest.raises(ValueError):
                    session.take_selection(window)
                return
            await session.search(prepared, r)
            result = session.take_selection(window)
            for rank, client in clients.items():
                assert len(client.calls) == 4
                assert client.calls[0].expected_index_version is None
                assert all(
                    call.expected_index_version == versions[rank]
                    for call in client.calls[1:]
                )
                assert all(call.kv_head // 2 == rank for call in client.calls)
            specs = union_query_head_selections(
                result,
                mapping=QueryHeadMapping(8, 4),
                layout_fingerprint=r.compute_fingerprint,
                max_union_tokens=2,
            )
            assert {s.kv_head for s in specs} == {0, 1, 2, 3}
            assert all(s.index_version == versions[s.kv_head // 2] for s in specs)
            await r.close()
            assert all(
                not c._closed for c in clients.values()
            )  # Borrowed, not destroyed.
            with pytest.raises(ShardSearchError, match="closed"):
                r.version_scope(ident(manifest.key.transfer_id))

    asyncio.run(run())
