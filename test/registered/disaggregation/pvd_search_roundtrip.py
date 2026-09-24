"""Reusable local-HTTP exact-search acceptance check.

The caller supplies Prompt KV and a prediction pipeline. The strict smoke uses
actual target-model K/Q; unit tests supply synthetic tensors. P->V is a local
byte copy with FakeTransferEngine registration, NOT evidence of RDMA.
"""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace

import torch
from aiohttp.test_utils import TestServer
from sglang.srt.disaggregation.pvd.control_server import create_shard_app
from sglang.srt.disaggregation.pvd.kv_packer import (
    describe_kv_layout,
    pack_full_prompt_kv_head_shard,
)
from sglang.srt.disaggregation.pvd.probe_search import (
    ProbeSearchRoute,
    ProbeSearchSession,
)
from sglang.srt.disaggregation.pvd.prompt_index import (
    PromptIndexManager,
    SearchRequestIdentity,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED, QueryHeadMapping
from sglang.srt.disaggregation.pvd.protocol import (
    KVEntryKey,
    KVEntryManifest,
    KVLayoutSignature,
    KVShardManifest,
)
from sglang.srt.disaggregation.pvd.search_client import (
    PVDShardSearchClient,
    SearchRefused,
    SearchScope,
)
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec, pack_sparse_kv
from sglang.srt.disaggregation.pvd.sparse_union import union_query_head_selections
from sglang.srt.disaggregation.pvd.sparse_working_set import CPUSparseWorkingSet
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from sglang.srt.disaggregation.pvd.vector_store import VectorKVStore


def independent_topk(prompt_k, query_rows, top_k):
    """Scalar dot products + Python sorting, not the index's selection helper."""
    best = {}
    for query in query_rows:
        scores = [
            sum(float(a) * float(b) for a, b in zip(query, key, strict=True))
            for key in prompt_k.tolist()
        ]
        for token in sorted(range(len(scores)), key=lambda t: (-scores[t], t))[:top_k]:
            best[token] = max(best.get(token, -float("inf")), scores[token])
    ordered = sorted(best, key=lambda t: (-best[t], t))
    return tuple(ordered), tuple(best[t] for t in ordered)


async def verify_exact_roundtrip(pool, prefix, pipeline, *, page_size=2, top_k=2):
    space = pipeline.probe_config.target_model_id
    query_heads = pipeline.probe_config.head_count
    assert pipeline.probe_config.head_start == 0
    total_heads = pool.k_buffer[0].shape[1]
    assert total_heads == 2, "this acceptance fixture exercises two V head shards"
    prompt_tokens = len(prefix.tokens)
    page_count = (prompt_tokens + page_size - 1) // page_size
    extra = deepcopy(describe_kv_layout(pool))
    extra["component_token_shapes"] = [
        [1, s[1]] for s in extra["component_token_shapes"]
    ]
    extra["component_bytes_per_token"] = [
        v // total_heads for v in extra["component_bytes_per_token"]
    ]
    layout = KVLayoutSignature(
        model_id=space,
        model_revision="offline-fixture",
        kv_dtype="torch.float32",
        page_size=page_size,
        num_layers=len(pool.k_buffer),
        total_kv_heads=total_heads,
        kv_heads_per_rank=1,
        head_dim=pool.k_buffer[0].shape[-1],
        tp_size=2,
        pp_size=1,
        tensor_layout=extra["tensor_layout"],
        extra=extra,
    )
    packed = [
        pack_full_prompt_kv_head_shard(
            pool,
            range(page_count),
            page_size=page_size,
            head_start=rank,
            head_count=1,
        )
        for rank in range(2)
    ]
    shards = [
        KVShardManifest(
            rank=rank,
            rail="cpu-test",
            expected_bytes=value.expected_bytes,
            page_count=page_count,
            last_page_valid_tokens=prompt_tokens % page_size or page_size,
            layer_start=0,
            layer_end=len(pool.k_buffer),
        )
        for rank, value in enumerate(packed)
    ]
    manifest = KVEntryManifest(
        key=KVEntryKey.new(space, prefix.request_id),
        layout=layout,
        prompt_token_count=prompt_tokens,
        shards=shards,
    )
    mapping = QueryHeadMapping(query_heads, total_heads)
    scope = SearchScope(prompt_tokens, page_size, layout.head_dim, "ip")
    checked, refusals, max_error, sparse_checked, union_checked = 0, 0, 0.0, 0, 0
    for rank in range(2):
        budget = TransferBudget(4 << 20, 8)
        index = PromptIndexManager(vector_space=space, metric="ip", budget=budget)
        store = VectorKVStore(
            rank=rank,
            world_size=2,
            rail="cpu-test",
            device="cpu",
            total_pages=page_count * 2,
            page_bytes=packed[rank].expected_bytes // page_count,
            endpoint=f"cpu-v{rank}",
            transfer_engine=FakeTransferEngine(),
            allow_cpu_for_tests=True,
            prompt_index=index,
        )
        session = ProbeSearchSession(prefix.request_id, manifest.key.transfer_id)
        try:
            entry = store.create_entry(manifest)
            store.begin_p_write(manifest.key)
            offset = entry.allocation.start_page * store.page_bytes
            store.pool[offset : offset + packed[rank].expected_bytes] = packed[
                rank
            ].tensor
            store.commit_p_write(manifest.key, packed[rank].expected_bytes)
            window = session.begin(
                prefix,
                target_tokens=prefix.committed_position + 4,
                query_positions=tuple(
                    range(len(prefix.tokens), len(prefix.tokens) + 2)
                ),
            )
            routes = tuple(
                ProbeSearchRoute(
                    query_head=qhead,
                    identity=SearchRequestIdentity(
                        vector_space=space,
                        positional_encoding=ROPE_APPLIED,
                        entry_transfer_id=manifest.key.transfer_id,
                        layer=layer,
                        kv_head=rank,
                    ),
                    scope=scope,
                    top_k=top_k,
                )
                for layer in pipeline.probe_config.layers
                for qhead in range(query_heads)
                if mapping.kv_head_for(qhead) == rank
            )
            prepared = session.prepare(
                window, pipeline, routes=routes, head_mapping=mapping
            )
            async with TestServer(create_shard_app(store)) as server:
                client = PVDShardSearchClient(str(server.make_url("")))
                try:
                    first = prepared.queries[0]
                    try:
                        await client.search(
                            first.route.identity,
                            queries=first.rows,
                            top_k=top_k,
                            scope=scope,
                        )
                    except SearchRefused as exc:
                        assert exc.code == "index_not_ready"
                        refusals += 1
                    else:
                        raise AssertionError("index-not-ready request was accepted")
                    assert store.progress_prompt_indexes()["built"] == 1
                    await session.search(prepared, client)
                    selection = session.take_selection(window)
                    query_groups = selection.query_groups or tuple(
                        (index,) for index in range(len(selection.queries))
                    )
                    for members, result in zip(
                        query_groups, selection.selections, strict=True
                    ):
                        rows = tuple(
                            row
                            for index in members
                            for row in selection.queries[index].rows
                        )
                        expected_ids, expected_scores = independent_topk(
                            pool.k_buffer[result.identity.layer][:prompt_tokens, rank],
                            rows,
                            top_k,
                        )
                        assert result.token_ids == expected_ids
                        assert result.page_ids == tuple(
                            sorted({t // page_size for t in expected_ids})
                        )
                        torch.testing.assert_close(
                            torch.tensor(result.scores),
                            torch.tensor(expected_scores),
                            atol=2e-5,
                            rtol=2e-4,
                        )
                        max_error = max(
                            max_error,
                            max(
                                abs(a - b)
                                for a, b in zip(
                                    result.scores, expected_scores, strict=True
                                )
                            ),
                        )
                        assert all(t < prompt_tokens for t in result.token_ids), (
                            "padding selected"
                        )
                        checked += len(members)
                        # Step 2: the exact selected K/V bytes for this
                        # same-KV-head response, with no D install. The index
                        # identity comes from
                        # V's gate, independently of the returned selection.
                        current = index.gate_for(manifest.key.transfer_id).descriptor
                        spec = SparseKVSpec(
                            request_id=prefix.request_id,
                            incarnation=window.incarnation,
                            operation_id=window.operation_id,
                            target_tokens=window.target_tokens,
                            entry_transfer_id=manifest.key.transfer_id,
                            index_version=result.index_version,
                            id_mapping_version=result.id_mapping_version,
                            layout_fingerprint=layout.fingerprint,
                            layer=result.identity.layer,
                            kv_head=result.identity.kv_head,
                            token_ids=result.token_ids,
                        )
                        payload_budget = TransferBudget(1 << 20, 1)
                        with pack_sparse_kv(
                            store.pool[offset : offset + packed[rank].expected_bytes],
                            layout=layout,
                            shard=shards[rank],
                            spec=spec,
                            entry_transfer_id=manifest.key.transfer_id,
                            index_version=current.index_version,
                            id_mapping_version=current.id_mapping_version,
                            budget=payload_budget,
                        ) as payload:
                            expected_kv = torch.stack(
                                [
                                    component[list(result.token_ids), rank]
                                    for component in (
                                        pool.k_buffer[spec.layer],
                                        pool.v_buffer[spec.layer],
                                    )
                                ]
                            )
                            torch.testing.assert_close(
                                payload.tensor, expected_kv, rtol=0, atol=0
                            )
                        assert payload_budget.snapshot()["used_staging_bytes"] == 0
                        sparse_checked += 1
                    # CPU per-shard bank only: this is not a coordinated TP
                    # install or a live Decode attention backend.
                    union_specs = union_query_head_selections(
                        selection,
                        mapping=mapping,
                        layout_fingerprint=layout.fingerprint,
                        max_union_tokens=prompt_tokens,
                    )
                    bank_budget = TransferBudget(1 << 20, 2)
                    packing_budget = TransferBudget(1 << 20, len(union_specs))
                    bank = CPUSparseWorkingSet(
                        request_id=prefix.request_id,
                        incarnation=window.incarnation,
                        entry_transfer_id=manifest.key.transfer_id,
                        layout_fingerprint=layout.fingerprint,
                        expected_groups=tuple(
                            (s.layer, s.kv_head) for s in union_specs
                        ),
                        prompt_tokens=prompt_tokens,
                        head_dim=layout.head_dim,
                        max_union_tokens=prompt_tokens,
                        budget=bank_budget,
                    )
                    try:
                        initial_specs = tuple(
                            replace(
                                s,
                                operation_id="initial-fixture",
                                target_tokens=0,
                                token_ids=tuple(range(prompt_tokens)),
                            )
                            for s in union_specs
                        )
                        for specs in (initial_specs, union_specs):
                            with ExitStack() as stack:
                                rows = [
                                    stack.enter_context(
                                        pack_sparse_kv(
                                            store.pool[
                                                offset : offset
                                                + packed[rank].expected_bytes
                                            ],
                                            layout=layout,
                                            shard=shards[rank],
                                            spec=spec,
                                            entry_transfer_id=manifest.key.transfer_id,
                                            index_version=current.index_version,
                                            id_mapping_version=current.id_mapping_version,
                                            budget=packing_budget,
                                        )
                                    )
                                    for spec in specs
                                ]
                                bank.stage(rows)
                            assert packing_budget.snapshot()["used_staging_bytes"] == 0
                            bank.install(specs[0].target_tokens)
                        with bank.read() as groups:
                            for spec in union_specs:
                                _, data = groups[(spec.layer, spec.kv_head)]
                                expected = torch.stack(
                                    [
                                        component[list(spec.token_ids), rank]
                                        for component in (
                                            pool.k_buffer[spec.layer],
                                            pool.v_buffer[spec.layer],
                                        )
                                    ]
                                )
                                torch.testing.assert_close(
                                    data, expected, rtol=0, atol=0
                                )
                                union_checked += 1
                    finally:
                        bank.close()
                    assert bank_budget.snapshot()["used_staging_bytes"] == 0
                    pinned = replace(
                        first.route.identity,
                        expected_index_version=selection.selections[0].index_version,
                        expected_id_mapping_version=selection.selections[
                            0
                        ].id_mapping_version,
                    )
                    for bad in (
                        replace(pinned, vector_space="another-target"),
                        replace(pinned, positional_encoding="none"),
                        replace(pinned, kv_head=1 - rank),
                        replace(pinned, expected_index_version="stale-index"),
                        replace(pinned, expected_id_mapping_version="stale-mapping"),
                    ):
                        try:
                            await client.search(
                                bad, queries=first.rows, top_k=top_k, scope=scope
                            )
                        except SearchRefused:
                            refusals += 1
                        else:
                            raise AssertionError("bad search identity was accepted")
                    # Pins must reject an index rebuilt for the same Entry.
                    index.close(manifest.key.transfer_id)
                    index.note_kv_readable(manifest.key.transfer_id)
                    assert store.progress_prompt_indexes()["built"] == 1
                    try:
                        await client.search(
                            pinned, queries=first.rows, top_k=top_k, scope=scope
                        )
                    except SearchRefused:
                        refusals += 1
                    else:
                        raise AssertionError("stale rebuilt index accepted")
                finally:
                    await client.close()
        finally:
            session.close()
            store.close()
        assert budget.snapshot()["used_staging_bytes"] == 0
    return {
        "real_http": True,
        "v_shards": 2,
        "layer_head_results_checked": checked,
        "negative_checks": refusals,
        "max_score_abs_error": max_error,
        "sparse_kv_payloads_checked": sparse_checked,
        "cpu_union_groups_installed": union_checked,
        "rdma_cagra_or_quality_validated": False,
    }
