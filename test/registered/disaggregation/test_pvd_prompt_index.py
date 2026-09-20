"""The V store driving IndexGate: build, search, release, and independence.

Real VectorKVStore, real packing, real extraction, exact CPU backend. No cuVS,
no GPU. Queries are rows taken from the extracted vectors, so these tests show
that the plumbing preserves identity -- not that retrieval is any good.
"""

import pytest
import torch
from test_pvd_prompt_vectors import FakePool, pack_shard, storage_layout
from test_pvd_vector_lifecycle import DelayedTransferEngine

from sglang.srt.disaggregation.pvd.index_lifecycle import IndexState
from sglang.srt.disaggregation.pvd.index_search import (
    BruteForceIndexBackend,
    IndexSearchError,
)
from sglang.srt.disaggregation.pvd.prompt_index import (
    PromptIndexManager,
    SearchRequestIdentity,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import (
    NO_POSITIONAL_ENCODING,
    ROPE_APPLIED,
)
from sglang.srt.disaggregation.pvd.protocol import KVEntryKey, KVEntryManifest
from sglang.srt.disaggregation.pvd.request_state import EntryShardState
from sglang.srt.disaggregation.pvd.vector_store import VectorKVStore

SPACE = "target/model-8b"
PROMPT_TOKENS = 8


def build_entry(rank=0, prompt_tokens=PROMPT_TOKENS):
    """A manifest/layout pair produced by the real packer, plus its bytes."""
    pool = FakePool()
    layout = storage_layout(pool)
    packed, shard, _ = pack_shard(pool, layout, rank=rank, prompt_tokens=prompt_tokens)
    manifest = KVEntryManifest(
        key=KVEntryKey.new("model-instance", f"req-{rank}"),
        layout=layout,
        prompt_token_count=prompt_tokens,
        shards=[
            pack_shard(pool, layout, rank=r, prompt_tokens=prompt_tokens)[1]
            for r in (0, 1)
        ],
    )
    return pool, layout, manifest, packed, shard


def make_store(manifest, *, index=None, rank=0):
    page_bytes = (
        manifest.shards[rank].expected_bytes // manifest.shards[rank].page_count
    )
    return VectorKVStore(
        rank=rank,
        world_size=2,
        rail=f"mlx5_{rank}",
        device="cpu",
        total_pages=16,
        page_bytes=page_bytes,
        endpoint=f"v{rank}",
        transfer_engine=DelayedTransferEngine(),
        allow_cpu_for_tests=True,
        prompt_index=index,
    )


def stored_entry(index=None, rank=0, prompt_tokens=PROMPT_TOKENS):
    """Create, fill and commit one Entry, exactly as the legacy path does."""
    pool, layout, manifest, packed, shard = build_entry(rank, prompt_tokens)
    store = make_store(manifest, index=index, rank=rank)
    entry = store.create_entry(manifest)
    store.begin_p_write(manifest.key)
    offset = entry.allocation.start_page * store.page_bytes
    store.pool[offset : offset + shard.expected_bytes] = packed.tensor
    store.commit_p_write(manifest.key, shard.expected_bytes)
    return store, manifest, pool, layout


def manager(**kwargs):
    kwargs.setdefault("vector_space", SPACE)
    kwargs.setdefault("metric", "l2")
    return PromptIndexManager(**kwargs)


def ident(transfer_id, layer=0, kv_head=0, **kwargs):
    """A caller's stated query identity. Every field is the caller's own.

    Nothing here is read back out of the manager or the index: that is the
    point of the contract, and a helper that peeked would hide the bug these
    tests exist to prevent.
    """
    kwargs.setdefault("vector_space", SPACE)
    kwargs.setdefault("positional_encoding", ROPE_APPLIED)
    return SearchRequestIdentity(
        entry_transfer_id=transfer_id, layer=layer, kv_head=kv_head, **kwargs
    )


# --------------------------------------------------------------------------
# Off by default
# --------------------------------------------------------------------------


def test_a_store_without_an_index_is_unchanged():
    store, manifest, _, _ = stored_entry()
    assert store.prompt_index is None
    assert store.entries[manifest.key].state is EntryShardState.STORED
    assert store.progress_prompt_indexes() == {
        "built": 0,
        "failed": 0,
        "deferred": 0,
        "skipped": 0,
    }


# --------------------------------------------------------------------------
# Build happens only after the KV is stored, and only when driven
# --------------------------------------------------------------------------


def test_storing_marks_the_gate_readable_but_builds_nothing_yet():
    index = manager()
    store, manifest, _, _ = stored_entry(index)
    gate = index.gate_for(manifest.key.transfer_id)
    assert gate.kv_readable
    assert gate.state is IndexState.ABSENT
    assert gate.deliverable
    assert not gate.searchable


def test_no_gate_is_readable_before_the_entry_is_stored():
    index = manager()
    pool, layout, manifest, packed, shard = build_entry()
    store = make_store(manifest, index=index)
    store.create_entry(manifest)
    store.begin_p_write(manifest.key)
    assert index.gate_for(manifest.key.transfer_id) is None
    assert store.progress_prompt_indexes()["built"] == 0


def test_driving_progress_builds_the_index():
    index = manager()
    store, manifest, _, _ = stored_entry(index)
    assert store.progress_prompt_indexes() == {
        "built": 1,
        "failed": 0,
        "deferred": 0,
        "skipped": 0,
    }
    gate = index.gate_for(manifest.key.transfer_id)
    assert gate.state is IndexState.READY
    assert gate.searchable
    assert gate.descriptor.vector_space == SPACE


def test_a_built_index_is_not_rebuilt_on_later_passes():
    index = manager()
    store, _, _, _ = stored_entry(index)
    store.progress_prompt_indexes()
    assert store.progress_prompt_indexes() == {
        "built": 0,
        "failed": 0,
        "deferred": 0,
        "skipped": 0,
    }


# --------------------------------------------------------------------------
# Delivery never depends on the index
# --------------------------------------------------------------------------


def test_the_entry_is_stored_and_deliverable_with_no_index_built():
    index = manager()
    store, manifest, _, _ = stored_entry(index)
    entry = store.entries[manifest.key]
    assert entry.state is EntryShardState.STORED
    assert not entry.release_requested
    assert index.gate_for(manifest.key.transfer_id).deliverable


def test_a_failed_build_leaves_the_entry_stored_and_deliverable():
    class BrokenBackend:
        name = "broken"

        def build(self, *a, **k):
            raise RuntimeError("cuvs exploded")

        def search(self, *a, **k):  # pragma: no cover - never reached
            raise AssertionError

    index = manager(backend=BrokenBackend())
    store, manifest, _, _ = stored_entry(index)
    assert store.progress_prompt_indexes() == {
        "built": 0,
        "failed": 1,
        "deferred": 0,
        "skipped": 0,
    }
    gate = index.gate_for(manifest.key.transfer_id)
    assert gate.state is IndexState.FAILED
    assert "cuvs exploded" in gate.error
    assert gate.deliverable
    assert store.entries[manifest.key].state is EntryShardState.STORED


def test_a_failed_build_is_retried_up_to_the_bound_then_stops():
    class BrokenBackend:
        name = "broken"

        def build(self, *a, **k):
            raise RuntimeError("transient")

        def search(self, *a, **k):  # pragma: no cover
            raise AssertionError

    index = manager(backend=BrokenBackend(), max_build_attempts=2)
    store, manifest, _, _ = stored_entry(index)
    assert store.progress_prompt_indexes()["failed"] == 1
    assert store.progress_prompt_indexes()["failed"] == 1
    gate = index.gate_for(manifest.key.transfer_id)
    assert gate.exhausted
    assert store.progress_prompt_indexes() == {
        "built": 0,
        "failed": 0,
        "deferred": 0,
        "skipped": 0,
    }


def test_an_unindexable_layout_fails_the_build_not_the_entry():
    """The core fixture layout has a single component; extraction refuses it."""
    index = manager()
    store, manifest, _, _ = stored_entry(index)
    entry = store.entries[manifest.key]
    object.__setattr__(entry.manifest, "layer_end", entry.manifest.layer_start + 1)
    index.gate_for(manifest.key.transfer_id)._state = IndexState.ABSENT
    assert store.progress_prompt_indexes()["failed"] == 1
    assert store.entries[manifest.key].state is EntryShardState.STORED


# --------------------------------------------------------------------------
# Searching
# --------------------------------------------------------------------------


def searchable(rank=0, prompt_tokens=PROMPT_TOKENS):
    index = manager()
    store, manifest, pool, layout = stored_entry(index, rank, prompt_tokens)
    store.progress_prompt_indexes()
    return index, store, manifest, pool, layout


def test_a_search_returns_the_original_token_and_page():
    index, _, manifest, _, layout = searchable()
    transfer_id = manifest.key.transfer_id
    record = index._entries[transfer_id]
    for (layer, head), item in sorted(record.vectors.items()):
        for row in (0, 3, PROMPT_TOKENS - 1):
            result = index.search(
                ident(transfer_id, layer, head),
                queries=item.vectors[row : row + 1],
                top_k=1,
            )
            selection = result.selection
            assert result.index_version
            assert result.validated[0] == "vector_space"
            assert selection.token_ids == (row,)
            assert selection.page_ids == (row // layout.page_size,)
            assert (selection.layer, selection.kv_head) == (layer, head)


def test_the_second_shard_indexes_the_upper_global_heads():
    index, _, manifest, _, _ = searchable(rank=1)
    record = index._entries[manifest.key.transfer_id]
    assert sorted({head for _, head in record.indexes}) == [2, 3]


def test_searching_before_the_build_is_refused_with_the_gate_state():
    index = manager()
    _, manifest, _, _ = stored_entry(index)
    with pytest.raises(Exception, match="absent"):
        index.search(
            ident(manifest.key.transfer_id), queries=torch.zeros(1, 8), top_k=1
        )


def test_a_query_with_the_wrong_positional_encoding_is_refused():
    index, _, manifest, _, _ = searchable()
    item = next(iter(index._entries[manifest.key.transfer_id].vectors.values()))
    with pytest.raises(Exception, match="differently-encoded"):
        index.search(
            ident(
                manifest.key.transfer_id,
                item.layer,
                item.kv_head,
                positional_encoding=NO_POSITIONAL_ENCODING,
            ),
            queries=item.vectors[:1],
            top_k=1,
        )


def test_an_unindexed_head_is_refused():
    index, _, manifest, _, _ = searchable()
    with pytest.raises(IndexSearchError, match="no index for layer"):
        index.search(
            ident(manifest.key.transfer_id, 0, 99),
            queries=torch.zeros(1, 8),
            top_k=1,
        )


def test_searching_an_unknown_entry_is_refused():
    with pytest.raises(IndexSearchError, match="no index gate"):
        manager().search(ident("nobody"), queries=torch.zeros(1, 8), top_k=1)


def test_one_index_serves_many_searches():
    index, _, manifest, _, _ = searchable()
    item = next(iter(index._entries[manifest.key.transfer_id].vectors.values()))
    for _ in range(5):
        index.search(
            ident(manifest.key.transfer_id, item.layer, item.kv_head),
            queries=item.vectors[:1],
            top_k=1,
        )
    assert index.gate_for(manifest.key.transfer_id).searchable


# --------------------------------------------------------------------------
# Release
# --------------------------------------------------------------------------


def test_releasing_the_entry_closes_its_index():
    index, store, manifest, _, _ = searchable()
    store.release_entry(manifest.key)
    store._progress_releases()
    store._free_allocation(store.entries[manifest.key])
    assert index.gate_for(manifest.key.transfer_id) is None
    with pytest.raises(IndexSearchError, match="no index gate"):
        index.search(
            ident(manifest.key.transfer_id), queries=torch.zeros(1, 8), top_k=1
        )


def test_an_entry_that_left_stored_is_not_indexed():
    """The gate outlives a state change; only a STORED entry may be read."""
    index = manager()
    store, manifest, _, _ = stored_entry(index)
    store.entries[manifest.key].state = EntryShardState.FAILED
    assert store.progress_prompt_indexes() == {
        "built": 0,
        "failed": 0,
        "deferred": 0,
        "skipped": 0,
    }
    assert index.gate_for(manifest.key.transfer_id).state is IndexState.ABSENT


def test_a_released_entry_is_never_indexed():
    index = manager()
    store, manifest, _, _ = stored_entry(index)
    store.release_entry(manifest.key)
    assert store.progress_prompt_indexes()["built"] == 0
    gate = index.gate_for(manifest.key.transfer_id)
    assert gate is None or not gate.searchable


def test_an_entry_whose_guard_is_releasing_is_skipped_not_read():
    """ResourceGuard.pin refuses once release is requested; we must not read."""
    index = manager()
    store, manifest, _, _ = stored_entry(index)
    entry = store.entries[manifest.key]
    # Hold a pin so the release is requested but cannot complete, which is the
    # window where a reader must refuse rather than race the free.
    entry.allocation_guard.pin("someone-else")
    entry.allocation_guard.request_release()
    assert store.progress_prompt_indexes() == {
        "built": 0,
        "failed": 0,
        "deferred": 0,
        "skipped": 1,
    }
    assert index.gate_for(manifest.key.transfer_id).state is IndexState.ABSENT


def test_closing_an_unknown_entry_is_harmless():
    manager().close("never-seen")


def test_the_snapshot_reports_backend_and_per_entry_state():
    index, _, manifest, _, _ = searchable()
    snapshot = index.snapshot()
    assert snapshot["backend"] == "brute_force"
    assert snapshot["vector_space"] == SPACE
    entry = snapshot["entries"][manifest.key.transfer_id]
    assert entry["state"] == "ready"
    assert entry["indexed_heads"] == 3 * 2


# --------------------------------------------------------------------------
# Budget accounting: a copy is refunded on failure and on close
# --------------------------------------------------------------------------


def budgeted(backend=None, limit=1 << 20):
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

    budget = TransferBudget(staging_bytes=limit, max_inflight=4)
    kwargs = {"budget": budget}
    if backend is not None:
        kwargs["backend"] = backend
    return manager(**kwargs), budget


class BrokenBackend:
    """A backend that declares its costs honestly and then fails to build.

    The footprints are real so that the failure under test is the build
    itself; a backend missing them would fail earlier, for a different
    reason, and the test would stop covering what it names.
    """

    name = "broken"
    device = torch.device("cpu")

    def build_footprint(self, rows, dim, *, metric):
        return rows * dim * 4

    def build_scratch_footprint(self, rows, dim, *, metric):
        return rows * dim * 4

    def search_footprint(self, rows, dim, num_queries, top_k):  # pragma: no cover
        return 0

    def build(self, *a, **k):
        raise RuntimeError("backend down")

    def search(self, *a, **k):  # pragma: no cover
        raise AssertionError


def test_a_successful_build_charges_the_vector_copies():
    index, budget = budgeted()
    store, _, _, _ = stored_entry(index)
    store.progress_prompt_indexes()
    assert budget.snapshot()["used_staging_bytes"] > 0


def test_a_failed_build_refunds_what_extraction_reserved():
    """Otherwise a permanently failing Entry holds budget until restart."""
    index, budget = budgeted(BrokenBackend())
    store, _, _, _ = stored_entry(index)
    assert store.progress_prompt_indexes()["failed"] == 1
    assert budget.snapshot()["used_staging_bytes"] == 0


def test_repeated_failures_do_not_accumulate_budget():
    index, budget = budgeted(BrokenBackend())
    store, _, _, _ = stored_entry(index)
    for _ in range(3):
        store.progress_prompt_indexes()
    assert budget.snapshot()["used_staging_bytes"] == 0


def test_closing_refunds_the_vector_copies():
    """The success path: every served Entry must give its bytes back."""
    index, budget = budgeted()
    store, manifest, _, _ = stored_entry(index)
    store.progress_prompt_indexes()
    assert budget.snapshot()["used_staging_bytes"] > 0
    index.close(manifest.key.transfer_id)
    assert budget.snapshot()["used_staging_bytes"] == 0


def test_releasing_the_entry_refunds_through_the_store():
    index, budget = budgeted()
    store, manifest, _, _ = stored_entry(index)
    store.progress_prompt_indexes()
    store.release_entry(manifest.key)
    store._progress_releases()
    store._free_allocation(store.entries[manifest.key])
    assert budget.snapshot()["used_staging_bytes"] == 0


def test_many_entries_do_not_exhaust_the_budget_when_each_is_released():
    index, budget = budgeted(limit=1 << 20)
    for _ in range(6):
        store, manifest, _, _ = stored_entry(index)
        assert store.progress_prompt_indexes()["built"] == 1
        store.release_entry(manifest.key)
        store._progress_releases()
        store._free_allocation(store.entries[manifest.key])
    assert budget.snapshot()["used_staging_bytes"] == 0


def test_a_build_finishing_after_close_refunds_and_does_not_resurrect():
    """close() can land mid-build; the result is discarded, not installed."""
    index, budget = budgeted()
    pool, layout, manifest, packed, shard = build_entry()
    transfer_id = manifest.key.transfer_id
    index.open(transfer_id)
    index.note_kv_readable(transfer_id)

    real_build = index.backend.build

    def close_then_build(*a, **k):
        index.close(transfer_id)
        return real_build(*a, **k)

    index.backend.build = close_then_build
    assert (
        index.build(transfer_id, packed.tensor, layout=layout, manifest=shard) is False
    )
    assert index.gate_for(transfer_id) is None
    assert budget.snapshot()["used_staging_bytes"] == 0


# --------------------------------------------------------------------------
# Query shape
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [torch.zeros(8), torch.zeros(1, 1, 8), [0.0] * 8])
def test_a_query_that_is_not_two_dimensional_is_refused_clearly(bad):
    index, _, manifest, _, _ = searchable()
    with pytest.raises(IndexSearchError, match="2-D"):
        index.search(ident(manifest.key.transfer_id), queries=bad, top_k=1)


# --------------------------------------------------------------------------
# V serving integration: shard HTTP routes and the launcher
# --------------------------------------------------------------------------


def shard_client(store):
    from aiohttp.test_utils import TestClient, TestServer

    from sglang.srt.disaggregation.pvd.control_server import create_shard_app

    return TestClient(TestServer(create_shard_app(store)))


def test_the_index_routes_report_disabled_when_no_index_is_configured():
    import asyncio

    async def scenario():
        store, _, _, _ = stored_entry()
        async with shard_client(store) as http:
            assert await (await http.post("/internal/v1/indexes/progress")).json() == {
                "enabled": False
            }
            assert await (await http.get("/internal/v1/indexes")).json() == {
                "enabled": False
            }
            refused = await http.post(
                "/internal/v1/indexes/search",
                json={
                    "transfer_id": "x",
                    "layer": 0,
                    "kv_head": 0,
                    "queries": [[0.0]],
                    "vector_space": SPACE,
                    "positional_encoding": ROPE_APPLIED,
                },
            )
            assert refused.status == 400

    asyncio.run(scenario())


def test_progress_and_search_over_http_return_original_tokens():
    import asyncio

    async def scenario():
        index = manager()
        store, manifest, _, layout = stored_entry(index)
        transfer_id = manifest.key.transfer_id
        async with shard_client(store) as http:
            progressed = await (await http.post("/internal/v1/indexes/progress")).json()
            assert progressed == {
                "enabled": True,
                "built": 1,
                "failed": 0,
                "deferred": 0,
                "skipped": 0,
            }
            snapshot = await (await http.get("/internal/v1/indexes")).json()
            assert snapshot["entries"][transfer_id]["state"] == "ready"

            (layer, head), item = sorted(index._entries[transfer_id].vectors.items())[0]
            row = 5
            reply = await http.post(
                "/internal/v1/indexes/search",
                json={
                    "transfer_id": transfer_id,
                    "layer": layer,
                    "kv_head": head,
                    "queries": item.vectors[row : row + 1].tolist(),
                    "top_k": 1,
                    "vector_space": SPACE,
                    "positional_encoding": ROPE_APPLIED,
                },
            )
            assert reply.status == 200
            body = await reply.json()
            assert body["token_ids"] == [row]
            assert body["page_ids"] == [row // layout.page_size]
            assert (body["layer"], body["kv_head"]) == (layer, head)
            assert body["id_mapping_version"]
            assert body["index_version"]
            assert "vector_space" in body["validated"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"queries": []}, "non-empty list"),
        ({"queries": "not-a-list"}, "non-empty list"),
        ({"queries": [[0.0], [0.0, 0.0]]}, "equal-length"),
        ({"queries": [[0.0] * 8] * 65}, "at most 64 queries"),
        ({"top_k": 0}, "positive integer"),
        ({"top_k": True}, "positive integer"),
        ({"top_k": 513}, "must not exceed 512"),
    ],
)
def test_a_malformed_or_unbounded_search_is_refused(payload, expected):
    import asyncio

    async def scenario():
        index = manager()
        store, manifest, _, _ = stored_entry(index)
        store.progress_prompt_indexes()
        body = {
            "transfer_id": manifest.key.transfer_id,
            "layer": 0,
            "kv_head": 0,
            "queries": [[0.0] * 8],
            "top_k": 1,
            "vector_space": SPACE,
            "positional_encoding": ROPE_APPLIED,
        }
        body.update(payload)
        async with shard_client(store) as http:
            reply = await http.post("/internal/v1/indexes/search", json=body)
            assert reply.status == 400
            # The specific refusal matters: a generic downstream error would
            # mean the bound or shape check never ran.
            assert expected in (await reply.text())

    asyncio.run(scenario())


def test_searching_a_not_ready_index_over_http_is_refused():
    import asyncio

    async def scenario():
        index = manager()
        store, manifest, _, _ = stored_entry(index)
        async with shard_client(store) as http:
            reply = await http.post(
                "/internal/v1/indexes/search",
                json={
                    "transfer_id": manifest.key.transfer_id,
                    "layer": 0,
                    "kv_head": 0,
                    "queries": [[0.0] * 8],
                    "top_k": 1,
                    "vector_space": SPACE,
                    "positional_encoding": ROPE_APPLIED,
                },
            )
            assert reply.status == 400

    asyncio.run(scenario())


def test_the_launcher_builds_no_index_unless_a_vector_space_is_given():
    from sglang.srt.disaggregation.pvd.server import _build_prompt_index, build_parser

    args = build_parser().parse_args(
        [
            "--advertise-host",
            "v",
            "--transfer-staging-budget-bytes",
            "1073741824",
            "--transfer-max-inflight",
            "64",
            "--total-pages",
            "8",
            "--page-bytes",
            "32",
        ]
    )
    assert args.prompt_index_vector_space is None
    assert args.prompt_index_metric == "ip"
    assert _build_prompt_index(args) is None


def test_the_launcher_accepts_a_vector_space_and_metric():
    from sglang.srt.disaggregation.pvd.server import _build_prompt_index, build_parser

    args = build_parser().parse_args(
        [
            "--advertise-host",
            "v",
            "--transfer-staging-budget-bytes",
            "1073741824",
            "--transfer-max-inflight",
            "64",
            "--total-pages",
            "8",
            "--page-bytes",
            "32",
            "--prompt-index-vector-space",
            "target/model-8b",
            "--prompt-index-metric",
            "l2",
            "--prompt-index-budget-bytes",
            "4194304",
        ]
    )
    assert args.prompt_index_vector_space == "target/model-8b"
    assert args.prompt_index_metric == "l2"
    built = _build_prompt_index(args)
    assert built is not None
    assert built.vector_space == "target/model-8b"
    assert built.metric == "l2"


def test_an_unknown_metric_is_rejected_at_startup():
    from sglang.srt.disaggregation.pvd.server import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--advertise-host",
                "v",
                "--transfer-staging-budget-bytes",
                "1",
                "--transfer-max-inflight",
                "1",
                "--total-pages",
                "8",
                "--page-bytes",
                "32",
                "--prompt-index-metric",
                "cosine",
            ]
        )


# --------------------------------------------------------------------------
# Complete index memory accounting
#
# The manager retains two copies per indexed head: the one extraction makes
# and the one the backend keeps. Charging only the first understates a V
# worker's real footprint by a factor of two, and the worker finds out by
# running out of memory rather than by being refused.
# --------------------------------------------------------------------------


def retained_bytes(index, transfer_id):
    """What this manager is actually holding, counted from the tensors.

    Counts every tensor a built index keeps, not just its vectors: the exact
    backend also retains squared norms on the L2 path, and an accounting
    check that ignored them would pass while under-charging.
    """
    record = index._entries[transfer_id]
    return sum(
        v.vectors.numel() * v.vectors.element_size() for v in record.vectors.values()
    ) + sum(
        tensor.numel() * tensor.element_size()
        for built in record.indexes.values()
        for tensor in built.retained_tensors()
    )


def test_both_retained_copies_are_charged_not_just_extractions():
    index, budget = budgeted()
    store, manifest, _, _ = stored_entry(index)
    assert store.progress_prompt_indexes()["built"] == 1
    transfer_id = manifest.key.transfer_id
    held = retained_bytes(index, transfer_id)
    charged = budget.snapshot()["used_staging_bytes"]
    # Counted from the tensors themselves, so this stays true if either copy
    # changes dtype or the backend starts keeping something different.
    assert charged == held
    # And the two copies really are distinct memory, not one aliased twice.
    record = index._entries[transfer_id]
    pointers = {v.vectors.data_ptr() for v in record.vectors.values()}
    pointers |= {i.handle.data_ptr() for i in record.indexes.values()}
    assert len(pointers) == len(record.vectors) + len(record.indexes)


def test_capacity_is_refused_before_the_backend_allocates_anything():
    """A refusal must cost nothing: no copy, and no build attempt spent."""

    class CountingBackend(BruteForceIndexBackend):
        def __init__(self):
            super().__init__()
            self.builds = 0

        def build(self, *a, **k):
            self.builds += 1
            return super().build(*a, **k)

    backend = CountingBackend()
    # Room for the extracted vectors but not for what the backend would keep.
    index = manager(backend=backend, budget=None)
    store, manifest, _, _ = stored_entry(index)
    store.progress_prompt_indexes()
    transfer_id = manifest.key.transfer_id
    vectors_only = sum(
        v.vectors.numel() * v.vectors.element_size()
        for v in index._entries[transfer_id].vectors.values()
    )

    backend = CountingBackend()
    index, budget = budgeted(backend, limit=vectors_only + 16)
    store, manifest, _, _ = stored_entry(index)
    assert store.progress_prompt_indexes() == {
        "built": 0,
        "failed": 0,
        "deferred": 1,
        "skipped": 0,
    }
    assert backend.builds == 0, "the backend was called despite no capacity"
    assert budget.snapshot()["used_staging_bytes"] == 0


def test_no_capacity_is_backpressure_not_a_spent_build_attempt():
    """Otherwise a busy minute permanently un-indexes an Entry."""
    index, budget = budgeted(limit=64, backend=None)
    store, manifest, _, _ = stored_entry(index)
    gate = index.gate_for(manifest.key.transfer_id)
    for _ in range(5):
        # Deferred, never failed: that distinction is the whole fix.
        assert store.progress_prompt_indexes() == {
            "built": 0,
            "failed": 0,
            "deferred": 1,
            "skipped": 0,
        }
    assert gate.state is IndexState.ABSENT
    assert gate.attempts == 0
    assert gate.deferrals == 5
    assert not gate.exhausted
    # Still a candidate: this is the whole point of the distinction.
    assert index.wants_build(manifest.key.transfer_id)


def test_a_deferred_build_succeeds_once_there_is_room_again():
    index, budget = budgeted(limit=64)
    store, manifest, _, _ = stored_entry(index)
    assert store.progress_prompt_indexes()["built"] == 0
    budget._staging_bytes = 1 << 20
    assert store.progress_prompt_indexes()["built"] == 1
    gate = index.gate_for(manifest.key.transfer_id)
    assert gate.state is IndexState.READY
    assert gate.attempts == 1 and gate.deferrals == 1


def test_repeated_build_and_release_cycles_do_not_drift():
    """Charge and refund must balance exactly, not approximately."""
    index, budget = budgeted(limit=1 << 20)
    baseline = None
    for _ in range(8):
        store, manifest, _, _ = stored_entry(index)
        assert store.progress_prompt_indexes()["built"] == 1
        charged = budget.snapshot()["used_staging_bytes"]
        baseline = charged if baseline is None else baseline
        assert charged == baseline, "the same shard charged a different amount"
        index.close(manifest.key.transfer_id)
        assert budget.snapshot()["used_staging_bytes"] == 0
        assert budget.snapshot()["reservations"] == 0


def test_a_retry_after_failure_does_not_collide_with_the_first_attempt():
    """Per-attempt owners: a second attempt must not hit 'already reserved'."""

    class FlakyBackend(BruteForceIndexBackend):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def build(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("first attempt fails")
            return super().build(*a, **k)

    index, budget = budgeted(FlakyBackend())
    store, manifest, _, _ = stored_entry(index)
    assert store.progress_prompt_indexes()["failed"] == 1
    assert budget.snapshot()["used_staging_bytes"] == 0
    assert store.progress_prompt_indexes()["built"] == 1
    assert budget.snapshot()["used_staging_bytes"] == retained_bytes(
        index, manifest.key.transfer_id
    )


def test_close_during_a_search_holds_the_charge_until_the_search_returns():
    """A refund while a search still holds the tensors is over-commitment."""
    import threading

    entered = threading.Event()
    may_finish = threading.Event()

    class BlockingBackend(BruteForceIndexBackend):
        def search(self, index, queries, *, top_k):
            entered.set()
            assert may_finish.wait(5)
            return super().search(index, queries, top_k=top_k)

    index, budget = budgeted(BlockingBackend())
    store, manifest, _, _ = stored_entry(index)
    store.progress_prompt_indexes()
    transfer_id = manifest.key.transfer_id
    (layer, head), item = sorted(index._entries[transfer_id].vectors.items())[0]
    charged = budget.snapshot()["used_staging_bytes"]
    assert charged > 0

    done = []

    def run():
        done.append(
            index.search(
                ident(transfer_id, layer, head),
                queries=item.vectors[:1],
                top_k=1,
            )
        )

    worker = threading.Thread(target=run)
    worker.start()
    assert entered.wait(5)
    index.close(transfer_id)
    # Detached at once -- no new search can start ...
    assert index.gate_for(transfer_id) is None
    # ... but the memory the running search is reading is still charged.
    assert budget.snapshot()["used_staging_bytes"] >= charged
    may_finish.set()
    worker.join(5)
    assert not worker.is_alive()
    assert len(done) == 1
    assert budget.snapshot()["used_staging_bytes"] == 0
    assert budget.snapshot()["reservations"] == 0


def test_a_search_charges_bounded_scratch_and_gives_it_back():
    index, budget = budgeted()
    store, manifest, _, _ = stored_entry(index)
    store.progress_prompt_indexes()
    transfer_id = manifest.key.transfer_id
    (layer, head), item = sorted(index._entries[transfer_id].vectors.items())[0]
    retained = budget.snapshot()["used_staging_bytes"]
    index.search(ident(transfer_id, layer, head), queries=item.vectors[:1], top_k=1)
    assert budget.snapshot()["used_staging_bytes"] == retained
    assert budget.snapshot()["reservations"] == 2


def test_a_search_with_no_scratch_headroom_is_refused_not_run():
    index, budget = budgeted()
    store, manifest, _, _ = stored_entry(index)
    store.progress_prompt_indexes()
    transfer_id = manifest.key.transfer_id
    (layer, head), item = sorted(index._entries[transfer_id].vectors.items())[0]
    budget._staging_bytes = budget.snapshot()["used_staging_bytes"]
    with pytest.raises(Exception, match="capacity"):
        index.search(ident(transfer_id, layer, head), queries=item.vectors[:1], top_k=1)
    # The reader count came back down, so the Entry is not wedged.
    assert index._entries[transfer_id].users == 0
    assert budget.snapshot()["reservations"] == 2


def test_the_launcher_refuses_a_vector_space_without_a_budget():
    """Otherwise the serving path is exactly the uncharged one again."""
    from sglang.srt.disaggregation.pvd.server import _build_prompt_index, build_parser

    argv = [
        "--advertise-host",
        "v",
        "--transfer-staging-budget-bytes",
        "1073741824",
        "--transfer-max-inflight",
        "64",
        "--total-pages",
        "8",
        "--page-bytes",
        "32",
        "--prompt-index-vector-space",
        "target/model-8b",
    ]
    args = build_parser().parse_args(argv)
    assert args.prompt_index_budget_bytes is None
    with pytest.raises(ValueError, match="prompt-index-budget-bytes"):
        _build_prompt_index(args)

    args = build_parser().parse_args(argv + ["--prompt-index-budget-bytes", "4194304"])
    built = _build_prompt_index(args)
    assert built.budget is not None
    assert built.budget.snapshot()["staging_bytes"] == 4194304


def test_the_index_budget_is_not_the_transfer_budget():
    """Index copies must not eat headroom a transfer was admitted against."""
    from sglang.srt.disaggregation.pvd.server import _build_prompt_index, build_parser

    args = build_parser().parse_args(
        [
            "--advertise-host",
            "v",
            "--transfer-staging-budget-bytes",
            "1073741824",
            "--transfer-max-inflight",
            "64",
            "--total-pages",
            "8",
            "--page-bytes",
            "32",
            "--prompt-index-vector-space",
            "target/model-8b",
            "--prompt-index-budget-bytes",
            "4194304",
        ]
    )
    built = _build_prompt_index(args)
    assert built.budget.snapshot()["staging_bytes"] == 4194304
    assert (
        built.budget.snapshot()["staging_bytes"] != args.transfer_staging_budget_bytes
    )


# --------------------------------------------------------------------------
# The caller's query identity
#
# Every one of these queries is the right shape, the right dtype and carries a
# legal positional-encoding label. Shape is not identity: what makes them
# wrong is whose vectors they are, and only the caller can say that.
# --------------------------------------------------------------------------


def test_a_query_from_another_vector_space_is_refused_despite_a_valid_shape():
    index, _, manifest, _, _ = searchable()
    transfer_id = manifest.key.transfer_id
    (layer, head), item = sorted(index._entries[transfer_id].vectors.items())[0]
    # Byte-for-byte a vector this index would happily return.
    queries = item.vectors[:1].clone()
    with pytest.raises(Exception, match="draft/model-1b"):
        index.search(
            ident(transfer_id, layer, head, vector_space="draft/model-1b"),
            queries=queries,
            top_k=1,
        )


def test_the_manager_does_not_answer_for_the_caller():
    """The refusal must come from the caller's claim, not the manager's."""
    index, _, manifest, _, _ = searchable()
    transfer_id = manifest.key.transfer_id
    (layer, head), item = sorted(index._entries[transfer_id].vectors.items())[0]
    # Move the manager's configured space away from what was actually built.
    # Both assertions below invert if authorize_search is handed
    # self.vector_space again.
    index.vector_space = "some/other-space"
    # A caller holding the real space is still served: the built descriptor
    # is the authority, and the manager's current field is not consulted.
    index.search(ident(transfer_id, layer, head), queries=item.vectors[:1], top_k=1)
    # A caller claiming the manager's new field is refused, because the claim
    # is checked against what was built, not against the field.
    with pytest.raises(Exception, match="some/other-space"):
        index.search(
            ident(transfer_id, layer, head, vector_space="some/other-space"),
            queries=item.vectors[:1],
            top_k=1,
        )


def test_a_stale_index_or_mapping_pin_is_refused():
    index, _, manifest, _, _ = searchable()
    transfer_id = manifest.key.transfer_id
    (layer, head), item = sorted(index._entries[transfer_id].vectors.items())[0]
    for field, message in (
        ("expected_index_version", "index has been rebuilt"),
        ("expected_id_mapping_version", "id mapping has been rebuilt"),
    ):
        with pytest.raises(Exception, match=message):
            index.search(
                ident(transfer_id, layer, head, **{field: "from-a-previous-life"}),
                queries=item.vectors[:1],
                top_k=1,
            )


def test_a_pinned_version_that_matches_is_reported_as_checked():
    index, _, manifest, _, _ = searchable()
    transfer_id = manifest.key.transfer_id
    (layer, head), item = sorted(index._entries[transfer_id].vectors.items())[0]
    first = index.search(
        ident(transfer_id, layer, head), queries=item.vectors[:1], top_k=1
    )
    # An unpinned search says so: the versions are absent from `validated`.
    assert "index_version" not in first.validated
    assert "id_mapping_version" not in first.validated
    second = index.search(
        ident(
            transfer_id,
            layer,
            head,
            expected_index_version=first.index_version,
            expected_id_mapping_version=first.id_mapping_version,
        ),
        queries=item.vectors[:1],
        top_k=1,
    )
    assert "index_version" in second.validated
    assert "id_mapping_version" in second.validated


def test_a_query_for_another_entry_or_head_is_refused():
    index, _, manifest, _, _ = searchable()
    transfer_id = manifest.key.transfer_id
    (layer, head), item = sorted(index._entries[transfer_id].vectors.items())[0]
    with pytest.raises(IndexSearchError, match="no index gate"):
        index.search(
            ident("some-other-entry", layer, head), queries=item.vectors[:1], top_k=1
        )
    with pytest.raises(IndexSearchError, match="no index for layer"):
        index.search(
            ident(transfer_id, layer, head + 7), queries=item.vectors[:1], top_k=1
        )
    with pytest.raises(IndexSearchError, match="no index for layer"):
        index.search(
            ident(transfer_id, layer + 7, head), queries=item.vectors[:1], top_k=1
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("vector_space", ""),
        ("vector_space", None),
        ("positional_encoding", "rotary-ish"),
        ("entry_transfer_id", "   "),
    ],
)
def test_an_incomplete_or_illegal_identity_is_refused(field, value):
    fields = {
        "vector_space": SPACE,
        "positional_encoding": ROPE_APPLIED,
        "entry_transfer_id": "t",
        "layer": 0,
        "kv_head": 0,
    }
    fields[field] = value
    with pytest.raises(IndexSearchError):
        SearchRequestIdentity(**fields)


def test_search_refuses_a_bare_transfer_id():
    """The old shape must not keep working by accident."""
    index, _, manifest, _, _ = searchable()
    with pytest.raises(IndexSearchError, match="SearchRequestIdentity"):
        index.search(manifest.key.transfer_id, queries=torch.zeros(1, 8), top_k=1)


def test_http_refuses_a_search_that_states_no_vector_space():
    import asyncio

    async def scenario():
        index = manager()
        store, manifest, _, _ = stored_entry(index)
        store.progress_prompt_indexes()
        transfer_id = manifest.key.transfer_id
        (layer, head), item = sorted(index._entries[transfer_id].vectors.items())[0]
        body = {
            "transfer_id": transfer_id,
            "layer": layer,
            "kv_head": head,
            "queries": item.vectors[:1].tolist(),
            "top_k": 1,
            "positional_encoding": ROPE_APPLIED,
        }
        async with shard_client(store) as http:
            reply = await http.post("/internal/v1/indexes/search", json=body)
            assert reply.status == 400
            assert "vector_space" in (await reply.text())
            # And a stated-but-wrong space is refused on the same request.
            reply = await http.post(
                "/internal/v1/indexes/search",
                json={**body, "vector_space": "draft/model-1b"},
            )
            assert reply.status == 400
            assert "draft/model-1b" in (await reply.text())
            # The same request with the caller's real space is answered.
            reply = await http.post(
                "/internal/v1/indexes/search", json={**body, "vector_space": SPACE}
            )
            assert reply.status == 200

    asyncio.run(scenario())


def test_http_refuses_a_stale_version_pin():
    import asyncio

    async def scenario():
        index = manager()
        store, manifest, _, _ = stored_entry(index)
        store.progress_prompt_indexes()
        transfer_id = manifest.key.transfer_id
        (layer, head), item = sorted(index._entries[transfer_id].vectors.items())[0]
        body = {
            "transfer_id": transfer_id,
            "layer": layer,
            "kv_head": head,
            "queries": item.vectors[:1].tolist(),
            "top_k": 1,
            "vector_space": SPACE,
            "positional_encoding": ROPE_APPLIED,
        }
        async with shard_client(store) as http:
            first = await (
                await http.post("/internal/v1/indexes/search", json=body)
            ).json()
            reply = await http.post(
                "/internal/v1/indexes/search",
                json={**body, "expected_index_version": "idx:stale:1"},
            )
            assert reply.status == 400
            assert "rebuilt" in (await reply.text())
            reply = await http.post(
                "/internal/v1/indexes/search",
                json={**body, "expected_index_version": first["index_version"]},
            )
            assert reply.status == 200
            assert "index_version" in (await reply.json())["validated"]

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# Device consistency
#
# On a GPU V worker the KV pool is on cuda:N. The exact backend is CPU by
# declared policy, so the copy extraction already makes is where the device
# transition happens -- and the pool itself never moves.
# --------------------------------------------------------------------------


def test_the_manager_places_copies_on_the_backends_device():
    index = manager()
    store, manifest, _, _ = stored_entry(index)
    store.progress_prompt_indexes()
    record = index._entries[manifest.key.transfer_id]
    assert index.backend_device == index.backend.device
    for item in record.vectors.values():
        assert item.vectors.device == index.backend.device
    for built in record.indexes.values():
        assert built.handle.device == index.backend.device


def test_indexing_does_not_move_the_authoritative_kv_pool():
    """The pool is registered memory; relocating it would invalidate the MR."""
    index = manager()
    store, manifest, _, _ = stored_entry(index)
    pool, device, ptr = store.pool, store.pool.device, store.pool.data_ptr()
    store.progress_prompt_indexes()
    assert store.pool is pool
    assert store.pool.device == device
    assert store.pool.data_ptr() == ptr
    # And nothing the index holds points into the pool.
    span = (ptr, ptr + store.pool.numel())
    record = index._entries[manifest.key.transfer_id]
    for item in record.vectors.values():
        assert not span[0] <= item.vectors.data_ptr() < span[1]
    for built in record.indexes.values():
        assert not span[0] <= built.handle.data_ptr() < span[1]


def test_a_cpu_query_searches_a_cpu_index_end_to_end():
    """What the HTTP route actually produces, against what is built."""
    index, _, manifest, _, _ = searchable()
    transfer_id = manifest.key.transfer_id
    (layer, head), item = sorted(index._entries[transfer_id].vectors.items())[0]
    # Exactly how control_server builds it: JSON numbers, host tensor.
    queries = torch.tensor(item.vectors[3:4].tolist(), dtype=torch.float32)
    assert queries.device == torch.device("cpu")
    result = index.search(ident(transfer_id, layer, head), queries=queries, top_k=1)
    assert result.selection.token_ids == (3,)


def test_the_snapshot_reports_the_device_and_the_budget():
    """An operator must be able to see the policy that is in force."""
    index, budget = budgeted()
    store, _, _, _ = stored_entry(index)
    store.progress_prompt_indexes()
    snapshot = index.snapshot()
    assert snapshot["device"] == str(index.backend.device)
    assert snapshot["budget"]["used_staging_bytes"] > 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device in this environment"
)
def test_a_cuda_pool_still_yields_a_host_index():
    """UNVERIFIED without GPU hardware: skipped on every CPU-only run.

    This is the production shape: pool on cuda:0, exact index on the host,
    queries on the host. It is the one case a CPU-only suite cannot cover.
    """
    index = manager()
    store, manifest, _, _ = stored_entry(index)
    store.pool = store.pool.to("cuda")
    store.progress_prompt_indexes()
    record = index._entries[manifest.key.transfer_id]
    assert store.pool.device.type == "cuda"
    for item in record.vectors.values():
        assert item.vectors.device == torch.device("cpu")
    for built in record.indexes.values():
        assert built.handle.device == torch.device("cpu")


def test_the_manager_hands_extraction_the_backends_device():
    """Otherwise a host backend ends up holding GPU copies of every prompt.

    A CPU-only run cannot tell ``device=cpu`` from ``device=None``, so the
    backend here declares 'meta' -- a real non-CPU device for placement
    purposes -- and records what it was actually given.
    """
    from sglang.srt.disaggregation.pvd.index_search import BuiltIndex

    class RecordingBackend(BruteForceIndexBackend):
        def __init__(self):
            super().__init__()
            self.seen = []

        @property
        def device(self):
            return torch.device("meta")

        def build(self, vectors, *, vector_space, metric):
            self.seen.append(vectors.device)
            return BuiltIndex(
                vector_space=vector_space,
                metric=metric,
                dim=int(vectors.shape[1]),
                count=int(vectors.shape[0]),
                handle=vectors,
            )

    backend = RecordingBackend()
    index = manager(backend=backend)
    store, _, _, _ = stored_entry(index)
    assert store.progress_prompt_indexes()["built"] == 1
    assert backend.seen, "the backend was never reached"
    assert set(backend.seen) == {torch.device("meta")}


def test_a_late_build_cannot_refund_a_newer_attempts_reservation():
    """Per-attempt owners. Sharing one would make a stale build free live bytes."""
    import threading

    blocked = threading.Event()
    entered = threading.Event()

    class SlowFirstBuild(BruteForceIndexBackend):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def build(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                entered.set()
                assert blocked.wait(5)
            return super().build(*a, **k)

    index, budget = budgeted(SlowFirstBuild())
    pool, layout, manifest, packed, shard = build_entry()
    transfer_id = manifest.key.transfer_id
    index.open(transfer_id)
    index.note_kv_readable(transfer_id)

    first = threading.Thread(
        target=lambda: index.build(
            transfer_id, packed.tensor, layout=layout, manifest=shard
        )
    )
    first.start()
    assert entered.wait(5)

    # The Entry is closed and re-opened while that build is still in flight,
    # and a second build succeeds under a fresh identity.
    index.close(transfer_id)
    index.open(transfer_id)
    index.note_kv_readable(transfer_id)
    assert index.build(transfer_id, packed.tensor, layout=layout, manifest=shard)
    live = retained_bytes(index, transfer_id)
    # Both attempts' copies exist right now and both are charged; the stale
    # build is also still inside the backend, so its build scratch is live
    # too. Sharing one owner would collapse all of that into a single charge.
    charged = budget.snapshot()["used_staging_bytes"]
    assert charged > 2 * live

    # Now let the stale build finish. It must refund its own charge and only
    # its own: the newer attempt's copies are still there to be searched.
    blocked.set()
    first.join(5)
    assert not first.is_alive()
    assert budget.snapshot()["used_staging_bytes"] == live
    assert retained_bytes(index, transfer_id) == live
    assert index.gate_for(transfer_id).state is IndexState.READY


# --------------------------------------------------------------------------
# Index versions across incarnations
#
# A version derived from the gate's attempt counter repeats: close and
# reopen resets attempts to 1, so the second incarnation's first build was
# handed the identifier a caller had pinned against the first one. The
# optional id-mapping pin happened to change too -- but an optional pin
# cannot be what makes a mandatory identity correct.
# --------------------------------------------------------------------------


def rebuild_in_place(index, transfer_id, packed, layout, shard):
    """Close and reopen the same Entry identity, then rebuild it."""
    index.close(transfer_id)
    index.open(transfer_id)
    index.note_kv_readable(transfer_id)
    assert index.build(transfer_id, packed.tensor, layout=layout, manifest=shard)
    return index.gate_for(transfer_id).descriptor.index_version


def built_manager(**kwargs):
    _, layout, manifest, packed, shard = build_entry()
    index = manager(**kwargs)
    transfer_id = manifest.key.transfer_id
    index.open(transfer_id)
    index.note_kv_readable(transfer_id)
    assert index.build(transfer_id, packed.tensor, layout=layout, manifest=shard)
    return index, transfer_id, packed, layout, shard


def test_a_version_pinned_to_a_previous_incarnation_is_refused():
    index, transfer_id, packed, layout, shard = built_manager()
    (layer, head), item = sorted(index._entries[transfer_id].vectors.items())[0]
    queries = item.vectors[:1].clone()
    old = index.gate_for(transfer_id).descriptor.index_version

    new = rebuild_in_place(index, transfer_id, packed, layout, shard)
    # The counter really does restart, which is what made the old scheme
    # collide; the version must not restart with it.
    assert index.gate_for(transfer_id).attempts == 1
    assert new != old

    # Pinning ONLY the index version, with no mapping pin to compensate.
    with pytest.raises(Exception, match="index has been rebuilt"):
        index.search(
            ident(transfer_id, layer, head, expected_index_version=old),
            queries=queries,
            top_k=1,
        )
    result = index.search(
        ident(transfer_id, layer, head, expected_index_version=new),
        queries=queries,
        top_k=1,
    )
    assert "index_version" in result.validated
    assert result.index_version == new


def test_every_rebuild_gets_its_own_version():
    index, transfer_id, packed, layout, shard = built_manager()
    seen = {index.gate_for(transfer_id).descriptor.index_version}
    for _ in range(6):
        seen.add(rebuild_in_place(index, transfer_id, packed, layout, shard))
    assert len(seen) == 7, f"versions repeated: {sorted(seen)}"


def test_retrying_after_a_failure_also_changes_the_version():
    """Attempts within one incarnation must not reuse an identifier either."""

    class FailsOnce(BruteForceIndexBackend):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def build(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("first attempt fails")
            return super().build(*a, **k)

    _, layout, manifest, packed, shard = build_entry()
    index = manager(backend=FailsOnce())
    transfer_id = manifest.key.transfer_id
    index.open(transfer_id)
    index.note_kv_readable(transfer_id)
    assert (
        index.build(transfer_id, packed.tensor, layout=layout, manifest=shard) is False
    )
    assert index.build(transfer_id, packed.tensor, layout=layout, manifest=shard)
    first = index.gate_for(transfer_id).descriptor.index_version
    second = rebuild_in_place(index, transfer_id, packed, layout, shard)
    assert first != second


def test_a_late_build_cannot_stamp_the_new_incarnations_version():
    """An old build finishing after close/reopen must be discarded whole."""
    import threading

    entered = threading.Event()
    blocked = threading.Event()

    class SlowFirstBuild(BruteForceIndexBackend):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def build(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                entered.set()
                assert blocked.wait(5)
            return super().build(*a, **k)

    index, budget = budgeted(SlowFirstBuild())
    _, layout, manifest, packed, shard = build_entry()
    transfer_id = manifest.key.transfer_id
    index.open(transfer_id)
    index.note_kv_readable(transfer_id)

    stale = threading.Thread(
        target=lambda: index.build(
            transfer_id, packed.tensor, layout=layout, manifest=shard
        )
    )
    stale.start()
    assert entered.wait(5)
    index.close(transfer_id)
    index.open(transfer_id)
    index.note_kv_readable(transfer_id)
    assert index.build(transfer_id, packed.tensor, layout=layout, manifest=shard)
    current = index.gate_for(transfer_id).descriptor.index_version

    blocked.set()
    stale.join(5)
    assert not stale.is_alive()
    # The live incarnation still carries its own version, and the stale
    # build's charge is gone while the live one's remains.
    assert index.gate_for(transfer_id).descriptor.index_version == current
    assert budget.snapshot()["used_staging_bytes"] == retained_bytes(index, transfer_id)

    (layer, head), item = sorted(index._entries[transfer_id].vectors.items())[0]
    result = index.search(
        ident(transfer_id, layer, head, expected_index_version=current),
        queries=item.vectors[:1],
        top_k=1,
    )
    assert result.index_version == current


def test_build_scratch_is_charged_during_the_build_and_given_back_after():
    """Peak build bytes and retained build bytes have different lifetimes."""
    import threading

    entered = threading.Event()
    blocked = threading.Event()
    observed = {}

    class Observe(BruteForceIndexBackend):
        def build(self, *a, **k):
            built = super().build(*a, **k)
            if not entered.is_set():
                entered.set()
                assert blocked.wait(5)
            return built

    index, budget = budgeted(Observe())
    _, layout, manifest, packed, shard = build_entry()
    transfer_id = manifest.key.transfer_id
    index.open(transfer_id)
    index.note_kv_readable(transfer_id)
    worker = threading.Thread(
        target=lambda: index.build(
            transfer_id, packed.tensor, layout=layout, manifest=shard
        )
    )
    worker.start()
    assert entered.wait(5)
    observed["during"] = budget.snapshot()["used_staging_bytes"]
    blocked.set()
    worker.join(5)
    assert not worker.is_alive()
    after = budget.snapshot()["used_staging_bytes"]
    # Scratch was charged while the build was running and is gone once the
    # index is installed, leaving exactly what the index holds.
    assert observed["during"] > after
    assert after == retained_bytes(index, transfer_id)


def test_two_searches_at_once_are_both_charged_then_both_refunded():
    import threading

    ready = threading.Barrier(3, timeout=10)
    release = threading.Event()

    class BlockingBackend(BruteForceIndexBackend):
        def search(self, index, queries, *, top_k):
            ready.wait()
            assert release.wait(5)
            return super().search(index, queries, top_k=top_k)

    index, budget = budgeted(BlockingBackend())
    store, manifest, _, _ = stored_entry(index)
    store.progress_prompt_indexes()
    transfer_id = manifest.key.transfer_id
    (layer, head), item = sorted(index._entries[transfer_id].vectors.items())[0]
    retained = budget.snapshot()["used_staging_bytes"]

    def run():
        index.search(ident(transfer_id, layer, head), queries=item.vectors[:1], top_k=1)

    workers = [threading.Thread(target=run) for _ in range(2)]
    for worker in workers:
        worker.start()
    ready.wait()
    # Both searches hold distinct scratch reservations at the same time.
    assert budget.snapshot()["reservations"] == 4
    assert budget.snapshot()["used_staging_bytes"] > retained
    release.set()
    for worker in workers:
        worker.join(5)
        assert not worker.is_alive()
    assert budget.snapshot()["used_staging_bytes"] == retained
    assert budget.snapshot()["reservations"] == 2
