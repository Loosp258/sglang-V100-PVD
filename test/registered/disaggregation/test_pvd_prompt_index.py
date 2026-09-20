"""The V store driving IndexGate: build, search, release, and independence.

Real VectorKVStore, real packing, real extraction, exact CPU backend. No cuVS,
no GPU. Queries are rows taken from the extracted vectors, so these tests show
that the plumbing preserves identity -- not that retrieval is any good.
"""

import pytest
import torch
from sglang.srt.disaggregation.pvd.index_lifecycle import IndexState
from sglang.srt.disaggregation.pvd.index_search import IndexSearchError
from sglang.srt.disaggregation.pvd.prompt_index import PromptIndexManager
from sglang.srt.disaggregation.pvd.prompt_vectors import (
    NO_POSITIONAL_ENCODING,
    ROPE_APPLIED,
)
from sglang.srt.disaggregation.pvd.protocol import KVEntryKey, KVEntryManifest
from sglang.srt.disaggregation.pvd.request_state import EntryShardState
from sglang.srt.disaggregation.pvd.vector_store import VectorKVStore
from test_pvd_prompt_vectors import FakePool, pack_shard, storage_layout
from test_pvd_vector_lifecycle import DelayedTransferEngine

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


# --------------------------------------------------------------------------
# Off by default
# --------------------------------------------------------------------------


def test_a_store_without_an_index_is_unchanged():
    store, manifest, _, _ = stored_entry()
    assert store.prompt_index is None
    assert store.entries[manifest.key].state is EntryShardState.STORED
    assert store.progress_prompt_indexes() == {"built": 0, "failed": 0, "skipped": 0}


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
    assert store.progress_prompt_indexes() == {"built": 1, "failed": 0, "skipped": 0}
    gate = index.gate_for(manifest.key.transfer_id)
    assert gate.state is IndexState.READY
    assert gate.searchable
    assert gate.descriptor.vector_space == SPACE


def test_a_built_index_is_not_rebuilt_on_later_passes():
    index = manager()
    store, _, _, _ = stored_entry(index)
    store.progress_prompt_indexes()
    assert store.progress_prompt_indexes() == {"built": 0, "failed": 0, "skipped": 0}


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
    assert store.progress_prompt_indexes() == {"built": 0, "failed": 1, "skipped": 0}
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
    assert store.progress_prompt_indexes() == {"built": 0, "failed": 0, "skipped": 0}


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
            selection = index.search(
                transfer_id,
                layer=layer,
                kv_head=head,
                queries=item.vectors[row : row + 1],
                top_k=1,
                positional_encoding=ROPE_APPLIED,
            )
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
            manifest.key.transfer_id,
            layer=0,
            kv_head=0,
            queries=torch.zeros(1, 8),
            top_k=1,
            positional_encoding=ROPE_APPLIED,
        )


def test_a_query_with_the_wrong_positional_encoding_is_refused():
    index, _, manifest, _, _ = searchable()
    item = next(iter(index._entries[manifest.key.transfer_id].vectors.values()))
    with pytest.raises(Exception, match="differently-encoded"):
        index.search(
            manifest.key.transfer_id,
            layer=item.layer,
            kv_head=item.kv_head,
            queries=item.vectors[:1],
            top_k=1,
            positional_encoding=NO_POSITIONAL_ENCODING,
        )


def test_an_unindexed_head_is_refused():
    index, _, manifest, _, _ = searchable()
    with pytest.raises(IndexSearchError, match="no index for layer"):
        index.search(
            manifest.key.transfer_id,
            layer=0,
            kv_head=99,
            queries=torch.zeros(1, 8),
            top_k=1,
            positional_encoding=ROPE_APPLIED,
        )


def test_searching_an_unknown_entry_is_refused():
    with pytest.raises(IndexSearchError, match="no index gate"):
        manager().search(
            "nobody",
            layer=0,
            kv_head=0,
            queries=torch.zeros(1, 8),
            top_k=1,
            positional_encoding=ROPE_APPLIED,
        )


def test_one_index_serves_many_searches():
    index, _, manifest, _, _ = searchable()
    item = next(iter(index._entries[manifest.key.transfer_id].vectors.values()))
    for _ in range(5):
        index.search(
            manifest.key.transfer_id,
            layer=item.layer,
            kv_head=item.kv_head,
            queries=item.vectors[:1],
            top_k=1,
            positional_encoding=ROPE_APPLIED,
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
            manifest.key.transfer_id,
            layer=0,
            kv_head=0,
            queries=torch.zeros(1, 8),
            top_k=1,
            positional_encoding=ROPE_APPLIED,
        )


def test_an_entry_that_left_stored_is_not_indexed():
    """The gate outlives a state change; only a STORED entry may be read."""
    index = manager()
    store, manifest, _, _ = stored_entry(index)
    store.entries[manifest.key].state = EntryShardState.FAILED
    assert store.progress_prompt_indexes() == {"built": 0, "failed": 0, "skipped": 0}
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
    assert store.progress_prompt_indexes() == {"built": 0, "failed": 0, "skipped": 1}
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
