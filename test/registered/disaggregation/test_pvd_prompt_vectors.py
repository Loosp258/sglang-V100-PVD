"""Prompt K extraction from the real stored EntryShard representation.

Every test starts from `kv_packer`'s actual packed buffer and the storage
layout/manifest metadata, not from a pre-extracted tensor. Queries here are
synthetic -- rows taken from the extracted vectors themselves -- so these
tests prove mapping and identity correctness, NOT real-model retrieval
quality, recall, or anything about CAGRA or V100S.
"""

import copy

import pytest
import torch
from sglang.srt.disaggregation.pvd.index_search import (
    BruteForceIndexBackend,
    merge_selections,
    select,
)
from sglang.srt.disaggregation.pvd.kv_packer import (
    PVD_TENSOR_LAYOUT,
    describe_kv_layout,
    pack_full_prompt_kv_head_shard,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import (
    NO_POSITIONAL_ENCODING,
    ROPE_APPLIED,
    PromptVectorError,
    QueryHeadMapping,
    extract_prompt_k,
)
from sglang.srt.disaggregation.pvd.protocol import KVLayoutSignature, KVShardManifest

ENTRY = "entry-1"
MAPPING = "map-v1"
PAGE_SIZE = 4
LAYERS = 3
TOTAL_KV_HEADS = 4
HEAD_DIM = 8
POOL_TOKENS = 64


class FakePool:
    """A rank-local MHA/GQA pool: k_buffer then v_buffer, [tokens, heads, dim]."""

    def __init__(
        self, layers=LAYERS, heads=TOTAL_KV_HEADS, dim=HEAD_DIM, dtype=torch.float16
    ):
        generator = torch.Generator().manual_seed(20260920)
        shape = (POOL_TOKENS, heads, dim)
        self.k_buffer = [
            torch.randn(shape, generator=generator).to(dtype) for _ in range(layers)
        ]
        self.v_buffer = [
            torch.randn(shape, generator=generator).to(dtype) for _ in range(layers)
        ]
        self.start_layer = 0
        self.end_layer = layers


def storage_layout(pool, *, total_kv_heads=TOTAL_KV_HEADS, page_size=PAGE_SIZE):
    """Build the two-shard V layout the way PVDKVManager.storage_layout does."""
    compute = describe_kv_layout(pool)
    source_heads = compute["component_token_shapes"][0][0]
    heads = total_kv_heads // 2
    extra = copy.deepcopy(compute)
    extra["component_token_shapes"] = [
        [heads, shape[1]] for shape in compute["component_token_shapes"]
    ]
    extra["component_bytes_per_token"] = [
        value // source_heads * heads for value in compute["component_bytes_per_token"]
    ]
    return KVLayoutSignature(
        model_id="test-model",
        model_revision="rev",
        kv_dtype=compute["component_dtypes"][0],
        page_size=page_size,
        num_layers=len(pool.k_buffer),
        total_kv_heads=total_kv_heads,
        kv_heads_per_rank=heads,
        head_dim=compute["component_token_shapes"][0][1],
        tp_size=2,
        pp_size=1,
        tensor_layout=PVD_TENSOR_LAYOUT,
        extra=extra,
    )


def pack_shard(pool, layout, *, rank, prompt_tokens, pages=None):
    heads = layout.kv_heads_per_rank
    page_count = -(-prompt_tokens // layout.page_size)
    pages = list(range(page_count)) if pages is None else list(pages)
    packed = pack_full_prompt_kv_head_shard(
        pool,
        pages,
        page_size=layout.page_size,
        head_start=rank * heads,
        head_count=heads,
    )
    manifest = KVShardManifest(
        rank=rank,
        rail=f"mlx5_{rank}",
        expected_bytes=packed.expected_bytes,
        page_count=page_count,
        last_page_valid_tokens=prompt_tokens % layout.page_size or layout.page_size,
        layer_start=pool.start_layer,
        layer_end=pool.end_layer,
    )
    return packed, manifest, pages


def extract(pool=None, *, rank=0, prompt_tokens=10, encoding=ROPE_APPLIED, **kwargs):
    pool = pool or FakePool()
    layout = kwargs.pop("layout", None) or storage_layout(pool)
    packed, manifest, pages = pack_shard(
        pool, layout, rank=rank, prompt_tokens=prompt_tokens
    )
    manifest = kwargs.pop("manifest", None) or manifest
    return (
        extract_prompt_k(
            packed.tensor,
            layout=layout,
            manifest=manifest,
            entry_transfer_id=ENTRY,
            id_mapping_version=MAPPING,
            positional_encoding=encoding,
            **kwargs,
        ),
        pool,
        pages,
    )


def source_k(pool, *, layer, global_head, pages, page_size, valid_tokens):
    rows = [p * page_size + i for p in pages for i in range(page_size)][:valid_tokens]
    return pool.k_buffer[layer][rows, global_head, :].to(torch.float32)


# --------------------------------------------------------------------------
# Shape, identity and coverage
# --------------------------------------------------------------------------


def test_every_owned_layer_and_kv_head_is_extracted():
    vectors, _, _ = extract()
    assert len(vectors) == LAYERS * (TOTAL_KV_HEADS // 2)
    assert {(v.layer, v.kv_head) for v in vectors} == {
        (layer, head) for layer in range(LAYERS) for head in (0, 1)
    }


def test_the_second_shard_owns_the_upper_global_kv_heads():
    vectors, _, _ = extract(rank=1)
    assert sorted({v.kv_head for v in vectors}) == [2, 3]


def test_each_vector_set_carries_its_entry_and_encoding():
    vectors, _, _ = extract()
    assert all(v.entry_transfer_id == ENTRY for v in vectors)
    assert all(v.positional_encoding == ROPE_APPLIED for v in vectors)
    assert all(v.source_dtype == "torch.float16" for v in vectors)


def test_vectors_are_token_by_head_dim():
    vectors, _, _ = extract(prompt_tokens=10)
    assert all(v.vectors.shape == (10, HEAD_DIM) for v in vectors)
    assert all(v.head_dim == HEAD_DIM and v.token_count == 10 for v in vectors)


# --------------------------------------------------------------------------
# K only, padding excluded, values correct
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [0, 1])
def test_extracted_values_are_the_stored_k_for_that_layer_and_head(rank):
    vectors, pool, pages = extract(rank=rank, prompt_tokens=10)
    for v in vectors:
        expected = source_k(
            pool,
            layer=v.layer,
            global_head=v.kv_head,
            pages=pages,
            page_size=PAGE_SIZE,
            valid_tokens=10,
        )
        assert torch.equal(v.vectors, expected)


def test_v_components_are_never_extracted():
    vectors, pool, pages = extract(prompt_tokens=8)
    for v in vectors:
        v_side = pool.v_buffer[v.layer][: v.token_count, v.kv_head, :].to(torch.float32)
        assert not torch.equal(v.vectors, v_side)


def test_padding_in_the_final_page_is_excluded():
    """10 prompt tokens over 4-token pages: 12 stored rows, 10 real ones."""
    vectors, pool, _ = extract(prompt_tokens=10)
    assert all(v.token_count == 10 for v in vectors)
    assert all(len(v.mapping) == 10 for v in vectors)
    padded_row = pool.k_buffer[0][11, 0, :].to(torch.float32)
    assert not any(torch.equal(v.vectors[-1], padded_row) for v in vectors)


@pytest.mark.parametrize(
    "prompt_tokens,expected", [(4, 4), (5, 5), (8, 8), (9, 9), (12, 12)]
)
def test_the_token_count_follows_the_prompt_not_the_pages(prompt_tokens, expected):
    vectors, _, _ = extract(prompt_tokens=prompt_tokens)
    assert vectors[0].token_count == expected


# --------------------------------------------------------------------------
# Mapping
# --------------------------------------------------------------------------


def test_rows_map_to_original_prompt_positions_and_pages():
    vectors, _, _ = extract(prompt_tokens=10)
    mapping = vectors[0].mapping
    assert mapping.version == MAPPING
    assert [mapping.token_of(r) for r in range(10)] == list(range(10))
    assert [mapping.page_of(r) for r in range(10)] == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]


def test_a_row_beyond_the_valid_tokens_is_refused():
    vectors, _, _ = extract(prompt_tokens=10)
    with pytest.raises(Exception):
        vectors[0].mapping.token_of(10)


# --------------------------------------------------------------------------
# Source preservation and ownership
# --------------------------------------------------------------------------


def test_extraction_does_not_mutate_the_stored_pool():
    pool = FakePool()
    before = [layer.clone() for layer in pool.k_buffer + pool.v_buffer]
    extract(pool)
    assert all(torch.equal(a, b) for a, b in zip(before, pool.k_buffer + pool.v_buffer))


def test_extracted_vectors_own_their_bytes():
    """Releasing or overwriting stored KV must not change a built index."""
    pool = FakePool()
    vectors, _, _ = extract(pool, prompt_tokens=8)
    snapshot = vectors[0].vectors.clone()
    for layer in pool.k_buffer:
        layer.zero_()
    assert torch.equal(vectors[0].vectors, snapshot)


def test_vectors_own_their_bytes_even_without_a_dtype_conversion():
    """dtype == stored dtype is the case where a view would alias the buffer."""
    pool = FakePool()
    layout = storage_layout(pool)
    packed, manifest, _ = pack_shard(pool, layout, rank=0, prompt_tokens=8)
    vectors = extract_prompt_k(
        packed.tensor,
        layout=layout,
        manifest=manifest,
        entry_transfer_id=ENTRY,
        id_mapping_version=MAPPING,
        positional_encoding=ROPE_APPLIED,
        dtype=torch.float16,
    )
    snapshot = vectors[0].vectors.clone()
    packed.tensor.zero_()
    assert torch.equal(vectors[0].vectors, snapshot)
    assert vectors[0].vectors.dtype == torch.float16


def test_writing_to_extracted_vectors_does_not_reach_the_pool():
    pool = FakePool()
    vectors, _, _ = extract(pool, prompt_tokens=8)
    original = pool.k_buffer[0][0, 0, :].clone()
    vectors[0].vectors[0].fill_(42.0)
    assert torch.equal(pool.k_buffer[0][0, 0, :], original)


def test_a_budget_is_charged_when_one_is_supplied():
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

    budget = TransferBudget(staging_bytes=1 << 20, max_inflight=4)
    extract(prompt_tokens=8, budget=budget, budget_owner="index:entry-1")
    used = budget.snapshot()["used_staging_bytes"]
    assert used == LAYERS * 2 * 8 * HEAD_DIM * 4


def test_a_budget_without_an_owner_is_refused():
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

    with pytest.raises(PromptVectorError, match="budget_owner"):
        extract(budget=TransferBudget(1 << 20, 4))


# --------------------------------------------------------------------------
# Filters use global ids
# --------------------------------------------------------------------------


def test_layers_and_heads_can_be_selected_by_global_id():
    vectors, _, _ = extract(rank=1, layers=[1], kv_heads=[3])
    assert [(v.layer, v.kv_head) for v in vectors] == [(1, 3)]


def test_asking_for_a_peer_shard_s_head_is_an_error_not_an_empty_result():
    with pytest.raises(PromptVectorError, match="KV head 3 is not held"):
        extract(rank=0, kv_heads=[3])


def test_asking_for_a_layer_outside_the_shard_is_refused():
    with pytest.raises(PromptVectorError, match="layer 9 is not held"):
        extract(layers=[9])


@pytest.mark.parametrize("bad", [[], [0, 0], [True], ["0"]])
def test_an_invalid_filter_is_refused(bad):
    with pytest.raises(PromptVectorError):
        extract(layers=bad)


# --------------------------------------------------------------------------
# Positional encoding semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "rope", None, 7])
def test_the_positional_encoding_must_be_stated_explicitly(bad):
    with pytest.raises(PromptVectorError, match="positional_encoding"):
        extract(encoding=bad)


def test_a_query_with_a_different_encoding_is_refused():
    vectors, _, _ = extract(encoding=ROPE_APPLIED)
    with pytest.raises(PromptVectorError, match="differently-encoded"):
        vectors[0].require_compatible_query(
            positional_encoding=NO_POSITIONAL_ENCODING, head_dim=HEAD_DIM
        )


def test_a_query_of_the_wrong_head_dim_is_refused():
    vectors, _, _ = extract()
    with pytest.raises(PromptVectorError, match="head_dim"):
        vectors[0].require_compatible_query(
            positional_encoding=ROPE_APPLIED, head_dim=HEAD_DIM + 1
        )


def test_a_matching_query_is_accepted():
    vectors, _, _ = extract()
    vectors[0].require_compatible_query(
        positional_encoding=ROPE_APPLIED, head_dim=HEAD_DIM
    )


# --------------------------------------------------------------------------
# Malformed layouts and manifests
# --------------------------------------------------------------------------


def broken_layout(pool, **changes):
    layout = storage_layout(pool)
    extra = copy.deepcopy(dict(layout.extra))
    extra.update(changes.pop("extra", {}))
    fields = dict(
        model_id=layout.model_id,
        model_revision=layout.model_revision,
        kv_dtype=layout.kv_dtype,
        page_size=layout.page_size,
        num_layers=layout.num_layers,
        total_kv_heads=layout.total_kv_heads,
        kv_heads_per_rank=layout.kv_heads_per_rank,
        head_dim=layout.head_dim,
        tp_size=layout.tp_size,
        pp_size=layout.pp_size,
        tensor_layout=layout.tensor_layout,
        extra=extra,
    )
    fields.update(changes)
    return KVLayoutSignature(**fields)


@pytest.mark.parametrize(
    "key",
    [
        "component_count",
        "component_dtypes",
        "component_token_shapes",
        "component_bytes_per_token",
    ],
)
def test_missing_layout_metadata_is_refused(key):
    pool = FakePool()
    extra = copy.deepcopy(dict(storage_layout(pool).extra))
    del extra[key]
    layout = broken_layout(pool)
    object.__setattr__(layout, "extra", extra)
    with pytest.raises(PromptVectorError, match="missing"):
        extract(pool, layout=layout)


def test_an_odd_component_count_is_refused():
    pool = FakePool()
    layout = broken_layout(pool, extra={"component_count": LAYERS * 2 - 1})
    with pytest.raises(PromptVectorError, match="paired K and V"):
        extract(pool, layout=layout)


def test_disagreeing_component_metadata_lengths_are_refused():
    pool = FakePool()
    layout = broken_layout(pool, extra={"component_dtypes": ["torch.float16"]})
    with pytest.raises(PromptVectorError, match="lengths disagree"):
        extract(pool, layout=layout)


def test_a_head_count_that_contradicts_the_component_shape_is_refused():
    pool = FakePool()
    layout = broken_layout(pool, kv_heads_per_rank=1)
    with pytest.raises(PromptVectorError):
        extract(pool, layout=layout)


def test_a_head_dim_that_contradicts_the_component_shape_is_refused():
    pool = FakePool()
    layout = broken_layout(pool, head_dim=HEAD_DIM * 2)
    with pytest.raises(PromptVectorError):
        extract(pool, layout=layout)


def test_a_byte_count_that_does_not_match_the_layout_is_refused():
    pool = FakePool()
    layout = storage_layout(pool)
    packed, manifest, _ = pack_shard(pool, layout, rank=0, prompt_tokens=8)
    with pytest.raises(PromptVectorError, match="bytes"):
        extract_prompt_k(
            packed.tensor[:-4],
            layout=layout,
            manifest=manifest,
            entry_transfer_id=ENTRY,
            id_mapping_version=MAPPING,
            positional_encoding=ROPE_APPLIED,
        )


def test_a_manifest_declaring_the_wrong_size_is_refused():
    pool = FakePool()
    layout = storage_layout(pool)
    packed, manifest, _ = pack_shard(pool, layout, rank=0, prompt_tokens=8)
    wrong = KVShardManifest(
        rank=manifest.rank,
        rail=manifest.rail,
        expected_bytes=manifest.expected_bytes + 16,
        page_count=manifest.page_count,
        last_page_valid_tokens=manifest.last_page_valid_tokens,
        layer_start=manifest.layer_start,
        layer_end=manifest.layer_end,
    )
    with pytest.raises(PromptVectorError, match="manifest declares"):
        extract_prompt_k(
            packed.tensor,
            layout=layout,
            manifest=wrong,
            entry_transfer_id=ENTRY,
            id_mapping_version=MAPPING,
            positional_encoding=ROPE_APPLIED,
        )


def test_a_layer_range_that_does_not_match_the_buffer_is_refused():
    pool = FakePool()
    layout = storage_layout(pool)
    packed, manifest, _ = pack_shard(pool, layout, rank=0, prompt_tokens=8)
    wrong = KVShardManifest(
        rank=0,
        rail=manifest.rail,
        expected_bytes=manifest.expected_bytes,
        page_count=manifest.page_count,
        last_page_valid_tokens=manifest.last_page_valid_tokens,
        layer_start=0,
        layer_end=LAYERS - 1,
    )
    with pytest.raises(PromptVectorError, match="layers but the buffer"):
        extract_prompt_k(
            packed.tensor,
            layout=layout,
            manifest=wrong,
            entry_transfer_id=ENTRY,
            id_mapping_version=MAPPING,
            positional_encoding=ROPE_APPLIED,
        )


def test_a_non_flat_buffer_is_refused():
    pool = FakePool()
    layout = storage_layout(pool)
    packed, manifest, _ = pack_shard(pool, layout, rank=0, prompt_tokens=8)
    with pytest.raises(PromptVectorError, match="flat byte buffer"):
        extract_prompt_k(
            packed.tensor.reshape(2, -1),
            layout=layout,
            manifest=manifest,
            entry_transfer_id=ENTRY,
            id_mapping_version=MAPPING,
            positional_encoding=ROPE_APPLIED,
        )


# --------------------------------------------------------------------------
# Query-head to KV-head mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q,kv,kind,group", [(4, 4, "mha", 1), (8, 4, "gqa", 2), (8, 1, "mqa", 8)]
)
def test_supported_attention_layouts(q, kv, kind, group):
    mapping = QueryHeadMapping(num_query_heads=q, total_kv_heads=kv)
    assert (mapping.kind, mapping.group_size) == (kind, group)


def test_gqa_groups_consecutive_query_heads_onto_one_kv_head():
    mapping = QueryHeadMapping(num_query_heads=8, total_kv_heads=4)
    assert [mapping.kv_head_for(h) for h in range(8)] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert mapping.query_heads_for(2) == (4, 5)


def test_mqa_routes_every_query_head_to_the_single_kv_head():
    mapping = QueryHeadMapping(num_query_heads=8, total_kv_heads=1)
    assert {mapping.kv_head_for(h) for h in range(8)} == {0}


@pytest.mark.parametrize("q,kv", [(6, 4), (5, 2), (7, 3)])
def test_a_non_divisible_attention_layout_is_rejected(q, kv):
    with pytest.raises(PromptVectorError, match="do not divide evenly"):
        QueryHeadMapping(num_query_heads=q, total_kv_heads=kv)


def test_fewer_query_heads_than_kv_heads_is_rejected():
    with pytest.raises(PromptVectorError, match="fewer query heads"):
        QueryHeadMapping(num_query_heads=2, total_kv_heads=4)


@pytest.mark.parametrize("bad", [-1, 8, True, 1.0])
def test_an_out_of_range_query_head_is_refused(bad):
    mapping = QueryHeadMapping(num_query_heads=8, total_kv_heads=4)
    with pytest.raises(PromptVectorError):
        mapping.kv_head_for(bad)


# --------------------------------------------------------------------------
# Integration: stored shard -> vectors -> exact index -> original tokens
# --------------------------------------------------------------------------


def test_a_stored_shard_round_trips_to_the_original_token_and_page():
    """Synthetic queries: this checks identity plumbing, not retrieval quality."""
    prompt_tokens = 10
    vectors, _, _ = extract(prompt_tokens=prompt_tokens)
    backend = BruteForceIndexBackend()
    for target_row in (0, 5, 9):
        for v in vectors:
            index = backend.build(
                v.vectors, vector_space="target/model-8b", metric="l2"
            )
            query = v.vectors[target_row : target_row + 1]
            v.require_compatible_query(
                positional_encoding=v.positional_encoding, head_dim=v.head_dim
            )
            selection = select(
                backend,
                index,
                query,
                layer=v.layer,
                kv_head=v.kv_head,
                mapping=v.mapping,
                top_k=1,
            )
            assert selection.token_ids == (target_row,)
            assert selection.page_ids == (target_row // PAGE_SIZE,)
            assert selection.kv_head == v.kv_head
            assert selection.layer == v.layer


def test_head_identity_survives_merging():
    vectors, _, _ = extract(prompt_tokens=8, layers=[0])
    backend = BruteForceIndexBackend()
    selections = []
    for v in vectors:
        index = backend.build(v.vectors, vector_space="s", metric="l2")
        selections.append(
            select(
                backend,
                index,
                v.vectors[0:1],
                layer=v.layer,
                kv_head=v.kv_head,
                mapping=v.mapping,
                top_k=1,
            )
        )
    merged = merge_selections(selections, policy="per_layer")
    assert set(merged) == {(0, 0), (0, 1)}


def test_two_shards_cover_every_global_head_without_overlap():
    pool = FakePool()
    heads = set()
    for rank in (0, 1):
        vectors, _, _ = extract(pool, rank=rank, prompt_tokens=8)
        shard_heads = {v.kv_head for v in vectors}
        assert not (heads & shard_heads)
        heads |= shard_heads
    assert heads == set(range(TOTAL_KV_HEADS))


def test_extraction_places_its_copy_where_it_is_told():
    """'meta' stands in for a device other than the pool's; see index_search."""
    pool = FakePool()
    layout = storage_layout(pool)
    packed, shard, _ = pack_shard(pool, layout, rank=0, prompt_tokens=8)
    kwargs = dict(
        layout=layout,
        manifest=shard,
        entry_transfer_id="t",
        id_mapping_version="map-1",
        positional_encoding=ROPE_APPLIED,
    )
    here = extract_prompt_k(packed.tensor, **kwargs)
    assert {v.vectors.device for v in here} == {packed.tensor.device}
    there = extract_prompt_k(packed.tensor, device="meta", **kwargs)
    assert {v.vectors.device for v in there} == {torch.device("meta")}
    # The source buffer is untouched by either: extraction copies, never moves.
    assert packed.tensor.device == here[0].vectors.device
