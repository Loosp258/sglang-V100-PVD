"""Actual packed CPU bytes; mapping only, not multi-source delivery evidence."""

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch
from sglang.srt.disaggregation.pvd.kv_packer import (
    describe_kv_layout,
    pack_full_prompt_kv,
)
from sglang.srt.disaggregation.pvd.protocol import (
    KVLayoutSignature,
    ProtocolValidationError,
)
from sglang.srt.disaggregation.pvd.sharding import (
    packed_fanin_transfer_slices,
    packed_transfer_slices,
    rank_packed_full_shard_fanin_plan,
    source_rank_and_head_offset,
    source_shard_intersections,
)


def layout(tp, *, page_size=2):
    heads = 12 // tp
    pool = NS(
        k_buffer=[torch.zeros((8, heads, 3), dtype=torch.float16) for _ in range(2)],
        v_buffer=[torch.zeros((8, heads, 3), dtype=torch.float16) for _ in range(2)],
    )
    description = describe_kv_layout(pool)
    return KVLayoutSignature(
        model_id="test",
        model_revision="1",
        kv_dtype="float16",
        page_size=page_size,
        num_layers=2,
        total_kv_heads=12,
        kv_heads_per_rank=heads,
        head_dim=3,
        tp_size=tp,
        pp_size=1,
        tensor_layout=description["tensor_layout"],
        extra=description,
    )


@pytest.mark.parametrize("v_tp,d_tp", [(2, 1), (3, 2), (4, 3), (2, 4), (1, 3), (3, 3)])
@pytest.mark.parametrize("page_size", [1, 2])
def test_fanin_reconstructs_every_destination_byte_once(v_tp, d_tp, page_size):
    # Global source independently defines component/token/head/dim values.
    components = [
        (torch.arange(8 * 12 * 3).reshape(8, 12, 3) + 300 * component).to(torch.float16)
        for component in range(4)
    ]
    pages = torch.tensor([2, 0, 3])  # Non-contiguous, non-sorted pages incl padding.

    def packed(tp, rank):
        heads = 12 // tp
        parts = [t[:, rank * heads : (rank + 1) * heads, :].clone() for t in components]
        return pack_full_prompt_kv(
            NS(k_buffer=parts[:2], v_buffer=parts[2:]), pages, page_size=page_size
        ).tensor

    storage, compute = (
        layout(v_tp, page_size=page_size),
        layout(d_tp, page_size=page_size),
    )
    sources = {rank: packed(v_tp, rank) for rank in range(v_tp)}
    for rank in range(d_tp):
        expected = packed(d_tp, rank)
        destination = torch.zeros_like(expected)
        writes = torch.zeros_like(expected, dtype=torch.int32)
        plans = packed_fanin_transfer_slices(
            storage, compute, compute_rank=rank, token_count=len(pages) * page_size
        )
        for source_rank, slices in plans.items():
            for part in slices:
                lo, ro, size = part.local_offset, part.remote_offset, part.length
                assert 0 <= lo < lo + size <= len(sources[source_rank])
                assert 0 <= ro < ro + size <= len(destination)
                destination[ro : ro + size] = sources[source_rank][lo : lo + size]
                writes[ro : ro + size] += 1
        assert torch.equal(destination, expected)
        assert torch.all(writes == 1), "missing or overlapping source writes"
        if len(plans) == 1:
            assert packed_transfer_slices(
                storage, compute, compute_rank=rank, token_count=len(pages) * page_size
            ) == next(iter(plans.values()))
        else:
            # Current coordinator/wire entry MUST still refuse fan-in.
            with pytest.raises(ProtocolValidationError, match="cross a V shard"):
                source_rank_and_head_offset(storage, compute, rank)
            with pytest.raises(ProtocolValidationError, match="cross a V shard"):
                packed_transfer_slices(
                    storage, compute, compute_rank=rank, token_count=3
                )


def test_non_aligned_shards_have_both_source_and_destination_offsets():
    parts = source_shard_intersections(layout(3), layout(2), 1)
    assert [
        (p.storage_rank, p.storage_head_offset, p.compute_head_offset, p.head_count)
        for p in parts
    ] == [(1, 2, 0, 2), (2, 0, 2, 4)]


@pytest.mark.parametrize("v_tp,d_tp", [(2, 1), (3, 1), (4, 1), (4, 2), (3, 3)])
@pytest.mark.parametrize("page_size", [1, 2])
def test_rank_packed_whole_shard_plan_reconstructs_canonical_bytes(
    v_tp, d_tp, page_size
):
    components = [
        (torch.arange(8 * 12 * 3).reshape(8, 12, 3) + 300 * component).to(torch.float16)
        for component in range(4)
    ]
    pages = torch.tensor([2, 0, 3])

    def packed(tp, rank):
        heads = 12 // tp
        parts = [t[:, rank * heads : (rank + 1) * heads, :].clone() for t in components]
        return pack_full_prompt_kv(
            NS(k_buffer=parts[:2], v_buffer=parts[2:]), pages, page_size=page_size
        ).tensor

    sources = {rank: packed(v_tp, rank) for rank in range(v_tp)}
    for compute_rank in range(d_tp):
        expected = packed(d_tp, compute_rank)
        plan = rank_packed_full_shard_fanin_plan(
            layout(v_tp, page_size=page_size),
            layout(d_tp, page_size=page_size),
            compute_rank=compute_rank,
            token_count=len(pages) * page_size,
        )
        assert plan.staging_bytes == expected.numel()
        assert len(plan.transfers) == v_tp // d_tp
        staging = torch.zeros(plan.staging_bytes, dtype=torch.uint8)
        wire_writes = torch.zeros_like(staging, dtype=torch.int32)
        for source_rank, transfer in plan.transfers:
            assert transfer.local_offset == 0
            assert transfer.length == sources[source_rank].numel()
            start = transfer.remote_offset
            stop = start + transfer.length
            staging[start:stop] = sources[source_rank]
            wire_writes[start:stop] += 1
        assert torch.all(wire_writes == 1)

        canonical = torch.zeros_like(expected)
        local_writes = torch.zeros_like(expected, dtype=torch.int32)
        for rule in plan.scatters:
            for token in range(rule.token_count):
                source = rule.source_offset + token * rule.source_stride
                destination = rule.destination_offset + token * rule.destination_stride
                canonical[destination : destination + rule.width] = staging[
                    source : source + rule.width
                ]
                local_writes[destination : destination + rule.width] += 1
        assert torch.equal(canonical, expected)
        assert torch.all(local_writes == 1)


@pytest.mark.parametrize("v_tp,d_tp,rank", [(2, 4, 0), (3, 2, 1), (4, 3, 1)])
def test_rank_packed_plan_refuses_partial_v_shard(v_tp, d_tp, rank):
    with pytest.raises(ProtocolValidationError, match="complete V source shards"):
        rank_packed_full_shard_fanin_plan(
            layout(v_tp), layout(d_tp), compute_rank=rank, token_count=4
        )


@pytest.mark.parametrize("tokens", [0, -1, True, 1.0, "1"])
def test_rank_packed_plan_refuses_invalid_token_count(tokens):
    with pytest.raises(ProtocolValidationError, match="token count"):
        rank_packed_full_shard_fanin_plan(
            layout(2), layout(1), compute_rank=0, token_count=tokens
        )


@pytest.mark.parametrize("rank", [-1, 1, True, 0.0, "0"])
def test_rank_must_be_an_exact_in_range_integer(rank):
    with pytest.raises(ProtocolValidationError, match="out of range"):
        source_shard_intersections(layout(2), layout(1), rank)


@pytest.mark.parametrize("tokens", [0, -1, True, 1.0, "1"])
def test_token_count_is_not_coerced(tokens):
    with pytest.raises(ProtocolValidationError, match="token count"):
        packed_fanin_transfer_slices(
            layout(2), layout(1), compute_rank=0, token_count=tokens
        )


@pytest.mark.parametrize(
    "fault", ["bytes", "shape-count", "head-count", "tp", "dtype", "model"]
)
def test_incompatible_or_ambiguous_layout_refused(fault):
    storage, compute = layout(2), layout(1)
    extra = dict(compute.extra)
    if fault == "bytes":
        extra["component_bytes_per_token"] = [72.5] * 4
    elif fault == "shape-count":
        extra["component_token_shapes"] = [[12, 3]]
    elif fault == "head-count":
        extra["component_token_shapes"] = [[6, 3]] * 4
    elif fault == "tp":
        compute = replace(compute, tp_size=1.0)
    elif fault == "dtype":
        extra["component_dtypes"] = ["torch.float32"] * 4
    else:
        compute = replace(compute, model_id="foreign")
    compute = replace(compute, extra=extra)
    with pytest.raises(ProtocolValidationError):
        packed_fanin_transfer_slices(storage, compute, compute_rank=0, token_count=3)
