"""Local rank-packed fan-in reorder after every V writer is terminal.

This module never reads a published receive region before the caller has
validated all writer proofs and fenced CUDA visibility. It is not an RDMA
completion mechanism.
"""

import torch
from sglang.srt.disaggregation.pvd.sharding import RankPackedFullShardPlan


def scatter_rank_packed_bytes(
    source: torch.Tensor,
    destination: torch.Tensor,
    plan: RankPackedFullShardPlan,
    *,
    backend: str = "torch",
) -> None:
    """Reorder V-rank-contiguous bytes into the canonical D component layout."""
    if backend not in ("torch", "triton"):
        raise ValueError("rank-packed scatter backend must be torch or triton")
    if (
        not isinstance(plan, RankPackedFullShardPlan)
        or not isinstance(source, torch.Tensor)
        or not isinstance(destination, torch.Tensor)
        or source.dtype != torch.uint8
        or destination.dtype != torch.uint8
        or source.device != destination.device
        or not source.is_contiguous()
        or not destination.is_contiguous()
        or source.numel() != plan.staging_bytes
        or destination.numel() != plan.staging_bytes
        or source.untyped_storage().data_ptr()
        == destination.untyped_storage().data_ptr()
    ):
        raise ValueError("distinct same-device rank-packed byte buffers required")
    for rule in plan.scatters:
        source_end = (
            rule.source_offset
            + (rule.token_count - 1) * rule.source_stride
            + rule.width
        )
        destination_end = (
            rule.destination_offset
            + (rule.token_count - 1) * rule.destination_stride
            + rule.width
        )
        if (
            min(
                rule.source_offset,
                rule.destination_offset,
                rule.source_stride,
                rule.destination_stride,
                rule.width,
                rule.token_count,
            )
            < 0
            or rule.width <= 0
            or rule.token_count <= 0
            or rule.source_stride < rule.width
            or rule.destination_stride < rule.width
            or source_end > source.numel()
            or destination_end > destination.numel()
        ):
            raise ValueError("rank-packed scatter rule is outside its buffers")
    if backend == "triton":
        if not source.is_cuda:
            raise ValueError("Triton rank-packed scatter requires CUDA buffers")
        components, parts, tokens, width = _uniform_rank_packed_shape(plan)
        from sglang.srt.disaggregation.pvd.triton_fanin_scatter import (
            scatter_uniform_rank_packed,
        )

        scatter_uniform_rank_packed(
            source,
            destination,
            components=components,
            parts=parts,
            tokens=tokens,
            width=width,
        )
        return
    for rule in plan.scatters:
        source_view = source.narrow(
            0,
            rule.source_offset,
            (rule.token_count - 1) * rule.source_stride + rule.width,
        ).as_strided((rule.token_count, rule.width), (rule.source_stride, 1))
        destination_view = destination.narrow(
            0,
            rule.destination_offset,
            (rule.token_count - 1) * rule.destination_stride + rule.width,
        ).as_strided((rule.token_count, rule.width), (rule.destination_stride, 1))
        destination_view.copy_(source_view)


def _uniform_rank_packed_shape(
    plan: RankPackedFullShardPlan,
) -> tuple[int, int, int, int]:
    """Prove the single-kernel byte mapping agrees with every wire/copy rule."""
    parts = len(plan.transfers)
    if not parts or not plan.scatters or len(plan.scatters) % parts:
        raise ValueError("Triton scatter requires a uniform whole-shard plan")
    components = len(plan.scatters) // parts
    first = plan.scatters[0]
    tokens, width = first.token_count, first.width
    source_bytes = components * tokens * width
    if tokens <= 0 or width <= 0 or plan.staging_bytes != parts * source_bytes:
        raise ValueError("Triton scatter requires a uniform whole-shard extent")
    if len({rank for rank, _ in plan.transfers}) != parts or any(
        transfer.local_offset != 0
        or transfer.remote_offset != part * source_bytes
        or transfer.length != source_bytes
        for part, (_, transfer) in enumerate(plan.transfers)
    ):
        raise ValueError("Triton scatter requires complete, ordered V partitions")
    for component in range(components):
        for part in range(parts):
            rule = plan.scatters[component * parts + part]
            if (
                rule.token_count != tokens
                or rule.width != width
                or rule.source_stride != width
                or rule.destination_stride != parts * width
                or rule.source_offset
                != part * source_bytes + component * tokens * width
                or rule.destination_offset
                != component * tokens * parts * width + part * width
            ):
                raise ValueError("Triton scatter cannot represent this copy rule")
    return components, parts, tokens, width
