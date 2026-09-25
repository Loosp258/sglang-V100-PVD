"""Local rank-packed fan-in reorder after every V writer is terminal.

This module never reads a published receive region before the caller has
validated all writer proofs and fenced CUDA visibility. It is not an RDMA
completion mechanism.
"""

import torch
from sglang.srt.disaggregation.pvd.sharding import RankPackedFullShardPlan


def scatter_rank_packed_bytes(
    source: torch.Tensor, destination: torch.Tensor, plan: RankPackedFullShardPlan
) -> None:
    """Reorder V-rank-contiguous bytes into the canonical D component layout."""
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
        or source.data_ptr() == destination.data_ptr()
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
