"""Pure layout and packed-buffer sharding rules for PVD 2.0.

The V group owns a stable storage TP layout. Legacy delivery requires each D
interval to fit in one V shard. The general byte planner also represents fan-in;
it does not activate multi-source delivery or change the legacy wire contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Tuple

from sglang.srt.disaggregation.pvd.protocol import (
    KVLayoutSignature,
    ProtocolValidationError,
    RemoteRegionDescriptor,
)


@dataclass(frozen=True)
class PackedTransferSlice:
    local_offset: int
    remote_offset: int
    length: int


@dataclass(frozen=True)
class HeadShardIntersection:
    storage_rank: int
    storage_head_offset: int
    compute_head_offset: int
    head_count: int


def _component_bytes(layout: KVLayoutSignature) -> List[int]:
    values = layout.extra.get("component_bytes_per_token")
    if not isinstance(values, list) or not values:
        raise ProtocolValidationError("layout has no component byte description")
    if any(type(value) is not int or value <= 0 for value in values):
        raise ProtocolValidationError("component bytes per token must be positive")
    return list(values)


def validate_compute_layout(
    storage: KVLayoutSignature, compute: KVLayoutSignature
) -> None:
    """Validate semantic compatibility while deliberately allowing TP changes."""
    for layout in (storage, compute):
        for name in ("tp_size", "total_kv_heads", "kv_heads_per_rank"):
            value = getattr(layout, name)
            if type(value) is not int or value <= 0:
                raise ProtocolValidationError(f"{name} must be a positive integer")
    fields = (
        "model_id",
        "model_revision",
        "kv_dtype",
        "page_size",
        "num_layers",
        "total_kv_heads",
        "head_dim",
        "pp_size",
        "tensor_layout",
    )
    mismatches = [
        name for name in fields if getattr(storage, name) != getattr(compute, name)
    ]
    if mismatches:
        raise ProtocolValidationError(
            "storage/compute KV layouts differ in: " + ", ".join(mismatches)
        )
    source_bytes = _component_bytes(storage)
    destination_bytes = _component_bytes(compute)
    if len(source_bytes) != len(destination_bytes):
        raise ProtocolValidationError("storage/compute component counts differ")
    if storage.extra.get("component_dtypes") != compute.extra.get("component_dtypes"):
        raise ProtocolValidationError("storage/compute component dtypes differ")
    source_shapes = storage.extra.get("component_token_shapes")
    destination_shapes = compute.extra.get("component_token_shapes")
    if (
        not isinstance(source_shapes, list)
        or not isinstance(destination_shapes, list)
        or len(source_shapes) != len(destination_shapes)
        or len(source_shapes) != len(source_bytes)
    ):
        raise ProtocolValidationError("storage/compute component shapes differ")
    for source_shape, destination_shape in zip(source_shapes, destination_shapes):
        if (
            not isinstance(source_shape, (list, tuple))
            or not isinstance(destination_shape, (list, tuple))
            or not source_shape
            or not destination_shape
            or source_shape[0] != storage.kv_heads_per_rank
            or destination_shape[0] != compute.kv_heads_per_rank
            or list(source_shape[1:]) != list(destination_shape[1:])
        ):
            raise ProtocolValidationError("storage/compute non-head shapes differ")
    if storage.total_kv_heads % storage.tp_size:
        raise ProtocolValidationError("V storage TP must evenly shard total KV heads")
    if compute.total_kv_heads % compute.tp_size:
        raise ProtocolValidationError(
            "PVD 2.0 currently requires compute TP to evenly shard KV heads"
        )
    if storage.kv_heads_per_rank != storage.total_kv_heads // storage.tp_size:
        raise ProtocolValidationError("invalid V storage KV-head count")
    if compute.kv_heads_per_rank != compute.total_kv_heads // compute.tp_size:
        raise ProtocolValidationError("invalid compute KV-head count")
    for source, destination in zip(source_bytes, destination_bytes):
        if source % storage.kv_heads_per_rank:
            raise ProtocolValidationError("V component cannot be split by KV head")
        if destination % compute.kv_heads_per_rank:
            raise ProtocolValidationError(
                "compute component cannot be split by KV head"
            )
        if (
            source // storage.kv_heads_per_rank
            != destination // compute.kv_heads_per_rank
        ):
            raise ProtocolValidationError("storage/compute KV-head byte sizes differ")


def source_rank_and_head_offset(
    storage: KVLayoutSignature,
    compute: KVLayoutSignature,
    compute_rank: int,
) -> Tuple[int, int]:
    parts = source_shard_intersections(storage, compute, compute_rank)
    if len(parts) != 1:
        raise ProtocolValidationError(
            "compute-rank KV heads cross a V shard boundary; topology is unsupported"
        )
    return parts[0].storage_rank, parts[0].storage_head_offset


def source_shard_intersections(
    storage: KVLayoutSignature,
    compute: KVLayoutSignature,
    compute_rank: int,
) -> Tuple[HeadShardIntersection, ...]:
    """Partition one D head interval among V shards, without wire activation.

    This planner permits fan-in. The existing one-source delivery protocol does
    not; its source_rank_and_head_offset entry point still refuses such plans.
    """
    validate_compute_layout(storage, compute)
    if type(compute_rank) is not int or not 0 <= compute_rank < compute.tp_size:
        raise ProtocolValidationError(f"compute rank {compute_rank} is out of range")
    start = compute_rank * compute.kv_heads_per_rank
    end = start + compute.kv_heads_per_rank
    parts = []
    cursor = start
    while cursor < end:
        rank, offset = divmod(cursor, storage.kv_heads_per_rank)
        count = min(end - cursor, storage.kv_heads_per_rank - offset)
        parts.append(HeadShardIntersection(rank, offset, cursor - start, count))
        cursor += count
    return tuple(parts)


def layout_from_destination(
    destination: RemoteRegionDescriptor,
) -> KVLayoutSignature:
    value = destination.backend_metadata.get("pvd_layout")
    if not isinstance(value, Mapping):
        raise ProtocolValidationError("D destination is missing pvd_layout metadata")
    return KVLayoutSignature.from_dict(value)


def packed_transfer_slices(
    storage: KVLayoutSignature,
    compute: KVLayoutSignature,
    *,
    compute_rank: int,
    token_count: int,
) -> List[PackedTransferSlice]:
    """Map one packed V shard into one packed D rank without GPU repacking."""
    source_rank, _ = source_rank_and_head_offset(storage, compute, compute_rank)
    return packed_fanin_transfer_slices(
        storage, compute, compute_rank=compute_rank, token_count=token_count
    )[source_rank]


def packed_fanin_transfer_slices(
    storage: KVLayoutSignature,
    compute: KVLayoutSignature,
    *,
    compute_rank: int,
    token_count: int,
) -> Mapping[int, List[PackedTransferSlice]]:
    """Byte-only plan keyed by V rank; disjoint writes exactly cover D's buffer.

    token_count includes the final page's padding, as kv_packer does. Returned
    offsets are relative, not MR addresses/permissions or completion receipts.
    Every source must eventually be independently fenced by the wire protocol.
    """
    if type(token_count) is not int or token_count <= 0:
        raise ProtocolValidationError("packed transfer token count must be positive")
    parts = source_shard_intersections(storage, compute, compute_rank)
    source_components = _component_bytes(storage)
    destination_components = _component_bytes(compute)
    by_rank = {part.storage_rank: [] for part in parts}
    source_component_base = 0
    destination_component_base = 0
    for source_bpt, destination_bpt in zip(source_components, destination_components):
        bytes_per_head = source_bpt // storage.kv_heads_per_rank
        for part in parts:
            for token in range(token_count):
                by_rank[part.storage_rank].append(
                    PackedTransferSlice(
                        local_offset=(
                            source_component_base
                            + token * source_bpt
                            + part.storage_head_offset * bytes_per_head
                        ),
                        remote_offset=(
                            destination_component_base
                            + token * destination_bpt
                            + part.compute_head_offset * bytes_per_head
                        ),
                        length=part.head_count * bytes_per_head,
                    )
                )
        source_component_base += token_count * source_bpt
        destination_component_base += token_count * destination_bpt
    return by_rank
