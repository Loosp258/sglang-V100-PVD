"""Pure layout and packed-buffer sharding rules for PVD 2.0.

The V group owns a stable storage TP layout.  P and D may use a different
compute TP layout as long as every compute-rank KV-head interval is wholly
contained in one V shard.  The first supported heterogeneous topology is
P TP1 -> V TP2 -> D TP4 for ordinary MHA/GQA KV pools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Tuple

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


def _component_bytes(layout: KVLayoutSignature) -> List[int]:
    values = layout.extra.get("component_bytes_per_token")
    if not isinstance(values, list) or not values:
        raise ProtocolValidationError("layout has no component byte description")
    result = [int(value) for value in values]
    if any(value <= 0 for value in result):
        raise ProtocolValidationError("component bytes per token must be positive")
    return result


def validate_compute_layout(
    storage: KVLayoutSignature, compute: KVLayoutSignature
) -> None:
    """Validate semantic compatibility while deliberately allowing TP changes."""
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
    if storage.extra.get("component_dtypes") != compute.extra.get(
        "component_dtypes"
    ):
        raise ProtocolValidationError("storage/compute component dtypes differ")
    source_shapes = storage.extra.get("component_token_shapes")
    destination_shapes = compute.extra.get("component_token_shapes")
    if (
        not isinstance(source_shapes, list)
        or not isinstance(destination_shapes, list)
        or len(source_shapes) != len(destination_shapes)
    ):
        raise ProtocolValidationError("storage/compute component shapes differ")
    for source_shape, destination_shape in zip(source_shapes, destination_shapes):
        if (
            not source_shape
            or not destination_shape
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
            raise ProtocolValidationError("compute component cannot be split by KV head")
        if source // storage.kv_heads_per_rank != destination // compute.kv_heads_per_rank:
            raise ProtocolValidationError("storage/compute KV-head byte sizes differ")


def source_rank_and_head_offset(
    storage: KVLayoutSignature,
    compute: KVLayoutSignature,
    compute_rank: int,
) -> Tuple[int, int]:
    validate_compute_layout(storage, compute)
    if compute_rank < 0 or compute_rank >= compute.tp_size:
        raise ProtocolValidationError(f"compute rank {compute_rank} is out of range")
    global_head_start = compute_rank * compute.kv_heads_per_rank
    source_rank = global_head_start // storage.kv_heads_per_rank
    source_head_offset = global_head_start % storage.kv_heads_per_rank
    if source_head_offset + compute.kv_heads_per_rank > storage.kv_heads_per_rank:
        raise ProtocolValidationError(
            "compute-rank KV heads cross a V shard boundary; topology is unsupported"
        )
    return source_rank, source_head_offset


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
    if token_count <= 0:
        raise ProtocolValidationError("packed transfer token count must be positive")
    _, source_head_offset = source_rank_and_head_offset(
        storage, compute, compute_rank
    )
    source_components = _component_bytes(storage)
    destination_components = _component_bytes(compute)
    slices: List[PackedTransferSlice] = []
    source_component_base = 0
    destination_component_base = 0
    for source_bpt, destination_bpt in zip(
        source_components, destination_components
    ):
        bytes_per_head = source_bpt // storage.kv_heads_per_rank
        source_head_byte_offset = source_head_offset * bytes_per_head
        for token in range(token_count):
            slices.append(
                PackedTransferSlice(
                    local_offset=(
                        source_component_base
                        + token * source_bpt
                        + source_head_byte_offset
                    ),
                    remote_offset=(
                        destination_component_base + token * destination_bpt
                    ),
                    length=destination_bpt,
                )
            )
        source_component_base += token_count * source_bpt
        destination_component_base += token_count * destination_bpt
    return slices
