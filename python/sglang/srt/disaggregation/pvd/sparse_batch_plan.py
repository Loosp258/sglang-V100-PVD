"""Checked sparse scatter reads from one registered immutable Entry.

This CPU plan grants no write permission or completion proof. The caller owns
the Entry allocation and index lease, authorizes the destination separately,
and keeps them through aggregate native completion. No KV is copied here.
"""

from dataclasses import dataclass

import torch
from sglang.srt.disaggregation.pvd.protocol import KVLayoutSignature, KVShardManifest
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError, _views
from sglang.srt.disaggregation.pvd.transfer_engine import MemorySlice, RegisteredMemory

MAX_DIRECT_SPARSE_SLICES = 128


@dataclass(frozen=True)
class SparseBatchPlan:
    slices: tuple[MemorySlice, ...]
    remote_offsets: tuple[int, ...]
    nbytes: int


def build_sparse_batch_plan(
    manifest, layout, shard, *, entry_transfer_id, index_version,
    id_mapping_version, allocation_offset, registration,
    max_slices=MAX_DIRECT_SPARSE_SLICES,
) -> SparseBatchPlan:
    """Keep spec/token order and cover exactly the existing sparse wire layout."""
    if (
        not isinstance(manifest, SparseDeliveryManifest)
        or not isinstance(registration, RegisteredMemory)
        or not isinstance(layout, KVLayoutSignature)
        or not isinstance(shard, KVShardManifest)
    ):
        raise SparsePayloadError("explicit sparse manifest and registration required")
    if type(max_slices) is not int or not 0 < max_slices <= MAX_DIRECT_SPARSE_SLICES:
        raise SparsePayloadError("direct sparse slice bound must be in [1, 128]")
    if 2 * sum(len(spec.token_ids) for spec in manifest.specs) > max_slices:
        raise SparsePayloadError("direct sparse native slice bound exceeded")
    if type(allocation_offset) is not int or allocation_offset < 0:
        raise SparsePayloadError("explicit nonnegative Entry allocation offset required")
    buffer = registration.buffer
    descriptor = registration.descriptor
    if (
        not isinstance(buffer, torch.Tensor)
        or buffer.dtype != torch.uint8
        or buffer.ndim != 1
        or not buffer.is_contiguous()
        or descriptor.address != buffer.data_ptr()
        or descriptor.length != buffer.numel()
        or descriptor.device != str(buffer.device)
    ):
        raise SparsePayloadError("original registered pool backing storage changed")
    # _views is the existing packer's full component dtype/shape/byte validator.
    # Its views are metadata only, with no GPU allocation, copy or kernel.
    if (
        descriptor.rank != shard.rank or descriptor.rail != shard.rail
        or shard.page_count <= 0
        or shard.expected_bytes <= 0
        or shard.expected_bytes % shard.page_count
        or allocation_offset % (shard.expected_bytes // shard.page_count)
        or allocation_offset + shard.expected_bytes > descriptor.length
    ):
        raise SparsePayloadError("Entry allocation is outside its original registered shard")
    source = buffer[allocation_offset:allocation_offset + shard.expected_bytes]
    views, dtype, valid_tokens = _views(source, layout, shard, allow_cuda=True)
    if (manifest.dtype, manifest.head_dim) != (str(dtype), layout.head_dim):
        raise SparsePayloadError("sparse manifest dtype/head dimension mismatch")
    identity = (entry_transfer_id, index_version, id_mapping_version)
    if any(
        not isinstance(value, str) or not value.strip() for value in identity
    ):
        raise SparsePayloadError("current Entry/index/mapping identities required")
    layers = len(views) // 2
    rows = shard.page_count * layout.page_size
    heads = layout.kv_heads_per_rank
    head_bytes = layout.head_dim * {
        torch.float16: 2, torch.bfloat16: 2, torch.float32: 4,
    }[dtype]
    slices, offsets = [], []
    group_offset = 0
    for spec in manifest.specs:
        if (spec.entry_transfer_id, spec.index_version, spec.id_mapping_version) != identity:
            raise SparsePayloadError("selection does not match current Entry/index/mapping")
        layer = spec.layer - shard.layer_start
        head = spec.kv_head - shard.rank * heads
        if (
            spec.layout_fingerprint != layout.fingerprint
            or not 0 <= layer < layers or not 0 <= head < heads
            or any(token >= valid_tokens for token in spec.token_ids)
        ):
            raise SparsePayloadError("selection is outside the Entry layout/shard/valid rows")
        count = len(spec.token_ids)
        for kind in (0, 1):
            # All K layers precede all V layers in component-major storage.
            component = layer + kind * layers
            for row, token in enumerate(spec.token_ids):
                offset = allocation_offset + (
                    (component * rows + token) * heads + head
                ) * head_bytes
                if not allocation_offset <= offset < offset + head_bytes <= (
                    allocation_offset + shard.expected_bytes
                ):
                    raise SparsePayloadError("sparse source slice escapes its Entry")
                slices.append(MemorySlice(registration, offset, head_bytes))
                offsets.append(group_offset + (kind * count + row) * head_bytes)
        group_offset += 2 * count * head_bytes
    if (
        group_offset != manifest.nbytes
        or sum(local.length for local in slices) != manifest.nbytes
        or tuple(offsets) != tuple(range(0, manifest.nbytes, head_bytes))
    ):
        raise SparsePayloadError("sparse batch does not exactly cover its destination")
    return SparseBatchPlan(tuple(slices), tuple(offsets), manifest.nbytes)
