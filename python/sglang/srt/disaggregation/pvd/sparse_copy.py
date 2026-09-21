"""Allocation-free paired sparse copies into an already-owned destination.

CPU completion is synchronous. With explicit CUDA opt-in this enqueues copies
on the caller's current stream; returning is NOT a GPU or RDMA completion fence.
The caller must hold BOTH buffers and source/index leases, and synchronize (or
retain event-protected ownership) on success AND exceptions before retiring
them. No registration, allocation, device move, stream switching or release
takes place here. Production CUDA serving is not enabled by this primitive.
"""

import torch
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError, _views


def copy_sparse_kv_into(
    packed,
    destination,
    *,
    manifest,
    layout,
    shard,
    entry_transfer_id,
    index_version,
    id_mapping_version,
    allow_cuda=False,
):
    """Validate every group BEFORE writing any bytes, then copy K and V rows.

    Inputs remain immutable for the call/queued GPU work. Runtime copy failures
    may leave a partially written destination; callers must discard that result,
    never mark it delivered, and drain already queued work before memory reuse.
    Even non-overlapping slices of the same allocation are refused, because the
    authoritative Entry must never double as output staging.
    """
    if not isinstance(manifest, SparseDeliveryManifest) or type(allow_cuda) is not bool:
        raise SparsePayloadError("explicit manifest and boolean CUDA opt-in required")
    views, dtype, valid_tokens = _views(packed, layout, shard, allow_cuda=allow_cuda)
    if (
        not isinstance(destination, torch.Tensor)
        or destination.device != packed.device
        or destination.dtype != torch.uint8
        or destination.ndim != 1
        or not destination.is_contiguous()
        or destination.numel() != manifest.nbytes
    ):
        raise SparsePayloadError(
            "destination must have the exact byte extent on the source device"
        )
    if packed.untyped_storage().data_ptr() == destination.untyped_storage().data_ptr():
        raise SparsePayloadError("source and destination must own distinct storage")
    if manifest.dtype != str(dtype) or manifest.head_dim != layout.head_dim:
        raise SparsePayloadError("manifest dtype/dimension differs from the stored KV")
    first = manifest.specs[0]
    if (
        first.entry_transfer_id,
        first.index_version,
        first.id_mapping_version,
    ) != (entry_transfer_id, index_version, id_mapping_version):
        raise SparsePayloadError("selection does not match current Entry/index/mapping")
    if first.layout_fingerprint != layout.fingerprint:
        raise SparsePayloadError("selection layout fingerprint mismatch")
    source_groups = []
    for spec in manifest.specs:
        layer = spec.layer - shard.layer_start
        head = spec.kv_head - shard.rank * layout.kv_heads_per_rank
        if not 0 <= layer < len(views) // 2 or not 0 <= head < layout.kv_heads_per_rank:
            raise SparsePayloadError("selected layer/head is not owned by this shard")
        if any(token >= valid_tokens for token in spec.token_ids):
            raise SparsePayloadError("selected token is outside Prompt or in padding")
        source_groups.append((views[layer], views[layer + len(views) // 2], head))
    # Also construct ALL destination views before the first copy: byte alignment
    # errors in a later view must not expose a partly updated valid-looking bank.
    try:
        outputs = manifest.payload_views(destination)
    except (ValueError, RuntimeError) as exc:
        raise SparsePayloadError(
            "destination cannot form aligned typed KV views"
        ) from exc
    try:
        with torch.no_grad():
            for payload, (keys, values, head) in zip(
                outputs, source_groups, strict=True
            ):
                for kind, source in enumerate((keys, values)):
                    for row, token in enumerate(payload.spec.token_ids):
                        payload.tensor[kind, row].copy_(source[token, head])
    finally:
        for payload in outputs:
            payload.close()  # drop local views, NEVER retire caller's allocation
