"""Offline sparse K/V payload contract. No serving, RDMA, or installation.

One payload contains one layer/KV-head selection. It never merges heads or
layers or chooses an attention policy. A future caller must pin the source
Entry and index for the entire scope; version strings alone are not a lease.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from sglang.srt.disaggregation.pvd.kv_packer import PVD_TENSOR_LAYOUT
from sglang.srt.disaggregation.pvd.protocol import KVLayoutSignature, KVShardManifest
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


class SparsePayloadError(ValueError):
    pass


@dataclass(frozen=True)
class SparseKVSpec:
    """Selection identity, not permission to write a Decode destination."""

    request_id: str
    incarnation: str
    operation_id: str
    target_tokens: int
    entry_transfer_id: str
    index_version: str
    id_mapping_version: str
    layout_fingerprint: str
    layer: int
    kv_head: int
    token_ids: tuple[int, ...]

    def __post_init__(self):
        for field in (
            "request_id",
            "incarnation",
            "operation_id",
            "entry_transfer_id",
            "index_version",
            "id_mapping_version",
            "layout_fingerprint",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise SparsePayloadError(f"{field} must be explicit non-empty text")
        for field in ("target_tokens", "layer", "kv_head"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise SparsePayloadError(f"{field} must be a non-negative integer")
        if (
            not isinstance(self.token_ids, tuple)
            or not self.token_ids
            or any(type(t) is not int or t < 0 for t in self.token_ids)
            or len(set(self.token_ids)) != len(self.token_ids)
        ):
            raise SparsePayloadError(
                "token ids must be nonempty, unique non-negative integers"
            )


class SparseKVPayload:
    """Scoped owning [K/V, selected-token, head-dim] tensor, source dtype.

    Positions stay absolute Prompt token ids. Consumers must finish using all
    tensor views inside the owning scope; arbitrary retained aliases cannot be
    revoked by Python. Do not hand this buffer to an async transport without
    adding the existing ResourceGuard/authorization/fence lifecycle first.
    """

    def __init__(self, spec, tensor):
        self.spec = spec
        self._tensor = tensor

    @property
    def tensor(self):
        if self._tensor is None:
            raise SparsePayloadError("sparse payload scope has ended")
        return self._tensor

    @property
    def nbytes(self):
        return self.tensor.numel() * self.tensor.element_size()

    def close(self):
        self._tensor = None


def _views(packed, layout, shard):
    if not isinstance(layout, KVLayoutSignature) or not isinstance(
        shard, KVShardManifest
    ):
        raise SparsePayloadError("explicit storage layout and shard manifest required")
    if (
        not isinstance(packed, torch.Tensor)
        or packed.device.type != "cpu"
        or packed.dtype != torch.uint8
        or packed.ndim != 1
        or not packed.is_contiguous()
    ):
        raise SparsePayloadError(
            "offline sparse packer needs a flat contiguous CPU byte buffer"
        )
    if layout.tensor_layout != PVD_TENSOR_LAYOUT:
        raise SparsePayloadError("unsupported storage tensor layout")
    extra = layout.extra
    layers = shard.layer_end - shard.layer_start
    if not 0 <= shard.layer_start < shard.layer_end <= layout.num_layers:
        raise SparsePayloadError("invalid source layer range")
    count = extra.get("component_count")
    dtypes = extra.get("component_dtypes", [])
    shapes = extra.get("component_token_shapes", [])
    sizes = extra.get("component_bytes_per_token", [])
    if (
        type(count) is not int
        or count != 2 * layers
        or not len(dtypes) == len(shapes) == len(sizes) == count
    ):
        raise SparsePayloadError("source must have paired per-layer K/V components")
    dtype = getattr(torch, layout.kv_dtype.split(".")[-1], None)
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise SparsePayloadError("unsupported source KV dtype")
    if (
        layout.page_size <= 0
        or shard.page_count <= 0
        or not 0 < shard.last_page_valid_tokens <= layout.page_size
        or layout.kv_heads_per_rank <= 0
        or layout.head_dim <= 0
        or shard.rank < 0
        or (shard.rank + 1) * layout.kv_heads_per_rank > layout.total_kv_heads
    ):
        raise SparsePayloadError("invalid source pages or head ownership")
    element_size = torch.empty(0, dtype=dtype).element_size()
    per_token = layout.kv_heads_per_rank * layout.head_dim * element_size
    expected_shape = [layout.kv_heads_per_rank, layout.head_dim]
    if any(
        str(dtype) != d
        or list(s) != expected_shape
        or type(n) is not int
        or n != per_token
        for d, s, n in zip(dtypes, shapes, sizes, strict=True)
    ):
        raise SparsePayloadError("source component dtype/shape/byte metadata disagree")
    rows = shard.page_count * layout.page_size
    span = rows * per_token
    if packed.numel() != count * span or shard.expected_bytes != count * span:
        raise SparsePayloadError("source byte count disagrees with manifest/layout")
    views = tuple(
        packed[i * span : (i + 1) * span].view(dtype).reshape(rows, *expected_shape)
        for i in range(count)
    )
    valid_tokens = (
        shard.page_count - 1
    ) * layout.page_size + shard.last_page_valid_tokens
    return views, dtype, valid_tokens


@contextmanager
def pack_sparse_kv(
    packed,
    *,
    layout: KVLayoutSignature,
    shard: KVShardManifest,
    spec: SparseKVSpec,
    entry_transfer_id: str,
    index_version: str,
    id_mapping_version: str,
    budget: TransferBudget,
):
    """Validate against independently supplied current source identity, then copy.

    Reserve output bytes before allocating it; copying one row at a time creates
    no full-Prompt K/V copies or gather tensor. CPU reference, not a GPU kernel.
    The source must remain pinned/immutable until this context exits.
    """
    if not isinstance(spec, SparseKVSpec) or not isinstance(budget, TransferBudget):
        raise SparsePayloadError("explicit selection spec and budget required")
    if not isinstance(layout, KVLayoutSignature):
        raise SparsePayloadError("explicit storage layout required")
    if (spec.entry_transfer_id, spec.index_version, spec.id_mapping_version) != (
        entry_transfer_id,
        index_version,
        id_mapping_version,
    ):
        raise SparsePayloadError("selection does not match current Entry/index/mapping")
    if spec.layout_fingerprint != layout.fingerprint:
        raise SparsePayloadError("selection layout fingerprint mismatch")
    views, dtype, valid_tokens = _views(packed, layout, shard)
    local_layer = spec.layer - shard.layer_start
    local_head = spec.kv_head - shard.rank * layout.kv_heads_per_rank
    if (
        not 0 <= local_layer < len(views) // 2
        or not 0 <= local_head < layout.kv_heads_per_rank
    ):
        raise SparsePayloadError("selected layer/head is not owned by this shard")
    if any(token >= valid_tokens for token in spec.token_ids):
        raise SparsePayloadError("selected token is outside Prompt or in padding")
    shape = (2, len(spec.token_ids), layout.head_dim)
    count = (
        2
        * len(spec.token_ids)
        * layout.head_dim
        * torch.empty(0, dtype=dtype).element_size()
    )
    owner = f"pvd-sparse-payload:{uuid.uuid4().hex}"
    budget.reserve(owner, count, 1)
    payload = None
    output = None
    try:
        output = torch.empty(shape, dtype=dtype)
        sources = (views[local_layer], views[local_layer + len(views) // 2])
        for kind, source in enumerate(sources):
            for row, token in enumerate(spec.token_ids):
                output[kind, row].copy_(source[token, local_head])
        payload = SparseKVPayload(spec, output)
        yield payload
    finally:
        if payload is not None:
            payload.close()
        output = None
        budget.release(owner)
