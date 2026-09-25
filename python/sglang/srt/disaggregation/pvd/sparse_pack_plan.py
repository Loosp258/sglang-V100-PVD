"""CPU-only byte plan for the optional V sparse gather/pack kernel."""

from dataclasses import dataclass

from sglang.srt.disaggregation.pvd.protocol import KVLayoutSignature, KVShardManifest
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError


class SparsePackCompletionUnknown(RuntimeError):
    """A GPU metadata upload may still own memory after local failure."""


@dataclass(frozen=True)
class SparsePackPlan:
    token_ids: tuple[int, ...]
    # local layer, local KV head, token start, token count, destination byte offset
    groups: tuple[tuple[int, int, int, int, int], ...]
    max_group_bytes: int
    metadata_bytes: int
    source_bytes: int
    destination_bytes: int


def build_sparse_pack_plan(manifest, layout, shard) -> SparsePackPlan:
    """Prove source ownership and exact destination coverage before GPU metadata allocation."""
    if (
        not isinstance(manifest, SparseDeliveryManifest)
        or not isinstance(layout, KVLayoutSignature)
        or not isinstance(shard, KVShardManifest)
    ):
        raise SparsePayloadError("explicit sparse manifest and storage shard required")
    element_bytes = {
        "torch.float16": 2,
        "torch.bfloat16": 2,
        "torch.float32": 4,
    }[manifest.dtype]
    rows = shard.page_count * layout.page_size
    valid_tokens = (
        shard.page_count - 1
    ) * layout.page_size + shard.last_page_valid_tokens
    layers = shard.layer_end - shard.layer_start
    heads = layout.kv_heads_per_rank
    head_start = shard.rank * heads
    head_bytes = layout.head_dim * element_bytes
    if (
        rows <= 0
        or layers <= 0
        or heads <= 0
        or not 0 < shard.last_page_valid_tokens <= layout.page_size
        or manifest.head_dim != layout.head_dim
        or manifest.dtype != f"torch.{layout.kv_dtype.split('.')[-1]}"
        or shard.expected_bytes != 2 * layers * rows * heads * head_bytes
    ):
        raise SparsePayloadError("sparse pack shard/layout dimensions are invalid")
    token_ids = []
    groups = []
    output_offset = 0
    max_group_bytes = 0
    for spec in manifest.specs:
        layer = spec.layer - shard.layer_start
        head = spec.kv_head - head_start
        if (
            not 0 <= layer < layers
            or not 0 <= head < heads
            or any(token >= valid_tokens for token in spec.token_ids)
            or spec.layout_fingerprint != layout.fingerprint
        ):
            raise SparsePayloadError(
                "sparse pack selection is outside this source shard"
            )
        groups.append((layer, head, len(token_ids), len(spec.token_ids), output_offset))
        token_ids.extend(spec.token_ids)
        group_bytes = 2 * len(spec.token_ids) * head_bytes
        max_group_bytes = max(max_group_bytes, group_bytes)
        output_offset += group_bytes
    if output_offset != manifest.nbytes:
        raise SparsePayloadError("sparse pack destination extent differs from manifest")
    return SparsePackPlan(
        tuple(token_ids),
        tuple(groups),
        max_group_bytes,
        8 * (len(token_ids) + 5 * len(groups)),
        2 * layers * rows * heads * head_bytes,
        output_offset,
    )
