"""Extract retrieval vectors from a stored EntryShard's packed Prompt KV.

Input is the real storage representation -- the byte buffer ``kv_packer``
produced, plus the validated ``KVLayoutSignature`` and ``KVShardManifest`` --
not an already-extracted tensor. Output is one vector set per (global layer,
global KV head), with a versioned row -> original prompt position mapping that
``index_search.select`` turns into logical token and page ids.

Positional encoding
-------------------
The stored K is **post-RoPE**: models apply ``rotary_emb(positions, q, k)``
before the attention layer writes k into the pool (see
``sglang/srt/models/llama.py``), so these vectors already carry rotation by
key position. Extraction therefore applies **no** positional transformation --
doing so would rotate twice -- and the probe must supply Q that is likewise
rotated, at the predicted positions. ``positional_encoding`` is an explicit
input rather than something guessed from metadata, because it is a property of
the model and backend, and ``require_compatible_query`` refuses a Q that
declares a different one.

Heads
-----
Layers and KV heads stay separate. Nothing here averages or concatenates
heads, and head identity is carried through into the selection. The
query-head -> KV-head grouping lives in ``QueryHeadMapping`` and takes the
query-head count explicitly, because ``KVLayoutSignature`` does not carry it.

Ownership and placement
-----------------------
Extracted vectors **own a copy**. The stored KV is never mutated, and the
index must not borrow storage inside the Entry's registered region: that would
tie index lifetime to the MR and put index reads in the path of its release.
The copy's bytes are charged through a budget when the caller supplies one.

The copy is also where the *device* transition happens, and it is the only
one: ``device`` names where the copy lands, defaulting to wherever the stored
shard already is. The authoritative KV pool is never moved to suit a
retrieval backend -- a host-resident backend gets a host-resident copy of the
slice it indexes, and the pool stays exactly where the transport registered
it. ``.to(dtype)`` converts a dtype and moves nothing, so the device is
always stated separately rather than inferred from a cast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import torch

from sglang.srt.disaggregation.pvd.index_search import IdMapping

#: Stored K already carries rotation by key position.
ROPE_APPLIED = "rope_applied"
#: Stored K carries no positional transformation.
NO_POSITIONAL_ENCODING = "none"

POSITIONAL_ENCODINGS = (ROPE_APPLIED, NO_POSITIONAL_ENCODING)


class PromptVectorError(ValueError):
    """A stored shard could not be interpreted as retrieval vectors."""


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromptVectorError(f"{name} must be a non-empty string")
    return value


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PromptVectorError(f"{name} must be a positive integer")
    return value


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, str(name).split(".")[-1], None)
    if not isinstance(dtype, torch.dtype):
        raise PromptVectorError(f"unsupported stored dtype {name!r}")
    return dtype


@dataclass(frozen=True)
class QueryHeadMapping:
    """Which KV head a query head reads, for supported GQA/MQA layouts.

    Only uniform grouping is supported: every KV head serves the same number
    of consecutive query heads. A non-divisible combination is rejected rather
    than rounded, because the remainder would silently mis-route heads.
    """

    num_query_heads: int
    total_kv_heads: int

    def __post_init__(self) -> None:
        _require_positive_int("num_query_heads", self.num_query_heads)
        _require_positive_int("total_kv_heads", self.total_kv_heads)
        if self.num_query_heads < self.total_kv_heads:
            raise PromptVectorError(
                "unsupported attention layout: fewer query heads than KV heads"
            )
        if self.num_query_heads % self.total_kv_heads:
            raise PromptVectorError(
                f"unsupported attention layout: {self.num_query_heads} query heads "
                f"do not divide evenly across {self.total_kv_heads} KV heads"
            )

    @property
    def group_size(self) -> int:
        return self.num_query_heads // self.total_kv_heads

    @property
    def kind(self) -> str:
        if self.group_size == 1:
            return "mha"
        return "mqa" if self.total_kv_heads == 1 else "gqa"

    def kv_head_for(self, query_head: int) -> int:
        if isinstance(query_head, bool) or not isinstance(query_head, int):
            raise PromptVectorError("query_head must be an integer")
        if not 0 <= query_head < self.num_query_heads:
            raise PromptVectorError(f"query head {query_head} is out of range")
        return query_head // self.group_size

    def query_heads_for(self, kv_head: int) -> Tuple[int, ...]:
        if isinstance(kv_head, bool) or not isinstance(kv_head, int):
            raise PromptVectorError("kv_head must be an integer")
        if not 0 <= kv_head < self.total_kv_heads:
            raise PromptVectorError(f"KV head {kv_head} is out of range")
        start = kv_head * self.group_size
        return tuple(range(start, start + self.group_size))


@dataclass(frozen=True)
class PromptKVectors:
    """One layer's, one KV head's Prompt K, ready for an index backend."""

    entry_transfer_id: str
    layer: int
    kv_head: int
    vectors: torch.Tensor
    mapping: IdMapping
    positional_encoding: str
    source_dtype: str

    @property
    def head_dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def token_count(self) -> int:
        return int(self.vectors.shape[0])

    def require_compatible_query(
        self, *, positional_encoding: str, head_dim: int
    ) -> None:
        """Refuse a Q that would be searched under different semantics."""
        if positional_encoding != self.positional_encoding:
            raise PromptVectorError(
                f"query is {positional_encoding!r} but stored K is "
                f"{self.positional_encoding!r}; retrieval would compare "
                "differently-encoded vectors"
            )
        if head_dim != self.head_dim:
            raise PromptVectorError(
                f"query head_dim {head_dim} does not match stored {self.head_dim}"
            )


def _validate_layout(
    layout: Any,
) -> Tuple[int, List[torch.dtype], List[List[int]], List[int]]:
    extra: Mapping[str, Any] = getattr(layout, "extra", None) or {}
    for key in (
        "component_count",
        "component_dtypes",
        "component_token_shapes",
        "component_bytes_per_token",
    ):
        if key not in extra:
            raise PromptVectorError(f"layout metadata is missing {key!r}")
    count = extra["component_count"]
    _require_positive_int("component_count", count)
    if count % 2:
        raise PromptVectorError(
            "expected paired K and V components; component_count is odd"
        )
    dtypes = [_torch_dtype(value) for value in extra["component_dtypes"]]
    shapes = [list(shape) for shape in extra["component_token_shapes"]]
    sizes = list(extra["component_bytes_per_token"])
    if not (len(dtypes) == len(shapes) == len(sizes) == count):
        raise PromptVectorError("layout component metadata lengths disagree")
    return count, dtypes, shapes, sizes


def extract_prompt_k(
    packed: torch.Tensor,
    *,
    layout: Any,
    manifest: Any,
    entry_transfer_id: str,
    id_mapping_version: str,
    positional_encoding: str,
    dtype: torch.dtype = torch.float32,
    device: Optional[object] = None,
    layers: Optional[Sequence[int]] = None,
    kv_heads: Optional[Sequence[int]] = None,
    budget: Any = None,
    budget_owner: Optional[str] = None,
) -> List[PromptKVectors]:
    """Turn one stored EntryShard into per-(layer, KV head) Prompt K vectors.

    K only: the V components of the packed buffer are skipped entirely.
    Padding in the final page is excluded using the manifest's
    ``last_page_valid_tokens``, so no vector is ever built from a token the
    prompt does not have.

    ``layers`` and ``kv_heads`` filter by **global** id and are validated
    against what this shard actually owns, so asking for a peer shard's head
    is an error rather than an empty result.

    ``device`` places the extracted copies. ``None`` keeps them wherever the
    stored shard is, which is what a caller that only wants to read its own
    KV wants; an index owner passes its backend's declared device so the
    vectors are born where they will be used.
    """
    _require_text("entry_transfer_id", entry_transfer_id)
    _require_text("id_mapping_version", id_mapping_version)
    if positional_encoding not in POSITIONAL_ENCODINGS:
        raise PromptVectorError(
            f"positional_encoding must be one of {POSITIONAL_ENCODINGS}; it is a "
            "property of the model and backend, not something to infer here"
        )
    if not isinstance(packed, torch.Tensor):
        raise PromptVectorError("packed KV must be a tensor")
    if packed.ndim != 1:
        raise PromptVectorError("packed KV must be a flat byte buffer")

    count, dtypes, shapes, sizes = _validate_layout(layout)
    local_layers = count // 2
    layer_start = int(getattr(manifest, "layer_start"))
    layer_end = int(getattr(manifest, "layer_end"))
    if not 0 <= layer_start < layer_end <= int(layout.num_layers):
        raise PromptVectorError("manifest layer range is outside the layout")
    if layer_end - layer_start != local_layers:
        raise PromptVectorError(
            f"manifest covers {layer_end - layer_start} layers but the buffer "
            f"holds {local_layers}"
        )

    page_size = _require_positive_int("page_size", int(layout.page_size))
    page_count = _require_positive_int("page_count", int(manifest.page_count))
    last_valid = int(manifest.last_page_valid_tokens)
    if not 0 < last_valid <= page_size:
        raise PromptVectorError("last_page_valid_tokens is outside the page")
    total_rows = page_count * page_size
    valid_tokens = (page_count - 1) * page_size + last_valid

    expected = sum(sizes) * total_rows
    byte_view = packed if packed.dtype == torch.uint8 else packed.view(torch.uint8)
    if byte_view.numel() != expected:
        raise PromptVectorError(
            f"packed KV has {byte_view.numel()} bytes, the layout requires {expected}"
        )
    declared = int(getattr(manifest, "expected_bytes", expected))
    if declared != expected:
        raise PromptVectorError(
            f"manifest declares {declared} bytes, the layout requires {expected}"
        )

    heads_per_rank = _require_positive_int(
        "kv_heads_per_rank", int(layout.kv_heads_per_rank)
    )
    head_dim = _require_positive_int("head_dim", int(layout.head_dim))
    rank = int(getattr(manifest, "rank"))
    if rank < 0:
        raise PromptVectorError("manifest rank must be non-negative")
    head_base = rank * heads_per_rank
    if head_base + heads_per_rank > int(layout.total_kv_heads):
        raise PromptVectorError(
            "this shard's KV heads fall outside the layout's total_kv_heads"
        )

    owned_layers = range(layer_start, layer_end)
    owned_heads = range(head_base, head_base + heads_per_rank)
    wanted_layers = _resolve(layers, owned_layers, "layer")
    wanted_heads = _resolve(kv_heads, owned_heads, "KV head")

    target_device = None if device is None else torch.device(device)

    if budget is not None:
        _require_text("budget_owner", budget_owner)
        element = torch.empty(0, dtype=dtype).element_size()
        budget.reserve(
            budget_owner,
            len(wanted_layers) * len(wanted_heads) * valid_tokens * head_dim * element,
            0,
        )

    mapping = IdMapping(
        version=id_mapping_version,
        token_ids=tuple(range(valid_tokens)),
        page_size=page_size,
    )

    offset = 0
    results: List[PromptKVectors] = []
    for index in range(count):
        span = sizes[index] * total_rows
        # Only the first half are K components; V is skipped, not extracted.
        if index < local_layers and (layer_start + index) in wanted_layers:
            shape = shapes[index]
            if len(shape) != 2:
                raise PromptVectorError(
                    "PVD retrieval requires token/head/dimension KV components"
                )
            if shape[0] != heads_per_rank or shape[1] != head_dim:
                raise PromptVectorError(
                    f"component {index} is {shape}, the layout declares "
                    f"[{heads_per_rank}, {head_dim}]"
                )
            slab = (
                byte_view[offset : offset + span]
                .view(dtypes[index])
                .reshape(total_rows, heads_per_rank, head_dim)
            )
            for head in wanted_heads:
                # copy=True: the index owns its bytes and never aliases
                # stored KV, so releasing the Entry's MR cannot pull data out
                # from under a live index. Device and dtype are both named,
                # because a cast alone would leave the copy on the pool's
                # device and tie a host backend to GPU memory.
                source = slab[:valid_tokens, head - head_base, :]
                vectors = source.to(
                    device=target_device or source.device, dtype=dtype, copy=True
                )
                results.append(
                    PromptKVectors(
                        entry_transfer_id=entry_transfer_id,
                        layer=layer_start + index,
                        kv_head=head,
                        vectors=vectors,
                        mapping=mapping,
                        positional_encoding=positional_encoding,
                        source_dtype=str(dtypes[index]),
                    )
                )
        offset += span
    return results


def _resolve(
    requested: Optional[Sequence[int]], owned: range, what: str
) -> Tuple[int, ...]:
    if requested is None:
        return tuple(owned)
    values = list(requested)
    if not values:
        raise PromptVectorError(f"an empty {what} selection extracts nothing")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise PromptVectorError(f"{what} ids must be integers")
        if value not in owned:
            raise PromptVectorError(
                f"{what} {value} is not held by this shard "
                f"({owned.start}..{owned.stop - 1})"
            )
    if len(set(values)) != len(values):
        raise PromptVectorError(f"duplicate {what} in the selection")
    return tuple(sorted(values))
