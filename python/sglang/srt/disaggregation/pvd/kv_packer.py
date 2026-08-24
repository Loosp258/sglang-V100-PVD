"""Deterministic full-prompt KV packing for V storage.

P and D use the same component order. Each component is page-major and contains
all tokens of every selected page, including padding in the final page. The
manifest separately records ``last_page_valid_tokens``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch


PVD_TENSOR_LAYOUT = "sglang-pvd-v1/component-page-token-major"


class UnsupportedKVPoolError(TypeError):
    pass


@dataclass(frozen=True)
class PVDPackedKV:
    tensor: torch.Tensor
    page_count: int
    page_size: int
    component_bytes_per_page: List[int]

    @property
    def expected_bytes(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()


def kv_components(kv_pool: Any) -> List[torch.Tensor]:
    """Return the canonical local-rank component list without copying."""
    if hasattr(kv_pool, "full_kv_pool"):
        kv_pool = kv_pool.full_kv_pool
    if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
        return list(kv_pool.k_buffer) + list(kv_pool.v_buffer)
    if hasattr(kv_pool, "kv_buffer"):
        value = kv_pool.kv_buffer
        return list(value) if isinstance(value, (list, tuple)) else [value]
    raise UnsupportedKVPoolError(
        f"PVD v1 cannot obtain tensor components from {type(kv_pool).__name__}"
    )


def describe_kv_layout(kv_pool: Any) -> Dict[str, Any]:
    components = kv_components(kv_pool)
    return {
        "tensor_layout": PVD_TENSOR_LAYOUT,
        "component_count": len(components),
        "component_dtypes": [str(tensor.dtype) for tensor in components],
        "component_token_shapes": [list(tensor.shape[1:]) for tensor in components],
        "component_bytes_per_token": [
            tensor[0].numel() * tensor.element_size() for tensor in components
        ],
    }


def _page_token_indices(
    page_indices: torch.Tensor, page_size: int, *, device: torch.device
) -> torch.Tensor:
    pages = page_indices.to(device=device, dtype=torch.long).reshape(-1, 1)
    offsets = torch.arange(page_size, device=device, dtype=torch.long).reshape(1, -1)
    return (pages * page_size + offsets).reshape(-1)


def pack_full_prompt_kv(
    kv_pool: Any,
    page_indices: Sequence[int] | torch.Tensor,
    *,
    page_size: int,
) -> PVDPackedKV:
    components = kv_components(kv_pool)
    if not components:
        raise UnsupportedKVPoolError("KV pool has no components")
    pages = torch.as_tensor(page_indices, dtype=torch.long)
    if pages.numel() == 0:
        raise ValueError("full prompt KV requires at least one page")

    chunks = []
    component_bytes = []
    for component in components:
        token_indices = _page_token_indices(pages, page_size, device=component.device)
        selected = component.index_select(0, token_indices).contiguous()
        byte_view = selected.view(torch.uint8).reshape(-1)
        chunks.append(byte_view)
        component_bytes.append(byte_view.numel() // pages.numel())
    packed = torch.cat(chunks).contiguous()
    return PVDPackedKV(
        tensor=packed,
        page_count=pages.numel(),
        page_size=page_size,
        component_bytes_per_page=component_bytes,
    )


def unpack_full_prompt_kv(
    packed: torch.Tensor,
    kv_pool: Any,
    page_indices: Sequence[int] | torch.Tensor,
    *,
    page_size: int,
) -> None:
    if not packed.is_contiguous():
        raise ValueError("packed PVD KV tensor must be contiguous")
    components = kv_components(kv_pool)
    pages = torch.as_tensor(page_indices, dtype=torch.long)
    if pages.numel() == 0:
        raise ValueError("destination requires at least one page")
    byte_view = packed.view(torch.uint8).reshape(-1)
    offset = 0
    for component in components:
        tokens = pages.numel() * page_size
        element_count = tokens * component[0].numel()
        length = element_count * component.element_size()
        if offset + length > byte_view.numel():
            raise ValueError("packed PVD KV is shorter than the destination layout")
        selected = byte_view[offset : offset + length].view(component.dtype)
        selected = selected.reshape(tokens, *component.shape[1:])
        token_indices = _page_token_indices(pages, page_size, device=component.device)
        component.index_copy_(0, token_indices, selected)
        offset += length
    if offset != byte_view.numel():
        raise ValueError("packed PVD KV has trailing bytes not described by layout")
