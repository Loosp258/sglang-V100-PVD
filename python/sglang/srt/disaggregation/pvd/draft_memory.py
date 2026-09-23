"""Measure the retained tensors of a loaded, private Llama/Qwen2 draft.

This is an accounting floor, not a CUDA peak-memory limit. Model loading,
allocator cache, backend workspaces and future pool growth are not included.
The caller must keep a separate margin and measure peak memory on its device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from sglang.srt.disaggregation.pvd.draft_sglang import DraftCapabilityError


@dataclass(frozen=True)
class DraftRetainedTensors:
    weights_bytes: int
    request_map_bytes: int
    kv_bytes: int
    allocator_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.request_map_bytes
            + self.kv_bytes
            + self.allocator_bytes
        )


def measure_draft_retained_tensors(model_runner: Any) -> DraftRetainedTensors:
    """Count distinct storage allocations in the supported private-pool layout.

    Aliased parameters/buffers and views are charged once by storage identity.
    Unknown layouts fail closed instead of returning a plausible undercount.
    A value from this function can be passed to ``persistent_bytes`` only as
    the known-tensor charge, never as proof of a hard memory bound.
    """
    model = getattr(model_runner, "model", None)
    request_pool = getattr(model_runner, "req_to_token_pool", None)
    kv_pool = getattr(model_runner, "token_to_kv_pool", None)
    allocator = getattr(model_runner, "token_to_kv_pool_allocator", None)
    if any(value is None for value in (model, request_pool, kv_pool, allocator)):
        raise DraftCapabilityError("draft model and both private pools must be loaded")
    if not callable(getattr(model, "parameters", None)) or not callable(
        getattr(model, "buffers", None)
    ):
        raise DraftCapabilityError("draft model does not expose tensor weights")

    seen: set[tuple[str, int | None, int]] = set()

    def charge(value: Any, label: str) -> int:
        if not isinstance(value, torch.Tensor):
            raise DraftCapabilityError(f"{label} is not a tensor")
        try:
            storage = value.untyped_storage()
            size = storage.nbytes()
            pointer = storage.data_ptr()
        except Exception as exc:
            raise DraftCapabilityError(f"{label} storage cannot be inspected") from exc
        if size < 0 or (size and pointer == 0):
            raise DraftCapabilityError(f"{label} has invalid storage")
        if not size:
            return 0
        key = (value.device.type, value.device.index, pointer)
        if key in seen:
            return 0
        seen.add(key)
        return size

    weights = sum(charge(t, "model parameter") for t in model.parameters())
    weights += sum(charge(t, "model buffer") for t in model.buffers())
    if weights == 0:
        raise DraftCapabilityError("draft model exposes no retained tensor weights")
    request_map = charge(
        getattr(request_pool, "req_to_token", None), "request-to-token map"
    )
    if request_map == 0:
        raise DraftCapabilityError("draft request-to-token map is empty")

    kv = 0
    for name in ("k_buffer", "v_buffer"):
        values = getattr(kv_pool, name, None)
        if not isinstance(values, (list, tuple)) or not values:
            raise DraftCapabilityError(f"draft KV pool has no supported {name}")
        kv += sum(charge(t, name) for t in values)
    if kv == 0:
        raise DraftCapabilityError("draft KV buffers are empty")

    indices = sum(
        charge(getattr(allocator, name, None), name)
        for name in ("free_pages", "release_pages")
    )
    return DraftRetainedTensors(weights, request_map, kv, indices)
