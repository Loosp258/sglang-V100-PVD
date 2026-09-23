"""Retained draft storage is charged once and unknown layouts are refused."""

from types import SimpleNamespace as NS

import pytest
import torch
from sglang.srt.disaggregation.pvd.draft_memory import (
    measure_draft_retained_tensors,
)
from sglang.srt.disaggregation.pvd.draft_sglang import DraftCapabilityError


def runner():
    model = torch.nn.Linear(3, 3, bias=False)
    # The buffer aliases the parameter: its storage must not be charged twice.
    model.register_buffer("tied", model.weight.view(-1))
    request_map = torch.zeros((2, 5), dtype=torch.int32)
    k = torch.zeros((4, 2), dtype=torch.float16)
    v = torch.zeros((4, 2), dtype=torch.float16)
    free_pages = torch.arange(4, dtype=torch.int64)
    return NS(
        model=model,
        req_to_token_pool=NS(req_to_token=request_map),
        token_to_kv_pool=NS(k_buffer=[k], v_buffer=[v]),
        token_to_kv_pool_allocator=NS(
            free_pages=free_pages, release_pages=torch.empty(0, dtype=torch.int64)
        ),
    )


def test_retained_storage_is_deduplicated_and_broken_down():
    draft = runner()
    result = measure_draft_retained_tensors(draft)
    assert result.weights_bytes == draft.model.weight.untyped_storage().nbytes()
    assert result.request_map_bytes == 2 * 5 * 4
    assert result.kv_bytes == 2 * 4 * 2 * 2
    assert result.allocator_bytes == 4 * 8
    assert result.total_bytes == sum(
        (
            result.weights_bytes,
            result.request_map_bytes,
            result.kv_bytes,
            result.allocator_bytes,
        )
    )
    # A KV view must not silently double charge its underlying allocation.
    draft.token_to_kv_pool.v_buffer = [draft.token_to_kv_pool.k_buffer[0].view(4, 2)]
    assert measure_draft_retained_tensors(draft).kv_bytes == 4 * 2 * 2


@pytest.mark.parametrize(
    "corrupt,reason",
    [
        (lambda d: setattr(d, "model", None), "model and both private pools"),
        (
            lambda d: setattr(d.req_to_token_pool, "req_to_token", None),
            "request-to-token map is not a tensor",
        ),
        (
            lambda d: setattr(d.token_to_kv_pool, "k_buffer", []),
            "no supported k_buffer",
        ),
        (
            lambda d: setattr(d.token_to_kv_pool_allocator, "free_pages", None),
            "free_pages is not a tensor",
        ),
    ],
)
def test_unknown_or_incomplete_layout_refused(corrupt, reason):
    draft = runner()
    corrupt(draft)
    with pytest.raises(DraftCapabilityError, match=reason):
        measure_draft_retained_tensors(draft)
