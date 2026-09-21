"""Allocator wrappers must not hide shared target K/V storage."""

from types import SimpleNamespace as NS

import pytest
import torch
from sglang.srt.disaggregation.pvd.draft_sglang import (
    DraftWorkerError,
    require_private_pools,
)


def worker(k, v):
    return NS(
        get_memory_pool=lambda: (
            NS(req_to_token=torch.zeros((4, 8), dtype=torch.int32)),
            NS(_kvcache=NS(k_buffer=[k], v_buffer=[v])),
        )
    )


@pytest.mark.parametrize("shared", ["k", "v", "both"])
def test_separate_allocators_cannot_hide_shared_backing_storage(shared):
    k, v = torch.zeros(8, 2, 4), torch.zeros(8, 2, 4)
    target = worker(k, v)
    draft = worker(
        k.view_as(k) if shared != "v" else k.clone(),
        v.view_as(v) if shared != "k" else v.clone(),
    )
    with pytest.raises(DraftWorkerError, match="same memory"):
        require_private_pools(draft, target)


def test_distinct_backing_storage_is_verified_and_cycles_terminate():
    k, v = torch.zeros(8, 2, 4), torch.zeros(8, 2, 4)
    target, draft = worker(k, v), worker(k.clone(), v.clone())
    assert require_private_pools(draft, target).storage_verified
    # Malformed wrappers must not lead to infinite storage traversal.
    cycle = NS()
    cycle._kvcache = cycle
    unknown = NS(get_memory_pool=lambda: (torch.zeros(2), cycle))
    assert not require_private_pools(unknown, target).storage_verified
