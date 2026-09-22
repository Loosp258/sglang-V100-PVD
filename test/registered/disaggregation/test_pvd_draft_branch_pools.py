"""Branch-local request handles over shared private pool storage."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch
from sglang.srt.disaggregation.pvd.draft_forward_adapter import PrivatePoolAllocator
from sglang.srt.disaggregation.pvd.draft_sglang import (
    DraftCapabilityError,
    DraftPlacement,
)
from test_pvd_draft_forward import (
    REAL_POOLS,
    RealShapedKVAllocator,
    RealShapedReqPool,
    needs_real_pools,
)
from test_pvd_draft_sglang import FakeAllocator, factory, prefix, provider


@pytest.mark.parametrize("real", [False, pytest.param(True, marks=needs_real_pools)])
def test_two_live_handles_use_distinct_private_request_owners(real):
    if real:
        request_type, kv_type = REAL_POOLS
        requests = request_type(
            size=4, max_context_len=32, device="cpu", enable_memory_saver=False
        )
        kv = kv_type.__new__(kv_type)
        kv.size, kv.page_size, kv.device, kv.need_sort = 64, 1, "cpu", False
        kv.clear()
    else:
        requests, kv = RealShapedReqPool(), RealShapedKVAllocator()
    prototype = PrivatePoolAllocator(requests, kv)
    fac = factory(allocator=prototype)
    first = fac.open(branch_id="a", prefix_tokens=3, max_tokens=2)
    second = fac.open(branch_id="b", prefix_tokens=3, max_tokens=2)
    try:
        first.prepare_prefix((1, 2, 3))
        prepared = second.prepare_prefix((1, 4, 5))
        assert first.request_index != second.request_index
        assert set(first.owned_kv).isdisjoint(second.owned_kv)
        assert prototype._request.req_pool_idx is None
        slot = second.request_index
        mapping = requests.req_to_token[slot].clone()
        first.release()
        torch.testing.assert_close(requests.req_to_token[slot], mapping)
        assert slot not in requests.free_slots
        assert len(second.generate(prepared, 2)) == 2
    finally:
        first.release()
        second.release()
    assert len(requests.free_slots) == 4 and len(kv.free_pages) == 64


def test_cleanup_is_serialized_with_execution_not_only_with_other_cleanup():
    held = []

    class CheckingAllocator(FakeAllocator):
        def free_kv(self, locations):
            held.append(made._execution_lock.locked())
            super().free_kv(locations)

    made = provider(factory(allocator=CheckingAllocator()))
    with made.branch():
        made.predict(prefix(), 2)
    assert held == [True]


def test_factory_refuses_allocator_without_branch_ownership_contract():
    allocator = FakeAllocator()
    allocator.fork_for_branch = None
    with pytest.raises(DraftCapabilityError, match="branch-local"):
        factory(allocator=allocator)
    assert not allocator.live_requests and not allocator.live_kv


def test_two_admitted_branches_retain_independent_maps_until_each_exits():
    requests, kv = RealShapedReqPool(), RealShapedKVAllocator()
    made = provider(
        factory(allocator=PrivatePoolAllocator(requests, kv)),
        placement=DraftPlacement(
            scratch_budget_bytes=1 << 20,
            persistent_budget_bytes=1024,
            max_concurrent_branches=2,
        ),
    )
    ready, finish = Event(), Event()

    def second():
        try:
            with made.branch():
                made.predict(prefix("b"), 2)
                slot = made._require_branch().handle.request_index
                mapping = requests.req_to_token[slot].clone()
                ready.set()
                assert finish.wait(5)
                torch.testing.assert_close(requests.req_to_token[slot], mapping)
                assert slot not in requests.free_slots
        finally:
            ready.set()  # don't hide a worker failure behind a timeout

    with ThreadPoolExecutor(1) as threads:
        try:
            with made.branch():
                made.predict(prefix("a"), 2)
                future = threads.submit(second)
                assert ready.wait(5)
                assert made.active_branches == 2
                assert len(requests.free_slots) == 2
            assert made.active_branches == 1
        finally:
            finish.set()
        future.result(timeout=5)
    assert made.active_branches == 0 and not made.quarantined
    assert len(requests.free_slots) == 4 and len(kv.free_pages) == 64
    assert made.scratch_budget.snapshot()["used_staging_bytes"] == 0
