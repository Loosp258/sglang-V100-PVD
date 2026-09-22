"""Observe real CPU tensor storage at refund, not just the final budget total."""

import pytest
import torch
from sglang.srt.disaggregation.pvd.index_search import BruteForceIndexBackend
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_prompt_index import build_entry, ident, manager
from torch.multiprocessing.reductions import StorageWeakRef


@pytest.mark.parametrize(
    "scenario", ["close", "late_build", "failed_build", "search", "failed_search"]
)
def test_retained_storage_is_gone_before_its_budget_is_refunded(scenario):
    watched = []
    refunds = []

    class Budget(TransferBudget):
        def release(self, owner):
            if owner.endswith((":vectors", ":index")):
                refunds.append(sum(not ref.expired() for ref in watched))
            super().release(owner)

    class Backend(BruteForceIndexBackend):
        calls = 0

        def build(self, vectors, **kwargs):
            self.calls += 1
            watched.append(StorageWeakRef(vectors.untyped_storage()))
            built = super().build(vectors, **kwargs)
            watched.append(StorageWeakRef(built.handle.untyped_storage()))
            if scenario == "failed_build" and self.calls == 2:
                raise RuntimeError("partial build failed after allocating")
            if scenario == "late_build" and self.calls == 1:
                index.close(transfer_id)
            return built

        def search(self, *args, **kwargs):
            result = super().search(*args, **kwargs)
            index.close(transfer_id)
            if scenario == "failed_search":
                raise RuntimeError("search failed after close")
            return result

    budget = Budget(staging_bytes=1 << 20, max_inflight=4)
    index = manager(backend=Backend(), budget=budget)
    _, layout, manifest, packed, shard = build_entry()
    transfer_id = manifest.key.transfer_id
    index.note_kv_readable(transfer_id)
    success = index.build(transfer_id, packed.tensor, layout=layout, manifest=shard)
    assert success is (scenario in ("close", "search", "failed_search"))
    if scenario == "close":
        index.close(transfer_id)
    elif scenario == "search":
        index.search(ident(transfer_id), queries=torch.zeros(1, 8), top_k=1)
    elif scenario == "failed_search":
        with pytest.raises(RuntimeError, match="search failed after close"):
            index.search(ident(transfer_id), queries=torch.zeros(1, 8), top_k=1)
    assert refunds and refunds == [0] * len(refunds)
    assert budget.snapshot()["used_staging_bytes"] == 0


def test_failed_search_traceback_does_not_pin_refunded_scratch():
    watched = []
    refunds = []

    class Budget(TransferBudget):
        def release(self, owner):
            if owner.startswith("prompt-index-search:"):
                refunds.append([ref.expired() for ref in watched])
            super().release(owner)

    class Backend(BruteForceIndexBackend):
        def search(self, *args, **kwargs):
            scratch = torch.zeros(32)
            watched.append(StorageWeakRef(scratch.untyped_storage()))
            raise RuntimeError("search failed with local scratch")

    budget = Budget(staging_bytes=1 << 20, max_inflight=4)
    index = manager(backend=Backend(), budget=budget)
    _, layout, manifest, packed, shard = build_entry()
    transfer_id = manifest.key.transfer_id
    index.note_kv_readable(transfer_id)
    assert index.build(transfer_id, packed.tensor, layout=layout, manifest=shard)
    retained = budget.snapshot()["used_staging_bytes"]
    with pytest.raises(RuntimeError, match="local scratch") as error:
        index.search(ident(transfer_id), queries=torch.zeros(1, 8), top_k=1)
    assert refunds == [[True]]
    assert error.value.__traceback__ is not None  # keep locations for diagnosis
    assert budget.snapshot()["used_staging_bytes"] == retained
    index.close(transfer_id)
    assert budget.snapshot()["used_staging_bytes"] == 0
