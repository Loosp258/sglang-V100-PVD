"""Index native-completion contract via CPU backend fault injection."""

import pytest
import torch
from sglang.srt.disaggregation.pvd.index_search import (
    BruteForceIndexBackend,
    IndexCompletionUnknown,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_prompt_index import ident, manager, stored_entry


class CompletionBackend(BruteForceIndexBackend):
    def __init__(self):
        super().__init__()
        self.fail_fence = self.fail_dispose = self.fail_search = False
        self.fences = 0
        self.disposed = []

    def synchronize(self):
        self.fences += 1
        if self.fail_fence:
            raise RuntimeError("native completion unknown")

    def dispose(self, index):
        self.disposed.append(id(index))
        if self.fail_dispose:
            raise RuntimeError("native disposal unknown")

    def search(self, *args, **kwargs):
        if self.fail_search:
            self.fail_fence = True
            raise RuntimeError("native search launched then failed")
        return super().search(*args, **kwargs)


def setup():
    backend = CompletionBackend()
    budget = TransferBudget(1 << 20, 32)
    index = manager(backend=backend, budget=budget)
    store, manifest, _, _ = stored_entry(index)
    return backend, budget, index, store, manifest


def test_successful_build_search_close_fence_and_dispose_before_refund():
    backend, budget, index, store, manifest = setup()
    assert store.progress_prompt_indexes()["built"] == 1
    record = index._entries[manifest.key.transfer_id]
    queries = next(iter(record.vectors.values())).vectors[0:1]
    assert backend.fences > 0
    index.search(ident(manifest.key.transfer_id), queries=queries, top_k=1)
    before = backend.fences
    expected = len(record.indexes)
    index.close(manifest.key.transfer_id)
    assert backend.fences > before and len(backend.disposed) == expected
    assert budget.snapshot()["reservations"] == 0
    assert not index.quarantined
    store.close()


def test_build_unknown_keeps_actual_entry_allocation_pin_and_all_budget():
    backend, budget, index, store, manifest = setup()
    backend.fail_fence = True
    assert store.progress_prompt_indexes()["failed"] == 1
    entry = store.entries[manifest.key]
    assert index.quarantined and index.snapshot()["retained_operations"] == 1
    assert budget.snapshot()["used_staging_bytes"] > 0
    assert len(store._quarantined_index_sources) == 1
    assert store._quarantined_index_sources[0][0] is entry
    calls = backend.fences
    # No retry even if a later sync would return success.
    backend.fail_fence = False
    assert store.progress_prompt_indexes()["built"] == 0
    assert backend.fences == calls
    store.close()
    assert not entry.resources_released
    assert store._pool_guard.value is not None


def test_failed_search_and_close_keep_queries_handle_scratch_and_record():
    backend, budget, index, store, manifest = setup()
    assert store.progress_prompt_indexes()["built"] == 1
    record = index._entries[manifest.key.transfer_id]
    query = next(iter(record.vectors.values())).vectors[0:1].clone()
    retained = budget.snapshot()["used_staging_bytes"]
    backend.fail_search = True
    with pytest.raises(IndexCompletionUnknown, match="completion unknown"):
        index.search(ident(manifest.key.transfer_id), queries=query, top_k=1)
    assert record.users == 1 and index.quarantined
    assert budget.snapshot()["used_staging_bytes"] > retained
    assert any(value is query for value in index._retained_operations[0])
    index.close(manifest.key.transfer_id)
    assert record.indexes and record.vectors
    with pytest.raises(IndexCompletionUnknown):
        index.search(ident(manifest.key.transfer_id), queries=query, top_k=1)
    store.close()  # Query uses owned index copies, not Entry's original MR.


def test_native_disposal_failure_retains_record_and_prevents_retry():
    backend, budget, index, store, manifest = setup()
    store.progress_prompt_indexes()
    record = index._entries[manifest.key.transfer_id]
    retained = budget.snapshot()["used_staging_bytes"]
    backend.fail_dispose = True
    index.close(manifest.key.transfer_id)
    assert index.quarantined and index._retained_records == [record]
    assert budget.snapshot()["used_staging_bytes"] == retained
    assert record.indexes and record.vectors
    attempts = len(backend.disposed)
    index.close(manifest.key.transfer_id)
    assert len(backend.disposed) == attempts
    store.close()


def test_partial_build_failure_disposes_completed_heads_then_refunds():
    backend, budget, index, store, manifest = setup()
    original = backend.build
    count = 0

    def fail_second(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("second head failed")
        return original(*args, **kwargs)

    backend.build = fail_second
    assert store.progress_prompt_indexes()["failed"] == 1
    assert len(backend.disposed) == 1
    assert budget.snapshot()["reservations"] == 0
    assert not index.quarantined and not store._quarantined_index_sources
    store.close()


def test_native_unknown_before_return_keeps_exception_owned_allocations():
    import gc
    import weakref

    backend, budget, index, store, manifest = setup()
    references = []

    def unknown(*args, **kwargs):
        native_buffer = torch.ones(16)
        references.append(weakref.ref(native_buffer))
        raise IndexCompletionUnknown("native build handle unknown")

    backend.build = unknown
    assert store.progress_prompt_indexes()["failed"] == 1
    gc.collect()
    assert references[0]() is not None
    assert backend.fences == 0, "a reported UNKNOWN must not be retried"
    assert budget.snapshot()["used_staging_bytes"] > 0
    assert store.snapshot()["quarantined_index_sources"] == 1
    store.close()


def test_cancelled_build_still_fences_and_refunds_but_does_not_swallow_interrupt():
    backend, budget, index, store, manifest = setup()

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("cancel build")

    backend.build = interrupt
    with pytest.raises(KeyboardInterrupt, match="cancel build"):
        store.progress_prompt_indexes()
    assert backend.fences > 0
    assert budget.snapshot()["reservations"] == 0
    assert not index.quarantined and not store._quarantined_index_sources
    store.close()


def test_each_head_build_finishes_before_reusing_single_scratch_reservation():
    backend, budget, index, store, manifest = setup()
    build = backend.build
    pending = []

    def async_build(*args, **kwargs):
        assert not pending, "two native heads overlap one scratch reservation"
        pending.append(True)
        return build(*args, **kwargs)

    def finish():
        pending.clear()

    backend.build, backend.synchronize = async_build, finish
    assert store.progress_prompt_indexes()["built"] == 1
    assert not pending
    store.close()
    assert budget.snapshot()["reservations"] == 0


def test_cuda_duck_backend_cannot_silently_skip_disposal():
    backend, budget, index, store, manifest = setup()
    assert store.progress_prompt_indexes()["built"] == 1
    retained = budget.snapshot()["used_staging_bytes"]
    # Policy fault injection: no CUDA allocation is performed by this test.
    index.backend_device = torch.device("cuda:0")
    backend.dispose = None
    index.close(manifest.key.transfer_id)
    assert index.quarantined
    assert budget.snapshot()["used_staging_bytes"] == retained
    assert index.snapshot()["retained_records"] == 1
    store.close()


def test_quarantine_before_publication_never_marks_a_build_ready():
    backend, budget, index, store, manifest = setup()
    release = index._release_owners

    def peer_fails_at_publication(owners):
        release(owners)
        if any(owner.endswith(":build-scratch") for owner in owners):
            index._quarantine("concurrent native operation became unknown")

    index._release_owners = peer_fails_at_publication
    assert store.progress_prompt_indexes()["failed"] == 1
    assert not index.gate_for(manifest.key.transfer_id).searchable
    assert index.quarantined
    assert index.snapshot()["retained_operations"] == 1
    assert budget.snapshot()["used_staging_bytes"] > 0
    store.close()
