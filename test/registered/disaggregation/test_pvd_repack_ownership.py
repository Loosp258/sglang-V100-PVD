"""Heterogeneous full-Prompt repacking must obey the same native ownership rules."""

from types import SimpleNamespace

import torch
from sglang.srt.disaggregation.pvd.request_state import DeliveryState
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferBudget,
    TransportState,
)
from test_pvd_core import ENTRY_BYTES, make_manifest
from test_pvd_vector_lifecycle import ready_store, reserve


def ready():
    engine, store, entry = ready_store()
    budget = TransferBudget(1024, 32)
    engine.lifecycle_manager = SimpleNamespace(budget=budget)
    return engine, store, entry, budget


def test_repack_allocation_failure_refunds_without_submit(monkeypatch):
    engine, store, entry, budget = ready()
    delivery, target = reserve(engine, store, entry, heterogeneous=True)

    def fail(*_, **__):
        raise RuntimeError("packing allocation failed")

    # Old implementation allocates via cat; fixed implementation preallocates.
    with monkeypatch.context() as patch:
        patch.setattr(torch, "cat", fail)
        patch.setattr(torch, "empty", fail)
        store.start_delivery(entry.key, delivery.delivery_id)
    assert delivery.state == DeliveryState.FAILED
    assert delivery.transfer_handle is None
    assert store.fence_write(delivery.authorization.identity)["fenced"]
    assert budget.snapshot()["used_staging_bytes"] == 0
    store.close()
    engine.release_memory(target)


def test_repack_registration_outcome_unknown_keeps_tensor_and_budget(monkeypatch):
    engine, store, entry, budget = ready()
    delivery, target = reserve(engine, store, entry, heterogeneous=True)

    def fail(*_, **__):
        raise RuntimeError("registration may have reached native")

    monkeypatch.setattr(engine, "register_memory", fail)
    store.start_delivery(entry.key, delivery.delivery_id)
    assert delivery.local_terminal == TransportState.UNKNOWN
    assert delivery.staging_guard.value is not None
    assert store.snapshot()["isolated_reason"]
    assert budget.snapshot()["used_staging_bytes"] > 0
    assert not store.fence_write(delivery.authorization.identity)["fenced"]
    store.close()
    assert not entry.resources_released
    engine.release_memory(target)  # controlled registration stub never submitted


def test_same_delivery_id_on_different_entries_has_distinct_budget_owners():
    engine, store, first, budget = ready()
    manifest = make_manifest("other-request")
    second = store.create_entry(manifest)
    store.begin_p_write(second.key)
    offset = second.allocation.start_page * store.page_bytes
    store.pool[offset : offset + ENTRY_BYTES] = torch.arange(
        ENTRY_BYTES, dtype=torch.uint8
    )
    store.commit_p_write(second.key, ENTRY_BYTES)
    one, target1 = reserve(engine, store, first, "same-id", heterogeneous=True)
    two, target2 = reserve(engine, store, second, "same-id", heterogeneous=True)
    store.start_delivery(first.key, "same-id")
    store.start_delivery(second.key, "same-id")
    assert budget.snapshot()["reservations"] == 2
    engine.finish(one.transfer_handle)
    store.progress_transfers()
    assert budget.snapshot()["used_staging_bytes"] > 0
    assert budget.snapshot()["reservations"] == 1
    engine.finish(two.transfer_handle)
    store.progress_transfers()
    assert budget.snapshot()["used_staging_bytes"] == 0
    expected = (
        torch.arange(ENTRY_BYTES, dtype=torch.uint8)
        .reshape(-1, 4)[:, :2]
        .contiguous()
        .flatten()
    )
    torch.testing.assert_close(target1.buffer, expected)
    torch.testing.assert_close(target2.buffer, expected)
    store.close()
    engine.release_memory(target1)
    engine.release_memory(target2)


def test_repack_copy_failure_is_owned_then_refunded_before_submit(monkeypatch):
    engine, store, entry, budget = ready()
    delivery, target = reserve(engine, store, entry, heterogeneous=True)

    def fail(*_, **__):
        assert delivery.staging_guard.value is not None
        assert budget.snapshot()["used_staging_bytes"] == target.descriptor.length
        raise RuntimeError("packing copy failed")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "copy_", fail)
        store.start_delivery(entry.key, delivery.delivery_id)
    assert delivery.state == DeliveryState.FAILED
    assert not engine.pending
    assert delivery.staging_guard.value is None
    assert budget.snapshot()["used_staging_bytes"] == 0
    assert store.fence_write(delivery.authorization.identity)["fenced"]
    store.close()
    engine.release_memory(target)


def test_single_final_allocation_needs_no_concatenation_peak(monkeypatch):
    engine, store, entry, _ = ready()
    delivery, target = reserve(engine, store, entry, heterogeneous=True)
    budget = TransferBudget(target.descriptor.length, 1)
    engine.lifecycle_manager = SimpleNamespace(budget=budget)

    def fail(*_, **__):
        raise AssertionError("no materialized components or concatenation allowed")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "cat", fail)
        patch.setattr(torch.Tensor, "contiguous", fail)
        store.start_delivery(entry.key, delivery.delivery_id)
    assert delivery.state == DeliveryState.V_WRITING
    assert budget.snapshot()["used_staging_bytes"] == target.descriptor.length
    engine.finish(delivery.transfer_handle)
    store.progress_transfers()
    assert delivery.state == DeliveryState.DELIVERED
    assert budget.snapshot()["used_staging_bytes"] == 0
    expected = (
        torch.arange(ENTRY_BYTES, dtype=torch.uint8)
        .reshape(-1, 4)[:, :2]
        .contiguous()
        .flatten()
    )
    torch.testing.assert_close(target.buffer, expected)
    store.close()
    engine.release_memory(target)
