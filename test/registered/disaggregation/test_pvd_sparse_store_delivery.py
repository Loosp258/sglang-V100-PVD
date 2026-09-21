"""Real V store/index/byte copying with controlled async transport, not RDMA."""

import uuid
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.request_state import (
    DeliveryState,
    InvalidStateTransition,
)
from sglang.srt.disaggregation.pvd.sparse_delivery import SPARSE_DELIVERY_KEY
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferBudget,
    TransportState,
)
from sglang.srt.disaggregation.pvd.vector_store import EntryConflictError
from test_pvd_prompt_index import manager, stored_entry
from test_pvd_sparse_delivery import manifest_for


def ready():
    index = manager(budget=TransferBudget(1 << 20, 32))
    store, entry_manifest, pool, layout = stored_entry(index)
    budget = TransferBudget(65536, 32)
    store.transfer_engine.lifecycle_manager = SimpleNamespace(budget=budget)
    store.progress_prompt_indexes()
    manifest = manifest_for(index, entry_manifest.key, layout)
    return store, entry_manifest, pool, manifest, budget


def destination(store, manifest, **metadata):
    return store.transfer_engine.register_memory(
        torch.zeros(manifest.nbytes, dtype=torch.uint8),
        endpoint="D-sparse",
        rank=0,
        rail=store.rail,
        metadata={
            "pvd_receiver_epoch": "D-epoch",
            "pvd_generation": uuid.uuid4().hex,
            SPARSE_DELIVERY_KEY: manifest.to_dict(),
            **metadata,
        },
    )


def test_exact_sparse_bytes_reuse_real_delivery_ack_and_no_full_prompt_copy():
    store, entry, pool, manifest, budget = ready()
    engine = store.transfer_engine
    target = destination(store, manifest)
    try:
        delivery = store.reserve_delivery(entry.key, "sparse", target.descriptor)
        assert delivery.to_dict()["sparse_fingerprint"] == manifest.fingerprint
        store.start_delivery(entry.key, "sparse")
        assert delivery.state == DeliveryState.V_WRITING
        assert budget.snapshot()["used_staging_bytes"] == manifest.nbytes
        assert not target.buffer.any()
        with pytest.raises(InvalidStateTransition):
            store.ack_delivery(entry.key, "sparse")
        engine.finish(delivery.transfer_handle)
        store.poll_delivery(entry.key, "sparse")
        assert delivery.state == DeliveryState.DELIVERED
        for payload in manifest.payload_views(target.buffer):
            s = payload.spec
            expected = torch.stack(
                [
                    x[s.layer][list(s.token_ids), s.kv_head]
                    for x in (pool.k_buffer, pool.v_buffer)
                ]
            )
            torch.testing.assert_close(payload.tensor, expected, rtol=0, atol=0)
        assert (
            engine.total_put_bytes == manifest.nbytes < entry.shards[0].expected_bytes
        )
        assert budget.snapshot()["used_staging_bytes"] == 0
        store.ack_delivery(entry.key, "sparse")
        # The Entry can be used for another operation; no destructive KV read.
        next_manifest = replace(
            manifest,
            specs=tuple(
                replace(s, operation_id="again", target_tokens=8)
                for s in manifest.specs
            ),
        )
        next_target = destination(store, next_manifest)
        next_delivery = store.reserve_delivery(
            entry.key, "next", next_target.descriptor
        )
        store.start_delivery(entry.key, "next")
        engine.finish(next_delivery.transfer_handle)
        store.poll_delivery(entry.key, "next")
        store.ack_delivery(entry.key, "next")
        torch.testing.assert_close(target.buffer, next_target.buffer)
        engine.release_memory(next_target)
    finally:
        store.close()
        engine.release_memory(target)


@pytest.mark.parametrize("success", [True, False])
def test_cancel_retains_staging_entry_and_budget_until_late_terminal(success):
    store, entry, _, manifest, budget = ready()
    engine = store.transfer_engine
    target = destination(store, manifest)
    delivery = store.reserve_delivery(entry.key, "sparse", target.descriptor)
    store.start_delivery(entry.key, "sparse")
    source = engine.pending[delivery.transfer_handle.transfer_id][0].registration
    store.cancel_entry(entry.key, "request closed")
    assert not store.fence_write(delivery.authorization.identity)["fenced"]
    assert budget.snapshot()["used_staging_bytes"] == manifest.nbytes
    assert source.descriptor.region_id not in engine.released
    assert not store.entries[entry.key].resources_released
    engine.finish(delivery.transfer_handle, success)
    store.progress_transfers()
    assert store.fence_write(delivery.authorization.identity)["fenced"]
    assert store.entries[entry.key].resources_released
    assert engine.released.count(source.descriptor.region_id) == 1
    assert budget.snapshot()["used_staging_bytes"] == 0
    store.close()
    engine.release_memory(target)


@pytest.mark.parametrize(
    "mode", ["bytes", "layout", "entry", "rank_head", "legacy", "dense"]
)
def test_invalid_sparse_destination_never_gets_authorized(mode):
    store, entry, _, manifest, budget = ready()
    target = destination(store, manifest)
    try:
        desc = target.descriptor
        metadata = dict(desc.backend_metadata)
        if mode == "bytes":
            desc = replace(desc, length=desc.length + 2)
        elif mode == "legacy":
            del metadata["pvd_receiver_epoch"]
        elif mode == "dense":
            metadata["pvd_layout"] = entry.layout.to_dict()
        else:
            changes = {
                "layout": {"layout_fingerprint": "other"},
                "entry": {"entry_transfer_id": "other"},
                "rank_head": {"kv_head": 99},
            }[mode]
            metadata[SPARSE_DELIVERY_KEY] = replace(
                manifest, specs=tuple(replace(s, **changes) for s in manifest.specs)
            ).to_dict()
        desc = replace(desc, backend_metadata=metadata)
        with pytest.raises(EntryConflictError):
            store.reserve_delivery(entry.key, "invalid", desc)
        assert not store.entries[entry.key].deliveries
        assert budget.snapshot()["used_staging_bytes"] == 0
    finally:
        store.close()
        store.transfer_engine.release_memory(target)


@pytest.mark.parametrize("mode", ["stale_index", "budget"])
def test_start_failure_releases_unsubmitted_packing_and_closes_gate(mode):
    store, entry, _, manifest, budget = ready()
    target = destination(store, manifest)
    delivery = store.reserve_delivery(entry.key, "sparse", target.descriptor)
    try:
        if mode == "stale_index":
            store.prompt_index.close(entry.key.transfer_id)
        else:
            budget.reserve("other-owner", 65536, 0)
        store.start_delivery(entry.key, "sparse")
        assert delivery.state == DeliveryState.FAILED
        assert delivery.transfer_handle is None
        assert store.fence_write(delivery.authorization.identity)["fenced"]
        assert budget.snapshot()["used_staging_bytes"] == (
            65536 if mode == "budget" else 0
        )
    finally:
        budget.release("other-owner")
        store.close()
        store.transfer_engine.release_memory(target)


def test_short_success_is_not_deliverable():
    store, entry, _, manifest, budget = ready()
    target = destination(store, manifest)
    delivery = store.reserve_delivery(entry.key, "sparse", target.descriptor)
    store.start_delivery(entry.key, "sparse")
    store.transfer_engine.finish(delivery.transfer_handle)
    delivery.transfer_handle.transferred_bytes -= 1
    store.poll_delivery(entry.key, "sparse")
    assert delivery.state == DeliveryState.FAILED
    assert budget.snapshot()["used_staging_bytes"] == 0
    store.close()
    store.transfer_engine.release_memory(target)


def test_unregister_failure_keeps_budget_until_retry(monkeypatch):
    store, entry, _, manifest, budget = ready()
    engine = store.transfer_engine
    target = destination(store, manifest)
    delivery = store.reserve_delivery(entry.key, "sparse", target.descriptor)
    store.start_delivery(entry.key, "sparse")
    release = engine.release_memory

    def fail(_):
        raise RuntimeError("native unregister failed")

    monkeypatch.setattr(engine, "release_memory", fail)
    engine.finish(delivery.transfer_handle)
    store.progress_transfers()
    assert budget.snapshot()["used_staging_bytes"] == manifest.nbytes
    monkeypatch.setattr(engine, "release_memory", release)
    store.progress_transfers()
    assert budget.snapshot()["used_staging_bytes"] == 0
    store.ack_delivery(entry.key, "sparse")
    store.close()
    engine.release_memory(target)


def test_registration_exception_quarantines_instead_of_freeing_unknown_memory(
    monkeypatch,
):
    store, entry, _, manifest, budget = ready()
    engine = store.transfer_engine
    target = destination(store, manifest)
    delivery = store.reserve_delivery(entry.key, "sparse", target.descriptor)

    def fail(*a, **kw):
        raise RuntimeError("registration outcome unknown")

    monkeypatch.setattr(engine, "register_memory", fail)
    store.start_delivery(entry.key, "sparse")
    assert delivery.local_terminal == TransportState.UNKNOWN
    assert store._isolated_reason
    assert not store.fence_write(delivery.authorization.identity)["fenced"]
    assert budget.snapshot()["used_staging_bytes"] == manifest.nbytes
    store.close()
    assert not store.entries[entry.key].resources_released
    # Intentionally quarantined until process teardown; do not fake a fence.
    engine.release_memory(
        target
    )  # target was never published to this failed registration
