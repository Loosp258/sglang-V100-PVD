"""Direct Entry scatter lifecycle with real CPU store and controlled transport."""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.request_state import DeliveryState
from sglang.srt.disaggregation.pvd.server import _validate_args, build_parser
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine, TransferHandle, TransferStatus
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget, TransportState
from sglang.srt.disaggregation.pvd.vector_store import EntryConflictError, VectorKVStore
from test_pvd_prompt_index import build_entry, manager
from test_pvd_sparse_delivery import manifest_for
from test_pvd_sparse_store_delivery import destination


class BatchEngine(FakeTransferEngine):
    def __init__(self):
        super().__init__()
        self.lifecycle_manager = SimpleNamespace(budget=TransferBudget(65536, 32))
        self.batch_calls, self.pending, self.released = [], {}, []

    def submit_batch_put(self, slices, remote, *, remote_offsets):
        handle = TransferHandle(uuid.uuid4().hex, transport_state=TransportState.IN_FLIGHT)
        self.batch_calls.append((slices, remote, remote_offsets))
        self.pending[handle.transfer_id] = (slices, remote, remote_offsets)
        return handle

    def finish(self, handle, *, state=TransportState.TERMINAL_SUCCESS, short=False):
        if state == TransportState.UNKNOWN:
            handle.status = TransferStatus.FAILED
            handle.transport_state = state
            handle.error = "aggregate completion uncertain"
            return
        slices, remote, offsets = self.pending.pop(handle.transfer_id)
        if state == TransportState.TERMINAL_SUCCESS:
            for local, offset in zip(slices, offsets, strict=True):
                result = FakeTransferEngine.submit_put(self, local, remote, remote_offset=offset)
                assert result.status == TransferStatus.SUCCESS
                handle.transferred_bytes += result.transferred_bytes
            handle.transferred_bytes -= int(short)
            handle.status = TransferStatus.SUCCESS
        else:
            handle.status = TransferStatus.FAILED
        handle.transport_state = state

    def abort(self, handle):
        # Cancellation requests draining; it provides no terminal proof.
        if not handle.transport_state.is_locally_safe_to_release:
            handle.transport_state = TransportState.DRAINING

    def release_memory(self, registration):
        self.released.append(registration.descriptor.region_id)
        super().release_memory(registration)


def ready():
    engine, index = BatchEngine(), manager(budget=TransferBudget(1 << 20, 32))
    pool, layout, entry_manifest, packed, shard = build_entry(prompt_tokens=10)
    store = VectorKVStore(
        rank=0, world_size=2, rail=shard.rail, device="cpu", total_pages=32,
        page_bytes=shard.expected_bytes // shard.page_count, endpoint="V",
        transfer_engine=engine, allow_cpu_for_tests=True, prompt_index=index,
        direct_sparse_batch_put=True,
    )
    # Occupy the first three pages to prove direct slices use allocation_base.
    blocker = store.create_entry(replace(entry_manifest, key=type(entry_manifest.key).new("model-instance", "blocker")))
    entry = store.create_entry(entry_manifest)
    store.begin_p_write(entry.key)
    offset = entry.allocation.start_page * store.page_bytes
    store.pool[offset:offset + shard.expected_bytes] = packed.tensor
    store.commit_p_write(entry.key, shard.expected_bytes)
    store.progress_prompt_indexes()
    manifest = manifest_for(index, entry.key, layout)
    target = destination(store, manifest)
    return engine, store, entry, blocker, pool, manifest, target


def finish_and_close(engine, store, target):
    # Only controlled local tests may explicitly resolve their fake operations.
    for entry in list(store.entries.values()):
        store.cancel_entry(entry.key, "CPU test cleanup")
        for delivery in entry.deliveries.values():
            handle = delivery.transfer_handle
            if handle and handle.transfer_id in engine.pending:
                engine.finish(handle)
    store.progress_transfers()
    store.close()
    engine.release_memory(target)


def test_direct_sparse_exact_bytes_use_original_mr_without_staging(monkeypatch):
    engine, store, entry, _, pool, manifest, target = ready()
    registrations_before = len(engine._regions)
    monkeypatch.setattr(store, "_prepare_sparse_source", lambda *a: pytest.fail("silent pack fallback"))
    delivery = store.reserve_delivery(entry.key, "direct", target.descriptor)
    store.start_delivery(entry.key, "direct")
    assert delivery.state == DeliveryState.V_WRITING
    assert len(engine._regions) == registrations_before
    assert all(local.registration is store.registration for local in engine.batch_calls[0][0])
    assert delivery.packing_index_lease is not None
    assert engine.lifecycle_manager.budget.snapshot()["used_staging_bytes"] == 0
    assert store.start_delivery(entry.key, "direct") is delivery
    assert len(engine.batch_calls) == 1
    engine.finish(delivery.transfer_handle)
    store.poll_delivery(entry.key, "direct")
    assert delivery.state == DeliveryState.DELIVERED
    assert delivery.packing_index_lease is None
    assert delivery.staging_guard.value is None
    for payload in manifest.payload_views(target.buffer):
        spec = payload.spec
        expected = torch.stack([values[spec.layer][list(spec.token_ids), spec.kv_head] for values in (pool.k_buffer, pool.v_buffer)])
        assert torch.equal(payload.tensor, expected)
    assert delivery.transfer_handle.transferred_bytes == manifest.nbytes
    store.ack_delivery(entry.key, "direct")
    assert store.registration.descriptor.region_id not in engine.released
    finish_and_close(engine, store, target)


@pytest.mark.parametrize("terminal", [TransportState.TERMINAL_SUCCESS, TransportState.TERMINAL_FAILED])
def test_cancel_keeps_entry_and_pool_until_aggregate_terminal(terminal):
    engine, store, entry, _, _, _, target = ready()
    delivery = store.reserve_delivery(entry.key, "direct", target.descriptor)
    store.start_delivery(entry.key, "direct")
    store.cancel_entry(entry.key, "cancel during batch")
    assert not entry.resources_released
    assert not store.fence_write(delivery.authorization.identity)["fenced"]
    assert delivery.packing_index_lease is not None
    engine.finish(delivery.transfer_handle, state=terminal)
    store.progress_transfers()
    assert entry.resources_released
    assert store.fence_write(delivery.authorization.identity)["fenced"]
    assert delivery.staging_guard.value is None
    assert store.registration.descriptor.region_id not in engine.released
    finish_and_close(engine, store, target)


def test_unknown_aggregate_retains_every_owner_and_prevents_page_reuse():
    engine, store, entry, _, _, _, target = ready()
    delivery = store.reserve_delivery(entry.key, "direct", target.descriptor)
    store.start_delivery(entry.key, "direct")
    engine.finish(delivery.transfer_handle, state=TransportState.UNKNOWN)
    store.cancel_entry(entry.key, "lost aggregate reply")
    store.progress_transfers()
    store.close()
    assert not entry.resources_released
    assert not store.fence_write(delivery.authorization.identity)["fenced"]
    assert delivery.packing_index_lease is not None
    assert delivery.staging_guard.value is not None
    assert not store.snapshot()["ready"]
    assert store.registration.descriptor.region_id not in engine.released
    # Production has no UNKNOWN repair. Finish only the controlled fake to
    # release CPU test tensors; this is not a serving recovery operation.
    finish_and_close(engine, store, target)


def test_short_aggregate_success_never_marks_delivery_readable():
    engine, store, entry, _, _, _, target = ready()
    delivery = store.reserve_delivery(entry.key, "direct", target.descriptor)
    store.start_delivery(entry.key, "direct")
    engine.finish(delivery.transfer_handle, short=True)
    store.poll_delivery(entry.key, "direct")
    assert delivery.state == DeliveryState.FAILED
    assert "byte count" in delivery.error
    finish_and_close(engine, store, target)


def test_cancel_during_cpu_plan_prevents_native_submit_and_releases_leases(monkeypatch):
    engine, store, entry, _, _, _, target = ready()
    prepared, proceed = threading.Event(), threading.Event()
    original = store._prepare_sparse_batch_source
    def hold(*args):
        result = original(*args)
        prepared.set()
        assert proceed.wait(10)
        return result
    monkeypatch.setattr(store, "_prepare_sparse_batch_source", hold)
    delivery = store.reserve_delivery(entry.key, "direct", target.descriptor)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(store.start_delivery, entry.key, "direct")
        assert prepared.wait(10)
        store.cancel_entry(entry.key, "cancel before adapter")
        assert not entry.resources_released
        proceed.set()
        future.result(10)
    assert not engine.batch_calls
    assert delivery.local_terminal == TransportState.NOT_SUBMITTED
    assert entry.resources_released
    assert delivery.staging_guard.value is None
    finish_and_close(engine, store, target)


def test_changed_generation_retry_and_oversize_manifest_are_rejected():
    engine, store, entry, _, _, manifest, target = ready()
    store.reserve_delivery(entry.key, "direct", target.descriptor)
    changed = replace(target.descriptor, backend_metadata={**target.descriptor.backend_metadata, "pvd_generation": "other"})
    with pytest.raises(EntryConflictError, match="different destination"):
        store.reserve_delivery(entry.key, "direct", changed)
    large = replace(manifest, specs=(replace(manifest.specs[0], token_ids=tuple(range(65))),))
    large_target = destination(store, large)
    with pytest.raises(EntryConflictError):
        store.reserve_delivery(entry.key, "oversize", large_target.descriptor)
    assert not engine.batch_calls
    engine.release_memory(large_target)
    finish_and_close(engine, store, target)


def test_server_flag_is_default_off_and_rejects_fake_transport():
    parser = build_parser()
    assert not parser.get_default("experimental_direct_sparse_batch_put")
    args = parser.parse_args([
        "--advertise-host", "127.0.0.1", "--transfer-staging-budget-bytes", "65536",
        "--transfer-max-inflight", "2", "--total-pages", "16", "--page-bytes", "384",
        "--experimental-direct-sparse-batch-put", "--transfer-backend", "fake",
        "--allow-fake-transport", "--allow-cpu-for-tests", "--no-strict-rdma-preflight",
        "--rails", "mlx5_0,mlx5_1",
    ])
    with pytest.raises(ValueError, match="direct sparse batch"):
        _validate_args(args)
