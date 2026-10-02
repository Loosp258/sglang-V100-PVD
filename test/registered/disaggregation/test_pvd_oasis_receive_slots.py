"""CPU policy/fault tests for exclusive CUDA receive-slot owners, not RDMA proof."""

import dataclasses
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.oasis_receive_slots import OasisReceiveSlotPool
from sglang.srt.disaggregation.pvd.protocol import (
    KVEntryKey,
    PVD_GENERATION_METADATA_KEY,
    PVD_RECEIVER_EPOCH_METADATA_KEY,
    PVD_TRANSFER_LIFECYCLE_PROTOCOL,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.sparse_delivery import (
    SPARSE_DELIVERY_KEY,
    SparseDeliveryManifest,
)
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
from sglang.srt.disaggregation.pvd.sparse_receiver import SparseReceiveError
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine, MemorySlice
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget, TransportState


class RecordingEngine(FakeTransferEngine):
    def __init__(self):
        super().__init__()
        self.registered, self.released = [], []
        self.release_threads = []
        self.raise_register = self.raise_release = False
        self.corrupt_descriptor = None

    def register_memory(self, buffer, **kwargs):
        registration = super().register_memory(buffer, **kwargs)
        self.registered.append(registration)
        if self.raise_register:
            raise RuntimeError("native registration reply lost")
        if self.corrupt_descriptor is not None:
            registration.descriptor = self.corrupt_descriptor(registration.descriptor)
        return registration

    def release_memory(self, registration):
        assert any(registration is original for original in self.registered)
        self.released.append(registration)
        self.release_threads.append(threading.get_ident())
        if self.raise_release:
            raise RuntimeError("native unregister reply lost")
        return super().release_memory(registration)


def case(*, capacity_bytes=4096, slots_per_rank=2, byte_budget=1 << 20):
    engine = RecordingEngine()
    budget = TransferBudget(byte_budget, 8)
    pool = OasisReceiveSlotPool(engine, budget, device="cuda:0", receiver_epoch="D",
        slots_per_rank=slots_per_rank, capacity_bytes=capacity_bytes)
    # Explicit policy test: actual CPU bytes, never a production CPU fallback.
    pool.device = torch.device("cpu")
    ordering_calls = []
    ordering = SimpleNamespace(prepare=lambda b: ordering_calls.append(b.data_ptr()))
    key = KVEntryKey("model", "prompt", "entry")
    return SimpleNamespace(pool=pool, engine=engine, budget=budget, ordering=ordering,
                           ordering_calls=ordering_calls, key=key)


def request(c, *, rank=0, counts=(2, 2), generation=None, delivery_id=None):
    identity = WriteIdentity(PVD_TRANSFER_LIFECYCLE_PROTOCOL, f"V{rank}", "D",
        delivery_id or uuid.uuid4().hex, "pending-registration",
        generation or uuid.uuid4().hex, rank, c.key)
    manifest = SparseDeliveryManifest(tuple(SparseKVSpec("req", "inc", identity.transfer_id,
        1, "entry", "index", "mapping", "layout", 3, 2 * rank + i, tuple(range(n)))
        for i, n in enumerate(counts)), "torch.float16", 128)
    return manifest, identity


def acquire(c, *, rank=0, counts=(2, 2), generation=None, delivery_id=None):
    manifest, identity = request(c, rank=rank, counts=counts, generation=generation,
                                 delivery_id=delivery_id)
    lease = c.pool.acquire(manifest, identity, endpoint=f"D{rank}", rail=f"rail{rank}",
                           device="cpu", ordering=c.ordering)
    return lease, manifest


def test_sequential_different_extents_reuse_physical_owner_and_exact_byte_contract():
    c = case()
    previous = None
    for counts, marker in (((3, 2), 23), ((1,), 47), ((2, 3), 89)):
        lease, manifest = acquire(c, counts=counts)
        original = c.engine.registered[0]
        descriptor = lease.registration.descriptor
        assert lease.registration is not original
        assert lease.buffer.numel() == manifest.nbytes == descriptor.length
        assert descriptor.region_id == original.descriptor.region_id
        assert descriptor.address == original.descriptor.address
        assert descriptor.backend_metadata[SPARSE_DELIVERY_KEY] == manifest.to_dict()
        lease.identity.validate_destination(descriptor)
        if previous is not None:
            assert previous.region_id == lease.identity.region_id
            with pytest.raises(ValueError, match="generation"):
                previous.validate_destination(descriptor)
        assert SPARSE_DELIVERY_KEY not in original.descriptor.backend_metadata
        assert original.descriptor.backend_metadata[PVD_GENERATION_METADATA_KEY].startswith("physical:")
        # Fake terminal success proves this synchronous byte copy in this test.
        backing = torch.full((manifest.nbytes,), marker, dtype=torch.uint8)
        source = FakeTransferEngine().register_memory(backing, endpoint="V", rank=0, rail="rail0")
        handle = c.engine.submit_put(MemorySlice(source, 0, manifest.nbytes), descriptor)
        assert handle.transport_state == TransportState.TERMINAL_SUCCESS
        assert handle.transferred_bytes == manifest.nbytes
        assert torch.equal(lease.buffer, backing)
        installed = lease.buffer.clone()
        FakeTransferEngine().release_memory(source)
        previous = lease.identity
        lease.release_after_proof()
        assert lease.buffer is lease.registration is None
        assert torch.equal(installed, backing)
        assert c.budget.snapshot()["used_staging_bytes"] == c.pool.capacity_bytes
        assert c.budget.snapshot()["used_inflight"] == 0
    assert len(c.engine.registered) == len(c.ordering_calls) == 1
    assert c.pool.snapshot()["acquired_leases"] == c.pool.snapshot()["returned_leases"] == 3
    c.pool.close()
    assert c.engine.released == c.engine.registered
    assert c.budget.snapshot()["used_staging_bytes"] == 0
    assert c.pool.snapshot()["physical_bytes"] == 0


def test_both_executors_reuse_four_slots_and_parent_retires_original_handles():
    c = case()
    region_sets = []
    for _ in range(2):  # Bootstrap and Decode use distinct worker executors.
        acquired = threading.Barrier(3)
        returned = threading.Barrier(3)

        def worker():
            leases = [acquire(c, rank=r)[0] for r in (0, 1)]
            acquired.wait(timeout=10)
            regions = {l.identity.region_id for l in leases}
            returned.wait(timeout=10)
            for lease in leases:
                lease.release_after_proof()
            return regions

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(worker) for _ in range(2)]
            acquired.wait(timeout=10)
            assert c.pool.snapshot()["leased_slots"] == 4
            assert c.pool.snapshot()["physical_bytes"] == 4 * c.pool.capacity_bytes
            with pytest.raises(SparseReceiveError, match="live or unknown"):
                c.pool.close()
            returned.wait(timeout=10)
            region_sets.append(set().union(*(f.result(timeout=10) for f in futures)))
    assert region_sets[0] == region_sets[1]
    assert len(c.engine.registered) == len(c.ordering_calls) == 4
    c.pool.close()
    assert len(c.engine.released) == 4
    assert set(c.engine.release_threads) == {threading.get_ident()}
    assert c.pool.snapshot()["physical_register_calls"] == 4
    assert c.pool.snapshot()["physical_releases"] == 4
    assert c.pool.snapshot()["closed"]


def test_busy_slots_never_expand_rank_capacity():
    c = case()
    leases = [acquire(c)[0] for _ in range(2)]
    with pytest.raises(SparseReceiveError, match="busy"):
        acquire(c)
    assert len(c.engine.registered) == 2
    assert c.budget.snapshot()["used_staging_bytes"] == 2 * c.pool.capacity_bytes
    for lease in leases:
        lease.release_after_proof()
    c.pool.close()


@pytest.mark.parametrize("field", ["generation", "delivery_id"])
def test_never_reuse_previous_logical_write_identity(field):
    c = case()
    lease, _ = acquire(c, **{field: "already-used"})
    lease.release_after_proof()
    with pytest.raises(SparseReceiveError, match="already used"):
        acquire(c, **{field: "already-used"})
    assert len(c.engine.registered) == 1
    c.pool.close()


def test_byte_budget_admission_happens_before_allocation_or_register(monkeypatch):
    c = case(byte_budget=1024)
    monkeypatch.setattr(c.pool, "_allocate_buffer", lambda: pytest.fail("allocation before budget"))
    with pytest.raises(RuntimeError, match="capacity"):
        acquire(c)
    assert not c.engine.registered
    assert c.budget.snapshot()["used_staging_bytes"] == 0
    assert c.pool.snapshot()["physical_register_calls"] == 0


@pytest.mark.parametrize("failed_step", ["allocation", "ordering"])
def test_failure_before_native_registration_refunds_proven_safe_slot(monkeypatch, failed_step):
    c = case()

    def fail(*_):
        raise RuntimeError("before-native failure")

    if failed_step == "allocation":
        monkeypatch.setattr(c.pool, "_allocate_buffer", fail)
    else:
        c.ordering.prepare = fail
    with pytest.raises(RuntimeError, match="before-native"):
        acquire(c)
    assert not c.engine.registered
    assert c.pool.snapshot()["physical_slots"] == 0
    assert c.budget.snapshot()["used_staging_bytes"] == 0
    c.pool.close()


def test_native_registration_unknown_retains_buffer_and_budget_without_replacement():
    c = case()
    c.engine.raise_register = True
    with pytest.raises(RuntimeError, match="reply lost"):
        acquire(c)
    slot = c.pool._slots[0]
    assert slot.buffer is not None and slot.registration_unknown
    assert c.pool.snapshot()["unknown_slots"] == 1
    assert c.budget.snapshot()["used_staging_bytes"] == c.pool.capacity_bytes
    assert c.pool.snapshot()["leased_slots"] == 0
    c.engine.raise_register = False
    with pytest.raises(SparseReceiveError, match="quarantined"):
        acquire(c)
    with pytest.raises(SparseReceiveError, match="live or unknown"):
        c.pool.close()
    assert len(c.engine.registered) == 1 and not c.engine.released


@pytest.mark.parametrize("field,value", [
    ("rank", 1), ("rail", "wrong"), ("endpoint", "wrong"),
    ("device", "cuda:1"), ("length", 4095), ("address", 123),
])
def test_wrong_physical_descriptor_is_quarantined_before_publication(field, value):
    c = case()
    c.engine.corrupt_descriptor = lambda d: dataclasses.replace(d, **{field: value})
    with pytest.raises(SparseReceiveError, match="registration differs"):
        acquire(c)
    assert c.pool.snapshot()["unknown_slots"] == 1
    assert c.pool.snapshot()["acquired_leases"] == 0
    assert c.budget.snapshot()["used_staging_bytes"] == c.pool.capacity_bytes
    assert not c.engine.released


@pytest.mark.parametrize("reason", ["RDMA terminal unknown", "CUDA ordering unknown", "D2H completion unknown"])
def test_unknown_lease_can_never_return_to_idle(reason):
    c = case()
    lease, _ = acquire(c)
    buffer, registration = lease.buffer, lease.registration
    lease.quarantine(reason)
    with pytest.raises(SparseReceiveError, match="cannot be recycled"):
        lease.release_after_proof()
    with pytest.raises(SparseReceiveError, match="quarantined"):
        acquire(c, rank=1)
    with pytest.raises(SparseReceiveError, match="live or unknown"):
        c.pool.close()
    assert lease.buffer is buffer and lease.registration is registration
    assert c.pool.snapshot()["leased_slots"] == 1
    assert c.pool.snapshot()["unknown_slots"] == 1
    assert c.pool.snapshot()["quarantine"] == reason
    assert c.budget.snapshot()["used_staging_bytes"] == c.pool.capacity_bytes
    assert not c.engine.released


def test_unregister_unknown_is_sticky_and_keeps_original_registration():
    c = case()
    lease, _ = acquire(c)
    original = c.engine.registered[0]
    lease.release_after_proof()
    c.engine.raise_release = True
    with pytest.raises(RuntimeError, match="reply lost"):
        c.pool.close()
    assert c.pool._slots[0].registration is original
    assert c.budget.snapshot()["used_staging_bytes"] == c.pool.capacity_bytes
    c.engine.raise_release = False
    with pytest.raises(SparseReceiveError, match="live or unknown"):
        c.pool.close()
    assert len(c.engine.released) == 1  # No speculative native unregister retry.
    assert c.pool.snapshot()["physical_releases"] == 0
    assert c.pool.snapshot()["unknown_slots"] == 1


def test_double_return_or_foreign_pool_never_releases_other_write_slot():
    c, other = case(), case()
    lease, _ = acquire(c)
    with pytest.raises(SparseReceiveError, match="exact live"):
        other.pool._return_lease(lease)
    assert c.pool.snapshot()["leased_slots"] == 1
    lease.release_after_proof()
    with pytest.raises(SparseReceiveError, match="exact live"):
        lease.release_after_proof()
    c.pool.close()
    other.pool.close()
    c.pool.close()  # Successful pool close is idempotent.
    assert len(c.engine.released) == 1


@pytest.mark.parametrize("change", ["extent", "heads", "dtype", "layer", "Entry", "epoch", "region"])
def test_invalid_logical_destinations_fail_before_charging_or_registration(change):
    c = case()
    manifest, identity = request(c)
    if change == "extent":
        manifest, _ = request(c, counts=(8, 8))
    elif change == "heads":
        manifest = dataclasses.replace(manifest,
            specs=tuple(dataclasses.replace(s, kv_head=s.kv_head + 2) for s in manifest.specs))
    elif change == "dtype":
        manifest = dataclasses.replace(manifest, dtype="torch.float32")
    elif change == "layer":
        manifest = dataclasses.replace(manifest, specs=(manifest.specs[0],
            dataclasses.replace(manifest.specs[1], layer=4)))
    elif change == "Entry":
        identity = dataclasses.replace(identity, key=KVEntryKey("model", "prompt", "wrong-entry"))
    elif change == "epoch":
        identity = dataclasses.replace(identity, receiver_epoch="stale-D")
    else:
        identity = dataclasses.replace(identity, region_id="old-published-region")
    with pytest.raises(SparseReceiveError):
        c.pool.acquire(manifest, identity, endpoint="D0", rail="rail0", device="cpu", ordering=c.ordering)
    assert c.budget.snapshot()["used_staging_bytes"] == 0
    assert not c.engine.registered
    c.pool.close()


def test_rank_never_changes_native_session_or_rail_after_registration():
    c = case()
    lease, _ = acquire(c)
    lease.release_after_proof()
    manifest, identity = request(c)
    with pytest.raises(SparseReceiveError, match="cannot change"):
        c.pool.acquire(manifest, identity, endpoint="D-new", rail="rail0", device="cpu", ordering=c.ordering)
    assert len(c.engine.registered) == 1
    c.pool.close()


def test_each_delivery_generation_does_not_modify_physical_metadata():
    c = case()
    lease, _ = acquire(c)
    physical = c.engine.registered[0]
    before = dict(physical.descriptor.backend_metadata)
    assert before[PVD_RECEIVER_EPOCH_METADATA_KEY] == lease.identity.receiver_epoch
    assert before[PVD_GENERATION_METADATA_KEY] != lease.identity.generation
    alias = lease.registration.descriptor
    alias.backend_metadata["local-test-mutation"] = True
    assert physical.descriptor.backend_metadata == before
    lease.release_after_proof()
    c.pool.close()


@pytest.mark.parametrize("kwargs", [
    {"device": "cpu"}, {"device": "cuda"}, {"slots_per_rank": 0},
    {"slots_per_rank": 5}, {"capacity_bytes": 0}, {"capacity_bytes": 2 * 2048 * 512 + 1},
    {"receiver_epoch": ""}, {"ranks": (0,)},
])
def test_constructor_requires_explicit_bounded_cuda_request(kwargs):
    arguments = dict(device="cuda:0", receiver_epoch="D", slots_per_rank=2, capacity_bytes=4096)
    arguments.update(kwargs)
    with pytest.raises(SparseReceiveError):
        OasisReceiveSlotPool(RecordingEngine(), TransferBudget(1 << 20, 8), **arguments)
