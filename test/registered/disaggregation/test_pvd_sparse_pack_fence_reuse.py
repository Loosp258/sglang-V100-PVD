"""Explicit CPU CUDA-policy doubles; no native CUDA/RDMA speed claim."""

from types import SimpleNamespace
from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest
import torch
from sglang.srt.disaggregation.pvd import server, vector_store
from sglang.srt.disaggregation.pvd.request_state import DeliveryState
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransportState
from test_pvd_cuda_sparse_packing import args, cuda_policy
from test_pvd_prompt_vectors import FakePool


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("failure", [None, "copy", "cancel", "pack_fence", "register", "submit"])
def test_reuse_counts_and_lifetimes_in_success_cancel_and_unknown(monkeypatch, reuse, failure):
    store, entry, manifest, budget, target, delivery, _ = cuda_policy(monkeypatch)
    store.reuse_sparse_pack_fence = reuse
    engine = store.transfer_engine
    original_copy = vector_store.copy_sparse_kv_into
    original_register, original_submit = engine.register_memory, engine.submit_put
    index_record = store.prompt_index._entries[entry.key.transfer_id]
    events = []

    def copy(*a, **kw):
        assert index_record.users == 1
        original_copy(*a, **kw)
        events.append("copy")
        if failure == "copy": raise RuntimeError("partly failed copy")
        if failure == "cancel": store.cancel_entry(entry.key, "packing cancellation")

    def fence(device):
        events.append("fence")
        if events.count("fence") == 1:
            assert index_record.users == 1 and delivery.packing_index_lease is not None
            if failure == "pack_fence": raise RuntimeError("pack fence unknown")

    def register(*a, **kw):
        assert delivery.packing_completion_proven and index_record.users == 0
        events.append("register")
        if failure == "register": raise RuntimeError("registration unknown")
        return original_register(*a, **kw)

    def submit(*a, **kw):
        events.append("put")
        if failure == "submit": raise RuntimeError("submission unknown")
        return original_submit(*a, **kw)

    monkeypatch.setattr(vector_store, "copy_sparse_kv_into", copy)
    monkeypatch.setattr(torch.cuda, "synchronize", fence)
    monkeypatch.setattr(engine, "register_memory", register)
    monkeypatch.setattr(engine, "submit_put", submit)
    store.start_delivery(entry.key, delivery.delivery_id)
    source = delivery.source_profile.snapshot()
    eligible = reuse and failure not in ("pack_fence", "register")
    assert source["reuse_pack_fence"] is reuse
    assert source["outer_fence_reused"] is eligible
    assert source["phases"]["pack_fence"]["calls"] == 1
    assert source["phases"]["outer_fence"]["calls"] == int(not eligible)
    assert events.count("fence") == (1 if eligible else 2)
    # Fake adapter has no third fence. Real adapter is tested independently.
    if failure in ("copy", "cancel"):
        assert "put" not in events and delivery.local_terminal is TransportState.NOT_SUBMITTED
        assert budget.snapshot()["used_staging_bytes"] == 0
        assert store.fence_write(delivery.authorization.identity)["fenced"]
    elif failure in ("pack_fence", "register", "submit"):
        assert delivery.local_terminal is TransportState.UNKNOWN
        assert budget.snapshot()["used_staging_bytes"] == manifest.nbytes
        assert delivery.staging_guard.value is not None
        assert not store.fence_write(delivery.authorization.identity)["fenced"]
        if failure == "pack_fence":
            assert not delivery.packing_completion_proven
            assert index_record.users == 1 and delivery.packing_index_lease is not None
    else:
        assert delivery.state is DeliveryState.V_WRITING
        before = list(events)
        assert store.start_delivery(entry.key, delivery.delivery_id) is delivery
        assert events == before  # repeat starts cannot repack or reuse stale proof
        assert budget.snapshot()["used_staging_bytes"] == manifest.nbytes
        engine.finish(delivery.transfer_handle)
        store.progress_transfers()
        assert delivery.state is DeliveryState.DELIVERED
        oracle = FakePool()  # same seeded fixture, independent unpacked K/V
        for payload in manifest.payload_views(target.buffer):
            spec = payload.spec
            expected = torch.stack([values[spec.layer][list(spec.token_ids), spec.kv_head]
                for values in (oracle.k_buffer, oracle.v_buffer)])
            assert torch.equal(payload.tensor, expected)
        store.ack_delivery(entry.key, delivery.delivery_id)
        assert budget.snapshot()["used_staging_bytes"] == 0
    store.close()
    engine.release_memory(target)


@pytest.mark.parametrize("rank", [0, 1])
def test_option_default_and_both_rank_constructor_wiring(monkeypatch, rank):
    assert args().experimental_reuse_sparse_pack_fence is False
    configured = args("--experimental-cuda-sparse-packing", "--experimental-reuse-sparse-pack-fence")
    configured.transfer_backend = "fake"  # wiring only, not startup qualification
    monkeypatch.setattr(server, "_build_prompt_index", lambda _: object())
    monkeypatch.setattr(server, "VectorKVStore", lambda **kw: SimpleNamespace(**kw))
    store, _ = server._create_store(configured, rank=rank, local_rank=rank, rails=["mlx5_2", "mlx5_3"])
    assert store.reuse_sparse_pack_fence is True
    assert store.device == f"cuda:{rank}"


@pytest.mark.parametrize("bad", ["no_cuda", "direct", "contiguous", "fake"])
def test_startup_refuses_ineligible_modes(bad):
    configured = args("--experimental-cuda-sparse-packing", "--experimental-reuse-sparse-pack-fence",
        "--prompt-index-vector-space", "test", "--prompt-index-budget-bytes", "1024")
    if bad == "no_cuda": configured.experimental_cuda_sparse_packing = False
    if bad == "direct": configured.experimental_direct_sparse_batch_put = True
    if bad == "contiguous": configured.experimental_contiguous_sparse_packing = True
    if bad == "fake": configured.transfer_backend = "fake"
    with pytest.raises(ValueError, match="fake transport" if bad == "fake" else "CUDA"):
        server._validate_args(configured)


def test_private_proof_resets_before_preparation_and_falls_back(monkeypatch):
    store, entry, _, _, target, delivery, _ = cuda_policy(monkeypatch)
    store.reuse_sparse_pack_fence = True
    delivery.packing_completion_proven = True  # stale marker must not authorize skip
    calls = []
    def fail(*a, **kw): raise RuntimeError("failed before pack fence")
    monkeypatch.setattr(store, "_prepare_delivery_source", fail)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))
    store.start_delivery(entry.key, delivery.delivery_id)
    assert len(calls) == 1 and not delivery.packing_completion_proven
    assert not delivery.source_profile.snapshot()["outer_fence_reused"]
    assert delivery.local_terminal is TransportState.NOT_SUBMITTED
    store.close()
    store.transfer_engine.release_memory(target)


def test_profile_allocation_failure_closes_unsubmitted_authorization(monkeypatch):
    store, entry, _, budget, target, delivery, _ = cuda_policy(monkeypatch)
    def fail(*args, **kwargs): raise MemoryError("injected bounded profile allocation failure")
    monkeypatch.setattr(vector_store, "VSourceProfile", fail)
    store.start_delivery(entry.key, delivery.delivery_id)
    assert delivery.local_terminal is TransportState.NOT_SUBMITTED
    assert not delivery.submitting and delivery.staging_guard is None
    assert store.fence_write(delivery.authorization.identity)["fenced"]
    assert budget.snapshot()["used_staging_bytes"] == 0
    store.close()
    store.transfer_engine.release_memory(target)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("reuse", [False, True])
def test_both_rank_packed_bytes_match_independent_unpacked_oracle(monkeypatch, rank, reuse):
    from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
    from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
    from test_pvd_cuda_sparse_packing import DevicePolicyDouble
    from test_pvd_prompt_index import manager, stored_entry
    from test_pvd_sparse_store_delivery import destination

    index = manager(budget=TransferBudget(1 << 20, 32))
    store, entry, oracle, layout = stored_entry(index, rank=rank)
    budget = TransferBudget(65536, 32)
    store.transfer_engine.lifecycle_manager = SimpleNamespace(budget=budget)
    store.progress_prompt_indexes()
    descriptor = index.gate_for(entry.key.transfer_id).descriptor
    spec = SparseKVSpec("consumer", "inc", "op", 4, entry.key.transfer_id,
        descriptor.index_version, descriptor.id_mapping_version, layout.fingerprint,
        0, rank * 2, (3, 0, 7))
    manifest = SparseDeliveryManifest((spec, replace(spec, layer=1,
        kv_head=rank * 2 + 1, token_ids=(1, 4))), layout.kv_dtype, layout.head_dim)
    target = destination(store, manifest)
    store.pool = DevicePolicyDouble(store.pool)
    store.allow_cuda_sparse_packing, store.reuse_sparse_pack_fence = True, reuse
    original_empty = torch.empty
    def allocate(*args, **kwargs):
        kwargs.pop("device", None)  # explicit CPU policy double
        return original_empty(*args, **kwargs)
    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    delivery = store.reserve_delivery(entry.key, "two-rank-byte-fixture", target.descriptor)
    store.start_delivery(entry.key, delivery.delivery_id)
    store.transfer_engine.finish(delivery.transfer_handle)
    store.progress_transfers()
    for payload in manifest.payload_views(target.buffer):
        s = payload.spec
        expected = torch.stack([values[s.layer][list(s.token_ids), s.kv_head]
            for values in (oracle.k_buffer, oracle.v_buffer)])
        assert torch.equal(payload.tensor, expected)
    assert delivery.source_profile.snapshot()["outer_fence_reused"] is reuse
    store.ack_delivery(entry.key, delivery.delivery_id)
    assert budget.snapshot()["used_staging_bytes"] == 0
    store.close()
    store.transfer_engine.release_memory(target)


@pytest.mark.parametrize("bad", [None, "pack_fence", "outer", "reuse", "kernel", "cuda", "bytes", "submit", "nan"])
def test_live_gate_refuses_unqualified_source_profiles(bad):
    from sglang.srt.disaggregation.pvd.v_source_profile import VSourceProfile
    path = Path(__file__).resolve().parents[3] / "benchmark/pvd_oasis_delivery_validation.py"
    spec = importlib.util.spec_from_file_location("source_delivery_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = VSourceProfile(nbytes=512, cuda=True, kernel="torch", reuse_pack_fence=True)
    profile.record_outer_fence_reuse()
    for phase in profile.snapshot()["phases"]:
        if phase != "outer_fence":
            with profile.measure(phase): pass
    value = profile.snapshot()
    if bad == "pack_fence": value["phases"]["pack_fence"]["successes"] = 0
    if bad == "outer": value["phases"]["outer_fence"]["calls"] = 1
    if bad == "reuse": value["outer_fence_reused"] = False
    if bad == "kernel": value["kernel"] = "triton"
    if bad == "cuda": value["cuda"] = False
    if bad == "bytes": value["nbytes"] = 256
    if bad == "submit": value["phases"]["submit"]["calls"] = 0
    if bad == "nan": value["phases"]["register"]["seconds"] = float("nan")
    if bad:
        with pytest.raises(AssertionError):
            module.validate_v_source_profile(value, reuse=True, nbytes=512)
    else:
        module.validate_v_source_profile(value, reuse=True, nbytes=512)
