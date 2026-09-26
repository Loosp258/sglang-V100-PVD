"""CUDA ownership policy with explicit CPU doubles; real CUDA tests stay separate."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd import server, vector_store
from sglang.srt.disaggregation.pvd.request_state import DeliveryState
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransportState
from test_pvd_sparse_store_delivery import destination, ready


class DevicePolicyDouble:
    """Only exposes CUDA policy while slicing real CPU bytes. NOT GPU execution."""

    device = torch.device("cuda:1")
    is_cuda = True

    def __init__(self, buffer):
        self.buffer = buffer

    def __getitem__(self, key):
        return self.buffer[key]


def cuda_policy(monkeypatch):
    store, entry, _, manifest, budget = ready()
    target = destination(store, manifest)
    store.pool = DevicePolicyDouble(store.pool)
    store.allow_cuda_sparse_packing = True
    allocated_devices = []
    real_empty = torch.empty

    def allocate(*args, **kwargs):
        allocated_devices.append(kwargs.pop("device", None))
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    delivery = store.reserve_delivery(entry.key, "cuda-policy", target.descriptor)
    return store, entry, manifest, budget, target, delivery, allocated_devices


@pytest.mark.parametrize("failure", [None, "copy", "cancel"])
def test_pack_drains_under_index_lease_before_registration_and_put(
    monkeypatch, failure
):
    store, entry, manifest, budget, target, delivery, devices = cuda_policy(monkeypatch)
    events = []
    engine = store.transfer_engine
    real_copy = vector_store.copy_sparse_kv_into
    real_register = engine.register_memory
    real_submit = engine.submit_put
    record = store.prompt_index._entries[entry.key.transfer_id]

    def copy(*args, **kwargs):
        events.append("copy")
        assert kwargs["allow_cuda"] is True
        assert record.users == 1
        assert budget.snapshot()["used_staging_bytes"] == manifest.nbytes
        real_copy(*args, **kwargs)  # CPU bytes, not CUDA semantics
        if failure == "copy":
            raise RuntimeError("partial copy failed")
        if failure == "cancel":
            store.cancel_entry(entry.key, "cancel during packing")
            assert delivery.staging_guard.value is not None

    def synchronize(device):
        assert device == torch.device("cuda:1")
        if events.count("sync") == 0:
            assert record.users == 1
            assert delivery.packing_index_lease is not None
            assert not store.entries[entry.key].resources_released
        events.append("sync")

    def register(*args, **kwargs):
        assert record.users == 0
        assert delivery.packing_index_lease is None
        assert events[:2] == ["copy", "sync"]
        events.append("register")
        return real_register(*args, **kwargs)

    def submit(*args, **kwargs):
        assert "sync" in events and "register" in events
        events.append("put")
        return real_submit(*args, **kwargs)

    monkeypatch.setattr(vector_store, "copy_sparse_kv_into", copy)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(engine, "register_memory", register)
    monkeypatch.setattr(engine, "submit_put", submit)
    store.start_delivery(entry.key, delivery.delivery_id)
    assert devices == [torch.device("cuda:1")]
    assert record.users == 0
    assert delivery.packing_index_lease is None
    if failure is None:
        assert delivery.state is DeliveryState.V_WRITING
        assert events[-1] == "put"
        assert budget.snapshot()["used_staging_bytes"] == manifest.nbytes
        engine.finish(delivery.transfer_handle)
        store.progress_transfers()
        assert delivery.state is DeliveryState.DELIVERED
        store.ack_delivery(entry.key, delivery.delivery_id)
    else:
        assert "put" not in events
        assert delivery.local_terminal is TransportState.NOT_SUBMITTED
        assert store.fence_write(delivery.authorization.identity)["fenced"]
        if failure == "copy":
            assert "register" not in events
    assert budget.snapshot()["used_staging_bytes"] == 0
    store.close()
    engine.release_memory(target)


def test_failed_cuda_drain_keeps_all_owners_even_if_outer_sync_later_succeeds(
    monkeypatch,
):
    store, entry, manifest, budget, target, delivery, _ = cuda_policy(monkeypatch)
    record = store.prompt_index._entries[entry.key.transfer_id]
    calls = []

    def synchronize(device):
        calls.append(device)
        if len(calls) == 1:
            raise RuntimeError("CUDA completion unknown")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    store.start_delivery(entry.key, delivery.delivery_id)
    assert len(calls) == 2
    assert delivery.local_terminal is TransportState.UNKNOWN
    assert delivery.transfer_handle is None
    assert delivery.packing_index_lease is not None
    assert delivery.staging_guard.value is not None
    assert record.users == 1
    assert store.snapshot()["isolated_reason"]
    store.prompt_index.close(entry.key.transfer_id)
    store.close()
    assert record.users == 1
    assert store.prompt_index.budget.snapshot()["used_staging_bytes"] > 0
    assert budget.snapshot()["used_staging_bytes"] == manifest.nbytes
    assert not store.entries[entry.key].resources_released
    assert not store.fence_write(delivery.authorization.identity)["fenced"]
    store.transfer_engine.release_memory(target)  # no PUT was submitted by the double


def test_cuda_store_without_opt_in_still_refuses_before_authorization():
    store, entry, _, manifest, _ = ready()
    target = destination(store, manifest)
    store.pool = DevicePolicyDouble(store.pool)
    with pytest.raises(vector_store.EntryConflictError, match="explicit CUDA opt-in"):
        store.reserve_delivery(entry.key, "disabled", target.descriptor)
    assert not store.entries[entry.key].deliveries
    assert store.snapshot()["sparse_packing_mode"] == "cpu_reference_only"
    assert store.snapshot()["sparse_pack_kernel"] == "torch"
    store.close()
    store.transfer_engine.release_memory(target)


def args(*extra):
    return server.build_parser().parse_args(
        [
            "--advertise-host",
            "localhost",
            "--total-pages",
            "8",
            "--page-bytes",
            "8",
            "--transfer-staging-budget-bytes",
            "1024",
            "--transfer-max-inflight",
            "2",
            *extra,
        ]
    )


def test_cuda_packing_flag_is_opt_in_and_explicitly_not_d_activation(caplog):
    assert not args().experimental_cuda_sparse_packing
    configured = args(
        "--experimental-cuda-sparse-packing",
        "--prompt-index-vector-space",
        "test",
        "--prompt-index-budget-bytes",
        "1024",
    )
    server._validate_args(configured)
    assert "D predictive serving is not enabled" in caplog.text


def test_triton_sparse_packing_flag_requires_existing_cuda_path():
    assert not args().experimental_triton_sparse_packing
    configured = args("--experimental-triton-sparse-packing")
    with pytest.raises(ValueError, match="requires CUDA sparse packing"):
        server._validate_args(configured)


def test_triton_sparse_packing_option_reaches_both_v_shards(monkeypatch):
    configured = args(
        "--experimental-cuda-sparse-packing", "--experimental-triton-sparse-packing"
    )
    configured.transfer_backend = "fake"  # constructor wiring only
    monkeypatch.setattr(server, "_build_prompt_index", lambda _: object())
    monkeypatch.setattr(server, "VectorKVStore", lambda **kw: SimpleNamespace(**kw))
    for rank in (0, 1):
        store, _ = server._create_store(
            configured, rank=rank, local_rank=rank, rails=["mlx5_2", "mlx5_3"]
        )
        assert store.fused_cuda_sparse_packing is True


@pytest.mark.parametrize("unknown_completion", [False, True])
def test_fused_pack_metadata_budget_is_held_until_cuda_fence(
    monkeypatch, unknown_completion
):
    store, entry, manifest, budget, _, delivery, _ = cuda_policy(monkeypatch)
    store.fused_cuda_sparse_packing = True
    assert store.snapshot()["sparse_pack_kernel"] == "triton"
    events = []
    module = types.ModuleType("sglang.srt.disaggregation.pvd.triton_sparse_pack")

    class FakeWorkspace:
        def __init__(self, selected, *, budget, owner, **unused):
            self.manifest_fingerprint = selected.fingerprint
            self.layout_fingerprint = unused["layout"].fingerprint
            self.shard = unused["shard"]
            self.device = torch.device("cpu")  # fake policy, real CPU bytes
            self.budget, self.owner = budget, owner
            budget.reserve(owner, 64, 0)
            events.append("reserve")

        def release_after_fence(self):
            events.append("refund")
            self.budget.release(self.owner)

    def launch(source, destination, workspace, **kwargs):
        assert delivery.packing_workspace is workspace
        assert budget.snapshot()["used_staging_bytes"] == manifest.nbytes + 64
        destination.fill_(27)
        events.append("launch")

    module.SparsePackWorkspace = FakeWorkspace
    module.launch_sparse_pack = launch
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def synchronize(device):
        events.append("sync")
        assert device == torch.device("cuda:1")
        if unknown_completion:
            raise RuntimeError("device completion unknown")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    store.start_delivery(entry.key, delivery.delivery_id)
    if unknown_completion:
        assert delivery.packing_workspace is not None
        assert "refund" not in events
        assert budget.snapshot()["used_staging_bytes"] == manifest.nbytes + 64
        assert store.snapshot()["isolated_reason"] is not None
    else:
        assert delivery.packing_workspace is None
        assert events[:4] == ["reserve", "launch", "sync", "refund"]
        assert budget.snapshot()["used_staging_bytes"] == manifest.nbytes
        engine = store.transfer_engine
        engine.finish(delivery.transfer_handle)
        store.progress_transfers()
        store.ack_delivery(entry.key, delivery.delivery_id)
        assert budget.snapshot()["used_staging_bytes"] == 0
        store.close()


def test_unknown_metadata_upload_isolates_v_even_if_later_sync_succeeds(monkeypatch):
    from sglang.srt.disaggregation.pvd.sparse_pack_plan import (
        SparsePackCompletionUnknown,
    )

    store, entry, manifest, budget, _, delivery, _ = cuda_policy(monkeypatch)
    store.fused_cuda_sparse_packing = True
    module = types.ModuleType("sglang.srt.disaggregation.pvd.triton_sparse_pack")

    def failed_workspace(selected, *, budget, owner, **unused):
        budget.reserve(owner, 64, 0)
        raise SparsePackCompletionUnknown("metadata upload completion unknown")

    module.SparsePackWorkspace = failed_workspace
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    store.start_delivery(entry.key, delivery.delivery_id)
    assert delivery.local_terminal is TransportState.UNKNOWN
    assert store.snapshot()["isolated_reason"] is not None
    assert budget.snapshot()["used_staging_bytes"] == manifest.nbytes + 64
    assert delivery.transfer_handle is None


@pytest.mark.parametrize("missing", ["space", "budget", "cuda"])
def test_cuda_packing_launch_refuses_incomplete_configuration(missing):
    configured = args(
        "--experimental-cuda-sparse-packing",
        "--prompt-index-vector-space",
        "test",
        "--prompt-index-budget-bytes",
        "1024",
    )
    if missing == "space":
        configured.prompt_index_vector_space = None
    elif missing == "budget":
        configured.prompt_index_budget_bytes = None
    else:
        configured.transfer_backend = "fake"
        configured.allow_fake_transport = True
        configured.strict_rdma_preflight = False
        configured.allow_cpu_for_tests = True
    with pytest.raises(ValueError, match="experimental CUDA sparse packing requires"):
        server._validate_args(configured)


@pytest.mark.parametrize("rank", [0, 1])
def test_both_shard_creation_paths_forward_the_explicit_option(monkeypatch, rank):
    configured = args("--experimental-cuda-sparse-packing")
    configured.transfer_backend = "fake"  # constructor wiring only, no validation claim
    monkeypatch.setattr(server, "_build_prompt_index", lambda _: object())
    monkeypatch.setattr(server, "VectorKVStore", lambda **kw: SimpleNamespace(**kw))
    store, _ = server._create_store(
        configured, rank=rank, local_rank=rank, rails=["mlx5_2", "mlx5_3"]
    )
    assert store.allow_cuda_sparse_packing is True
    assert store.device == f"cuda:{rank}"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="real CUDA required; unverified"
)
@pytest.mark.parametrize("copy_failure", [False, True])
def test_actual_cuda_store_pack_lifetime_and_bytes(monkeypatch, copy_failure):
    """Actual CUDA packing, still a fake transfer engine -- never RDMA evidence."""
    from sglang.srt.disaggregation.pvd.sparse_delivery import SPARSE_DELIVERY_KEY
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
    from test_pvd_prompt_index import build_entry, manager
    from test_pvd_sparse_delivery import manifest_for
    from test_pvd_vector_lifecycle import DelayedTransferEngine

    pool, layout, entry_manifest, packed, shard = build_entry()
    engine = DelayedTransferEngine()
    budget = TransferBudget(65536, 32)
    engine.lifecycle_manager = SimpleNamespace(budget=budget)
    index = manager(budget=TransferBudget(1 << 20, 32))
    store = vector_store.VectorKVStore(
        rank=0,
        world_size=2,
        rail=shard.rail,
        device="cuda:0",
        total_pages=16,
        page_bytes=shard.expected_bytes // shard.page_count,
        endpoint="test-v",
        transfer_engine=engine,
        prompt_index=index,
        allow_cuda_sparse_packing=True,
    )
    entry = store.create_entry(entry_manifest)
    store.begin_p_write(entry.key)
    offset = entry.allocation.start_page * store.page_bytes
    store.pool[offset : offset + packed.tensor.numel()].copy_(packed.tensor)
    torch.cuda.synchronize(store.pool.device)
    store.commit_p_write(entry.key, shard.expected_bytes)
    assert store.progress_prompt_indexes()["built"] == 1
    manifest = manifest_for(index, entry.key, layout)
    target = engine.register_memory(
        torch.zeros(manifest.nbytes, dtype=torch.uint8, device=store.pool.device),
        endpoint="test-d",
        rank=0,
        rail=store.rail,
        metadata={
            "pvd_receiver_epoch": "test-D",
            "pvd_generation": "gen",
            SPARSE_DELIVERY_KEY: manifest.to_dict(),
        },
    )
    delivery = store.reserve_delivery(entry.key, "actual-cuda", target.descriptor)
    copy = vector_store.copy_sparse_kv_into

    def fail_after_copy(*args, **kwargs):
        copy(*args, **kwargs)
        raise RuntimeError("fault after real queued CUDA copies")

    try:
        if copy_failure:
            monkeypatch.setattr(vector_store, "copy_sparse_kv_into", fail_after_copy)
        store.start_delivery(entry.key, delivery.delivery_id)
        assert delivery.packing_index_lease is None
        if copy_failure:
            assert delivery.state is DeliveryState.FAILED
            assert delivery.transfer_handle is None
            assert store.fence_write(delivery.authorization.identity)["fenced"]
        else:
            assert delivery.staging_guard.value.device == store.pool.device
            engine.finish(delivery.transfer_handle)
            torch.cuda.synchronize(store.pool.device)  # fake-engine copy completion
            store.progress_transfers()
            for payload in manifest.payload_views(target.buffer):
                spec = payload.spec
                expected = torch.stack(
                    [
                        values[spec.layer][list(spec.token_ids), spec.kv_head]
                        for values in (pool.k_buffer, pool.v_buffer)
                    ]
                ).to(store.pool.device)
                torch.testing.assert_close(payload.tensor, expected, rtol=0, atol=0)
            store.ack_delivery(entry.key, delivery.delivery_id)
        assert budget.snapshot()["used_staging_bytes"] == 0
    finally:
        store.close()
        engine.release_memory(target)
