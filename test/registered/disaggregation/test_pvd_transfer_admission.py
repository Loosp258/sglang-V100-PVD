"""Explicit budgets, bounded progress and quarantine admission.

Capacity is decided before any staging tensor exists, and no state transition
refunds a reservation that still occupies memory.
"""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.arg_groups.pvd_disaggregation_hook import handle_pvd_disaggregation
from sglang.srt.disaggregation.pvd.protocol import KVEntryKey
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferBudget,
    TransferCapacityError,
    budget_of,
)
from sglang.srt.disaggregation.pvd.transfer_progress import (
    PVD_TRANSFER_CAPABILITY,
    TransferProgress,
    require_capability,
)
from sglang.srt.disaggregation.pvd.upload_manager import PVDUploadManager
from test_pvd_decode_lifecycle import SENDER_EPOCH, make_session
from test_pvd_upload_lifecycle import make_pair

# --------------------------------------------------------------------------
# Explicit configuration
# --------------------------------------------------------------------------


def base_args(**overrides):
    values = dict(
        disaggregation_topology="pvd",
        disaggregation_mode="decode",
        pvd_kv_refresh_interval=16,
        disable_overlap_schedule=False,
        pvd_vector_coordinator_url="http://v:9100",
        pvd_vector_groups=None,
        tp_size=2,
        dp_size=1,
        enable_dp_attention=False,
        pp_size=1,
        pvd_rank_rails="mlx5_0,mlx5_1",
        disaggregation_ib_device=None,
        disaggregation_transfer_backend="mooncake",
        pvd_strict_rdma_preflight=True,
        speculative_algorithm=None,
        enable_hierarchical_cache=False,
        enable_hisparse=False,
        enable_prefill_context_parallel=False,
        disaggregation_decode_enable_radix_cache=False,
        pvd_model_instance_id="model",
        pvd_transfer_staging_budget_bytes=1 << 30,
        pvd_transfer_max_inflight=64,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "field",
    ["pvd_transfer_staging_budget_bytes", "pvd_transfer_max_inflight"],
)
@pytest.mark.parametrize("bad", [None, 0, -1, True, False, 1.5, "1024"])
def test_budget_config_requires_an_explicit_positive_integer(field, bad):
    args = base_args(**{field: bad})
    with pytest.raises(ValueError) as excinfo:
        handle_pvd_disaggregation(args)
    message = str(excinfo.value)
    assert field.replace("pvd_", "--pvd-").replace("_", "-") in message


def test_valid_budgets_are_accepted_and_ordinary_pd_is_untouched():
    args = base_args()
    handle_pvd_disaggregation(args)
    assert args.pvd_transfer_staging_budget_bytes == 1 << 30
    assert args.pvd_transfer_max_inflight == 64

    pd = SimpleNamespace(disaggregation_topology="pd", disable_overlap_schedule=False)
    handle_pvd_disaggregation(pd)
    assert not pd.disable_overlap_schedule
    # Ordinary PD never grows a budget attribute.
    assert not hasattr(pd, "pvd_transfer_staging_budget_bytes")


# --------------------------------------------------------------------------
# Budget arithmetic
# --------------------------------------------------------------------------


def test_budget_rejects_before_allocation():
    budget = TransferBudget(staging_bytes=64, max_inflight=1)
    budget.reserve("a", 64, 1)
    with pytest.raises(TransferCapacityError):
        budget.reserve("b", 1, 1)
    budget.release("a")
    budget.reserve("b", 64, 1)


def test_repeated_rejections_do_not_grow_usage():
    budget = TransferBudget(staging_bytes=64, max_inflight=1)
    budget.reserve("isolated", 64, 1)
    for index in range(100):
        with pytest.raises(TransferCapacityError):
            budget.reserve(f"rejected-{index}", 64, 1)
    snapshot = budget.snapshot()
    assert snapshot["used_staging_bytes"] == 64
    assert snapshot["used_inflight"] == 1
    assert snapshot["reservations"] == 1


def test_same_owner_is_idempotent_but_cannot_change_its_reservation():
    budget = TransferBudget(staging_bytes=64, max_inflight=2)
    budget.reserve("owner", 32, 1)
    budget.reserve("owner", 32, 1)
    assert budget.snapshot()["used_staging_bytes"] == 32
    with pytest.raises(ValueError):
        budget.reserve("owner", 33, 1)


def test_two_adapters_share_one_limit():
    budget = TransferBudget(staging_bytes=100, max_inflight=4)
    budget.reserve("adapter-a", 60, 1)
    with pytest.raises(TransferCapacityError):
        budget.reserve("adapter-b", 60, 1)
    budget.reserve("adapter-b", 40, 1)
    assert budget.snapshot()["used_staging_bytes"] == 100


# --------------------------------------------------------------------------
# Reserve before allocate
# --------------------------------------------------------------------------


def test_decode_staging_is_refused_before_torch_empty_is_called():
    session, manager, _, engine, _ = make_session()
    manager.transfer_budget = TransferBudget(staging_bytes=8, max_inflight=4)
    allocations = []
    original = torch.empty

    def counting_empty(*args, **kwargs):
        allocations.append(args)
        return original(*args, **kwargs)

    torch.empty = counting_empty
    try:
        with pytest.raises(TransferCapacityError):
            session.prepare([1, 2])
    finally:
        torch.empty = original
    # The refusal happened before any staging tensor existed.
    assert allocations == []
    assert session.staging is None
    assert session.registration is None


def test_decode_staging_refund_waits_for_deregistration():
    async def scenario():
        session, manager, client, engine, _ = make_session()
        budget = TransferBudget(staging_bytes=1 << 20, max_inflight=4)
        manager.transfer_budget = budget
        session.prepare([1, 2])
        used = budget.snapshot()["used_staging_bytes"]
        assert used > 0
        session.identities = {0: session.expected_identity(0, SENDER_EPOCH)}
        client.fence_reply = {"fenced": False}
        assert await session.close() is False
        # Still pinned, so still charged.
        assert budget.snapshot()["used_staging_bytes"] == used
        client.fence_reply = {"fenced": True}
        assert await session.progress_close() is True
        assert await session.close() is True
        assert budget.snapshot()["used_staging_bytes"] == 0

    asyncio.run(scenario())


def test_reused_decode_staging_is_not_charged_twice():
    session, manager, _, _, _ = make_session()
    budget = TransferBudget(staging_bytes=1 << 20, max_inflight=4)
    manager.transfer_budget = budget
    first = session.prepare([1, 2])
    charged = budget.snapshot()["used_staging_bytes"]
    session.release_refresh()
    session.clock.complete(first["delivery_id"])
    session.req.output_ids.extend([1, 2, 3, 4])
    session.prepare([1, 2])
    assert budget.snapshot()["used_staging_bytes"] == charged
    assert budget.snapshot()["reservations"] == 1


def test_fake_engine_has_no_budget_but_production_adapter_requires_one():
    assert budget_of(FakeTransferEngine()) is None
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
        TransferLifecycleManager,
    )

    budget = TransferBudget(staging_bytes=16, max_inflight=1)
    engine = SimpleNamespace(lifecycle_manager=TransferLifecycleManager(budget))
    assert budget_of(engine) is budget


# --------------------------------------------------------------------------
# Quarantine and slots
# --------------------------------------------------------------------------


def test_inflight_slots_are_bounded_and_quarantine_blocks_admission():
    from sglang.srt.disaggregation.pvd.transfer_engine import TransferHandle
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
        ResourceGuard,
        TransferLifecycleManager,
    )

    budget = TransferBudget(staging_bytes=1024, max_inflight=2)
    manager = TransferLifecycleManager(budget)
    guards = [ResourceGuard(object(), lambda: None) for _ in range(4)]
    handles = [TransferHandle(f"t{i}") for i in range(4)]
    manager.attach(handles[0], guards[0], 16)
    manager.attach(handles[1], guards[1], 16)
    with pytest.raises(TransferCapacityError):
        manager.attach(handles[2], guards[2], 16)
    assert budget.snapshot()["used_inflight"] == 2

    manager.mark_unknown(handles[0], "native status lost")
    # Quarantine never refunds, and it stops all later admission.
    assert budget.snapshot()["used_inflight"] == 2
    with pytest.raises(RuntimeError, match="quarantined"):
        manager.attach(handles[3], guards[3], 16)
    assert manager.snapshot()["quarantined"] is True


# --------------------------------------------------------------------------
# Tombstone domains
# --------------------------------------------------------------------------


def test_tombstone_domains_are_bounded_and_reject_rather_than_forget():
    async def scenario():
        pair = await make_pair(req_id="domain-a")
        manager = pair.manager
        manager.max_tombstone_domains = 1
        identity = pair.lease.upload_identities[0]
        other = KVEntryKey("model-instance", "domain-b", "domain-b")
        second = type(identity)(
            protocol=identity.protocol,
            sender_epoch=identity.sender_epoch,
            receiver_epoch=identity.receiver_epoch,
            transfer_id="upload:domain-b:v0",
            region_id=identity.region_id,
            generation=identity.generation,
            shard_rank=0,
            key=other,
        )
        # A second domain does not fit, and the answer is refusal, not the
        # deletion of records that still have to reject late requests.
        with pytest.raises(RuntimeError, match="tombstone capacity"):
            manager.open(identity=second, coordinator=pair.client)
        assert manager.snapshot()["rejected_domains"] == 1
        await pair.shutdown()

    asyncio.run(scenario())


def test_a_closed_domain_still_refuses_a_late_start_after_compaction():
    async def scenario():
        pair = await make_pair(req_id="compact")
        task = pair.submit(0)
        await pair.settle()
        pair.finish_native(0, success=True)
        assert await task
        identity = pair.lease.upload_identities[0]
        assert pair.record(0) is None

        # Shard 1 was opened at lease time and never submitted; abandon it so
        # the domain has no live record left.
        pair.manager.abandon(pair.lease.upload_identities[1].transfer_id, "not needed")
        await pair.tick()

        removed = pair.manager.close_domain(pair.manifest.key)
        assert removed >= 1
        snapshot = pair.manager.snapshot()
        assert snapshot["closed_domains"] == 1
        # Compaction dropped the per-identity tombstones, but the domain still
        # refuses the late start they existed to block.
        with pytest.raises(RuntimeError, match="closed"):
            pair.manager.open(identity=identity, coordinator=pair.client)

    asyncio.run(scenario())


def test_a_domain_with_live_records_is_never_compacted():
    async def scenario():
        pair = await make_pair(req_id="live")
        pair.submit(0)
        await pair.settle()
        with pytest.raises(RuntimeError, match="live records"):
            pair.manager.close_domain(pair.manifest.key)
        await pair.shutdown()

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# Bounded progress
# --------------------------------------------------------------------------


def test_progress_backs_off_instead_of_spinning_and_creates_no_tasks():
    async def scenario():
        pair = await make_pair(req_id="backoff")
        task = pair.submit(0)
        await pair.settle()
        pair.finish_native(0, success=True)
        pair.client.sync_failures = 10_000
        progress = TransferProgress(
            upload_manager=pair.manager,
            decode_owner=SimpleNamespace(pending_decode_closes=[]),
            backoff_seconds=60.0,
            max_backoff_seconds=60.0,
        )
        first = await progress.tick_control()
        assert first["skipped"] is False
        assert first["backing_off"] is True
        before = len(asyncio.all_tasks())
        for _ in range(50):
            result = await progress.tick_control()
            assert result["skipped"] is True
        # Fifty ticks under backoff created no coroutines and issued no work.
        assert len(asyncio.all_tasks()) == before
        assert progress.snapshot()["skipped_for_backoff"] == 50
        with pytest.raises(Exception):
            await task
        await pair.shutdown()

    asyncio.run(scenario())


def test_progress_tick_observes_native_state_without_releasing():
    async def scenario():
        pair = await make_pair(req_id="observe")
        task = pair.submit(0)
        await pair.settle()
        progress = TransferProgress(upload_manager=pair.manager)
        result = progress.tick()
        assert result["observed"] >= 1
        assert not pair.pages_reusable(0)
        pair.finish_native(0, success=True)
        progress.tick()
        # Observation alone reaches the terminal state but releases nothing:
        # only the control step can report it to V.
        assert pair.entry(0).upload_terminal is None
        assert await progress.tick_control() is not None
        assert pair.entry(0).upload_terminal is not None
        assert await task

    asyncio.run(scenario())


def test_progress_snapshot_reports_capability_and_outstanding():
    manager = PVDUploadManager()
    owner = SimpleNamespace(pending_decode_closes=[object()])
    progress = TransferProgress(upload_manager=manager, decode_owner=owner)
    snapshot = progress.snapshot()
    assert snapshot["capability"] == PVD_TRANSFER_CAPABILITY
    assert snapshot["decode_closes_outstanding"] == 1
    assert snapshot["uploads"]["outstanding"] == 0
    assert progress.outstanding() == 1


def test_progress_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        TransferProgress(max_records_per_tick=0)
    with pytest.raises(ValueError):
        TransferProgress(backoff_seconds=0)
    with pytest.raises(ValueError):
        TransferProgress(backoff_seconds=10, max_backoff_seconds=1)


# --------------------------------------------------------------------------
# Capability admission
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "peer", [None, {}, {"capabilities": []}, {"capabilities": ["something-else"]}]
)
def test_a_peer_without_the_capability_is_refused(peer):
    with pytest.raises(RuntimeError, match=PVD_TRANSFER_CAPABILITY):
        require_capability(peer, "V")


def test_a_peer_advertising_the_capability_is_accepted():
    require_capability({"capabilities": [PVD_TRANSFER_CAPABILITY]}, "V")
