"""Rank agreement under adversarial delivery order; no distributed/GPU claim."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import permutations

import pytest
import torch
from sglang.srt.disaggregation.pvd.sparse_install import (
    CPUInstallGroup,
    InstallProtocolError,
    RankInstallCoordinator,
    RankInstallReceipt,
)
from sglang.srt.disaggregation.pvd.sparse_payload import (
    SparseKVPayload,
    SparseKVSpec,
    SparsePayloadError,
)
from sglang.srt.disaggregation.pvd.sparse_working_set import CPUSparseWorkingSet
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


def protocol(request="r", incarnation="inc", ranks=(0, 1)):
    return RankInstallCoordinator(
        request,
        incarnation,
        "entry",
        rank_layouts={r: f"layout-{r}" for r in ranks},
        interval=4,
        lead_tokens=1,
    )


def receipt(epoch, rank):
    return RankInstallReceipt(
        epoch, rank, f"staged-{epoch.round}-{rank}", f"layout-{rank}"
    )


def complete(coordinator, epoch):
    ranks = coordinator.snapshot()["expected_ranks"]
    for rank in ranks:
        r = receipt(epoch, rank)
        coordinator.prepared(r)
        coordinator.parked(r, epoch.target_tokens)
    assert coordinator.decide_install(epoch)
    for rank in ranks:
        coordinator.applied(receipt(epoch, rank))


@pytest.mark.parametrize("ranks", [(0,), (0, 1), (2, 5, 8)])
def test_requires_every_declared_rank_not_a_hardcoded_tp2(ranks):
    c = protocol(ranks=ranks)
    epoch = c.begin(0)
    for rank in ranks:
        c.prepared(receipt(epoch, rank))
    assert not c.can_decode(0)
    assert not c.decide_install(epoch)  # ready is not drained
    for rank in ranks:
        c.parked(receipt(epoch, rank), 0)
    assert c.decide_install(epoch)
    for rank in ranks[:-1]:
        c.applied(receipt(epoch, rank))
        assert not c.can_decode(0)
        assert c.snapshot()["installed_tokens"] is None
    c.applied(receipt(epoch, ranks[-1]))
    assert c.can_decode(0)
    assert c.snapshot()["round"] == 1
    assert c.snapshot()["next_boundary"] == 4


@pytest.mark.parametrize("order", list(permutations((0, 1, 2))))
def test_ack_reordering_and_duplicates_commit_exactly_once(order):
    c = protocol(ranks=(0, 1, 2))
    epoch = c.begin(0)
    for rank in reversed(order):
        r = receipt(epoch, rank)
        c.prepared(r)
        c.prepared(r)
        c.parked(r, 0)
        c.parked(r, 0)
    assert c.decide_install(epoch)
    assert c.decide_install(epoch)
    for rank in order:
        c.applied(receipt(epoch, rank))
        c.applied(receipt(epoch, rank))
    assert c.snapshot()["round"] == 1


def test_late_rank_waits_at_boundary_and_new_request_does_not_reset_old_round():
    c = protocol()
    complete(c, c.begin(0))
    epoch = c.begin(3)
    c.prepared(receipt(epoch, 0))
    assert c.can_decode(3)
    snapshot = c.snapshot()
    new = protocol("new")
    complete(new, new.begin(0))
    new.cancel()
    assert c.snapshot() == snapshot
    c.parked(receipt(epoch, 0), 4)
    assert not c.decide_install(epoch)
    assert not c.can_decode(4)
    with pytest.raises(ValueError, match="past an uninstalled"):
        c.can_decode(5)
    c.prepared(receipt(epoch, 1))
    c.parked(receipt(epoch, 1), 4)
    assert c.decide_install(epoch)
    c.applied(receipt(epoch, 1))
    assert not c.can_decode(4)
    c.applied(receipt(epoch, 0))
    assert c.can_decode(4)
    assert c.snapshot()["next_boundary"] == 8


@pytest.mark.parametrize(
    "change",
    [
        "request_id",
        "incarnation",
        "entry_transfer_id",
        "operation_id",
        "round",
        "target_tokens",
    ],
)
def test_foreign_or_stale_identity_cannot_prepare(change):
    c = protocol()
    epoch = c.begin(0)
    bad = replace(
        epoch, **{change: 1 if change in ("round", "target_tokens") else "other"}
    )
    before = c.snapshot()
    with pytest.raises(InstallProtocolError, match="stale or foreign"):
        c.prepared(receipt(bad, 0))
    assert c.snapshot() == before


def test_conflicting_prepared_bank_and_unknown_rank_are_refused():
    c = protocol()
    epoch = c.begin(0)
    r = receipt(epoch, 0)
    c.prepared(r)
    for bad in (
        replace(r, staging_id="replacement"),
        replace(r, layout_fingerprint="wrong"),
        receipt(epoch, 8),
    ):
        with pytest.raises(InstallProtocolError):
            c.prepared(bad)
    assert c.snapshot()["prepared"] == (0,)


def test_early_ack_or_early_park_cannot_advance_clock():
    c = protocol()
    complete(c, c.begin(0))
    epoch = c.begin(3)
    r = receipt(epoch, 0)
    with pytest.raises(InstallProtocolError, match="not prepared"):
        c.parked(r, 4)
    c.prepared(r)
    with pytest.raises(InstallProtocolError, match="exact boundary"):
        c.parked(r, 3)
    with pytest.raises(InstallProtocolError, match="not been decided"):
        c.applied(r)
    assert c.can_decode(3)


@pytest.mark.parametrize("action", ["cancel", "fail"])
def test_partial_install_terminal_failure_never_reopens_decode(action):
    c = protocol()
    epoch = c.begin(0)
    for rank in (0, 1):
        c.prepared(receipt(epoch, rank))
        c.parked(receipt(epoch, rank), 0)
    c.decide_install(epoch)
    c.applied(receipt(epoch, 0))
    if action == "cancel":
        c.cancel("EOS/cancel during install")
    else:
        c.fail(epoch, "rank 1 timeout")
    assert not c.can_decode(0)
    assert c.snapshot()["installed_tokens"] is None
    for late in (
        lambda: c.applied(receipt(epoch, 1)),
        lambda: c.prepared(receipt(epoch, 1)),
        lambda: c.begin(0),
    ):
        with pytest.raises(InstallProtocolError, match="terminal"):
            late()


def test_previous_round_ack_and_callback_thread_cannot_change_active_round():
    c = protocol()
    initial = c.begin(0)
    complete(c, initial)
    pending = c.begin(3)
    with pytest.raises(InstallProtocolError, match="stale or foreign"):
        c.applied(receipt(initial, 0))
    with (
        ThreadPoolExecutor(1) as executor,
        pytest.raises(InstallProtocolError, match="owner thread"),
    ):
        executor.submit(c.prepared, receipt(pending, 0)).result()
    assert c.snapshot()["prepared"] == ()


def cpu_group(request="r"):
    budgets = {r: TransferBudget(4096, 2) for r in (0, 1)}
    banks = {
        r: CPUSparseWorkingSet(
            request_id=request,
            incarnation="inc",
            entry_transfer_id="entry",
            layout_fingerprint=f"layout-{r}",
            expected_groups=((0, r),),
            prompt_tokens=4,
            head_dim=2,
            max_union_tokens=2,
            budget=budgets[r],
        )
        for r in (0, 1)
    }
    return CPUInstallGroup(banks, interval=4, lead_tokens=1), banks, budgets


def payload(epoch, rank):
    tokens = (0, 1, 2, 3) if epoch.target_tokens == 0 else (1, 3)
    spec = SparseKVSpec(
        epoch.request_id,
        epoch.incarnation,
        epoch.operation_id,
        epoch.target_tokens,
        epoch.entry_transfer_id,
        "index",
        "map",
        f"layout-{rank}",
        0,
        rank,
        tokens,
    )
    return SparseKVPayload(
        spec, torch.full((2, len(tokens), 2), float(epoch.round + rank))
    )


def stage_all(group, epoch):
    for rank in (0, 1):
        p = payload(epoch, rank)
        try:
            group.stage(epoch, rank, [p])
        finally:
            p.close()


def test_cpu_banks_wait_for_reader_then_install_the_same_epoch():
    group, _, budgets = cpu_group()
    initial = group.begin(0)
    group.stage(initial, 0, [payload(initial, 0)])
    assert not group.try_install(initial, {0: 0, 1: 0})
    with pytest.raises(InstallProtocolError, match="wait or abort"), group.read(0, 0):
        pytest.fail("partial initial bank was exposed")
    group.stage(initial, 1, [payload(initial, 1)])
    assert group.try_install(initial, {0: 0, 1: 0})
    epoch = group.begin(3)
    stage_all(group, epoch)
    with group.read(0, 3) as old:
        assert not group.try_install(epoch, {0: 4, 1: 4})
        assert old[(0, 0)][0].target_tokens == 0
        assert group.coordinator.snapshot()["applied"] == ()
    assert group.try_install(epoch, {0: 4, 1: 4})
    for rank in (0, 1):
        with group.read(rank, 4) as current:
            assert current[(0, rank)][0].operation_id == epoch.operation_id
            assert current[(0, rank)][0].token_ids == (1, 3)
    group.close()
    assert all(b.snapshot()["used_staging_bytes"] == 0 for b in budgets.values())


@pytest.mark.parametrize("when", ["before_apply", "after_apply", "cancel"])
def test_cpu_partial_failure_blocks_all_reads_and_releases_on_explicit_close(
    monkeypatch, when
):
    group, banks, budgets = cpu_group()
    initial = group.begin(0)
    stage_all(group, initial)
    group.try_install(initial, {0: 0, 1: 0})
    epoch = group.begin(3)
    stage_all(group, epoch)
    original = banks[1].install

    def fail(count, *, candidate):
        assert not group.coordinator.can_decode(4)
        with pytest.raises(InstallProtocolError), group.read(0, 4):
            pytest.fail("rank 0's new bank exposed before rank 1 ACK")
        if when == "after_apply":
            original(count, candidate=candidate)
        if when == "cancel":
            group.coordinator.cancel("cancel during install")
        raise RuntimeError("injected rank 1 failure")

    monkeypatch.setattr(banks[1], "install", fail)
    with pytest.raises(RuntimeError, match="rank 1 failure"):
        group.try_install(epoch, {0: 4, 1: 4})
    assert group.coordinator.snapshot()["installed_tokens"] == 0
    for rank in (0, 1):
        with pytest.raises(InstallProtocolError), group.read(rank, 4):
            pytest.fail("failed request resumed")
    assert any(b.snapshot()["used_staging_bytes"] > 0 for b in budgets.values())
    group.close()
    assert all(b.snapshot()["used_staging_bytes"] == 0 for b in budgets.values())


def test_candidate_replacement_is_detected_before_any_rank_installs():
    group, banks, _ = cpu_group()
    epoch = group.begin(0)
    stage_all(group, epoch)
    # Deliberate violation of driver's exclusive ownership, simulate replacement.
    banks[1].discard_next()
    banks[1].stage([payload(epoch, 1)])
    with pytest.raises(SparsePayloadError, match="stale or foreign"):
        group.try_install(epoch, {0: 0, 1: 0})
    assert group.coordinator.snapshot()["applied"] == ()
    group.close()


def test_cpu_close_retains_a_live_reader_until_drain():
    group, _, budgets = cpu_group()
    epoch = group.begin(0)
    stage_all(group, epoch)
    group.try_install(epoch, {0: 0, 1: 0})
    with group.read(0, 0):
        with pytest.raises(InstallProtocolError, match="readers must drain"):
            group.close()
        assert budgets[0].snapshot()["used_staging_bytes"] > 0
        assert not group.coordinator.can_decode(0)
    group.close()
    assert all(b.snapshot()["used_staging_bytes"] == 0 for b in budgets.values())


def test_disagreeing_boundary_does_not_install_any_rank():
    group, _, _ = cpu_group()
    epoch = group.begin(0)
    stage_all(group, epoch)
    with pytest.raises(InstallProtocolError, match="same boundary"):
        group.try_install(epoch, {0: 0, 1: 1})
    assert group.coordinator.snapshot()["applied"] == ()
    assert group.try_install(epoch, {0: 0, 1: 0})
    group.close()


@pytest.mark.parametrize("field", ["round", "target_tokens"])
def test_boolean_counts_cannot_alias_valid_epoch(field):
    c = protocol()
    epoch = c.begin(0)
    with pytest.raises(InstallProtocolError, match="invalid installation epoch"):
        replace(epoch, **{field: False})
    with pytest.raises(InstallProtocolError, match="invalid rank receipt"):
        replace(receipt(epoch, 0), rank=False)


def test_wrong_staging_ack_and_old_failure_do_not_advance_or_kill_new_round():
    c = protocol()
    old = c.begin(0)
    complete(c, old)
    epoch = c.begin(3)
    for rank in (0, 1):
        c.prepared(receipt(epoch, rank))
        c.parked(receipt(epoch, rank), 4)
    assert c.decide_install(epoch)
    with pytest.raises(InstallProtocolError, match="differs from prepared"):
        c.applied(replace(receipt(epoch, 0), staging_id="another-bank"))
    with pytest.raises(InstallProtocolError, match="stale or foreign"):
        c.fail(old, "late timeout")
    assert c.snapshot()["applied"] == ()
    assert c.snapshot()["state"] == "installing"
    c.applied(receipt(epoch, 0))
    c.applied(receipt(epoch, 1))
    assert c.can_decode(4)


def test_cancel_before_prepare_keeps_late_payload_out_of_banks():
    group, _, budgets = cpu_group()
    epoch = group.begin(0)
    group.coordinator.cancel()
    with pytest.raises(InstallProtocolError, match="terminal"):
        group.stage(epoch, 0, [payload(epoch, 0)])
    assert all(b.snapshot()["used_staging_bytes"] == 0 for b in budgets.values())
    group.close()


def test_wrong_payload_operation_does_not_reserve_a_bank():
    group, _, budgets = cpu_group()
    epoch = group.begin(0)
    p = payload(replace(epoch, operation_id="wrong"), 0)
    with pytest.raises(InstallProtocolError, match="not for the active epoch"):
        group.stage(epoch, 0, [p])
    assert budgets[0].snapshot()["used_staging_bytes"] == 0
    assert group.coordinator.snapshot()["prepared"] == ()
    group.close()


def test_new_cpu_request_does_not_touch_old_staged_buffers_or_clock():
    old, banks, budgets = cpu_group("old")
    epoch = old.begin(0)
    stage_all(old, epoch)
    old.try_install(epoch, {0: 0, 1: 0})
    pending = old.begin(3)
    stage_all(old, pending)
    before = old.coordinator.snapshot()
    stamps = {r: bank.install_candidate() for r, bank in banks.items()}
    charges = {r: budget.snapshot() for r, budget in budgets.items()}
    new, _, _ = cpu_group("new")
    initial = new.begin(0)
    stage_all(new, initial)
    new.try_install(initial, {0: 0, 1: 0})
    new.close()
    assert old.coordinator.snapshot() == before
    assert {r: bank.install_candidate() for r, bank in banks.items()} == stamps
    assert {r: budget.snapshot() for r, budget in budgets.items()} == charges
    assert old.try_install(pending, {0: 4, 1: 4})
    old.close()
