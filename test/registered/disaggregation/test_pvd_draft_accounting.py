"""Shared draft budgets and bounded metadata, with explicit runner doubles."""

import pytest
from sglang.srt.disaggregation.pvd.draft_sglang import (
    DraftPlacement,
    DraftWorkerError,
)
from sglang.srt.disaggregation.pvd.prediction import PredictionConfigError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferBudget,
    TransferCapacityError,
)
from test_pvd_draft_sglang import factory, provider


def test_independent_providers_do_not_alias_persistent_budget_owners():
    budget = TransferBudget(2048, 1)
    first = provider(factory(persistent_bytes=1024), persistent_budget=budget)
    second = provider(factory(persistent_bytes=1024), persistent_budget=budget)
    assert first.factory is not second.factory
    assert budget.snapshot()["used_staging_bytes"] == 2048
    assert budget.snapshot()["reservations"] == 2


def test_second_provider_cannot_bypass_shared_capacity_with_same_byte_count():
    budget = TransferBudget(1024, 1)
    first = provider(factory(persistent_bytes=1024), persistent_budget=budget)
    with pytest.raises(TransferCapacityError):
        provider(factory(persistent_bytes=1024), persistent_budget=budget)
    assert first.persistent_budget.snapshot()["used_staging_bytes"] == 1024


def test_positive_persistent_bytes_require_an_explicit_budget():
    with pytest.raises(PredictionConfigError, match="persistent"):
        provider(factory(), placement=DraftPlacement(scratch_budget_bytes=4096))


def test_explicit_shared_budget_is_charged_even_without_a_local_budget_setting():
    budget = TransferBudget(2048, 1)
    provider(
        factory(),
        placement=DraftPlacement(scratch_budget_bytes=4096),
        persistent_budget=budget,
    )
    assert budget.snapshot()["used_staging_bytes"] == 1024


def test_persistent_and_scratch_cannot_share_one_accounting_budget():
    budget = TransferBudget(8192, 1)
    with pytest.raises(PredictionConfigError, match="separate"):
        provider(factory(), persistent_budget=budget, scratch_budget=budget)
    assert budget.snapshot()["used_staging_bytes"] == 0


@pytest.mark.parametrize("declared", [True, 1.5, "1024", -1])
def test_factory_persistent_size_is_not_silently_coerced(declared):
    fac = factory()
    fac.persistent_bytes = lambda: declared
    budget = TransferBudget(4096, 1)
    with pytest.raises(PredictionConfigError, match="non-negative integer"):
        provider(fac, persistent_budget=budget)
    assert budget.snapshot()["reservations"] == 0


def test_external_capacity_does_not_override_explicit_per_provider_bound():
    budget = TransferBudget(4096, 1)
    with pytest.raises(TransferCapacityError):
        provider(
            factory(),
            persistent_budget=budget,
            placement=DraftPlacement(
                scratch_budget_bytes=4096, persistent_budget_bytes=512
            ),
        )
    assert budget.snapshot()["reservations"] == 0


def test_factory_diagnostics_do_not_grow_for_every_prediction_round():
    fac = factory()
    made = provider(fac)
    for _ in range(100):
        with made.branch():
            pass
    assert len(fac.opened) <= 64
    assert fac.opened_count == 100
    assert made.scratch_budget.snapshot()["used_staging_bytes"] == 0


@pytest.mark.parametrize(
    ("scratch", "error"),
    [
        (16, TransferCapacityError),
        (True, PredictionConfigError),
        (1.5, PredictionConfigError),
        (-1, PredictionConfigError),
        (RuntimeError("sizing failed"), RuntimeError),
    ],
)
def test_failed_branch_admission_releases_opened_handle_before_returning_slot(
    scratch, error
):
    events = []
    fac = factory()
    made = provider(
        fac,
        placement=DraftPlacement(
            scratch_budget_bytes=8, persistent_budget_bytes=4096
        ),
    )

    class OpenedHandle:
        def scratch_bytes(self):
            if isinstance(scratch, Exception):
                raise scratch
            return scratch

        def release(self):
            events.append(
                (
                    "release",
                    made.active_branches,
                    made.scratch_budget.snapshot()["used_staging_bytes"],
                )
            )

    fac.open = lambda **_: OpenedHandle()
    with pytest.raises(error):
        with made.branch():
            pytest.fail("an unreserved handle must not enter the branch")
    assert events == [("release", 1, 0)]
    assert made.active_branches == 0
    assert not made.degraded
    assert made.scratch_budget.snapshot()["reservations"] == 0


def test_failed_opened_handle_cleanup_quarantines_provider():
    fac = factory()
    made = provider(
        fac,
        placement=DraftPlacement(
            scratch_budget_bytes=8, persistent_budget_bytes=4096
        ),
    )

    class FailingRelease:
        def scratch_bytes(self):
            return 16

        def release(self):
            assert made.active_branches == 1
            raise RuntimeError("completion unknown")

    handle = FailingRelease()
    fac.open = lambda **_: handle
    with pytest.raises(DraftWorkerError, match="quarantined"):
        with made.branch():
            pytest.fail("an unreserved handle must not enter the branch")
    assert made.degraded
    assert made.active_branches == 0
    assert len(made.quarantined) == 1
    assert made._retained[made.quarantined[0].branch_id].handle is handle
    with pytest.raises(TransferCapacityError, match="quarantined"):
        with made.branch():
            pytest.fail("a quarantined provider must not issue another handle")
