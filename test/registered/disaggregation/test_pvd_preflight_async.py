"""Async GPUDirect preflight must wait for completion and retain unknown MRs."""

import weakref

import pytest
import torch
from sglang.srt.disaggregation.pvd import preflight
from sglang.srt.disaggregation.pvd.transfer_engine import (
    FakeTransferEngine,
    TransferStatus,
)


class DelayedEngine(FakeTransferEngine):
    def __init__(self, pending_polls):
        super().__init__()
        self.pending_polls = pending_polls
        self.registrations = []
        self.releases = []

    def register_memory(self, *args, **kwargs):
        registration = super().register_memory(*args, **kwargs)
        self.registrations.append(registration)
        return registration

    def release_memory(self, registration):
        self.releases.append(registration)
        super().release_memory(registration)

    def poll(self, handle):
        if self.pending_polls:
            self.pending_polls -= 1
            return TransferStatus.PENDING
        return super().poll(handle)


def test_preflight_waits_for_async_terminal_success(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    engine = DelayedEngine(3)
    report = preflight.run_rank_preflight(
        rank=0,
        rails=("mlx5_0",),
        device="cpu",
        engine=engine,
        strict=False,
        transfer_timeout_seconds=0.5,
    )
    assert report.local_gpu_transfer
    assert len(engine.releases) == 2


def test_preflight_timeout_retains_both_unknown_regions(monkeypatch):
    before = preflight.unknown_preflight_owner_count()
    engine = DelayedEngine(1000)
    with pytest.raises(preflight.PVDPreflightError, match="restart the process"):
        preflight.run_rank_preflight(
            rank=0,
            rails=("mlx5_0",),
            device="cpu",
            engine=engine,
            strict=False,
            transfer_timeout_seconds=0.001,
        )
    assert len(engine.registrations) == 2
    assert engine.releases == []
    assert preflight.unknown_preflight_owner_count() == before + 1
    assert preflight._unknown_preflight_owners[-1][0] is engine
    # This fake engine has no in-flight native transfer; clean its global test
    # registry only after asserting the production path retained both MRs.
    for registration in engine.registrations:
        engine.release_memory(registration)


def test_unknown_owner_survives_caller_dropping_engine():
    engine = DelayedEngine(1000)
    retained = weakref.ref(engine)
    with pytest.raises(preflight.PVDPreflightError, match="restart the process"):
        preflight.run_rank_preflight(
            rank=0,
            rails=("mlx5_0",),
            device="cpu",
            engine=engine,
            strict=False,
            transfer_timeout_seconds=0.001,
        )
    del engine
    assert retained() is preflight._unknown_preflight_owners[-1][0]


@pytest.mark.parametrize("poll_failure", ("exception", "cancelled"))
def test_preflight_unknown_poll_retains_both_regions(poll_failure):
    class UnknownEngine(DelayedEngine):
        def poll(self, handle):
            if poll_failure == "exception":
                raise RuntimeError("native QP state unknown")
            return TransferStatus.CANCELLED

    engine = UnknownEngine(0)
    with pytest.raises(preflight.PVDPreflightError, match="restart the process"):
        preflight.run_rank_preflight(
            rank=0,
            rails=("mlx5_0",),
            device="cpu",
            engine=engine,
            strict=False,
        )
    assert len(engine.registrations) == 2
    assert engine.releases == []
    for registration in engine.registrations:
        engine.release_memory(registration)
