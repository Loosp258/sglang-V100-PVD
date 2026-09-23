"""A D GPU's two HCA destinations keep separate registration owners."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from sglang.srt.disaggregation.pvd.multi_rail_receive import (
    MultiRailReceiveError,
    RailMappedReceiveEngine,
)
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine


class RailAdapter(FakeTransferEngine):
    def __init__(self, rail):
        super().__init__()
        self.rail = rail
        self.release_attempts = 0
        self.fail_release = False

    def release_memory(self, registration):
        self.release_attempts += 1
        if self.fail_release:
            raise RuntimeError("native unregister not complete")
        super().release_memory(registration)


class UnhealthyRailAdapter(RailAdapter):
    def health(self):
        return {**super().health(), "healthy": False}


def test_two_rails_register_and_release_through_exact_native_owner():
    engines = {rail: RailAdapter(rail) for rail in ("mlx5_0", "mlx5_1")}
    receiver = RailMappedReceiveEngine(engines)
    records = {
        rail: receiver.register_memory(
            torch.empty(32, dtype=torch.uint8),
            endpoint="D",
            rank=rank,
            rail=rail,
        )
        for rank, rail in enumerate(engines)
    }
    assert {r.descriptor.rail for r in records.values()} == set(engines)
    assert receiver.health()["registered_destinations"] == 2
    with pytest.raises(MultiRailReceiveError, match="unconfigured"):
        receiver.register_memory(
            torch.empty(8, dtype=torch.uint8),
            endpoint="D",
            rank=0,
            rail="mlx5_2",
        )
    with pytest.raises(MultiRailReceiveError, match="cannot submit"):
        receiver.submit_put(None, None)
    for rail, registration in records.items():
        receiver.release_memory(registration)
        assert engines[rail].release_attempts == 1
    assert receiver.health()["registered_destinations"] == 0
    with pytest.raises(MultiRailReceiveError, match="already retired"):
        receiver.release_memory(records["mlx5_0"])


def test_failing_child_unregister_keeps_exact_registration_for_retry():
    engine = RailAdapter("mlx5_1")
    receiver = RailMappedReceiveEngine({"mlx5_1": engine})
    registration = receiver.register_memory(
        torch.empty(16, dtype=torch.uint8),
        endpoint="D",
        rank=1,
        rail="mlx5_1",
    )
    engine.fail_release = True
    with pytest.raises(RuntimeError, match="native unregister"):
        receiver.release_memory(registration)
    assert receiver.health()["registered_destinations"] == 1
    engine.fail_release = False
    receiver.release_memory(registration)
    assert engine.release_attempts == 2
    assert receiver.health()["registered_destinations"] == 0


def test_reused_adapter_or_mislabelled_rail_is_refused():
    engine = RailAdapter("mlx5_0")
    with pytest.raises(MultiRailReceiveError, match="distinct explicit"):
        RailMappedReceiveEngine({"mlx5_0": engine, "mlx5_1": engine})
    with pytest.raises(MultiRailReceiveError, match="distinct explicit"):
        RailMappedReceiveEngine({"mlx5_1": engine})


def test_startup_thread_can_hand_off_receive_adapter_to_control_thread():
    receiver = RailMappedReceiveEngine({"mlx5_0": RailAdapter("mlx5_0")})

    def control_turn():
        registration = receiver.register_memory(
            torch.empty(16, dtype=torch.uint8),
            endpoint="D",
            rank=0,
            rail="mlx5_0",
        )
        receiver.release_memory(registration)
        return receiver.health()["registered_destinations"]

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(control_turn).result() == 0


def test_unhealthy_child_blocks_new_receive_registration():
    receiver = RailMappedReceiveEngine(
        {
            "mlx5_0": RailAdapter("mlx5_0"),
            "mlx5_1": UnhealthyRailAdapter("mlx5_1"),
        }
    )
    health = receiver.health()
    assert health["healthy"] is False
    assert health["rails"]["mlx5_0"]["healthy"] is True
    assert health["rails"]["mlx5_1"]["healthy"] is False
    with pytest.raises(MultiRailReceiveError, match="unhealthy"):
        receiver.register_memory(
            torch.empty(16, dtype=torch.uint8),
            endpoint="D",
            rank=0,
            rail="mlx5_0",
        )
    assert receiver.health()["registered_destinations"] == 0
