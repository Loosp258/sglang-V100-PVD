"""Native two-HCA startup policy, with the hardware boundary replaced."""

import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from sglang.srt.disaggregation.pvd import multi_rail_receive, preflight
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


class NativeAdapter(FakeTransferEngine):
    constructed: ClassVar[list] = []

    def __init__(self, *, hostname, gpu_id, rail, budget):
        super().__init__()
        self.rail = rail
        self._engine = SimpleNamespace(
            gpu_id=gpu_id, get_ib_device=lambda: rail, hostname=hostname
        )
        self.lifecycle_manager = SimpleNamespace(budget=budget)
        self.constructed.append(self)


@pytest.fixture
def native_boundary(monkeypatch):
    NativeAdapter.constructed = []
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.pvd.mooncake_engine",
        SimpleNamespace(MooncakePVDTransferEngine=NativeAdapter),
    )
    monkeypatch.setattr(multi_rail_receive.platform, "system", lambda: "Linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(multi_rail_receive.Path, "is_dir", lambda self: True)
    monkeypatch.setattr(preflight, "_has_active_port", lambda path: True)
    monkeypatch.setattr(
        preflight,
        "run_rank_preflight",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.delenv("MC_FORCE_TCP", raising=False)
    monkeypatch.delenv("MOONCAKE_PROTOCOL", raising=False)
    return calls


def test_native_group_uses_existing_and_new_session_on_same_gpu(native_boundary):
    budget = TransferBudget(staging_bytes=4096, max_inflight=2)
    existing = NativeAdapter(hostname="D", gpu_id=0, rail="mlx5_2", budget=budget)
    group = multi_rail_receive.create_native_receive_group(
        hostname="D",
        gpu_id=0,
        rails=("mlx5_2", "mlx5_3"),
        transfer_budget=budget,
        existing_adapter=existing,
    )
    assert group.adapters["mlx5_2"] is existing
    assert group.adapters["mlx5_3"] is not existing
    assert [item["rank"] for item in native_boundary] == [0, 1]
    assert [item["device"] for item in native_boundary] == ["cuda:0"] * 2
    assert all(item["strict"] for item in native_boundary)
    assert len(NativeAdapter.constructed) == 2


@pytest.mark.parametrize(
    "bad",
    (
        {"rails": ("mlx5_2", "mlx5_2")},
        {"rails": ("mlx5_2", "bad/rail")},
        {"gpu_id": 2},
    ),
)
def test_native_group_rejects_bad_topology_before_initializing(native_boundary, bad):
    kwargs = {
        "hostname": "D",
        "gpu_id": 0,
        "rails": ("mlx5_2", "mlx5_3"),
        "transfer_budget": TransferBudget(staging_bytes=4096, max_inflight=2),
    }
    kwargs.update(bad)
    with pytest.raises((ValueError, preflight.PVDPreflightError)):
        multi_rail_receive.create_native_receive_group(**kwargs)
    assert not NativeAdapter.constructed
    assert not native_boundary


def test_native_group_rejects_tcp_before_initializing(native_boundary, monkeypatch):
    monkeypatch.setenv("MC_FORCE_TCP", "1")
    with pytest.raises(ValueError, match="Linux CUDA RDMA"):
        multi_rail_receive.create_native_receive_group(
            hostname="D",
            gpu_id=0,
            rails=("mlx5_2", "mlx5_3"),
            transfer_budget=TransferBudget(staging_bytes=4096, max_inflight=2),
        )
    assert not NativeAdapter.constructed


def test_native_group_refuses_foreign_existing_adapter(native_boundary):
    budget = TransferBudget(staging_bytes=4096, max_inflight=2)
    existing = NativeAdapter(hostname="D", gpu_id=1, rail="mlx5_2", budget=budget)
    with pytest.raises(ValueError, match="different GPU"):
        multi_rail_receive.create_native_receive_group(
            hostname="D",
            gpu_id=0,
            rails=("mlx5_2", "mlx5_3"),
            transfer_budget=budget,
            existing_adapter=existing,
        )
    assert len(NativeAdapter.constructed) == 1
    assert not native_boundary


def test_native_group_rejects_wrong_selected_hca(native_boundary, monkeypatch):
    original = NativeAdapter.__init__

    def choose_wrong_hca(self, **kwargs):
        original(self, **kwargs)
        self._engine.get_ib_device = lambda: "mlx5_wrong"

    monkeypatch.setattr(NativeAdapter, "__init__", choose_wrong_hca)
    with pytest.raises(ValueError, match="did not select required HCA"):
        multi_rail_receive.create_native_receive_group(
            hostname="D",
            gpu_id=0,
            rails=("mlx5_2", "mlx5_3"),
            transfer_budget=TransferBudget(staging_bytes=4096, max_inflight=2),
        )
    assert not native_boundary


def test_native_group_requires_every_rail_active(native_boundary, monkeypatch):
    monkeypatch.setattr(
        preflight,
        "_has_active_port",
        lambda path: path.name == "mlx5_2",
    )
    with pytest.raises(ValueError, match="mlx5_3 has no ACTIVE port"):
        multi_rail_receive.create_native_receive_group(
            hostname="D",
            gpu_id=0,
            rails=("mlx5_2", "mlx5_3"),
            transfer_budget=TransferBudget(staging_bytes=4096, max_inflight=2),
        )
    assert not NativeAdapter.constructed
