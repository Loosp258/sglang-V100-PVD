"""D resolves only the Gateway-selected Entry and preflighted V rails."""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from sglang.srt.disaggregation.pvd.client import (
    PVDSelectedShardRoute,
    PVDSelectedShardRoutes,
)
from sglang.srt.disaggregation.pvd.conn import PVDConnectionError, PVDKVManager
from sglang.srt.disaggregation.pvd.multi_rail_receive import RailMappedReceiveEngine
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine


class DeferredControl:
    def __init__(self):
        self.coroutine = None

    def submit(self, coroutine):
        self.coroutine = coroutine
        future = Future()
        try:
            future.set_result(asyncio.run(coroutine))
        except BaseException as exc:
            future.set_exception(exc)
        return future


class SelectedClient:
    def __init__(self, selected):
        self.selected = selected
        self.keys = []

    async def selected_shard_routes(self, key):
        self.keys.append(key)
        return self.selected


def manager_for(rails):
    manager = object.__new__(PVDKVManager)
    manager.model_instance_id = "model"
    manager.sparse_receive_registry = object()
    manager.sparse_receive_engine = RailMappedReceiveEngine(
        {rail: FakeTransferEngine() for rail in rails}
    )
    manager.control = DeferredControl()
    req = SimpleNamespace(pvd_transfer_id="entry", pvd_vector_group_id="chosen")
    key = manager.key_for(req)
    selected = PVDSelectedShardRoutes(
        SimpleNamespace(key=key),
        (
            PVDSelectedShardRoute(0, "http://v0", "epoch0", "mlx5_0"),
            PVDSelectedShardRoute(1, "http://v1", "epoch1", "mlx5_1"),
        ),
    )
    client = SelectedClient(selected)
    manager.clients = {"chosen": client}
    return manager, req, selected, client


def test_selected_entry_and_rails_are_checked_before_request_allocation():
    manager, req, selected, client = manager_for(("mlx5_0", "mlx5_1"))
    assert manager.start_selected_cuda_routes(req).result() is selected
    assert client.keys == [manager.key_for(req)]


def test_unconfigured_v_hca_or_entry_change_fails_closed():
    manager, req, selected, client = manager_for(("mlx5_0",))
    with pytest.raises(PVDConnectionError, match="no preflighted receive adapter"):
        manager.start_selected_cuda_routes(req).result()
    assert len(client.keys) == 1

    manager.sparse_receive_engine = RailMappedReceiveEngine(
        {rail: FakeTransferEngine() for rail in ("mlx5_0", "mlx5_1")}
    )
    client.selected = PVDSelectedShardRoutes(
        SimpleNamespace(key=SimpleNamespace(transfer_id="other")),
        selected.shards,
    )
    with pytest.raises(PVDConnectionError, match="identity changed"):
        manager.start_selected_cuda_routes(req).result()


def test_uninitialized_sparse_receiver_cannot_discover_routes():
    manager, req, _, client = manager_for(("mlx5_0", "mlx5_1"))
    manager.sparse_receive_registry = None
    with pytest.raises(PVDConnectionError, match="not initialized"):
        manager.start_selected_cuda_routes(req)
    assert not client.keys


def test_selected_request_factory_uses_only_manager_owned_d_resources(monkeypatch):
    from sglang.srt.disaggregation.pvd import cuda_routed_request

    manager, req, selected, _ = manager_for(("mlx5_0", "mlx5_1"))
    registry = SimpleNamespace(
        engine=manager.sparse_receive_engine, _owner=lambda: None
    )
    manager.sparse_receive_registry = registry
    manager.transfer_engine = SimpleNamespace(
        health=lambda: {"healthy": True, "session_id": "D0"}
    )
    manager.tp_size = 1
    manager.tp_rank = 0
    manager.rail = "mlx5_0"
    layout = object()
    manager.layout = lambda: layout
    calls = []
    assembly = object()
    monkeypatch.setattr(
        cuda_routed_request,
        "assemble_routed_cuda_request",
        lambda *args, **kwargs: calls.append((args, kwargs)) or assembly,
    )
    supplied = {
        "group": object(),
        "pipeline": object(),
        "head_mapping": object(),
        "vector_space": "model-space",
        "metric": "l2",
        "top_k": 4,
        "max_union_tokens": 16,
        "max_head_dim": 128,
        "copy_budget": object(),
        "aggregate_budget": object(),
        "poll_interval_seconds": 0.01,
    }
    assert manager.assemble_selected_cuda_request(req, selected, **supplied) is assembly
    assert calls == [
        (
            (selected,),
            {
                **supplied,
                "compute_layout": layout,
                "compute_rank": 0,
                "registry": registry,
                "d_endpoint": "D0",
                "d_rail": "mlx5_0",
                "d_rails": {0: "mlx5_0", 1: "mlx5_1"},
            },
        )
    ]
    manager.sparse_receive_registry = None
    with pytest.raises(PVDConnectionError, match="not initialized"):
        manager.assemble_selected_cuda_request(req, selected, **supplied)
    manager.sparse_receive_registry = SimpleNamespace(
        engine=object(), _owner=lambda: None
    )
    with pytest.raises(PVDConnectionError, match="not initialized"):
        manager.assemble_selected_cuda_request(req, selected, **supplied)
    assert len(calls) == 1


def test_unhealthy_d_transport_blocks_route_discovery_and_assembly(monkeypatch):
    from sglang.srt.disaggregation.pvd import cuda_routed_request

    class UnhealthyFake(FakeTransferEngine):
        def health(self):
            return {**super().health(), "healthy": False}

    manager, req, selected, _ = manager_for(("mlx5_0", "mlx5_1"))
    manager.sparse_receive_engine = RailMappedReceiveEngine(
        {"mlx5_0": FakeTransferEngine(), "mlx5_1": UnhealthyFake()}
    )
    with pytest.raises(PVDConnectionError, match="sparse receive transport is unhealthy"):
        manager.start_selected_cuda_routes(req).result()

    manager.sparse_receive_registry = SimpleNamespace(
        engine=manager.sparse_receive_engine, _owner=lambda: None
    )
    manager.transfer_engine = SimpleNamespace(
        health=lambda: {"healthy": True, "session_id": "D0"}
    )
    manager.tp_size = 1
    manager.tp_rank = 0
    manager.rail = "mlx5_0"
    calls = []
    monkeypatch.setattr(
        cuda_routed_request,
        "assemble_routed_cuda_request",
        lambda *args, **kwargs: calls.append(1),
    )
    supplied = {
        "group": object(),
        "pipeline": object(),
        "head_mapping": object(),
        "vector_space": "space",
        "metric": "l2",
        "top_k": 4,
        "max_union_tokens": 16,
        "max_head_dim": 128,
        "copy_budget": object(),
        "aggregate_budget": object(),
        "poll_interval_seconds": 0.01,
    }
    with pytest.raises(PVDConnectionError, match="sparse receive transport is unhealthy"):
        manager.assemble_selected_cuda_request(req, selected, **supplied)
    assert not calls

    manager.sparse_receive_engine = RailMappedReceiveEngine(
        {"mlx5_0": FakeTransferEngine(), "mlx5_1": FakeTransferEngine()}
    )
    manager.sparse_receive_registry.engine = manager.sparse_receive_engine
    manager.transfer_engine = SimpleNamespace(
        health=lambda: {"healthy": False, "session_id": "D0"}
    )
    with pytest.raises(PVDConnectionError, match="compute transport is unhealthy"):
        manager.assemble_selected_cuda_request(req, selected, **supplied)
    assert not calls
