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
