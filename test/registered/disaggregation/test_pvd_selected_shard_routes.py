"""Selected V shard discovery is bound to the stored Entry, not caller hints."""

import asyncio

import pytest
from aiohttp.test_utils import TestServer
from sglang.srt.disaggregation.pvd.client import (
    PVDControlPlaneError,
    PVDCoordinatorClient,
)
from sglang.srt.disaggregation.pvd.control_server import create_coordinator_app
from sglang.srt.disaggregation.pvd.coordinator import (
    CoordinatorError,
    LocalShardClient,
    VectorCoordinator,
)
from test_pvd3 import make_ready_entry, make_vector


def test_selected_entry_discovers_exact_live_shard_routes_over_http():
    async def run():
        engine, stores, _ = make_vector()
        coordinator = VectorCoordinator(
            [
                LocalShardClient(store, shard_url=f"http://v{rank}.example:920{rank}")
                for rank, store in enumerate(stores)
            ]
        )
        key = await make_ready_entry(coordinator, engine, "selected")
        async with TestServer(create_coordinator_app(coordinator)) as server:
            client = PVDCoordinatorClient(str(server.make_url("")))
            try:
                found = await client.selected_shard_routes(key)
            finally:
                await client.close()
        assert found.manifest.key == key
        assert found.manifest.prompt_token_count == 5
        assert tuple(route.rank for route in found.shards) == (0, 1)
        assert tuple(route.url for route in found.shards) == (
            "http://v0.example:9200",
            "http://v1.example:9201",
        )
        assert all(
            route.sender_epoch == stores[route.rank].worker_epoch
            and route.rail == found.manifest.shard(route.rank).rail
            for route in found.shards
        )

    asyncio.run(run())


def test_unadvertised_or_unready_shard_never_yields_route(monkeypatch):
    async def run():
        engine, stores, coordinator = make_vector()
        key = await make_ready_entry(coordinator, engine, "selected")
        with pytest.raises(CoordinatorError, match="advertised"):
            await coordinator.selected_shard_routes(key)
        engine, stores, _ = make_vector()
        coordinator = VectorCoordinator(
            [
                LocalShardClient(store, shard_url=f"http://v{rank}:920{rank}")
                for rank, store in enumerate(stores)
            ]
        )
        with pytest.raises(CoordinatorError, match="not stored"):
            await coordinator.selected_shard_routes(key)
        await make_ready_entry(coordinator, engine, "selected")
        original = stores[1].snapshot

        def stale():
            return {**original(), "ready": False}

        monkeypatch.setattr(stores[1], "snapshot", stale)
        with pytest.raises(CoordinatorError, match="stale or not ready"):
            await coordinator.selected_shard_routes(key)

    asyncio.run(run())


def test_coordinator_client_rejects_wrong_entry_duplicate_or_changed_rail(monkeypatch):
    async def run():
        engine, stores, _ = make_vector()
        coordinator = VectorCoordinator(
            [
                LocalShardClient(store, shard_url=f"http://v{rank}:920{rank}")
                for rank, store in enumerate(stores)
            ]
        )
        key = await make_ready_entry(coordinator, engine, "selected")
        base_reply = await coordinator.selected_shard_routes(key)
        client = PVDCoordinatorClient("http://v-coordinator")

        async def bad_request(_path, _payload):
            return reply

        monkeypatch.setattr(client, "_request", bad_request)
        for altered in (
            {**base_reply, "shards": [base_reply["shards"][0]] * 2},
            {
                **base_reply,
                "shards": [
                    base_reply["shards"][0],
                    {**base_reply["shards"][1], "rail": "bad"},
                ],
            },
            {
                **base_reply,
                "manifest": {
                    **base_reply["manifest"],
                    "key": {**key.to_dict(), "transfer_id": "other"},
                },
            },
        ):
            reply = altered
            with pytest.raises(PVDControlPlaneError, match="invalid selected shard"):
                await client.selected_shard_routes(key)

    asyncio.run(run())
