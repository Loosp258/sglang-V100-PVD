"""P-to-D initial KV handshake and native completion proof."""

import asyncio
import types
import unittest
from unittest import mock

import torch
from aiohttp import web

from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer
from sglang.srt.disaggregation.pvd.direct_bootstrap import (
    DirectBootstrapClient,
    publish_and_send_direct,
)
from sglang.srt.disaggregation.pvd.protocol import (
    PVD_GENERATION_METADATA_KEY,
    PVD_RECEIVER_EPOCH_METADATA_KEY,
)
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine


class DirectBootstrapTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        server = object.__new__(CommonKVBootstrapServer)
        server.app = web.Application()
        server.lock = asyncio.Lock()
        server.pvd_direct_entries = {}
        server._setup_routes()
        self.runner = web.AppRunner(server.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.client = DirectBootstrapClient("127.0.0.1", port, poll_interval=0.001)

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def test_full_kv_goes_directly_to_d_with_terminal_proof(self):
        engine = FakeTransferEngine()
        source = torch.arange(128, dtype=torch.uint8)
        target = torch.zeros_like(source)
        generation, receiver_epoch = "generation-d", "epoch-d"
        registration = engine.register_memory(
            target,
            endpoint="pvd-decode",
            rank=0,
            rail="mlx5_0",
            metadata={
                PVD_GENERATION_METADATA_KEY: generation,
                PVD_RECEIVER_EPOCH_METADATA_KEY: receiver_epoch,
            },
        )
        identity = {"transfer_id": "transfer-a", "delivery_id": "delivery-a"}
        task = asyncio.create_task(
            publish_and_send_direct(
                client=self.client,
                key=types.SimpleNamespace(transfer_id="transfer-a"),
                delivery_id="delivery-a",
                manifest=types.SimpleNamespace(to_dict=lambda: {"prompt": 128}),
                first_token=types.SimpleNamespace(to_dict=lambda: {"output_token_id": 5}),
                packed=source,
                engine=engine,
                sender_epoch="epoch-p",
                rail="mlx5_0",
            )
        )
        entry = await self.client.wait("entry", identity)
        self.assertEqual(entry["expected_bytes"], 128)
        await self.client.post(
            "destination",
            {
                **identity,
                "destination": registration.descriptor.to_dict(),
                "receiver_epoch": receiver_epoch,
                "generation": generation,
            },
        )
        proof = await self.client.wait("terminal", identity)
        self.assertEqual(proof["state"], "terminal_success")
        self.assertEqual(proof["transferred_bytes"], 128)
        await self.client.post("ack", identity)
        await task
        self.assertTrue(torch.equal(source, target))
        self.assertEqual(engine.health()["registered_regions"], 1)
        engine.release_memory(registration)

    async def test_destination_replay_cannot_change_generation(self):
        identity = {"transfer_id": "transfer-b", "delivery_id": "delivery-b"}
        await self.client.post(
            "entry",
            {
                **identity,
                "manifest": {"prompt": 2},
                "first_token": {"output_token_id": 1},
                "expected_bytes": 2,
                "sender_epoch": "epoch-p",
            },
        )
        destination = {
            **identity,
            "destination": {"region_id": "region"},
            "receiver_epoch": "epoch-d",
            "generation": "first",
        }
        await self.client.post("destination", destination)
        with self.assertRaisesRegex(Exception, "destination changed"):
            await self.client.post(
                "destination", {**destination, "generation": "second"}
            )

        with self.assertRaisesRegex(Exception, "byte count differs"):
            await self.client.post(
                "terminal",
                {
                    **identity,
                    "state": "terminal_success",
                    "transferred_bytes": 1,
                    "sender_epoch": "epoch-p",
                    "receiver_epoch": "epoch-d",
                    "generation": "first",
                },
            )

    async def test_rendezvous_capacity_is_bounded(self):
        payload = {
            "manifest": {},
            "first_token": {},
            "expected_bytes": 1,
            "sender_epoch": "epoch-p",
        }
        with mock.patch(
            "sglang.srt.disaggregation.common.conn._PVD_DIRECT_MAX_ENTRIES", 1
        ):
            await self.client.post(
                "entry", {**payload, "transfer_id": "a", "delivery_id": "a"}
            )
            with self.assertRaisesRegex(Exception, "capacity exceeded"):
                await self.client.post(
                    "entry", {**payload, "transfer_id": "b", "delivery_id": "b"}
                )


if __name__ == "__main__":
    unittest.main()
