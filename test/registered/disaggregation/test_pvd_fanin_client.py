"""D MR lifecycle through real coordinator HTTP; fake payload, no GPU/RDMA."""

import asyncio
import copy
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from sglang.srt.disaggregation.pvd.control_server import create_coordinator_app
from sglang.srt.disaggregation.pvd.full_kv_fanin_client import (
    FanInHTTPClient,
    FullKVFanInDelivery,
)
from sglang.srt.disaggregation.pvd.protocol import ProtocolValidationError
from test_pvd_fanin_coordinator import group
from test_pvd_fanin_store import setup
from test_pvd_fanin_writer import DelayedEngine


@asynccontextmanager
async def connected(c):
    parent, actor, epochs = group(c)
    async with TestClient(TestServer(create_coordinator_app(parent))) as server:
        client = FanInHTTPClient(
            str(server.make_url("")), timeout_seconds=5, max_response_bytes=65536
        )
        delivery = FullKVFanInDelivery(c.receiver, source_epochs=epochs, client=client)
        try:
            yield parent, actor, client, delivery
        finally:
            await client.close()


async def finish_network(delivery):
    await delivery.reserve()
    await delivery.start()
    for _ in range(16):
        if delivery.ready:
            return
        await delivery.poll()
    raise AssertionError(delivery.state)


def test_real_http_all_writers_then_local_reader_then_ack():
    async def run():
        with setup(publish=False) as c:
            async with connected(c) as (parent, _, _, delivery):
                # Consumer pin is distinct from network lifetime.
                c.guard.pin("local-import")
                c.guard.request_release()
                await finish_network(delivery)
                expected = []
                for token in range(8):
                    expected.extend(c.raw[0][4 * token : 4 * token + 4].tolist())
                    expected.extend(c.raw[1][4 * token : 4 * token + 4].tolist())
                assert c.target.buffer.tolist() == expected
                with pytest.raises(ProtocolValidationError, match="install ACK"):
                    delivery.close()
                # Byte import comparison above is CPU-only, not a CUDA fence.
                assert (await delivery.ack_after_install())["state"] == "released"
                delivery.close()
                assert c.guard.value is c.target
                c.guard.unpin("local-import")
                assert c.guard.value is None
                assert parent.entries[c.key].active_delivery_count == 0

    asyncio.run(run())


@pytest.mark.parametrize("operation", ["reserve", "start", "ack"])
def test_lost_http_reply_retains_mr_and_fences_original_epochs(operation):
    async def run():
        with setup(publish=False) as c:
            async with connected(c) as (parent, _, client, delivery):
                if operation == "ack":
                    await finish_network(delivery)
                elif operation == "start":
                    await delivery.reserve()
                original = client.request

                async def lost(op, payload):
                    result = await original(op, payload)
                    if op == operation:
                        raise TimeoutError("reply lost after peer processed")
                    return result

                client.request = lost
                c.guard.request_release()
                call = (
                    delivery.ack_after_install
                    if operation == "ack"
                    else getattr(delivery, operation)
                )
                with pytest.raises(TimeoutError):
                    await call()
                assert c.guard.value is c.target
                result = await delivery.poll()
                assert result["fenced"]
                delivery.close()
                assert c.guard.value is None
                assert parent.entries[c.key].active_delivery_count == 0

    asyncio.run(run())


def test_cancelled_rpc_task_keeps_all_identities_before_any_response():
    async def run():
        with setup(publish=False) as c:
            async with connected(c) as (_, _, client, delivery):
                seen = asyncio.Event()
                original = client.request

                async def blocked(op, payload):
                    result = await original(op, payload)
                    if op == "reserve":
                        seen.set()
                        await asyncio.Event().wait()
                    return result

                client.request = blocked
                task = asyncio.create_task(delivery.reserve())
                await seen.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                c.guard.request_release()
                with pytest.raises(
                    ProtocolValidationError, match="writers must be fenced"
                ):
                    delivery.close()
                assert c.guard.value is c.target
                await delivery.poll()
                delivery.close()
                assert c.guard.value is None

    asyncio.run(run())


def test_inflight_cancel_waits_for_every_actual_handle():
    async def run():
        with setup(publish=False, engine=DelayedEngine()) as c:
            async with connected(c) as (_, _, _, delivery):
                await delivery.reserve()
                await delivery.start()
                c.guard.request_release()
                result = await delivery.fence()
                assert not result["fenced"] and not delivery.ready
                with pytest.raises(ProtocolValidationError):
                    delivery.close()
                for handle, *_ in c.engine.submitted:
                    c.engine.complete(handle)
                await delivery.poll()
                delivery.close()
                assert c.guard.value is None

    asyncio.run(run())


@pytest.mark.parametrize(
    "corruption",
    [
        "epoch",
        "rank",
        "missing_writer",
        "proof_rank",
        "bytes",
        "fenced",
        "released",
        "plan",
    ],
)
def test_corrupt_envelope_cannot_adopt_proofs_or_free_mr(corruption):
    async def run():
        with setup(publish=False) as c:
            async with connected(c) as (_, actor, client, delivery):
                await delivery.reserve()
                await actor.start("fanin", 0)
                for _ in range(16):
                    good = await actor.poll("fanin", 0)
                    if good["state"] == "delivered":
                        break
                assert good["state"] == "delivered"
                bad = copy.deepcopy(good)
                if corruption == "epoch":
                    bad["write_identities"]["1"]["sender_epoch"] = "restarted"
                elif corruption == "rank":
                    bad["destination_rank"] = True
                elif corruption == "missing_writer":
                    bad["write_identities"].pop("1")
                elif corruption == "proof_rank":
                    bad["writer_proofs"]["1"]["source_rank"] = 0
                elif corruption == "bytes":
                    bad["writer_proofs"]["1"]["transferred_bytes"] -= 1
                elif corruption == "fenced":
                    bad["fenced"] = 1
                elif corruption == "released":
                    bad["state"], bad["entry_count_held"] = "released", False
                else:
                    bad["plan_fingerprint"] = "other"
                original = client.request

                async def corrupt(*args):
                    return bad

                client.request = corrupt
                c.guard.request_release()
                with pytest.raises(ProtocolValidationError):
                    await delivery.poll()
                with pytest.raises(
                    ProtocolValidationError, match="writers must be fenced"
                ):
                    delivery.close()
                assert c.guard.value is c.target
                client.request = original
                await delivery.poll()
                delivery.close()
                assert c.guard.value is None

    asyncio.run(run())


def test_bad_epoch_map_is_rejected_before_publication():
    with setup(publish=False) as c:

        class Client:
            async def request(self, *args):
                raise AssertionError("must not send")

        with pytest.raises(ProtocolValidationError, match="epoch map"):
            FullKVFanInDelivery(c.receiver, source_epochs={"0": "V0"}, client=Client())
        # Invalid setup did not publish: no remote proof is required to close.
        c.guard.request_release()
        c.receiver.close()
        assert c.guard.value is None


def test_cancel_before_first_rpc_creates_fences_and_releases():
    async def run():
        with setup(publish=False) as c:
            async with connected(c) as (parent, _, _, delivery):
                delivery.cancel()
                c.guard.request_release()
                assert (await delivery.poll())["state"] == "cancelled"
                delivery.close()
                assert c.guard.value is None
                assert all(e.active_delivery_count == 0 for e in c.entries)
                assert parent.entries[c.key].active_delivery_count == 0

    asyncio.run(run())


def test_cancel_during_successful_rpc_never_makes_network_ready_again():
    async def run():
        with setup(publish=False) as c:
            async with connected(c) as (_, _, client, delivery):
                await finish_network(delivery)
                original = client.request

                async def cancel_then_return(op, payload):
                    result = await original(op, payload)
                    delivery.cancel()
                    return result

                client.request = cancel_then_return
                await delivery.poll()
                assert not delivery.ready
                client.request = original
                await delivery.poll()
                delivery.close()

    asyncio.run(run())


def test_closed_http_is_not_a_fence():
    async def run():
        with setup(publish=False) as c:
            async with connected(c) as (_, _, client, delivery):
                await delivery.reserve()
                await client.close()
                c.guard.request_release()
                with pytest.raises(RuntimeError, match="HTTP client is closed"):
                    await delivery.poll()
                with pytest.raises(
                    ProtocolValidationError, match="writers must be fenced"
                ):
                    delivery.close()
                assert c.guard.value is c.target

    asyncio.run(run())


def test_local_release_failure_cannot_republish_after_network_unpin():
    async def run():
        with setup(publish=False) as c:
            async with connected(c) as (_, _, _, delivery):
                await finish_network(delivery)
                await delivery.ack_after_install()
                original = c.engine.release_memory
                failed = False

                def fail_once(memory):
                    nonlocal failed
                    if memory is c.target and not failed:
                        failed = True
                        raise RuntimeError("local MR cleanup failed")
                    return original(memory)

                c.engine.release_memory = fail_once
                c.guard.request_release()
                with pytest.raises(RuntimeError, match="local MR cleanup failed"):
                    delivery.close()
                assert c.guard.value is c.target
                with pytest.raises(RuntimeError, match="delivery closed"):
                    await delivery.reserve()
                c.guard.request_release()
                assert c.guard.value is None

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["large", "redirect", "list", "failure"])
def test_http_response_limit_status_and_object_contract(mode):
    async def run():
        calls = []

        async def handler(request):
            calls.append(request.path)
            if mode == "large":
                return web.Response(text="x" * 2048)
            if mode == "redirect":
                raise web.HTTPFound("/v1/fanin/poll")
            if mode == "list":
                return web.json_response([])
            return web.Response(status=503)

        app = web.Application()
        app.router.add_post("/v1/fanin/{operation}", handler)
        async with TestClient(TestServer(app)) as server:
            client = FanInHTTPClient(
                str(server.make_url("")), timeout_seconds=2, max_response_bytes=100
            )
            try:
                with pytest.raises((ProtocolValidationError, RuntimeError)):
                    await client.request("reserve", {})
                assert calls == ["/v1/fanin/reserve"]
            finally:
                await client.close()

    asyncio.run(run())
