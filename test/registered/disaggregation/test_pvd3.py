"""PVD 3.0 correctness tests; CPU tensors and the real fake-transport stores."""

import asyncio
import time
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.kv_packer import (
    pack_full_prompt_kv,
    unpack_full_prompt_kv,
)


def test_refresh_preserves_generated_tokens_in_partial_prompt_page():
    source = SimpleNamespace(k_buffer=[torch.arange(12).reshape(12, 1, 1)], v_buffer=[])
    target = SimpleNamespace(k_buffer=[torch.full((20, 1, 1), -9)], v_buffer=[])
    packed = pack_full_prompt_kv(source, [0, 2], page_size=4).tensor
    unpack_full_prompt_kv(packed, target, [1, 3], page_size=4, prompt_token_count=5)
    assert target.k_buffer[0].flatten().tolist() == [
        -9,
        -9,
        -9,
        -9,
        0,
        1,
        2,
        3,
        -9,
        -9,
        -9,
        -9,
        8,
        -9,
        -9,
        -9,
        -9,
        -9,
        -9,
        -9,
    ]
    target.k_buffer[0][13:] = 77
    unpack_full_prompt_kv(packed, target, [1, 3], page_size=4, prompt_token_count=5)
    assert target.k_buffer[0][13:].flatten().tolist() == [77] * 7


def test_malformed_refresh_does_not_partially_overwrite_kv():
    source = SimpleNamespace(k_buffer=[torch.ones(8, 1, 1)], v_buffer=[])
    target = SimpleNamespace(k_buffer=[torch.full((8, 1, 1), 99.0)], v_buffer=[])
    packed = pack_full_prompt_kv(source, [0, 1], page_size=4).tensor
    with pytest.raises(ValueError):
        unpack_full_prompt_kv(
            torch.cat([packed, torch.zeros(1, dtype=torch.uint8)]),
            target,
            [0, 1],
            page_size=4,
            prompt_token_count=5,
        )
    assert torch.all(target.k_buffer[0] == 99)


def test_refresh_clock_counts_only_completed_decode_tokens():
    from sglang.srt.disaggregation.pvd.retrieval import RefreshClock

    clock = RefreshClock("delivery", interval=4)
    assert clock.due(0)
    first = clock.begin(0)
    assert first == "delivery:refresh:0"
    with pytest.raises(ValueError, match="in flight"):
        clock.begin(0)
    with pytest.raises(ValueError, match="stale"):
        clock.complete("delivery:refresh:1")
    clock.complete(first)
    assert not clock.due(3)
    assert clock.due(4)
    second = clock.begin(4)
    clock.complete(second)
    assert not clock.due(7)
    assert clock.due(8)
    with pytest.raises(ValueError):
        clock.due(2)
    with pytest.raises(ValueError):
        RefreshClock("delivery", interval=0)


@pytest.mark.parametrize("interval", [0, -1, True, "4"])
def test_pvd_rejects_invalid_refresh_interval(interval):
    from sglang.srt.arg_groups.pvd_disaggregation_hook import handle_pvd_disaggregation

    args = SimpleNamespace(
        disaggregation_topology="pvd", pvd_kv_refresh_interval=interval
    )
    with pytest.raises(ValueError, match="positive integer"):
        handle_pvd_disaggregation(args)


def test_pd_config_is_unchanged_and_pvd_decode_disables_overlap():
    from sglang.srt.arg_groups.pvd_disaggregation_hook import handle_pvd_disaggregation

    pd = SimpleNamespace(disaggregation_topology="pd", disable_overlap_schedule=False)
    handle_pvd_disaggregation(pd)
    assert not pd.disable_overlap_schedule
    args = SimpleNamespace(
        disaggregation_topology="pvd",
        disaggregation_mode="decode",
        pvd_kv_refresh_interval=4,
        disable_overlap_schedule=False,
        pvd_vector_coordinator_url="http://v:9100",
        pvd_vector_groups=None,
        tp_size=2,
        dp_size=1,
        enable_dp_attention=False,
        pp_size=1,
        pvd_rank_rails="mlx5_0,mlx5_0",
        disaggregation_transfer_backend="mooncake",
        pvd_strict_rdma_preflight=True,
        speculative_algorithm=None,
        enable_hierarchical_cache=False,
        enable_hisparse=False,
        enable_prefill_context_parallel=False,
        disaggregation_decode_enable_radix_cache=False,
        pvd_model_instance_id="model",
        pvd_transfer_staging_budget_bytes=1 << 30,
        pvd_transfer_max_inflight=64,
    )
    handle_pvd_disaggregation(args)
    assert args.disable_overlap_schedule
    assert args.disable_radix_cache


async def make_ready_entry(coordinator, engine, req_id):
    from sglang.srt.disaggregation.pvd.protocol import (
        FirstTokenMetadata,
        KVEntryKey,
        KVEntryManifest,
        KVLayoutSignature,
        KVShardManifest,
    )
    from sglang.srt.disaggregation.pvd.transfer_engine import MemorySlice

    manifest = KVEntryManifest(
        key=KVEntryKey("model", req_id, req_id),
        layout=KVLayoutSignature(
            model_id="model",
            model_revision="r",
            kv_dtype="float16",
            page_size=4,
            num_layers=1,
            total_kv_heads=4,
            kv_heads_per_rank=2,
            head_dim=1,
            tp_size=2,
            pp_size=1,
            tensor_layout="flat-test-layout",
            extra={
                "component_bytes_per_token": [4],
                "component_count": 1,
                "component_dtypes": ["torch.float16"],
                "component_token_shapes": [[2, 1]],
            },
        ),
        prompt_token_count=5,
        shards=[
            KVShardManifest(
                rank=rank,
                rail=f"mlx5_{rank}",
                expected_bytes=32,
                page_count=2,
                last_page_valid_tokens=1,
                layer_start=0,
                layer_end=1,
            )
            for rank in range(2)
        ],
    )
    entry = await coordinator.create_entry(manifest)
    for rank in range(2):
        source = engine.register_memory(
            torch.full((32,), rank + 10, dtype=torch.uint8),
            endpoint="p",
            rank=rank,
            rail=f"mlx5_{rank}",
        )
        engine.submit_put(MemorySlice(source, 0, 32), entry.target_regions[rank])
        await coordinator.commit_shard(
            manifest.key,
            rank,
            32,
            FirstTokenMetadata(output_token_id=7) if rank == 0 else None,
        )
        engine.release_memory(source)
    return manifest.key


def make_vector():
    from sglang.srt.disaggregation.pvd.coordinator import (
        LocalShardClient,
        VectorCoordinator,
    )
    from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine
    from sglang.srt.disaggregation.pvd.vector_store import VectorKVStore

    engine = FakeTransferEngine()
    stores = [
        VectorKVStore(
            rank=rank,
            world_size=2,
            rail=f"mlx5_{rank}",
            device="cpu",
            total_pages=16,
            page_bytes=16,
            endpoint=f"v{rank}",
            transfer_engine=engine,
            allow_cpu_for_tests=True,
        )
        for rank in range(2)
    ]
    return engine, stores, VectorCoordinator([LocalShardClient(s) for s in stores])


def test_batched_retrieval_reuses_entry_with_independent_deliveries():
    async def scenario():
        engine, stores, coordinator = make_vector()
        keys = [
            await make_ready_entry(coordinator, engine, name) for name in ("a", "b")
        ]
        targets = [
            {
                rank: engine.register_memory(
                    torch.zeros(32, dtype=torch.uint8),
                    endpoint=f"d-{seq}",
                    rank=rank,
                    rail=f"mlx5_{rank}",
                )
                for rank in range(2)
            }
            for seq in range(2)
        ]
        for epoch in range(2):
            requests = [
                {
                    "key": key.to_dict(),
                    "sequence_id": key.req_id,
                    "delivery_id": f"{key.req_id}:{epoch}",
                    "selection": "full_prompt",
                    "destinations": {
                        str(r): reg.descriptor.to_dict()
                        for r, reg in targets[i].items()
                    },
                }
                for i, key in enumerate(keys)
            ]
            results = await coordinator.retrieve(requests)
            assert [r["sequence_id"] for r in results] == ["a", "b"]
            for i, result in enumerate(results):
                assert result["state"] == "delivered"
                assert result["token_ranges"] == [[0, 5]]
                for rank, reg in targets[i].items():
                    assert reg.buffer.tolist() == [rank + 10] * 32
                    reg.buffer.zero_()
                await coordinator.ack_delivery(result["delivery_id"])
        assert all(coordinator.entries[k].active_delivery_count == 0 for k in keys)
        for key in keys:
            await coordinator.release_entry(key)
        assert all(store.allocator.available_pages == 16 for store in stores)

    asyncio.run(scenario())


def test_consumer_lease_prevents_expiry_without_pin_forever():
    async def scenario():
        engine, stores, coordinator = make_vector()
        key = await make_ready_entry(coordinator, engine, "long-decode")
        lease = await coordinator.renew_consumer(key, "d-session")
        assert lease["renew_after_seconds"] > 0
        now = time.monotonic()
        coordinator.entries[key].expires_at = now - 1
        await coordinator.reap_expired(now)
        assert coordinator.entries[key].state.value == "stored"
        await coordinator.release_consumer(key, "d-session")
        await coordinator.reap_expired(now + 1000)
        assert coordinator.entries[key].state.value == "released"
        assert all(s.allocator.available_pages == 16 for s in stores)

    asyncio.run(scenario())


def test_retrieval_rejects_unsupported_search_before_transfer():
    async def scenario():
        engine, _, coordinator = make_vector()
        key = await make_ready_entry(coordinator, engine, "a")
        with pytest.raises(ValueError, match="selection"):
            await coordinator.retrieve(
                [
                    {
                        "key": key.to_dict(),
                        "sequence_id": "a",
                        "delivery_id": "round-0",
                        "selection": "cagra",
                        "destinations": {},
                    }
                ]
            )
        assert not coordinator.deliveries

    asyncio.run(scenario())


def test_router_admission_validates_batched_identities():
    async def scenario():
        _, _, coordinator = make_vector()
        request = {
            "pvd_transfer_id": ["a", "b"],
            "pvd_delivery_id": ["da", "db"],
            "pvd_vector_group_id": ["v", "v"],
            "text": ["hello", "world"],
        }
        assert await coordinator.admit_request(request) == {"accepted": ["a", "b"]}
        assert await coordinator.admit_request(request) == {"accepted": ["a", "b"]}
        with pytest.raises(ValueError):
            await coordinator.admit_request({**request, "pvd_delivery_id": ["da"]})

    asyncio.run(scenario())


def test_http_admission_lease_and_retrieval_contract():
    async def scenario():
        from aiohttp.test_utils import TestServer
        from sglang.srt.disaggregation.pvd.client import PVDCoordinatorClient
        from sglang.srt.disaggregation.pvd.control_server import create_coordinator_app

        engine, _, coordinator = make_vector()
        # This assertion checks the endpoint before opening any sockets.
        app = create_coordinator_app(coordinator)
        assert "/v1/retrieve" in {r.resource.canonical for r in app.router.routes()}
        async with TestServer(app) as server:
            client = PVDCoordinatorClient(str(server.make_url("")))
            try:
                await client.admit_request(
                    {
                        "pvd_transfer_id": "a",
                        "pvd_delivery_id": "da",
                        "pvd_vector_group_id": "v",
                        # Long prompts can exceed aiohttp's default 1 MiB.
                        "text": "hello" * 300_000,
                    }
                )
                key = await make_ready_entry(coordinator, engine, "a")
                await client.renew_consumer(key, "da")
                targets = {
                    rank: engine.register_memory(
                        torch.zeros(32, dtype=torch.uint8),
                        endpoint="d",
                        rank=rank,
                        rail=f"mlx5_{rank}",
                    )
                    for rank in range(2)
                }
                response = await client.retrieve(
                    [
                        {
                            "key": key.to_dict(),
                            "sequence_id": "a",
                            "delivery_id": "da:0",
                            "selection": "full_prompt",
                            "destinations": {
                                str(r): reg.descriptor.to_dict()
                                for r, reg in targets.items()
                            },
                        }
                    ]
                )
                assert response["results"][0]["state"] == "delivered"
                assert targets[1].buffer.tolist() == [11] * 32
                await client.ack_delivery("da:0")
                await client.release_consumer(key, "da")
            finally:
                await client.close()

    asyncio.run(scenario())


def _with_close_retention(manager):
    """Give a fake manager the Task 6 retention API.

    An undrained Decode close is owned by the manager, not by the request, so
    every fake that drives a session close needs somewhere to hand it.
    """
    manager.pending_decode_closes = []
    manager.retain_pending_close = lambda session: manager.pending_decode_closes.append(
        session
    )
    return manager


def test_decode_session_reuses_staging_and_validates_reply_before_unpack():
    from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeSession
    from sglang.srt.disaggregation.pvd.protocol import KVEntryKey
    from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine

    pool = SimpleNamespace(k_buffer=[torch.full((16, 1, 1), -9.0)], v_buffer=[])
    req = SimpleNamespace(
        origin_input_ids=list(range(5)), output_ids=[7], pvd_delivery_id="d"
    )
    key = KVEntryKey("model", "a", "a")
    manager = SimpleNamespace(
        key_for=lambda r: key,
        client_for=lambda r: object(),
        kv_pool=pool,
        page_size=4,
        tp_rank=0,
        rail="mlx5_0",
        transfer_engine=FakeTransferEngine(),
        layout=lambda: SimpleNamespace(to_dict=lambda: {}),
        scheduler=SimpleNamespace(
            server_args=SimpleNamespace(pvd_kv_refresh_interval=4)
        ),
    )
    _with_close_retention(manager)
    session = PVDDecodeSession(manager, req)
    first = session.prepare([1, 2])
    ptr = session.staging.data_ptr()
    # The destination is pinned from the moment its descriptor is published.
    assert first["destination"]["backend_metadata"]["pvd_receiver_epoch"]
    assert first["destination"]["backend_metadata"]["pvd_generation"]
    source = SimpleNamespace(
        k_buffer=[torch.arange(8, dtype=torch.float32).reshape(8, 1, 1)], v_buffer=[]
    )
    session.staging.copy_(pack_full_prompt_kv(source, [0, 1], page_size=4).tensor)
    reply = {
        "key": key.to_dict(),
        "sequence_id": "a",
        "delivery_id": first["delivery_id"],
        "selection": "full_prompt",
        "token_ranges": [[0, 5]],
        "state": "delivered",
    }
    with pytest.raises(ValueError, match="identity"):
        session.unpack({**reply, "delivery_id": "old"})
    assert torch.all(pool.k_buffer[0] == -9)
    session.unpack(reply)
    # Staging may not be reused while the previous refresh is still pinned.
    with pytest.raises(RuntimeError, match="unfenced"):
        session.prepare([1, 2])
    session.release_refresh()
    session.clock.complete(reply["delivery_id"])
    assert pool.k_buffer[0][4:9].flatten().tolist() == [0, 1, 2, 3, 4]
    pool.k_buffer[0][9:] = 88
    req.output_ids.extend([1, 2, 3, 4])
    second = session.prepare([1, 2])
    assert second["delivery_id"] != first["delivery_id"]
    # A new generation per refresh: a late write from the first one cannot be
    # accepted as belonging to the second.
    assert (
        second["destination"]["backend_metadata"]["pvd_generation"]
        != first["destination"]["backend_metadata"]["pvd_generation"]
    )
    assert session.staging.data_ptr() == ptr
    session.unpack({**reply, "delivery_id": second["delivery_id"]})
    assert pool.k_buffer[0][9:].flatten().tolist() == [88] * 7


def test_cancel_before_retrieve_fences_late_request():
    async def scenario():
        from sglang.srt.disaggregation.pvd.transfer_authorization import (
            PVD_TRANSFER_LIFECYCLE_PROTOCOL,
            WriteIdentity,
        )

        engine, _, coordinator = make_vector()
        key = await make_ready_entry(coordinator, engine, "a")
        identity = WriteIdentity(
            protocol=PVD_TRANSFER_LIFECYCLE_PROTOCOL,
            sender_epoch="vector-epoch",
            receiver_epoch="decode-epoch",
            transfer_id="late-round:d0",
            region_id="unpublished-region",
            generation="decode-generation",
            shard_rank=0,
            key=key,
        )
        # Closing the coordinator ID gate blocks a late retrieve, but without a
        # previously saved shard authorization it must not claim a safe fence.
        with pytest.raises(Exception, match="unknown delivery"):
            await coordinator.fence_retrieval("late-round", [identity.to_dict()])
        reply = await coordinator.retrieve(
            [
                {
                    "key": key.to_dict(),
                    "sequence_id": "a",
                    "delivery_id": "late-round",
                    "selection": "full_prompt",
                    "destinations": {},
                }
            ]
        )
        assert reply[0]["state"] == "failed"
        assert "fenced" in reply[0]["error"]
        assert not coordinator.deliveries

    asyncio.run(scenario())


def test_fence_rejects_delivery_without_authoritative_store_identity():
    async def scenario():
        from sglang.srt.disaggregation.pvd.request_state import DeliveryState
        from sglang.srt.disaggregation.pvd.transfer_authorization import (
            PVD_TRANSFER_LIFECYCLE_PROTOCOL,
            WriteIdentity,
        )

        engine, _, coordinator = make_vector()
        key = await make_ready_entry(coordinator, engine, "a")
        destinations = {
            r: engine.register_memory(
                torch.zeros(32, dtype=torch.uint8),
                endpoint="d",
                rank=r,
                rail=f"mlx5_{r}",
            ).descriptor
            for r in range(2)
        }
        await coordinator.reserve_delivery(
            key=key, delivery_id="slow-write", destinations=destinations
        )
        # A coordinator timeout is not evidence that the remote shard stopped.
        coordinator.deliveries["slow-write"].state = DeliveryState.FAILED

        identities = [
            WriteIdentity(
                protocol=PVD_TRANSFER_LIFECYCLE_PROTOCOL,
                sender_epoch=f"vector-epoch-{rank}",
                receiver_epoch="decode-epoch",
                transfer_id=f"slow-write:d{rank}",
                region_id=destinations[rank].region_id,
                generation="decode-generation",
                shard_rank=rank,
                key=key,
            ).to_dict()
            for rank in range(2)
        ]
        with pytest.raises(Exception, match="identity set"):
            await coordinator.fence_retrieval("slow-write", identities)

    asyncio.run(scenario())


def test_shard_fence_blocks_a_delayed_reserve():
    async def scenario():
        engine, stores, coordinator = make_vector()
        key = await make_ready_entry(coordinator, engine, "a")
        destination = engine.register_memory(
            torch.zeros(32, dtype=torch.uint8), endpoint="d", rank=0, rail="mlx5_0"
        ).descriptor
        # An ID-only tombstone stops late reservation, but is not MR proof.
        assert stores[0].fence_delivery(key, "late:d0")["fenced"] is False
        with pytest.raises(Exception, match="fenced"):
            stores[0].reserve_delivery(key, "late:d0", destination)

    asyncio.run(scenario())


def test_fenced_old_delivery_cannot_write_a_reregistered_destination():
    async def scenario():
        engine, stores, coordinator = make_vector()
        key = await make_ready_entry(coordinator, engine, "reuse-after-fence")
        buffer = torch.full((32,), 77, dtype=torch.uint8)
        old = engine.register_memory(buffer, endpoint="d", rank=0, rail="mlx5_0")
        stores[0].reserve_delivery(key, "old:d0", old.descriptor)
        stores[0].fence_delivery(key, "old:d0")
        engine.release_memory(old)
        new = engine.register_memory(buffer, endpoint="d", rank=0, rail="mlx5_0")
        try:
            assert new.descriptor.address == old.descriptor.address
            assert new.descriptor.region_id != old.descriptor.region_id
            with pytest.raises(Exception, match="fenced"):
                stores[0].start_delivery(key, "old:d0")
            assert torch.all(buffer == 77)
        finally:
            engine.release_memory(new)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["timeout", "wrong-delivery", "unfenced"])
def test_decode_retains_mr_until_matching_fence_confirmation(failure):
    """An unproven fence must retain the MR, never release it."""
    from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeSession
    from sglang.srt.disaggregation.pvd.protocol import KVEntryKey
    from sglang.srt.disaggregation.pvd.transfer_engine import (
        FakeTransferEngine,
        MemorySlice,
        TransferStatus,
    )

    async def scenario():
        key = KVEntryKey("model", "fence", "fence")

        class Client:
            calls = 0
            allow = False

            async def fence_retrieval(self, delivery_id, identities):
                Client.calls += 1
                if Client.allow:
                    return {
                        "delivery_id": delivery_id,
                        "fenced": True,
                        "identities": identities,
                    }
                if failure == "timeout":
                    raise TimeoutError("V has not confirmed completion")
                if failure == "wrong-delivery":
                    return {
                        "delivery_id": "wrong",
                        "fenced": True,
                        "identities": identities,
                    }
                return {
                    "delivery_id": delivery_id,
                    "fenced": False,
                    "identities": identities,
                }

            async def poll_delivery(self, delivery_id):
                raise AssertionError("identities were already saved")

            async def release_consumer(self, key, consumer_id):
                return None

        client = Client()
        engine = FakeTransferEngine()
        pool = SimpleNamespace(k_buffer=[torch.full((16, 1, 1), -9.0)], v_buffer=[])
        req = SimpleNamespace(
            origin_input_ids=list(range(5)), output_ids=[7], pvd_delivery_id="d"
        )
        manager = _with_close_retention(
            SimpleNamespace(
                key_for=lambda r: key,
                client_for=lambda r: client,
                kv_pool=pool,
                page_size=4,
                tp_rank=0,
                rail="mlx5_0",
                transfer_engine=engine,
                layout=lambda: SimpleNamespace(to_dict=lambda: {}),
                scheduler=SimpleNamespace(
                    server_args=SimpleNamespace(pvd_kv_refresh_interval=4)
                ),
            )
        )
        session = PVDDecodeSession(manager, req)
        session.prepare([1, 2])
        registration = session.registration
        session.identities = {0: session.expected_identity(0, "v-worker-epoch")}

        def mr_is_live():
            return (
                engine.submit_put(
                    MemorySlice(registration, 0, 16), registration.descriptor
                ).status
                == TransferStatus.SUCCESS
            )

        assert await session.progress_close() is False
        assert session.registration is registration
        assert mr_is_live()

        drained = await session.close()
        assert drained is False
        # Ownership moved to the manager, not away from the process.
        assert session in manager.pending_decode_closes
        assert session.registration is registration
        assert mr_is_live()

        Client.allow = True
        assert await session.progress_close() is True
        assert mr_is_live(), "fencing alone must not deregister the MR"
        assert await session.close() is True
        assert session.registration is None
        assert not mr_is_live()

    asyncio.run(scenario())


def test_decode_admission_does_not_start_kv_delivery():
    from sglang.srt.disaggregation.base.conn import KVPoll
    from sglang.srt.disaggregation.pvd.conn import PVDKVReceiver

    async def scenario():
        engine, _, coordinator = make_vector()
        key = await make_ready_entry(coordinator, engine, "admit")
        entry = coordinator.entries[key]
        req = SimpleNamespace(
            origin_input_ids=list(range(5)),
            output_ids=[],
            pvd_delivery_id="d",
            bootstrap_room=123,
        )
        manager = SimpleNamespace(
            key_for=lambda r: key,
            client_for=lambda r: object(),
            decode_runtime_for=lambda r: object(),
            layout=lambda: entry.manifest.layout,
            local_shard_manifest=lambda n: entry.manifest.shard(0),
            decode_sessions={},
            transfer_engine=engine,
            tp_rank=0,
            page_size=4,
            scheduler=SimpleNamespace(
                server_args=SimpleNamespace(pvd_kv_refresh_interval=4)
            ),
            metadata_buffers=SimpleNamespace(
                output_ids=torch.zeros((1, 1), dtype=torch.long),
                cached_tokens=torch.zeros((1, 4), dtype=torch.long),
                bootstrap_room=torch.zeros((1, 1), dtype=torch.long),
            ),
        )
        receiver = PVDKVReceiver(mgr=manager, req=req)
        receiver._entry_record = entry.to_dict()
        receiver._send_metadata([1, 2], aux_index=0)
        assert receiver.poll() == KVPoll.Success
        assert manager.metadata_buffers.output_ids[0, 0].item() == 7
        assert receiver.session.staging is None
        assert not coordinator.deliveries
        receiver.clear()
        assert key in manager.decode_sessions

    asyncio.run(scenario())


def test_cancel_waiting_session_releases_lease_without_finished_flag():
    from sglang.srt.disaggregation.pvd.conn import _AsyncControlLoop
    from sglang.srt.disaggregation.pvd.decode_refresh import (
        PVDDecodeRefresher,
        PVDDecodeSession,
    )

    engine, _, coordinator = make_vector()
    control = _AsyncControlLoop()
    key = control.submit(
        make_ready_entry(coordinator, engine, "cancel-waiting")
    ).result(timeout=10)
    control.submit(coordinator.renew_consumer(key, "d-waiting")).result(timeout=10)

    class LocalClient:
        async def release_consumer(self, key, consumer_id):
            return await coordinator.release_consumer(key, consumer_id)

    req = SimpleNamespace(pvd_delivery_id="d-waiting", finished=lambda: False)
    manager = SimpleNamespace(
        key_for=lambda r: key,
        client_for=lambda r: LocalClient(),
        control=control,
        tp_rank=0,
        scheduler=SimpleNamespace(
            server_args=SimpleNamespace(pvd_kv_refresh_interval=4)
        ),
    )
    session = PVDDecodeSession(manager, req)
    manager.decode_sessions = {key: session}
    try:
        PVDDecodeRefresher(manager).release_request(req)
        session._close_future.result(timeout=10)
        assert not manager.decode_sessions
        assert not coordinator.entries[key].consumer_leases
    finally:
        control.loop.call_soon_threadsafe(control.loop.stop)
        control.thread.join(timeout=10)
        control.loop.close()


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("identity_fault", [False, True, "regression", "prepare"])
@pytest.mark.parametrize(
    "bootstrap_case", ["off", "success", "cancel-retrieve", "cancel-ack"]
)
def test_two_rank_batch_refresh_periodicity_and_rank0_lease_failure(
    tp_size, identity_fault, bootstrap_case
):
    """Real coordinator and tensors, with a thread barrier for TP collectives."""
    import threading
    from concurrent.futures import Future, ThreadPoolExecutor
    from dataclasses import replace

    from sglang.srt.disaggregation.pvd.conn import _AsyncControlLoop
    from sglang.srt.disaggregation.pvd.decode_refresh import (
        PVDDecodeRefresher,
        PVDDecodeSession,
    )

    engine, _, coordinator = make_vector()
    control = _AsyncControlLoop()
    key = control.submit(make_ready_entry(coordinator, engine, "batch")).result(
        timeout=10
    )
    barrier = threading.Barrier(tp_size, timeout=10)
    slots = [None] * tp_size
    retrieve_permission, ack_permission = Future(), Future()
    if bootstrap_case == "off":
        retrieve_permission.set_result(None)
        ack_permission.set_result(None)

    class LocalClient:
        async def retrieve(self, sequences):
            await asyncio.wrap_future(retrieve_permission)
            return {"results": await coordinator.retrieve(sequences)}

        async def ack_delivery(self, delivery_id):
            await asyncio.wrap_future(ack_permission)
            return (await coordinator.ack_delivery(delivery_id)).to_dict()

        async def poll_delivery(self, delivery_id):
            return (await coordinator.poll_delivery(delivery_id)).to_dict()

        async def fence_retrieval(self, delivery_id, identities):
            return await coordinator.fence_retrieval(delivery_id, identities)

        async def release_consumer(self, key, consumer_id):
            return await coordinator.release_consumer(key, consumer_id)

    client = LocalClient()
    managers, sessions, refreshers = [], [], []
    heads = 4 // tp_size
    storage_layout = coordinator.entries[key].manifest.layout
    layout = replace(
        storage_layout,
        tp_size=tp_size,
        kv_heads_per_rank=heads,
        extra={
            **storage_layout.extra,
            "component_bytes_per_token": [heads * 2],
            "component_token_shapes": [[heads, 1]],
        },
    )
    for rank in range(tp_size):

        def gather(value, rank=rank):
            slots[rank] = value
            barrier.wait()
            result = list(slots)
            barrier.wait()
            return result

        req = SimpleNamespace(
            origin_input_ids=list(range(5)),
            output_ids=[7],
            pvd_delivery_id="batch-d",
            req_pool_idx=0,
            finished=lambda: False,
        )
        manager = SimpleNamespace(
            key_for=lambda r: key,
            client_for=lambda r: client,
            vector_group_for=lambda r: "v",
            pending_decode_closes=[],
            clients={"v": client},
            kv_pool=SimpleNamespace(
                k_buffer=[torch.full((16, heads, 1), -9, dtype=torch.float16)],
                v_buffer=[],
            ),
            page_size=4,
            tp_rank=rank,
            rail=f"mlx5_{rank // (tp_size // 2)}",
            transfer_engine=engine,
            layout=lambda: layout,
            control=control,
            gather_rank_objects=gather,
            scheduler=SimpleNamespace(
                server_args=SimpleNamespace(pvd_kv_refresh_interval=4),
                req_to_token_pool=SimpleNamespace(
                    req_to_token=torch.arange(16).reshape(1, 16)
                ),
            ),
        )
        session = PVDDecodeSession(manager, req)
        manager.decode_sessions = {key: session}
        _with_close_retention(manager)
        managers.append(manager)
        sessions.append(session)
        refreshers.append(PVDDecodeRefresher(manager))

    try:
        with ThreadPoolExecutor(max_workers=tp_size) as executor:

            def step():
                futures = [
                    executor.submit(r.refresh, [s.req])
                    for r, s in zip(refreshers, sessions)
                ]
                return [future.result(timeout=15) for future in futures]

            def bootstrap_step(method):
                futures = [
                    executor.submit(
                        getattr(r, method),
                        *(([s.req],) if method == "start_bootstrap" else ()),
                    )
                    for r, s in zip(refreshers, sessions)
                ]
                return [future.result(timeout=15) for future in futures]

            if identity_fault:
                if identity_fault == "regression":
                    sessions[-1].clock.last_tokens = 4
                elif identity_fault == "prepare":
                    def refuse_prepare(pages):
                        raise RuntimeError("rank-local preparation failure")
                    sessions[-1].prepare = refuse_prepare
                else:
                    sessions[-1].clock.round = 3
                errors = (
                    step() if bootstrap_case == "off"
                    else [errors for _, errors in bootstrap_step("start_bootstrap")]
                )
                assert all(len(items) == 1 for items in errors)
                assert not coordinator.deliveries
                # A descriptor that never left D must not pin memory forever.
                assert all(s._refresh_owner is None for s in sessions)
                return

            if bootstrap_case == "off":
                assert step() == [[] for _ in range(tp_size)]
            else:
                assert bootstrap_step("start_bootstrap") == [([], [])] * tp_size
                # An unresolved network future must return control, not stall
                # this scheduler pass. No initial clock is completed yet.
                assert bootstrap_step("poll_bootstrap") == [([], [])] * tp_size
                assert not coordinator.deliveries
                assert all(s.clock.round == 0 for s in sessions)
                assert all(s._initial_receipt is None for s in sessions)
                if bootstrap_case == "cancel-retrieve":
                    sessions[-1]._closed = True  # rank-local cancellation
                retrieve_permission.set_result(None)
                refreshers[0]._bootstrap[2].result(timeout=10)
                result = bootstrap_step("poll_bootstrap")
                if bootstrap_case == "cancel-retrieve":
                    assert all(len(errors) == 1 for _, errors in result)
                    assert all(torch.all(m.kv_pool.k_buffer[0] == -9) for m in managers)
                    assert all(s.clock.round == 0 for s in sessions)
                    return
                assert result == [([], [])] * tp_size
                # Even after installation, an unfinished ACK is not runnable.
                assert bootstrap_step("poll_bootstrap") == [([], [])] * tp_size
                assert all(s.clock.round == 0 for s in sessions)
                assert all(s._initial_receipt is None for s in sessions)
                if bootstrap_case == "cancel-ack":
                    sessions[-1]._closed = True
                ack_permission.set_result(None)
                refreshers[0]._bootstrap[2].result(timeout=10)
                result = bootstrap_step("poll_bootstrap")
                if bootstrap_case == "cancel-ack":
                    assert all(len(errors) == 1 for _, errors in result)
                    assert all(s.clock.round == 0 for s in sessions)
                    assert all(s._initial_receipt is None for s in sessions)
                    return
                assert all(
                    completed == [s.req] and not errors
                    for (completed, errors), s in zip(result, sessions)
                )
                assert all(not r.bootstrap_pending for r in refreshers)
                assert all(
                    s.clock.round == 1 and s.clock.last_tokens == 0 for s in sessions
                )
            assert len(coordinator.deliveries) == 1
            assert all(s._initial_receipt.req is s.req for s in sessions)
            pointers = [s.staging.data_ptr() for s in sessions]
            for manager in managers:
                manager.kv_pool.k_buffer[0][5:] = 42
            for s in sessions:
                s.req.output_ids.extend([1, 2, 3])
            assert step() == [[] for _ in range(tp_size)]
            assert len(coordinator.deliveries) == 1
            for s in sessions:
                s.req.output_ids.append(4)
            assert step() == [[] for _ in range(tp_size)]
            assert len(coordinator.deliveries) == 2
            assert [s.staging.data_ptr() for s in sessions] == pointers
            for rank, manager in enumerate(managers):
                tensor = manager.kv_pool.k_buffer[0]
                assert tensor[:5].contiguous().view(torch.uint8).flatten().tolist() == [
                    10 + rank // (tp_size // 2)
                ] * (5 * heads * 2)
                assert torch.all(tensor[5:] == 42)
            # Only rank 0 sees an async lease failure: both ranks must take the
            # same collective path and abort without launching another write.
            sessions[0].lease_error = "lease lost"
            errors = step()
            assert all(
                len(items) == 1 and "lease lost" in items[0][1] for items in errors
            )
            assert len(coordinator.deliveries) == 2
    finally:
        barrier.abort()
        for session in sessions:
            # Published cancellations drain through a matching identity fence;
            # pre-publication failures have already dropped their local pins.
            drained = control.submit(session.close()).result(timeout=10)
            assert drained
            assert session.registration is None
            assert session.staging is None
        control.loop.call_soon_threadsafe(control.loop.stop)
        control.thread.join(timeout=10)
        control.loop.close()
