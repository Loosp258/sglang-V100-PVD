import asyncio
import dataclasses
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from sglang.srt.disaggregation.pvd.coordinator import (
    CoordinatorError,
    DeliveryRecord,
    LocalShardClient,
    VectorCoordinator,
)
from sglang.srt.disaggregation.pvd.protocol import (
    KVEntryKey,
    RemoteRegionDescriptor,
)
from sglang.srt.disaggregation.pvd.request_state import DeliveryState, EntryState
from sglang.srt.disaggregation.pvd.transfer_authorization import (
    PVD_TRANSFER_LIFECYCLE_PROTOCOL,
    WriteAuthorization,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransportState,
)


def make_identity(**changes):
    values = {
        "protocol": PVD_TRANSFER_LIFECYCLE_PROTOCOL,
        "sender_epoch": "vector-epoch-1",
        "receiver_epoch": "decode-epoch-1",
        "transfer_id": "delivery-1:d0",
        "region_id": "region-1",
        "generation": "generation-1",
        "shard_rank": 0,
        "key": KVEntryKey("model-1", "request-1", "entry-1"),
    }
    values.update(changes)
    return WriteIdentity(**values)


def make_destination(rank, **metadata_changes):
    metadata = {
        "pvd_receiver_epoch": "decode-epoch-1",
        "pvd_generation": f"generation-{rank}",
        "layout_token": f"layout-{rank}",
    }
    metadata.update(metadata_changes)
    return RemoteRegionDescriptor(
        endpoint="decode-worker",
        region_id=f"region-{rank}",
        address=4096 + rank * 128,
        length=128,
        device="cpu",
        rank=rank,
        rail=f"mlx5_{rank % 2}",
        backend_metadata=metadata,
    )


def make_delivery_identity(rank, **changes):
    values = {
        "protocol": PVD_TRANSFER_LIFECYCLE_PROTOCOL,
        "sender_epoch": f"vector-epoch-{rank // 2}",
        "receiver_epoch": "decode-epoch-1",
        "transfer_id": f"delivery-1:d{rank}",
        "region_id": f"region-{rank}",
        "generation": f"generation-{rank}",
        "shard_rank": rank,
        "key": KVEntryKey("model-1", "request-1", "entry-1"),
    }
    values.update(changes)
    return WriteIdentity(**values)


class IdentityFenceShard:
    def __init__(self, rank, *, legacy_reply=False):
        self.rank = rank
        self.legacy_reply = legacy_reply
        self.fenced = []

    async def fence_delivery(self, identity):
        self.fenced.append(identity)
        if self.legacy_reply:
            return {"fenced": True}
        return {**identity.to_dict(), "fenced": True}


def make_fence_coordinator(*, legacy_rank=None):
    shards = [
        IdentityFenceShard(rank, legacy_reply=rank == legacy_rank) for rank in range(2)
    ]
    coordinator = VectorCoordinator(shards)
    identities = {rank: make_delivery_identity(rank) for rank in range(4)}
    destinations = {rank: make_destination(rank) for rank in range(4)}
    coordinator.deliveries["delivery-1"] = DeliveryRecord(
        delivery_id="delivery-1",
        entry_key=KVEntryKey("model-1", "request-1", "entry-1"),
        destinations=destinations,
        source_shards={0: 0, 1: 0, 2: 1, 3: 1},
        write_identities=identities,
        # Business failure is not transport proof; the identity fence below is.
        state=DeliveryState.FAILED,
        created_at=time.monotonic(),
        deadline=time.monotonic() + 30,
    )
    return coordinator, shards, identities


def test_identity_round_trip_is_exact_and_rejects_tampering():
    identity = make_identity()
    encoded = identity.to_dict()

    assert WriteIdentity.from_dict(encoded) == identity

    tampered = dict(encoded)
    tampered["receiver_epoch"] = "decode-epoch-2"
    assert WriteIdentity.from_dict(tampered) != identity


@pytest.mark.parametrize(
    "change",
    [
        {"protocol": "pvd_transfer_lifecycle_v0"},
        {"sender_epoch": ""},
        {"shard_rank": True},
        {"key": {"model_instance_id": "model-1", "req_id": "request-1"}},
        {"extra": "rejected"},
    ],
)
def test_identity_rejects_old_or_incomplete_wire_values(change):
    encoded = make_identity().to_dict()
    encoded.update(change)

    with pytest.raises(ValueError):
        WriteIdentity.from_dict(encoded)


def test_authorization_rejects_closed_late_submission():
    identity = make_identity()
    authorization = WriteAuthorization(identity, ResourceGuard(object(), lambda: None))

    authorization.close()

    with pytest.raises(ValueError, match="closed"):
        authorization.begin(identity)
    assert authorization.fence(identity)["fenced"] is False


def test_authorization_rejects_a_second_or_different_begin():
    identity = make_identity()
    authorization = WriteAuthorization(identity, ResourceGuard(object(), lambda: None))

    authorization.begin(identity)
    with pytest.raises(ValueError, match="already begun"):
        authorization.begin(identity)
    with pytest.raises(ValueError, match="identity"):
        authorization.begin(make_identity(region_id="other-region"))


def test_authorization_fence_requires_closed_terminal_identity():
    identity = make_identity()
    authorization = WriteAuthorization(identity, ResourceGuard(object(), lambda: None))

    authorization.begin(identity)
    authorization.close()
    with pytest.raises(ValueError, match="terminal"):
        authorization.observe_terminal(identity, TransportState.UNKNOWN)
    assert authorization.fence(identity)["fenced"] is False

    authorization.observe_terminal(identity, TransportState.TERMINAL_SUCCESS)
    assert authorization.fence(identity) == {**identity.to_dict(), "fenced": True}


def test_authorization_rejects_not_submitted_after_begin():
    identity = make_identity()
    authorization = WriteAuthorization(identity, ResourceGuard(object(), lambda: None))
    authorization.begin(identity)
    authorization.close()

    with pytest.raises(ValueError, match="NOT_SUBMITTED"):
        authorization.observe_terminal(identity, TransportState.NOT_SUBMITTED)


def test_authorization_pins_until_closed_not_submitted_confirmation():
    released = []
    guard = ResourceGuard(object(), lambda: released.append(True))
    identity = make_identity()

    authorization = WriteAuthorization(identity, guard)
    guard.request_release()
    assert released == []

    authorization.close()
    authorization.observe_terminal(identity, TransportState.NOT_SUBMITTED)
    assert released == [True]


def test_same_identity_authorizations_do_not_share_a_resource_pin():
    released = []
    guard = ResourceGuard(object(), lambda: released.append(True))
    identity = make_identity()
    first = WriteAuthorization(identity, guard)
    second = WriteAuthorization(identity, guard)
    guard.request_release()
    first.close()
    first.observe_terminal(identity, TransportState.NOT_SUBMITTED)
    assert released == []
    second.close()
    second.observe_terminal(identity, TransportState.NOT_SUBMITTED)
    assert released == [True]


def test_authorization_retries_failed_local_release_without_reopening_gate():
    attempts = []

    def release():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("unregister failed")

    guard = ResourceGuard(object(), release)
    identity = make_identity()
    authorization = WriteAuthorization(identity, guard)
    guard.request_release()
    authorization.close()
    with pytest.raises(RuntimeError, match="unregister failed"):
        authorization.observe_terminal(identity, TransportState.NOT_SUBMITTED)
    assert guard.value is not None
    authorization.observe_terminal(identity, TransportState.NOT_SUBMITTED)
    assert guard.value is None
    assert len(attempts) == 2
    with pytest.raises(ValueError, match="closed"):
        authorization.begin(identity)


def test_authorization_rejects_untyped_terminal_state():
    identity = make_identity()
    authorization = WriteAuthorization(identity, ResourceGuard(object(), lambda: None))
    authorization.close()
    with pytest.raises(ValueError, match="terminal"):
        authorization.observe_terminal(identity, "terminal_success")
    assert authorization.fence(identity)["fenced"] is False


def test_fence_returns_stored_identity_instead_of_echoing_tampered_input():
    identity = make_identity()
    authorization = WriteAuthorization(identity, ResourceGuard(object(), lambda: None))
    authorization.close()
    authorization.observe_terminal(identity, TransportState.NOT_SUBMITTED)

    assert authorization.fence(make_identity(receiver_epoch="decode-epoch-old")) == {
        **identity.to_dict(),
        "fenced": False,
    }


class AuthorizingReserveShard:
    def __init__(self, rank, *, substitute_destination=False):
        self.rank = rank
        self.substitute_destination = substitute_destination

    async def reserve_delivery(self, key, delivery_id, destination):
        returned_destination = destination
        if self.substitute_destination:
            returned_destination = dataclasses.replace(
                destination, endpoint="substituted-worker"
            )
        return {
            "state": DeliveryState.D_RESERVED.value,
            "destination": returned_destination.to_dict(),
            "write_identity": make_delivery_identity(
                destination.rank,
                transfer_id=delivery_id,
                key=key,
                sender_epoch=f"vector-epoch-{self.rank}",
            ).to_dict(),
        }

    async def cancel_delivery(self, key, delivery_id, reason):
        return {"state": DeliveryState.CANCELLED.value}


def make_reserving_coordinator(*, substitute_rank=None):
    shards = [
        AuthorizingReserveShard(rank, substitute_destination=rank == substitute_rank)
        for rank in range(2)
    ]
    coordinator = VectorCoordinator(shards)
    key = KVEntryKey("model-1", "request-1", "entry-1")
    coordinator.entries[key] = SimpleNamespace(
        state=EntryState.STORED,
        active_delivery_count=0,
    )
    return coordinator, key


def test_reserve_stores_authoritative_identity_and_exact_destination():
    async def scenario():
        coordinator, key = make_reserving_coordinator()
        destinations = {rank: make_destination(rank) for rank in range(2)}

        delivery = await coordinator.reserve_delivery(
            key=key,
            delivery_id="delivery-1",
            destinations=destinations,
        )

        assert delivery.write_identities == {
            rank: make_delivery_identity(rank, sender_epoch=f"vector-epoch-{rank}")
            for rank in range(2)
        }
        assert delivery.destinations == destinations

    asyncio.run(scenario())


def test_reserve_rejects_same_identity_with_substituted_destination():
    async def scenario():
        coordinator, key = make_reserving_coordinator(substitute_rank=1)

        with pytest.raises(CoordinatorError, match="destination"):
            await coordinator.reserve_delivery(
                key=key,
                delivery_id="delivery-1",
                destinations={rank: make_destination(rank) for rank in range(2)},
            )

    asyncio.run(scenario())


def test_identity_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_identity().region_id = "changed"


def test_coordinator_fence_validates_exact_tp4_identity_set():
    async def scenario():
        coordinator, shards, identities = make_fence_coordinator()

        reply = await coordinator.fence_retrieval(
            "delivery-1",
            [identities[rank].to_dict() for rank in range(4)],
        )

        assert reply == {
            "delivery_id": "delivery-1",
            "identities": [identities[rank].to_dict() for rank in range(4)],
            "fenced": True,
        }
        assert shards[0].fenced == [identities[0], identities[1]]
        assert shards[1].fenced == [identities[2], identities[3]]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "bad_identities",
    [
        "missing",
        "duplicate",
        "wrong-key",
        "old-sender-epoch",
        "old-receiver-epoch",
        "wrong-generation",
        "wrong-region",
    ],
)
def test_coordinator_fence_rejects_incomplete_or_tampered_identity_set(bad_identities):
    async def scenario():
        coordinator, _, identities = make_fence_coordinator()
        supplied = [identities[rank].to_dict() for rank in range(4)]
        if bad_identities == "missing":
            supplied.pop()
        elif bad_identities == "duplicate":
            supplied[-1] = supplied[0]
        elif bad_identities == "wrong-key":
            supplied[2] = make_delivery_identity(
                2, key=KVEntryKey("model-1", "other-request", "entry-1")
            ).to_dict()
        elif bad_identities == "old-sender-epoch":
            supplied[2] = make_delivery_identity(
                2, sender_epoch="vector-epoch-old"
            ).to_dict()
        elif bad_identities == "old-receiver-epoch":
            supplied[2] = make_delivery_identity(
                2, receiver_epoch="decode-epoch-old"
            ).to_dict()
        elif bad_identities == "wrong-generation":
            supplied[2] = make_delivery_identity(
                2, generation="generation-old"
            ).to_dict()
        else:
            supplied[2] = make_delivery_identity(2, region_id="region-old").to_dict()

        with pytest.raises(CoordinatorError, match="identity"):
            await coordinator.fence_retrieval("delivery-1", supplied)

    asyncio.run(scenario())


def test_coordinator_rejects_legacy_shard_fence_reply():
    async def scenario():
        coordinator, _, identities = make_fence_coordinator(legacy_rank=1)

        with pytest.raises(CoordinatorError, match="identity"):
            await coordinator.fence_retrieval(
                "delivery-1",
                [identities[rank].to_dict() for rank in range(4)],
            )

    asyncio.run(scenario())


def test_local_shard_does_not_promote_legacy_boolean_fence():
    class LegacyStore:
        rank = 0

        def fence_delivery(self, key, delivery_id):
            pytest.fail("legacy cancellation must not release unprotected V pages")

    async def scenario():
        identity = make_delivery_identity(0)

        reply = await LocalShardClient(LegacyStore()).fence_delivery(identity)

        assert reply == {
            **identity.to_dict(),
            "fenced": False,
            "reason": "transport_terminal_unverified",
        }

    asyncio.run(scenario())


def test_http_fence_round_trip_carries_full_identities_and_rejects_legacy_reply():
    async def scenario():
        from aiohttp.test_utils import TestServer
        from sglang.srt.disaggregation.pvd.client import (
            PVDControlPlaneError,
            PVDCoordinatorClient,
        )
        from sglang.srt.disaggregation.pvd.control_server import create_coordinator_app

        coordinator, _, identities = make_fence_coordinator()
        async with TestServer(create_coordinator_app(coordinator)) as server:
            client = PVDCoordinatorClient(str(server.make_url("")))
            try:
                reply = await client.fence_retrieval(
                    "delivery-1",
                    [identities[rank].to_dict() for rank in range(4)],
                )
                assert reply["identities"] == [
                    identities[rank].to_dict() for rank in range(4)
                ]
                assert reply["fenced"] is True
            finally:
                await client.close()

        coordinator, _, identities = make_fence_coordinator(legacy_rank=1)
        async with TestServer(create_coordinator_app(coordinator)) as server:
            client = PVDCoordinatorClient(str(server.make_url("")))
            try:
                with pytest.raises(PVDControlPlaneError, match="identity"):
                    await client.fence_retrieval(
                        "delivery-1",
                        [identities[rank].to_dict() for rank in range(4)],
                    )
            finally:
                await client.close()

    asyncio.run(scenario())


def test_internal_http_shard_fence_passes_identity_but_stays_fail_closed():
    class LegacyStore:
        rank = 0

        def __init__(self):
            self.calls = []

        def fence_delivery(self, key, delivery_id):
            self.calls.append((key, delivery_id))
            return {"delivery_id": delivery_id, "fenced": True}

    async def scenario():
        from aiohttp.test_utils import TestServer
        from sglang.srt.disaggregation.pvd.control_server import (
            HttpShardClient,
            create_shard_app,
        )

        store = LegacyStore()
        identity = make_delivery_identity(0)
        async with TestServer(create_shard_app(store)) as server:
            client = HttpShardClient(0, str(server.make_url("")))
            try:
                reply = await client.fence_delivery(identity)
            finally:
                await client.close()

        assert store.calls == []
        assert reply == {
            **identity.to_dict(),
            "fenced": False,
            "reason": "transport_terminal_unverified",
        }

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "malformed", ["legacy", "epoch", "missing", "duplicate", "id", "boolean"]
)
def test_client_rejects_untrusted_successful_http_fence_reply(malformed):
    async def scenario():
        from aiohttp import web
        from aiohttp.test_utils import TestServer
        from sglang.srt.disaggregation.pvd.client import (
            PVDControlPlaneError,
            PVDCoordinatorClient,
        )

        identities = [make_delivery_identity(rank).to_dict() for rank in range(2)]

        async def fence(request):
            payload = await request.json()
            assert payload == {"delivery_id": "delivery-1", "identities": identities}
            reply = {
                "delivery_id": "delivery-1",
                "identities": [dict(v) for v in identities],
                "fenced": True,
            }
            if malformed == "legacy":
                reply.pop("identities")
            elif malformed == "epoch":
                reply["identities"][0]["sender_epoch"] = "stale-sender"
            elif malformed == "missing":
                reply["identities"].pop()
            elif malformed == "duplicate":
                reply["identities"][1] = reply["identities"][0]
            elif malformed == "id":
                reply["delivery_id"] = "other-delivery"
            else:
                reply["fenced"] = 1
            return web.json_response(reply)

        app = web.Application()
        app.router.add_post("/v1/retrievals/fence", fence)
        async with TestServer(app) as server:
            client = PVDCoordinatorClient(str(server.make_url("")))
            try:
                with pytest.raises(PVDControlPlaneError, match="fence"):
                    await client.fence_retrieval("delivery-1", identities)
            finally:
                await client.close()

    asyncio.run(scenario())


def test_forged_fence_does_not_close_an_existing_authorization():
    async def scenario():
        coordinator, shards, identities = make_fence_coordinator()
        supplied = [v.to_dict() for v in identities.values()]
        supplied[0]["generation"] = "stale-generation"
        with pytest.raises(CoordinatorError):
            await coordinator.fence_retrieval("delivery-1", supplied)
        assert "delivery-1" not in coordinator._fenced_retrievals
        assert all(not shard.fenced for shard in shards)

    asyncio.run(scenario())


def test_fence_blocks_direct_reserve_and_start_even_when_unconfirmed():
    async def scenario():
        coordinator, _, identities = make_fence_coordinator(legacy_rank=1)
        coordinator.deliveries["delivery-1"].state = DeliveryState.D_RESERVED
        key = identities[0].key
        coordinator.entries[key] = SimpleNamespace(state=EntryState.STORED)
        with pytest.raises(CoordinatorError, match="identity"):
            await coordinator.fence_retrieval(
                "delivery-1", [v.to_dict() for v in identities.values()]
            )
        with pytest.raises(CoordinatorError, match="fenced"):
            await coordinator.reserve_delivery(
                key=key,
                delivery_id="delivery-1",
                destinations={rank: make_destination(rank) for rank in range(2)},
            )
        with pytest.raises(CoordinatorError, match="fenced"):
            await coordinator.start_delivery("delivery-1")

    asyncio.run(scenario())


def test_reservation_owns_a_snapshot_of_destination_metadata():
    async def scenario():
        coordinator, key = make_reserving_coordinator()
        destinations = {rank: make_destination(rank) for rank in range(2)}
        delivery = await coordinator.reserve_delivery(
            key=key, delivery_id="delivery-1", destinations=destinations
        )
        destinations[0].backend_metadata["layout_token"] = "substituted-layout"
        assert delivery.destinations[0].backend_metadata["layout_token"] == "layout-0"
        with pytest.raises(CoordinatorError, match="different parameters"):
            await coordinator.reserve_delivery(
                key=key, delivery_id="delivery-1", destinations=destinations
            )

    asyncio.run(scenario())


def test_begin_close_race_never_reopens_authorization():
    identity = make_identity()
    released = []
    guard = ResourceGuard(object(), lambda: released.append(True))
    authorization = WriteAuthorization(identity, guard)
    guard.request_release()
    barrier = threading.Barrier(2)

    def begin():
        barrier.wait(timeout=5)
        try:
            authorization.begin(identity)
        except ValueError:
            return False
        return True

    def close():
        barrier.wait(timeout=5)
        authorization.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        beginning = executor.submit(begin)
        closing = executor.submit(close)
        began = beginning.result(timeout=5)
        closing.result(timeout=5)
    assert released == []
    assert authorization.fence(identity)["fenced"] is False
    with pytest.raises(ValueError, match="closed"):
        authorization.begin(identity)
    authorization.observe_terminal(
        identity,
        TransportState.TERMINAL_SUCCESS if began else TransportState.NOT_SUBMITTED,
    )
    assert released == [True]


def test_terminal_replay_calls_release_once_and_callback_can_inspect_fence():
    identity = make_identity()
    calls = []

    def release():
        calls.append(authorization.fence(identity)["fenced"])

    guard = ResourceGuard(object(), release)
    authorization = WriteAuthorization(identity, guard)
    guard.request_release()
    authorization.close()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                authorization.observe_terminal, identity, TransportState.NOT_SUBMITTED
            )
            for _ in range(2)
        ]
        for future in futures:
            future.result(timeout=5)
    assert calls == [True]


@pytest.mark.parametrize(
    "state", [TransportState.IN_FLIGHT, TransportState.DRAINING, TransportState.UNKNOWN]
)
def test_nonterminal_state_never_unpins(state):
    released = []
    identity = make_identity()
    guard = ResourceGuard(object(), lambda: released.append(True))
    authorization = WriteAuthorization(identity, guard)
    authorization.begin(identity)
    authorization.close()
    guard.request_release()
    with pytest.raises(ValueError, match="terminal"):
        authorization.observe_terminal(identity, state)
    assert released == []
    assert authorization.fence(identity)["fenced"] is False


@pytest.mark.parametrize(
    "bad_reply",
    ["unreachable", "missing-identity", "old-epoch", "non-object"],
)
def test_one_bad_shard_prevents_aggregate_fence(bad_reply):
    async def scenario():
        coordinator, shards, identities = make_fence_coordinator()

        async def fence(identity):
            if bad_reply == "unreachable":
                raise TimeoutError("rank1 control connection lost")
            if bad_reply == "non-object":
                return None
            reply = {**identity.to_dict(), "fenced": True}
            if bad_reply == "missing-identity":
                del reply["region_id"]
            elif bad_reply == "old-epoch":
                reply["sender_epoch"] = "old-worker"
            else:
                reply["fenced"] = False
            return reply

        shards[1].fence_delivery = fence
        with pytest.raises(CoordinatorError):
            await coordinator.fence_retrieval(
                "delivery-1", [v.to_dict() for v in identities.values()]
            )
        assert "delivery-1" in coordinator._fenced_retrievals

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        {"delivery_id": "delivery-1"},
        {"delivery_id": "delivery-1", "identities": {}},
        {"delivery_id": "delivery-1", "identities": [{"fenced": True}]},
    ],
)
def test_http_fence_rejects_legacy_requests_without_touching_shards(payload):
    async def scenario():
        from aiohttp.test_utils import TestClient, TestServer
        from sglang.srt.disaggregation.pvd.control_server import create_coordinator_app

        coordinator, shards, _ = make_fence_coordinator()
        async with TestClient(
            TestServer(create_coordinator_app(coordinator))
        ) as client:
            response = await client.post("/v1/retrievals/fence", json=payload)
            assert response.status in (400, 409)
            assert "error" in await response.json()
        assert all(not shard.fenced for shard in shards)
        assert "delivery-1" not in coordinator._fenced_retrievals

    asyncio.run(scenario())


@pytest.mark.parametrize("missing_ranks", [{0}, {0, 1}])
def test_lifecycle_reservation_rejects_missing_worker_authorizations(missing_ranks):
    async def scenario():
        coordinator, key = make_reserving_coordinator()

        async def legacy_reserve(key, delivery_id, destination):
            return {
                "state": DeliveryState.D_RESERVED.value,
                "destination": destination.to_dict(),
            }

        for rank in missing_ranks:
            coordinator.shards[rank].reserve_delivery = legacy_reserve
        with pytest.raises(CoordinatorError, match="identity"):
            await coordinator.reserve_delivery(
                key=key,
                delivery_id="delivery-1",
                destinations={rank: make_destination(rank) for rank in range(2)},
            )

    asyncio.run(scenario())


def test_matching_pending_fence_is_not_a_success_or_business_release():
    async def scenario():
        coordinator, shards, identities = make_fence_coordinator()
        record = coordinator.deliveries["delivery-1"]
        record.state = DeliveryState.D_RESERVED

        async def pending(identity):
            return {**identity.to_dict(), "fenced": False}

        shards[1].fence_delivery = pending
        reply = await coordinator.fence_retrieval(
            "delivery-1", [v.to_dict() for v in identities.values()]
        )
        assert reply["fenced"] is False
        assert reply["identities"] == [v.to_dict() for v in identities.values()]
        assert record.state == DeliveryState.D_RESERVED
        assert "delivery-1" in coordinator._fenced_retrievals

    asyncio.run(scenario())
