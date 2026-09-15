"""D receive-buffer lifetime, identity fences and pending-refresh handling.

These cases run the real VectorKVStore, VectorCoordinator, WriteAuthorization
and PVDDecodeSession. Only the native transport boundary is controllable.
CPU tensors and a simulated CUDA synchronization boundary do not prove
GPUDirect/RDMA ordering on real hardware.
"""

import asyncio
import dataclasses
import uuid
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeSession
from sglang.srt.disaggregation.pvd.protocol import (
    PVD_GENERATION_METADATA_KEY,
    PVD_RECEIVER_EPOCH_METADATA_KEY,
    KVEntryKey,
)
from sglang.srt.disaggregation.pvd.transfer_engine import (
    FakeTransferEngine,
    MemorySlice,
    TransferStatus,
)
from test_pvd3 import make_ready_entry, make_vector

SENDER_EPOCH = "v-worker-epoch"


def _with_close_retention(manager):
    manager.pending_decode_closes = []
    manager.retain_pending_close = lambda session: manager.pending_decode_closes.append(
        session
    )
    return manager


class RecordingClient:
    """Fence/poll client whose replies each test controls exactly."""

    def __init__(self):
        self.fence_reply = None
        self.fence_error = None
        self.poll_reply = None
        self.poll_error = None
        self.fence_calls = []
        self.poll_calls = []
        self.released = []

    async def fence_retrieval(self, delivery_id, identities):
        self.fence_calls.append((delivery_id, identities))
        if self.fence_error is not None:
            raise self.fence_error
        reply = dict(self.fence_reply or {})
        reply.setdefault("delivery_id", delivery_id)
        reply.setdefault("identities", identities)
        return reply

    async def poll_delivery(self, delivery_id):
        self.poll_calls.append(delivery_id)
        if self.poll_error is not None:
            raise self.poll_error
        return self.poll_reply

    async def release_consumer(self, key, consumer_id):
        self.released.append(consumer_id)
        return None


def make_session(client=None, *, tp_rank=0, key=None, layout=None, rail="mlx5_0"):
    key = key or KVEntryKey("model", "decode", "decode")
    client = client or RecordingClient()
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
            tp_rank=tp_rank,
            rail=rail,
            transfer_engine=engine,
            layout=(lambda: layout)
            if layout is not None
            else (lambda: SimpleNamespace(to_dict=lambda: {})),
            scheduler=SimpleNamespace(
                server_args=SimpleNamespace(pvd_kv_refresh_interval=4)
            ),
        )
    )
    return PVDDecodeSession(manager, req), manager, client, engine, pool


def mr_live(engine, registration):
    return (
        engine.submit_put(
            MemorySlice(registration, 0, 16), registration.descriptor
        ).status
        == TransferStatus.SUCCESS
    )


def reply_for(session, identities):
    return {
        "write_identities": {
            str(rank): identity.to_dict() for rank, identity in identities.items()
        }
    }


# --------------------------------------------------------------------------
# Destination publication
# --------------------------------------------------------------------------


def test_destination_is_pinned_and_stamped_before_publication():
    session, _, _, engine, _ = make_session()
    payload = session.prepare([1, 2])
    metadata = payload["destination"]["backend_metadata"]
    assert metadata[PVD_RECEIVER_EPOCH_METADATA_KEY] == session.receiver_epoch
    assert metadata[PVD_GENERATION_METADATA_KEY] == session.generation
    # Pinned the moment the descriptor exists, so a release request cannot
    # deregister it while a remote writer may still be authorized.
    assert session._refresh_owner is not None
    registration = session.registration
    session.receive_guard.request_release()
    assert mr_live(engine, registration)
    session.release_refresh()
    assert not mr_live(engine, registration)


def test_each_refresh_publishes_a_new_generation():
    session, _, _, _, _ = make_session()
    first = session.prepare([1, 2])
    session.release_refresh()
    session.clock.complete(first["delivery_id"])
    session.req.output_ids.extend([1, 2, 3, 4])
    second = session.prepare([1, 2])
    assert (
        first["destination"]["backend_metadata"][PVD_GENERATION_METADATA_KEY]
        != second["destination"]["backend_metadata"][PVD_GENERATION_METADATA_KEY]
    )
    assert session.registration.descriptor.region_id == (
        session.registration.descriptor.region_id
    )


def test_staging_cannot_be_reused_while_a_refresh_is_unfenced():
    session, _, _, _, _ = make_session()
    session.prepare([1, 2])
    with pytest.raises(RuntimeError, match="unfenced"):
        session.prepare([1, 2])


# --------------------------------------------------------------------------
# Identity adoption
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("receiver_epoch", "another-d-epoch"),
        ("generation", "0" * 32),
        ("region_id", "0" * 32),
        ("transfer_id", "d:refresh:0:d9"),
        ("shard_rank", 3),
    ],
)
def test_adopt_rejects_an_identity_this_rank_did_not_publish(field, value):
    session, _, _, _, _ = make_session()
    session.prepare([1, 2])
    good = session.expected_identity(0, SENDER_EPOCH)
    forged = dataclasses.replace(good, **{field: value})
    with pytest.raises(ValueError):
        session.adopt_identities(reply_for(session, {0: forged}))
    assert not session.identities


def test_adopt_requires_this_ranks_identity():
    session, _, _, _, _ = make_session(tp_rank=0)
    session.prepare([1, 2])
    other = dataclasses.replace(
        session.expected_identity(0, SENDER_EPOCH),
        shard_rank=1,
        transfer_id=f"{session.clock.pending[0]}:d1",
    )
    with pytest.raises(ValueError, match="omits this D rank"):
        session.adopt_identities(reply_for(session, {1: other}))


def test_adopt_rejects_a_legacy_reply_without_identities():
    session, _, _, _, _ = make_session()
    session.prepare([1, 2])
    with pytest.raises(ValueError, match="no write identities"):
        session.adopt_identities({"state": "delivered"})
    with pytest.raises(ValueError, match="no write identities"):
        session.adopt_identities({"write_identities": {}})


def test_adopt_accepts_peer_ranks_without_verifying_their_private_fields():
    """Rank 0 carries peer identities to fence, but does not own their fields."""
    session, _, _, _, _ = make_session(tp_rank=0)
    session.prepare([1, 2])
    own = session.expected_identity(0, SENDER_EPOCH)
    peer = dataclasses.replace(
        own,
        shard_rank=1,
        transfer_id=f"{session.clock.pending[0]}:d1",
        # A peer rank legitimately has its own region, generation and V epoch.
        region_id=uuid.uuid4().hex,
        generation=uuid.uuid4().hex,
        receiver_epoch="peer-d-epoch",
        sender_epoch="another-v-shard-epoch",
    )
    session.adopt_identities(reply_for(session, {0: own, 1: peer}))
    assert set(session.identities) == {0, 1}


# --------------------------------------------------------------------------
# Fencing and retention
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode", ["http-error", "pending", "wrong-delivery", "missing-fenced-flag"]
)
def test_unproven_fence_never_releases_the_receive_buffer(mode):
    async def scenario():
        session, manager, client, engine, _ = make_session()
        session.prepare([1, 2])
        session.identities = {0: session.expected_identity(0, SENDER_EPOCH)}
        registration = session.registration
        if mode == "http-error":
            client.fence_error = TimeoutError("V unreachable")
        elif mode == "pending":
            client.fence_reply = {"fenced": False}
        elif mode == "wrong-delivery":
            client.fence_reply = {"delivery_id": "other", "fenced": True}
        else:
            client.fence_reply = {}

        assert await session.progress_close() is False
        assert session.registration is registration
        assert mr_live(engine, registration)

        assert await session.close() is False
        assert session in manager.pending_decode_closes
        assert session.registration is registration
        assert mr_live(engine, registration)
        assert client.released == []

        client.fence_error = None
        client.fence_reply = {"fenced": True}
        assert await session.progress_close() is True
        assert await session.close() is True
        assert session.registration is None
        assert not mr_live(engine, registration)

    asyncio.run(scenario())


def test_close_recovers_saved_identities_before_fencing():
    async def scenario():
        session, _, client, engine, _ = make_session()
        session.prepare([1, 2])
        identity = session.expected_identity(0, SENDER_EPOCH)
        # No identities were adopted: the retrieval never completed.
        assert not session.identities
        client.poll_reply = reply_for(session, {0: identity})
        client.fence_reply = {"fenced": True}
        assert await session.progress_close() is True
        assert client.poll_calls == [session.clock.pending[0]]
        assert session.identities == {0: identity}

    asyncio.run(scenario())


def test_unrecoverable_identities_retain_the_buffer():
    async def scenario():
        session, manager, client, engine, _ = make_session()
        session.prepare([1, 2])
        registration = session.registration
        client.poll_error = RuntimeError("unknown delivery")
        assert await session.progress_close() is False
        assert client.fence_calls == []
        assert await session.close() is False
        assert session in manager.pending_decode_closes
        assert mr_live(engine, registration)

    asyncio.run(scenario())


def test_a_forged_recovered_identity_cannot_unlock_the_fence():
    async def scenario():
        session, _, client, engine, _ = make_session()
        session.prepare([1, 2])
        forged = dataclasses.replace(
            session.expected_identity(0, SENDER_EPOCH),
            generation=uuid.uuid4().hex,
        )
        client.poll_reply = reply_for(session, {0: forged})
        client.fence_reply = {"fenced": True}
        assert await session.progress_close() is False
        assert client.fence_calls == []
        assert mr_live(engine, session.registration)

    asyncio.run(scenario())


def test_retained_sessions_are_driven_by_the_manager_not_the_request():
    async def scenario():
        from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeRefresher

        session, manager, client, engine, _ = make_session()
        session.prepare([1, 2])
        session.identities = {0: session.expected_identity(0, SENDER_EPOCH)}
        registration = session.registration
        client.fence_reply = {"fenced": False}
        assert await session.close() is False
        # The request object is gone; only the manager still holds this.
        session.req = None
        refresher = PVDDecodeRefresher(manager)
        assert await refresher.progress_pending_closes() == 1
        assert mr_live(engine, registration)
        client.fence_reply = {"fenced": True}
        assert await refresher.progress_pending_closes() == 0
        assert not manager.pending_decode_closes
        assert not mr_live(engine, registration)

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# End-to-end against the real V coordinator
# --------------------------------------------------------------------------


def test_live_refresh_fences_with_real_v_identities():
    """Real V shards issue the identities; the fence consumes exactly those."""

    async def scenario():
        engine, stores, coordinator = make_vector()
        key = await make_ready_entry(coordinator, engine, "decode-live")

        class Client:
            async def fence_retrieval(self, delivery_id, identities):
                return await coordinator.fence_retrieval(delivery_id, identities)

            async def poll_delivery(self, delivery_id):
                return (await coordinator.poll_delivery(delivery_id)).to_dict()

            async def release_consumer(self, key, consumer_id):
                return await coordinator.release_consumer(key, consumer_id)

        client = Client()
        storage_layout = coordinator.entries[key].manifest.layout
        sessions, payloads = [], []
        for rank in range(2):
            session, manager, _, _, _ = make_session(
                client,
                tp_rank=rank,
                key=key,
                layout=storage_layout,
                rail=f"mlx5_{rank}",
            )
            manager.transfer_engine = engine
            sessions.append(session)
            payloads.append(session.prepare([0, 1]))

        head = payloads[0]
        results = await coordinator.retrieve(
            [
                {
                    **{k: v for k, v in head.items() if k != "destination"},
                    "destinations": {
                        str(rank): payloads[rank]["destination"] for rank in range(2)
                    },
                }
            ]
        )
        assert len(results) == 1
        result = results[0]
        assert result["state"] == "delivered", result.get("error")

        registrations = [s.registration for s in sessions]
        for rank, session in enumerate(sessions):
            # Each rank adopts the set, but verifies only its own authorization.
            session.adopt_identities(result)
            assert set(session.identities) == {0, 1}
            assert session.identities[rank].receiver_epoch == session.receiver_epoch
            assert session.identities[rank].region_id == (
                registrations[rank].descriptor.region_id
            )
            assert session.identities[rank].generation == session.generation
        # V stamped its own storage-shard incarnations as the senders.
        senders = {i.sender_epoch for i in sessions[0].identities.values()}
        assert senders == {stores[0].worker_epoch, stores[1].worker_epoch}

        assert await sessions[0].progress_close() is True
        assert mr_live(engine, registrations[0]), "fencing must not deregister"
        assert await sessions[0].close() is True
        assert not mr_live(engine, registrations[0])
        # Fencing rank 0 leaves rank 1's own destination untouched.
        assert mr_live(engine, registrations[1])
        assert sessions[1].registration is registrations[1]

    asyncio.run(scenario())


def test_v_writing_is_polled_to_a_terminal_state_not_treated_as_failure():
    async def scenario():
        from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeRefresher

        calls = []

        class Client:
            async def poll_delivery(self, delivery_id):
                calls.append(delivery_id)
                if len(calls) < 3:
                    return {"state": "v_writing", "error": None}
                return {
                    "state": "delivered",
                    "error": None,
                    "write_identities": {"0": {"placeholder": True}},
                }

        refresher = PVDDecodeRefresher(SimpleNamespace())
        result = await refresher._drive_delivery(
            Client(), {"delivery_id": "x", "state": "v_writing"}
        )
        assert result["state"] == "delivered"
        assert len(calls) == 3
        assert result["write_identities"] == {"0": {"placeholder": True}}

    asyncio.run(scenario())


def test_a_delivery_that_never_terminates_is_reported_as_failed():
    async def scenario():
        from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeRefresher

        class Client:
            async def poll_delivery(self, delivery_id):
                return {"state": "v_writing", "error": None}

        refresher = PVDDecodeRefresher(SimpleNamespace())
        refresher.POLL_ATTEMPTS = 3
        refresher.POLL_INTERVAL_SECONDS = 0
        result = await refresher._drive_delivery(
            Client(), {"delivery_id": "x", "state": "v_writing"}
        )
        # Failure here is a business result. It releases nothing: the session's
        # destination pin is dropped only by release_refresh or a fence.
        assert result["state"] == "failed"
        assert "terminal state" in result["error"]

    asyncio.run(scenario())


def test_generated_token_kv_survives_a_refresh_of_the_shared_tail_page():
    from sglang.srt.disaggregation.pvd.kv_packer import pack_full_prompt_kv

    session, _, _, _, pool = make_session()
    payload = session.prepare([1, 2])
    source = SimpleNamespace(
        k_buffer=[torch.arange(8, dtype=torch.float32).reshape(8, 1, 1)], v_buffer=[]
    )
    session.staging.copy_(pack_full_prompt_kv(source, [0, 1], page_size=4).tensor)
    # Tokens beyond the 5-token prompt live in the prompt's final, partially
    # filled page. They are D-generated KV and must not be overwritten.
    pool.k_buffer[0][9:] = 88
    session.unpack(
        {
            "key": session.key.to_dict(),
            "sequence_id": session.key.req_id,
            "delivery_id": payload["delivery_id"],
            "selection": "full_prompt",
            "token_ranges": [[0, 5]],
            "state": "delivered",
        }
    )
    assert pool.k_buffer[0][4:9].flatten().tolist() == [0, 1, 2, 3, 4]
    assert torch.all(pool.k_buffer[0][9:] == 88)
