"""Combined shard RPC with real HTTP, store gates and controlled CPU writes."""

import asyncio
import threading
from dataclasses import replace

import pytest
import torch
from sglang.srt.disaggregation.pvd.coordinator import CoordinatorError
from sglang.srt.disaggregation.pvd.request_state import DeliveryState
from sglang.srt.disaggregation.pvd.sparse_receiver import SparseReceiveRegistry
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget, TransportState
from test_pvd_sparse_receiver import receiving


async def combined(case, destination=None):
    # These client-level probes own the prepared destination, so publication
    # must be latched before awaiting even when the response is lost.
    case.record._published = True
    return await case.client.reserve_and_start_delivery(
        case.entry.key,
        case.record.identity.transfer_id,
        destination or case.record._registration.descriptor,
    )


def delivery(case):
    return case.store.entries[case.entry.key].deliveries[
        case.record.identity.transfer_id
    ]


def test_combined_rpc_writes_exact_sparse_payload_and_preserves_terminal_proof():
    async def run():
        async with receiving() as c:
            reply = await combined(c)
            assert reply["state"] == DeliveryState.V_WRITING.value
            assert reply["transport_state"] == TransportState.IN_FLIGHT.value
            assert reply["write_identity"] == c.record.identity.to_dict()
            assert reply["write_fence"] == {
                **c.record.identity.to_dict(),
                "fenced": False,
            }
            assert reply["destination"] == c.record._registration.descriptor.to_dict()
            assert reply["sparse_fingerprint"] == c.manifest.fingerprint
            assert reply["transferred_bytes"] == 0
            assert len(c.engine.pending) == 1
            assert c.engine.total_put_bytes == 0
            c.engine.finish(delivery(c).transfer_handle)
            completed = await c.client.poll_delivery(
                c.entry.key, c.record.identity.transfer_id
            )
            assert completed["state"] == DeliveryState.DELIVERED.value
            assert completed["transferred_bytes"] == c.manifest.nbytes
            assert completed["write_fence"]["fenced"]
            for payload in c.manifest.payload_views(c.record._registration.buffer):
                spec = payload.spec
                expected = torch.stack(
                    (
                        c.pool.k_buffer[spec.layer][list(spec.token_ids), spec.kv_head],
                        c.pool.v_buffer[spec.layer][list(spec.token_ids), spec.kv_head],
                    )
                )
                torch.testing.assert_close(payload.tensor, expected, rtol=0, atol=0)
            await c.client.ack_delivery(c.entry.key, c.record.identity.transfer_id)

    asyncio.run(run())


def test_exact_retry_after_lost_reply_never_resubmits(monkeypatch):
    async def run():
        async with receiving() as c:
            request = c.client._request

            async def lose_reply(method, path, payload=None):
                await request(method, path, payload)
                raise TimeoutError("reply lost after native submission")

            monkeypatch.setattr(c.client, "_request", lose_reply)
            with pytest.raises(TimeoutError, match="reply lost"):
                await combined(c)
            first = delivery(c)
            first_handle = first.transfer_handle
            assert len(c.engine.pending) == 1
            assert c.record.identity.region_id not in c.engine.released
            monkeypatch.setattr(c.client, "_request", request)
            for _ in range(2):
                reply = await combined(c)
                assert reply["state"] == DeliveryState.V_WRITING.value
                assert delivery(c) is first
                assert delivery(c).transfer_handle is first_handle
                assert len(c.engine.pending) == 1
            c.engine.finish(first_handle)
            await c.client.poll_delivery(c.entry.key, c.record.identity.transfer_id)
            reply = await combined(c)
            assert reply["state"] == DeliveryState.DELIVERED.value
            assert not c.engine.pending
            assert c.engine.total_put_bytes == c.manifest.nbytes

    asyncio.run(run())


@pytest.mark.parametrize("field", ["region_id", "generation", "metadata"])
def test_retry_with_changed_destination_rejected_before_another_write(field):
    async def run():
        async with receiving() as c:
            await combined(c)
            first_handle = delivery(c).transfer_handle
            descriptor = c.record._registration.descriptor
            if field == "region_id":
                changed = replace(descriptor, region_id="other-receive-region")
            else:
                metadata = dict(descriptor.backend_metadata)
                metadata[
                    "pvd_generation" if field == "generation" else "extra"
                ] = "other"
                changed = replace(descriptor, backend_metadata=metadata)
            with pytest.raises(CoordinatorError, match="different destination"):
                await combined(c, changed)
            assert delivery(c).transfer_handle is first_handle
            assert len(c.engine.pending) == 1

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["cancelled", "absent_fence"])
def test_cancel_or_absent_fence_tombstone_refuses_combined_retry(mode):
    async def run():
        async with receiving() as c:
            if mode == "cancelled":
                await combined(c)
                cancelled = await c.client.cancel_delivery(
                    c.entry.key, c.record.identity.transfer_id, "request cancelled"
                )
                assert cancelled["state"] == DeliveryState.CANCELLED.value
                assert not cancelled["write_fence"]["fenced"]
                original_handle = delivery(c).transfer_handle
            else:
                proof = await c.client.fence_delivery(c.record.identity)
                assert proof["fenced"]
                assert not c.engine.pending
            with pytest.raises(CoordinatorError, match="fenced|isolated"):
                await combined(c)
            if mode == "cancelled":
                assert delivery(c).transfer_handle is original_handle
                assert len(c.engine.pending) == 1
            else:
                assert not c.store.entries[c.entry.key].deliveries
                assert not c.engine.pending

    asyncio.run(run())


def test_fence_between_reserve_and_start_prevents_native_submission(monkeypatch):
    async def run():
        async with receiving() as c:
            reserve = c.store.reserve_delivery

            def fence_after_reserve(*args):
                result = reserve(*args)
                assert c.store.fence_write(c.record.identity)["fenced"]
                return result

            monkeypatch.setattr(c.store, "reserve_delivery", fence_after_reserve)
            with pytest.raises(CoordinatorError, match="fenced"):
                await combined(c)
            assert not c.engine.pending
            assert delivery(c).local_terminal == TransportState.NOT_SUBMITTED
            assert c.store.fence_write(c.record.identity)["fenced"]

    asyncio.run(run())


def test_unknown_native_write_retains_source_destination_and_budget():
    async def run():
        async with receiving() as c:
            await combined(c)
            d = delivery(c)
            d.transfer_handle.transport_state = TransportState.UNKNOWN
            source = c.engine.pending[d.transfer_handle.transfer_id][0].registration
            reply = await c.client.fence_delivery(c.record.identity)
            assert not reply["fenced"]
            assert source.descriptor.region_id not in c.engine.released
            assert c.record.identity.region_id not in c.engine.released
            assert c.engine.lifecycle_manager.budget.snapshot()[
                "used_staging_bytes"
            ] == c.manifest.nbytes
            assert c.registry.budget.snapshot()["used_staging_bytes"] == c.manifest.nbytes
            assert not await c.record.close()
            with pytest.raises(CoordinatorError, match="fenced|isolated"):
                await combined(c)
            # Only the controlled native terminal proof permits cleanup.
            c.engine.finish(d.transfer_handle)
            c.store.progress_transfers()
            assert await c.record.close()
            assert c.engine.lifecycle_manager.budget.snapshot()[
                "used_staging_bytes"
            ] == 0
            assert c.registry.budget.snapshot()["used_staging_bytes"] == 0

    asyncio.run(run())


def test_validated_calls_share_one_worker_and_keep_http_loop_responsive(monkeypatch):
    async def run():
        async with receiving() as c:
            reserve = c.store.reserve_delivery
            start = c.store.start_delivery
            thread_ids = []
            entered, release = threading.Event(), threading.Event()
            loop_thread = threading.get_ident()

            def checked_reserve(*args):
                thread_ids.append(threading.get_ident())
                return reserve(*args)

            def blocked_start(*args):
                thread_ids.append(threading.get_ident())
                entered.set()
                assert release.wait(5), "test did not release native worker"
                return start(*args)

            monkeypatch.setattr(c.store, "reserve_delivery", checked_reserve)
            monkeypatch.setattr(c.store, "start_delivery", blocked_start)
            task = asyncio.create_task(combined(c))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                # The handler awaits the worker; other HTTP service stays usable.
                async with c.client._session.get(
                    c.client.base_url + "/internal/health"
                ) as response:
                    assert response.status == 200
                    await response.json()
            finally:
                release.set()
                await task
            assert len(thread_ids) == 2
            assert thread_ids[0] == thread_ids[1] != loop_thread

    asyncio.run(run())


def test_receive_registry_combined_flag_uses_one_rpc_and_exact_proof(monkeypatch):
    async def run():
        async with receiving() as c:
            registry = SparseReceiveRegistry(
                c.engine, TransferBudget(65536, 32), receiver_epoch="D-combined",
                combine_reserve_start=True,
            )
            record = registry.prepare(
                c.manifest, key=c.entry.key, rank=0, rail=c.store.rail,
                endpoint="D", sender_epoch=c.store.worker_epoch, client=c.client,
            )

            async def separate_call(*args):
                raise AssertionError("combined mode must not issue a separate RPC")

            monkeypatch.setattr(c.client, "reserve_delivery", separate_call)
            monkeypatch.setattr(c.client, "start_delivery", separate_call)
            try:
                assert not await record.start()
                assert record.snapshot()["source_started"]
                assert record.profile["combined_calls"] == 1
                assert record.profile["reserve_calls"] == 0
                assert record.profile["start_calls"] == 0
                d = c.store.entries[c.entry.key].deliveries[record.identity.transfer_id]
                assert not await record.poll()
                c.engine.finish(d.transfer_handle)
                assert await record.poll()
                record.stage(c.group, c.epoch)
                assert c.group.try_install(c.epoch, {0: 4})
                await record.ack()
                assert await record.close()
                assert not registry.snapshot()
                assert registry.budget.snapshot()["used_staging_bytes"] == 0
            finally:
                for d in c.store.entries[c.entry.key].deliveries.values():
                    handle = d.transfer_handle
                    if handle and not handle.transport_state.is_locally_safe_to_release:
                        c.engine.finish(handle)
                c.store.progress_transfers()
                await registry.close()

    asyncio.run(run())
