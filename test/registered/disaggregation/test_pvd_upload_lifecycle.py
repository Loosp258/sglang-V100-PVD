"""P -> V upload identity, terminal confirmation and cancellation cleanup.

Every case below runs the real VectorKVStore, allocator, page guards,
WriteAuthorization, VectorCoordinator and PVDPrefillRuntime.  Only the native
transport boundary is controllable: DelayedTransferEngine records a PUT and
does nothing until the test calls finish().  CPU tensors and a simulated CUDA
synchronization boundary do not prove GPUDirect/RDMA ordering on real hardware.
"""

import asyncio
import os
import uuid
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd import worker_epoch as worker_epoch_module
from sglang.srt.disaggregation.pvd.coordinator import (
    CoordinatorError,
    LocalShardClient,
    VectorCoordinator,
)
from sglang.srt.disaggregation.pvd.protocol import (
    PVD_GENERATION_METADATA_KEY,
    PVD_TRANSFER_LIFECYCLE_PROTOCOL,
    FirstTokenMetadata,
    KVEntryKey,
    WriteIdentity,
    upload_transfer_id,
)
from sglang.srt.disaggregation.pvd.request_state import EntryShardState, EntryState
from sglang.srt.disaggregation.pvd.runtime import (
    PVDDataPlaneError,
    PVDPrefillRuntime,
)
from sglang.srt.disaggregation.pvd.transfer_engine import MemorySlice, TransferStatus
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransportState
from sglang.srt.disaggregation.pvd.upload_manager import PVDUploadManager
from test_pvd_core import ENTRY_BYTES, make_manifest, make_store
from test_pvd_vector_lifecycle import DelayedTransferEngine

MODEL_INSTANCE = "model-instance"


# --------------------------------------------------------------------------
# In-process coordinator client with the PVDCoordinatorClient surface.
# --------------------------------------------------------------------------


class DirectCoordinatorClient:
    """Same call surface as PVDCoordinatorClient, without HTTP.

    ``drop_sync_replies`` simulates a lost terminal notification: V applies the
    report, but P never learns that it landed.  ``sync_failures`` simulates an
    unreachable V before the report is applied at all.
    """

    def __init__(self, coordinator: VectorCoordinator) -> None:
        self.coordinator = coordinator
        self.drop_sync_replies = False
        self.sync_failures = 0
        self.sync_calls = []

    async def create_entry(
        self, manifest, *, uploader_epoch=None, uploader_epochs=None
    ):
        kwargs = {"uploader_epoch": uploader_epoch}
        if uploader_epochs is not None:
            kwargs["uploader_epochs"] = uploader_epochs
        record = await self.coordinator.create_entry(manifest, **kwargs)
        return record.to_dict()

    async def commit_shard(self, key, rank, received_bytes, first_token=None):
        record = await self.coordinator.commit_shard(
            key, rank, received_bytes, first_token
        )
        return record.to_dict()

    async def cancel_entry(self, key, reason):
        record = await self.coordinator.cancel_entry(key, reason)
        return record.to_dict()

    async def sync_upload(self, identity, state, closed):
        self.sync_calls.append((identity.transfer_id, state, closed))
        if self.sync_failures > 0:
            self.sync_failures -= 1
            raise RuntimeError("V coordinator is unreachable")
        reply = await self.coordinator.sync_upload(identity, state, closed)
        if self.drop_sync_replies:
            raise RuntimeError("upload sync reply was lost in transit")
        return reply


# --------------------------------------------------------------------------
# Fixture
# --------------------------------------------------------------------------


class UploadPair:
    """One P worker uploading a two-shard Entry into two real V stores."""

    def __init__(self, req_id="upload-req", epoch=None):
        self.engine = DelayedTransferEngine()
        self.stores = [make_store(rank, self.engine) for rank in range(2)]
        self.coordinator = VectorCoordinator(
            [LocalShardClient(store) for store in self.stores]
        )
        self.client = DirectCoordinatorClient(self.coordinator)
        self.manager = PVDUploadManager()
        self.epoch = epoch or f"p-epoch-{uuid.uuid4().hex}"
        self.runtime = PVDPrefillRuntime(
            model_instance_id=MODEL_INSTANCE,
            coordinator=self.client,
            transfer_engine=self.engine,
            upload_manager=self.manager,
            worker_epoch=self.epoch,
            poll_interval_seconds=0,
        )
        self.manifest = make_manifest(req_id)
        self.lease = None
        self.sources = {}
        self.tasks = {}

    async def create(self):
        self.lease = await self.runtime.create_entry(
            req_id=self.manifest.key.req_id,
            transfer_id=self.manifest.key.transfer_id,
            layout=self.manifest.layout,
            prompt_token_count=self.manifest.prompt_token_count,
            shards={shard.rank: shard for shard in self.manifest.shards},
        )
        return self.lease

    def _source(self, rank):
        if rank not in self.sources:
            tensor = torch.full((ENTRY_BYTES,), rank + 1, dtype=torch.uint8)
            registration = self.engine.register_memory(
                tensor, endpoint="pvd-prefill", rank=rank, rail=f"mlx5_{rank}"
            )
            self.sources[rank] = (tensor, registration)
        return self.sources[rank]

    def submit(self, rank, *, first_token=True):
        """Start publish_shard. It stays pending until finish_native()."""
        _, registration = self._source(rank)
        token = FirstTokenMetadata(output_token_id=7) if rank == 0 else None
        if not first_token:
            token = None
        task = asyncio.ensure_future(
            self.runtime.publish_shard(
                lease=self.lease,
                rank=rank,
                local=MemorySlice(registration, 0, ENTRY_BYTES),
                first_token=token,
            )
        )
        self.tasks[rank] = task
        return task

    async def settle(self):
        """Let pending coroutines reach their next await point."""
        for _ in range(8):
            await asyncio.sleep(0)

    def record(self, rank):
        return self.manager.get(upload_transfer_id(self.manifest.key, rank))

    def handle(self, rank):
        record = self.record(rank)
        return None if record is None else record.handle

    def finish_native(self, rank, success=True):
        self.engine.finish(self.handle(rank), success=success)

    async def tick(self):
        return await self.manager.progress()

    async def shutdown(self):
        """Cancel publish coroutines still waiting on a native terminal.

        Cancelling the coroutine is exactly what the scheduler does on abort;
        it must not release anything, which the assertions above check.
        """
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)

    def entry(self, rank):
        return self.stores[rank].entries[self.manifest.key]

    def pages_reusable(self, rank):
        store = self.stores[rank]
        return (
            self.entry(rank).resources_released
            and store.allocator.available_pages == store.allocator.total_pages
        )

    def shard_stored(self, rank):
        return self.entry(rank).state == EntryShardState.STORED

    def entry_is_stored(self):
        return self.coordinator.entries[self.manifest.key].state == EntryState.STORED


async def make_pair(**kwargs):
    pair = UploadPair(**kwargs)
    await pair.create()
    return pair


async def make_tp2_pair():
    """Two process incarnations, separate managers, one shared Entry."""
    pair = UploadPair("tp2-upload", epoch="p-rank0")
    managers = [pair.manager, PVDUploadManager()]
    runtimes = [
        pair.runtime,
        PVDPrefillRuntime(
            model_instance_id=MODEL_INSTANCE,
            coordinator=pair.client,
            transfer_engine=pair.engine,
            upload_manager=managers[1],
            worker_epoch="p-rank1",
            poll_interval_seconds=0,
        ),
    ]
    epochs = {0: "p-rank0", 1: "p-rank1"}
    leases = await asyncio.gather(
        *(
            runtime.create_entry(
                req_id=pair.manifest.key.req_id,
                transfer_id=pair.manifest.key.transfer_id,
                layout=pair.manifest.layout,
                prompt_token_count=pair.manifest.prompt_token_count,
                shards={s.rank: s for s in pair.manifest.shards},
                uploader_epochs=epochs,
                owned_shard_ranks={rank},
            )
            for rank, runtime in enumerate(runtimes)
        )
    )
    return pair, managers, runtimes, leases


def test_tp2_distinct_uploaders_publish_one_entry():
    async def scenario():
        pair, managers, runtimes, leases = await make_tp2_pair()
        assert leases[0].upload_identities == leases[1].upload_identities
        tasks = []
        handles = []
        for rank in (0, 1):
            assert managers[rank].outstanding() == 1
            assert (
                managers[rank].get(upload_transfer_id(pair.manifest.key, 1 - rank))
                is None
            )
            _, registration = pair._source(rank)
            tasks.append(
                asyncio.create_task(
                    runtimes[rank].publish_shard(
                        lease=leases[rank],
                        rank=rank,
                        local=MemorySlice(registration, 0, ENTRY_BYTES),
                        first_token=FirstTokenMetadata(output_token_id=7)
                        if rank == 0
                        else None,
                    )
                )
            )
        await pair.settle()
        for rank in (0, 1):
            handle = (
                managers[rank].get(upload_transfer_id(pair.manifest.key, rank)).handle
            )
            handles.append(handle)
            assert handle.transport_state == TransportState.IN_FLIGHT
        assert not pair.entry_is_stored()
        for handle in handles:
            pair.engine.finish(handle)
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert pair.entry_is_stored()
        assert all(manager.outstanding() == 0 for manager in managers)

    asyncio.run(scenario())


def test_tp2_cancel_one_sender_cannot_drain_other_sender():
    async def scenario():
        pair, managers, runtimes, leases = await make_tp2_pair()
        _, registration = pair._source(1)
        task = asyncio.create_task(
            runtimes[1].publish_shard(
                lease=leases[1],
                rank=1,
                local=MemorySlice(registration, 0, ENTRY_BYTES),
            )
        )
        try:
            await pair.settle()
            handle = managers[1].get(upload_transfer_id(pair.manifest.key, 1)).handle
            managers[0].abandon_key(pair.manifest.key, "rank0 cancelled before submit")
            await pair.coordinator.cancel_entry(pair.manifest.key, "request cancelled")
            await managers[0].progress()
            assert pair.pages_reusable(0)
            assert pair.entry(1).upload_terminal is None
            assert not pair.pages_reusable(1)
            assert handle.transport_state == TransportState.IN_FLIGHT
            assert {call[0] for call in pair.client.sync_calls} == {
                upload_transfer_id(pair.manifest.key, 0)
            }
            # A late native success only drains the cancelled Entry.
            pair.engine.finish(handle)
            await asyncio.gather(task, return_exceptions=True)
            await managers[1].progress()
            assert pair.pages_reusable(1)
            assert not pair.entry_is_stored()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_tp2_cannot_submit_another_ranks_shard():
    async def scenario():
        pair, managers, runtimes, leases = await make_tp2_pair()
        _, registration = pair._source(1)
        with pytest.raises(PVDDataPlaneError, match="does not own"):
            await runtimes[0].publish_shard(
                lease=leases[0],
                rank=1,
                local=MemorySlice(registration, 0, ENTRY_BYTES),
            )
        assert managers[0].get(upload_transfer_id(pair.manifest.key, 1)) is None
        assert managers[1].get(upload_transfer_id(pair.manifest.key, 1)).handle is None
        assert pair.entry(1).upload_terminal is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "bad_map",
    [
        {},
        {0: "p0"},
        {0: "p0", 1: "p1", 2: "p2"},
        {0: "p0", 1: ""},
        {0: "p0", 1: 7},
        {False: "p0", 1: "p1"},
        {0.0: "p0", 1: "p1"},
        {"00": "p0", "1": "p1"},
        {0: "p0", "0": "p0", 1: "p1"},
        ["p0", "p1"],
    ],
)
def test_invalid_uploader_map_rejected_before_allocation(bad_map):
    async def scenario():
        pair = UploadPair()
        with pytest.raises(ValueError):
            await pair.coordinator.create_entry(pair.manifest, uploader_epochs=bad_map)
        assert not pair.coordinator.entries
        assert all(store.allocator.allocated_pages == 0 for store in pair.stores)

    asyncio.run(scenario())


def test_uploader_map_is_idempotent_but_cannot_change_owner():
    async def scenario():
        pair, _, _, _ = await make_tp2_pair()
        epochs = {0: "p-rank0", 1: "p-rank1"}
        entry = await pair.coordinator.create_entry(
            pair.manifest, uploader_epochs=epochs
        )
        assert entry.uploader_epochs == epochs
        # Caller mutation cannot change the stored authorization domain.
        epochs[1] = "restarted-p-rank1"
        with pytest.raises(CoordinatorError, match="uploader epoch"):
            await pair.coordinator.create_entry(pair.manifest, uploader_epochs=epochs)
        assert entry.uploader_epochs[1] == "p-rank1"
        with pytest.raises(ValueError, match="not both"):
            await pair.coordinator.create_entry(
                pair.manifest,
                uploader_epoch="p-rank0",
                uploader_epochs=epochs,
            )
        with pytest.raises(CoordinatorError, match="uploader epoch"):
            await pair.coordinator.create_entry(pair.manifest)

    asyncio.run(scenario())


@pytest.mark.parametrize("tp_size", [1, 2])
def test_real_sender_passes_uploader_map_and_owns_only_local_shards(tp_size):
    from sglang.srt.disaggregation.pvd.conn import PVDKVSender

    async def scenario():
        pair = UploadPair("sender-wiring", epoch="p0")
        manifest = pair.manifest
        epochs = {rank: f"p{rank if tp_size == 2 else 0}" for rank in (0, 1)}
        senders, managers = [], []
        for rank in range(tp_size):
            uploads = PVDUploadManager()
            runtime = PVDPrefillRuntime(
                model_instance_id=MODEL_INSTANCE,
                coordinator=pair.client,
                transfer_engine=pair.engine,
                upload_manager=uploads,
                worker_epoch=epochs[rank],
            )

            def gather(value, expected_rank=rank):
                assert value["epoch"] == epochs[expected_rank]
                return [
                    {
                        "rank": r,
                        "epoch": epochs[r],
                        "shard": manifest.shard(r).to_dict(),
                    }
                    for r in (0, 1)
                ]

            manager = SimpleNamespace(
                tp_size=tp_size,
                tp_rank=rank,
                worker_epoch=epochs[rank],
                key_for=lambda req: manifest.key,
                client_for=lambda req: pair.client,
                prefill_runtime_for=lambda req, rt=runtime: rt,
                storage_layout=lambda: manifest.layout,
                storage_shard_manifest=lambda n, r, layout: manifest.shard(r),
                local_shard_manifest=lambda n, r=rank: manifest.shard(r),
                gather_rank_objects=gather,
                control=SimpleNamespace(submit=asyncio.create_task),
                abandon_uploads=lambda key, reason, um=uploads: um.abandon_key(
                    key, reason
                ),
            )
            senders.append(
                PVDKVSender(
                    mgr=manager,
                    req=SimpleNamespace(
                        origin_input_ids=list(range(manifest.prompt_token_count)),
                    ),
                )
            )
            managers.append(uploads)
        leases = await asyncio.gather(*(sender._create_future for sender in senders))
        for rank, lease in enumerate(leases):
            assert {
                r: i.sender_epoch for r, i in lease.upload_identities.items()
            } == epochs
            assert lease.owned_shard_ranks == ({0, 1} if tp_size == 1 else {rank})
        senders[0].clear()
        await managers[0].progress()
        if tp_size == 2:
            assert pair.entry(1).upload_terminal is None
            assert not managers[1].get(upload_transfer_id(manifest.key, 1)).abandoned

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# Identity shape
# --------------------------------------------------------------------------


def test_tp1_uploads_to_two_v_shards_have_distinct_identities():
    async def scenario():
        pair = await make_pair()
        identities = pair.lease.upload_identities
        assert set(identities) == {0, 1}
        # One P compute rank, one sender epoch, two storage shards.
        assert {i.sender_epoch for i in identities.values()} == {pair.epoch}
        assert identities[0].transfer_id != identities[1].transfer_id
        assert identities[0].shard_rank == 0 and identities[1].shard_rank == 1
        assert identities[0].region_id != identities[1].region_id
        assert identities[0].receiver_epoch != identities[1].receiver_epoch
        for rank, identity in identities.items():
            assert identity.protocol == PVD_TRANSFER_LIFECYCLE_PROTOCOL
            assert identity.key == pair.manifest.key
            assert identity.transfer_id == upload_transfer_id(pair.manifest.key, rank)
            identity.validate_destination(pair.lease.target_regions[rank])

    asyncio.run(scenario())


def test_upload_transfer_id_names_the_storage_shard_not_the_compute_rank():
    key = KVEntryKey.new(MODEL_INSTANCE, "req-a")
    other = KVEntryKey.new(MODEL_INSTANCE, "req-b")
    assert upload_transfer_id(key, 0) != upload_transfer_id(key, 1)
    assert upload_transfer_id(key, 0) != upload_transfer_id(other, 0)
    assert upload_transfer_id(key, 1).endswith(":v1")
    with pytest.raises(Exception):
        upload_transfer_id(key, -1)
    with pytest.raises(Exception):
        upload_transfer_id(key, True)


def test_second_uploader_epoch_cannot_join_an_existing_entry():
    async def scenario():
        pair = await make_pair()
        with pytest.raises(CoordinatorError, match="uploader epoch"):
            await pair.coordinator.create_entry(
                pair.manifest, uploader_epoch="a-different-p-epoch"
            )
        # A legacy caller cannot fall back once lifecycle metadata exists.
        with pytest.raises(CoordinatorError, match="uploader epoch"):
            await pair.coordinator.create_entry(pair.manifest)

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# PENDING is not failure, and is not STORED either
# --------------------------------------------------------------------------


def test_first_pending_poll_does_not_cancel_the_upload():
    async def scenario():
        pair = await make_pair()
        task = pair.submit(0)
        await pair.settle()
        # The old single-poll publish treated this PENDING as a failure and
        # cancelled the entry. The write is still in flight.
        assert not task.done()
        assert pair.handle(0).transport_state == TransportState.IN_FLIGHT
        assert pair.entry(0).state == EntryShardState.P_WRITING
        assert not pair.entry(0).release_requested

        pair.finish_native(0, success=True)
        assert await task
        assert pair.shard_stored(0)

    asyncio.run(scenario())


def test_pending_upload_does_not_publish_stored_or_kv_ready():
    async def scenario():
        pair = await make_pair()
        for rank in (0, 1):
            pair.submit(rank)
        await pair.settle()
        for rank in (0, 1):
            assert not pair.shard_stored(rank)
        assert not pair.entry_is_stored()

        # A byte-complete commit alone must not publish STORED either.
        await pair.coordinator.commit_shard(pair.manifest.key, 1, ENTRY_BYTES, None)
        assert not pair.shard_stored(1)
        assert not pair.entry_is_stored()

        for rank in (0, 1):
            pair.finish_native(rank, success=True)
        assert await pair.tasks[0]
        assert await pair.tasks[1]
        assert pair.shard_stored(0) and pair.shard_stored(1)
        assert pair.entry_is_stored()

    asyncio.run(scenario())


def test_byte_complete_commit_without_terminal_keeps_pages_and_state():
    async def scenario():
        pair = await make_pair()
        pair.submit(0)
        await pair.settle()
        entry = pair.entry(0)
        # Commit arriving before the transport terminal records the bytes but
        # publishes nothing and releases nothing.
        pair.stores[0].commit_p_write(pair.manifest.key, ENTRY_BYTES)
        assert entry.upload_committed_bytes == ENTRY_BYTES
        assert entry.state == EntryShardState.P_WRITING
        assert entry.upload_pending
        assert not pair.pages_reusable(0)

        pair.finish_native(0, success=True)
        await pair.tick()
        assert entry.state == EntryShardState.STORED
        assert await pair.tasks[0]

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# Cancellation: V retains, late transport releases without resurrecting
# --------------------------------------------------------------------------


def test_v_cancel_retains_pages_until_matching_closed_terminal():
    async def scenario():
        pair = await make_pair()
        task = pair.submit(0)
        await pair.settle()

        await pair.coordinator.cancel_entry(pair.manifest.key, "injected V cancel")
        entry = pair.entry(0)
        assert entry.release_requested
        assert entry.upload_close_requested
        # Cancellation is a request to the sender, never a release.
        assert not pair.pages_reusable(0)
        assert pair.stores[0].allocator.allocated_pages == entry.allocation.page_count

        # A sync that is not closed picks up the close request and still
        # releases nothing.
        await pair.tick()
        assert pair.record(0).submission_forbidden
        assert not pair.pages_reusable(0)

        pair.finish_native(0, success=False)
        await pair.tick()
        assert pair.pages_reusable(0)
        assert entry.upload_terminal == TransportState.TERMINAL_FAILED
        assert not pair.shard_stored(0)
        with pytest.raises(Exception):
            await task

    asyncio.run(scenario())


def test_late_success_after_business_cancel_reclaims_without_publishing():
    async def scenario():
        pair = await make_pair()
        task = pair.submit(0)
        await pair.settle()
        await pair.coordinator.cancel_entry(pair.manifest.key, "client disconnected")

        # The WRITE lands after the request was already abandoned.
        pair.finish_native(0, success=True)
        await pair.tick()

        entry = pair.entry(0)
        assert entry.upload_terminal == TransportState.TERMINAL_SUCCESS
        assert pair.pages_reusable(0)
        # Reclamation only: the request is not resurrected.
        assert not pair.shard_stored(0)
        assert not pair.entry_is_stored()
        # The publish coroutine fails on the cancelled entry. Its failure is a
        # business result; it did not release anything by itself.
        with pytest.raises(Exception):
            await task

    asyncio.run(scenario())


def test_ttl_expiry_cannot_release_an_unconfirmed_upload():
    async def scenario():
        import time as _time

        pair = await make_pair()
        pair.submit(0)
        await pair.settle()
        pair.stores[0].reap_expired(_time.monotonic() + 10_000)
        entry = pair.entry(0)
        assert entry.upload_close_requested
        assert not pair.pages_reusable(0)
        # Only a matching closed terminal may release.
        pair.finish_native(0, success=False)
        await pair.tick()
        assert pair.pages_reusable(0)
        await pair.shutdown()

    asyncio.run(scenario())


def test_store_close_cannot_release_an_unconfirmed_upload():
    async def scenario():
        pair = await make_pair()
        pair.submit(0)
        await pair.settle()
        entry = pair.entry(0)
        pair.stores[0].close()
        assert entry.upload_close_requested
        assert not pair.pages_reusable(0)
        assert pair.stores[0].allocator.allocated_pages == entry.allocation.page_count
        # The pool MR stays registered while any allocation is still protected.
        assert pair.stores[0].registration.descriptor.region_id in pair.engine._regions
        await pair.shutdown()

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# Pre-native rejection
# --------------------------------------------------------------------------


def test_local_rejection_before_native_submit_closes_the_authorization():
    async def scenario():
        pair = await make_pair()
        # Rail mismatch is rejected by the adapter before any native call.
        _, registration = pair._source(0)

        def rejecting_submit(local, remote, *, remote_offset=0):
            from sglang.srt.disaggregation.pvd.transfer_engine import TransferHandle

            handle = TransferHandle(uuid.uuid4().hex)
            handle.status = TransferStatus.FAILED
            handle.error = "PVD source preparation failed: injected"
            return handle

        pair.engine.submit_put = rejecting_submit
        task = pair.submit(0)
        with pytest.raises(PVDDataPlaneError):
            await task

        await pair.tick()
        entry = pair.entry(0)
        # begin consumed the one-shot gate, so a proven pre-native rejection is
        # recorded as a safe failed gate, not as "never began".
        assert entry.upload_begun
        assert entry.upload_terminal == TransportState.TERMINAL_FAILED
        assert pair.pages_reusable(0)
        assert not pair.shard_stored(0)

    asyncio.run(scenario())


def test_abandoned_upload_that_never_submitted_reports_not_submitted():
    """A request aborted before send must still free V's destination."""

    async def scenario():
        pair = await make_pair()
        # Taking the lease already registers a record per shard, so a sender
        # that is aborted before send() still has something that can report.
        assert pair.manager.outstanding() == 2
        record = pair.record(0)
        assert record is not None and record.handle is None

        # Nothing has been reported yet: V cannot act on "still preparing".
        await pair.tick()
        assert pair.entry(0).upload_terminal is None
        assert not pair.pages_reusable(0)

        pair.manager.abandon(record.transfer_id, "sender cleared")
        await pair.tick()
        assert pair.entry(0).upload_terminal == TransportState.TERMINAL_FAILED
        assert pair.pages_reusable(0)
        # The other shard is untouched by this one's reclamation.
        assert not pair.pages_reusable(1)

    asyncio.run(scenario())


def test_rejected_foreign_lease_never_reports_another_process_terminal():
    """An invalid foreign lease cannot grant local ownership of its writes."""

    async def scenario():
        pair = UploadPair("rejected-lease")
        # Make V's published identity fail P's own epoch check.
        pair.runtime.worker_epoch = "an-unexpected-epoch"
        original_create = pair.client.create_entry

        async def create_with_foreign_epoch(manifest, *, uploader_epoch=None):
            return await original_create(manifest, uploader_epoch=pair.epoch)

        pair.client.create_entry = create_with_foreign_epoch
        with pytest.raises(PVDDataPlaneError, match="uploader epoch"):
            await pair.create()

        assert pair.manager.outstanding() == 0
        await pair.tick()
        assert pair.client.sync_calls == []
        for rank in (0, 1):
            store = pair.stores[rank]
            entry = store.entries[pair.manifest.key]
            assert entry.upload_terminal is None
            assert not entry.resources_released

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# Manager ownership after the sender disappears
# --------------------------------------------------------------------------


def test_records_progress_after_the_sender_task_is_cancelled():
    """PVDKVSender.abort()/clear() retire the sender, not the upload."""

    async def scenario():
        pair = await make_pair()
        task = pair.submit(0)
        await pair.settle()
        handle = pair.handle(0)

        # Same ordering as PVDKVSender.abort(): forbid further submission,
        # then ask V to cancel. Neither releases the destination.
        pair.manager.abandon(
            pair.lease.upload_identities[0].transfer_id, "sender removed"
        )
        await pair.coordinator.cancel_entry(pair.manifest.key, "request aborted")

        # The scheduler drops the request: the publish coroutine goes away and
        # is never polled again.
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert pair.record(0) is not None
        assert not pair.pages_reusable(0)

        # The manager, not the sender, drives the remaining work.
        await pair.tick()
        assert not pair.pages_reusable(0)
        pair.engine.finish(handle, success=True)
        await pair.tick()
        assert pair.pages_reusable(0)
        assert pair.record(0) is None
        # A late success reclaims. It does not republish the request.
        assert not pair.shard_stored(0)
        assert not pair.entry_is_stored()

    asyncio.run(scenario())


def test_one_submission_per_identity_is_enforced_on_p():
    """V's begin gate is one-shot; P must not issue a second write either."""

    async def scenario():
        pair = await make_pair()
        first = pair.submit(0)
        await pair.settle()
        record = pair.record(0)
        handle = record.handle

        with pytest.raises(PVDDataPlaneError, match="already been submitted"):
            await pair.runtime.publish_shard(
                lease=pair.lease,
                rank=0,
                local=MemorySlice(pair._source(0)[1], 0, ENTRY_BYTES),
            )
        # The original handle is still the one being tracked.
        assert pair.record(0).handle is handle

        # Shard 1 has never been submitted, so abandoning it exercises the
        # forbidden branch rather than the already-claimed one.
        pair.manager.abandon(pair.lease.upload_identities[1].transfer_id, "aborted")
        with pytest.raises(PVDDataPlaneError, match="no longer be submitted"):
            await pair.runtime.publish_shard(
                lease=pair.lease,
                rank=1,
                local=MemorySlice(pair._source(1)[1], 0, ENTRY_BYTES),
            )
        assert not pair.pages_reusable(0)
        assert not pair.pages_reusable(1)

        first.cancel()
        await pair.shutdown()

    asyncio.run(scenario())


def test_retired_identity_cannot_be_submitted_again():
    async def scenario():
        pair = await make_pair()
        task = pair.submit(0)
        await pair.settle()
        pair.finish_native(0, success=True)
        assert await task
        assert pair.record(0) is None
        with pytest.raises(RuntimeError, match="closed"):
            pair.manager.open(
                identity=pair.lease.upload_identities[0], coordinator=pair.client
            )

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# Lost and retried notifications
# --------------------------------------------------------------------------


def test_lost_terminal_notification_is_retried_idempotently():
    async def scenario():
        pair = await make_pair()
        task = pair.submit(0)
        await pair.settle()
        pair.finish_native(0, success=True)

        # V is unreachable: nothing is released and the record survives every
        # retry. The count is deliberately larger than the number of ticks so
        # the assertions do not depend on coroutine scheduling order.
        pair.client.sync_failures = 1000
        for _ in range(3):
            await pair.tick()
        assert pair.record(0) is not None
        assert pair.entry(0).upload_terminal is None
        assert not pair.shard_stored(0)
        assert not pair.pages_reusable(0)

        # The publish coroutine fails because V never acknowledged the
        # terminal within its bounded attempts. That is a business failure; it
        # released nothing, and the record is still owned by the manager.
        with pytest.raises(PVDDataPlaneError, match="not acknowledged"):
            await task
        assert pair.record(0) is not None

        pair.client.sync_failures = 0
        await pair.tick()
        assert pair.entry(0).upload_terminal == TransportState.TERMINAL_SUCCESS
        assert pair.record(0) is None

        # A replayed identical terminal report is accepted idempotently.
        reply = await pair.coordinator.sync_upload(
            pair.lease.upload_identities[0],
            TransportState.TERMINAL_SUCCESS,
            True,
        )
        assert reply["terminal_ack"] is True

    asyncio.run(scenario())


def test_lost_reply_after_v_applied_the_report_is_idempotent_on_retry():
    """V applied the terminal, but P never saw the acknowledgement."""

    async def scenario():
        pair = await make_pair()
        task = pair.submit(0)
        await pair.settle()
        pair.finish_native(0, success=True)

        pair.client.drop_sync_replies = True
        with pytest.raises(PVDDataPlaneError, match="not acknowledged"):
            await task
        # V did apply it; only the reply was lost.
        assert pair.entry(0).upload_terminal == TransportState.TERMINAL_SUCCESS
        assert pair.record(0) is not None

        pair.client.drop_sync_replies = False
        await pair.tick()
        assert pair.record(0) is None
        assert pair.entry(0).upload_terminal == TransportState.TERMINAL_SUCCESS

    asyncio.run(scenario())


def test_conflicting_terminal_reports_are_rejected():
    async def scenario():
        pair = await make_pair()
        task = pair.submit(0)
        await pair.settle()
        pair.finish_native(0, success=True)
        await pair.tick()
        with pytest.raises(Exception, match="different terminal state"):
            await pair.coordinator.sync_upload(
                pair.lease.upload_identities[0],
                TransportState.TERMINAL_FAILED,
                True,
            )
        assert await task

    asyncio.run(scenario())


def test_non_terminal_state_cannot_arrive_closed():
    async def scenario():
        pair = await make_pair()
        pair.submit(0)
        await pair.settle()
        for state in (TransportState.IN_FLIGHT, TransportState.DRAINING):
            with pytest.raises(Exception, match="terminal"):
                await pair.coordinator.sync_upload(
                    pair.lease.upload_identities[0], state, True
                )
        assert not pair.pages_reusable(0)
        await pair.shutdown()

    asyncio.run(scenario())


def test_unknown_transport_state_isolates_and_never_releases():
    async def scenario():
        pair = await make_pair()
        pair.submit(0)
        await pair.settle()
        reply = await pair.coordinator.sync_upload(
            pair.lease.upload_identities[0], TransportState.UNKNOWN, True
        )
        assert reply["terminal_ack"] is False
        assert reply["close_requested"] is True
        assert not pair.pages_reusable(0)
        assert pair.stores[0].snapshot()["isolated_reason"]
        await pair.shutdown()

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# Identity forgery
# --------------------------------------------------------------------------


def _tamper(identity, **changes):
    import dataclasses

    return dataclasses.replace(identity, **changes)


def test_wrong_epoch_or_generation_cannot_release_an_allocation():
    async def scenario():
        pair = await make_pair()
        pair.submit(0)
        await pair.settle()
        identity = pair.lease.upload_identities[0]
        forgeries = [
            _tamper(identity, sender_epoch="another-p-epoch"),
            _tamper(identity, receiver_epoch="another-v-epoch"),
            _tamper(identity, generation=uuid.uuid4().hex),
            _tamper(identity, region_id=uuid.uuid4().hex),
            _tamper(identity, shard_rank=1),
            _tamper(identity, transfer_id="upload:forged:v0"),
            _tamper(identity, key=KVEntryKey.new(MODEL_INSTANCE, "other-req")),
        ]
        for forged in forgeries:
            with pytest.raises(Exception):
                await pair.coordinator.sync_upload(
                    forged, TransportState.TERMINAL_SUCCESS, True
                )
            assert not pair.pages_reusable(0)
            assert pair.entry(0).upload_terminal is None
        await pair.shutdown()

    asyncio.run(scenario())


def test_terminal_for_one_shard_does_not_release_the_other():
    async def scenario():
        pair = await make_pair()
        for rank in (0, 1):
            pair.submit(rank)
        await pair.settle()
        pair.finish_native(0, success=False)
        await pair.tick()
        assert pair.pages_reusable(0)
        assert not pair.pages_reusable(1)
        assert pair.entry(1).upload_terminal is None
        with pytest.raises(Exception):
            await pair.tasks[0]
        # Shard 1's write is still in flight. Cancelling its publish coroutine
        # releases nothing.
        await pair.shutdown()
        assert not pair.pages_reusable(1)

    asyncio.run(scenario())


def test_begin_requires_the_matching_identity():
    engine = DelayedTransferEngine()
    store = make_store(0, engine)
    manifest = make_manifest("begin-guard")
    entry = store.create_entry(manifest, uploader_epoch="p-epoch")
    with pytest.raises(Exception, match="lifecycle write identity"):
        store.begin_p_write(manifest.key)
    forged = _tamper(entry.upload_identity, sender_epoch="other")
    with pytest.raises(Exception, match="does not match"):
        store.begin_p_write(manifest.key, forged)
    store.begin_p_write(manifest.key, entry.upload_identity)
    # One-shot gate.
    with pytest.raises(Exception, match="already begun"):
        store.begin_p_write(manifest.key, entry.upload_identity)


def test_legacy_entry_rejects_a_lifecycle_identity():
    engine = DelayedTransferEngine()
    store = make_store(0, engine)
    manifest = make_manifest("legacy-entry")
    entry = store.create_entry(manifest)
    assert entry.upload_identity is None
    identity = WriteIdentity(
        protocol=PVD_TRANSFER_LIFECYCLE_PROTOCOL,
        sender_epoch="p-epoch",
        receiver_epoch=store.worker_epoch,
        transfer_id=upload_transfer_id(manifest.key, 0),
        region_id=entry.target_region.region_id,
        generation=entry.target_region.backend_metadata[PVD_GENERATION_METADATA_KEY],
        shard_rank=0,
        key=manifest.key,
    )
    with pytest.raises(Exception, match="no lifecycle upload authorization"):
        store.begin_p_write(manifest.key, identity)
    with pytest.raises(Exception, match="no lifecycle upload authorization"):
        store.sync_upload(identity, TransportState.TERMINAL_SUCCESS, True)
    # The legacy commit path is unchanged for legacy entries.
    store.begin_p_write(manifest.key)
    store.commit_p_write(manifest.key, ENTRY_BYTES)
    assert entry.state == EntryShardState.STORED
    assert not entry.upload_pending


# --------------------------------------------------------------------------
# Real HTTP control plane
# --------------------------------------------------------------------------


def test_upload_sync_over_real_http():
    async def scenario():
        from aiohttp.test_utils import TestClient, TestServer
        from sglang.srt.disaggregation.pvd.client import PVDCoordinatorClient
        from sglang.srt.disaggregation.pvd.control_server import (
            create_coordinator_app,
        )

        engine = DelayedTransferEngine()
        stores = [make_store(rank, engine) for rank in range(2)]
        coordinator = VectorCoordinator([LocalShardClient(store) for store in stores])
        server = TestServer(create_coordinator_app(coordinator))
        async with TestClient(server) as http:
            client = PVDCoordinatorClient(
                str(server.make_url("")), session=http.session
            )
            manifest = make_manifest("http-upload")
            created = await client.create_entry(manifest, uploader_epoch="p-http")
            identities = {
                int(rank): WriteIdentity.from_dict(value)
                for rank, value in created["upload_identities"].items()
            }
            assert set(identities) == {0, 1}

            reply = await client.sync_upload(
                identities[0], TransportState.IN_FLIGHT, False
            )
            assert reply["terminal_ack"] is False
            assert reply["close_requested"] is False
            assert stores[0].allocator.allocated_pages > 0

            await client.cancel_entry(manifest.key, "http cancel")
            reply = await client.sync_upload(
                identities[0], TransportState.IN_FLIGHT, False
            )
            assert reply["close_requested"] is True
            assert not stores[0].entries[manifest.key].resources_released

            reply = await client.sync_upload(
                identities[0], TransportState.TERMINAL_FAILED, True
            )
            assert reply["terminal_ack"] is True
            assert stores[0].entries[manifest.key].resources_released

            # A malformed closed report is refused over HTTP too.
            with pytest.raises(Exception):
                await client.sync_upload(identities[1], TransportState.IN_FLIGHT, True)
            assert not stores[1].entries[manifest.key].resources_released

    asyncio.run(scenario())


def test_rejected_partial_lease_abandons_only_valid_local_authorizations():
    async def scenario():
        pair = UploadPair("partially-invalid-lease")
        original_create = pair.client.create_entry

        async def corrupt_one_identity(manifest, **kwargs):
            reply = await original_create(manifest, **kwargs)
            reply["upload_identities"]["1"]["sender_epoch"] = "foreign-process"
            return reply

        pair.client.create_entry = corrupt_one_identity
        with pytest.raises(PVDDataPlaneError, match="uploader epoch"):
            await pair.create()
        assert pair.manager.outstanding() == 1
        assert pair.record(0).abandoned
        assert pair.record(1) is None
        await pair.tick()
        assert pair.pages_reusable(0)
        assert not pair.pages_reusable(1)
        assert pair.entry(1).upload_terminal is None

    asyncio.run(scenario())


def test_runtime_rejects_inconsistent_ownership_before_contacting_v():
    async def scenario():
        pair = UploadPair(epoch="p0")
        with pytest.raises(PVDDataPlaneError, match="owned shards"):
            await pair.runtime.create_entry(
                req_id=pair.manifest.key.req_id,
                transfer_id=pair.manifest.key.transfer_id,
                layout=pair.manifest.layout,
                prompt_token_count=pair.manifest.prompt_token_count,
                shards={s.rank: s for s in pair.manifest.shards},
                uploader_epochs={0: "p0", 1: "p1"},
                owned_shard_ranks={0, 1},
            )
        assert not pair.coordinator.entries
        assert pair.manager.outstanding() == 0

    asyncio.run(scenario())


def test_tp2_uploader_map_over_coordinator_and_shard_http():
    async def scenario():
        from aiohttp.test_utils import TestClient, TestServer
        from sglang.srt.disaggregation.pvd.client import PVDCoordinatorClient
        from sglang.srt.disaggregation.pvd.control_server import (
            HttpShardClient,
            create_coordinator_app,
            create_shard_app,
        )

        engine = DelayedTransferEngine()
        stores = [make_store(rank, engine) for rank in (0, 1)]
        shard_server = TestServer(create_shard_app(stores[1]))
        async with TestClient(shard_server) as shard_http:
            coordinator = VectorCoordinator(
                [
                    LocalShardClient(stores[0]),
                    HttpShardClient(
                        1, str(shard_server.make_url("")), session=shard_http.session
                    ),
                ]
            )
            server = TestServer(create_coordinator_app(coordinator))
            async with TestClient(server) as http:
                client = PVDCoordinatorClient(
                    str(server.make_url("")), session=http.session
                )
                manifest = make_manifest("http-tp2")
                epochs = {0: "http-p0", 1: "http-p1"}
                replies = await asyncio.gather(
                    *(
                        client.create_entry(manifest, uploader_epochs=epochs)
                        for _ in (0, 1)
                    )
                )
                assert replies[0] == replies[1]
                assert replies[0]["uploader_epochs"] == {"0": "http-p0", "1": "http-p1"}
                for rank in (0, 1):
                    assert (
                        stores[rank].entries[manifest.key].upload_identity.sender_epoch
                        == epochs[rank]
                    )
                # Exercise server validation directly, bypassing client checks.
                for invalid in ({"0": "p0"}, {"0": "p0", "1": ""}):
                    response = await http.post(
                        "/v1/entries",
                        json={
                            "manifest": make_manifest("bad-http").to_dict(),
                            "uploader_epochs": invalid,
                        },
                    )
                    assert response.status == 400
                assert len(coordinator.entries) == 1

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# Worker epoch
# --------------------------------------------------------------------------


def test_worker_epoch_is_stable_within_a_process():
    first = worker_epoch_module.worker_epoch()
    assert first == worker_epoch_module.worker_epoch()
    assert worker_epoch_module.current_worker_epoch() == first
    assert first.startswith(f"{os.getpid()}-")


def test_worker_epoch_changes_when_the_pid_changes(monkeypatch):
    real_pid = os.getpid()
    original = worker_epoch_module.worker_epoch()
    assert original.startswith(f"{real_pid}-")
    # Stand in for a forked child that inherited this module's state.
    monkeypatch.setattr(worker_epoch_module.os, "getpid", lambda: real_pid + 1)
    forked = worker_epoch_module.worker_epoch()
    assert forked != original
    assert forked.startswith(f"{real_pid + 1}-")
    monkeypatch.undo()
    # Back in the parent, the inherited value is not reused either.
    assert worker_epoch_module.worker_epoch() not in (forked, original)


def test_fork_hook_clears_inherited_state():
    before = worker_epoch_module.worker_epoch()
    worker_epoch_module._before_fork()
    worker_epoch_module._after_fork_in_child()
    assert worker_epoch_module.current_worker_epoch() is None
    assert worker_epoch_module.worker_epoch() != before


def test_prefill_runtime_requires_epoch_and_manager_together():
    engine = DelayedTransferEngine()
    with pytest.raises(ValueError, match="worker epoch"):
        PVDPrefillRuntime(
            model_instance_id=MODEL_INSTANCE,
            coordinator=None,
            transfer_engine=engine,
            upload_manager=PVDUploadManager(),
        )
    with pytest.raises(ValueError, match="worker epoch"):
        PVDPrefillRuntime(
            model_instance_id=MODEL_INSTANCE,
            coordinator=None,
            transfer_engine=engine,
            worker_epoch="p-epoch",
        )
