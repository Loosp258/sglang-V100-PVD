"""Chunked upload keeps each V shard fenced until exact terminal proof."""

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from sglang.srt.disaggregation.pvd.coordinator import (
    LocalShardClient,
    VectorCoordinator,
)
from sglang.srt.disaggregation.pvd.prompt_chunks import PromptChunkIdentity
from sglang.srt.disaggregation.pvd.protocol import FirstTokenMetadata
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransportState
from sglang.srt.disaggregation.pvd.vector_store import VectorKVStore
from test_pvd_core import ENTRY_BYTES, PAGE_BYTES, make_manifest


def make_chunk_store(rank, engine):
    index = SimpleNamespace(
        backend=SimpleNamespace(supports_extend=True),
        open=lambda transfer_id: None,
        note_kv_readable=lambda transfer_id: None,
        close=lambda transfer_id: None,
    )
    store = VectorKVStore(
        rank=rank,
        world_size=2,
        rail=f"mlx5_{rank}",
        device="cpu",
        total_pages=8,
        page_bytes=PAGE_BYTES,
        endpoint=f"v{rank}",
        transfer_engine=engine,
        allow_cpu_for_tests=True,
        prompt_index=index,
        enable_chunked_upload=True,
    )
    store.progress_prompt_indexes = lambda *, wait_for_lock=False: {}
    return store


def test_local_group_kicks_both_shard_indexes_before_final_chunk():
    asyncio.run(_local_group_kicks_both_shard_indexes_before_final_chunk())


async def _local_group_kicks_both_shard_indexes_before_final_chunk():
    engine = FakeTransferEngine()
    stores = [make_chunk_store(rank, engine) for rank in (0, 1)]
    clients = [LocalShardClient(store) for store in stores]
    started = [threading.Event() for _ in stores]
    for rank, store in enumerate(stores):
        def progress(*, wait_for_lock=False, rank=rank):
            assert wait_for_lock
            started[rank].set()

        store.progress_prompt_indexes = progress
    coordinator = VectorCoordinator(clients)
    manifest = replace(make_manifest("local-index-kick"), upload_mode="chunked_cagra")
    entry = await coordinator.create_entry(manifest, uploader_epoch="sender")
    for rank in (0, 1):
        reply = await coordinator.begin_chunk(manifest.key, rank, 0, 1)
        chunk = PromptChunkIdentity.from_dict(
            reply,
            layout=manifest.layout,
            shard=manifest.shard(rank),
            base=entry.upload_identities[rank],
        )
        await coordinator.sync_upload(chunk.write, TransportState.TERMINAL_SUCCESS, True)
        committed = await coordinator.commit_chunk(chunk.write, chunk.chunk_bytes)
        assert not committed["finished"]
    for event in started:
        assert await asyncio.wait_for(asyncio.to_thread(event.wait), 2)
    await asyncio.gather(*(client.drain_index_tasks() for client in clients))


def test_two_chunks_per_shard_require_terminal_and_bytes_before_stored():
    asyncio.run(_two_chunks_per_shard_require_terminal_and_bytes_before_stored())


async def _two_chunks_per_shard_require_terminal_and_bytes_before_stored():
    engine = FakeTransferEngine()
    stores = [make_chunk_store(rank, engine) for rank in (0, 1)]
    coordinator = VectorCoordinator([LocalShardClient(store) for store in stores])
    manifest = replace(make_manifest(), upload_mode="chunked_cagra")
    entry = await coordinator.create_entry(manifest, uploader_epoch="sender")
    for rank in (0, 1):
        base = entry.upload_identities[rank]
        for first_page in (0, 1):
            reply = await coordinator.begin_chunk(manifest.key, rank, first_page, 1)
            chunk = PromptChunkIdentity.from_dict(
                reply,
                layout=manifest.layout,
                shard=manifest.shard(rank),
                base=base,
            )
            with pytest.raises(Exception, match="no closed successful PUT"):
                await coordinator.commit_chunk(chunk.write, chunk.chunk_bytes)
            terminal = await coordinator.sync_upload(
                chunk.write, TransportState.TERMINAL_SUCCESS, True
            )
            assert terminal["terminal_ack"] is True
            committed = await coordinator.commit_chunk(chunk.write, chunk.chunk_bytes)
            assert committed["complete_pages"] == first_page + 1
            assert (
                await coordinator.commit_chunk(chunk.write, chunk.chunk_bytes)
            ) == committed
        assert stores[rank].entries[manifest.key].received_bytes == 0
        with pytest.raises(Exception, match="chunk progress is unavailable"):
            await coordinator.begin_chunk(manifest.key, rank, 1, 1)
        await coordinator.sync_upload(base, TransportState.TERMINAL_SUCCESS, True)
        await coordinator.commit_shard(
            manifest.key,
            rank,
            ENTRY_BYTES,
            FirstTokenMetadata(output_token_id=7) if rank == 0 else None,
        )
    assert all(stores[rank].entries[manifest.key].received_bytes == ENTRY_BYTES for rank in (0, 1))
    assert coordinator.entries[manifest.key].state.value == "stored"


def test_unknown_child_write_pins_destination_after_cancel():
    asyncio.run(_unknown_child_write_pins_destination_after_cancel())


async def _unknown_child_write_pins_destination_after_cancel():
    engine = FakeTransferEngine()
    stores = [make_chunk_store(rank, engine) for rank in (0, 1)]
    coordinator = VectorCoordinator([LocalShardClient(store) for store in stores])
    manifest = replace(make_manifest("unknown-chunk"), upload_mode="chunked_cagra")
    entry = await coordinator.create_entry(manifest, uploader_epoch="sender")
    reply = await coordinator.begin_chunk(manifest.key, 0, 0, 1)
    chunk = PromptChunkIdentity.from_dict(
        reply,
        layout=manifest.layout,
        shard=manifest.shard(0),
        base=entry.upload_identities[0],
    )
    unknown = await coordinator.sync_upload(chunk.write, TransportState.UNKNOWN, False)
    assert unknown["terminal_ack"] is False
    await coordinator.cancel_entry(manifest.key, "request cancelled")
    assert not stores[0].entries[manifest.key].resources_released
    assert stores[0].entries[manifest.key].allocation_guard.value is not None
    await coordinator.sync_upload(chunk.write, TransportState.TERMINAL_SUCCESS, True)
    for _ in range(3):
        stores[0]._progress_releases()
        await asyncio.sleep(0)
    assert stores[0].entries[manifest.key].resources_released
