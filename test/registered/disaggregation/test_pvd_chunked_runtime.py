"""P chunk publication, V prefix indexing and final Entry commit together."""

import asyncio

import torch
from sglang.srt.disaggregation.pvd.coordinator import (
    LocalShardClient,
    VectorCoordinator,
)
from sglang.srt.disaggregation.pvd.kv_packer import pack_full_prompt_kv_head_shard
from sglang.srt.disaggregation.pvd.prompt_index import PromptIndexManager
from sglang.srt.disaggregation.pvd.protocol import FirstTokenMetadata
from sglang.srt.disaggregation.pvd.runtime import PVDPrefillRuntime
from sglang.srt.disaggregation.pvd.transfer_engine import (
    FakeTransferEngine,
    TransferHandle,
    TransferStatus,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransportState
from sglang.srt.disaggregation.pvd.upload_manager import PVDUploadManager
from sglang.srt.disaggregation.pvd.vector_store import VectorKVStore
from test_pvd_chunked_prompt_index import ExtendableExact
from test_pvd_prompt_index import SPACE, build_entry
from test_pvd_prompt_vectors import pack_shard
from test_pvd_upload_lifecycle import DirectCoordinatorClient


class FakeBatchEngine(FakeTransferEngine):
    def submit_batch_put(self, slices, remote, *, remote_offsets):
        children = [
            self.submit_put(item, remote, remote_offset=offset)
            for item, offset in zip(slices, remote_offsets, strict=True)
        ]
        assert all(item.status is TransferStatus.SUCCESS for item in children)
        return TransferHandle(
            transfer_id="fake-batch-" + children[0].transfer_id,
            status=TransferStatus.SUCCESS,
            transferred_bytes=sum(item.transferred_bytes for item in children),
            transport_state=TransportState.TERMINAL_SUCCESS,
        )


def test_prefill_chunks_build_private_prefix_and_publish_same_graph():
    asyncio.run(_run())


async def _run():
    pool, layout, manifest, _, _ = build_entry(prompt_tokens=12)
    engine = FakeBatchEngine()
    stores = []
    for rank in (0, 1):
        shard = manifest.shard(rank)
        stores.append(VectorKVStore(
            rank=rank, world_size=2, rail=f"mlx5_{rank}", device="cpu",
            total_pages=8, page_bytes=shard.expected_bytes // shard.page_count,
            endpoint=f"v{rank}", transfer_engine=engine,
            allow_cpu_for_tests=True,
            prompt_index=PromptIndexManager(
                vector_space=SPACE, backend=ExtendableExact(),
            ),
            enable_chunked_upload=True,
        ))
    shard_clients = [LocalShardClient(s) for s in stores]
    coordinator = VectorCoordinator(shard_clients)
    client = DirectCoordinatorClient(coordinator)
    client.begin_chunk = coordinator.begin_chunk
    client.commit_chunk = coordinator.commit_chunk
    runtime = PVDPrefillRuntime(
        model_instance_id=manifest.key.model_instance_id,
        coordinator=client, transfer_engine=engine,
        upload_manager=PVDUploadManager(), worker_epoch="sender",
        poll_interval_seconds=0,
    )
    lease = await runtime.create_entry(
        req_id=manifest.key.req_id, transfer_id=manifest.key.transfer_id,
        layout=layout, prompt_token_count=12,
        shards={rank: manifest.shard(rank) for rank in (0, 1)},
        upload_mode="chunked_cagra",
    )
    for first_page, page_count in ((0, 1), (1, 2)):
        tasks = []
        for rank in (0, 1):
            packed = pack_full_prompt_kv_head_shard(
                pool, list(range(first_page, first_page + page_count)),
                page_size=layout.page_size,
                head_start=rank * layout.kv_heads_per_rank,
                head_count=layout.kv_heads_per_rank,
            )
            tasks.append(runtime.publish_chunk_tensor_shard(
                lease=lease, rank=rank, tensor=packed.tensor,
                first_page=first_page, page_count=page_count,
                first_token=(
                    FirstTokenMetadata(output_token_id=7)
                    if rank == 0 and first_page else None
                ),
            ))
        await asyncio.gather(*tasks)
        await asyncio.gather(*(client.drain_index_tasks() for client in shard_clients))
        for rank, store in enumerate(stores):
            gate = store.prompt_index.gate_for(manifest.key.transfer_id)
            assert gate.searchable is bool(first_page)
            if first_page:
                expected = pack_shard(pool, layout, rank=rank, prompt_tokens=12)[0]
                entry = store.entries[manifest.key]
                offset = entry.allocation.start_page * store.page_bytes
                assert torch.equal(
                    store.pool[offset:offset + expected.expected_bytes],
                    expected.tensor,
                )
    assert coordinator.entries[manifest.key].state.value == "stored"
    for store in stores:
        store.close()
