"""Upload synthetic Prompt KV from P over Mooncake to two live V CAGRA ranks.

This gate exercises native P->V WRITE, Entry commit, V index build/search and
release. Queries are copied synthetic K rows, not real target-model Q.
"""

import argparse
import asyncio
import copy
import json
import sys
import time
import uuid


async def _json(session, method, url, payload=None):
    async with session.request(method, url, json=payload) as response:
        body = await response.json()
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}: {body}")
        return body


async def run(args):
    import aiohttp
    import torch
    from sglang.srt.disaggregation.pvd.client import PVDCoordinatorClient
    from sglang.srt.disaggregation.pvd.kv_packer import (
        PVD_TENSOR_LAYOUT,
        describe_kv_layout,
        pack_full_prompt_kv_head_shard,
    )
    from sglang.srt.disaggregation.pvd.mooncake_engine import (
        MooncakePVDTransferEngine,
    )
    from sglang.srt.disaggregation.pvd.protocol import (
        FirstTokenMetadata,
        KVLayoutSignature,
        KVShardManifest,
    )
    from sglang.srt.disaggregation.pvd.runtime import PVDPrefillRuntime
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
    from sglang.srt.disaggregation.pvd.upload_manager import PVDUploadManager

    if not torch.cuda.is_available() or args.expected_gpu not in torch.cuda.get_device_name(0):
        raise RuntimeError("requested P GPU is not the expected device")

    class Pool:
        start_layer = 0
        end_layer = 2

        def __init__(self):
            generator = torch.Generator().manual_seed(20260924)
            shape = (1024, 2, 32)
            self.k_buffer = [
                torch.nn.functional.normalize(
                    torch.randn(shape, generator=generator), dim=-1
                ).half()
                for _ in range(2)
            ]
            self.v_buffer = [
                torch.randn(shape, generator=generator).half() for _ in range(2)
            ]

    pool = Pool()
    compute = describe_kv_layout(pool)
    extra = copy.deepcopy(compute)
    extra["component_token_shapes"] = [
        [1, shape[1]] for shape in compute["component_token_shapes"]
    ]
    extra["component_bytes_per_token"] = [
        value // 2 for value in compute["component_bytes_per_token"]
    ]
    layout = KVLayoutSignature(
        model_id="synthetic-target", model_revision="native-pv-gate-v1",
        kv_dtype=compute["component_dtypes"][0], page_size=4,
        num_layers=2, total_kv_heads=2, kv_heads_per_rank=1,
        head_dim=32, tp_size=2, pp_size=1,
        tensor_layout=PVD_TENSOR_LAYOUT, extra=extra,
    )
    packed = {
        rank: pack_full_prompt_kv_head_shard(
            pool, list(range(256)), page_size=4,
            head_start=rank, head_count=1,
        )
        for rank in (0, 1)
    }
    shards = {
        rank: KVShardManifest(
            rank=rank, rail=args.rail, expected_bytes=item.expected_bytes,
            page_count=256, last_page_valid_tokens=4,
            layer_start=0, layer_end=2,
        )
        for rank, item in packed.items()
    }
    page_bytes = shards[0].expected_bytes // shards[0].page_count
    if page_bytes != args.page_bytes or any(
        item.expected_bytes != shards[0].expected_bytes for item in shards.values()
    ):
        raise RuntimeError("V page geometry does not match synthetic Prompt KV")

    coordinator = PVDCoordinatorClient(args.coordinator_url)
    upload_manager = PVDUploadManager()
    engine = MooncakePVDTransferEngine(
        hostname=args.prefill_host, gpu_id=0, rail=args.rail,
        budget=TransferBudget(8 << 20, 4),
    )
    runtime = PVDPrefillRuntime(
        model_instance_id="synthetic-pv-gate", coordinator=coordinator,
        transfer_engine=engine, upload_manager=upload_manager,
        worker_epoch="synthetic-p:" + uuid.uuid4().hex,
    )
    report = {
        "schema": "pvd-native-upload-index-gpu-v1",
        "status": "failed", "rail": args.rail,
        "page_bytes": page_bytes,
    }
    lease = None
    try:
        lease = await runtime.create_entry(
            req_id="synthetic-req", transfer_id=uuid.uuid4().hex,
            layout=layout, prompt_token_count=1024, shards=shards,
            owned_shard_ranks={0, 1},
        )
        gpu_tensors = {
            rank: item.tensor.to(device="cuda:0", non_blocking=False)
            for rank, item in packed.items()
        }
        results = await asyncio.gather(
            *(
                runtime.publish_tensor_shard(
                    lease=lease, rank=rank, tensor=gpu_tensors[rank],
                    endpoint="synthetic-pv-gate", rail=args.rail,
                    first_token=(
                        FirstTokenMetadata(output_token_id=1)
                        if rank == 0 else None
                    ),
                    deadline=time.monotonic() + 30,
                )
                for rank in (0, 1)
            )
        )
        report["committed_shards"] = len(results)
        async with aiohttp.ClientSession() as session:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                snapshots = [
                    await _json(
                        session, "GET",
                        f"{args.vector_base_url}:{args.shard_port_base + rank}"
                        "/internal/v1/indexes",
                    )
                    for rank in (0, 1)
                ]
                records = [
                    snapshot["entries"].get(lease.manifest.key.transfer_id)
                    for snapshot in snapshots
                ]
                if all(record and record["searchable"] for record in records):
                    break
                await asyncio.sleep(0.2)
            else:
                raise RuntimeError(f"V indexes did not become searchable: {records}")
            if any(record["indexed_heads"] != 2 for record in records):
                raise AssertionError("V did not build both layer/head indexes")
            hits = []
            for rank in (0, 1):
                queries = [pool.k_buffer[0][token, rank].float().tolist()
                           for token in range(4)]
                answer = await _json(
                    session, "POST",
                    f"{args.vector_base_url}:{args.shard_port_base + rank}"
                    "/internal/v1/indexes/search",
                    {
                        "vector_space": "synthetic-target",
                        "positional_encoding": "rope_applied",
                        "transfer_id": lease.manifest.key.transfer_id,
                        "layer": 0, "kv_head": rank,
                        "queries": queries, "top_k": 4,
                    },
                )
                found = sum(token in answer["token_ids"] for token in range(4))
                hits.append(found)
                if found != 4 or answer["kv_head"] != rank:
                    raise AssertionError("native V search lost synthetic self-hits")
            report["self_hits_per_rank"] = hits
            report["indexed_heads_per_rank"] = [
                record["indexed_heads"] for record in records
            ]
        await coordinator.release_entry(lease.manifest.key)
        async with aiohttp.ClientSession() as session:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                after = [
                    await _json(
                        session, "GET",
                        f"{args.vector_base_url}:{args.shard_port_base + rank}"
                        "/internal/v1/indexes",
                    )
                    for rank in (0, 1)
                ]
                if all(
                    lease.manifest.key.transfer_id not in snapshot["entries"]
                    and snapshot["budget"]["used_staging_bytes"] == (640 << 20)
                    and not snapshot["quarantined"]
                    for snapshot in after
                ):
                    break
                await asyncio.sleep(0.1)
            else:
                raise RuntimeError("V did not release Entry indexes to the root cap")
        report["released_entry"] = True
        report["root_only_after_release"] = True
        report["status"] = "passed"
    finally:
        if lease is not None and report["status"] != "passed":
            await coordinator.cancel_entry(lease.manifest.key, "synthetic gate failed")
        await coordinator.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-host", required=True)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--vector-base-url", required=True)
    parser.add_argument("--shard-port-base", type=int, required=True)
    parser.add_argument("--rail", required=True)
    parser.add_argument("--page-bytes", type=int, required=True)
    parser.add_argument("--expected-gpu", required=True)
    args = parser.parse_args(argv)
    report = asyncio.run(run(args))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
