"""Cross-node V->D sparse Mooncake WRITE gate with synthetic Prompt KV.

Run after run_pvd_native_upload_index_gpu.py --retain-entry. This verifies
terminal transfer proof, CUDA receive ordering, exact selected K/V bytes and
safe Delivery closure. It does NOT install a Decode working set or run a model.
"""

import argparse
import asyncio
import json
import sys
import time
import uuid


async def run(args):
    import aiohttp
    import torch
    from sglang.srt.disaggregation.pvd.client import PVDCoordinatorClient
    from sglang.srt.disaggregation.pvd.control_server import HttpShardClient
    from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import (
        CUDASparseReceiveRegistry,
    )
    from sglang.srt.disaggregation.pvd.mooncake_engine import (
        MooncakePVDTransferEngine,
    )
    from sglang.srt.disaggregation.pvd.protocol import KVEntryKey
    from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
    from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

    if (
        not torch.cuda.is_available()
        or args.expected_gpu not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("requested D GPU is not the expected device")
    key = KVEntryKey("synthetic-pv-gate", "synthetic-req", args.transfer_id)
    generator = torch.Generator().manual_seed(20260924)
    keys, values = [], []
    for _ in range(2):
        shape = (1024, 2, 32)
        keys.append(
            torch.nn.functional.normalize(
                torch.randn(shape, generator=generator), dim=-1
            ).half()
        )
    for _ in range(2):
        values.append(torch.randn(shape, generator=generator).half())

    engine = MooncakePVDTransferEngine(
        hostname=args.decode_host,
        gpu_id=0,
        rail=args.rail,
        budget=TransferBudget(8 << 20, 8),
    )
    budget = TransferBudget(8 << 20, 8)
    registry = CUDASparseReceiveRegistry(
        engine, budget, receiver_epoch="synthetic-d:" + uuid.uuid4().hex,
        device="cuda:0",
    )
    clients = {
        rank: HttpShardClient(
            rank, f"{args.vector_base_url}:{args.shard_port_base + rank}"
        )
        for rank in (0, 1)
    }
    coordinator = PVDCoordinatorClient(args.coordinator_url)
    report = {
        "schema": "pvd-native-sparse-receive-gpu-v1",
        "status": "failed",
        "transfer_id": args.transfer_id,
        "rail": args.rail,
        "verified_groups_per_rank": [],
    }
    records = []
    try:
        async with aiohttp.ClientSession() as session:
            for rank in (0, 1):
                async with session.get(
                    f"{args.vector_base_url}:{args.shard_port_base + rank}"
                    "/internal/health"
                ) as response:
                    health = await response.json()
                    if response.status != 200:
                        raise RuntimeError(f"V health HTTP {response.status}: {health}")
                if (
                    health["rank"] != rank
                    or health["rail"] != args.rail
                    or health["sparse_packing_mode"]
                    != "cuda_synchronous_experimental"
                    or not health["ready"]
                    or not any(
                        entry["key"]["transfer_id"] == key.transfer_id
                        for entry in health["entries"]
                    )
                ):
                    raise RuntimeError(f"V rank {rank} not ready for sparse Delivery")
                # Query identity and versions come from the V search response;
                # the selected logical token ids are the actual Delivery input.
                async with session.post(
                    f"{args.vector_base_url}:{args.shard_port_base + rank}"
                    "/internal/v1/indexes/search",
                    json={
                        "vector_space": "synthetic-target",
                        "positional_encoding": "rope_applied",
                        "transfer_id": key.transfer_id,
                        "layer": 0,
                        "kv_head": rank,
                        "queries": [keys[0][1, rank].float().tolist()],
                        "top_k": 2,
                    },
                ) as response:
                    answer = await response.json()
                    if response.status != 200:
                        raise RuntimeError(f"V search HTTP {response.status}: {answer}")
                tokens = tuple(answer["token_ids"])
                if not tokens or answer["kv_head"] != rank:
                    raise AssertionError("V returned an empty or wrong-head selection")
                specs = tuple(
                    SparseKVSpec(
                        request_id=key.req_id,
                        incarnation="synthetic-d-gate",
                        operation_id="native-sparse:" + args.transfer_id,
                        target_tokens=4,
                        entry_transfer_id=key.transfer_id,
                        index_version=answer["index_version"],
                        id_mapping_version=answer["id_mapping_version"],
                        layout_fingerprint=args.layout_fingerprint,
                        layer=layer,
                        kv_head=rank,
                        token_ids=tokens,
                    )
                    for layer in (0, 1)
                )
                manifest = SparseDeliveryManifest(specs, "torch.float16", 32)
                record = registry.prepare(
                    manifest,
                    key=key,
                    rank=rank,
                    rail=args.rail,
                    endpoint="synthetic-d-gate",
                    sender_epoch=health["worker_epoch"],
                    client=clients[rank],
                )
                records.append(record)
                await record.start()
                deadline = time.monotonic() + 30
                while not record.snapshot()["ready"] and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                    await record.poll()
                if not record.snapshot()["ready"]:
                    raise TimeoutError(f"V rank {rank} sparse WRITE did not complete")
                # A remote terminal-success proof was observed before the
                # CPU-initiated CUDA synchronization and any GPU read.
                registry.ordering.after_remote_write(record._registration)
                views = manifest.payload_views(record._buffer)
                try:
                    for spec, payload in zip(specs, views, strict=True):
                        expected = torch.stack(
                            (
                                keys[spec.layer][list(tokens), rank],
                                values[spec.layer][list(tokens), rank],
                            )
                        ).to("cuda:0")
                        torch.testing.assert_close(payload.tensor, expected, rtol=0, atol=0)
                finally:
                    for payload in views:
                        payload.close()
                report["verified_groups_per_rank"].append(len(specs))
                report.setdefault("received_bytes_per_rank", []).append(
                    manifest.nbytes
                )
        for record in records:
            if not await record.close():
                raise RuntimeError("D destination could not be safely closed")
        if registry.snapshot() or budget.snapshot()["used_staging_bytes"] != 0:
            raise AssertionError("D retained a registered sparse destination")
        await coordinator.release_entry(key)
        report["released_entry"] = True
        report["status"] = "passed"
    finally:
        errors = await registry.close()
        if errors:
            report["cleanup_errors"] = errors
        for client in clients.values():
            await client.close()
        await coordinator.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-host", required=True)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--vector-base-url", required=True)
    parser.add_argument("--shard-port-base", type=int, required=True)
    parser.add_argument("--transfer-id", required=True)
    parser.add_argument("--layout-fingerprint", required=True)
    parser.add_argument("--rail", required=True)
    parser.add_argument("--expected-gpu", required=True)
    report = asyncio.run(run(parser.parse_args(argv)))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
