"""Real Qwen D target-Q search and V-to-D native sparse KV receive gate.

Requires the retained real Prompt Entry from run_pvd_qwen_native_upload_gpu.py.
Checks two V ranks, one layer and one Q/KV head per rank. This does not install
a working set or run a generated-token Decode forward.
"""

import argparse
import asyncio
import sys
import threading
import time
import uuid


def _validate(runner, args, *, checkpoint=False):
    if not checkpoint or type(runner.model).__name__ != "Qwen2ForCausalLM":
        raise ValueError("a real Qwen2.5 checkpoint is required")

    import torch
    from sglang.srt.disaggregation.pvd.client import PVDCoordinatorClient
    from sglang.srt.disaggregation.pvd.control_server import HttpShardClient
    from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import (
        CUDASparseReceiveRegistry,
    )
    from sglang.srt.disaggregation.pvd.cuda_target_probe import CUDAQwen2TargetProbe
    from sglang.srt.disaggregation.pvd.draft_forward_adapter import (
        DraftForwardAdapter,
        PrivatePoolAllocator,
    )
    from sglang.srt.disaggregation.pvd.draft_runner_sglang import DraftForwardInputs
    from sglang.srt.disaggregation.pvd.mooncake_engine import (
        MooncakePVDTransferEngine,
    )
    from sglang.srt.disaggregation.pvd.prediction import (
        CommittedPrefix,
        DraftPrediction,
        ProbeConfig,
    )
    from sglang.srt.disaggregation.pvd.prompt_index import SearchRequestIdentity
    from sglang.srt.disaggregation.pvd.protocol import KVEntryKey
    from sglang.srt.disaggregation.pvd.search_client import (
        PVDShardSearchClient,
        SearchScope,
    )
    from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
    from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

    config = runner.model.config
    if (
        config.num_key_value_heads != 4
        or config.num_attention_heads != 28
        or config.num_hidden_layers != 28
        or runner.model_config.head_dim != 128
        or args.expected_gpu not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("this bounded gate expects Qwen2.5-7B on V100S")
    token_count = 1024
    generator = torch.Generator().manual_seed(20260924)
    tokens = tuple(torch.randint(3, 1000, (token_count,), generator=generator).tolist())
    allocator = PrivatePoolAllocator(
        runner.req_to_token_pool, runner.token_to_kv_pool_allocator
    )
    slot, rows = allocator.alloc_request(), []
    try:
        rows = allocator.alloc_kv(token_count)
        allocator.write_mapping(slot, 0, rows)
        adapter = DraftForwardAdapter(
            runner,
            architecture="Qwen2ForCausalLM",
            attention_backend="torch_native",
            bytes_per_token=28 * 4 * 128 * 2 * 2,
            device="cuda:0",
        )
        adapter.forward(
            DraftForwardInputs(
                "extend",
                tokens,
                tuple(range(token_count)),
                (token_count,),
                (slot,),
                tuple(rows),
                (0,),
                (token_count,),
            )
        )
        torch.cuda.synchronize("cuda:0")
        expected = {
            head: torch.stack(
                (
                    runner.token_to_kv_pool.get_key_buffer(0)[rows, head],
                    runner.token_to_kv_pool.get_value_buffer(0)[rows, head],
                )
            ).clone()
            for head in (0, 2)
        }
    finally:
        if rows:
            torch.cuda.synchronize("cuda:0")
            allocator.clear_mapping(slot)
            allocator.free_kv(rows)
        allocator.free_request(slot)

    space = "qwen2.5-7b-real-target"
    probe_budget = TransferBudget(256 << 20, 1)
    lock = threading.Lock()
    probe = CUDAQwen2TargetProbe(
        runner,
        ProbeConfig(space, (0,), head_start=0, head_count=15),
        device="cuda:0",
        execution_lock=lock,
        target_model_id=space,
        max_tokens=token_count + 2,
        max_predict_tokens=1,
        transient_bytes_bound=64 << 20,
        budget=probe_budget,
    )
    prefix = CommittedPrefix("real-qwen-prompt", tokens, 0, "prompt")
    with probe.branch():
        query = probe.capture(
            prefix, DraftPrediction(prefix.request_id, prefix.version, (42,))
        )[0]
        if (
            query.positional_encoding != "rope_applied"
            or query.vector_space != space
            or query.positions != (token_count,)
            or query.layer != 0
        ):
            raise AssertionError("D target Q has a foreign identity or position")
        q = {
            0: query.vectors[0, 0].float().clone(),
            1: query.vectors[0, 14].float().clone(),
        }
    if probe_budget.snapshot()["used_staging_bytes"] or lock.locked():
        raise AssertionError("D target Q probe retained private resources")

    async def search_and_receive():
        key = KVEntryKey(space, "real-qwen-prompt", args.transfer_id)
        engine = MooncakePVDTransferEngine(
            hostname=args.decode_host,
            gpu_id=0,
            rail=args.rail,
            budget=TransferBudget(16 << 20, 4),
        )
        budget = TransferBudget(16 << 20, 4)
        registry = CUDASparseReceiveRegistry(
            engine,
            budget,
            receiver_epoch="real-qwen-d:" + uuid.uuid4().hex,
            device="cuda:0",
        )
        endpoints = {
            rank: f"{args.vector_base_url}:{args.shard_port_base + rank}"
            for rank in (0, 1)
        }
        controls = {
            rank: HttpShardClient(rank, url) for rank, url in endpoints.items()
        }
        searches = {
            rank: PVDShardSearchClient(url) for rank, url in endpoints.items()
        }
        coordinator = PVDCoordinatorClient(args.coordinator_url)
        records = []
        report = {
            "entry_key": key.to_dict(),
            "real_target_post_rope_q": True,
            "real_checkpoint_prompt_kv": True,
            "top_k": 10,
            "ranks": [],
        }
        try:
            for rank, head in ((0, 0), (1, 2)):
                health = await controls[rank].health()
                if (
                    health["rank"] != rank
                    or health["rail"] != args.rail
                    or health["sparse_packing_mode"]
                    != "cuda_synchronous_experimental"
                    or not health["ready"]
                ):
                    raise RuntimeError(f"V rank {rank} cannot serve sparse KV")
                result = await searches[rank].search(
                    SearchRequestIdentity(
                        space, "rope_applied", key.transfer_id, 0, head
                    ),
                    queries=[q[rank].tolist()],
                    top_k=10,
                    scope=SearchScope(token_count, 4, 128, "ip"),
                )
                selected = tuple(result.token_ids)
                if len(selected) != 10:
                    raise AssertionError("V did not return ten logical Prompt tokens")
                # Independent model run on D is a cross-node value oracle.
                # If the two checkpoints/forward paths differ, fail the gate.
                scores = (q[rank].to("cuda:0") @ expected[head][0].float().T)
                exact = torch.topk(scores, 10).indices.tolist()
                recall = len(set(selected) & set(exact)) / 10
                spec = SparseKVSpec(
                    request_id=key.req_id,
                    incarnation="real-qwen-d-gate",
                    operation_id="search-receive:" + key.transfer_id,
                    target_tokens=0,
                    entry_transfer_id=key.transfer_id,
                    index_version=result.index_version,
                    id_mapping_version=result.id_mapping_version,
                    layout_fingerprint=args.layout_fingerprint,
                    layer=0,
                    kv_head=head,
                    token_ids=selected,
                )
                manifest = SparseDeliveryManifest((spec,), "torch.float16", 128)
                record = registry.prepare(
                    manifest,
                    key=key,
                    rank=rank,
                    rail=args.rail,
                    endpoint="real-qwen-d-gate",
                    sender_epoch=health["worker_epoch"],
                    client=controls[rank],
                )
                records.append(record)
                await record.start()
                deadline = time.monotonic() + 45
                while not record.snapshot()["ready"] and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                    await record.poll()
                if not record.snapshot()["ready"]:
                    raise TimeoutError(f"V rank {rank} sparse WRITE timed out")
                registry.ordering.after_remote_write(record._registration)
                views = manifest.payload_views(record._buffer)
                try:
                    actual = views[0].tensor
                    oracle = expected[head][:, list(selected)].contiguous()
                    torch.testing.assert_close(actual, oracle, rtol=0, atol=0)
                finally:
                    for payload in views:
                        payload.close()
                report["ranks"].append(
                    {
                        "storage_rank": rank,
                        "query_head": 0 if rank == 0 else 14,
                        "kv_head": head,
                        "recall_at_10": recall,
                        "received_bytes": manifest.nbytes,
                        "kv_bit_exact": True,
                    }
                )
            for record in records:
                if not await record.close():
                    raise RuntimeError("D sparse destination was not fenced")
            if registry.snapshot() or budget.snapshot()["used_staging_bytes"]:
                raise AssertionError("D retained sparse destination memory")
            await coordinator.release_entry(key)
            report["released_entry"] = True
            return report
        finally:
            errors = await registry.close()
            for client in controls.values():
                await client.close()
            for client in searches.values():
                await client.close()
            await coordinator.close()
            if errors:
                raise RuntimeError(f"D retained unsafe sparse resources: {errors}")

    return asyncio.run(search_and_receive())


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
    args, model_args = parser.parse_known_args(argv)
    from run_pvd_cuda_probe_smoke import main as run_model

    return run_model(
        model_args,
        validator=lambda runner, checkpoint=False: _validate(
            runner, args, checkpoint=checkpoint
        ),
        schema="pvd-qwen2.5-native-v-to-d-search-receive-v1",
    )


if __name__ == "__main__":
    sys.exit(main())
