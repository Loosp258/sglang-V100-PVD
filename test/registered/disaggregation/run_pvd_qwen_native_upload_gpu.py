"""Upload real Qwen2.5 Prompt KV from P to two native V CAGRA ranks.

Uses the local checkpoint and deterministic prompt token IDs. Samples the
first token greedily from the actual Prefill logits and leaves the Entry on V
for the next D-side real-query gate.
"""

import argparse
import asyncio
import copy
import sys
import time
import uuid
from types import SimpleNamespace


def _validate(runner, args, *, checkpoint=False):
    if not checkpoint or type(runner.model).__name__ != "Qwen2ForCausalLM":
        raise ValueError("a real Qwen2.5 checkpoint is required")

    import aiohttp
    import torch
    from sglang.srt.disaggregation.pvd.client import PVDCoordinatorClient
    from sglang.srt.disaggregation.pvd.draft_forward_adapter import (
        DraftForwardAdapter,
        PrivatePoolAllocator,
    )
    from sglang.srt.disaggregation.pvd.draft_runner_sglang import DraftForwardInputs
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

    config = runner.model.config
    if (
        config.num_key_value_heads != 4
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
        first_logits = adapter.forward(
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
        first_token_id = int(first_logits.argmax(-1).item())
        if not 0 <= first_token_id < config.vocab_size:
            raise AssertionError("Prefill sampled a token outside Qwen vocabulary")
        torch.cuda.synchronize("cuda:0")
        pool = SimpleNamespace(
            start_layer=0,
            end_layer=28,
            k_buffer=[
                runner.token_to_kv_pool.get_key_buffer(layer)[rows].clone()
                for layer in range(28)
            ],
            v_buffer=[
                runner.token_to_kv_pool.get_value_buffer(layer)[rows].clone()
                for layer in range(28)
            ],
        )
    finally:
        if rows:
            torch.cuda.synchronize("cuda:0")
            allocator.clear_mapping(slot)
            allocator.free_kv(rows)
        allocator.free_request(slot)

    compute = describe_kv_layout(pool)
    extra = copy.deepcopy(compute)
    extra["component_token_shapes"] = [
        [2, shape[1]] for shape in compute["component_token_shapes"]
    ]
    extra["component_bytes_per_token"] = [
        value // 2 for value in compute["component_bytes_per_token"]
    ]
    space = "qwen2.5-7b-real-target"
    layout = KVLayoutSignature(
        model_id=space,
        model_revision="cloudlab-local-checkpoint",
        kv_dtype=compute["component_dtypes"][0],
        page_size=4,
        num_layers=28,
        total_kv_heads=4,
        kv_heads_per_rank=2,
        head_dim=128,
        tp_size=2,
        pp_size=1,
        tensor_layout=PVD_TENSOR_LAYOUT,
        extra=extra,
    )
    packed = {
        rank: pack_full_prompt_kv_head_shard(
            pool,
            range(token_count // 4),
            page_size=4,
            head_start=rank * 2,
            head_count=2,
        )
        for rank in (0, 1)
    }
    page_bytes = packed[0].expected_bytes // (token_count // 4)
    if page_bytes != args.page_bytes or any(
        item.expected_bytes != packed[0].expected_bytes for item in packed.values()
    ):
        raise RuntimeError("V page geometry differs from actual Qwen Prompt KV")
    shards = {
        rank: KVShardManifest(
            rank=rank,
            rail=args.rail,
            expected_bytes=item.expected_bytes,
            page_count=token_count // 4,
            last_page_valid_tokens=4,
            layer_start=0,
            layer_end=28,
        )
        for rank, item in packed.items()
    }

    async def upload():
        coordinator = PVDCoordinatorClient(args.coordinator_url)
        engine = MooncakePVDTransferEngine(
            hostname=args.prefill_host,
            gpu_id=0,
            rail=args.rail,
            budget=TransferBudget(128 << 20, 4),
        )
        runtime = PVDPrefillRuntime(
            model_instance_id=space,
            coordinator=coordinator,
            transfer_engine=engine,
            upload_manager=PVDUploadManager(),
            worker_epoch="real-p:" + uuid.uuid4().hex,
        )
        lease = None
        try:
            lease = await runtime.create_entry(
                req_id="real-qwen-prompt",
                transfer_id=uuid.uuid4().hex,
                layout=layout,
                prompt_token_count=token_count,
                shards=shards,
                owned_shard_ranks={0, 1},
            )
            tensors = {
                rank: item.tensor.to("cuda:0") for rank, item in packed.items()
            }
            await asyncio.gather(
                *(
                    runtime.publish_tensor_shard(
                        lease=lease,
                        rank=rank,
                        tensor=tensors[rank],
                        endpoint="real-qwen-p",
                        rail=args.rail,
                        first_token=(
                            FirstTokenMetadata(output_token_id=first_token_id)
                            if rank == 0 else None
                        ),
                        deadline=time.monotonic() + 90,
                    )
                    for rank in (0, 1)
                )
            )
            async with aiohttp.ClientSession() as session:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    snapshots = []
                    for rank in (0, 1):
                        async with session.get(
                            f"{args.vector_base_url}:{args.shard_port_base + rank}"
                            "/internal/v1/indexes"
                        ) as response:
                            snapshot = await response.json()
                            if response.status != 200:
                                raise RuntimeError(
                                    f"V rank {rank} index HTTP {response.status}"
                                )
                            snapshots.append(snapshot)
                    records = [
                        snapshot["entries"].get(lease.manifest.key.transfer_id)
                        for snapshot in snapshots
                    ]
                    if all(record and record["searchable"] for record in records):
                        break
                    await asyncio.sleep(0.5)
                else:
                    raise TimeoutError("V did not build real Qwen Prompt indexes")
                if any(record["indexed_heads"] != 56 for record in records):
                    raise AssertionError("V did not index all 28 layers and two heads")
            return {
                "entry_key": lease.manifest.key.to_dict(),
                "layout_fingerprint": layout.fingerprint,
                "prompt_tokens": token_count,
                "page_bytes": page_bytes,
                "indexed_heads_per_rank": [
                    record["indexed_heads"] for record in records
                ],
                "retained_entry": True,
                "real_checkpoint_kv": True,
                "prefill_first_token_id": first_token_id,
            }
        except BaseException:
            if lease is not None:
                await coordinator.cancel_entry(lease.manifest.key, "real P gate failed")
            raise
        finally:
            await coordinator.close()

    return asyncio.run(upload())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-host", required=True)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--vector-base-url", required=True)
    parser.add_argument("--shard-port-base", type=int, required=True)
    parser.add_argument("--rail", required=True)
    parser.add_argument("--page-bytes", type=int, required=True)
    parser.add_argument("--expected-gpu", required=True)
    args, model_args = parser.parse_known_args(argv)
    from run_pvd_cuda_probe_smoke import main as run_model

    return run_model(
        model_args,
        validator=lambda runner, checkpoint=False: _validate(
            runner, args, checkpoint=checkpoint
        ),
        schema="pvd-qwen2.5-native-p-to-v-upload-v1",
    )


if __name__ == "__main__":
    sys.exit(main())
