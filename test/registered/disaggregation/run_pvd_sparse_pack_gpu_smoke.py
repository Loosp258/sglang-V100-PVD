"""Standalone real-CUDA byte check for V sparse gather/pack (no pytest needed).

This is a kernel correctness smoke, not an end-to-end performance benchmark.
It allocates bounded synthetic tensors and does not contact a serving V.
"""

import argparse
import json

import torch
from sglang.srt.disaggregation.pvd.kv_packer import PVD_TENSOR_LAYOUT
from sglang.srt.disaggregation.pvd.protocol import KVLayoutSignature, KVShardManifest
from sglang.srt.disaggregation.pvd.sparse_copy import copy_sparse_kv_into
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_pack_plan import build_sparse_pack_plan
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


def check(device, rank, dtype, *, multiblock):
    from sglang.srt.disaggregation.pvd.triton_sparse_pack import SparsePackWorkspace

    layers, rows, heads, head_dim = (
        2,
        128 if multiblock else 12,
        2,
        128 if multiblock else 8,
    )
    selected_a = tuple(range(64, 0, -1)) if multiblock else (7, 0, 9)
    selected_b = tuple(range(65, 1, -1)) if multiblock else (8, 1)
    element_bytes = torch.empty((), dtype=dtype).element_size()
    per_token = heads * head_dim * element_bytes
    source_bytes = 2 * layers * rows * per_token
    layout = KVLayoutSignature(
        model_id="synthetic-v100s-smoke",
        model_revision="byte-check-v1",
        kv_dtype=str(dtype),
        page_size=4,
        num_layers=layers,
        total_kv_heads=2 * heads,
        kv_heads_per_rank=heads,
        head_dim=head_dim,
        tp_size=2,
        pp_size=1,
        tensor_layout=PVD_TENSOR_LAYOUT,
        extra={
            "component_count": 2 * layers,
            "component_dtypes": [str(dtype)] * (2 * layers),
            "component_token_shapes": [[heads, head_dim]] * (2 * layers),
            "component_bytes_per_token": [per_token] * (2 * layers),
        },
    )
    shard = KVShardManifest(
        rank=rank,
        rail="mlx5_0",
        expected_bytes=source_bytes,
        page_count=rows // 4,
        last_page_valid_tokens=2,
        layer_start=0,
        layer_end=layers,
    )
    specs = (
        SparseKVSpec(
            "req",
            "inc",
            "op",
            16,
            "entry",
            "index",
            "mapping",
            layout.fingerprint,
            0,
            rank * heads + 1,
            selected_a,
        ),
        SparseKVSpec(
            "req",
            "inc",
            "op",
            16,
            "entry",
            "index",
            "mapping",
            layout.fingerprint,
            1,
            rank * heads,
            selected_b,
        ),
    )
    manifest = SparseDeliveryManifest(specs, str(dtype), head_dim)
    plan = build_sparse_pack_plan(manifest, layout, shard)
    # Random exact integers avoid FP16 overflow/repeated rows masking a wrong
    # token or head offset in the large multi-block case.
    generator = torch.Generator(device=device).manual_seed(1603 + rank)
    values = torch.randint(
        0,
        256,
        (source_bytes // element_bytes,),
        generator=generator,
        device=device,
        dtype=torch.int32,
    ).to(dtype)
    source = values.contiguous().view(torch.uint8)
    actual = torch.empty(manifest.nbytes, dtype=torch.uint8, device=device)
    reference = torch.empty_like(actual)
    kwargs = {
        "manifest": manifest,
        "layout": layout,
        "shard": shard,
        "entry_transfer_id": "entry",
        "index_version": "index",
        "id_mapping_version": "mapping",
        "allow_cuda": True,
    }
    budget = TransferBudget(plan.metadata_bytes, 1)
    workspace = SparsePackWorkspace(
        manifest,
        shard=shard,
        layout=layout,
        device=device,
        budget=budget,
        owner=f"smoke-rank{rank}-{dtype}",
    )
    completed = False
    try:
        copy_sparse_kv_into(source, actual, **kwargs, fused_workspace=workspace)
        copy_sparse_kv_into(source, reference, **kwargs)
        torch.cuda.synchronize(device)
        completed = True
        if not torch.equal(actual, reference):
            difference = torch.nonzero(actual != reference).flatten()[0].item()
            raise AssertionError(
                f"rank {rank} {dtype} multiblock={multiblock}: "
                f"byte mismatch at offset {difference}"
            )
    finally:
        if not completed:
            torch.cuda.synchronize(device)
        workspace.release_after_fence()
    if budget.snapshot()["used_staging_bytes"] != 0:
        raise AssertionError("metadata budget was not refunded after CUDA fence")
    return {
        "rank": rank,
        "dtype": str(dtype),
        "multiblock": multiblock,
        "compared_bytes": manifest.nbytes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--current-device",
        default=None,
        help="Optional thread-current CUDA device to test cross-device launch safety",
    )
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        parser.error("a real indexed CUDA device is required")
    if args.current_device is not None:
        torch.cuda.set_device(args.current_device)
    current_before = torch.cuda.current_device()
    results = [
        check(device, rank, dtype, multiblock=multiblock)
        for rank in (0, 1)
        for dtype in (torch.float16, torch.bfloat16, torch.float32)
        for multiblock in (False, True)
    ]
    if torch.cuda.current_device() != current_before:
        raise AssertionError("sparse pack did not restore the caller's CUDA device")
    print(
        json.dumps(
            {
                "device": str(device),
                "current_device": current_before,
                "checks": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
