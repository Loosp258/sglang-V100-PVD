"""Exercise a V PromptIndexManager with several real cuVS graphs under one cap.

This is a component gate, not a production serving, target-Q recall, or
56-graph capacity test. It uses synthetic, packed Prompt K on one V100S.
"""

import argparse
import copy
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--expect-cuvs-version", required=True)
    args = parser.parse_args(argv)

    import cuvs
    import torch
    from sglang.srt.disaggregation.pvd.cagra_backend import CagraIndexBackend
    from sglang.srt.disaggregation.pvd.kv_packer import (
        PVD_TENSOR_LAYOUT,
        describe_kv_layout,
        pack_full_prompt_kv_head_shard,
    )
    from sglang.srt.disaggregation.pvd.prompt_index import (
        PromptIndexManager,
        SearchRequestIdentity,
    )
    from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED
    from sglang.srt.disaggregation.pvd.protocol import (
        KVLayoutSignature,
        KVShardManifest,
    )
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

    if not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise RuntimeError("requested CUDA device index is unavailable")
    gpu = torch.cuda.get_device_name(args.device)
    if args.expected_gpu not in gpu or cuvs.__version__ != args.expect_cuvs_version:
        raise RuntimeError("GPU or cuVS candidate does not match the request")

    class Pool:
        start_layer = 0
        end_layer = 2

        def __init__(self):
            generator = torch.Generator().manual_seed(20260924)
            shape = (1024, 2, 32)
            self.k_buffer = [
                torch.randn(shape, generator=generator).half() for _ in range(2)
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
        model_id="synthetic-shared-cap", model_revision="gate-v1",
        kv_dtype=compute["component_dtypes"][0], page_size=4,
        num_layers=2, total_kv_heads=2, kv_heads_per_rank=1,
        head_dim=32, tp_size=2, pp_size=1, tensor_layout=PVD_TENSOR_LAYOUT,
        extra=extra,
    )
    packed = pack_full_prompt_kv_head_shard(
        pool, list(range(256)), page_size=4, head_start=0, head_count=1
    )
    shard = KVShardManifest(
        rank=0, rail="mlx5_0", expected_bytes=packed.expected_bytes,
        page_count=256, last_page_valid_tokens=4,
        layer_start=0, layer_end=2,
    )

    root_cap, graph_cap = 640 << 20, 512 << 20
    budget = TransferBudget(root_cap + (32 << 20), 1)
    backend = CagraIndexBackend(
        device=f"cuda:{args.device}", native_bytes_per_index=graph_cap,
        graph_degree=32, intermediate_degree=64, itopk_size=64,
        global_native_cap_bytes=root_cap,
    )
    manager = PromptIndexManager(
        vector_space="synthetic-target", backend=backend, metric="ip",
        budget=budget,
    )
    report = {
        "schema": "pvd-cagra-shared-manager-gpu-v1", "status": "failed",
        "gpu": gpu, "device": args.device, "cuvs_version": cuvs.__version__,
        "root_cap_bytes": root_cap, "per_graph_cap_bytes": graph_cap,
        "initial_charged_bytes": budget.snapshot()["used_staging_bytes"],
    }
    if report["initial_charged_bytes"] != root_cap:
        raise AssertionError("manager did not reserve exactly one native root cap")
    entry_ids = ("synthetic-entry-a", "synthetic-entry-b")
    try:
        for entry_id in entry_ids:
            manager.note_kv_readable(entry_id)
            if not manager.build(entry_id, packed.tensor, layout=layout, manifest=shard):
                raise AssertionError(f"index build did not become ready: {entry_id}")
        if len(backend._owners) != 4:
            raise AssertionError("expected two native graphs for each of two Entries")
        root_live = backend.runtime.global_allocated_bytes()
        if root_live <= 0 or root_live > root_cap:
            raise AssertionError("native graphs escaped the shared RMM cap")
        charged = budget.snapshot()["used_staging_bytes"]
        vectors = sum(
            item.vectors.numel() * item.vectors.element_size()
            for record in manager._entries.values()
            for item in record.vectors.values()
        )
        if charged != root_cap + vectors:
            raise AssertionError("retained index copies were not charged separately")
        identity = SearchRequestIdentity(
            vector_space="synthetic-target", positional_encoding=ROPE_APPLIED,
            entry_transfer_id=entry_ids[0], layer=0, kv_head=0,
        )
        queries = manager._entries[entry_ids[0]].vectors[(0, 0)].vectors[:4]
        result = manager.search(identity, queries=queries, top_k=4)
        if not result.selection.token_ids:
            raise AssertionError("native manager search returned no token IDs")
        report.update(
            graph_count=4, root_allocated_bytes=root_live,
            charged_with_entries_bytes=charged,
            vector_copy_bytes=vectors,
            search_token_count=len(result.selection.token_ids),
        )
    finally:
        for entry_id in entry_ids:
            manager.close(entry_id)
    report["charged_after_close_bytes"] = budget.snapshot()["used_staging_bytes"]
    report["root_after_close_bytes"] = backend.runtime.global_allocated_bytes()
    if (
        report["charged_after_close_bytes"] == root_cap
        and report["root_after_close_bytes"] == 0
        and not backend._owners
    ):
        report["status"] = "passed"
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
