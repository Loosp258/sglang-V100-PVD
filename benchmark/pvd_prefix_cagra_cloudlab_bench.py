"""Isolated native CAGRA A/B for complete-Prompt graph reuse on one V GPU.

The same prebuilt K tensors, backend instance, parameters and query are used
for both arms. ABBA order follows one unmeasured warmup. This is a build-path
microbenchmark; the prototype is not wired into the V service.
"""

# ruff: noqa: I001 -- cuVS must load before Torch/SGLang in this CloudLab wheel.

from __future__ import annotations

import argparse
import json
import statistics
import time

# The validated CloudLab wheel needs cuVS loaded before Torch/SGLang.
from cuvs.neighbors import cagra as _cagra  # noqa: F401

import torch
from sglang.srt.disaggregation.pvd.cagra_backend import CagraIndexBackend

from benchmark.pvd_prefix_index_reuse import PrefixIndexReuseExperiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--rows", type=int, default=2304)
    parser.add_argument("--prefix-rows", type=int)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--entries", type=int, default=4)
    args = parser.parse_args()
    if args.prefix_rows is None:
        args.prefix_rows = args.rows
    if min(args.heads, args.dim, args.entries) < 1 or args.rows <= 16:
        parser.error("positive shape and rows > 16 required")
    if not 16 < args.prefix_rows <= args.rows:
        parser.error("prefix rows must be in (16, rows]")
    if 0 < args.rows - args.prefix_rows <= 16:
        parser.error("native CAGRA tail requires more than 16 rows")

    device = torch.device(args.device)
    backend = CagraIndexBackend(
        device=device,
        native_bytes_per_index=536870912,
        global_native_cap_bytes=671088640,
        graph_degree=8,
        intermediate_degree=16,
        itopk_size=64,
    )
    generator = torch.Generator(device=device).manual_seed(20260929)
    # Inputs are prepared once, outside every timed region. Each Entry owns
    # distinct storage, even when its complete K values are identical.
    common = {
        (0, head): torch.randn(
            args.prefix_rows,
            args.dim,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        for head in range(args.heads)
    }
    entry_vectors = []
    for _ in range(args.entries):
        values = {}
        for head, prefix in common.items():
            tail = torch.randn(
                args.rows - args.prefix_rows,
                args.dim,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            values[head] = torch.cat((prefix, tail)).contiguous()
        entry_vectors.append(values)
    query = entry_vectors[0][(0, 0)][100:101].contiguous()
    torch.cuda.synchronize(device)

    warm = backend.build(
        entry_vectors[0][(0, 0)], vector_space="matched-prompt", metric="ip"
    )
    backend.dispose(warm)
    backend.synchronize()
    print("warmup_complete", flush=True)

    results = []
    for arm in ("baseline", "shared", "shared", "baseline"):
        ready_seconds = []
        verify_seconds = 0.0
        graph_builds = 0
        group = None
        torch.cuda.reset_peak_memory_stats(device)
        try:
            if arm == "shared":
                group = PrefixIndexReuseExperiment(
                    backend,
                    vector_space="matched-prompt",
                    metric="ip",
                    retain_idle=True,
                )
            for entry_no in range(args.entries):
                vectors = entry_vectors[entry_no]
                start = time.perf_counter()
                if arm == "baseline":
                    indexes = {
                        head: backend.build(value, vector_space="matched-prompt", metric="ip")
                        for head, value in vectors.items()
                    }
                    backend.synchronize()
                    graph_builds += len(indexes)
                else:
                    entry, timing = group.build(
                        vectors,
                        prefix_rows=args.prefix_rows,
                        prefix_identity="exact-shared-prompt-k-prefix",
                    )
                    graph_builds += timing.prefix_builds + timing.tail_builds
                    verify_seconds += timing.verify_seconds
                ready_seconds.append(time.perf_counter() - start)
                # Validate a native search after the Entry becomes ready.
                if arm == "baseline":
                    rows, scores = backend.search(indexes[(0, 0)], query, top_k=4)
                    backend.dispose(indexes[(0, 0)])
                    for head, index in indexes.items():
                        if head != (0, 0):
                            backend.dispose(index)
                else:
                    rows, scores = group.search(entry, (0, 0), query, top_k=4)
                    group.close(entry)
                search_preview = {
                    "rows": rows[0].tolist(),
                    "scores": [round(float(score), 5) for score in scores[0].tolist()],
                    "self_hit": 100 in rows[0].tolist(),
                }
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "entry": entry_no + 1,
                            "ready_seconds": round(ready_seconds[-1], 6),
                            "search_preview": search_preview,
                        }
                    ),
                    flush=True,
                )
        finally:
            if group is not None and "exact-shared-prompt-k-prefix" in group.groups:
                group.evict("exact-shared-prompt-k-prefix")
        result = {
            "arm": arm,
            "ready_seconds": ready_seconds,
            "sum_ready_seconds": sum(ready_seconds),
            "verify_seconds": verify_seconds,
            "graph_builds": graph_builds,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "native_retained_bytes_after_arm": backend.runtime.global_limit.get_allocated_bytes(),
        }
        results.append(result)
        print(json.dumps(result), flush=True)

    summary = {
        "device": str(device),
        "shape": vars(args),
        "order": [result["arm"] for result in results],
        "baseline_median_sum_ready_seconds": statistics.median(
            result["sum_ready_seconds"] for result in results if result["arm"] == "baseline"
        ),
        "shared_median_sum_ready_seconds": statistics.median(
            result["sum_ready_seconds"] for result in results if result["arm"] == "shared"
        ),
        "baseline_graph_builds_per_arm": args.entries * args.heads,
        "shared_graph_builds_per_arm": args.heads * (
            args.entries + 1 if args.prefix_rows < args.rows else 1
        ),
        "native_retained_bytes_final": backend.runtime.global_limit.get_allocated_bytes(),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
