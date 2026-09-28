"""Matched V100S search cost and synthetic recall for full vs split CAGRA."""

# ruff: noqa: I001 -- cuVS must load before Torch/SGLang in this CloudLab wheel.

from __future__ import annotations

import argparse
import json
import statistics
import time

from cuvs.neighbors import cagra as _cagra  # noqa: F401

import torch
from sglang.srt.disaggregation.pvd.cagra_backend import CagraIndexBackend

from benchmark.pvd_prefix_index_reuse import PrefixIndexReuseExperiment


def overlap(left: torch.Tensor, right: torch.Tensor) -> int:
    return sum(
        len(set(a).intersection(b)) for a, b in zip(left.tolist(), right.tolist())
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--rows", type=int, default=2304)
    parser.add_argument("--prefix-rows", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    if (
        min(args.heads, args.dim, args.queries, args.top_k) < 1
        or not 16 < args.prefix_rows < args.rows
        or args.rows - args.prefix_rows <= 16
        or args.top_k > 64
    ):
        parser.error("invalid native CAGRA shape")

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
    vectors = {
        (0, head): torch.randn(
            args.rows, args.dim, device=device, generator=generator
        )
        for head in range(args.heads)
    }
    queries = {
        head: torch.randn(args.queries, args.dim, device=device, generator=generator)
        for head in vectors
    }
    backend.synchronize()
    full = {}
    experiment = PrefixIndexReuseExperiment(
        backend, vector_space="matched-prompt", metric="ip"
    )
    entry = None
    try:
        for head, value in vectors.items():
            full[head] = backend.build(
                value, vector_space="matched-prompt", metric="ip"
            )
        entry, timing = experiment.build(
            vectors,
            prefix_rows=args.prefix_rows,
            prefix_identity="same-exact-prompt-k-prefix",
        )
        print(
            json.dumps(
                {
                    "built_full_graphs": len(full),
                    "built_prefix_graphs": timing.prefix_builds,
                    "built_tail_graphs": timing.tail_builds,
                    "native_retained_bytes": backend.runtime.global_limit.get_allocated_bytes(),
                }
            ),
            flush=True,
        )

        def search(arm):
            result = {}
            backend.synchronize()
            start = time.perf_counter()
            for head, query in queries.items():
                if arm == "full":
                    rows, _ = backend.search(full[head], query, top_k=args.top_k)
                else:
                    rows, _ = experiment.search(
                        entry, head, query, top_k=args.top_k
                    )
                result[head] = rows
            backend.synchronize()
            return time.perf_counter() - start, result

        # Both kernels and their search workspaces are warmed before timing.
        search("full")
        search("split")
        timings = {"full": [], "split": []}
        last = {}
        for arm in ("full", "split", "split", "full"):
            elapsed, rows = search(arm)
            timings[arm].append(elapsed)
            last[arm] = rows
            print(json.dumps({"arm": arm, "search_seconds": elapsed}), flush=True)

        exact_match = {"full": 0, "split": 0, "agreement": 0}
        for head, value in vectors.items():
            exact = torch.topk(queries[head] @ value.T, args.top_k, dim=1).indices
            exact = exact.cpu()
            full_rows = last["full"][head].cpu()
            split_rows = last["split"][head].cpu()
            exact_match["full"] += overlap(full_rows, exact)
            exact_match["split"] += overlap(split_rows, exact)
            exact_match["agreement"] += overlap(split_rows, full_rows)
        denominator = args.heads * args.queries * args.top_k
        print(
            "SUMMARY "
            + json.dumps(
                {
                    "shape": vars(args),
                    "order": ["full", "split", "split", "full"],
                    "full_search_median_seconds": statistics.median(timings["full"]),
                    "split_search_median_seconds": statistics.median(timings["split"]),
                    "full_recall_at_k": exact_match["full"] / denominator,
                    "split_recall_at_k": exact_match["split"] / denominator,
                    "full_split_overlap_at_k": exact_match["agreement"] / denominator,
                }
            ),
            flush=True,
        )
    finally:
        if entry is not None:
            experiment.close(entry)
        for index in full.values():
            backend.dispose(index)
        backend.synchronize()
        print(
            "native_retained_bytes_final="
            + str(backend.runtime.global_limit.get_allocated_bytes()),
            flush=True,
        )


if __name__ == "__main__":
    main()
