"""Bounded V-side exact-search A/B; no serving, Mooncake or index lifetime claim."""

import argparse
import json
import os
import statistics
import time

import torch
from sglang.srt.disaggregation.pvd.index_search import BruteForceIndexBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--rows", type=int, default=915)
    parser.add_argument("--queries", type=int, default=7)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    if not (
        2 <= args.heads <= 64
        and 8 <= args.rows <= 2048
        and 1 <= args.queries <= 64
        and 1 <= args.dim <= 256
        and 1 <= args.top_k <= min(8, args.rows)
        and 3 <= args.repetitions <= 100
    ):
        parser.error("all dimensions and repetitions must stay within search bounds")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        parser.error("an explicit available cuda:N device is required")
    backend = BruteForceIndexBackend(device=device)
    rng = torch.Generator().manual_seed(915)
    indexes = tuple(
        backend.build(
            torch.randn(args.rows, args.dim, generator=rng).to(device),
            vector_space="pvd-grouped-topk-benchmark",
            metric="ip",
        )
        for _ in range(args.heads)
    )
    queries = tuple(
        torch.randn(args.queries, args.dim, generator=rng).to(device)
        for _ in indexes
    )
    original = os.environ.get("PVD_GROUPED_EXACT_TOPK_REDUCE")
    times = {"stable_sort": [], "topk_reduce": []}
    references = {}
    try:
        for iteration in range(args.repetitions + 2):
            # Alternate order so warm-up, clock and thermal effects do not
            # systematically favor one implementation.
            labels = ("stable_sort", "topk_reduce")
            if iteration % 2:
                labels = labels[::-1]
            for label in labels:
                if label == "topk_reduce":
                    os.environ["PVD_GROUPED_EXACT_TOPK_REDUCE"] = "1"
                else:
                    os.environ.pop("PVD_GROUPED_EXACT_TOPK_REDUCE", None)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                answer = backend.search_grouped(indexes, queries, top_k=args.top_k)
                torch.cuda.synchronize(device)
                if iteration >= 2:
                    times[label].append((time.perf_counter() - started) * 1000)
                if label not in references:
                    references[label] = answer
        for (sort_rows, sort_scores), (reduce_rows, reduce_scores) in zip(
            references["stable_sort"], references["topk_reduce"], strict=True
        ):
            if not torch.equal(sort_rows, reduce_rows) or not torch.equal(
                sort_scores, reduce_scores
            ):
                raise ValueError("Top-K reduction differs from stable full sort")
    finally:
        if original is None:
            os.environ.pop("PVD_GROUPED_EXACT_TOPK_REDUCE", None)
        else:
            os.environ["PVD_GROUPED_EXACT_TOPK_REDUCE"] = original
    print(
        json.dumps(
            {
                "schema": "pvd.grouped_topk.microbenchmark.v1",
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device),
                "shape": {
                    "heads": args.heads,
                    "rows": args.rows,
                    "queries": args.queries,
                    "dim": args.dim,
                    "top_k": args.top_k,
                },
                "repetitions": args.repetitions,
                "equal_results": True,
                "median_ms": {
                    name: statistics.median(values) for name, values in times.items()
                },
                "p95_nearest_rank_ms": {
                    name: sorted(values)[max(0, int(0.95 * len(values) + 0.999) - 1)]
                    for name, values in times.items()
                },
                "not_end_to_end": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
