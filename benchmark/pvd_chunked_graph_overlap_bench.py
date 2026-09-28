"""Measure the CAGRA cost floor for overlapping graph builds with KV arrival.

This is a V-side native build experiment. Each arm receives the same K and
uses the same backend. Arrival gaps are modeled from measured build durations;
no P prefill, RDMA traffic or serving protocol is emulated here.
"""

# ruff: noqa: I001 -- The CloudLab cuVS wheel must load before Torch.
from __future__ import annotations

import argparse
import json
import statistics
import time

from cuvs.neighbors import cagra as _cagra  # noqa: F401

import torch

from sglang.srt.disaggregation.pvd.cagra_backend import CagraIndexBackend


def build_heads(backend, vectors, start, end):
    indexes = []
    begun = time.perf_counter()
    try:
        for matrix in vectors:
            indexes.append(
                backend.build(
                    matrix[start:end].contiguous(),
                    vector_space="chunk-arrival-test",
                    metric="ip",
                )
            )
        backend.synchronize()
        return indexes, time.perf_counter() - begun
    except BaseException:
        backend.synchronize()
        for index in indexes:
            backend.dispose(index)
        raise


def dispose_heads(backend, indexes):
    backend.synchronize()
    for index in indexes:
        backend.dispose(index)


def modeled_ready(full_seconds, first_seconds, rest_seconds, arrival_gap):
    baseline = arrival_gap + full_seconds
    chunked = max(arrival_gap, first_seconds) + rest_seconds
    return {
        "gap_seconds": arrival_gap,
        "baseline_ready_seconds": baseline,
        "chunked_ready_seconds": chunked,
        "saved_seconds": baseline - chunked,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--rows", type=int, default=2304)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--first-rows", type=int, default=2048)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--gaps", type=float, nargs="+", default=[0, 1, 2, 5, 10])
    args = parser.parse_args()
    if min(args.heads, args.dim, args.repetitions) < 1:
        parser.error("heads, dim and repetitions must be positive")
    if not 16 < args.first_rows < args.rows - 16:
        parser.error("both chunks must have more than 16 rows for native CAGRA")
    if any(gap < 0 for gap in args.gaps):
        parser.error("arrival gaps must be non-negative")

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
    # Own one immutable GPU buffer per head and share those exact bytes across
    # all arms. Input allocation and random generation stay outside timing.
    vectors = [
        torch.randn(
            args.rows,
            args.dim,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        for _ in range(args.heads)
    ]
    backend.synchronize()
    warm, _ = build_heads(backend, vectors[:1], 0, args.rows)
    dispose_heads(backend, warm)
    print("warmup_complete", flush=True)

    observations = {"full": [], "chunked_first": [], "chunked_rest": []}
    order = ["full", "chunked", "chunked", "full"] * args.repetitions
    for arm in order:
        first = rest = full = None
        try:
            if arm == "full":
                full, duration = build_heads(backend, vectors, 0, args.rows)
                observations["full"].append(duration)
            else:
                first, first_duration = build_heads(
                    backend, vectors, 0, args.first_rows
                )
                rest, rest_duration = build_heads(
                    backend, vectors, args.first_rows, args.rows
                )
                observations["chunked_first"].append(first_duration)
                observations["chunked_rest"].append(rest_duration)
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "first_seconds": first_duration if arm == "chunked" else None,
                        "rest_seconds": rest_duration if arm == "chunked" else None,
                        "full_seconds": duration if arm == "full" else None,
                        "native_retained_bytes": backend.runtime.global_limit.get_allocated_bytes(),
                    }
                ),
                flush=True,
            )
        finally:
            for indexes in (rest, first, full):
                if indexes is not None:
                    dispose_heads(backend, indexes)
        if backend.runtime.global_limit.get_allocated_bytes() != 0:
            raise RuntimeError("native CAGRA allocations survived an arm")

    medians = {key: statistics.median(values) for key, values in observations.items()}
    full_seconds = medians["full"]
    first_seconds = medians["chunked_first"]
    rest_seconds = medians["chunked_rest"]
    result = {
        "shape": vars(args),
        "order": order,
        "observations": observations,
        "medians": medians,
        "minimum_gap_for_positive_saving_seconds": max(
            0.0, first_seconds + rest_seconds - full_seconds
        ),
        "scenarios": [
            modeled_ready(full_seconds, first_seconds, rest_seconds, gap)
            for gap in args.gaps
        ],
        "native_retained_bytes_final": backend.runtime.global_limit.get_allocated_bytes(),
    }
    print("SUMMARY " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
