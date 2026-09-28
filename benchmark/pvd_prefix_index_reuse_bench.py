"""Measure prefix reuse with a real exact copy builder on synthetic Prompt K.

Run: python -m benchmark.pvd_prefix_index_reuse_bench
This reports local exact-copy time and the number of index builds required
when both prefix and suffix must have an index. It does not claim to measure
cuVS CAGRA on the V100S deployment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from time import perf_counter

import torch

from benchmark.pvd_prefix_index_reuse import PrefixIndexReuseExperiment


@dataclass
class ExactIndex:
    handle: torch.Tensor
    count: int


class ExactCopyBackend:
    def build(self, vectors, *, vector_space, metric):
        if not torch.isfinite(vectors).all():
            raise ValueError("non-finite K")
        return ExactIndex(vectors.contiguous().clone(), len(vectors))

    def search(self, index, queries, *, top_k):
        scores = queries @ index.handle.T
        ordered, rows = torch.sort(scores, dim=1, descending=True, stable=True)
        return rows[:, :top_k], ordered[:, :top_k]

    def synchronize(self):
        pass

    def dispose(self, index):
        pass


def run(entries: int, heads: int, rows: int, prefix_rows: int, dim: int):
    if min(entries, heads, dim) < 1 or not 0 < prefix_rows <= rows:
        raise ValueError("invalid benchmark shape")
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20260929)
    common = torch.randn(heads, prefix_rows, dim, generator=generator)
    tails = torch.randn(entries, heads, rows - prefix_rows, dim, generator=generator)
    backend = ExactCopyBackend()

    start = perf_counter()
    for entry_no in range(entries):
        for head in range(heads):
            vectors = torch.cat((common[head], tails[entry_no, head]))
            backend.build(vectors, vector_space="benchmark", metric="ip")
    baseline_seconds = perf_counter() - start

    experiment = PrefixIndexReuseExperiment(
        backend, vector_space="benchmark", metric="ip", retain_idle=True
    )
    last_record = None
    timings = []
    start = perf_counter()
    for entry_no in range(entries):
        vectors = {
            (0, head): torch.cat((common[head], tails[entry_no, head]))
            for head in range(heads)
        }
        record, timing = experiment.build(
            vectors, prefix_rows=prefix_rows, prefix_identity="common-prefix"
        )
        if entry_no < entries - 1:
            experiment.close(record)
        else:
            last_record = record
        timings.append(timing)
    shared_seconds = perf_counter() - start

    query = torch.randn(1, dim, generator=generator)
    full = backend.build(
        torch.cat((common[0], tails[-1, 0])), vector_space="benchmark", metric="ip"
    )
    expected = backend.search(full, query, top_k=4)
    actual = experiment.search(last_record, (0, 0), query, top_k=4)
    if not torch.equal(actual[0], expected[0]):
        raise AssertionError("split search changed exact top-k rows")
    experiment.close(last_record)
    experiment.evict("common-prefix")
    return {
        "shape": {
            "entries": entries,
            "heads": heads,
            "rows": rows,
            "prefix_rows": prefix_rows,
            "dim": dim,
        },
        "baseline_exact_copy_seconds": round(baseline_seconds, 6),
        "shared_exact_copy_seconds_including_verification": round(shared_seconds, 6),
        "prefix_verification_seconds": round(
            sum(t.verify_seconds for t in timings), 6
        ),
        "baseline_total_index_builds": entries * heads,
        "shared_prefix_builds": sum(t.prefix_builds for t in timings),
        "shared_tail_builds": sum(t.tail_builds for t in timings),
        "baseline_all_graph_builds": entries * heads,
        "shared_all_graph_builds": (
            heads + (entries * heads if rows > prefix_rows else 0)
        ),
        "exact_top_k_equal": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", type=int, default=4)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--rows", type=int, default=2304)
    parser.add_argument("--prefix-rows", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
