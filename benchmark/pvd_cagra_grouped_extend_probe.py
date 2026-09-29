"""Check native grouped CAGRA build+extend and per-head filtered IDs."""

from cuvs.neighbors import cagra, filters

import json
import time

import cupy as cp
import numpy as np


def sync():
    cp.cuda.get_current_stream().synchronize()


def head_filter(group_size, rows, prefix, local_head):
    total = group_size * rows
    bits = np.zeros((total + 31) // 32, dtype=np.uint32)
    locations = (
        range(local_head * prefix, (local_head + 1) * prefix),
        range(
            group_size * prefix + local_head * (rows - prefix),
            group_size * prefix + (local_head + 1) * (rows - prefix),
        ),
    )
    for section in locations:
        for vector_id in section:
            bits[vector_id // 32] |= np.uint32(1 << (vector_id % 32))
    return filters.from_bitset(cp.asarray(bits))


def main():
    cp.cuda.Device(0).use()
    rng = cp.random.RandomState(20260929)
    heads, rows, prefix, dim, top_k = 8, 1024, 512, 128, 10
    vectors = rng.standard_normal((heads, rows, dim), dtype=cp.float32)
    queries = rng.standard_normal((heads, 16, dim), dtype=cp.float32)
    exact = [
        cp.argsort(-(queries[h] @ vectors[h].T), axis=1)[:, :top_k]
        .get().tolist()
        for h in range(heads)
    ]
    params = cagra.IndexParams(
        metric="inner_product", graph_degree=8,
        intermediate_graph_degree=16, build_algo="ivf_pq",
    )
    search_params = cagra.SearchParams(itopk_size=128)
    results = []
    for group_size in (1, 2, 4, 8):
        build_seconds = extend_seconds = 0.0
        recalls, invalid = [], 0
        for start_head in range(0, heads, group_size):
            selected = vectors[start_head:start_head + group_size]
            first = cp.ascontiguousarray(selected[:, :prefix].reshape(-1, dim))
            remaining = cp.ascontiguousarray(selected[:, prefix:].reshape(-1, dim))
            started = time.perf_counter()
            index = cagra.build(params, first)
            sync()
            build_seconds += time.perf_counter() - started
            started = time.perf_counter()
            cagra.extend(cagra.ExtendParams(), index, remaining)
            sync()
            extend_seconds += time.perf_counter() - started
            for local_head in range(group_size):
                head = start_head + local_head
                _, ids = cagra.search(
                    search_params, index, queries[head], top_k,
                    filter=head_filter(group_size, rows, prefix, local_head),
                )
                sync()
                found = cp.asarray(ids).get()
                first_begin = local_head * prefix
                second_begin = group_size * prefix + local_head * (rows - prefix)
                prefix_ids = (found >= first_begin) & (found < first_begin + prefix)
                tail_ids = (found >= second_begin) & (
                    found < second_begin + rows - prefix
                )
                invalid += int(np.count_nonzero(~(prefix_ids | tail_ids)))
                token_ids = np.where(
                    prefix_ids, found - first_begin,
                    prefix + found - second_begin,
                )
                recalls.extend(
                    len(set(row) & set(wanted)) / top_k
                    for row, wanted in zip(token_ids.tolist(), exact[head])
                )
            del index
        results.append({
            "group_size": group_size, "graph_count": heads // group_size,
            "build_seconds": build_seconds, "extend_seconds": extend_seconds,
            "invalid_result_ids": invalid,
            "exact_top10_recall_mean": sum(recalls) / len(recalls),
        })
    print(json.dumps({
        "heads": heads, "rows_per_head": rows, "prefix_rows": prefix,
        "itopk_size": 128, "results": results,
    }), flush=True)


if __name__ == "__main__":
    main()
