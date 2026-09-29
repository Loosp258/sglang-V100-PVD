"""Real Qwen2.5 K/Q versus native grouped-head CAGRA and an exact oracle.

Use the existing local checkpoint on one isolated V100S. This is a retrieval
quality/plumbing probe, not a three-node PVD request or a latency measurement.
"""

import os
import time
from statistics import median


def grouped_probe(sources, queries, *, prompt_count, prompt_mode, group_size):
    import cupy as cp
    import numpy as np
    import torch
    from cuvs.neighbors import cagra, filters

    keys = sorted(sources)
    if len(keys) != 8 or any(len(keys) % size for size in (1, 2, 4, 8)):
        raise AssertionError("expected four layers with two KV heads each")
    k = [cp.from_dlpack(sources[key]) for key in keys]
    q = {key: cp.from_dlpack(value) for key, value in queries.items()}
    top_k = 10
    head_queries = {
        (layer, kv_head): cp.concatenate(
            [
                q[layer, query_head]
                for query_head in sorted({head for _, head in queries})
                if query_head // group_size == kv_head
            ]
        )
        for layer, kv_head in keys
    }
    exact = {
        key: cp.argsort(-(head_queries[key] @ k[position].T), axis=1)
        [:, :top_k].get().tolist()
        for position, key in enumerate(keys)
    }
    if os.environ.get("PVD_CAGRA_DIAGNOSE") == "1":
        report = []
        for key in ((0, 0), (0, 1), (2, 0), (3, 0)):
            vectors = k[keys.index(key)]
            current_queries = head_queries[key]
            dot_products = current_queries @ vectors.T
            top_ids = cp.asarray(exact[key], dtype=cp.int64)
            precise = current_queries.astype(cp.float64) @ vectors.astype(cp.float64).T
            precise_ids = cp.argsort(-precise, axis=1)[:, :top_k]
            precise_sorted = cp.sort(precise, axis=1)[:, ::-1]
            norms = cp.linalg.norm(vectors, axis=1)
            centered = cp.ascontiguousarray(vectors - vectors.mean(axis=0))
            centered_norms = cp.linalg.norm(centered, axis=1)
            centered_ids = cp.argsort(
                -(current_queries @ centered.T), axis=1
            )[:, :top_k]
            quality = []
            for representation, candidate_vectors in (
                ("original", vectors), ("mean_centered", centered)
            ):
                candidate_scores = current_queries @ candidate_vectors.T
                for degree, intermediate in ((8, 16), (32, 64), (64, 128)):
                    params = cagra.IndexParams(
                        metric="inner_product", graph_degree=degree,
                        intermediate_graph_degree=intermediate,
                        build_algo="ivf_pq",
                    )
                    index = cagra.build(params, candidate_vectors)
                    for width in (64, 256):
                        distances, ids = cagra.search(
                            cagra.SearchParams(itopk_size=width),
                            index, current_queries, top_k,
                        )
                        cp.cuda.get_current_stream().synchronize()
                        ids = cp.asarray(ids).astype(cp.int64)
                        actual_scores = cp.take_along_axis(
                            candidate_scores, ids, axis=1
                        )
                        score_error = float(
                            cp.max(cp.abs(actual_scores - distances))
                        )
                        found = ids.get().tolist()
                        quality.append({
                            "representation": representation,
                            "graph_degree": degree,
                            "intermediate_degree": intermediate,
                            "itopk_size": width,
                            "mean_exact_top10_recall": sum(
                                len(set(row) & set(wanted)) / top_k
                                for row, wanted in zip(found, exact[key])
                            ) / len(found),
                            "score_max_abs_error": score_error,
                        })
                    del index
            report.append({
                "layer": key[0], "kv_head": key[1],
                "k_norm_median": float(cp.median(norms)),
                "k_norm_p95": float(cp.percentile(norms, 95)),
                "k_centered_norm_median": float(cp.median(centered_norms)),
                "exact_top10_k_norm_median": float(cp.median(norms[top_ids])),
                "fp32_fp64_top10_overlap": sum(
                    len(set(row) & set(wanted)) / top_k
                    for row, wanted in zip(
                        top_ids.get().tolist(), precise_ids.get().tolist()
                    )
                ) / len(top_ids),
                "original_centered_top10_overlap": sum(
                    len(set(row) & set(wanted)) / top_k
                    for row, wanted in zip(
                        top_ids.get().tolist(), centered_ids.get().tolist()
                    )
                ) / len(top_ids),
                "fp32_fp64_score_max_abs_error": float(
                    cp.max(cp.abs(dot_products.astype(cp.float64) - precise))
                ),
                "fp64_top1_top10_gap_mean": float(
                    cp.mean(precise_sorted[:, 0] - precise_sorted[:, top_k - 1])
                ),
                "fp64_top10_top11_gap_mean": float(
                    cp.mean(precise_sorted[:, top_k - 1] - precise_sorted[:, top_k])
                ),
                "quality": quality,
            })
        return {
            "model": "Qwen2.5-7B-Instruct", "prompt_tokens": prompt_count,
            "prompt_mode": prompt_mode, "diagnostics": report,
        }
    params = cagra.IndexParams(
        metric="inner_product", graph_degree=8,
        intermediate_graph_degree=16, build_algo="ivf_pq",
    )
    itopk_size = int(os.environ.get("PVD_CAGRA_GROUP_ITOPK", "64"))
    if itopk_size not in (64, 128, 256, 512, 1024, 2048):
        raise ValueError("PVD_CAGRA_GROUP_ITOPK must be 64, 128, 256, 512, 1024 or 2048")
    search_params = cagra.SearchParams(itopk_size=itopk_size)
    warm = cagra.build(params, k[0])
    cagra.search(search_params, warm, head_queries[keys[0]], top_k)
    cp.cuda.get_current_stream().synchronize()
    del warm
    cp.get_default_memory_pool().free_all_blocks()
    results = []
    for size in (1, 2, 4, 8):
        build_times, search_times, recalls = [], [], []
        invalid_ids = 0
        cases = []
        for first in range(0, len(keys), size):
            dataset = cp.ascontiguousarray(cp.concatenate(k[first:first + size]))
            started = time.perf_counter()
            index = cagra.build(params, dataset)
            cp.cuda.get_current_stream().synchronize()
            build_times.append(time.perf_counter() - started)
            for local_head in range(size):
                key = keys[first + local_head]
                bitset_filter = None
                if size > 1:
                    bitset = np.zeros((size * prompt_count + 31) // 32, dtype=np.uint32)
                    for vector_id in range(
                        local_head * prompt_count, (local_head + 1) * prompt_count
                    ):
                        bitset[vector_id // 32] |= np.uint32(1 << (vector_id % 32))
                    bitset_filter = filters.from_bitset(cp.asarray(bitset))
                started = time.perf_counter()
                _, found = cagra.search(
                    search_params, index, head_queries[key], top_k,
                    filter=bitset_filter,
                )
                cp.cuda.get_current_stream().synchronize()
                search_times.append(time.perf_counter() - started)
                found = cp.asarray(found).get()
                invalid_ids += int(np.count_nonzero(
                    (found < local_head * prompt_count)
                    | (found >= (local_head + 1) * prompt_count)
                ))
                translated = (found - local_head * prompt_count).tolist()
                recalls_for_head = [
                    len(set(row) & set(wanted)) / top_k
                    for row, wanted in zip(translated, exact[key])
                ]
                recalls.extend(recalls_for_head)
                cases.append({
                    "layer": key[0], "kv_head": key[1],
                    "recall_at_10": sum(recalls_for_head) / len(recalls_for_head),
                })
            del index, dataset
            cp.cuda.get_current_stream().synchronize()
        results.append({
            "group_size": size, "graph_count": len(build_times),
            "build_seconds": sum(build_times),
            "median_search_seconds_per_head": median(search_times),
            "recall_at_10_mean": sum(recalls) / len(recalls),
            "recall_at_10_min_query": min(recalls),
            "invalid_result_ids": invalid_ids,
            "cases": cases,
        })
    torch.cuda.synchronize()
    return {
        "model": "Qwen2.5-7B-Instruct", "prompt_tokens": prompt_count,
        "prompt_mode": prompt_mode, "layers": [0, 1, 2, 3],
        "kv_heads_per_layer": sorted({head for _, head in keys}),
        "query_heads_per_layer": sorted({head for _, head in queries}),
        "top_k": top_k, "itopk_size": itopk_size, "results": results,
    }


def full_shard_centered_probe(
    sources, queries, *, prompt_count, prompt_mode, q_heads_per_kv
):
    """Compare grouped graphs and optional exact KNN seeding on real K/Q."""
    import gc

    import cupy as cp
    import numpy as np
    from cuvs.neighbors import cagra, filters

    keys = sorted(sources)
    if len(keys) != 56 or len({layer for layer, _ in keys}) != 28:
        raise AssertionError("expected one V shard of 28 layers x two KV heads")
    rows, dim, top_k = prompt_count, 128, 10
    q = {key: cp.from_dlpack(value) for key, value in queries.items()}
    head_queries = {
        (layer, kv_head): cp.concatenate(
            [
                q[layer, query_head]
                for query_head in sorted({head for _, head in queries})
                if query_head // q_heads_per_kv == kv_head
            ]
        )
        for layer, kv_head in keys
    }
    raw_k = [cp.from_dlpack(sources[key]) for key in keys]
    prefix = int(os.environ.get("PVD_CAGRA_GROUP_PREFIX", "0"))
    if prefix and not 256 <= prefix < rows:
        raise ValueError("PVD_CAGRA_GROUP_PREFIX must be in [256, rows)")
    chunk_rows = int(os.environ.get("PVD_CAGRA_GROUP_CHUNK_ROWS", "0"))
    if chunk_rows and (not prefix or chunk_rows != prefix):
        raise ValueError("PVD_CAGRA_GROUP_CHUNK_ROWS requires an equal prefix")
    boundaries = [0, prefix] if prefix else [0, rows]
    if prefix:
        step = chunk_rows or rows - prefix
        boundaries.extend(range(prefix + step, rows, step))
        boundaries.append(rows)
    requested_boundaries = os.environ.get("PVD_CAGRA_GROUP_BOUNDARIES")
    if requested_boundaries:
        requested = [int(value) for value in requested_boundaries.split(",")]
        if (
            len(requested) < 3
            or requested[0] != 0
            or requested[1] != prefix
            or requested[-1] != rows
            or any(a >= b for a, b in zip(requested, requested[1:]))
        ):
            raise ValueError("PVD_CAGRA_GROUP_BOUNDARIES must span 0, prefix, rows")
        boundaries = requested
    exact = {
        key: cp.argsort(
            -(head_queries[key].astype(cp.float64)
              @ raw_k[position].astype(cp.float64).T),
            axis=1,
        )[:, :top_k].get().tolist()
        for position, key in enumerate(keys)
    }
    started = time.perf_counter()
    means = [vectors[:prefix].mean(axis=0) for vectors in raw_k] if prefix else [
        vectors.mean(axis=0) for vectors in raw_k
    ]
    centered_k = [
        cp.ascontiguousarray(vectors - mean)
        for vectors, mean in zip(raw_k, means)
    ]
    cp.cuda.get_current_stream().synchronize()
    centering_seconds = time.perf_counter() - started
    centered_exact_overlaps = []
    for position, key in enumerate(keys):
        centered_exact = cp.argsort(
            -(head_queries[key].astype(cp.float64)
              @ centered_k[position].astype(cp.float64).T),
            axis=1,
        )[:, :top_k].get().tolist()
        centered_exact_overlaps.extend(
            len(set(a) & set(b)) / top_k
            for a, b in zip(exact[key], centered_exact)
        )

    itopk_size = int(os.environ.get("PVD_CAGRA_GROUP_ITOPK", "128"))
    if itopk_size not in (64, 128, 256, 512, 1024, 2048):
        raise ValueError("PVD_CAGRA_GROUP_ITOPK must be 64, 128, 256, 512, 1024 or 2048")
    build_algo = os.environ.get("PVD_CAGRA_GROUP_BUILD_ALGO", "ivf_pq")
    if build_algo not in (
        "ivf_pq", "nn_descent", "iterative_cagra_search",
        "exact_block_knn", "exact_global_knn",
    ):
        raise ValueError("unsupported CAGRA build algorithm")
    exact_degree = int(os.environ.get("PVD_CAGRA_EXACT_GRAPH_DEGREE", "8"))
    if exact_degree not in (8, 16, 32):
        raise ValueError("exact CAGRA graph degree must be 8, 16 or 32")
    params = None if build_algo.startswith("exact_") else cagra.IndexParams(
        metric="inner_product", graph_degree=8,
        intermediate_graph_degree=16, build_algo=build_algo,
    )

    def build_graph(dataset, group_size):
        if params is not None:
            return cagra.build(params, dataset), None
        # For a 512-row KV-head prefix, exact candidate edges need only a
        # small GEMM. Keep the CuPy graph alive with its CAGRA index.
        if len(dataset) % group_size:
            raise ValueError("each grouped KV head must have the same row count")
        per_head = len(dataset) // group_size
        if per_head <= exact_degree:
            raise ValueError("exact graph degree must be below rows per KV head")
        graph = cp.empty((len(dataset), exact_degree), dtype=cp.uint32)
        blocks = (
            [(0, len(dataset))]
            if build_algo == "exact_global_knn" else
            [(head * per_head, (head + 1) * per_head)
             for head in range(group_size)]
        )
        for begin, end in blocks:
            matrix = dataset[begin:end]
            scores = matrix @ matrix.T
            cp.fill_diagonal(scores, -cp.inf)
            neighbors = cp.argpartition(
                scores, -exact_degree, axis=1
            )[:, -exact_degree:]
            top_scores = cp.take_along_axis(scores, neighbors, axis=1)
            neighbors = cp.take_along_axis(
                neighbors, cp.argsort(-top_scores, axis=1), axis=1
            )
            graph[begin:end] = neighbors.astype(cp.uint32) + begin
        return cagra.from_graph(
            graph, dataset, metric="inner_product"
        ), graph

    search_params = cagra.SearchParams(itopk_size=itopk_size)
    warm, warm_graph = build_graph(
        centered_k[0][:prefix] if prefix else centered_k[0], 1
    )
    cagra.search(search_params, warm, head_queries[keys[0]], top_k)
    cp.cuda.get_current_stream().synchronize()
    del warm, warm_graph
    gc.collect()
    requested_sizes = os.environ.get("PVD_CAGRA_GROUP_SIZES")
    group_sizes = (
        [int(value) for value in requested_sizes.split(",")]
        if requested_sizes else [1, 2, 4, 8, 56]
    )
    if not group_sizes or any(size < 1 or len(keys) % size for size in group_sizes):
        raise ValueError("PVD_CAGRA_GROUP_SIZES must divide 56")
    if os.environ.get("PVD_CAGRA_GROUP_REVERSE") == "1":
        group_sizes.reverse()
    results = []
    for size in group_sizes:
        build_seconds = extend_seconds = assembly_seconds = filter_seconds = 0.0
        extend_by_chunk = [0.0] * (len(boundaries) - 2)
        search_seconds, head_recalls, head_results = [], [], []
        invalid_ids = 0
        for first in range(0, len(keys), size):
            started = time.perf_counter()
            selected = centered_k[first:first + size]
            dataset = cp.ascontiguousarray(cp.concatenate(
                [value[boundaries[0]:boundaries[1]] for value in selected]
            ))
            cp.cuda.get_current_stream().synchronize()
            assembly_seconds += time.perf_counter() - started
            started = time.perf_counter()
            index, graph = build_graph(dataset, size)
            cp.cuda.get_current_stream().synchronize()
            build_seconds += time.perf_counter() - started
            tail_datasets = []
            for chunk_number, (begin, end) in enumerate(
                zip(boundaries[1:-1], boundaries[2:])
            ):
                tail_dataset = cp.ascontiguousarray(cp.concatenate(
                    [value[begin:end] for value in selected]
                ))
                tail_datasets.append(tail_dataset)
                started = time.perf_counter()
                cagra.extend(cagra.ExtendParams(), index, tail_dataset)
                cp.cuda.get_current_stream().synchronize()
                elapsed = time.perf_counter() - started
                extend_seconds += elapsed
                extend_by_chunk[chunk_number] += elapsed
            for local_head in range(size):
                key = keys[first + local_head]
                prefilter = None
                if size > 1:
                    started = time.perf_counter()
                    bitset = np.zeros((size * rows + 31) // 32, dtype=np.uint32)
                    sections = tuple(
                        (
                            size * begin + local_head * (end - begin),
                            size * begin + (local_head + 1) * (end - begin),
                        )
                        for begin, end in zip(boundaries[:-1], boundaries[1:])
                    )
                    for begin, end in sections:
                        for vector_id in range(begin, end):
                            bitset[vector_id // 32] |= np.uint32(1 << (vector_id % 32))
                    prefilter = filters.from_bitset(cp.asarray(bitset))
                    cp.cuda.get_current_stream().synchronize()
                    filter_seconds += time.perf_counter() - started
                started = time.perf_counter()
                _, found = cagra.search(
                    search_params, index, head_queries[key], top_k,
                    filter=prefilter,
                )
                cp.cuda.get_current_stream().synchronize()
                search_seconds.append(time.perf_counter() - started)
                found = cp.asarray(found).get().astype(np.int64)
                valid = np.zeros(found.shape, dtype=bool)
                translated = np.full(found.shape, -1, dtype=np.int64)
                for begin, end in zip(boundaries[:-1], boundaries[1:]):
                    group_begin = size * begin + local_head * (end - begin)
                    in_chunk = (found >= group_begin) & (
                        found < group_begin + end - begin
                    )
                    valid |= in_chunk
                    translated[in_chunk] = begin + found[in_chunk] - group_begin
                head_invalid = int(np.count_nonzero(~valid))
                translated = translated.tolist()
                invalid_ids += head_invalid
                recall = sum(
                    len(set(row) & set(wanted)) / top_k
                    for row, wanted in zip(translated, exact[key])
                ) / len(translated)
                head_recalls.append(recall)
                head_results.append({
                    "layer": key[0], "kv_head": key[1],
                    "recall_at_10": recall, "invalid_result_ids": head_invalid,
                })
            del index, graph, dataset, tail_datasets
            gc.collect()
            cp.cuda.get_current_stream().synchronize()
        results.append({
            "group_size": size, "graph_count": len(keys) // size,
            "assembly_seconds": assembly_seconds,
            "build_seconds": build_seconds,
            "extend_seconds": extend_seconds,
            "extend_seconds_by_chunk": extend_by_chunk,
            "filter_setup_seconds": filter_seconds,
            "median_search_seconds_per_head": median(search_seconds),
            "mean_recall_at_10": sum(head_recalls) / len(head_recalls),
            "min_head_recall_at_10": min(head_recalls),
            "invalid_result_ids": invalid_ids,
            "heads": head_results,
        })
    return {
        "model": "Qwen2.5-7B-Instruct", "prompt_tokens": prompt_count,
        "prompt_mode": prompt_mode, "local_heads": len(keys),
        "prefix_rows": prefix,
        "chunk_boundaries": boundaries,
        "queries_per_head": 2, "itopk_size": itopk_size,
        "build_algo": build_algo,
        "exact_graph_degree": exact_degree if build_algo.startswith("exact_") else None,
        "centered_exact_top10_overlap_mean": sum(centered_exact_overlaps)
        / len(centered_exact_overlaps),
        "centered_exact_top10_overlap_min": min(centered_exact_overlaps),
        "centering_seconds": centering_seconds,
        "group_order": group_sizes, "results": results,
    }


def _prompt_tokens(model_path, count):
    mode = os.environ.get("PVD_CAGRA_RECALL_PROMPT", "language")
    if mode == "random":
        import torch

        generator = torch.Generator().manual_seed(20260924)
        return mode, tuple(
            torch.randint(3, 1000, (count,), generator=generator).tolist()
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if mode == "online":
        trial = int(os.environ.get("PVD_CAGRA_ONLINE_TRIAL", "0"))
        prompt = f"Case {trial}. " + "EEFTRITON " * 430
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        if len(encoded) != count:
            raise AssertionError(
                f"online Prompt has {len(encoded)} tokens, expected {count}"
            )
        return mode, tuple(encoded)
    if mode != "language":
        raise ValueError("PVD_CAGRA_RECALL_PROMPT must be language, online or random")
    passages = (
        (
            "A router selects a prefill worker, a vector worker, "
            "and a decode worker for each request."
        ),
        (
            "The prefill worker computes prompt keys and values "
            "before transferring them to the vector store."
        ),
        "Each shard owns a bounded set of heads and maintains its own index lifecycle.",
        (
            "The decode worker issues a retrieval query before a refresh boundary "
            "and waits if the result is late."
        ),
        (
            "A transport descriptor identifies the destination region, "
            "generation, and permitted byte range."
        ),
        (
            "The experiment compares approximate graph search "
            "with an exact dot-product oracle."
        ),
        (
            "A patient researcher records latency, recall, output quality, "
            "and memory pressure separately."
        ),
        (
            "Natural-language questions can include dates, measurements, "
            "algorithms, and unrelated facts."
        ),
    )
    text = " ".join(
        f"Section {section}: {passages[section % len(passages)]}"
        for section in range(count)
    )
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) < count:
        raise AssertionError(
            "natural-language fixture is shorter than the requested prompt"
        )
    return mode, tuple(encoded[:count])


def latency_grid_probe(sources, *, prompt_count, prompt_mode):
    """Measure full and one-extend exact degree-16 graphs on real model K."""
    import gc

    import cupy as cp

    from cuvs.neighbors import cagra

    keys = sorted(sources)
    if len(keys) != 56:
        raise ValueError("latency grid requires all 56 local KV heads")
    degree = 16
    specs = []
    for item in os.environ["PVD_CAGRA_LATENCY_GRID"].split(","):
        n, prefix = map(int, item.split(":"))
        if not 256 <= n <= prompt_count or (prefix and not 256 <= prefix < n):
            raise ValueError(f"invalid latency grid point {item}")
        specs.append((n, prefix))
    repeats = int(os.environ.get("PVD_CAGRA_LATENCY_REPEATS", "2"))
    if not 1 <= repeats <= 5:
        raise ValueError("latency grid repeats must be in [1, 5]")
    raw = [cp.from_dlpack(sources[key]) for key in keys]
    stream = cp.cuda.get_current_stream()

    def seed(dataset, rows):
        graph = cp.empty((4 * rows, degree), dtype=cp.uint32)
        for head in range(4):
            begin, end = head * rows, (head + 1) * rows
            matrix = dataset[begin:end]
            scores = matrix @ matrix.T
            cp.fill_diagonal(scores, -cp.inf)
            neighbors = cp.argpartition(scores, -degree, axis=1)[:, -degree:]
            top_scores = cp.take_along_axis(scores, neighbors, axis=1)
            neighbors = cp.take_along_axis(
                neighbors, cp.argsort(-top_scores, axis=1), axis=1
            )
            graph[begin:end] = neighbors.astype(cp.uint32) + begin
        return cagra.from_graph(graph, dataset, metric="inner_product"), graph

    # CUDA library initialization belongs outside the measured grid.
    warm = cp.arange(4 * 256 * 128, dtype=cp.float32).reshape(4 * 256, 128)
    warm_index, warm_graph = seed(warm, 256)
    cagra.extend(cagra.ExtendParams(), warm_index, warm[:4 * 64].copy())
    stream.synchronize()
    del warm_index, warm_graph, warm
    gc.collect()

    results = []
    for repeat in range(repeats):
        ordered = specs if repeat % 2 == 0 else list(reversed(specs))
        for n, prefix in ordered:
            first_rows = prefix or n
            build_seconds = extend_seconds = assembly_seconds = 0.0
            for first in range(0, len(raw), 4):
                selected = raw[first:first + 4]
                started = time.perf_counter()
                means = [value[:first_rows].mean(axis=0) for value in selected]
                initial = cp.ascontiguousarray(cp.concatenate([
                    value[:first_rows] - mean
                    for value, mean in zip(selected, means)
                ]))
                tail = None
                if prefix:
                    tail = cp.ascontiguousarray(cp.concatenate([
                        value[prefix:n] - mean
                        for value, mean in zip(selected, means)
                    ]))
                stream.synchronize()
                assembly_seconds += time.perf_counter() - started
                started = time.perf_counter()
                index, graph = seed(initial, first_rows)
                stream.synchronize()
                build_seconds += time.perf_counter() - started
                if tail is not None:
                    started = time.perf_counter()
                    cagra.extend(cagra.ExtendParams(), index, tail)
                    stream.synchronize()
                    extend_seconds += time.perf_counter() - started
                del index, graph, initial, tail
                gc.collect()
                stream.synchronize()
            results.append({
                "n": n, "prefix": prefix, "tail": n - prefix if prefix else 0,
                "repeat": repeat, "assembly_seconds": assembly_seconds,
                "build_seconds": build_seconds,
                "extend_seconds": extend_seconds,
            })
    return {
        "model": "Qwen2.5-7B-Instruct", "prompt_mode": prompt_mode,
        "prompt_tokens": prompt_count, "kv_heads": len(keys),
        "kv_head_ids": sorted({head for _, head in keys}),
        "group_size": 4, "graph_count": 14, "degree": degree,
        "build_algorithm": "exact_per_head_knn_from_graph",
        "extend_algorithm": "cuvs_cagra_native_extend",
        "measurements": results,
    }


def validate(runner, *, checkpoint=False):
    if not checkpoint:
        raise ValueError("a real Qwen2.5 checkpoint is required")

    import torch
    from sglang.srt.disaggregation.pvd.draft_forward_adapter import (
        DraftForwardAdapter,
        PrivatePoolAllocator,
    )
    from sglang.srt.disaggregation.pvd.draft_runner_sglang import DraftForwardInputs

    if type(runner.model).__name__ != "Qwen2ForCausalLM":
        raise ValueError("real Qwen2 model required")
    device = torch.device("cuda:0")
    prompt_count = int(os.environ.get("PVD_CAGRA_RECALL_ROWS", "1024"))
    max_rows = 4096 if os.environ.get("PVD_CAGRA_LATENCY_GRID") else 2304
    if not 256 <= prompt_count <= max_rows:
        raise ValueError(f"PVD_CAGRA_RECALL_ROWS must be in [256, {max_rows}]")
    top_k = 10
    full_shard = os.environ.get("PVD_CAGRA_GROUP_FULL_SHARD") == "1"
    layers = (
        tuple(range(runner.model.config.num_hidden_layers))
        if full_shard else (0, 1, 2, 3)
    )
    group_size = (
        runner.model.config.num_attention_heads
        // runner.model.config.num_key_value_heads
    )
    if group_size != 7:
        raise ValueError("this Qwen2.5-7B acceptance expects seven Q heads per KV head")
    kv_head_start = int(os.environ.get("PVD_CAGRA_KV_HEAD_START", "0"))
    if kv_head_start not in (0, 2):
        raise ValueError("PVD_CAGRA_KV_HEAD_START must be 0 or 2")
    kv_heads = (kv_head_start, kv_head_start + 1)
    query_heads = tuple(
        head * group_size + offset for head in kv_heads for offset in (0, 1)
    )
    if runner.model.config.num_attention_heads <= max(query_heads):
        raise ValueError("model has too few query heads for this V rank")
    prompt_mode, tokens = _prompt_tokens(runner.model_config.model_path, prompt_count)
    allocator = PrivatePoolAllocator(
        runner.req_to_token_pool, runner.token_to_kv_pool_allocator
    )
    slot, rows = allocator.alloc_request(), []
    sources = {}
    prefill_chunk_rows = int(os.environ.get("PVD_CAGRA_PREFILL_CHUNK_ROWS", "0"))
    if prefill_chunk_rows and not 256 <= prefill_chunk_rows <= prompt_count:
        raise ValueError("PVD_CAGRA_PREFILL_CHUNK_ROWS is outside the Prompt")
    prefill_boundaries = list(range(0, prompt_count, prefill_chunk_rows or prompt_count))
    prefill_boundaries.append(prompt_count)
    chunks = {(layer, head): [] for layer in layers for head in kv_heads}
    try:
        rows = allocator.alloc_kv(prompt_count)
        allocator.write_mapping(slot, 0, rows)
        adapter = DraftForwardAdapter(
            runner,
            architecture="Qwen2ForCausalLM",
            attention_backend="torch_native",
            bytes_per_token=(
                runner.model.config.num_hidden_layers
                * runner.model.config.num_key_value_heads
                * runner.model_config.head_dim
                * 2
                * 2
            ),
            device=device,
        )
        for begin, end in zip(prefill_boundaries[:-1], prefill_boundaries[1:]):
            logits = adapter.forward(
                DraftForwardInputs(
                    "extend",
                    tokens[begin:end],
                    tuple(range(begin, end)),
                    (end,),
                    (slot,),
                    tuple(rows[begin:end]),
                    (begin,),
                    (end - begin,),
                )
            )
            torch.cuda.synchronize(device)
            for layer in layers:
                for kv_head in kv_heads:
                    chunks[layer, kv_head].append(
                        runner.token_to_kv_pool.get_key_buffer(layer)[
                            rows[begin:end], kv_head
                        ].clone().float().contiguous()
                    )
        next_token = int(logits.argmax().item())
        sources = {
            key: torch.cat(value).contiguous() for key, value in chunks.items()
        }
    finally:
        if rows:
            torch.cuda.synchronize(device)
            allocator.clear_mapping(slot)
            allocator.free_kv(rows)
        allocator.free_request(slot)
    if any(
        source.shape != (prompt_count, runner.model_config.head_dim)
        for source in sources.values()
    ):
        raise AssertionError("target Prompt K extraction has an unexpected shape")
    if os.environ.get("PVD_CAGRA_LATENCY_GRID"):
        if not full_shard:
            raise ValueError("latency grid requires PVD_CAGRA_GROUP_FULL_SHARD=1")
        return latency_grid_probe(
            sources, prompt_count=prompt_count, prompt_mode=prompt_mode
        )

    # Observe the actual attention input at the next-token position. The
    # attention module receives Q after the model applies RoPE; this avoids
    # reimplementing the model's Q projection or positional transform.
    captured = {}
    hooks = []

    def capture_q(module, args, *, layer):
        captured[layer] = (
            args[0][-1]
            .reshape(
                runner.model.config.num_attention_heads, runner.model_config.head_dim
            )
            .detach()
            .clone()
            .float()
        )

    slot, rows = allocator.alloc_request(), []
    try:
        for layer in layers:
            hooks.append(
                runner.model.model.layers[
                    layer
                ].self_attn.attn.register_forward_pre_hook(
                    lambda module, args, layer=layer: capture_q(
                        module, args, layer=layer
                    )
                )
            )
        full_tokens = tokens + (next_token,)
        rows = allocator.alloc_kv(len(full_tokens))
        allocator.write_mapping(slot, 0, rows)
        adapter.forward(
            DraftForwardInputs(
                "extend",
                full_tokens,
                tuple(range(len(full_tokens))),
                (len(full_tokens),),
                (slot,),
                tuple(rows),
                (0,),
                (len(full_tokens),),
            )
        )
        torch.cuda.synchronize(device)
    finally:
        for hook in hooks:
            hook.remove()
        if rows:
            torch.cuda.synchronize(device)
            allocator.clear_mapping(slot)
            allocator.free_kv(rows)
        allocator.free_request(slot)
    if set(captured) != set(layers):
        raise AssertionError("real target forward did not expose every Q layer")
    queries = {}
    for layer, query_matrix in captured.items():
        if query_matrix.shape != (
            runner.model.config.num_attention_heads,
            runner.model_config.head_dim,
        ):
            raise AssertionError("post-RoPE target Q has unexpected shape")
        for query_head in query_heads:
            queries[layer, query_head] = (
                query_matrix[query_head].reshape(1, -1).contiguous()
            )

    if full_shard:
        result = full_shard_centered_probe(
            sources, queries, prompt_count=prompt_count,
            prompt_mode=prompt_mode, q_heads_per_kv=group_size,
        )
    else:
        result = grouped_probe(
            sources, queries, prompt_count=prompt_count,
            prompt_mode=prompt_mode, group_size=group_size,
        )
    result["prefill_chunk_boundaries"] = prefill_boundaries
    result["kv_heads"] = kv_heads
    return result



def main(argv=None):
    # Import cuVS before SGLang's package initializer imports torch in the
    # pinned CloudLab environment.
    import cuvs
    from run_pvd_cuda_probe_smoke import main as run_model
    from sglang.srt.model_executor.model_runner import ModelRunner

    required_version = "25.10.00"
    if cuvs.__version__ != required_version:
        raise RuntimeError(
            f"this V100S acceptance gate requires cuVS {required_version}"
        )
    # This standalone probe uses torch_native attention. Its base backend has
    # no CUDA-graph fill value, while ModelRunner's optional prefill kernel
    # warmup asks for one even with CUDA graph disabled. The measured forward
    # and retrieval kernels still execute normally in validate().
    ModelRunner.kernel_warmup = lambda self: None
    return run_model(
        argv,
        validator=validate,
        schema="pvd-qwen2.5-real-q-grouped-cagra-v1",
    )


if __name__ == "__main__":
    raise SystemExit(main())
