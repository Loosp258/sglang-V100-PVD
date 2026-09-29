"""Native CAGRA evaluation of bounded Draft-Q candidate allocation on V100S.

Use the *same centered one-head graph* and native Top-16 result for uniform
and adaptive policies. Policies change the consumed prefix of that result and
the resulting per-layer/KV-head token union, not the graph or search query.
"""

from __future__ import annotations

# cuVS must load before CuPy in the isolated CloudLab environment.
from cuvs.neighbors import cagra

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import cupy as cp
import numpy as np
import torch


POLICY_NAMES = ("uniform4", "uniform8", "adaptive", "uniform16")


def synchronize():
    cp.cuda.get_current_stream().synchronize()


def mean(values):
    return statistics.mean(values)


def overlap(candidate, exact):
    return len(set(candidate) & set(exact)) / len(exact)


def exact_top4(queries, keys):
    scores = queries.reshape(-1, 128) @ keys.T
    return np.argsort(-scores, axis=1)[:, :4].reshape(2, 7, 4)


def summarize_policy(group_values):
    return {
        "mean_true_top4_coverage_student_branch": mean(
            value["student_coverage"] for value in group_values),
        "mean_true_top4_coverage_target_branch": mean(
            value["target_coverage"] for value in group_values),
        "mean_union_tokens_per_layer_kv_head": mean(
            value["union_tokens"] for value in group_values),
        "max_union_tokens_per_layer_kv_head": max(
            value["union_tokens"] for value in group_values),
        "fraction_over_128_union_cap": mean(
            value["union_tokens"] > 128 for value in group_values),
        "estimated_fresh_kv_mib": sum(
            value["union_tokens"] for value in group_values
        ) * 128 * 2 * 2 / 2**20,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--itopk-size", type=int, default=256)
    args = parser.parse_args()
    cp.cuda.Device(0).use()
    fixtures = torch.load(args.fixture, map_location="cpu", weights_only=False)
    calibration = json.loads(args.calibration.read_text())
    if args.max_records:
        fixtures = fixtures[:args.max_records]
    if (not fixtures or len(calibration["validation_exact_summary"]) < len(fixtures)
            or len(calibration["adaptive_policy"]) != 28):
        raise ValueError("fixture and calibration sizes differ")
    policy = np.asarray(calibration["adaptive_policy"], dtype=np.int32)
    if policy.shape != (28, 28):
        raise ValueError("adaptive policy must cover all 28 layers and Q heads")
    params = cagra.IndexParams(
        metric="inner_product", graph_degree=16,
        intermediate_graph_degree=32, build_algo="ivf_pq",
    )
    search_params = cagra.SearchParams(itopk_size=args.itopk_size)
    # Warm the exact graph/search shape without including first-use setup.
    warm_record = fixtures[0]
    warm_k = warm_record["prompt_k"][0, :, 0].float().numpy()
    warm_k = np.ascontiguousarray(warm_k - warm_k.mean(axis=0))
    warm_index = cagra.build(params, cp.asarray(warm_k))
    warm_q = warm_record["predicted_q"][0, :, :7].float().numpy()
    cagra.search(search_params, warm_index,
                 cp.asarray(np.ascontiguousarray(warm_q.reshape(-1, 128))), 16)
    synchronize()
    del warm_index
    gc.collect()
    synchronize()

    records_out = []
    for fixture, exact_summary in zip(
            fixtures, calibration["validation_exact_summary"]):
        if fixture["name"] != exact_summary["name"]:
            raise ValueError("validation fixture order differs")
        rows = fixture["prompt_tokens"]
        by_policy = {name: [] for name in POLICY_NAMES}
        native_true_branch, native_true_target = [], []
        build_seconds, pred_search_seconds, true_search_seconds = [], [], []
        for layer in range(28):
            for kv_head in range(4):
                first = kv_head * 7
                last = first + 7
                keys = fixture["prompt_k"][layer, :, kv_head].float().numpy()
                keys = np.ascontiguousarray(keys - keys.mean(axis=0))
                predicted = np.ascontiguousarray(
                    fixture["predicted_q"][layer, :, first:last]
                    .float().numpy().reshape(14, 128))
                student_true = np.ascontiguousarray(
                    fixture["student_branch_target_q"][layer, :, first:last]
                    .float().numpy().reshape(14, 128))
                target_true = np.ascontiguousarray(
                    fixture["target_branch_target_q"][layer, :, first:last]
                    .float().numpy().reshape(14, 128))
                branch_exact = exact_top4(student_true, keys)
                target_exact = exact_top4(target_true, keys)
                dataset = cp.asarray(keys)
                pred_cp = cp.asarray(predicted)
                true_cp = cp.asarray(student_true)
                started = time.perf_counter()
                index = cagra.build(params, dataset)
                synchronize()
                build_seconds.append(time.perf_counter() - started)
                started = time.perf_counter()
                _, found_pred = cagra.search(search_params, index, pred_cp, 16)
                synchronize()
                pred_search_seconds.append(time.perf_counter() - started)
                started = time.perf_counter()
                _, found_true = cagra.search(search_params, index, true_cp, 16)
                synchronize()
                true_search_seconds.append(time.perf_counter() - started)
                found_pred = cp.asarray(found_pred).get().reshape(2, 7, 16)
                found_true = cp.asarray(found_true).get().reshape(2, 7, 16)
                if (np.any(found_pred < 0) or np.any(found_pred >= rows)
                        or np.any(found_true < 0) or np.any(found_true >= rows)):
                    raise RuntimeError("native CAGRA returned invalid Prompt IDs")
                for position in range(2):
                    for head in range(7):
                        native_true_branch.append(overlap(
                            found_true[position, head, :4],
                            branch_exact[position, head]))
                        native_true_target.append(overlap(
                            found_true[position, head, :4],
                            target_exact[position, head]))
                for name in POLICY_NAMES:
                    selected = []
                    student_hits, target_hits = [], []
                    for position in range(2):
                        for head in range(7):
                            k = (int(policy[layer, first + head])
                                 if name == "adaptive" else
                                 int(name.removeprefix("uniform")))
                            candidates = found_pred[position, head, :k]
                            selected.extend(candidates.tolist())
                            student_hits.append(overlap(
                                candidates,
                                branch_exact[position, head]))
                            target_hits.append(overlap(
                                candidates,
                                target_exact[position, head]))
                    by_policy[name].append({
                        "student_coverage": mean(student_hits),
                        "target_coverage": mean(target_hits),
                        "union_tokens": len(set(selected)),
                    })
                del index, dataset, pred_cp, true_cp
                gc.collect()
                synchronize()
        summaries = {name: summarize_policy(by_policy[name])
                     for name in POLICY_NAMES}
        record_out = {
            "name": fixture["name"],
            "prompt_tokens": rows,
            "low_margin_fallback": exact_summary["low_margin_fallback"],
            "native_true_q_student_branch_top4_recall": mean(native_true_branch),
            "native_true_q_student_vs_target_trajectory_top4_recall": mean(
                native_true_target),
            "policies": summaries,
            "native_build_seconds": sum(build_seconds),
            "native_pred_search_ms_median_per_graph": statistics.median(
                pred_search_seconds) * 1000,
            "native_true_search_ms_median_per_graph": statistics.median(
                true_search_seconds) * 1000,
        }
        records_out.append(record_out)
        print(json.dumps({"completed_record": fixture["name"],
                          "native_branch_true_q": record_out[
                              "native_true_q_student_branch_top4_recall"],
                          "policy_student_coverage": {
                              key: value[
                                  "mean_true_top4_coverage_student_branch"]
                              for key, value in summaries.items()},
                          "adaptive_cap_overflow": summaries["adaptive"][
                              "fraction_over_128_union_cap"]}), flush=True)
    overall = {}
    for name in POLICY_NAMES:
        overall[name] = {
            "mean_true_top4_coverage_student_branch": mean(
                row["policies"][name][
                    "mean_true_top4_coverage_student_branch"]
                for row in records_out),
            "mean_true_top4_coverage_target_branch": mean(
                row["policies"][name][
                    "mean_true_top4_coverage_target_branch"]
                for row in records_out),
            "mean_fresh_kv_mib_per_request": mean(
                row["policies"][name]["estimated_fresh_kv_mib"]
                for row in records_out),
            "fraction_groups_over_128_cap": mean(
                row["policies"][name]["fraction_over_128_union_cap"]
                for row in records_out),
        }
    fallback_records = [row for row in records_out
                        if row["low_margin_fallback"]]
    mixed = mean(
        (row["native_true_q_student_branch_top4_recall"]
         if row["low_margin_fallback"] else
         row["policies"]["adaptive"][
             "mean_true_top4_coverage_student_branch"])
        for row in records_out
    )
    report = {
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "records": len(records_out),
        "queries_per_layer_kv_head": 14,
        "native_top_k_requested": 16,
        "native_itopk_size": args.itopk_size,
        "graph_degree": 16,
        "intermediate_degree": 32,
        "builder": "cuvs CAGRA IVF-PQ, separately centered K per layer/KV head",
        "overall_policies": overall,
        "native_true_q_student_branch_top4_recall": mean(
            row["native_true_q_student_branch_top4_recall"]
            for row in records_out),
        "low_margin_fallback_requests": len(fallback_records),
        "low_margin_fallback_fraction": len(fallback_records) / len(records_out),
        "adaptive_with_request_level_true_q_fallback_branch_recall": mixed,
        "per_record": records_out,
        "note": "All policy arms consume the same native Top-16 search on each centered graph; search timing is common, candidate transfer bytes are estimated from unique IDs. True-Q fallback uses the target Q on Draft tokens and omits the cost of producing it. No online D latency or answer quality.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"completed": True,
                      "overall_policies": overall,
                      "native_true_q_student_branch_top4_recall": report[
                          "native_true_q_student_branch_top4_recall"],
                      "fallback_fraction": report[
                          "low_margin_fallback_fraction"]}), flush=True)


if __name__ == "__main__":
    main()
