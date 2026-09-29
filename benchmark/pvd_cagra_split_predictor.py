"""Fit and use a bounded graph-latency model for the exact-16 PVD probe.

The optimizer needs an arrival curve from the P/V transport. GPU timings alone
cannot predict whether a prefix build overlaps the remaining Prefill/upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _features(kind: str, n: int, prefix: int) -> list[float]:
    p = (prefix or n) / 1024
    if kind == "build":
        return [1.0, p, p * p]
    m = (n - prefix) / 1024
    return [1.0, p, m, p * m, m * m, p * p]


def fit(rows: list[dict]) -> dict:
    """Least squares on repeated measurements; do not extrapolate at predict."""
    if not rows or not any(row["prefix"] for row in rows):
        raise ValueError("need complete-build and split measurements")
    result = {"schema": "pvd-exact16-split-latency-v1"}
    build_rows = [row for row in rows if row["prefix"] == 0]
    # Split measurements give additional observations at the same build size.
    build_rows += [row for row in rows if row["prefix"]]
    extend_rows = [row for row in rows if row["prefix"]]
    for kind, subset, field in (
        ("build", build_rows, "build_seconds"),
        ("extend", extend_rows, "extend_seconds"),
    ):
        x = np.asarray([
            _features(kind, row["n"], row["prefix"]) for row in subset
        ], dtype=float)
        y = np.asarray([row[field] for row in subset], dtype=float)
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        result[kind] = {
            "coefficients": coefficients.tolist(),
            "training_points": len(subset),
            "train_mae_ms": float(np.mean(np.abs(x @ coefficients - y)) * 1000),
            "train_p95_abs_error_ms": float(
                np.percentile(np.abs(x @ coefficients - y), 95) * 1000
            ),
        }
    result["domain"] = {
        "min_n": min(row["n"] for row in rows),
        "max_n": max(row["n"] for row in rows),
        "min_prefix": min(row["prefix"] for row in extend_rows),
        "max_prefix": max(row["prefix"] for row in extend_rows),
        "min_tail": min(row["n"] - row["prefix"] for row in extend_rows),
        "max_tail": max(row["n"] - row["prefix"] for row in extend_rows),
    }
    return result


def predict(model: dict, n: int, prefix: int) -> dict:
    """Return graph times in seconds within the measured input rectangle."""
    domain = model["domain"]
    if not domain["min_n"] <= n <= domain["max_n"]:
        raise ValueError("Prompt length is outside the measured range")
    if prefix:
        tail = n - prefix
        if not (
            domain["min_prefix"] <= prefix <= domain["max_prefix"]
            and domain["min_tail"] <= tail <= domain["max_tail"]
        ):
            raise ValueError("prefix or tail is outside the measured range")
    build = float(np.dot(
        _features("build", n, prefix), model["build"]["coefficients"]
    ))
    extend = float(np.dot(
        _features("extend", n, prefix), model["extend"]["coefficients"]
    )) if prefix else 0.0
    return {"build_seconds": max(0.0, build), "extend_seconds": max(0.0, extend)}


def optimize(model: dict, n: int, head_start_by_prefix: dict[int, float]) -> list[dict]:
    """Rank graph-ready offsets from full-KV arrival by measured head starts.

    head_start_by_prefix[p] is a nonnegative number of seconds by which p rows
    arrive at V before all n rows. Every candidate uses the same full-arrival
    reference; choose a two-rank split by taking max of rank offsets.
    """
    choices = [{"prefix": 0, "graph_ready_after_full_seconds":
                predict(model, n, 0)["build_seconds"]}]
    for prefix, head_start in head_start_by_prefix.items():
        if head_start < 0:
            raise ValueError("head start must be nonnegative")
        times = predict(model, n, prefix)
        choices.append({
            "prefix": prefix,
            "graph_ready_after_full_seconds": max(
                0.0, times["build_seconds"] - head_start
            ) + times["extend_seconds"],
            **times,
        })
    return sorted(choices, key=lambda item: item["graph_ready_after_full_seconds"])


def optimize_two_rank(
    models: list[dict], n: int, arrivals: list[dict], uncertainty_ms: float = 0.0
) -> dict:
    """Choose a common split using each rank's actual or estimated arrival times.

    Arrival timestamps use one common origin in seconds. For each rank:
    baseline_full_seconds is complete-KV arrival without splitting;
    split_full_seconds is complete-KV arrival with split upload;
    prefix_seconds maps prefix length to that prefix's arrival at V.
    """
    if len(models) != 2 or len(arrivals) != 2:
        raise ValueError("two models and two arrival traces are required")
    if uncertainty_ms < 0:
        raise ValueError("uncertainty threshold must be nonnegative")

    def graph_times_for_rank(model: dict, trace: dict, prefix: int) -> dict:
        measured = trace.get("online_graph_seconds", {}).get(str(prefix))
        if measured is None:
            return predict(model, n, prefix)
        build = float(measured["build_seconds"])
        extend = float(measured["extend_seconds"])
        if build <= 0 or extend < 0 or (prefix == 0 and extend != 0):
            raise ValueError("invalid online graph calibration")
        return {"build_seconds": build, "extend_seconds": extend}

    try:
        full_builds = [
            graph_times_for_rank(model, trace, 0)["build_seconds"]
            for model, trace in zip(models, arrivals)
        ]
    except ValueError as exc:
        return {
            "recommendation": 0,
            "reason": f"uncalibrated Prompt length: {exc}",
            "choices": [],
        }
    prefixes = set(map(int, arrivals[0]["prefix_seconds"])) & set(
        map(int, arrivals[1]["prefix_seconds"])
    )
    baseline_rank_ready = [
        float(trace["baseline_full_seconds"]) + build
        for build, trace in zip(full_builds, arrivals)
    ]
    baseline = max(baseline_rank_ready)
    choices = []
    rejected = {}
    for prefix in sorted(prefixes):
        rank_ready = []
        graph_times = []
        try:
            graph_times = [
                graph_times_for_rank(model, trace, prefix)
                for model, trace in zip(models, arrivals)
            ]
        except ValueError as exc:
            rejected[str(prefix)] = str(exc)
            continue
        for times, trace in zip(graph_times, arrivals):
            first_arrival = float(trace["prefix_seconds"][str(prefix)])
            full_arrival = float(trace["split_full_seconds"])
            if first_arrival > full_arrival:
                raise ValueError("prefix cannot arrive after complete KV")
            rank_ready.append(max(
                first_arrival + times["build_seconds"], full_arrival
            ) + times["extend_seconds"])
        ready = max(rank_ready)
        choices.append({
            "prefix": prefix,
            "rank_ready_seconds": rank_ready,
            "both_ready_seconds": ready,
            "gain_vs_full_seconds": baseline - ready,
            "rank_graph_times": graph_times,
        })
    choices.sort(key=lambda item: item["both_ready_seconds"])
    best = choices[0] if choices else None
    safe_gain = best and best["gain_vs_full_seconds"] * 1000 > uncertainty_ms
    return {
        "baseline_rank_ready_seconds": baseline_rank_ready,
        "baseline_both_ready_seconds": baseline,
        "uncertainty_threshold_ms": uncertainty_ms,
        "recommendation": best["prefix"] if safe_gain else 0,
        "choices": choices,
        "rejected_candidates": rejected,
    }


def evaluate(model: dict, rows: list[dict]) -> dict:
    errors = {"build": [], "extend": [], "sum": []}
    for row in rows:
        forecast = predict(model, row["n"], row["prefix"])
        errors["build"].append(abs(
            forecast["build_seconds"] - row["build_seconds"]
        ) * 1000)
        if row["prefix"]:
            errors["extend"].append(abs(
                forecast["extend_seconds"] - row["extend_seconds"]
            ) * 1000)
            errors["sum"].append(abs(
                forecast["build_seconds"] + forecast["extend_seconds"]
                - row["build_seconds"] - row["extend_seconds"]
            ) * 1000)
    return {
        kind: {
            "count": len(values),
            "mae_ms": float(np.mean(values)),
            "p95_abs_error_ms": float(np.percentile(values, 95)),
            "max_abs_error_ms": float(max(values)),
        }
        for kind, values in errors.items() if values
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("measurements", type=Path, nargs="+")
    fit_parser.add_argument("--output", type=Path, required=True)
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("model", type=Path)
    predict_parser.add_argument("n", type=int)
    predict_parser.add_argument("--head-starts", type=Path)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("model", type=Path)
    evaluate_parser.add_argument("measurements", type=Path, nargs="+")
    optimize_parser = sub.add_parser("optimize")
    optimize_parser.add_argument("model", type=Path)
    optimize_parser.add_argument("n", type=int)
    optimize_parser.add_argument("arrivals", type=Path)
    optimize_parser.add_argument("--model-rank1", type=Path)
    optimize_parser.add_argument("--uncertainty-ms", type=float, default=125.0)
    policy_parser = sub.add_parser("compile-policy")
    policy_parser.add_argument("model", type=Path)
    policy_parser.add_argument("arrivals", type=Path)
    policy_parser.add_argument("--output", type=Path, required=True)
    policy_parser.add_argument("--uncertainty-ms", type=float, default=125.0)
    args = parser.parse_args()
    if args.command == "fit":
        rows = []
        for path in args.measurements:
            raw = json.loads(path.read_text())
            if isinstance(raw, dict) and (
                raw.get("degree") != 16
                or raw.get("group_size") != 4
                or raw.get("graph_count") != 14
                or raw.get("build_algorithm") != "exact_per_head_knn_from_graph"
                or raw.get("extend_algorithm") != "cuvs_cagra_native_extend"
            ):
                raise ValueError(f"incompatible measurement configuration: {path}")
            rows.extend(raw["measurements"] if isinstance(raw, dict) else raw)
        model = fit(rows)
        model["training_sources"] = [str(path) for path in args.measurements]
        args.output.write_text(json.dumps(model, indent=2) + "\n")
        print(json.dumps(model, indent=2))
    elif args.command == "evaluate":
        model = json.loads(args.model.read_text())
        summaries = {}
        combined = []
        for path in args.measurements:
            raw = json.loads(path.read_text())
            rows = raw["measurements"] if isinstance(raw, dict) else raw
            summaries[str(path)] = evaluate(model, rows)
            combined.extend(rows)
        summaries["combined"] = evaluate(model, combined)
        print(json.dumps(summaries, indent=2))
    elif args.command == "optimize":
        model = json.loads(args.model.read_text())
        other = json.loads(args.model_rank1.read_text()) if args.model_rank1 else model
        arrivals = json.loads(args.arrivals.read_text())
        print(json.dumps(optimize_two_rank(
            [model, other], args.n, arrivals["ranks"], args.uncertainty_ms
        ), indent=2))
    elif args.command == "compile-policy":
        model = json.loads(args.model.read_text())
        arrivals = json.loads(args.arrivals.read_text())
        choices = {}
        for n_text, trace in arrivals["by_prompt"].items():
            n = int(n_text)
            result = optimize_two_rank(
                [model, model], n, trace["ranks"], args.uncertainty_ms
            )
            choices[str(n)] = {
                "prefix": result["recommendation"],
                "predicted_gain_seconds": (
                    next((item["gain_vs_full_seconds"] for item in result["choices"]
                          if item["prefix"] == result["recommendation"]), 0.0)
                ),
            }
        policy = {
            "schema": "pvd-exact16-split-policy-v1",
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "arrivals_sha256": hashlib.sha256(args.arrivals.read_bytes()).hexdigest(),
            "uncertainty_ms": args.uncertainty_ms,
            "choices": choices,
        }
        args.output.write_text(json.dumps(policy, indent=2) + "\n")
        print(json.dumps(policy, indent=2))
    else:
        model = json.loads(args.model.read_text())
        if args.head_starts:
            starts = {
                int(key): float(value)
                for key, value in json.loads(args.head_starts.read_text()).items()
            }
            print(json.dumps(optimize(model, args.n, starts), indent=2))
        else:
            print(json.dumps(predict(model, args.n, 0), indent=2))


if __name__ == "__main__":
    main()
