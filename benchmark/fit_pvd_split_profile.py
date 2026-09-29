"""Fit an N-only PVD split planner from representative online traces.

The served planner needs only Prompt length and this fixed hardware profile.
Calibration traces are consumed once here, never at request time.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


POLICY_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "python/sglang/srt/disaggregation/pvd/split_upload_policy.py"
)
spec = importlib.util.spec_from_file_location("pvd_split_upload_policy", POLICY_SOURCE)
policy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy_module)


def split_arms(source):
    if "splits" in source:
        return source["splits"].values()
    return [source["split"]]


def phase_rows(sources, rank):
    arrival, build, extend = [], [], []
    for source in sources:
        for arm in [source["full"], *split_arms(source)]:
            for item in arm:
                n = item["tokens"]
                for event in item["ranks"][str(rank)]:
                    if event["kind"] == "graph":
                        prefix = event["pages"]
                        arrival.append((prefix, event["start_seconds"]))
                        build.append((prefix, event["seconds"]))
                    elif event["first_page"] == 0:
                        # Only a full-upload graph start measures complete-KV
                        # arrival. In a split upload, final-step start is gated
                        # by first-graph completion and is only an upper bound.
                        arrival.append((n, event["start_seconds"]))
                        build.append((n, event["seconds"]))
                    else:
                        extend.append((n - event["first_page"], event["seconds"]))
    return arrival, build, extend


def fit_rank(sources, rank):
    arrival, build, extend = phase_rows(sources, rank)
    if len(arrival) < 6 or len(extend) < 6:
        raise ValueError("insufficient online split calibration samples")

    def fit(rows, powers):
        features = np.asarray([
            [(count / 1024) ** power for power in range(powers)]
            for count, _ in rows
        ], dtype=float)
        observations = np.asarray([seconds for _, seconds in rows], dtype=float)
        coefficients = np.linalg.lstsq(features, observations, rcond=None)[0]
        residuals = observations - features @ coefficients
        return coefficients.tolist(), {
            "samples": len(rows),
            "mae_seconds": float(np.mean(np.abs(residuals))),
            "max_abs_error_seconds": float(np.max(np.abs(residuals))),
        }

    coefficients, metrics = {}, {}
    for name, rows, powers in (
        ("arrival", arrival, 2), ("build", build, 3), ("extend", extend, 3)
    ):
        coefficients[name], metrics[name] = fit(rows, powers)
    return coefficients, metrics


def paired_gain_errors(profile, sources):
    errors = []
    for source in sources:
        full_by_case = {item["case"]: item for item in source["full"]}
        for arm in split_arms(source):
            for item in arm:
                full = full_by_case.get(item["case"])
                if full is None or full["tokens"] != item["tokens"]:
                    continue
                prefix = next((event["pages"] for event in item["ranks"]["0"]
                               if event["kind"] == "graph"), 0)
                if not prefix:
                    continue
                n = item["tokens"]
                plan = policy_module.plan_split(profile, n)
                predicted = next((choice["both_ready_seconds"] for choice in plan["choices"]
                                  if choice["prefix"] == prefix), None)
                if predicted is None:
                    continue
                actual_full = max(full["ranks"][str(rank)][-1]["end_seconds"]
                                  for rank in (0, 1))
                actual_split = max(item["ranks"][str(rank)][-1]["end_seconds"]
                                   for rank in (0, 1))
                predicted_gain = plan["full_ready_seconds"] - predicted
                actual_gain = actual_full - actual_split
                errors.append({
                    "n": n, "prefix": prefix, "case": item["case"],
                    "predicted_gain_seconds": predicted_gain,
                    "actual_gain_seconds": actual_gain,
                    "gain_error_seconds": predicted_gain - actual_gain,
                })
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", type=Path, nargs="+")
    parser.add_argument("--holdout", type=Path, action="append", default=[])
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prefill_chunk_tokens <= 0 or args.page_size <= 0:
        raise ValueError("positive chunk and page sizes required")
    training = [json.loads(path.read_text()) for path in args.training]
    holdout = [json.loads(path.read_text()) for path in args.holdout]
    all_n = [item["tokens"] for source in training for item in source["full"]]
    ranks, fit_metrics = [], []
    for rank in (0, 1):
        coefficients, metrics = fit_rank(training, rank)
        ranks.append(coefficients)
        fit_metrics.append(metrics)
    profile = {
        "schema": "pvd-exact16-parametric-v1",
        "config": {
            "graph_degree": 16, "group_heads": 4,
            "prefill_chunk_tokens": args.prefill_chunk_tokens,
            "page_size": args.page_size,
        },
        "domain": {
            "min_prompt_tokens": min(all_n),
            "max_prompt_tokens": max(all_n),
            "max_prefix_tokens": args.prefill_chunk_tokens
                * ((max(all_n) - 64) // args.prefill_chunk_tokens),
            "min_tail_tokens": 64,
        },
        "ranks": ranks,
        "uncertainty_seconds": 0.0,
        "training_sources_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.training
        },
        "fit_metrics": fit_metrics,
    }
    policy_module.validate_profile(profile, page_size=args.page_size)
    train_errors = paired_gain_errors(profile, training)
    holdout_errors = paired_gain_errors(profile, holdout)
    if not train_errors:
        raise ValueError("no paired graph-ready comparisons")
    # Keep the holdout out of both the coefficient fit and the decision guard.
    # A small apparent win cannot authorize a split on this noisy profile.
    profile["uncertainty_seconds"] = max(
        0.125, max(abs(row["gain_error_seconds"]) for row in train_errors) + 0.05
    )
    profile["gain_validation"] = {
        "training": train_errors, "holdout": holdout_errors,
        "training_max_abs_error_seconds": max(
            abs(row["gain_error_seconds"]) for row in train_errors
        ),
        "holdout_max_abs_error_seconds": max(
            (abs(row["gain_error_seconds"]) for row in holdout_errors),
            default=None,
        ),
    }
    args.output.write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps({
        "uncertainty_seconds": profile["uncertainty_seconds"],
        "domain": profile["domain"],
        "plans": {
            str(n): policy_module.plan_split(profile, n)
            for n in sorted(set(all_n))
        },
    }, indent=2))


if __name__ == "__main__":
    main()
