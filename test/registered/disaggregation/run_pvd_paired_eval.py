"""Collect and compare bounded, sequential PVD output/latency observations.

The mode label is operator-declared, NOT verified by this script. Token match
is a diagnostic, not a semantic quality score. Use fixed server revisions and
record GPU/transport evidence separately.
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _dataset(path):
    items = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not item["id"].strip()
            or not isinstance(item.get("text"), str)
            or not item["text"].strip()
            or len(item["text"]) > 200_000
        ):
            raise ValueError(f"invalid dataset item at line {line_number}")
        items.append({"id": item["id"], "text": item["text"]})
        if len(items) > 1000:
            raise ValueError("dataset exceeds 1000 requests")
    if not items or len({item["id"] for item in items}) != len(items):
        raise ValueError("dataset must be non-empty with unique IDs")
    digest = hashlib.sha256(
        json.dumps(items, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return items, digest


def _request(url, text, max_new_tokens, timeout_seconds):
    body = {
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if response.status != 200:
            raise ValueError(f"Gateway returned HTTP {response.status}")
        payload = json.loads(response.read(2 * 1024 * 1024))
    elapsed = time.perf_counter() - started
    metadata = payload.get("meta_info") if isinstance(payload, dict) else None
    tokens = payload.get("output_ids") if isinstance(payload, dict) else None
    if (
        not isinstance(metadata, dict)
        or not isinstance(tokens, list)
        or len(tokens) != max_new_tokens
        or any(type(token) is not int or token < 0 for token in tokens)
        or metadata.get("completion_tokens") != max_new_tokens
    ):
        raise ValueError("Gateway did not return the requested complete token IDs")
    return {
        "output_ids": tokens,
        "prompt_tokens": metadata.get("prompt_tokens"),
        "completion_tokens": len(tokens),
        "client_elapsed_seconds": elapsed,
    }


def collect(args):
    if not 1 <= args.max_new_tokens <= 256 or not 0 < args.timeout_seconds <= 600:
        raise ValueError("max tokens must be 1..256 and timeout must be 0..600 seconds")
    if not args.gateway_url.startswith(("http://", "https://")):
        raise ValueError("explicit HTTP(S) Gateway URL required")
    if Path(args.output).exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    items, digest = _dataset(args.dataset)
    results = []
    for item in items:
        result = _request(
            args.gateway_url,
            item["text"],
            args.max_new_tokens,
            args.timeout_seconds,
        )
        results.append({"id": item["id"], **result})
        print(
            f"{args.mode}: {item['id']} {result['client_elapsed_seconds']:.3f}s",
            file=sys.stderr,
        )
    report = {
        "schema": "pvd.paired_eval.v1",
        "mode_label": args.mode,
        "mode_verified_by_script": False,
        "config_id": args.config_id,
        "dataset_sha256": digest,
        "max_new_tokens": args.max_new_tokens,
        "sequential_requests": True,
        "results": results,
    }
    with Path(args.output).open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote {len(results)} observations to {args.output}")


def _percentile(values, percentile):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def compare_reports(full, predictive):
    for report, mode in ((full, "full"), (predictive, "predictive")):
        if (
            not isinstance(report, dict)
            or report.get("schema") != "pvd.paired_eval.v1"
            or report.get("mode_label") != mode
            or report.get("mode_verified_by_script") is not False
            or report.get("sequential_requests") is not True
        ):
            raise ValueError(f"invalid or mislabeled {mode} report")
    if (
        full["dataset_sha256"] != predictive["dataset_sha256"]
        or full["max_new_tokens"] != predictive["max_new_tokens"]
    ):
        raise ValueError("dataset or generation bounds differ")
    left, right = full["results"], predictive["results"]
    if not left or [r["id"] for r in left] != [r["id"] for r in right]:
        raise ValueError("report request IDs or order differ")
    exact = 0
    prefixes = []
    for baseline, candidate in zip(left, right, strict=True):
        a, b = baseline["output_ids"], candidate["output_ids"]
        expected = full["max_new_tokens"]
        if (
            len(a) != expected
            or len(b) != expected
            or any(type(token) is not int or token < 0 for token in a + b)
        ):
            raise ValueError("invalid token IDs in report")
        exact += a == b
        prefixes.append(
            next(
                (i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y),
                expected,
            )
        )
    latencies = {}
    for report in (full, predictive):
        values = [r["client_elapsed_seconds"] for r in report["results"]]
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in values
        ):
            raise ValueError("invalid client latency in report")
        latencies[report["mode_label"]] = {
            "median_seconds": statistics.median(values),
            "p95_nearest_rank_seconds": _percentile(values, 0.95),
        }
    return {
        "requests": len(left),
        "exact_output_match_fraction": exact / len(left),
        "mean_common_prefix_tokens": statistics.mean(prefixes),
        "latency": latencies,
        "quality_caveat": "Token agreement is not semantic quality; modes are operator-declared.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    sample = subcommands.add_parser("collect")
    sample.add_argument("--gateway-url", required=True)
    sample.add_argument("--dataset", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--mode", choices=("full", "predictive"), required=True)
    sample.add_argument("--config-id", required=True)
    sample.add_argument("--max-new-tokens", type=int, default=20)
    sample.add_argument("--timeout-seconds", type=float, default=180)
    comparison = subcommands.add_parser("compare")
    comparison.add_argument("--full", required=True)
    comparison.add_argument("--predictive", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            collect(args)
        else:
            full = json.loads(Path(args.full).read_text(encoding="utf-8"))
            predictive = json.loads(Path(args.predictive).read_text(encoding="utf-8"))
            print(json.dumps(compare_reports(full, predictive), indent=2))
    except (ValueError, OSError, json.JSONDecodeError, urllib.error.URLError) as exc:
        parser.exit(1, f"paired evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
