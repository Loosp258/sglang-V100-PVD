"""Inventory first; explicit, bounded CAGRA smoke test when hardware is available.

Collect environment information without importing GPU libraries (the default):
    python scripts/pvd/check_cagra.py

Run real build/search in the candidate Linux V100S environment:
    python scripts/pvd/check_cagra.py --mode smoke

Inventory collection is NOT a passed GPU test, even when its exit code is zero.
Smoke success covers only the reported synthetic configuration, not model
quality, RDMA or production PVD compatibility. Never installs dependencies,
changes package versions, or falls back to another algorithm/metric.
"""

import argparse
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--mode",
        choices=["inventory", "smoke"],
        default="inventory",
        help="inventory does not import CuPy/cuVS or require a GPU; smoke executes CAGRA",
    )
    result.add_argument("--device", type=int, default=0)
    result.add_argument("--expected-gpu", default="V100S")
    result.add_argument("--rows", type=int, default=4096)
    result.add_argument("--queries", type=int, default=32)
    result.add_argument("--dim", type=int, default=128)
    result.add_argument("--k", type=int, default=10)
    result.add_argument(
        "--metric", choices=["inner_product", "sqeuclidean"], default="inner_product"
    )
    result.add_argument(
        "--build-algo", choices=["nn_descent", "ivf_pq"], default="nn_descent"
    )
    result.add_argument(
        "--min-recall",
        type=float,
        default=0.90,
        help="Synthetic smoke threshold, NOT a generation-quality threshold",
    )
    return result


def validate(args):
    # Keep exact-reference and graph scratch costs small. Not a load benchmark.
    for name, lower, upper in (
        ("rows", 1024, 8192),
        ("queries", 1, 128),
        ("dim", 32, 256),
        ("k", 1, 32),
    ):
        if not lower <= getattr(args, name) <= upper:
            raise ValueError(
                f"{name} must be in [{lower}, {upper}] for this bounded probe"
            )
    if args.device < 0 or not args.expected_gpu.strip():
        raise ValueError(
            "device must be non-negative and expected-gpu must be explicit"
        )
    if not 0 < args.min_recall <= 1:
        raise ValueError("min-recall must be in (0, 1]")


def inventory():
    versions = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name", "")
        if name.lower().startswith(
            ("cuvs", "libcuvs", "pylibraft", "raft", "rmm", "cupy", "torch")
        ):
            versions[name] = dist.version
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        result["nvidia_smi"] = {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["nvidia_smi"] = {"error": str(exc)}
    return result


def probe(args, report):
    report["stage"] = "import_cupy"
    import cupy as cp
    import numpy as np

    report["stage"] = "check_gpu"
    cp.cuda.Device(args.device).use()
    props = cp.cuda.runtime.getDeviceProperties(args.device)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    report["gpu"] = {
        "name": name,
        "compute_capability": [props["major"], props["minor"]],
        "total_bytes": props["totalGlobalMem"],
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
    }
    if args.expected_gpu.lower() not in name.lower():
        raise RuntimeError(
            f"selected GPU {name!r} does not match {args.expected_gpu!r}"
        )

    report["stage"] = "import_cagra"
    from cuvs.neighbors import cagra

    report["cagra_module"] = getattr(cagra, "__file__", None)
    report["stage"] = "allocate"
    rng = np.random.default_rng(20260920)
    # Non-normalized independent queries test metric handling, not self lookup.
    host_data = rng.standard_normal((args.rows, args.dim)).astype(np.float32)
    host_queries = rng.standard_normal((args.queries, args.dim)).astype(np.float32)
    dataset, queries = cp.asarray(host_data), cp.asarray(host_queries)
    cp.cuda.runtime.deviceSynchronize()

    report["stage"] = "build"
    started = time.perf_counter()
    index = cagra.build(
        cagra.IndexParams(metric=args.metric, build_algo=args.build_algo), dataset
    )
    cp.cuda.runtime.deviceSynchronize()
    report["build_ms"] = 1000 * (time.perf_counter() - started)

    report["stage"] = "search"
    started = time.perf_counter()
    distances, neighbors = cagra.search(cagra.SearchParams(), index, queries, args.k)
    cp.cuda.runtime.deviceSynchronize()
    report["search_ms"] = 1000 * (time.perf_counter() - started)
    ids = cp.asnumpy(cp.asarray(neighbors))
    scores = cp.asnumpy(cp.asarray(distances))

    report["stage"] = "validate_results"
    report.update(
        validate_search_results(
            host_data,
            host_queries,
            ids,
            scores,
            metric=args.metric,
            k=args.k,
            min_recall=args.min_recall,
        )
    )
    report["stage"] = "complete"


def validate_search_results(
    host_data, host_queries, ids, scores, *, metric, k, min_recall
):
    """Independent CPU oracle for returned IDs AND their metric-valued scores.

    A recall-only test accepts a backend returning correct IDs but scores with
    the wrong sign/metric. PVD consumes scores too, so that is not acceptance.
    Import NumPy only here: inventory remains independent of GPU/numeric libs.
    Ties at the exact kth value are interchangeable; tolerance applies to
    numerical score verification, not to expanding the recall cutoff.
    """
    import numpy as np

    if metric not in ("inner_product", "sqeuclidean"):
        raise ValueError("unsupported CAGRA acceptance metric")
    if (
        type(k) is not int
        or type(min_recall) not in (int, float)
        or not 0 < min_recall <= 1
    ):
        raise ValueError("explicit integer k and finite recall threshold required")
    if (
        host_data.ndim != 2
        or host_queries.ndim != 2
        or host_data.shape[1] != host_queries.shape[1]
        or not 0 < k <= len(host_data)
        or len(host_queries) == 0
        or not np.isfinite(host_data).all()
        or not np.isfinite(host_queries).all()
    ):
        raise ValueError("finite matching data/query matrices and valid k required")
    if ids.shape != (len(host_queries), k) or scores.shape != ids.shape:
        raise RuntimeError("invalid search result shape")
    if (
        not np.issubdtype(ids.dtype, np.integer)
        or np.any(ids < 0)
        or np.any(ids >= len(host_data))
    ):
        raise RuntimeError("invalid neighbor IDs")
    if not np.isfinite(scores).all() or any(len(set(row.tolist())) != k for row in ids):
        raise RuntimeError("nonfinite scores or duplicate neighbor IDs")
    data = host_data.astype(np.float64, copy=False)
    queries = host_queries.astype(np.float64, copy=False)
    reference = queries @ data.T
    if metric == "sqeuclidean":
        reference = (
            np.sum(queries**2, axis=1)[:, None]
            + np.sum(data**2, axis=1)[None, :]
            - 2 * reference
        )
    expected_scores = np.take_along_axis(reference, ids.astype(np.int64), axis=1)
    if not np.allclose(scores, expected_scores, rtol=1e-3, atol=1e-3):
        raise RuntimeError(
            "CAGRA scores disagree with the selected metric and neighbor IDs"
        )
    ordering = -reference if metric == "inner_product" else reference
    selected = np.take_along_axis(ordering, ids.astype(np.int64), axis=1)
    cutoff = np.partition(ordering, k - 1, axis=1)[:, k - 1, None]
    recall = float(np.count_nonzero(selected <= cutoff) / ids.size)
    if recall < min_recall:
        raise RuntimeError(
            f"synthetic recall {recall:.4f} below threshold {min_recall}"
        )
    return {
        "synthetic_recall_at_k": recall,
        "score_max_abs_error": float(np.max(np.abs(scores - expected_scores))),
        "score_convention": "higher_dot_product"
        if metric == "inner_product"
        else "lower_squared_distance",
        "score_rtol": 1e-3,
        "score_atol": 1e-3,
        "recall_ties": "any exact kth-score tie is accepted",
    }


def main(argv=None):
    args = parser().parse_args(argv)
    report = {
        "schema": "pvd_cagra_probe_v2",
        "status": "failed",
        "cagra_test": "not_run",
        "stage": "validate_args",
        "config": {
            key: str(value)
            if isinstance(value, float) and not math.isfinite(value)
            else value
            for key, value in vars(args).items()
        },
    }
    try:
        validate(args)
        report["stage"] = "inventory"
        report["environment"] = inventory()
        if args.mode == "inventory":
            report["status"] = "collected"
        else:
            report["cagra_test"] = "running"
            probe(args, report)
            report["cagra_test"] = "passed"
            report["status"] = "passed"
    except Exception as exc:
        if report["cagra_test"] == "running":
            report["cagra_test"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))
    return 0 if report["status"] in ("collected", "passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
