"""Bounded real-GPU acceptance of PVD's CagraIndexBackend (not serving).

Run in an isolated cuVS candidate environment. A successful result validates
the adapter's native allocator bridge, build/search/dispose and synthetic
logical-row scores on one GPU; it does not validate model-query recall.
"""

import argparse
import json
import sys
import time


def _error_summary(exc):
    lines = str(exc).splitlines()
    return f"{type(exc).__name__}: {lines[0] if lines else ''}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--expect-cuvs-version", required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--index-count", type=int, default=1)
    parser.add_argument("--native-cap-bytes", type=int, default=536870912)
    args = parser.parse_args(argv)
    if not (
        1024 <= args.rows <= 8192
        and 1 <= args.queries <= 32
        and 32 <= args.dim <= 256
        and 1 <= args.top_k <= 32
        and 1 <= args.index_count <= 4
        and 67108864 <= args.native_cap_bytes <= 1073741824
        and args.device >= 0
        and args.expected_gpu.strip()
        and args.expect_cuvs_version.strip()
    ):
        parser.error("GPU probe bounds exceeded")

    import cuvs
    import torch

    from sglang.srt.disaggregation.pvd.cagra_backend import CagraIndexBackend

    if cuvs.__version__ != args.expect_cuvs_version:
        raise RuntimeError(
            f"imported cuVS {cuvs.__version__!r} differs from requested "
            f"{args.expect_cuvs_version!r}"
        )
    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise RuntimeError("requested CUDA device is unavailable")
    name = torch.cuda.get_device_name(args.device)
    if args.expected_gpu not in name:
        raise RuntimeError(f"expected {args.expected_gpu!r}, found {name!r}")
    device = torch.device(f"cuda:{args.device}")
    torch.manual_seed(240923)
    backend = CagraIndexBackend(
        device=device,
        native_bytes_per_index=args.native_cap_bytes,
        graph_degree=32,
        intermediate_degree=64,
        itopk_size=64,
    )
    built = []
    result = {
        "status": "failed",
        "gpu": name,
        "cuvs_version": cuvs.__version__,
        "rows": args.rows,
        "queries": args.queries,
        "dim": args.dim,
        "top_k": args.top_k,
        "index_count": args.index_count,
        "native_cap_bytes": args.native_cap_bytes,
    }
    try:
        build_started = time.monotonic()
        for n in range(args.index_count):
            vectors = torch.randn(
                args.rows, args.dim, device=device, dtype=torch.float32
            )
            queries = vectors[: args.queries].clone()
            index = backend.build(
                vectors, vector_space=f"pvd-cagra-gpu-probe-{n}", metric="ip"
            )
            built.append((index, vectors, queries))
        result["build_ms"] = round((time.monotonic() - build_started) * 1000, 3)
        result["native_retained_bytes"] = [
            int(index.handle.limit.get_allocated_bytes())
            for index, _, _ in built
        ]
        started = time.monotonic()
        max_abs_error = 0.0
        self_hits = 0
        for index, vectors, queries in built:
            rows, scores = backend.search(index, queries, top_k=args.top_k)
            exact_scores = queries @ vectors.T
            selected_scores = exact_scores.gather(1, rows.long())
            max_abs_error = max(
                max_abs_error, float((selected_scores - scores).abs().max().item())
            )
            self_hits += sum(int(i in rows[i].tolist()) for i in range(args.queries))
        result["search_ms"] = round((time.monotonic() - started) * 1000, 3)
        result.update(
            score_max_abs_error=max_abs_error,
            self_hits=self_hits,
            expected_self_hits=args.queries * args.index_count,
        )
        if max_abs_error > 1e-3 or self_hits != args.queries * args.index_count:
            raise AssertionError("CAGRA adapter score or self-neighbor check failed")
    except Exception as exc:
        # Native cuVS exceptions can include a many-frame C++ backtrace.
        # Keep the machine-readable probe bounded while retaining its cause.
        result["error"] = _error_summary(exc)
    finally:
        disposed = 0
        for index, _, _ in reversed(built):
            try:
                backend.dispose(index)
                disposed += 1
            except Exception as exc:
                result["dispose_error"] = _error_summary(exc)
                break
        result["disposed_count"] = disposed
        result["built_count"] = len(built)
    if (
        "error" not in result
        and "dispose_error" not in result
        and result["disposed_count"] == args.index_count
    ):
        result["status"] = "passed"
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
