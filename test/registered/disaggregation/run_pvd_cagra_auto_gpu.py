"""Real-GPU short-exact/long-CAGRA coexistence gate, not production serving."""

import argparse
import json
import sys
import time


def _summary(exc):
    lines = str(exc).splitlines()
    return f"{type(exc).__name__}: {lines[0] if lines else ''}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--expect-cuvs-version", required=True)
    parser.add_argument("--native-cap-bytes", type=int, default=536870912)
    args = parser.parse_args(argv)
    if (
        args.device < 0
        or not args.expected_gpu.strip()
        or not args.expect_cuvs_version.strip()
        or not 67108864 <= args.native_cap_bytes <= 1073741824
    ):
        parser.error("explicit bounded GPU/CAGRA settings required")

    import cuvs
    import torch
    from sglang.srt.disaggregation.pvd.cagra_backend import (
        CagraAutoIndexBackend,
        CagraIndexBackend,
    )

    if cuvs.__version__ != args.expect_cuvs_version:
        raise RuntimeError("imported cuVS version differs from requested version")
    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise RuntimeError("requested CUDA device is unavailable")
    gpu = torch.cuda.get_device_name(args.device)
    if args.expected_gpu not in gpu:
        raise RuntimeError("requested GPU identity differs from actual device")
    device = torch.device(f"cuda:{args.device}")
    torch.manual_seed(240924)
    backend = CagraAutoIndexBackend(
        CagraIndexBackend(
            device=device,
            native_bytes_per_index=args.native_cap_bytes,
            graph_degree=32,
            intermediate_degree=64,
            itopk_size=64,
        )
    )
    report = {
        "schema": "pvd-cagra-auto-gpu-v1",
        "status": "failed",
        "gpu": gpu,
        "cuvs_version": cuvs.__version__,
        "native_cap_bytes": args.native_cap_bytes,
        "short_rows": 16,
        "long_rows": 1024,
        "dim": 32,
        "top_k": 4,
        "production_serving_validated": False,
    }
    built = []
    try:
        for label, count in (("short", 16), ("long", 1024)):
            vectors = torch.randn(count, 32, device=device)
            query = vectors[:4].clone()
            started = time.monotonic()
            index = backend.build(vectors, vector_space=label, metric="ip")
            built.append((label, index, vectors, query))
            report[f"{label}_build_ms"] = round(
                (time.monotonic() - started) * 1000, 3
            )
            report[f"{label}_retained_bytes"] = (
                index.handle.numel() * index.handle.element_size()
                if isinstance(index.handle, torch.Tensor)
                else int(index.handle.limit.get_allocated_bytes())
            )
        if not isinstance(built[0][1].handle, torch.Tensor) or isinstance(
            built[1][1].handle, torch.Tensor
        ):
            raise AssertionError("short/long indexes selected the wrong backends")
        for label, index, vectors, query in built:
            rows, scores = backend.search(index, query, top_k=4)
            expected = (query @ vectors.T).gather(1, rows.long())
            error = float((scores - expected).abs().max().item())
            hits = sum(int(i in rows[i].tolist()) for i in range(4))
            report[f"{label}_score_max_abs_error"] = error
            report[f"{label}_self_hits"] = hits
            if error > 1e-3 or hits != 4 or rows.device != device:
                raise AssertionError(f"{label} search identity or score mismatch")
    except Exception as exc:
        report["error"] = _summary(exc)
    finally:
        disposed = 0
        for _, index, _, _ in reversed(built):
            try:
                backend.dispose(index)
                disposed += 1
            except Exception as exc:
                report["dispose_error"] = _summary(exc)
                break
        report["built_count"] = len(built)
        report["disposed_count"] = disposed
    if (
        "error" not in report
        and "dispose_error" not in report
        and report["built_count"] == report["disposed_count"] == 2
    ):
        report["status"] = "passed"
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
