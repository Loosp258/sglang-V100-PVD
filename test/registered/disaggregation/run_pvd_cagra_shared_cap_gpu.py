"""Prove a cuVS C-API allocation sees one nested per-index/global RMM cap."""

import argparse
import ctypes
import json
import sys


def _summary(exc):
    lines = str(exc).splitlines()
    return f"{type(exc).__name__}: {lines[0] if lines else ''}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--expect-cuvs-version", required=True)
    args = parser.parse_args(argv)

    import cuvs
    import torch
    from sglang.srt.disaggregation.pvd.cagra_backend import (
        CagraIndexBackend,
        CagraNativeRuntime,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    gpu = torch.cuda.get_device_name(0)
    if args.expected_gpu not in gpu or cuvs.__version__ != args.expect_cuvs_version:
        raise RuntimeError("CUDA GPU or cuVS candidate does not match the request")
    global_cap, per_index_cap, probe_bytes = 640 << 20, 512 << 20, 360 << 20
    runtime = CagraNativeRuntime(
        torch.device("cuda:0"), global_native_cap_bytes=global_cap
    )
    backend = CagraIndexBackend(
        device="cuda:0", native_bytes_per_index=per_index_cap,
        graph_degree=32, intermediate_degree=64, itopk_size=64,
        _runtime=runtime,
    )
    report = {
        "schema": "pvd-cagra-shared-native-cap-v1",
        "status": "failed",
        "gpu": gpu,
        "cuvs_version": cuvs.__version__,
        "global_cap_bytes": global_cap,
        "per_index_cap_bytes": per_index_cap,
        "probe_bytes": probe_bytes,
        "production_budget_integrated": False,
    }
    indexes = []
    first_pointer = ctypes.c_void_p()
    first_live = False
    unknown_native_owner = False
    try:
        for n in range(2):
            vectors = torch.randn((1024, 32), device="cuda:0")
            index = backend.build(vectors, vector_space=f"shared-{n}", metric="ip")
            indexes.append((index, vectors))
        retained = sum(
            int(index.handle.limit.get_allocated_bytes()) for index, _ in indexes
        )
        if retained <= 0 or runtime.global_allocated_bytes() != retained:
            raise AssertionError("shared root does not count both native indexes")
        report["retained_native_bytes"] = retained
        for index, vectors in indexes:
            rows, scores = backend.search(index, vectors[:4], top_k=4)
            expected = (vectors[:4] @ vectors.T).gather(1, rows.long())
            if (
                float((scores - expected).abs().max()) > 1e-3
                or sum(int(i in rows[i].tolist()) for i in range(4)) != 4
            ):
                raise AssertionError("nested native index search failed")

        first, second = (index.handle for index, _ in indexes)
        with runtime.scope(first):
            status = runtime.alloc(
                first.resources.get_c_obj(), ctypes.byref(first_pointer), probe_bytes
            )
            if status != 1 or not first_pointer.value:
                unknown_native_owner = bool(first_pointer.value)
                raise AssertionError("first C-API allocation under the root failed")
            first_live = True
            runtime.synchronize()
        before_rejection = runtime.global_allocated_bytes()
        if before_rejection < retained + probe_bytes:
            raise AssertionError("shared root did not account for the first probe")
        second_pointer = ctypes.c_void_p()
        with runtime.scope(second):
            status = runtime.alloc(
                second.resources.get_c_obj(), ctypes.byref(second_pointer),
                probe_bytes,
            )
            runtime.synchronize()
            if status == 1 or second_pointer.value:
                # The cap failed. A returned pointer has uncertain ownership;
                # retain native indexes/resources until this process exits.
                unknown_native_owner = bool(second_pointer.value)
                raise AssertionError("shared root allowed the over-cap second probe")
        if runtime.global_allocated_bytes() != before_rejection:
            raise AssertionError("rejected probe changed shared allocation count")
        report["second_probe_rejected_by_shared_cap"] = True
    except Exception as exc:
        report["error"] = _summary(exc)
    finally:
        if first_live:
            try:
                with runtime.scope(indexes[0][0].handle):
                    if runtime.free(
                        indexes[0][0].handle.resources.get_c_obj(),
                        first_pointer, probe_bytes,
                    ) != 1:
                        raise RuntimeError("first probe release was not proved")
                    runtime.synchronize()
            except Exception as exc:
                report["probe_release_error"] = _summary(exc)
        disposed = 0
        if "probe_release_error" not in report and not unknown_native_owner:
            for index, _ in reversed(indexes):
                try:
                    backend.dispose(index)
                    disposed += 1
                except Exception as exc:
                    report["dispose_error"] = _summary(exc)
                    break
        report["built_count"] = len(indexes)
        report["disposed_count"] = disposed
        report["native_owner_unknown"] = unknown_native_owner
        report["root_bytes_after_disposal"] = runtime.global_allocated_bytes()
    if (
        not any(key.endswith("error") for key in report)
        and report.get("second_probe_rejected_by_shared_cap") is True
        and report["built_count"] == report["disposed_count"] == 2
        and report["root_bytes_after_disposal"] == 0
    ):
        report["status"] = "passed"
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
