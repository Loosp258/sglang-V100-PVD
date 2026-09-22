"""Strict CUDA component acceptance. No-device/skipped/missing tests never pass.

Not model serving, CAGRA, RDMA, real multi-GPU TP or performance acceptance.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

CASES = {
    "test_pvd_cuda_receive_ordering": ("test_real_cuda_receive_sync_memops_roundtrip",),
    "test_pvd_cuda_sparse_packing": (
        "test_actual_cuda_store_pack_lifetime_and_bytes[False]",
        "test_actual_cuda_store_pack_lifetime_and_bytes[True]",
    ),
    "test_pvd_cuda_working_set": (
        "test_real_cuda_bank_copy_stream_reader_and_switch[dtype0]",
        "test_real_cuda_bank_copy_stream_reader_and_switch[dtype1]",
    ),
    "test_pvd_cuda_rank_install": (
        "test_real_cuda_local_participant_completes_handshake",
    ),
    "test_pvd_cuda_sparse_attention": (
        "test_real_cuda_tiled_math_matches_cpu_oracle[dtype0]",
        "test_real_cuda_tiled_math_matches_cpu_oracle[dtype1]",
        "test_real_cuda_attention_consumes_installed_banks_across_refresh",
    ),
}


class CUDAAcceptanceError(ValueError):
    pass


def validate_junit(xml, *, returncode):
    if returncode != 0:
        raise CUDAAcceptanceError(f"test process exited {returncode}")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise CUDAAcceptanceError("invalid or missing test evidence") from exc
    if root.tag not in ("testsuite", "testsuites"):
        raise CUDAAcceptanceError("unrecognized test evidence root")
    if any(node.tag in ("failure", "error", "skipped") for node in root.iter()):
        raise CUDAAcceptanceError("failed/error/skipped CUDA tests are not acceptance")
    expected = {(module, name) for module, names in CASES.items() for name in names}
    seen = set()
    for case in root.iter("testcase"):
        key = (case.get("classname", "").rsplit(".", 1)[-1], case.get("name"))
        if key not in expected or key in seen:
            raise CUDAAcceptanceError("unexpected or duplicate CUDA case")
        seen.add(key)
    if seen != expected:
        raise CUDAAcceptanceError("required CUDA cases were not all executed")
    return [f"{module}::{name}" for module, name in sorted(seen)]


def inventory(expected_gpu):
    import torch

    if not torch.cuda.is_available():
        raise CUDAAcceptanceError(
            "CUDA is unavailable in the active Python environment"
        )
    device = torch.cuda.get_device_properties(0)
    if expected_gpu and expected_gpu.lower() not in device.name.lower():
        raise CUDAAcceptanceError(
            f"expected GPU {expected_gpu!r}, found {device.name!r}"
        )
    return {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device": "cuda:0",
        "name": device.name,
        "capability": [device.major, device.minor],
        "total_memory": device.total_memory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-gpu", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    if not __debug__ or args.timeout_seconds <= 0:
        parser.error("assertions must be enabled and timeout must be positive")
    report = {
        "schema": "pvd-cuda-components-v1",
        "status": "blocked",
        "scope": "real CUDA components; local control; fake V payload transport",
        "production_gpu_rdma_validated": False,
        "cagra_validated": False,
        "model_forward_validated": False,
        "performance_validated": False,
    }
    try:
        report["runtime"] = inventory(args.expected_gpu)
    except Exception as exc:  # noqa: BLE001 -- report optional runtime/import failures, never claim acceptance
        report["reason"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(report, indent=2))
        return 2
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    env = dict(os.environ)
    env.pop("PYTHONOPTIMIZE", None)
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONPATH"] = str(repo / "python") + os.pathsep + env.get("PYTHONPATH", "")
    child = None
    try:
        with tempfile.TemporaryDirectory(prefix="pvd-cuda-acceptance-") as temporary:
            evidence = Path(temporary) / "results.xml"
            nodes = [
                str(here / f"{module}.py") + "::" + name
                for module, names in CASES.items()
                for name in names
            ]
            child = subprocess.run(
                [
                    sys.executable,
                    str(here / "run_pvd_cpu_tests.py"),
                    *nodes,
                    "-q",
                    "--tb=short",
                    "-p",
                    "no:cacheprovider",
                    f"--junitxml={evidence}",
                ],
                cwd=repo,
                env=env,
                text=True,
                capture_output=True,
                timeout=args.timeout_seconds,
                check=False,
            )
            report["passed_cases"] = validate_junit(
                evidence.read_text(encoding="utf-8"), returncode=child.returncode
            )
        report["status"] = "passed"
        code = 0
    except (OSError, subprocess.TimeoutExpired, CUDAAcceptanceError) as exc:
        report["status"] = "failed"
        report["reason"] = f"{type(exc).__name__}: {exc}"
        if child is not None:
            report["stdout_tail"] = child.stdout[-16000:]
            report["stderr_tail"] = child.stderr[-16000:]
        code = 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
