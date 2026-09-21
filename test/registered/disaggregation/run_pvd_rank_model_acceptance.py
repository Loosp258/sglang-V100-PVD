"""Strict CPU acceptance CLI: missing dependencies/failures are NOT skips."""

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from pvd_rank_model_acceptance import AcceptanceError, parse_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    if not __debug__ or args.timeout_seconds <= 0:
        parser.error("assertions must be enabled and timeout must be positive")
    script = Path(__file__).resolve().with_name("run_pvd_draft_cpu_smoke.py")
    repo = script.parents[3]
    run_id = uuid.uuid4().hex
    env = dict(os.environ)
    env.pop("PYTHONOPTIMIZE", None)
    env["PYTHONPATH"] = str(repo / "python") + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        str(script),
        "--rank-runtime-loop",
        "--acceptance-run-id",
        run_id,
    ]
    try:
        child = subprocess.run(
            command,
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
        report = parse_report(child.stdout, returncode=child.returncode, run_id=run_id)
    except (OSError, subprocess.TimeoutExpired, AcceptanceError) as exc:
        print(f"PVD CPU acceptance failed: {exc}", file=sys.stderr)
        if "child" in locals():
            print(child.stdout[-16000:], file=sys.stderr)
            print(child.stderr[-16000:], file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "scope": "CPU TP1; local rank control; fake payload transport",
                "python": sys.executable,
                "evidence": report,
                "production_gpu_rdma_validated": False,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
