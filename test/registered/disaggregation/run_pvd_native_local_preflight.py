"""Strict standalone Mooncake GPU/RDMA loopback preflight for one PVD rail.

This proves only the local HCA/GPU path. It does not prove cross-node reachability,
P/V/D serving, CAGRA, or latency. Run it in a fresh process so an uncertain
native completion cannot outlive its owner after an aborted validation.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--rail", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)

    report = {
        "schema": "pvd-native-local-preflight-v1",
        "status": "failed",
        "hostname": args.hostname,
        "rail": args.rail,
        "gpu_id": args.gpu_id,
        "cross_node_validated": False,
        "production_pvd_validated": False,
    }
    try:
        # Classic Mooncake reads this at native import time. Do not import it
        # before setting the flag; the adapter rechecks the pinned version.
        os.environ["MC_DISABLE_METACACHE"] = "1"
        root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(root / "python"))

        import torch
        from sglang.srt.disaggregation.pvd.mooncake_engine import (
            MooncakePVDTransferEngine,
        )
        from sglang.srt.disaggregation.pvd.preflight import run_rank_preflight
        from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

        if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
            raise ValueError("requested CUDA GPU is unavailable")
        engine = MooncakePVDTransferEngine(
            hostname=args.hostname,
            gpu_id=args.gpu_id,
            rail=args.rail,
            budget=TransferBudget(staging_bytes=1 << 20, max_inflight=4),
        )
        result = run_rank_preflight(
            rank=0,
            rails=(args.rail,),
            device=f"cuda:{args.gpu_id}",
            engine=engine,
            strict=True,
            transfer_timeout_seconds=args.timeout_seconds,
        )
        health = engine.health()
        if (
            health.get("healthy") is not True
            or health.get("registered_regions") != 0
            or health["lifecycle"]["tracked_transfers"] != 0
        ):
            raise RuntimeError("native preflight left an unhealthy or live owner")
        report.update(
            status="passed",
            gpu=torch.cuda.get_device_name(args.gpu_id),
            torch=torch.__version__,
            mooncake_version=health["mooncake_version"],
            metadata_policy=health["metadata_policy"],
            preflight=result.to_dict(),
        )
    except Exception as exc:  # noqa: BLE001 -- a failed preflight is never a skip
        report.update(
            reason=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=12),
        )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
