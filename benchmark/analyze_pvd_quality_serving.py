"""Attach actual V bootstrap, predicted-Q and bank-install evidence to QA rows."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re


def log_text(path):
    return gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text(encoding="utf-8")


def attach(rows, paths, arm):
    entries, boundaries = {}, {}
    active = None
    for path in paths:
        for line in log_text(path).splitlines():
            gate = re.search(r"initial fan-in graph gate READY: transfer_id=(\S+) "
                             r"wait_seconds=([\d.]+)", line)
            if gate:
                active = gate[1]
                entries.setdefault(active, {"refreshes": [], "joint_predictions": []})
                entries[active]["graph_gate_wait_seconds"] = float(gate[2])
            installed = re.search(r"initial KV installed from V: transfer_id=(\S+)", line)
            if installed:
                entries.setdefault(installed[1], {"refreshes": [], "joint_predictions": []})[
                    "initial_kv_from_v"] = True
            refresh = re.search(r"refresh ready: query_source=(\w+)", line)
            if refresh:
                entries[active]["refreshes"].append(refresh[1])
            prediction = re.search(r"joint Draft-Q: request=(\S+) .*target_forward_count=(\d+)", line)
            if prediction:
                entries[active]["joint_predictions"].append({
                    "request_id": prediction[1], "target_forward_count": int(prediction[2])})
            bank = re.search(r"boundary installed: request_id=(\S+) boundary=(\d+)", line)
            if bank:
                boundaries.setdefault(bank[1], []).append(int(bank[2]))
    result = []
    for row in rows:
        transfer_id = row["entry_transfer_id"]
        event = entries.get(transfer_id, {})
        if not event.get("initial_kv_from_v") or "graph_gate_wait_seconds" not in event:
            raise ValueError(f"missing V bootstrap proof for {row['id']}")
        points = boundaries.get(row["request_id"], [])
        if arm == "full" and (event["refreshes"] or points or event["joint_predictions"]):
            raise ValueError("full arm ran sparse prediction")
        if arm == "joint" and any(p["target_forward_count"] != 0
            or p["request_id"] != row["request_id"] for p in event["joint_predictions"]):
            raise ValueError("joint arm used a target forward or wrong request")
        result.append({"id": row["id"], "request_id": row["request_id"],
                       "entry_transfer_id": transfer_id, "benchmark": row["benchmark"],
                       **event, "installed_boundaries": points,
                       "sparse_bank_installed": bool(points),
                       "tokens_after_first_boundary": max(0, row["completion_tokens"] - min(points))
                           if points else 0})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = {}
    logs = []
    for arm in ("full", "target", "joint"):
        paths = sorted(args.folder.glob(f"d-quality-{arm}*.log*"))
        if not paths:
            raise ValueError(f"missing {arm} D logs")
        # Full pilot logs are deliberately not copied into this prefix. The
        # final full run and resumed last request have distinct Entry IDs.
        rows = [json.loads(line) for line in (args.folder / f"{arm}.jsonl").read_text(encoding="utf-8").splitlines()]
        arms[arm] = attach(rows, paths, arm)
        logs.extend(paths)
    payload = {"arms": arms, "summary": {
        arm: {benchmark: {
            "requests": len(subset := [row for row in rows if row["benchmark"] == benchmark]),
            "with_sparse_bank_install": sum(row["sparse_bank_installed"] for row in subset),
            "installed_boundaries": sum(len(row["installed_boundaries"]) for row in subset),
            "predicted_refreshes": sum(row["refreshes"].count("predicted") for row in subset),
            "committed_refreshes": sum(row["refreshes"].count("committed") for row in subset),
        } for benchmark in ("gsm8k", "hotpotqa")} for arm, rows in arms.items()},
        "log_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in logs}}
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
