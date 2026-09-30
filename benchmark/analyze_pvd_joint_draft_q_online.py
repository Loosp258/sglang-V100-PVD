"""Validate paired P/V/D logs and summarize the experimental learned-Q path."""

import argparse
import hashlib
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from analyze_pvd_split_online import attach, read_events, read_probe


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seconds(fields):
    return {key: float(value) for key, value in re.findall(
        r"(\w+_seconds)=([\d.]+)", fields)}


def read_decode(path):
    entries = {}
    active = None
    for line in path.read_text().splitlines():
        gate = re.search(r"initial fan-in graph gate READY: transfer_id=(\S+) "
                         r"wait_seconds=([\d.]+)", line)
        if gate:
            active = gate[1]
            require(active not in entries, "duplicate initial graph gate")
            entries[active] = {"graph_gate_wait_seconds": float(gate[2]),
                               "refreshes": [], "prediction_stages": []}
        installed = re.search(r"initial KV installed from V: transfer_id=(\S+) "
                              r"fanin_seconds=([\d.]+)", line)
        if installed:
            require(installed[1] == active, "initial KV Entry mismatch")
            entries[active]["initial_fanin_seconds"] = float(installed[2])
        if "PVD refresh ready:" in line:
            require(active is not None, "refresh before bootstrap")
            entries[active]["refreshes"].append({
                **seconds(line),
                "query_source": re.search(r"query_source=(\w+)", line)[1],
            })
        if "PVD CUDA prediction stages:" in line:
            entries[active]["prediction_stages"].append(seconds(line))
        joint = re.search(r"PVD joint Draft-Q: request=(\S+) "
                          r"prefix_tokens=(\d+) horizon=(\d+)", line)
        if joint:
            entries[active]["prediction_stages"].append({
                **seconds(line), "request_id": joint[1],
                "prefix_tokens": int(joint[2]), "horizon": int(joint[3]),
                "target_forward_count": int(re.search(
                    r"target_forward_count=(\d+)", line)[1]),
            })
    return entries


def read_split(path):
    entries = {}
    for line in path.read_text().splitlines():
        match = re.search(r"predicted Prompt split: transfer_id=(\S+) "
                          r"tokens=(\d+) prefix=(\d+) best_prefix=(\d+) "
                          r"predicted_gain_seconds=([\d.]+)", line)
        if match:
            entries[match[1]] = {
                "tokens": int(match[2]), "prefix": int(match[3]),
                "best_prefix": int(match[4]), "predicted_gain_seconds": float(match[5]),
            }
    return entries


def ready_events(path):
    entries = {}
    for line in path.read_text().splitlines():
        match = re.search(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}).*"
                          r"provisional graph READY: transfer_id=(\S+) rank=(\d+)", line)
        if match:
            stamp = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S,%f")
            entries.setdefault(match[2], {})[match[3]] = stamp.replace(
                tzinfo=timezone.utc).timestamp()
    return entries


def analyze_arm(folder, arm):
    tag = f"jointq-{arm}-20260930"
    v_path = folder / f"v-{tag}.log"
    probes = attach(read_probe(folder / f"{tag}.jsonl"), read_events(v_path))
    decode = read_decode(folder / f"d-{tag}.log")
    split = read_split(folder / f"p-{tag}.log")
    ready = ready_events(v_path)
    require(len(probes) == len(decode) == len(split) == 5, "missing request/Entry")
    for probe in probes:
        entry = probe["transfer_id"]
        require(probe["status"] == 200 and probe["error"] is None,
                "client request failed")
        require(probe["completion_tokens"] == (6 if probe["case"] == "warmup" else 16),
                "output token count changed")
        require(set(ready[entry]) == {"0", "1"}, "both V ranks must be READY")
        require(split[entry]["tokens"] == probe["prompt_tokens"], "split length mismatch")
        require(split[entry]["prefix"] == 0, "this trial selected a split")
        probe["split_decision"] = split[entry]
        probe["decode"] = decode[entry]
        probe["both_graphs_ready_seconds"] = max(ready[entry].values()) - probe["started_unix"]
        probe["graph_build_seconds_by_rank"] = {
            rank: events[0]["seconds"] for rank, events in probe["ranks"].items()
        }
        require(probe["both_graphs_ready_seconds"] < probe["first_event_seconds"],
                "first client token arrived before the graph barrier")
        require(decode[entry]["initial_fanin_seconds"] >= decode[entry]["graph_gate_wait_seconds"],
                "fanin must include its readiness wait")
        n = 1 if probe["case"] == "warmup" else 3
        require(len(decode[entry]["refreshes"]) == n, "unexpected refresh count")
        require(len(decode[entry]["prediction_stages"]) == n, "missing prediction timings")
        require(all(r["query_source"] == "predicted" for r in decode[entry]["refreshes"]),
                "committed-Q fallback occurred")
        if arm == "joint":
            require(all(s["target_forward_count"] == 0
                        and s["request_id"] == probe["request_id"]
                        and s["horizon"] == 8 for s in decode[entry]["prediction_stages"]),
                    "joint Q used a target forward or mismatched request")
    return probes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = {arm: analyze_arm(args.folder, arm) for arm in ("joint", "target")}
    by_case = {arm: {p["case"]: p for p in probes if p["case"] != "warmup"}
               for arm, probes in arms.items()}
    require(set(by_case["joint"]) == set(by_case["target"]), "cases differ")
    pairs = []
    for case, joint in by_case["joint"].items():
        target = by_case["target"][case]
        require((joint["prompt_sha256"], joint["prompt_tokens"], joint["completion_tokens"])
                == (target["prompt_sha256"], target["prompt_tokens"], target["completion_tokens"]),
                "paired Prompt or output length differs")
        gain = target["wall_seconds"] - joint["wall_seconds"]
        pair = {
            "case": case, "prompt_tokens": joint["prompt_tokens"],
            "joint_total_seconds": joint["wall_seconds"],
            "target_total_seconds": target["wall_seconds"],
            "saved_seconds": gain, "saved_fraction": gain / target["wall_seconds"],
            "output_match": joint["final_text_sha256"] == target["final_text_sha256"],
        }
        pairs.append(pair)
        print(json.dumps(pair))
    files = sorted(args.folder.glob("*"))
    payload = {
        "serving_commit": "4ff33838636ca2cea49678a1491e7514de45a52a",
        "arm_order": ["joint", "target"], "repetitions_per_case_per_arm": 1,
        "cold_warmup_excluded": True,
        "pairs": pairs, "arms": arms,
        "median_client_seconds": {arm: statistics.median(p["wall_seconds"]
            for p in by_case[arm].values()) for arm in arms},
        "raw_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in files if p.suffix in (".log", ".jsonl")},
        "limitations": ["No quality equivalence: natural-text outputs differ",
                        "No repeated or reversed-order trial",
                        "TP1 D and two V GPU ranks; serialized single request",
                        "No online exact-Q recall measurement"],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
