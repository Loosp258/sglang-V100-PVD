"""Combine client probes with V graph logs for online split calibration."""

import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path


EVENT = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}).*"
    r"PVD Prompt provisional (graph|final step): transfer_id=([\w-]+) "
    r"rank=(\d+) (?:pages=(\d+) first_page=(\d+)|first_page=(\d+) pages=(\d+)) "
    r"heads=\d+ .*seconds=(\d+\.\d+)"
)


def read_events(path):
    entries = {}
    for line in path.read_text().splitlines():
        match = EVENT.match(line)
        if not match:
            continue
        stamp, kind, transfer_id, rank, graph_pages, graph_first, final_first, final_pages, duration = match.groups()
        end = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=timezone.utc).timestamp()
        event = {
            "rank": int(rank),
            "kind": kind,
            "pages": int(graph_pages or final_pages),
            "prefix": int(graph_pages) if kind == "graph" else 0,
            "first_page": int(graph_first or final_first),
            "start_unix": end - float(duration),
            "end_unix": end,
            "seconds": float(duration),
        }
        entries.setdefault(transfer_id, []).append(event)
    return entries


def read_probe(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def attach(probes, entries):
    groups = sorted(entries.items(), key=lambda pair: min(e["start_unix"] for e in pair[1]))
    attached = []
    for probe in probes:
        start = probe["started_unix"]
        candidates = [
            (key, events) for key, events in groups
            if start <= min(e["start_unix"] for e in events) < start + probe["wall_seconds"] + 0.2
        ]
        if len(candidates) != 1:
            raise ValueError(f"expected one graph for case {probe['case']}: {len(candidates)}")
        key, events = candidates[0]
        if len(events) not in (2, 4):
            raise ValueError(f"incomplete graph events for {key}: {len(events)}")
        n = max(e["pages"] for e in events)
        attached.append({**probe, "tokens": n, "transfer_id": key, "ranks": {str(rank): [
            {**e, "start_seconds": e["start_unix"] - start,
             "end_seconds": e["end_unix"] - start}
            for e in events if e["rank"] == rank
        ] for rank in (0, 1)}})
    return attached


def calibration(full, split):
    by_prompt = {}
    for n in sorted({item["tokens"] for item in full} & {item["tokens"] for item in split}):
        f = [item for item in full if item["tokens"] == n]
        s = [item for item in split if item["tokens"] == n]
        prefix = next((event["prefix"] for item in s for event in item["ranks"]["0"] if event["kind"] == "graph"), 0)
        if not prefix:
            continue
        ranks = []
        for rank in (0, 1):
            r = str(rank)
            full_final = [next(e for e in item["ranks"][r] if e["kind"] == "final step") for item in f]
            split_first = [next(e for e in item["ranks"][r] if e["kind"] == "graph") for item in s]
            split_final = [next(e for e in item["ranks"][r] if e["kind"] == "final step") for item in s]
            median = statistics.median
            ranks.append({
                "baseline_full_seconds": median(e["start_seconds"] for e in full_final),
                "split_full_seconds": median(e["start_seconds"] for e in split_final),
                "prefix_seconds": {str(prefix): median(e["start_seconds"] for e in split_first)},
                "online_graph_seconds": {
                    "0": {"build_seconds": median(e["seconds"] for e in full_final), "extend_seconds": 0.0},
                    str(prefix): {
                        "build_seconds": median(e["seconds"] for e in split_first),
                        "extend_seconds": median(e["seconds"] for e in split_final),
                    },
                },
            })
        by_prompt[str(n)] = {"ranks": ranks, "pilot_samples_per_arm": [len(f), len(s)]}
    return {"by_prompt": by_prompt, "arrival_definition": "V graph execution start, an upper bound on complete chunk arrival"}


def merge_calibrations(parts):
    """Keep the same full baseline and candidate-specific final arrival."""
    merged = {"by_prompt": {}, "arrival_definition": parts[0]["arrival_definition"]}
    for part in parts:
        for n, trace in part["by_prompt"].items():
            if n not in merged["by_prompt"]:
                merged["by_prompt"][n] = {"ranks": [
                    {
                        "baseline_full_seconds": rank["baseline_full_seconds"],
                        "split_full_seconds": {},
                        "prefix_seconds": {},
                        "online_graph_seconds": {"0": rank["online_graph_seconds"]["0"]},
                    }
                    for rank in trace["ranks"]
                ]}
            for target, rank in zip(merged["by_prompt"][n]["ranks"], trace["ranks"]):
                if abs(target["baseline_full_seconds"] - rank["baseline_full_seconds"]) > 1e-9:
                    raise ValueError("split candidates use different full baselines")
                for prefix, arrival in rank["prefix_seconds"].items():
                    if prefix in target["prefix_seconds"]:
                        raise ValueError(f"duplicate split candidate {n}/{prefix}")
                    target["prefix_seconds"][prefix] = arrival
                    target["split_full_seconds"][prefix] = rank["split_full_seconds"]
                    target["online_graph_seconds"][prefix] = rank["online_graph_seconds"][prefix]
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("full", type=Path)
    parser.add_argument("split", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = read_events(args.events)
    full = attach(read_probe(args.full), entries)
    splits = {path.stem: attach(read_probe(path), entries) for path in args.split}
    if len(splits) != len(args.split):
        raise ValueError("split probe filenames must be distinct")
    payload = {
        "calibration": merge_calibrations([
            calibration(full, split) for split in splits.values()
        ]),
        "full": full, "splits": splits,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["calibration"], indent=2))


if __name__ == "__main__":
    main()
