"""Compare paired PVD V-to-D fan-in runs by Prompt hash and token length."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({row["case"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate case in {path}")
    return {row["case"]: row for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("gated", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, gated = read(args.baseline), read(args.gated)
    if set(baseline) != set(gated):
        raise ValueError("case IDs differ across arms")
    groups = defaultdict(list)
    for case in sorted(baseline):
        a, b = baseline[case], gated[case]
        for key in ("repetitions", "prompt_sha256", "completion_tokens", "final_text_sha256"):
            if a[key] != b[key]:
                raise ValueError(f"case {case} differs in {key}")
        if a["status"] != 200 or b["status"] != 200 or a.get("error") or b.get("error"):
            raise ValueError(f"case {case} failed")
        groups[a["repetitions"]].append({
            "case": case,
            "baseline_first_seconds": a["first_event_seconds"],
            "gated_first_seconds": b["first_event_seconds"],
            "first_delta_seconds": round(b["first_event_seconds"] - a["first_event_seconds"], 3),
            "baseline_wall_seconds": a["wall_seconds"],
            "gated_wall_seconds": b["wall_seconds"],
            "wall_delta_seconds": round(b["wall_seconds"] - a["wall_seconds"], 3),
        })
    summary = {}
    for reps, rows in groups.items():
        summary[str(reps)] = {
            "samples": len(rows),
            "median_baseline_first_seconds": statistics.median(row["baseline_first_seconds"] for row in rows),
            "median_gated_first_seconds": statistics.median(row["gated_first_seconds"] for row in rows),
            "median_first_delta_seconds": statistics.median(row["first_delta_seconds"] for row in rows),
            "median_baseline_wall_seconds": statistics.median(row["baseline_wall_seconds"] for row in rows),
            "median_gated_wall_seconds": statistics.median(row["gated_wall_seconds"] for row in rows),
            "median_wall_delta_seconds": statistics.median(row["wall_delta_seconds"] for row in rows),
            "cases": rows,
        }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
