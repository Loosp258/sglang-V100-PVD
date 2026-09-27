"""Summarize one local PVD Decode timeline without copying its request log."""

import argparse
import json
import re
import statistics
from pathlib import Path


def summarize(path):
    lines = Path(path).read_text(errors="replace").splitlines()
    batches = []
    events = {}
    for line in lines:
        match = re.search(r"PVD timeline event=([a-z_]+)", line)
        if match is None:
            continue
        event = match.group(1)
        events.setdefault(event, []).append(line)
        if event == "target_batch":
            token = re.search(r"committed_tokens=\((\d+),\)", line)
            duration = re.search(r"\bseconds=([0-9.]+)", line)
            start = re.search(r"\bt_start=([0-9.]+)", line)
            end = re.search(r"\bt_end=([0-9.]+)", line)
            if all(value is not None for value in (token, duration, start, end)):
                batches.append(
                    (
                        int(token.group(1)),
                        float(duration.group(1)),
                        float(start.group(1)),
                        float(end.group(1)),
                    )
                )
    segments = []
    for start in range(0, 128, 32):
        stop = start + 32
        values = [seconds for token, seconds, _, _ in batches if start <= token < stop]
        segments.append(
            {
                "tokens": f"{start}-{start + 31}",
                "count": len(values),
                "forward_sum_seconds": round(sum(values), 4),
                "forward_p50_seconds": round(statistics.median(values), 6)
                if values
                else None,
            }
        )
    stages = {}
    for event in (
        "prediction_step",
        "refresh_scheduled",
        "refresh_ready",
        "installed",
        "search_http",
    ):
        rows = events.get(event, [])
        timestamps = [
            float(match.group(1))
            for line in rows
            if (match := re.search(r"\bt=([0-9.]+)", line)) is not None
        ]
        durations = [
            float(match.group(1))
            for line in rows
            if (match := re.search(r"\bseconds=([0-9.]+)", line)) is not None
        ]
        starts = [
            float(match.group(1))
            for line in rows
            if (match := re.search(r"\bt_start=([0-9.]+)", line)) is not None
        ]
        ends = [
            float(match.group(1))
            for line in rows
            if (match := re.search(r"\bt_end=([0-9.]+)", line)) is not None
        ]
        stages[event] = {
            "count": len(rows),
            "first_t": min(timestamps) if timestamps else None,
            "last_t": max(timestamps) if timestamps else None,
            "sum_seconds": round(sum(durations), 4),
            "span_seconds": round(max(ends) - min(starts), 4)
            if starts and ends
            else (
                round(max(timestamps) - min(timestamps), 4)
                if len(timestamps) > 1
                else None
            ),
        }
    boundary = next((end for token, _, _, end in batches if token == 63), None)
    ready = stages["refresh_ready"]["first_t"]
    forward_sum = sum(seconds for _, seconds, _, _ in batches)
    forward_span = max(end for _, _, _, end in batches) - min(
        start for _, _, start, _ in batches
    )
    return {
        "target_batches": len(batches),
        "target_forward_sum_seconds": round(forward_sum, 4),
        "target_forward_span_seconds": round(forward_span, 4),
        "outside_target_forward_seconds": round(forward_span - forward_sum, 4),
        "boundary_63_to_ready_seconds": round(ready - boundary, 4)
        if ready is not None and boundary is not None
        else None,
        "segments": segments,
        "stages": stages,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.log), indent=2))
