"""Summarize one local PVD Decode timeline without copying its request log."""

import argparse
import json
import re
import statistics
from pathlib import Path


def summarize(path, *, batch_event="target_batch"):
    if batch_event not in ("target_batch", "full_kv_batch"):
        raise ValueError("unsupported batch event")
    lines = Path(path).read_text(errors="replace").splitlines()
    batches = []
    events = {}
    for line in lines:
        match = re.search(r"PVD timeline event=([a-z_]+)", line)
        if match is None:
            continue
        event = match.group(1)
        events.setdefault(event, []).append(line)
        if event == batch_event:
            token = re.search(r"committed_tokens=\((\d+),\)", line)
            duration = re.search(r"\bseconds=([0-9.]+)", line)
            run = re.search(r"\brun_seconds=([0-9.]+)", line)
            process = re.search(r"\bprocess_seconds=([0-9.]+)", line)
            start = re.search(r"\bt_start=([0-9.]+)", line)
            end = re.search(r"\bt_end=([0-9.]+)", line)
            if all(value is not None for value in (token, duration, start, end)):
                batches.append(
                    (
                        int(token.group(1)),
                        float(duration.group(1)),
                        float(start.group(1)),
                        float(end.group(1)),
                        float(run.group(1)) if run is not None else None,
                        float(process.group(1)) if process is not None else None,
                    )
                )
    gaps = [
        (index, batches[index][2] - batches[index - 1][3])
        for index in range(1, len(batches))
    ]
    segments = []
    for start in range(0, 128, 32):
        stop = start + 32
        values = [seconds for _, seconds, _, _, _, _ in batches[start:stop]]
        segment_gaps = [seconds for index, seconds in gaps if start <= index < stop]
        segments.append(
            {
                "batch_indices": f"{start}-{start + 31}",
                "count": len(values),
                "batch_sum_seconds": round(sum(values), 4),
                "batch_p50_seconds": round(statistics.median(values), 6)
                if values
                else None,
                "gap_sum_seconds": round(sum(segment_gaps), 4),
                "gap_p50_seconds": round(statistics.median(segment_gaps), 6)
                if segment_gaps
                else None,
                "gap_max_seconds": round(max(segment_gaps), 6)
                if segment_gaps
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
    boundary = next((end for token, _, _, end, _, _ in batches if token == 63), None)
    ready = stages["refresh_ready"]["first_t"]
    batch_sum = sum(seconds for _, seconds, _, _, _, _ in batches)
    batch_span = max(end for _, _, _, end, _, _ in batches) - min(
        start for _, _, start, _, _, _ in batches
    )
    return {
        "batch_event": batch_event,
        "batch_count": len(batches),
        "batch_sum_seconds": round(batch_sum, 4),
        "batch_span_seconds": round(batch_span, 4),
        "outside_batch_seconds": round(batch_span - batch_sum, 4),
        "largest_gaps": [
            {"before_batch_index": index, "seconds": round(seconds, 6)}
            for index, seconds in sorted(gaps, key=lambda row: row[1], reverse=True)[:5]
        ],
        "run_sum_seconds": (
            round(sum(run for _, _, _, _, run, _ in batches if run is not None), 4)
            if any(run is not None for _, _, _, _, run, _ in batches)
            else None
        ),
        "process_sum_seconds": (
            round(
                sum(
                    process for _, _, _, _, _, process in batches if process is not None
                ),
                4,
            )
            if any(process is not None for _, _, _, _, _, process in batches)
            else None
        ),
        "boundary_63_to_ready_seconds": round(ready - boundary, 4)
        if ready is not None and boundary is not None
        else None,
        "segments": segments,
        "stages": stages,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--batch-event",
        choices=("target_batch", "full_kv_batch"),
        default="target_batch",
    )
    args = parser.parse_args()
    print(json.dumps(summarize(args.log, batch_event=args.batch_event), indent=2))
