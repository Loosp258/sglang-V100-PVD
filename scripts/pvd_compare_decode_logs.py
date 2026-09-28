"""Summarize sequential PVD/full-KV Decode requests from a D server log.

The requested run is selected by the reset of its single-request formal batch
counter. Batch and gap totals partition the D formal-Decode window. Refresh
stage durations are concurrent with that window and must not be added to it.
"""

import argparse
import json
import re
import statistics
from pathlib import Path


def field(line, name):
    match = re.search(rf"\b{name}=([0-9.]+)", line)
    return float(match.group(1)) if match else None


def summarize(path, event, run_index):
    lines = Path(path).read_text(errors="replace").splitlines()
    batches = []
    for index, line in enumerate(lines):
        if f"PVD timeline event={event} members=1 " not in line:
            continue
        match = re.search(r"committed_tokens=\((\d+),\)", line)
        if match is None:
            continue
        batches.append(
            {
                "line": index,
                "token": int(match.group(1)),
                "start": field(line, "t_start"),
                "end": field(line, "t_end"),
                "seconds": field(line, "seconds"),
                "run_seconds": field(line, "run_seconds"),
                "process_seconds": field(line, "process_seconds"),
            }
        )
    groups = []
    for batch in batches:
        if not groups or batch["token"] <= groups[-1][-1]["token"]:
            groups.append([])
        groups[-1].append(batch)
    if not 0 <= run_index < len(groups):
        raise ValueError(f"run_index {run_index} outside {len(groups)} runs")
    run = groups[run_index]
    # A later pair may have single-member batches whose token counters reset
    # mid-run; the caller should select only isolated single-request runs.
    if any(batch["start"] is None or batch["end"] is None for batch in run):
        raise ValueError("incomplete formal batch timestamps")
    gaps = [
        run[index]["start"] - run[index - 1]["end"]
        for index in range(1, len(run))
    ]
    segments = []
    for start in range(0, len(run), 32):
        stop = min(start + 32, len(run))
        segment = run[start:stop]
        segment_gaps = gaps[max(start - 1, 0) : stop - 1]
        segments.append(
            {
                "batch_indices": f"{start}-{stop - 1}",
                "count": len(segment),
                "batch_seconds": round(sum(batch["seconds"] for batch in segment), 4),
                "batch_p50_seconds": round(
                    statistics.median(batch["seconds"] for batch in segment), 6
                ),
                "gap_seconds": round(sum(segment_gaps), 4),
                "gap_max_seconds": round(max(segment_gaps), 6)
                if segment_gaps
                else None,
            }
        )
    line_start = run[0]["line"]
    line_end = run[-1]["line"]
    events = []
    refreshes = []
    for line in lines[line_start : line_end + 1]:
        if "PVD timeline event=" in line and any(
            f"event={name}" in line
            for name in (
                "refresh_scheduled",
                "refresh_ready",
                "installed",
                "search_http",
            )
        ):
            name = re.search(r"event=([a-z_]+)", line).group(1)
            events.append(
                {
                    "event": name,
                    "boundary": field(line, "boundary"),
                    "committed_tokens": field(line, "committed_tokens"),
                    "t": field(line, "t"),
                    "start": field(line, "t_start"),
                    "end": field(line, "t_end"),
                    "seconds": field(line, "seconds"),
                }
            )
        if "PVD refresh ready:" in line:
            refreshes.append(
                {
                    name: field(line, name)
                    for name in (
                        "capture_seconds",
                        "search_seconds",
                        "union_seconds",
                        "delivery_seconds",
                        "total_seconds",
                    )
                }
            )
    batch_seconds = sum(batch["seconds"] for batch in run)
    span = run[-1]["end"] - run[0]["start"]
    by_token = {batch["token"]: batch for batch in run}
    refresh_windows = []
    scheduled = [event for event in events if event["event"] == "refresh_scheduled"]
    ready = [event for event in events if event["event"] == "refresh_ready"]
    for index, event in enumerate(scheduled):
        lead = int(event["committed_tokens"])
        boundary = int(event["boundary"])
        before_lead, after_lead = by_token.get(lead - 1), by_token.get(lead)
        before_boundary, after_boundary = (
            by_token.get(boundary - 1),
            by_token.get(boundary),
        )
        ready_at = ready[index]["t"] if index < len(ready) else None
        refresh_windows.append(
            {
                "boundary": boundary,
                "pause_after_lead_seconds": round(
                    after_lead["start"] - before_lead["end"], 4
                )
                if before_lead and after_lead
                else None,
                "ready_before_boundary_seconds": round(
                    before_boundary["end"] - ready_at, 4
                )
                if before_boundary and ready_at is not None
                else None,
                "boundary_gap_seconds": round(
                    after_boundary["start"] - before_boundary["end"], 4
                )
                if before_boundary and after_boundary
                else None,
            }
        )
    result = {
        "event": event,
        "run_index": run_index,
        "batch_count": len(run),
        "first_token_counter": run[0]["token"],
        "last_token_counter": run[-1]["token"],
        "formal_span_seconds": round(span, 4),
        "formal_batch_seconds": round(batch_seconds, 4),
        "between_batch_seconds": round(sum(gaps), 4),
        "run_batch_seconds": (
            round(sum(batch["run_seconds"] for batch in run), 4)
            if all(batch["run_seconds"] is not None for batch in run)
            else None
        ),
        "process_result_seconds": (
            round(sum(batch["process_seconds"] for batch in run), 4)
            if all(batch["process_seconds"] is not None for batch in run)
            else None
        ),
        "segments": segments,
        "refreshes": refreshes,
        "refresh_windows": refresh_windows,
        "events": events,
    }
    if abs(result["formal_span_seconds"] - batch_seconds - sum(gaps)) > 0.02:
        raise ValueError("formal batch and gap accounting does not close")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--event", choices=("target_batch", "full_kv_batch"), required=True)
    parser.add_argument("--run-index", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(summarize(args.log, args.event, args.run_index), indent=2))


if __name__ == "__main__":
    main()
