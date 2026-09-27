"""Summarize formal peer progress during one request's private prediction."""

import argparse
import json
import re
from pathlib import Path


def summarize(path: Path, predicting_request: str, peer_request: str) -> dict:
    prediction_times = []
    peer_commits = []
    for line in path.read_text(errors="replace").splitlines():
        if "PVD timeline event=" not in line:
            continue
        timestamp = re.search(r"\bt=([0-9.]+)", line)
        if timestamp is None:
            continue
        when = float(timestamp.group(1))
        if (
            "event=prediction_step" in line
            and f"request_id={predicting_request} " in line
        ):
            prediction_times.append(when)
        elif "event=formal_decode_committed" in line:
            for request_id, committed in re.findall(r"\('([^']+)', (\d+)\)", line):
                if request_id == peer_request:
                    peer_commits.append((when, int(committed)))
    if not prediction_times:
        raise ValueError("no private prediction steps for the selected request")
    start, end = min(prediction_times), max(prediction_times)
    overlap = [(when, token) for when, token in peer_commits if start <= when <= end]
    intervals = [later[0] - earlier[0] for earlier, later in zip(overlap, overlap[1:])]
    return {
        "prediction_steps": len(prediction_times),
        "prediction_span_seconds": round(end - start, 4),
        "peer_commits_during_prediction": len(overlap),
        "peer_first_committed_token": overlap[0][1] if overlap else None,
        "peer_last_committed_token": overlap[-1][1] if overlap else None,
        "peer_max_commit_gap_seconds": round(max(intervals), 4) if intervals else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("predicting_request")
    parser.add_argument("peer_request")
    args = parser.parse_args()
    print(json.dumps(summarize(args.log, args.predicting_request, args.peer_request)))
