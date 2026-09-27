"""Summarize concurrent CUDA kernel intervals in an Nsight SQLite export."""

import argparse
import json
import sqlite3
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--streams", nargs=2, type=int, required=True)
    args = parser.parse_args()

    connection = sqlite3.connect(args.sqlite)
    stream_a, stream_b = args.streams
    strings = dict(connection.execute("SELECT id, value FROM StringIds"))
    intervals = {}
    for stream in args.streams:
        intervals[stream] = connection.execute(
            "SELECT start, end, shortName, deviceId, contextId, globalPid "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId = ? ORDER BY start",
            (stream,),
        ).fetchall()

    a, b = intervals[stream_a], intervals[stream_b]
    i = j = count = duration_ns = 0
    examples = []
    while i < len(a) and j < len(b):
        left = max(a[i][0], b[j][0])
        right = min(a[i][1], b[j][1])
        if right > left:
            count += 1
            duration_ns += right - left
            if len(examples) < 8:
                examples.append(
                    {
                        "start_ns": left,
                        "duration_us": round((right - left) / 1e3, 3),
                        "a": strings.get(a[i][2], str(a[i][2])),
                        "b": strings.get(b[j][2], str(b[j][2])),
                    }
                )
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1

    streams = {}
    for stream in args.streams:
        rows = intervals[stream]
        launch_threads = connection.execute(
            "SELECT r.globalTid, COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r "
            "ON k.correlationId = r.correlationId "
            "WHERE k.streamId = ? GROUP BY r.globalTid ORDER BY COUNT(*) DESC",
            (stream,),
        ).fetchall()
        streams[str(stream)] = {
            "kernels": len(rows),
            "first_start_ns": rows[0][0],
            "last_end_ns": rows[-1][1],
            "device_context_process": list({(r[3], r[4], r[5]) for r in rows}),
            "launch_threads": launch_threads,
            "top_kernels": Counter(strings.get(r[2], str(r[2])) for r in rows).most_common(8),
        }
    print(
        json.dumps(
            {
                "streams": streams,
                "overlapping_kernel_pairs": count,
                "overlapping_kernel_duration_ms": round(duration_ns / 1e6, 3),
                "examples": examples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
