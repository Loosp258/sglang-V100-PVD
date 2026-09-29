"""Bounded identical-schedule Gateway probe for complete vs chunked PVD upload."""

import argparse
import hashlib
import json
import statistics
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://10.10.1.2:8001/generate")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--trial-offset", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=430)
    parser.add_argument("--output-tokens", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if not (1 <= args.rounds <= 5 and 1 <= args.repetitions <= 450):
        parser.error("probe bounds exceeded")
    if not 0 <= args.trial_offset <= 100:
        parser.error("trial offset exceeded")
    if not (1 <= args.output_tokens <= 8 and 0 < args.timeout <= 600):
        parser.error("output/timeout bounds exceeded")
    rows = []
    for trial in range(args.trial_offset, args.trial_offset + args.rounds):
        prompt = f"Case {trial}. " + "EEFTRITON " * args.repetitions
        data = json.dumps({
            "text": prompt,
            "sampling_params": {
                "temperature": 0, "max_new_tokens": args.output_tokens,
                "ignore_eos": True,
            },
        }).encode()
        request = urllib.request.Request(
            args.url, data=data, headers={"Content-Type": "application/json"},
        )
        started_unix = time.time()
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                status = response.status
                result = json.load(response)
        except Exception as exc:
            rows.append({"trial": trial, "error": repr(exc)})
            print(json.dumps(rows[-1]), flush=True)
            continue
        metadata = result.get("meta_info", {}) if isinstance(result, dict) else {}
        row = {
            "trial": trial, "wall_seconds": round(time.perf_counter() - started, 3),
            "started_unix": started_unix, "finished_unix": time.time(),
            "status": status,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_tokens": metadata.get("prompt_tokens"),
            "completion_tokens": metadata.get("completion_tokens"),
            "output_sha256": hashlib.sha256(result.get("text", "").encode()).hexdigest(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    succeeded = [row["wall_seconds"] for row in rows if "wall_seconds" in row]
    print(json.dumps({
        "schema": "pvd-chunked-online-probe-v1",
        "repetitions": args.repetitions, "requested_output_tokens": args.output_tokens,
        "median_wall_seconds": statistics.median(succeeded) if succeeded else None,
        "rows": rows,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
