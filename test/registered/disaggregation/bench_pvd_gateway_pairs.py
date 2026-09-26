"""Bounded paired-request wall-clock A/B for an already-running PVD gateway.

Run the identical invocation before and after restarting only D with an
alternative serving-limits file. This is client-side latency, not a kernel or
network-only benchmark. The script starts no services and changes no server.
"""

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.request


def request(url, prompt, output_tokens, timeout):
    body = json.dumps(
        {
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": output_tokens},
        }
    ).encode()
    call = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(call, timeout=timeout) as response:
        status = response.status
        result = json.load(response)
    if status != 200 or not isinstance(result, dict):
        raise RuntimeError(f"unexpected Gateway response status={status}")
    text = result.get("text")
    if not isinstance(text, str):
        raise RuntimeError("Gateway response has no text")
    meta = result.get("meta_info", {})
    if not isinstance(meta, dict):
        meta = {}
    return {
        "status": status,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "output_chars": len(text),
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://10.0.1.2:8001/generate")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    if not (1 <= args.rounds <= 20 and 1 <= args.repetitions <= 1000):
        parser.error("bounded rounds and prompt repetitions required")
    if not (1 <= args.output_tokens <= 128 and 1 <= args.timeout <= 600):
        parser.error("bounded output length and timeout required")
    prompts = ["EEFTRITON " * args.repetitions + f" Case {i}." for i in (0, 1)]
    rounds = []
    for _ in range(args.rounds):
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(request, args.url, p, args.output_tokens, args.timeout)
                for p in prompts
            ]
            responses = [future.result(timeout=args.timeout + 5) for future in futures]
        rounds.append(
            {
                "wall_seconds": round(time.perf_counter() - started, 3),
                "responses": responses,
            }
        )
    print(
        json.dumps(
            {
                "schema": "pvd-gateway-pair-ab-v1",
                "url": args.url,
                "repetitions": args.repetitions,
                "output_tokens_requested": args.output_tokens,
                "rounds": rounds,
                "median_wall_seconds": round(
                    statistics.median(row["wall_seconds"] for row in rounds), 3
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
