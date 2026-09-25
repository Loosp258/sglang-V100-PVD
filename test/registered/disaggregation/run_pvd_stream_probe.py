"""Observe bounded live streaming token timing through a PVD Gateway.

Only one request is sent. Coalesced or incomplete SSE events are reported,
not silently treated as true per-token TPOT. Transport and mode are not
inferred from HTTP alone.
"""

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request


def _observe(response, started, max_new_tokens):
    content_type = response.headers.get("Content-Type", "")
    if not content_type.lower().startswith("text/event-stream"):
        raise ValueError(f"Gateway did not return SSE: {content_type}")
    token_times = []
    event_count = coalesced_tokens = previous_count = 0
    saw_done = False
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            saw_done = True
            break
        event = json.loads(data)
        if not isinstance(event, dict):
            raise ValueError("non-object SSE event")
        meta = event.get("meta_info")
        count = meta.get("completion_tokens") if isinstance(meta, dict) else None
        if type(count) is not int or not previous_count <= count <= max_new_tokens:
            raise ValueError("SSE completion count is absent, regressed or excessive")
        event_count += 1
        step = count - previous_count
        if step:
            observed = time.perf_counter() - started
            token_times.extend([observed] * step)
            coalesced_tokens += max(0, step - 1)
        previous_count = count
    if not saw_done or len(token_times) != max_new_tokens:
        raise ValueError("SSE stream ended without all requested tokens and DONE")
    gaps = [later - earlier for earlier, later in zip(token_times, token_times[1:])]
    return {
        "tokens": len(token_times),
        "sse_events": event_count,
        "coalesced_tokens": coalesced_tokens,
        "ttft_seconds": token_times[0],
        "client_elapsed_seconds": time.perf_counter() - started,
        "median_observed_intertoken_seconds": statistics.median(gaps) if gaps else None,
        "max_observed_intertoken_seconds": max(gaps) if gaps else None,
        "token_observation_times_seconds": token_times,
        "true_tpot_observable": coalesced_tokens == 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    args = parser.parse_args(argv)
    if (
        not args.gateway_url.startswith(("http://", "https://"))
        or not args.text.strip()
        or len(args.text) > 200_000
        or not 1 <= args.max_new_tokens <= 256
        or not math.isfinite(args.timeout_seconds)
        or not 0 < args.timeout_seconds <= 600
    ):
        parser.error("bounded URL, prompt, token count and timeout required")
    body = {
        "text": args.text,
        "stream": True,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": True,
        },
    }
    request = urllib.request.Request(
        args.gateway_url.rstrip("/") + "/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
            if response.status != 200:
                raise ValueError(f"Gateway returned HTTP {response.status}")
            result = _observe(response, started, args.max_new_tokens)
    except (ValueError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"stream timing unavailable: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
