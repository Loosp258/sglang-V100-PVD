"""Bounded live PVD Gateway load probe with honest SSE timing.

This is a serving-path measurement, not proof of CAGRA selection, RDMA
completion, output quality, or a performance improvement over another mode.
Inspect V/D logs and service state separately for those claims.
"""

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _observe(response, started, expected_tokens):
    content_type = response.headers.get("Content-Type", "")
    if not content_type.lower().startswith("text/event-stream"):
        raise ValueError(f"Gateway did not return SSE: {content_type}")
    times = []
    prompt_tokens = None
    events = coalesced = previous = 0
    done = False
    last_finish_type = None
    last_text = None
    for raw in response:
        if len(raw) > 1 << 20:
            raise ValueError("oversized SSE line")
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        payload = json.loads(data)
        if isinstance(payload, dict) and "text" in payload:
            if not isinstance(payload["text"], str) or len(payload["text"]) > 100_000:
                raise ValueError("SSE output text is invalid or oversized")
            last_text = payload["text"]
        meta = payload.get("meta_info") if isinstance(payload, dict) else None
        finish = meta.get("finish_reason") if isinstance(meta, dict) else None
        last_finish_type = finish.get("type") if isinstance(finish, dict) else finish
        count = meta.get("completion_tokens") if isinstance(meta, dict) else None
        length = meta.get("prompt_tokens") if isinstance(meta, dict) else None
        if type(count) is not int or not previous <= count <= expected_tokens:
            raise ValueError("SSE completion count is absent, regressed or excessive")
        if type(length) is not int or length <= 0:
            raise ValueError("SSE prompt token count is absent or invalid")
        if prompt_tokens is not None and prompt_tokens != length:
            raise ValueError("SSE prompt token count changed")
        prompt_tokens = length
        events += 1
        step = count - previous
        if step:
            now = time.perf_counter() - started
            times.extend([now] * step)
            coalesced += max(0, step - 1)
        previous = count
    if not done or len(times) != expected_tokens:
        raise ValueError(
            "SSE stream ended without every requested token and DONE: "
            f"done={done} received={len(times)}/{expected_tokens} "
            f"events={events} finish_type={last_finish_type!r} "
            f"elapsed_seconds={time.perf_counter() - started:.3f}"
        )
    gaps = [b - a for a, b in itertools.pairwise(times)]
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": len(times),
        "sse_events": events,
        "coalesced_tokens": coalesced,
        "ttft_seconds": times[0],
        "elapsed_seconds": time.perf_counter() - started,
        "median_observed_gap_seconds": statistics.median(gaps) if gaps else None,
        "max_observed_gap_seconds": max(gaps) if gaps else None,
        "true_tpot_observable": coalesced == 0,
        "output_sha256": (
            hashlib.sha256(last_text.encode("utf-8")).hexdigest()
            if last_text is not None
            else None
        ),
    }


def _request(url, text, expected_tokens, timeout, barrier):
    body = json.dumps(
        {
            "text": text,
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": expected_tokens,
                "ignore_eos": True,
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    barrier.wait(timeout=timeout)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ValueError(f"Gateway returned HTTP {response.status}")
            result = _observe(response, started, expected_tokens)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"Gateway returned HTTP {exc.code}") from exc
    result["input_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return started, result


def collect(args):
    fixed_prefix = getattr(args, "fixed_prefix", False)
    replay_seed = getattr(args, "replay_seed", None)
    if (
        not args.gateway_url.startswith(("http://", "https://"))
        or not 1 <= args.clients <= 4
        or not 1 <= args.rounds <= 5
        or not 1 <= args.repetitions <= 300
        or not 1 <= args.max_new_tokens <= 32
        or not 0 < args.timeout_seconds <= 600
        or not 1 <= len(args.sentence) <= 500
        or len(args.sentence) * args.repetitions > 100_000
        or not 1 <= args.min_prompt_tokens <= args.max_prompt_tokens <= 8192
        or (fixed_prefix and (args.clients != 1 or args.rounds != 1))
        or (
            replay_seed is not None
            and (
                not isinstance(replay_seed, str)
                or re.fullmatch(r"[A-Za-z0-9_-]{1,32}", replay_seed) is None
                or fixed_prefix
            )
        )
    ):
        raise ValueError("bounded URL, clients, rounds, prompt and timeout required")
    run_id = replay_seed if replay_seed is not None else uuid.uuid4().hex[:12]
    rounds = []
    for round_number in range(args.rounds):
        barrier = threading.Barrier(args.clients)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.clients) as pool:
            jobs = [
                pool.submit(
                    _request,
                    args.gateway_url,
                    (
                        "PVD_LOAD_FIXED: "
                        if fixed_prefix
                        else f"PVD_LOAD_{run_id}_{round_number}_{client}: "
                    )
                    + (args.sentence + " ") * args.repetitions,
                    args.max_new_tokens,
                    args.timeout_seconds,
                    barrier,
                )
                for client in range(args.clients)
            ]
            observations = [job.result() for job in jobs]
        starts = [started for started, _ in observations]
        results = [result for _, result in observations]
        for result in results:
            if (
                not args.min_prompt_tokens
                <= result["prompt_tokens"]
                <= args.max_prompt_tokens
            ):
                raise ValueError(
                    f"actual Prompt {result['prompt_tokens']} tokens outside requested range"
                )
        wall = max(
            started + result["elapsed_seconds"] for started, result in observations
        ) - min(starts)
        rounds.append(
            {
                "round": round_number,
                "wall_seconds": wall,
                "generated_tokens_per_second": sum(
                    result["completion_tokens"] for result in results
                )
                / wall,
                "requests": results,
            }
        )
    all_results = [result for item in rounds for result in item["requests"]]
    return {
        "schema": "pvd.live_load.v1",
        "run_id": run_id,
        "fixed_prefix": fixed_prefix,
        "replay_seed": replay_seed,
        "mode_verified_by_script": False,
        "clients": args.clients,
        "rounds": rounds,
        "all_complete": True,
        "all_true_tpot_observable": all(
            result["true_tpot_observable"] for result in all_results
        ),
        "latency_p50_seconds": statistics.median(
            result["elapsed_seconds"] for result in all_results
        ),
        "latency_p95_nearest_rank_seconds": _percentile(
            [result["elapsed_seconds"] for result in all_results], 0.95
        ),
        "ttft_p50_seconds": statistics.median(
            result["ttft_seconds"] for result in all_results
        ),
        "ttft_p95_nearest_rank_seconds": _percentile(
            [result["ttft_seconds"] for result in all_results], 0.95
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--clients", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--sentence", default="Explain GPU RDMA in one short sentence.")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--min-prompt-tokens", type=int, default=1)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument(
        "--fixed-prefix",
        action="store_true",
        help="Use the same Prompt across single-client, single-round A/B runs",
    )
    parser.add_argument(
        "--replay-seed",
        help="Repeat exact per-client Prompts across separately launched A/B runs",
    )
    args = parser.parse_args(argv)
    try:
        result = collect(args)
    except (
        ValueError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        threading.BrokenBarrierError,
    ) as exc:
        parser.exit(1, f"live load failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.exit(main())
