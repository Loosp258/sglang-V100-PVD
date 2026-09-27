"""Deterministic, bounded fact-retrieval quality probe through a PVD Gateway.

Run the same seed once with predictive sparse D and once with full-KV D.
Use ``--target-prompt-tokens`` for reproducible long-context profiles. Those
profiles are scaled from the existing 48-record, approximately 1413-token
Gateway measurement; the report always records the Gateway's actual token
count. Compare ``input_set_sha256`` across modes to confirm identical inputs.
Add ``--stream`` to collect SSE timing. SGLang emits cumulative stream text by
default; use ``--stream-text-format delta`` when the server enables
``--incremental-streaming-output``.
The report contains expected-answer checks and hashes, not raw model output.
It does not establish general generation quality or retrieval recall.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import re
import statistics
import threading
import time
import urllib.error
import urllib.request

ITEMS = (
    "amber lantern",
    "cobalt compass",
    "cedar journal",
    "silver kite",
    "violet ticket",
    "marble clock",
    "copper flute",
    "linen atlas",
    "glass teapot",
    "granite badge",
    "willow brush",
    "indigo ribbon",
)
CITIES = (
    "Oslo",
    "Quito",
    "Kyoto",
    "Nairobi",
    "Lima",
    "Tallinn",
    "Accra",
    "Porto",
    "Seoul",
    "Hanoi",
    "Riga",
    "Cusco",
)
TARGET_PROMPT_RECORDS = {4096: 139, 8192: 278, 16384: 557}
MAX_RECORDS = 640
MAX_OUTPUT_CHARS = 100_000
MAX_SSE_LINE_BYTES = 1 << 20
MAX_SSE_EVENTS_PER_TOKEN = 4


def records_for_target_prompt_tokens(target_prompt_tokens: int) -> int:
    """Map approximate token profiles to records using the 48/1413 baseline."""
    if (
        type(target_prompt_tokens) is not int
        or target_prompt_tokens not in TARGET_PROMPT_RECORDS
    ):
        raise ValueError("target prompt tokens must be one of 4096, 8192, or 16384")
    return TARGET_PROMPT_RECORDS[target_prompt_tokens]


def make_prompt(seed: str, case: int, *, records: int) -> tuple[str, str]:
    """Make unique codes and distractors; ask for a fact near a random position."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", seed):
        raise ValueError("seed must be a bounded filename-safe string")
    if (
        type(case) is not int
        or case < 0
        or type(records) is not int
        or not 16 <= records <= MAX_RECORDS
    ):
        raise ValueError("case and record count are out of bounds")
    rng = random.Random(f"{seed}:{case}")
    codes = rng.sample(range(10_000, 100_000), records)
    chosen = rng.randrange(records)
    lines = [
        "Each archive record below has a distinct five-digit access code. "
        "Use only the record ID asked for at the end."
    ]
    for index, code in enumerate(codes):
        item = ITEMS[(index * 7 + case) % len(ITEMS)]
        origin = CITIES[(index * 5 + case) % len(CITIES)]
        destination = CITIES[(index * 3 + case + 1) % len(CITIES)]
        lines.append(
            f"Record ID R{index:03d}: the {item} travels from {origin} "
            f"to {destination}; its access code is {code}."
        )
    lines.append(
        f"What is the five-digit access code for record ID R{chosen:03d}? "
        "Reply with that code first, without explaining your choice."
    )
    return "\n".join(lines), str(codes[chosen])


def _request(url: str, text: str, expected: str, max_tokens: int, timeout: float):
    body = json.dumps(
        {
            "text": text,
            "stream": False,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_tokens,
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
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"Gateway returned HTTP {response.status}")
        raw = response.read((1 << 20) + 1)
    if len(raw) > 1 << 20:
        raise ValueError("Gateway response exceeds 1 MiB")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise ValueError("Gateway returned no generated text")
    output = payload["text"]
    if len(output) > MAX_OUTPUT_CHARS:
        raise ValueError("Gateway generated unbounded text")
    first_code = re.search(r"(?<!\d)\d{5}(?!\d)", output)
    meta = payload.get("meta_info")
    if not isinstance(meta, dict) or type(meta.get("prompt_tokens")) is not int:
        raise ValueError("Gateway returned no token accounting")
    return {
        "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "expected_code": expected,
        "first_code": first_code.group() if first_code is not None else None,
        "first_code_matches": first_code is not None and first_code.group() == expected,
        "prompt_tokens": meta["prompt_tokens"],
        "completion_tokens": meta.get("completion_tokens"),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _decode_window_p50(gaps: list[float]) -> tuple[float | None, float | None]:
    """Compare early and late Decode away from an M64 refresh boundary.

    A 128-token stream has 127 gaps. Its first and last 32 gaps sit on
    opposite sides of the first M64 boundary, even when that boundary stalls.
    Short streams cannot support both windows and report no phase estimate.
    """
    if len(gaps) < 64:
        return None, None
    return _nearest_rank(gaps[:32], 0.5), _nearest_rank(gaps[-32:], 0.5)


def _observe_stream(
    response,
    started: float,
    text: str,
    expected: str,
    max_tokens: int,
    text_format: str,
):
    """Parse a bounded SGLang SSE stream and hash its final generated text.

    SGLang's default stream text is cumulative. With
    ``--incremental-streaming-output``, it is emitted as deltas instead.
    """
    content_type = response.headers.get("Content-Type", "")
    if not content_type.lower().startswith("text/event-stream"):
        raise ValueError(f"Gateway did not return SSE: {content_type}")
    if text_format not in ("cumulative", "delta"):
        raise ValueError("stream text format must be cumulative or delta")

    event_count = coalesced_tokens = previous_count = 0
    prompt_tokens = None
    token_times = []
    output_parts = []
    cumulative_text = ""
    output_characters = 0
    saw_text = False
    saw_done = False

    while True:
        raw = response.readline(MAX_SSE_LINE_BYTES + 1)
        if not raw:
            break
        if len(raw) > MAX_SSE_LINE_BYTES:
            raise ValueError("SSE line exceeds 1 MiB")
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

        delta = event.get("text")
        if delta is not None and not isinstance(delta, str):
            raise ValueError("SSE text must be a string or null")
        if delta is not None:
            saw_text = True
            if text_format == "cumulative":
                if not delta.startswith(cumulative_text):
                    raise ValueError("cumulative SSE text regressed or changed")
                cumulative_text = delta
                output_characters = len(cumulative_text)
            else:
                output_characters += len(delta)
                output_parts.append(delta)
            if output_characters > MAX_OUTPUT_CHARS:
                raise ValueError("Gateway generated unbounded text")

        meta = event.get("meta_info")
        count = meta.get("completion_tokens") if isinstance(meta, dict) else None
        if type(count) is not int or not previous_count <= count <= max_tokens:
            raise ValueError("SSE completion count is absent, regressed or excessive")
        prompt_count = meta.get("prompt_tokens")
        if prompt_count is not None:
            if type(prompt_count) is not int or prompt_count < 0:
                raise ValueError("SSE prompt token count is invalid")
            if prompt_tokens is not None and prompt_tokens != prompt_count:
                raise ValueError("SSE prompt token count changed during generation")
            prompt_tokens = prompt_count

        event_count += 1
        if event_count > max_tokens * MAX_SSE_EVENTS_PER_TOKEN + 16:
            raise ValueError("SSE event count exceeds bounded limit")
        step = count - previous_count
        if step:
            observed = time.perf_counter() - started
            token_times.extend([observed] * step)
            coalesced_tokens += max(0, step - 1)
        previous_count = count

    if not saw_done or len(token_times) != max_tokens:
        raise ValueError("SSE stream ended without all requested tokens and DONE")
    if prompt_tokens is None:
        raise ValueError("SSE stream returned no prompt token accounting")
    if not saw_text:
        raise ValueError("SSE stream returned no generated text")

    output = cumulative_text if text_format == "cumulative" else "".join(output_parts)
    if len(output) != output_characters:
        raise ValueError("SSE output text length changed while parsing")
    first_code = re.search(r"(?<!\d)\d{5}(?!\d)", output)
    gaps = [later - earlier for earlier, later in zip(token_times, token_times[1:])]
    early_p50, late_p50 = _decode_window_p50(gaps)
    wall_seconds = time.perf_counter() - started
    ttft_seconds = token_times[0]
    completion_seconds = token_times[-1]
    return {
        "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "expected_code": expected,
        "first_code": first_code.group() if first_code is not None else None,
        "first_code_matches": first_code is not None and first_code.group() == expected,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": len(token_times),
        "elapsed_seconds": wall_seconds,
        "ttft_seconds": ttft_seconds,
        "intertoken_gap_p50_seconds": _nearest_rank(gaps, 0.50),
        "intertoken_gap_p95_seconds": _nearest_rank(gaps, 0.95),
        "intertoken_gap_max_seconds": max(gaps) if gaps else None,
        "early_32_gap_p50_seconds": early_p50,
        "late_32_gap_p50_seconds": late_p50,
        "completion_seconds": completion_seconds,
        "decode_seconds": completion_seconds - ttft_seconds,
        "wall_seconds": wall_seconds,
        "sse_events": event_count,
        "coalesced_tokens": coalesced_tokens,
        "true_tpot_observable": coalesced_tokens == 0,
        "_intertoken_gaps_seconds": gaps,
    }


def _request_stream(
    url: str,
    text: str,
    expected: str,
    max_tokens: int,
    timeout: float,
    text_format: str,
):
    body = json.dumps(
        {
            "text": text,
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_tokens,
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
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"Gateway returned HTTP {response.status}")
        return _observe_stream(
            response, started, text, expected, max_tokens, text_format
        )


def collect(
    url: str,
    seed: str,
    *,
    cases: int,
    records: int | None = None,
    target_prompt_tokens: int | None = None,
    max_tokens: int,
    timeout: float,
    concurrency: int = 2,
    synchronized_start: bool = False,
    stream: bool = False,
    stream_text_format: str = "cumulative",
):
    if not url.startswith(("http://", "https://")) or not 1 <= cases <= 16:
        raise ValueError("bounded Gateway URL and case count required")
    if not 1 <= max_tokens <= 256 or not 0 < timeout <= 600:
        raise ValueError("bounded generation and timeout required")
    if target_prompt_tokens is not None:
        if records is not None:
            raise ValueError("choose records or target_prompt_tokens, not both")
        records = records_for_target_prompt_tokens(target_prompt_tokens)
        if max_tokens not in (128, 256):
            raise ValueError("long-prompt profiles require 128 or 256 output tokens")
    elif records is None:
        records = 48
    if type(concurrency) is not int or not 1 <= concurrency <= 4:
        raise ValueError("concurrency must be an integer in [1, 4]")
    if type(synchronized_start) is not bool:
        raise ValueError("synchronized_start must be a bool")
    if type(stream) is not bool:
        raise ValueError("stream must be a bool")
    if stream_text_format not in ("cumulative", "delta"):
        raise ValueError("stream text format must be cumulative or delta")
    prompts = [make_prompt(seed, i, records=records) for i in range(cases)]
    input_hashes = [
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text, _ in prompts
    ]
    input_set_sha256 = hashlib.sha256(
        "\n".join(input_hashes).encode("ascii")
    ).hexdigest()
    results, waves = [], []
    total_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for offset in range(0, cases, concurrency):
            wave = prompts[offset : offset + concurrency]
            gate = threading.Barrier(len(wave)) if synchronized_start else None

            def send(case_index, pair):
                if gate is not None:
                    try:
                        gate.wait(timeout=min(timeout, 30.0))
                    except threading.BrokenBarrierError as exc:
                        raise ValueError("synchronized wave could not start") from exc
                if stream:
                    result = _request_stream(
                        url,
                        pair[0],
                        pair[1],
                        max_tokens,
                        timeout,
                        stream_text_format,
                    )
                else:
                    result = _request(url, pair[0], pair[1], max_tokens, timeout)
                result["case"] = case_index
                return result

            wave_started = time.perf_counter()
            futures = [
                pool.submit(send, offset + index, pair)
                for index, pair in enumerate(wave)
            ]
            results.extend(future.result() for future in futures)
            waves.append(
                {
                    "requests": len(wave),
                    "elapsed_seconds": time.perf_counter() - wave_started,
                }
            )
    stream_metrics = None
    if stream:
        observed_gaps = []
        for item in results:
            observed_gaps.extend(item.pop("_intertoken_gaps_seconds"))
        ttfts = [item["ttft_seconds"] for item in results]
        completion_times = [item["completion_seconds"] for item in results]
        wall_times = [item["wall_seconds"] for item in results]
        stream_metrics = {
            "requests": len(results),
            "ttft_median_seconds": statistics.median(ttfts),
            "ttft_p95_seconds": _nearest_rank(ttfts, 0.95),
            "intertoken_gap_p50_seconds": _nearest_rank(observed_gaps, 0.50),
            "intertoken_gap_p95_seconds": _nearest_rank(observed_gaps, 0.95),
            "intertoken_gap_max_seconds": max(observed_gaps) if observed_gaps else None,
            "completion_median_seconds": statistics.median(completion_times),
            "completion_p95_seconds": _nearest_rank(completion_times, 0.95),
            "wall_median_seconds": statistics.median(wall_times),
            "wall_p95_seconds": _nearest_rank(wall_times, 0.95),
            "true_tpot_requests": sum(item["true_tpot_observable"] for item in results),
            "coalesced_tokens": sum(item["coalesced_tokens"] for item in results),
        }
    report = {
        "schema": "pvd.fact_recall.v1",
        "seed": seed,
        "cases": cases,
        "records_per_case": records,
        "target_prompt_tokens": target_prompt_tokens,
        "input_set_sha256": input_set_sha256,
        "prompt_tokens_min": min(item["prompt_tokens"] for item in results),
        "prompt_tokens_median": statistics.median(
            item["prompt_tokens"] for item in results
        ),
        "prompt_tokens_max": max(item["prompt_tokens"] for item in results),
        "correct": sum(item["first_code_matches"] for item in results),
        "median_elapsed_seconds": statistics.median(
            item["elapsed_seconds"] for item in results
        ),
        "total_elapsed_seconds": time.perf_counter() - total_started,
        "concurrency": concurrency,
        "synchronized_start": synchronized_start,
        "waves": waves,
        "results": results,
        "mode_verified_by_script": False,
    }
    if stream:
        report["stream"] = True
        report["stream_text_format"] = stream_text_format
        report["stream_metrics"] = stream_metrics
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--cases", type=int, default=6)
    size = parser.add_mutually_exclusive_group()
    size.add_argument(
        "--records",
        type=int,
        help="records per case (default: 48; legacy workload)",
    )
    size.add_argument(
        "--target-prompt-tokens",
        type=int,
        choices=tuple(TARGET_PROMPT_RECORDS),
        help=(
            "approximate long-prompt profile; actual Gateway token counts are reported"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="output limit (default: 20 for legacy runs, 128 for long-prompt profiles)",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--synchronized-start", action="store_true")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="opt in to SSE TTFT, inter-token, completion, and wall timing",
    )
    parser.add_argument(
        "--stream-text-format",
        choices=("cumulative", "delta"),
        default="cumulative",
        help="Gateway SSE text framing (default matches SGLang server default)",
    )
    args = parser.parse_args()
    max_tokens = args.max_new_tokens
    if max_tokens is None:
        max_tokens = 128 if args.target_prompt_tokens is not None else 20
    if args.target_prompt_tokens is not None and max_tokens not in (128, 256):
        parser.error("--target-prompt-tokens requires --max-new-tokens 128 or 256")
    try:
        result = collect(
            args.gateway_url,
            args.seed,
            cases=args.cases,
            records=args.records,
            target_prompt_tokens=args.target_prompt_tokens,
            max_tokens=max_tokens,
            timeout=args.timeout_seconds,
            concurrency=args.concurrency,
            synchronized_start=args.synchronized_start,
            stream=args.stream,
            stream_text_format=args.stream_text_format,
        )
    except (ValueError, OSError, urllib.error.URLError) as exc:
        parser.exit(1, f"fact-recall probe failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
