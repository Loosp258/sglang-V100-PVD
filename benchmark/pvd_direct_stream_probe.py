"""Measure Gateway first streamed output and completion for one fixed Prompt."""

import argparse
import hashlib
import json
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://10.10.1.2:8001/generate")
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=6)
    args = parser.parse_args()
    prompt = f"Case {args.case}. " + "EEFTRITON " * 430
    payload = json.dumps(
        {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.output_tokens,
                "ignore_eos": True,
            },
            "stream": True,
        }
    ).encode()
    request = urllib.request.Request(
        args.url, data=payload, headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    first = None
    events = []
    with urllib.request.urlopen(request, timeout=90) as response:
        status = response.status
        content_type = response.headers.get("Content-Type")
        for raw in response:
            if raw.startswith(b"data:") and raw.strip() != b"data: [DONE]":
                if first is None:
                    first = time.perf_counter() - started
                events.append(raw[5:].strip().decode())
    elapsed = time.perf_counter() - started
    final = json.loads(events[-1]) if events else {}
    print(
        json.dumps(
            {
                "case": args.case,
                "status": status,
                "content_type": content_type,
                "first_event_seconds": round(first, 3) if first is not None else None,
                "wall_seconds": round(elapsed, 3),
                "events": len(events),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "final_text_sha256": hashlib.sha256(
                    final.get("text", "").encode()
                ).hexdigest(),
                "completion_tokens": final.get("meta_info", {}).get(
                    "completion_tokens"
                ),
                "error": final.get("error"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
