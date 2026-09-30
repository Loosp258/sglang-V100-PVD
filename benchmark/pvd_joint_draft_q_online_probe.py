"""Fixed matched Prompts, streaming response timing and answer text evidence."""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def request(url, name, text, output_tokens):
    payload = json.dumps({"text": text, "stream": True, "sampling_params": {
        "temperature": 0, "max_new_tokens": output_tokens, "ignore_eos": True}}).encode()
    started_unix, started = time.time(), time.perf_counter()
    first = None
    events = []
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        status = response.status
        for line in response:
            if line.startswith(b"data:") and line.strip() != b"data: [DONE]":
                if first is None:
                    first = time.perf_counter() - started
                events.append(json.loads(line[5:].strip()))
    total = time.perf_counter() - started
    final = events[-1] if events else {}
    meta = final.get("meta_info", {})
    return {"case": name, "started_unix": started_unix, "status": status,
            "first_event_seconds": first, "wall_seconds": total,
            "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
            "output_text": final.get("text", ""),
            "final_text_sha256": hashlib.sha256(final.get("text", "").encode()).hexdigest(),
            "request_id": meta.get("id"), "error": final.get("error"),
            "events": len(events)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://10.10.1.2:8001/generate")
    parser.add_argument("--arm", choices=("joint", "target"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--warmup-only", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    notes = (args.source_root / "benchmark/react/hotpotqa_100.jsonl").read_text().splitlines()
    natural = "\n".join(notes[80:])
    cases = [
        ("warmup", "Case 599. " + "EEFTRITON " * 430, 6),
        ("repeat-short", "Case 510. " + "EEFTRITON " * 200, 16),
        ("repeat-long", "Case 511. " + "EEFTRITON " * 430, 16),
        ("natural-short", tokenizer.decode(tokenizer.encode(natural,
            add_special_tokens=False)[:1009]), 16),
        ("natural-long", tokenizer.decode(tokenizer.encode(natural,
            add_special_tokens=False)[:2155]), 16),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for name, text, count in cases[:1] if args.warmup_only else cases:
            row = request(args.url, name, text, count)
            row["arm"] = args.arm
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            destination.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if row["status"] != 200 or row["error"] or row["completion_tokens"] != count:
                raise RuntimeError(f"invalid complete response: {row}")


if __name__ == "__main__":
    main()
