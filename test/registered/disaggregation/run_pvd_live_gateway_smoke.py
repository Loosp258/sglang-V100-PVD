"""Send one bounded live PVD Gateway request and print its full response.

Run only against an intentionally started three-node experiment. This script
does not start servers, infer transport success from HTTP alone, or benchmark.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--min-completion-tokens", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args(argv)
    if not (
        1 <= args.min_completion_tokens <= args.max_new_tokens <= 32
        and 0 < args.timeout_seconds <= 600
    ):
        parser.error("smoke bounds exceeded")
    marker = f"PVD_NATIVE_SMOKE_{uuid.uuid4().hex[:12]}"
    body = {
        "text": f"{marker}: Explain GPU RDMA in one short sentence.",
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
    try:
        with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
            status = response.status
            payload = response.read(1 << 20).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        payload = exc.read(1 << 20).decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 -- transport failure is a failed smoke
        print(json.dumps({"marker": marker, "status": "failed", "reason": str(exc)}))
        return 1
    try:
        result = json.loads(payload)
    except json.JSONDecodeError:
        result = payload
    metadata = result.get("meta_info") if isinstance(result, dict) else None
    completion_tokens = (
        metadata.get("completion_tokens") if isinstance(metadata, dict) else None
    )
    ok = (
        status == 200
        and isinstance(result, dict)
        and bool(result.get("text"))
        and isinstance(completion_tokens, int)
        and completion_tokens >= args.min_completion_tokens
    )
    print(
        json.dumps(
            {
                "marker": marker,
                "status": "passed" if ok else "failed",
                "http_status": status,
                "completion_tokens": completion_tokens,
                "response": result,
                "transport_validated_by_http_alone": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
