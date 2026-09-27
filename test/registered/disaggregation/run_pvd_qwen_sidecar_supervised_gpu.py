"""Run one real GPU0 sidecar request under D-parent ownership; no serving."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from sglang.srt.disaggregation.pvd.prediction import CommittedPrefix
from sglang.srt.disaggregation.pvd.probe_lane_identity import checkpoint_identity
from sglang.srt.disaggregation.pvd.probe_lane_protocol import ProbeLaneTicket
from sglang.srt.disaggregation.pvd.probe_lane_sidecar_process import (
    launch_probe_sidecar,
)
from sglang.srt.disaggregation.pvd.probe_search import ProbeWindow
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--prefix-text", default="The quick brown fox jumps")
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer

    model_path = Path(args.model_path).resolve(strict=True)
    draft_path = Path(args.draft_model_path).resolve(strict=True)
    config = json.loads((model_path / "config.json").read_text())
    layers = config["num_hidden_layers"]
    heads = config["num_attention_heads"]
    head_dim = config.get("head_dim", config["hidden_size"] // heads)
    if not 0 < layers <= 128 or not 0 < heads <= 128 or not 0 < head_dim <= 256:
        raise ValueError("bounded Qwen2 architecture required")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokens = tuple(tokenizer.encode(args.prefix_text, add_special_tokens=True))
    if not 0 < len(tokens) <= 126:
        raise ValueError("bounded nonempty prefix required")
    checkpoint = checkpoint_identity(model_path)
    budget = TransferBudget(16 << 20, 1)
    script = Path(__file__).parent / "run_pvd_qwen_sidecar_gpu.py"
    sidecar_args = [
        "--model-path",
        str(model_path),
        "--draft-model-path",
        str(draft_path),
        "--context-length",
        "128",
        "--max-total-tokens",
        "128",
        "--predict-tokens",
        "2",
        "--draft-mem-fraction-static",
        "0.1",
        "--draft-scratch-budget-bytes",
        str(256 << 20),
        "--draft-persistent-budget-bytes",
        str(2 << 30),
        "--draft-transient-bytes-bound",
        str(128 << 20),
        "--probe-budget-bytes",
        str(256 << 20),
        "--probe-transient-bytes-bound",
        str(128 << 20),
        "--reply-budget-bytes",
        str(16 << 20),
    ]
    started = time.monotonic()
    owner = launch_probe_sidecar(
        script,
        sidecar_args,
        checkpoint=checkpoint,
        target_model_id=args.target_model_id,
        reply_budget=budget,
        startup_timeout=180,
        directory_parent="/tmp",
    )
    ready_seconds = time.monotonic() - started
    try:
        prefix = CommittedPrefix("supervised-smoke", tokens, 0, "v1")
        window = ProbeWindow(
            "supervised-incarnation",
            "supervised-operation",
            "supervised-entry",
            prefix,
            2,
            (len(tokens), len(tokens) + 1),
        )
        ticket = ProbeLaneTicket.issue(
            window,
            target_model_id=args.target_model_id,
            weights_sha256=checkpoint.weights_sha256,
            tokenizer_sha256=checkpoint.tokenizer_sha256,
            layers=tuple(range(layers)),
            head_start=0,
            head_count=heads,
            head_dim=head_dim,
            max_reply_bytes=layers * 2 * heads * head_dim * 4,
            deadline_monotonic=time.monotonic() + 90,
        )

        async def request():
            async with owner.client.request(ticket) as queries:
                assert len(queries) == layers
                for query in queries:
                    assert query.vectors.device.type == "cpu"
                    assert query.vectors.shape == (2, heads, head_dim)
                    assert query.positional_encoding == "rope_applied"
            return len(queries)

        request_start = time.monotonic()
        returned_layers = asyncio.run(request())
        request_seconds = time.monotonic() - request_start
        assert budget.snapshot()["used_staging_bytes"] == 0
        owner.check_alive()
        result = {
            "schema": "pvd.probe.supervised.real-gpu.v1",
            "status": "passed",
            "model_layers": returned_layers,
            "ready_seconds": ready_seconds,
            "request_seconds": request_seconds,
            "weights_sha256": checkpoint.weights_sha256,
            "tokenizer_sha256": checkpoint.tokenizer_sha256,
            "sidecar_pid": owner.process.pid,
            "client_pid_verified": True,
            "reply_budget_refunded": True,
            "scheduler_serving_activated": False,
        }
    finally:
        owner.close()
    assert not owner.socket_dir.exists()
    result["sidecar_stopped_cleanly"] = True
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
