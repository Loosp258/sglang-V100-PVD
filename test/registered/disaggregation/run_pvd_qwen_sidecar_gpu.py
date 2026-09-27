"""Experimental GPU0 Qwen2 target+draft Unix sidecar; not a serving switch.

Run only in an isolated process. The D Scheduler PID must be known before
launch. No automatic fallback or model download is performed.
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import threading
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--socket-dir", required=True)
    parser.add_argument("--expected-d-pid", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--max-total-tokens", type=int, required=True)
    parser.add_argument("--predict-tokens", type=int, required=True)
    parser.add_argument("--draft-mem-fraction-static", type=float, required=True)
    parser.add_argument("--draft-scratch-budget-bytes", type=int, required=True)
    parser.add_argument("--draft-persistent-budget-bytes", type=int, required=True)
    parser.add_argument("--draft-prefix-cache-budget-bytes", type=int, default=0)
    parser.add_argument("--draft-transient-bytes-bound", type=int, required=True)
    parser.add_argument("--probe-budget-bytes", type=int, required=True)
    parser.add_argument("--probe-prefix-cache-budget-bytes", type=int, default=0)
    parser.add_argument("--probe-transient-bytes-bound", type=int, required=True)
    parser.add_argument("--reply-budget-bytes", type=int, required=True)
    parser.add_argument("--max-connections", type=int, default=4)
    args = parser.parse_args(argv)
    root = Path(args.socket_dir)
    if (
        args.expected_d_pid <= 0
        or args.predict_tokens not in (1, 2, 4, 8, 16)
        or args.max_total_tokens <= args.predict_tokens
        or args.context_length < args.max_total_tokens
        or not args.target_model_id.strip()
        or not root.is_dir()
        or root.is_symlink()
        or not 1 <= args.max_connections <= 8
    ):
        parser.error("bounded model, peer and private socket directory required")
    for name in (
        "draft_scratch_budget_bytes",
        "draft_persistent_budget_bytes",
        "draft_transient_bytes_bound",
        "probe_budget_bytes",
        "probe_transient_bytes_bound",
        "reply_budget_bytes",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.probe_prefix_cache_budget_bytes < 0:
        parser.error("--probe-prefix-cache-budget-bytes must be nonnegative")
    if args.draft_prefix_cache_budget_bytes < 0:
        parser.error("--draft-prefix-cache-budget-bytes must be nonnegative")

    def serve(target_runner, *, checkpoint=False):
        if not checkpoint:
            raise ValueError("a real target checkpoint is required")
        from sglang.srt.disaggregation.pvd.cuda_prediction_startup import (
            build_cuda_prediction_startup,
        )
        from sglang.srt.disaggregation.pvd.draft_sglang import DraftPlacement
        from sglang.srt.disaggregation.pvd.probe_lane_identity import (
            checkpoint_identity,
        )
        from sglang.srt.disaggregation.pvd.probe_lane_model import (
            ProbeLaneCUDAHandler,
        )
        from sglang.srt.disaggregation.pvd.probe_lane_unix import (
            ProbeLaneUnixServer,
        )
        from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

        identity = checkpoint_identity(os.path.realpath(args.model_path))
        scratch = TransferBudget(args.probe_budget_bytes, 2)
        prefix_budget = (
            TransferBudget(args.probe_prefix_cache_budget_bytes, 1)
            if args.probe_prefix_cache_budget_bytes
            else None
        )
        draft_prefix_budget = (
            TransferBudget(args.draft_prefix_cache_budget_bytes, 1)
            if args.draft_prefix_cache_budget_bytes
            else None
        )
        startup = build_cuda_prediction_startup(
            target_runner,
            draft_model_path=args.draft_model_path,
            draft_revision=None,
            draft_mem_fraction_static=args.draft_mem_fraction_static,
            target_model_id=args.target_model_id,
            placement=DraftPlacement(
                gpu_id=0,
                tp_rank=0,
                scratch_budget_bytes=args.draft_scratch_budget_bytes,
                persistent_budget_bytes=args.draft_persistent_budget_bytes,
                max_concurrent_branches=1,
            ),
            execution_lock=threading.RLock(),
            max_prefix_tokens=args.max_total_tokens - args.predict_tokens,
            predict_tokens=args.predict_tokens,
            draft_transient_bytes_bound=args.draft_transient_bytes_bound,
            probe_transient_bytes_bound=args.probe_transient_bytes_bound,
            target_scratch_budget=scratch,
            prefix_budget=prefix_budget,
            draft_prefix_cache_budget=draft_prefix_budget,
        )
        if (
            prefix_budget is not None
            and startup.pipeline.probe.prefix_cache_bytes
            > args.probe_prefix_cache_budget_bytes
        ):
            raise ValueError("sidecar prefix cache exceeds its explicit budget")
        handler = ProbeLaneCUDAHandler(
            startup.pipeline,
            weights_sha256=identity.weights_sha256,
            tokenizer_sha256=identity.tokenizer_sha256,
        )

        async def run():
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop.set)
            service = await ProbeLaneUnixServer(
                root,
                "probe.sock",
                expected_client_pid=args.expected_d_pid,
                target_model_id=args.target_model_id,
                weights_sha256=identity.weights_sha256,
                tokenizer_sha256=identity.tokenizer_sha256,
                handler=handler,
                reply_budget=TransferBudget(
                    args.reply_budget_bytes, args.max_connections
                ),
                max_connections=args.max_connections,
            ).start()

            async def reap_idle_cache():
                while not stop.is_set():
                    await asyncio.sleep(1)
                    handler.retire_idle_cache()

            reaper = asyncio.create_task(reap_idle_cache())
            stopped = asyncio.create_task(stop.wait())
            try:
                print(
                    json.dumps(
                        {
                            "schema": "pvd.probe.sidecar.ready.v1",
                            "pid": os.getpid(),
                            "socket": str(service.path),
                            "weights_sha256": identity.weights_sha256,
                            "tokenizer_sha256": identity.tokenizer_sha256,
                            "device": "cuda:0",
                        }
                    ),
                    flush=True,
                )
                done, _ = await asyncio.wait(
                    (stopped, reaper), return_when=asyncio.FIRST_COMPLETED
                )
                if reaper in done:
                    reaper.result()
            finally:
                reaper.cancel()
                stopped.cancel()
                await asyncio.gather(reaper, stopped, return_exceptions=True)
                await service.aclose()
                handler.close()
            return {"sidecar_stopped_cleanly": True}

        return asyncio.run(run())

    # Editable installs may prepend an older checkout during SGLang's import
    # chain. Pin the runner module to this script's isolated checkout before
    # model initialization; a mixed-source process must fail closed.
    from sglang.srt.model_executor import model_runner as runner_module

    expected_python = Path(__file__).resolve().parents[3] / "python"
    imported_runner = Path(runner_module.__file__).resolve()
    if not imported_runner.is_relative_to(expected_python):
        raise RuntimeError(
            "sidecar imported ModelRunner from another checkout: "
            f"{imported_runner} expected under {expected_python}"
        )

    from run_pvd_cuda_probe_smoke import main as launch_target

    return launch_target(
        [
            "--model-path",
            args.model_path,
            "--architecture",
            "qwen2",
            "--dtype",
            "float16",
            "--context-length",
            str(args.context_length),
            "--max-total-tokens",
            str(args.max_total_tokens),
        ],
        validator=serve,
        schema="pvd-qwen-probe-sidecar-v1",
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
