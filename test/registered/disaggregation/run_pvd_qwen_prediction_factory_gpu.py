"""Real-Qwen GPU smoke for the production draft/prediction startup factory.

This loads a target and an independent draft on one GPU, then runs one bounded
prediction-only Q branch. It does not start a Scheduler, V search or RDMA.
All checkpoint locations and memory ceilings are explicit CLI inputs.
"""

import argparse
import os
import sys
import threading


def validate_factory(
    target_runner,
    *,
    checkpoint,
    draft_model_path,
    draft_revision,
    draft_mem_fraction_static,
    draft_scratch_budget_bytes,
    draft_persistent_budget_bytes,
    draft_transient_bytes_bound,
    probe_budget_bytes,
    probe_transient_bytes_bound,
    prefix_text,
):
    if not checkpoint or type(target_runner.model).__name__ != "Qwen2ForCausalLM":
        raise ValueError("a real Qwen2 target checkpoint is required")

    import torch
    from run_pvd_qwen_dual_draft_gpu import (
        _assert_rlock_released,
        _assert_runner_unchanged,
        _runner_canaries,
    )
    from sglang.srt.disaggregation.pvd.cuda_prediction_startup import (
        build_cuda_prediction_startup,
    )
    from sglang.srt.disaggregation.pvd.draft_sglang import DraftPlacement
    from sglang.srt.disaggregation.pvd.prediction import CommittedPrefix
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
    from transformers import AutoTokenizer

    target_args = target_runner.server_args
    tokenizer = AutoTokenizer.from_pretrained(
        target_args.tokenizer_path or target_args.model_path,
        revision=target_args.revision,
        local_files_only=True,
    )
    prompt = tuple(int(x) for x in tokenizer.encode(prefix_text))
    if not prompt or len(prompt) + 2 > target_args.max_total_tokens:
        raise ValueError("Prompt plus draft continuation exceeds target capacity")
    target_before = _runner_canaries(target_runner)
    lock = threading.RLock()
    target_budget = TransferBudget(probe_budget_bytes, 8)
    startup = build_cuda_prediction_startup(
        target_runner,
        draft_model_path=draft_model_path,
        draft_revision=draft_revision,
        draft_mem_fraction_static=draft_mem_fraction_static,
        target_model_id="qwen2-target-factory-smoke",
        placement=DraftPlacement(
            gpu_id=target_runner.gpu_id,
            tp_rank=0,
            scratch_budget_bytes=draft_scratch_budget_bytes,
            persistent_budget_bytes=draft_persistent_budget_bytes,
            max_concurrent_branches=1,
        ),
        execution_lock=lock,
        max_prefix_tokens=len(prompt),
        predict_tokens=2,
        draft_transient_bytes_bound=draft_transient_bytes_bound,
        probe_transient_bytes_bound=probe_transient_bytes_bound,
        target_scratch_budget=target_budget,
        target_tokenizer=tokenizer,
    )
    if startup.target_scratch_budget is not target_budget:
        raise AssertionError("factory did not retain the shared target budget")
    if startup.pipeline.probe._execution_lock is not lock:
        raise AssertionError("target Q probe uses a different execution lock")

    before_cpu_rng = torch.get_rng_state().clone()
    before_cuda_rng = torch.cuda.get_rng_state(0).clone()
    prefix = CommittedPrefix(
        "qwen-factory-smoke", prompt, committed_position=0, version="factory-v1"
    )
    with startup.pipeline.query_branch(prefix) as queries:
        query_layers = len(queries)
        if query_layers != target_runner.model.config.num_hidden_layers:
            raise AssertionError("target Q missing layers")
        expected_positions = tuple(range(len(prompt), len(prompt) + 2))
        for query in queries:
            if (
                query.positions != expected_positions
                or query.vector_space != "qwen2-target-factory-smoke"
                or query.positional_encoding != "rope_applied"
                or not torch.isfinite(query.vectors).all()
            ):
                raise AssertionError("target Q identity or values invalid")
    del queries
    torch.cuda.synchronize(0)
    if not torch.equal(before_cpu_rng, torch.get_rng_state()) or not torch.equal(
        before_cuda_rng, torch.cuda.get_rng_state(0)
    ):
        raise AssertionError("prediction branch changed target RNG")
    _assert_runner_unchanged(target_runner, target_before)
    _assert_rlock_released(lock)
    if startup.pipeline.provider.degraded or startup.pipeline.provider.active_branches:
        raise AssertionError("draft branch did not retire")
    if target_budget.snapshot()["used_staging_bytes"]:
        raise AssertionError("target Q scratch reservation was not refunded")
    return {
        "target_model_path": os.path.realpath(target_args.model_path),
        "draft_model_path": os.path.realpath(draft_model_path),
        "draft_retained_bytes": startup.draft_retained_bytes,
        "prompt_tokens": len(prompt),
        "query_layers": query_layers,
        "query_positions": expected_positions,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "scope": "single-GPU draft factory only; no Scheduler, V search or RDMA",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--draft-revision", default=None)
    parser.add_argument("--draft-mem-fraction-static", type=float, required=True)
    parser.add_argument("--draft-scratch-budget-bytes", type=int, required=True)
    parser.add_argument("--draft-persistent-budget-bytes", type=int, required=True)
    parser.add_argument("--draft-transient-bytes-bound", type=int, required=True)
    parser.add_argument("--probe-budget-bytes", type=int, required=True)
    parser.add_argument("--probe-transient-bytes-bound", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--max-total-tokens", type=int, default=128)
    parser.add_argument(
        "--prefix-text", default="The quick brown fox jumps over the lazy dog"
    )
    args, passthrough = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    for label, path in (("target", args.model_path), ("draft", args.draft_model_path)):
        if not os.path.isabs(path) or not os.path.isfile(
            os.path.join(path, "config.json")
        ):
            parser.error(f"{label} checkpoint must be an absolute local directory")
    for name in (
        "draft_scratch_budget_bytes",
        "draft_persistent_budget_bytes",
        "draft_transient_bytes_bound",
        "probe_budget_bytes",
        "probe_transient_bytes_bound",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    def validator(target_runner, *, checkpoint=False):
        return validate_factory(
            target_runner,
            checkpoint=checkpoint,
            draft_model_path=args.draft_model_path,
            draft_revision=args.draft_revision,
            draft_mem_fraction_static=args.draft_mem_fraction_static,
            draft_scratch_budget_bytes=args.draft_scratch_budget_bytes,
            draft_persistent_budget_bytes=args.draft_persistent_budget_bytes,
            draft_transient_bytes_bound=args.draft_transient_bytes_bound,
            probe_budget_bytes=args.probe_budget_bytes,
            probe_transient_bytes_bound=args.probe_transient_bytes_bound,
            prefix_text=args.prefix_text,
        )

    from run_pvd_cuda_probe_smoke import main as run

    return run(
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
            *passthrough,
        ],
        validator=validator,
        schema="pvd-qwen-prediction-factory-v1",
    )


if __name__ == "__main__":
    raise SystemExit(main())
