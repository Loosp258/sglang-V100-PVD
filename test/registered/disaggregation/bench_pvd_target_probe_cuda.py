"""Isolated Qwen2.5 target-Q probe timing, without P/V/network startup.

Run with ``--model-path``, ``--architecture qwen2`` and a context of at least
1100 tokens. The common runner setup lives in run_pvd_cuda_probe_smoke.py.
Each sample checks Q equality and complete scratch-budget retirement. This
measures the probe alone, not end-to-end PVD throughput or retrieval quality.
"""

import statistics
import threading
import time


def validate(runner, *, checkpoint=False):
    if not checkpoint or type(runner.model).__name__ != "Qwen2ForCausalLM":
        raise ValueError("a real Qwen2 checkpoint is required")

    import torch
    from sglang.srt.disaggregation.pvd.cuda_target_probe import CUDAQwen2TargetProbe
    from sglang.srt.disaggregation.pvd.prediction import (
        CommittedPrefix,
        DraftPrediction,
        ProbeConfig,
    )
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

    prompt_count = 1094
    config = runner.model.config
    if runner.model_config.context_len < prompt_count + 2:
        raise ValueError("model context is too short for the measured probe")
    generator = torch.Generator().manual_seed(20260926)
    tokens = tuple(
        torch.randint(
            3, min(config.vocab_size, 1000), (prompt_count,), generator=generator
        ).tolist()
    )
    prefix = CommittedPrefix("probe-bench", tokens, 0, "fixed-prefix")
    prediction = DraftPrediction(prefix.request_id, prefix.version, (42, 43))
    budget = TransferBudget(256 << 20, 1)
    execution_lock = threading.Lock()
    probe = CUDAQwen2TargetProbe(
        runner,
        ProbeConfig(
            "qwen2.5-7b-probe-bench",
            tuple(range(config.num_hidden_layers)),
            head_start=0,
            head_count=config.num_attention_heads,
        ),
        device="cuda:0",
        execution_lock=execution_lock,
        target_model_id="qwen2.5-7b-probe-bench",
        max_tokens=prompt_count + 2,
        max_predict_tokens=2,
        transient_bytes_bound=64 << 20,
        budget=budget,
    )
    reference, samples_ms = None, []
    for round_index in range(6):
        torch.cuda.synchronize("cuda:0")
        started = time.perf_counter()
        with probe.branch():
            queries = probe.capture(prefix, prediction)
        torch.cuda.synchronize("cuda:0")
        elapsed_ms = (time.perf_counter() - started) * 1000
        if budget.snapshot()["used_staging_bytes"] != 0 or execution_lock.locked():
            raise AssertionError("probe failed to retire its scratch ownership")
        if reference is None:
            reference = tuple(query.vectors.clone() for query in queries)
        else:
            for query, expected in zip(queries, reference, strict=True):
                torch.testing.assert_close(query.vectors, expected, rtol=0, atol=0)
        if round_index:
            samples_ms.append(round(elapsed_ms, 3))
    return {
        "model": "Qwen2.5-7B-Instruct",
        "prompt_tokens": prompt_count,
        "predicted_tokens": 2,
        "layers": config.num_hidden_layers,
        "query_heads": config.num_attention_heads,
        "warmup_rounds": 1,
        "measured_rounds": len(samples_ms),
        "probe_ms": samples_ms,
        "median_probe_ms": round(statistics.median(samples_ms), 3),
        "same_q_each_round": True,
        "scratch_retired_each_round": True,
        "end_to_end_pvd_validated": False,
    }


def main(argv=None):
    from run_pvd_cuda_probe_smoke import main as run_model

    return run_model(argv, validator=validate, schema="pvd-target-probe-bench-v1")


if __name__ == "__main__":
    raise SystemExit(main())
