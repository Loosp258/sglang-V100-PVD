"""Isolated Qwen2.5 target-Q full-prefix versus retained-prefix experiment.

The persistent private pool here is deliberately test-owned, not a serving
feature: it has no request-close hook or production budget. It establishes
numerical Q parity and the per-round compute opportunity before such an owner
is added. Never invoke inside a serving process.
"""

import statistics
import threading
import time
from types import SimpleNamespace


def validate(runner, *, checkpoint=False):
    if not checkpoint or type(runner.model).__name__ != "Qwen2ForCausalLM":
        raise ValueError("a real Qwen2 checkpoint is required")

    import torch
    from sglang.srt.disaggregation.pvd.cuda_target_probe import CUDAQwen2TargetProbe
    from sglang.srt.disaggregation.pvd.draft_forward_adapter import (
        DraftForwardAdapter,
        PrivatePoolAllocator,
    )
    from sglang.srt.disaggregation.pvd.draft_runner_sglang import DraftForwardInputs
    from sglang.srt.disaggregation.pvd.prediction import (
        CommittedPrefix,
        DraftPrediction,
        ProbeConfig,
    )
    from sglang.srt.disaggregation.pvd.target_probe import PostRopeQueryCapture
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    config = runner.model.config
    prompt_count, predict_count = 1094, 2
    if runner.model_config.context_len < prompt_count + predict_count:
        raise ValueError("target context is too short")
    tokens = tuple(
        torch.randint(
            3,
            min(config.vocab_size, 1000),
            (prompt_count,),
            generator=torch.Generator().manual_seed(20260926),
        ).tolist()
    )
    prefix = CommittedPrefix("incremental-probe-bench", tokens, 0, "fixed-prefix")
    prediction = DraftPrediction(prefix.request_id, prefix.version, (42, 43))
    probe_config = ProbeConfig(
        "qwen2.5-7b-incremental-bench",
        tuple(range(config.num_hidden_layers)),
        head_start=0,
        head_count=config.num_attention_heads,
    )
    budget = TransferBudget(256 << 20, 1)
    lock = threading.Lock()
    probe = CUDAQwen2TargetProbe(
        runner,
        probe_config,
        device="cuda:0",
        execution_lock=lock,
        target_model_id=probe_config.target_model_id,
        max_tokens=prompt_count + predict_count,
        max_predict_tokens=predict_count,
        transient_bytes_bound=64 << 20,
        budget=budget,
    )
    with probe.branch():
        full = tuple(q.vectors.clone() for q in probe.capture(prefix, prediction))
    if budget.snapshot()["used_staging_bytes"]:
        raise AssertionError("baseline probe retained its scratch")

    capacity = prompt_count + predict_count
    requests = ReqToTokenPool(1, capacity, "cuda:0", False)
    pool = MHATokenToKVPool(
        capacity,
        1,
        probe.dtype,
        probe.kv_heads,
        probe.head_dim,
        probe.layers,
        "cuda:0",
        False,
    )
    kv = TokenToKVPoolAllocator(capacity, probe.dtype, "cuda:0", pool, False)
    allocator = PrivatePoolAllocator(requests, kv)
    backend = TorchNativeAttnBackend(
        SimpleNamespace(
            device="cuda:0", req_to_token_pool=requests, token_to_kv_pool=pool
        )
    )
    builder = DraftForwardAdapter(
        None,
        architecture="Qwen2ForCausalLM",
        attention_backend="torch_native",
        bytes_per_token=probe.layers * probe.kv_heads * probe.head_dim * 2 * 4,
        device="cuda:0",
    )
    slot, prefix_rows = None, []
    samples_ms = {"padded": [], "compact": []}
    largest_q_error = {"padded": 0.0, "compact": 0.0}
    tight_mismatch_count = {"padded": 0, "compact": 0}
    try:
        slot = allocator.alloc_request()
        prefix_rows = allocator.alloc_kv(prompt_count)
        allocator.write_mapping(slot, 0, prefix_rows)
        batch = builder.build_forward_batch(
            DraftForwardInputs(
                "extend",
                tokens,
                tuple(range(prompt_count)),
                (prompt_count,),
                (slot,),
                tuple(prefix_rows),
                (0,),
                (prompt_count,),
            )
        )
        backend.init_forward_metadata(batch)
        with (
            torch.inference_mode(),
            forward_context(ForwardContext(attn_backend=backend)),
        ):
            runner.model.model(batch.input_ids, batch.positions, batch)
        torch.cuda.synchronize("cuda:0")
        del batch

        for compact in (False, True):
            mode = "compact" if compact else "padded"
            for _ in range(5):
                suffix_rows = allocator.alloc_kv(predict_count)
                try:
                    allocator.write_mapping(slot, prompt_count, suffix_rows)
                    batch = builder.build_forward_batch(
                        DraftForwardInputs(
                            "extend",
                            prediction.tokens,
                            tuple(range(prompt_count, capacity)),
                            (capacity,),
                            (slot,),
                            tuple(suffix_rows),
                            (prompt_count,),
                            (predict_count,),
                        )
                    )
                    batch.pvd_compact_extend = compact
                    capture = PostRopeQueryCapture(
                        probe_config,
                        prefix,
                        predict_count,
                        query_heads=probe.query_heads,
                        head_dim=probe.head_dim,
                        forward_start=prompt_count,
                    )
                    capture.bind_positions(batch.positions)
                    batch.pvd_query_capture = capture
                    backend.init_forward_metadata(batch)
                    torch.cuda.synchronize("cuda:0")
                    started = time.perf_counter()
                    with (
                        torch.inference_mode(),
                        forward_context(ForwardContext(attn_backend=backend)),
                    ):
                        runner.model.model(batch.input_ids, batch.positions, batch)
                    incremental = capture.finish()
                    torch.cuda.synchronize("cuda:0")
                    samples_ms[mode].append(
                        round((time.perf_counter() - started) * 1000, 3)
                    )
                    for actual, expected in zip(incremental, full, strict=True):
                        delta = (actual.vectors - expected).abs()
                        largest_q_error[mode] = max(
                            largest_q_error[mode], float(delta.max())
                        )
                        tolerance = 3e-3 + 3e-3 * expected.abs()
                        tight_mismatch_count[mode] += int((delta > tolerance).sum())
                        torch.testing.assert_close(
                            actual.vectors, expected, rtol=5e-3, atol=2e-2
                        )
                finally:
                    torch.cuda.synchronize("cuda:0")
                    requests.req_to_token[slot, prompt_count:capacity] = 0
                    allocator.free_kv(suffix_rows)
                    torch.cuda.synchronize("cuda:0")
                    del batch
    finally:
        torch.cuda.synchronize("cuda:0")
        if slot is not None:
            allocator.clear_mapping(slot)
            if prefix_rows:
                allocator.free_kv(prefix_rows)
            allocator.free_request(slot)
        torch.cuda.synchronize("cuda:0")
    return {
        "model": "Qwen2.5-7B-Instruct",
        "prompt_tokens": prompt_count,
        "predicted_tokens": predict_count,
        "layers": config.num_hidden_layers,
        "incremental_ms": samples_ms,
        "median_incremental_ms": {
            mode: round(statistics.median(values), 3)
            for mode, values in samples_ms.items()
        },
        "max_q_abs_error_vs_full": largest_q_error,
        "tight_mismatches_vs_full": tight_mismatch_count,
        "private_pool_released": True,
        "serving_lifecycle_or_budget_implemented": False,
        "end_to_end_pvd_validated": False,
    }


def main(argv=None):
    from run_pvd_cuda_probe_smoke import main as run_model

    return run_model(argv, validator=validate, schema="pvd-incremental-probe-bench-v1")


if __name__ == "__main__":
    raise SystemExit(main())
