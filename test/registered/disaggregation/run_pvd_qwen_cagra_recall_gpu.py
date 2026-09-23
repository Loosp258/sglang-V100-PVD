"""Real Qwen2.5 target K/Q versus native CAGRA and an exact GPU oracle.

Use the existing local checkpoint on one isolated V100S. This is a retrieval
quality/plumbing probe, not a three-node PVD request or a latency measurement.
"""

import threading


def validate(runner, *, checkpoint=False):
    if not checkpoint:
        raise ValueError("a real Qwen2.5 checkpoint is required")

    import torch
    from sglang.srt.disaggregation.pvd.cagra_backend import CagraIndexBackend
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
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

    if type(runner.model).__name__ != "Qwen2ForCausalLM":
        raise ValueError("real Qwen2 model required")
    device = torch.device("cuda:0")
    prompt_count, top_k = 1024, 10
    generator = torch.Generator().manual_seed(20260924)
    tokens = tuple(
        torch.randint(3, 1000, (prompt_count,), generator=generator).tolist()
    )
    allocator = PrivatePoolAllocator(
        runner.req_to_token_pool, runner.token_to_kv_pool_allocator
    )
    slot, rows = allocator.alloc_request(), []
    source = None
    try:
        rows = allocator.alloc_kv(prompt_count)
        allocator.write_mapping(slot, 0, rows)
        adapter = DraftForwardAdapter(
            runner,
            architecture="Qwen2ForCausalLM",
            attention_backend="torch_native",
            bytes_per_token=(
                runner.model.config.num_hidden_layers
                * runner.model.config.num_key_value_heads
                * runner.model_config.head_dim
                * 2
                * 2
            ),
            device=device,
        )
        adapter.forward(
            DraftForwardInputs(
                "extend",
                tokens,
                tuple(range(prompt_count)),
                (prompt_count,),
                (slot,),
                tuple(rows),
                (0,),
                (prompt_count,),
            )
        )
        torch.cuda.synchronize(device)
        source = runner.token_to_kv_pool.get_key_buffer(0)[rows, 0].clone()
    finally:
        if rows:
            torch.cuda.synchronize(device)
            allocator.clear_mapping(slot)
            allocator.free_kv(rows)
        allocator.free_request(slot)
    if source.shape != (prompt_count, runner.model_config.head_dim):
        raise AssertionError("target Prompt K extraction has an unexpected shape")

    budget = TransferBudget(256 << 20, 1)
    lock = threading.Lock()
    space = "qwen2.5-7b-real-target"
    probe = CUDAQwen2TargetProbe(
        runner,
        ProbeConfig(space, (0,), head_start=0, head_count=1),
        device=device,
        execution_lock=lock,
        target_model_id=space,
        max_tokens=prompt_count + 2,
        max_predict_tokens=1,
        transient_bytes_bound=64 << 20,
        budget=budget,
    )
    prefix = CommittedPrefix("real-cagra", tokens, 0, "prompt")
    with probe.branch():
        result = probe.capture(
            prefix, DraftPrediction(prefix.request_id, prefix.version, (42,))
        )
        query = result[0]
        if (
            query.positional_encoding != "rope_applied"
            or query.vector_space != space
            or query.positions != (prompt_count,)
            or query.layer != 0
            or query.head_start != 0
        ):
            raise AssertionError("target Q identity or position is not compatible")
        q = query.vectors[0, 0].float().reshape(1, -1).contiguous().clone()
    if budget.snapshot()["used_staging_bytes"] != 0 or lock.locked():
        raise AssertionError("target Q probe retained private resources")

    vectors = source.float().contiguous()
    backend = CagraIndexBackend(
        device=device,
        native_bytes_per_index=512 << 20,
        graph_degree=32,
        intermediate_degree=64,
        itopk_size=64,
    )
    index = backend.build(vectors, vector_space=space, metric="ip")
    try:
        rows, scores = backend.search(index, q, top_k=top_k)
        exact = (q @ vectors.T).reshape(-1)
        expected = torch.topk(exact, top_k).indices
        actual = rows.reshape(-1).long()
        if (
            len(set(actual.tolist())) != top_k
            or bool((actual < 0).any())
            or bool((actual >= prompt_count).any())
        ):
            raise AssertionError("CAGRA returned invalid logical Prompt ids")
        score_error = float((scores.reshape(-1) - exact[actual]).abs().max())
        if score_error > 1e-2:
            raise AssertionError("CAGRA scores disagree with exact dot products")
        overlap = len(set(actual.tolist()) & set(expected.tolist()))
        return {
            "model": "Qwen2.5-7B-Instruct",
            "prompt_tokens": prompt_count,
            "layer": 0,
            "query_head": 0,
            "kv_head": 0,
            "post_rope_query": True,
            "native_cagra": True,
            "top_k": top_k,
            "recall_at_k": overlap / top_k,
            "score_max_abs_error": score_error,
            "probe_budget_restored": True,
        }
    finally:
        backend.dispose(index)
        torch.cuda.synchronize(device)


def main(argv=None):
    # Import cuVS before SGLang's package initializer imports torch. This is
    # essential for the pinned cuVS 25.02 candidate's native library loader.
    import cuvs
    from run_pvd_cuda_probe_smoke import main as run_model

    if cuvs.__version__ != "25.02.00":
        raise RuntimeError("this V100S acceptance gate requires cuVS 25.02.00")
    return run_model(
        argv,
        validator=validate,
        schema="pvd-qwen2.5-real-q-cagra-recall-v1",
    )


if __name__ == "__main__":
    raise SystemExit(main())
