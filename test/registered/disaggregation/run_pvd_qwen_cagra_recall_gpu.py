"""Real Qwen2.5 target K/Q versus native CAGRA and an exact GPU oracle.

Use the existing local checkpoint on one isolated V100S. This is a retrieval
quality/plumbing probe, not a three-node PVD request or a latency measurement.
"""

import os
import threading


def _prompt_tokens(model_path, count):
    mode = os.environ.get("PVD_CAGRA_RECALL_PROMPT", "language")
    if mode == "random":
        import torch

        generator = torch.Generator().manual_seed(20260924)
        return mode, tuple(
            torch.randint(3, 1000, (count,), generator=generator).tolist()
        )
    if mode != "language":
        raise ValueError("PVD_CAGRA_RECALL_PROMPT must be language or random")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    passages = (
        (
            "A router selects a prefill worker, a vector worker, "
            "and a decode worker for each request."
        ),
        (
            "The prefill worker computes prompt keys and values "
            "before transferring them to the vector store."
        ),
        "Each shard owns a bounded set of heads and maintains its own index lifecycle.",
        (
            "The decode worker issues a retrieval query before a refresh boundary "
            "and waits if the result is late."
        ),
        (
            "A transport descriptor identifies the destination region, "
            "generation, and permitted byte range."
        ),
        (
            "The experiment compares approximate graph search "
            "with an exact dot-product oracle."
        ),
        (
            "A patient researcher records latency, recall, output quality, "
            "and memory pressure separately."
        ),
        (
            "Natural-language questions can include dates, measurements, "
            "algorithms, and unrelated facts."
        ),
    )
    text = " ".join(
        f"Section {section}: {passages[section % len(passages)]}"
        for section in range(count)
    )
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) < count:
        raise AssertionError(
            "natural-language fixture is shorter than the requested prompt"
        )
    return mode, tuple(encoded[:count])


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
    prompt_count = int(os.environ.get("PVD_CAGRA_RECALL_ROWS", "1024"))
    if not 256 <= prompt_count <= 2048:
        raise ValueError("PVD_CAGRA_RECALL_ROWS must be in [256, 2048]")
    top_k = 10
    layers = (0, min(16, runner.model.config.num_hidden_layers - 1))
    query_heads = (0, 1, 7, 8)
    if runner.model.config.num_attention_heads < 9:
        raise ValueError(
            "this Qwen2.5-7B GQA acceptance requires at least nine query heads"
        )
    group_size = (
        runner.model.config.num_attention_heads
        // runner.model.config.num_key_value_heads
    )
    if group_size != 7:
        raise ValueError("this Qwen2.5-7B acceptance expects seven Q heads per KV head")
    prompt_mode, tokens = _prompt_tokens(runner.model_config.model_path, prompt_count)
    allocator = PrivatePoolAllocator(
        runner.req_to_token_pool, runner.token_to_kv_pool_allocator
    )
    slot, rows = allocator.alloc_request(), []
    sources = {}
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
        logits = adapter.forward(
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
        next_token = int(logits.argmax().item())
        for layer in layers:
            for kv_head in (0, 1):
                sources[layer, kv_head] = (
                    runner.token_to_kv_pool.get_key_buffer(layer)[rows, kv_head]
                    .clone()
                    .float()
                    .contiguous()
                )
    finally:
        if rows:
            torch.cuda.synchronize(device)
            allocator.clear_mapping(slot)
            allocator.free_kv(rows)
        allocator.free_request(slot)
    if any(
        source.shape != (prompt_count, runner.model_config.head_dim)
        for source in sources.values()
    ):
        raise AssertionError("target Prompt K extraction has an unexpected shape")

    # The private target-probe KV footprint grows with the bounded prompt.
    # At 2048 rows Qwen2.5-7B exceeds 256 MiB after scratch is included.
    budget = TransferBudget(512 << 20, 1)
    lock = threading.Lock()
    space = "qwen2.5-7b-real-target"
    probe = CUDAQwen2TargetProbe(
        runner,
        ProbeConfig(space, layers, head_start=0, head_count=9),
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
            prefix, DraftPrediction(prefix.request_id, prefix.version, (next_token,))
        )
        queries = {}
        for query in result:
            if (
                query.positional_encoding != "rope_applied"
                or query.vector_space != space
                or query.positions != (prompt_count,)
                or query.layer not in layers
                or query.head_start != 0
            ):
                raise AssertionError("target Q identity or position is not compatible")
            for query_head in query_heads:
                queries[query.layer, query_head] = (
                    query.vectors[0, query_head]
                    .float()
                    .reshape(1, -1)
                    .contiguous()
                    .clone()
                )
    if budget.snapshot()["used_staging_bytes"] != 0 or lock.locked():
        raise AssertionError("target Q probe retained private resources")

    backend = CagraIndexBackend(
        device=device,
        native_bytes_per_index=512 << 20,
        graph_degree=32,
        intermediate_degree=64,
        itopk_size=64,
    )
    cases = []
    for layer in layers:
        for kv_head in (0, 1):
            vectors = sources[layer, kv_head]
            index = backend.build(vectors, vector_space=space, metric="ip")
            try:
                for query_head in query_heads:
                    if query_head // group_size != kv_head:
                        continue
                    q = queries[layer, query_head]
                    rows, scores = backend.search(index, q, top_k=top_k)
                    exact = (q @ vectors.T).reshape(-1)
                    expected = torch.topk(exact, top_k).indices
                    actual = rows.reshape(-1).long()
                    if (
                        len(set(actual.tolist())) != top_k
                        or bool((actual < 0).any())
                        or bool((actual >= prompt_count).any())
                    ):
                        raise AssertionError(
                            "CAGRA returned invalid logical Prompt ids"
                        )
                    score_error = float(
                        (scores.reshape(-1) - exact[actual]).abs().max()
                    )
                    if score_error > 1e-2:
                        raise AssertionError(
                            "CAGRA scores disagree with exact dot products"
                        )
                    overlap = len(set(actual.tolist()) & set(expected.tolist()))
                    cases.append(
                        {
                            "layer": layer,
                            "query_head": query_head,
                            "kv_head": kv_head,
                            "recall_at_k": overlap / top_k,
                            "score_max_abs_error": score_error,
                        }
                    )
            finally:
                backend.dispose(index)
                torch.cuda.synchronize(device)
    if len(cases) != len(layers) * len(query_heads):
        raise AssertionError("not every configured layer/head was measured")
    return {
        "model": "Qwen2.5-7B-Instruct",
        "prompt_tokens": prompt_count,
        "prompt_mode": prompt_mode,
        "next_token_source": "target_greedy_logits",
        "post_rope_query": True,
        "native_cagra": True,
        "top_k": top_k,
        "case_count": len(cases),
        "mean_recall_at_k": sum(case["recall_at_k"] for case in cases) / len(cases),
        "min_recall_at_k": min(case["recall_at_k"] for case in cases),
        "score_max_abs_error": max(case["score_max_abs_error"] for case in cases),
        "cases": cases,
        "probe_budget_restored": True,
    }


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
        schema="pvd-qwen2.5-real-q-cagra-recall-v2",
    )


if __name__ == "__main__":
    raise SystemExit(main())
