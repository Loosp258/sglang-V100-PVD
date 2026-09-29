"""Real Qwen2.5 target K/Q versus native CAGRA and an exact GPU oracle.

Use the existing local checkpoint on one isolated V100S. This is a retrieval
quality/plumbing probe, not a three-node PVD request or a latency measurement.
"""

import os
import time
from statistics import median


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
    from sglang.srt.disaggregation.pvd.draft_forward_adapter import (
        DraftForwardAdapter,
        PrivatePoolAllocator,
    )
    from sglang.srt.disaggregation.pvd.draft_runner_sglang import DraftForwardInputs

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

    space = "qwen2.5-7b-real-target"
    # Observe the actual attention input at the next-token position. The
    # attention module receives Q after the model applies RoPE; this avoids
    # reimplementing the model's Q projection or positional transform.
    captured = {}
    hooks = []

    def capture_q(module, args, *, layer):
        captured[layer] = (
            args[0][-1]
            .reshape(
                runner.model.config.num_attention_heads, runner.model_config.head_dim
            )
            .detach()
            .clone()
            .float()
        )

    slot, rows = allocator.alloc_request(), []
    try:
        for layer in layers:
            hooks.append(
                runner.model.model.layers[
                    layer
                ].self_attn.attn.register_forward_pre_hook(
                    lambda module, args, layer=layer: capture_q(
                        module, args, layer=layer
                    )
                )
            )
        full_tokens = tokens + (next_token,)
        rows = allocator.alloc_kv(len(full_tokens))
        allocator.write_mapping(slot, 0, rows)
        adapter.forward(
            DraftForwardInputs(
                "extend",
                full_tokens,
                tuple(range(len(full_tokens))),
                (len(full_tokens),),
                (slot,),
                tuple(rows),
                (0,),
                (len(full_tokens),),
            )
        )
        torch.cuda.synchronize(device)
    finally:
        for hook in hooks:
            hook.remove()
        if rows:
            torch.cuda.synchronize(device)
            allocator.clear_mapping(slot)
            allocator.free_kv(rows)
        allocator.free_request(slot)
    if set(captured) != set(layers):
        raise AssertionError("real target forward did not expose every Q layer")
    queries = {}
    for layer, query_matrix in captured.items():
        if query_matrix.shape != (
            runner.model.config.num_attention_heads,
            runner.model_config.head_dim,
        ):
            raise AssertionError("post-RoPE target Q has unexpected shape")
        for query_head in query_heads:
            queries[layer, query_head] = (
                query_matrix[query_head].reshape(1, -1).contiguous()
            )

    backend = CagraIndexBackend(
        device=device,
        native_bytes_per_index=512 << 20,
        graph_degree=32,
        intermediate_degree=64,
        itopk_size=64,
    )
    extend_prefix = int(os.environ.get("PVD_CAGRA_RECALL_EXTEND_PREFIX", "0"))
    if extend_prefix and not 256 <= extend_prefix < prompt_count:
        raise ValueError("extend prefix must be in [256, prompt_count)")
    if extend_prefix and not backend.supports_extend:
        raise RuntimeError("selected cuVS binding cannot extend CAGRA")
    arm_order = os.environ.get("PVD_CAGRA_RECALL_ARM_ORDER", "full_first")
    if arm_order not in ("full_first", "extended_first"):
        raise ValueError("arm order must be full_first or extended_first")
    order = (
        ("extended", "full")
        if extend_prefix and arm_order == "extended_first"
        else (("full", "extended") if extend_prefix else ("full",))
    )
    cases = []
    builds = []
    for layer in layers:
        for kv_head in (0, 1):
            vectors = sources[layer, kv_head]
            for arm in order:
                build_started = time.perf_counter()
                index = backend.build(
                    vectors[:extend_prefix] if arm == "extended" else vectors,
                    vector_space=space,
                    metric="ip",
                )
                build_seconds = time.perf_counter() - build_started
                extend_seconds = 0.0
                if arm == "extended":
                    extend_started = time.perf_counter()
                    index = backend.extend(index, vectors[extend_prefix:])
                    extend_seconds = time.perf_counter() - extend_started
                builds.append(
                    {
                        "arm": arm,
                        "layer": layer,
                        "kv_head": kv_head,
                        "build_seconds": build_seconds,
                        "extend_seconds": extend_seconds,
                    }
                )
                try:
                    for query_head in query_heads:
                        if query_head // group_size != kv_head:
                            continue
                        q = queries[layer, query_head]
                        search_started = time.perf_counter()
                        rows, scores = backend.search(index, q, top_k=top_k)
                        search_seconds = time.perf_counter() - search_started
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
                                "arm": arm,
                                "layer": layer,
                                "query_head": query_head,
                                "kv_head": kv_head,
                                "recall_at_k": overlap / top_k,
                                "score_max_abs_error": score_error,
                                "search_seconds": search_seconds,
                                "ids": actual.tolist(),
                            }
                        )
                finally:
                    backend.dispose(index)
                    torch.cuda.synchronize(device)
    if len(cases) != len(layers) * len(query_heads) * (2 if extend_prefix else 1):
        raise AssertionError("not every configured layer/head was measured")
    arms = {}
    for arm in order:
        arm_cases = [case for case in cases if case["arm"] == arm]
        arm_builds = [item for item in builds if item["arm"] == arm]
        arms[arm] = {
            "mean_recall_at_k": sum(case["recall_at_k"] for case in arm_cases)
            / len(arm_cases),
            "min_recall_at_k": min(case["recall_at_k"] for case in arm_cases),
            "median_head_build_seconds": median(
                item["build_seconds"] for item in arm_builds
            ),
            "median_head_extend_seconds": median(
                item["extend_seconds"] for item in arm_builds
            ),
            "median_search_seconds": median(
                case["search_seconds"] for case in arm_cases
            ),
        }
    if extend_prefix:
        full_ids = {
            (case["layer"], case["query_head"]): set(case["ids"])
            for case in cases
            if case["arm"] == "full"
        }
        arms["full_extended_mean_overlap_at_k"] = sum(
            len(full_ids[case["layer"], case["query_head"]] & set(case["ids"])) / top_k
            for case in cases
            if case["arm"] == "extended"
        ) / len(full_ids)
    return {
        "model": "Qwen2.5-7B-Instruct",
        "prompt_tokens": prompt_count,
        "prompt_mode": prompt_mode,
        "next_token_source": "target_greedy_logits",
        "post_rope_query": True,
        "native_cagra": True,
        "top_k": top_k,
        "extend_prefix": extend_prefix,
        "arm_order": arm_order,
        "arms": arms,
        "builds": builds,
        "case_count": len(cases),
        "mean_recall_at_k": arms["full"]["mean_recall_at_k"],
        "min_recall_at_k": arms["full"]["min_recall_at_k"],
        "score_max_abs_error": max(case["score_max_abs_error"] for case in cases),
        "cases": cases,
        "q_source": "post_rope_attention_input_hook",
    }


def main(argv=None):
    # Import cuVS before SGLang's package initializer imports torch. This is
    # essential for the pinned cuVS 25.02 candidate's native library loader.
    import cuvs
    from run_pvd_cuda_probe_smoke import main as run_model
    from sglang.srt.model_executor.model_runner import ModelRunner

    required_version = (
        "25.10.00"
        if int(os.environ.get("PVD_CAGRA_RECALL_EXTEND_PREFIX", "0"))
        else "25.02.00"
    )
    if cuvs.__version__ != required_version:
        raise RuntimeError(
            f"this V100S acceptance gate requires cuVS {required_version}"
        )
    # This standalone probe uses torch_native attention. Its base backend has
    # no CUDA-graph fill value, while ModelRunner's optional prefill kernel
    # warmup asks for one even with CUDA graph disabled. The measured forward
    # and retrieval kernels still execute normally in validate().
    ModelRunner.kernel_warmup = lambda self: None
    return run_model(
        argv,
        validator=validate,
        schema="pvd-qwen2.5-real-q-cagra-recall-v3",
    )


if __name__ == "__main__":
    raise SystemExit(main())
