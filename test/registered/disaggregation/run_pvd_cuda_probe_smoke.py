"""Strict standalone real-CUDA probe smoke; random tiny Llama/Qwen2.

Owns process groups and the model. Never run inside a serving process. This
checks target-Q extraction, not CAGRA, sparse Decode, RDMA or performance.
"""

import argparse
import json
import os
import socket
import sys
import tempfile
import threading
import traceback


def validate(runner, *, checkpoint=False):
    import torch
    from sglang.srt.disaggregation.pvd.cuda_target_probe import (
        CUDALlamaTargetProbe,
        CUDAQwen2TargetProbe,
    )
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

    execution_lock = threading.Lock()
    budget = TransferBudget(128 << 20, 1)
    architecture = type(runner.model).__name__
    probe_type = {
        "LlamaForCausalLM": CUDALlamaTargetProbe,
        "Qwen2ForCausalLM": CUDAQwen2TargetProbe,
    }[architecture]
    probe = probe_type(
        runner,
        ProbeConfig("cuda-smoke-target", (0, 1), head_start=1, head_count=2),
        device="cuda:0",
        execution_lock=execution_lock,
        target_model_id="cuda-smoke-target",
        max_tokens=16,
        max_predict_tokens=2,
        transient_bytes_bound=64 << 20,
        budget=budget,
    )
    pools = runner.token_to_kv_pool.k_buffer + runner.token_to_kv_pool.v_buffer
    with torch.no_grad():
        for tensor in pools:
            (tensor[:1] if checkpoint else tensor).fill_(0.25)
    # A real 7B checkpoint must not be duplicated just for this test. Keep
    # bounded per-tensor canaries; the tiny fixture still checks every byte.
    before = [
        (tensor, tensor[:1].clone() if checkpoint else tensor.clone())
        for tensor in pools
    ]
    weights = [
        (p, p.detach().flatten()[:16].clone() if checkpoint else p.clone())
        for p in runner.model.parameters()
    ]
    mapping = runner.req_to_token_pool.req_to_token.clone()
    free = runner.token_to_kv_pool_allocator.free_pages.clone()
    slots = list(runner.req_to_token_pool.free_slots)
    rng = torch.cuda.get_rng_state(0).clone()
    prefix = CommittedPrefix("probe", (1, 4, 13, 7), 0, "prefix")
    prediction = DraftPrediction("probe", "prefix", (19, 27))
    with probe.branch():
        queries = probe.capture(prefix, prediction)
        predicted = [query.vectors.clone() for query in queries]
    committed = CommittedPrefix(
        "probe", prefix.tokens + prediction.tokens, 2, "committed"
    )
    with probe.branch():
        actual_queries = probe.capture_committed(committed, (5,))
        actual = [query.vectors.clone() for query in actual_queries]
    assert budget.snapshot()["used_staging_bytes"] == 0
    assert not execution_lock.locked()
    for current, old in before:
        torch.testing.assert_close(
            old, current[:1] if checkpoint else current, rtol=0, atol=0
        )
    for current, old in weights:
        torch.testing.assert_close(
            old,
            current.detach().flatten()[:16] if checkpoint else current,
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        mapping, runner.req_to_token_pool.req_to_token, rtol=0, atol=0
    )
    torch.testing.assert_close(
        free, runner.token_to_kv_pool_allocator.free_pages, rtol=0, atol=0
    )
    assert slots == runner.req_to_token_pool.free_slots
    assert torch.equal(rng, torch.cuda.get_rng_state(0))

    # Independent TEST oracle observes each layer's attention input after
    # RoPE, plus the layer-local QKV projection before RoPE. A large model may
    # share a RoPE module across layers, so hooking the module cannot identify
    # which layer produced a query. Probe code itself installs no hooks.
    oracle, raw, hooks = [], [], []
    allocator = PrivatePoolAllocator(
        runner.req_to_token_pool, runner.token_to_kv_pool_allocator
    )
    slot, rows = allocator.alloc_request(), []
    try:
        for layer in runner.model.model.layers[:2]:
            attention = layer.self_attn
            q_size = attention.q_size
            hooks.append(
                attention.qkv_proj.register_forward_hook(
                    lambda module, args, output, size=q_size: raw.append(
                        output[0][:, :size].detach().clone()
                    )
                )
            )
            hooks.append(
                attention.attn.register_forward_pre_hook(
                    lambda module, args: oracle.append(args[0].detach().clone())
                )
            )
        tokens = committed.tokens
        rows = allocator.alloc_kv(len(tokens))
        allocator.write_mapping(slot, 0, rows)
        adapter = DraftForwardAdapter(
            runner,
            architecture=architecture,
            attention_backend="torch_native",
            bytes_per_token=probe.layers * probe.kv_heads * probe.head_dim * 2 * 4,
            device="cuda:0",
        )
        with execution_lock:
            adapter.forward(
                DraftForwardInputs(
                    "extend",
                    tokens,
                    tuple(range(len(tokens))),
                    (len(tokens),),
                    (slot,),
                    tuple(rows),
                    (0,),
                    (len(tokens),),
                )
            )
            torch.cuda.synchronize("cuda:0")
    finally:
        torch.cuda.synchronize("cuda:0")
        for hook in hooks:
            hook.remove()
        allocator.clear_mapping(slot)
        allocator.free_kv(rows)
        allocator.free_request(slot)
        torch.cuda.synchronize("cuda:0")
    assert len(oracle) == len(raw) == len(predicted) == 2, (
        len(oracle),
        len(raw),
        len(predicted),
    )
    tolerance = 3e-3 if probe.dtype == torch.float16 else 2e-4
    errors = []
    for layer, value in enumerate(predicted):
        expected = oracle[layer].reshape(6, probe.query_heads, probe.head_dim)[4:, 1:3]
        unrotated = raw[layer].reshape(6, probe.query_heads, probe.head_dim)[4:, 1:3]
        torch.testing.assert_close(value, expected, rtol=tolerance, atol=tolerance)
        assert not torch.allclose(value, unrotated, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(
            actual[layer], expected[1:], rtol=tolerance, atol=tolerance
        )
        errors.append(float((value - expected).abs().max()))
    return {
        "architecture": architecture,
        "layers": 2,
        "max_q_abs_error": max(errors),
        "post_rope_oracle_matched": True,
        "actual_prefix_fallback_matched": True,
        "target_state_check": (
            "per-tensor bounded canaries, mapping and CUDA RNG unchanged"
            if checkpoint
            else "full weights/pools, mapping and CUDA RNG unchanged"
        ),
        "probe_budget_restored": True,
    }


def main(argv=None, *, validator=validate, schema="pvd-cuda-target-probe-v1"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--architecture", choices=("llama", "qwen2"), default="llama")
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args(argv)
    if not __debug__:
        parser.error("assertions must be enabled")
    if args.model_path and (
        not os.path.isabs(args.model_path)
        or not os.path.isfile(os.path.join(args.model_path, "config.json"))
    ):
        parser.error("--model-path must be an absolute local checkpoint directory")
    report = {
        "schema": schema,
        "status": "blocked",
        "fixture": (
            f"checkpoint at {args.model_path}"
            if args.model_path
            else f"random tiny {args.architecture}; not a production checkpoint"
        ),
        "production_gpu_rdma_validated": False,
        "performance_validated": False,
    }
    try:
        import torch

        if sys.platform != "linux" or not torch.cuda.is_available():
            report["reason"] = "requires a serving-capable Linux CUDA environment"
            print(json.dumps(report, indent=2))
            return 2
        if os.environ.get("SGLANG_USE_CPU_ENGINE") not in (None, "0"):
            raise RuntimeError("refusing CUDA smoke with CPU engine override")
        os.environ["HF_HUB_OFFLINE"] = "1"
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from sglang.srt.layers.dp_attention import initialize_dp_attention
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.server_args import ServerArgs
        from transformers import GenerationConfig, LlamaConfig, Qwen2Config

        torch.manual_seed(8128)
        torch.cuda.set_device(0)
        with tempfile.TemporaryDirectory(prefix="pvd-cuda-probe-") as directory:
            if args.model_path is None:
                config_type = (
                    LlamaConfig if args.architecture == "llama" else Qwen2Config
                )
                config = config_type(
                    vocab_size=128,
                    hidden_size=256,
                    intermediate_size=512,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=64,
                    architectures=[
                        "LlamaForCausalLM"
                        if args.architecture == "llama"
                        else "Qwen2ForCausalLM"
                    ],
                    tie_word_embeddings=False,
                )
                config.save_pretrained(directory)
                GenerationConfig(bos_token_id=1, eos_token_id=2).save_pretrained(
                    directory
                )
            server_args = ServerArgs(
                model_path=args.model_path or directory,
                device="cuda",
                dtype=args.dtype,
                load_format="auto" if args.model_path else "dummy",
                attention_backend="torch_native",
                page_size=1,
                max_total_tokens=128,
                max_running_requests=4,
                context_length=64,
                max_prefill_tokens=64,
                chunked_prefill_size=-1,
                mem_fraction_static=0.5,
                disable_cuda_graph=True,
                disable_overlap_schedule=True,
                disable_radix_cache=True,
            )
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            model_config = ModelConfig.from_server_args(server_args)
            init_distributed_environment(
                backend="nccl",
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
            )
            initialize_model_parallel(tensor_model_parallel_size=1, backend="nccl")
            initialize_dp_attention(server_args=server_args, model_config=model_config)
            runner = ModelRunner(
                model_config,
                mem_fraction_static=server_args.mem_fraction_static,
                gpu_id=0,
                tp_rank=0,
                tp_size=1,
                moe_ep_rank=0,
                moe_ep_size=1,
                pp_rank=0,
                pp_size=1,
                nccl_port=port,
                server_args=server_args,
                is_draft_worker=True,
            )
            expected_architecture = (
                "LlamaForCausalLM"
                if args.architecture == "llama"
                else "Qwen2ForCausalLM"
            )
            if type(runner.model).__name__ != expected_architecture:
                raise RuntimeError(
                    "loaded checkpoint architecture differs from --architecture"
                )
            if args.model_path is None:
                with torch.no_grad():
                    for name, parameter in runner.model.named_parameters():
                        if "norm" in name:
                            parameter.fill_(1)
                        else:
                            parameter.normal_(mean=0, std=0.12)
            report["evidence"] = validator(runner, checkpoint=bool(args.model_path))
            report.update(
                status="passed",
                device=torch.cuda.get_device_name(0),
                dtype=args.dtype,
                torch=torch.__version__,
                cuda=torch.version.cuda,
            )
    except Exception as exc:  # noqa: BLE001 -- strict standalone smoke reports failures, not skips
        report.update(
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=12),
        )
        print(json.dumps(report, indent=2))
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
