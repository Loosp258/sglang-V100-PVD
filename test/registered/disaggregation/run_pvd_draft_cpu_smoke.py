"""Opt-in, strict real-ModelRunner check; no mocks, downloads, or GPU claims.

Run in a serving-capable Linux Python environment with this checkout on
PYTHONPATH. A tiny randomly initialized Llama is a test fixture, NOT a choice
of production draft model. Import/initialization failures are errors, not skips.
The process owns all distributed state and should not run inside a server.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile

import torch

# Selected before any SGLang imports; this is the repository's real CPU
# dispatch mechanism, not a monkeypatch of an operator or model method.
os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


def main() -> None:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.disaggregation.pvd.draft_forward_adapter import (
        DraftForwardAdapter,
        PrivatePoolAllocator,
    )
    from sglang.srt.disaggregation.pvd.draft_runner_sglang import (
        DraftForwardInputs,
        SGLangDraftHandle,
    )
    from sglang.srt.disaggregation.pvd.draft_sglang import DraftCapabilities
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    from transformers import GenerationConfig, LlamaConfig

    torch.set_num_threads(2)
    torch.manual_seed(8128)
    with tempfile.TemporaryDirectory(prefix="pvd-draft-cpu-") as directory:
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            architectures=["LlamaForCausalLM"],
            tie_word_embeddings=False,
        )
        config.save_pretrained(directory)
        GenerationConfig(bos_token_id=1, eos_token_id=2).save_pretrained(directory)
        args = ServerArgs(
            model_path=directory,
            device="cpu",
            dtype="float32",
            load_format="dummy",
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
        model_config = ModelConfig.from_server_args(args)
        # A draft runner reuses process groups normally created by D's target
        # worker. This standalone test initializes real TP1 Gloo groups instead.
        init_distributed_environment(
            backend="gloo",
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")
        initialize_dp_attention(server_args=args, model_config=model_config)
        runner = ModelRunner(
            model_config,
            mem_fraction_static=args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            moe_ep_rank=0,
            moe_ep_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=port,
            server_args=args,
            is_draft_worker=True,
        )
        # Dummy loader weights can be so small that a broken attention path
        # barely moves logits. Make the fixture deterministic and sensitive
        # to context; still no checkpoint/model download and no quality claim.
        with torch.no_grad():
            for name, parameter in runner.model.named_parameters():
                if "norm" in name:
                    parameter.fill_(1.0)
                else:
                    parameter.normal_(mean=0.0, std=0.12)
        adapter = DraftForwardAdapter(
            runner,
            architecture="LlamaForCausalLM",
            attention_backend="torch_native",
            bytes_per_token=2 * 2 * 2 * 8 * 4,
            device="cpu",
        )
        req_pool = runner.req_to_token_pool
        kv_pool = runner.token_to_kv_pool_allocator
        before = (len(req_pool.free_slots), kv_pool.available_size())
        errors = []
        canary_deltas = []

        def full_prefix(tokens):
            allocator = PrivatePoolAllocator(req_pool, kv_pool)
            slot = allocator.alloc_request()
            rows = []
            try:
                rows = allocator.alloc_kv(len(tokens))
                allocator.write_mapping(slot, 0, rows)
                return adapter.forward(
                    DraftForwardInputs(
                        "extend",
                        tuple(tokens),
                        tuple(range(len(tokens))),
                        (len(tokens),),
                        (slot,),
                        tuple(rows),
                        (0,),
                        (len(tokens),),
                    )
                ).clone()
            finally:
                allocator.clear_mapping(slot)
                allocator.free_kv(rows)
                allocator.free_request(slot)

        for prefix in ((1, 5, 9), (1, 12, 7, 4, 21)):
            allocator = PrivatePoolAllocator(req_pool, kv_pool)
            slot = allocator.alloc_request()
            rows = []
            try:
                rows = allocator.alloc_kv(len(prefix))
                allocator.write_mapping(slot, 0, rows)
                cached = adapter.forward(
                    DraftForwardInputs(
                        "extend",
                        prefix,
                        tuple(range(len(prefix))),
                        (len(prefix),),
                        (slot,),
                        tuple(rows),
                        (0,),
                        (len(prefix),),
                    )
                )
                tokens = list(prefix)
                for step in range(3):
                    reference = full_prefix(tokens)
                    assert cached.shape == reference.shape == (64,)
                    assert torch.isfinite(cached).all()
                    torch.testing.assert_close(cached, reference, atol=2e-5, rtol=2e-4)
                    errors.append(float((cached - reference).abs().max()))
                    if step == 2:
                        break
                    # Fixed non-greedy tokens also exercise nontrivial cache reuse.
                    token = (17, 33)[step]
                    new_rows = allocator.alloc_kv(1)
                    rows.extend(new_rows)
                    allocator.write_mapping(slot, len(tokens), new_rows)
                    tokens.append(token)
                    cached = adapter.forward(
                        DraftForwardInputs(
                            "decode",
                            (token,),
                            (len(tokens) - 1,),
                            (len(tokens),),
                            (slot,),
                            tuple(new_rows),
                        )
                    )
                    # A corrupted prefix map must affect this actual backend.
                    # Restore it before the next comparison; the current row
                    # is rewritten identically by this repeated forward.
                    saved = req_pool.req_to_token[slot, : len(tokens) - 1].clone()
                    try:
                        req_pool.req_to_token[slot, : len(tokens) - 1] = rows[0]
                        broken = adapter.forward(
                            DraftForwardInputs(
                                "decode",
                                (token,),
                                (len(tokens) - 1,),
                                (len(tokens),),
                                (slot,),
                                tuple(new_rows),
                            )
                        )
                        delta = float((cached - broken).abs().max())
                        assert delta > 1e-4, "fixture cannot detect a corrupted KV map"
                        canary_deltas.append(delta)
                    finally:
                        req_pool.req_to_token[slot, : len(tokens) - 1] = saved
                    # The canary also wrote this token's deeper-layer KV using
                    # the corrupt context. Recompute it after restoring the map
                    # before a later token can read those rows.
                    cached = adapter.forward(
                        DraftForwardInputs(
                            "decode",
                            (token,),
                            (len(tokens) - 1,),
                            (len(tokens),),
                            (slot,),
                            tuple(new_rows),
                        )
                    )
            finally:
                allocator.clear_mapping(slot)
                allocator.free_kv(rows)
                allocator.free_request(slot)
            assert not torch.count_nonzero(req_pool.req_to_token[slot])
            assert (len(req_pool.free_slots), kv_pool.available_size()) == before

        # Exercise the actual branch handle, not just hand-built forward inputs.
        caps = DraftCapabilities(
            architectures=("LlamaForCausalLM",),
            attention_backends=("torch_native",),
            max_prefix_tokens=16,
            max_predict_tokens=3,
        )
        for interrupted in (False, True, False):
            handle = SGLangDraftHandle(
                "smoke",
                adapter,
                PrivatePoolAllocator(req_pool, kv_pool),
                max_prefix_tokens=16,
                max_tokens=3,
                capabilities=caps,
            )
            try:
                prepared = handle.prepare_prefix((1, 11, 22, 7))
                if not interrupted:
                    predicted = handle.generate(prepared, 3)
                    reference_tokens = [1, 11, 22, 7]
                    for token in predicted:
                        assert token == int(full_prefix(reference_tokens).argmax())
                        reference_tokens.append(token)
            finally:
                slot = handle.request_index
                handle.release()
                handle.release()  # idempotence on real pools
            assert handle.released
            assert not torch.count_nonzero(req_pool.req_to_token[slot])
            assert (len(req_pool.free_slots), kv_pool.available_size()) == before

        probe_evidence = None
        if "--probe" in sys.argv[1:]:
            from pvd_target_probe_smoke import validate_target_probe

            probe_evidence = validate_target_probe(runner, full_prefix)
        search_evidence = None
        if "--search" in sys.argv[1:]:
            from pvd_real_search_smoke import validate_real_search

            search_evidence = validate_real_search(runner)
        sparse_decode_evidence = None
        controlled_decode_evidence = None
        batch_decode_evidence = None
        if "--sparse-decode" in sys.argv[1:]:
            from pvd_sparse_decode_smoke import validate_sparse_decode

            sparse_decode_evidence = validate_sparse_decode(runner)
        if "--controlled-decode" in sys.argv[1:]:
            from pvd_controlled_decode_smoke import validate_controlled_decode

            controlled_decode_evidence = validate_controlled_decode(runner)
        if "--batch-decode" in sys.argv[1:]:
            from pvd_batch_decode_smoke import validate_batch_decode

            batch_decode_evidence = validate_batch_decode(runner)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "device": "cpu",
                    "dtype": "float32",
                    "fixture": "random tiny Llama, not a production model",
                    "backend": type(runner.attn_backend).__name__,
                    "forward_count": adapter.forward_count,
                    "comparisons": len(errors),
                    "max_abs_error": max(errors),
                    "corrupt_mapping_min_logit_delta": min(canary_deltas),
                    "real_handle_success_early_release_reuse": True,
                    "pool_capacity_restored": True,
                    "gpu_rdma_latency_validated": False,
                    "probe_evidence": probe_evidence,
                    "search_evidence": search_evidence,
                    "sparse_decode_evidence": sparse_decode_evidence,
                    "controlled_decode_evidence": controlled_decode_evidence,
                    "batch_decode_evidence": batch_decode_evidence,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
