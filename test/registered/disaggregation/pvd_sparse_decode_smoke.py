"""Actual Llama forwards through the opt-in sparse CPU attention adapter.

Native complete-KV logits are an independent all-selection oracle. A manual
softmax hook checks every sparse attention output, including head-specific
selections and generated KV. No Scheduler, GPU, transport or TP claims.
"""

import torch


def validate_sparse_decode(runner):
    from sglang.srt.disaggregation.pvd.draft_forward_adapter import (
        DraftForwardAdapter,
        PrivatePoolAllocator,
    )
    from sglang.srt.disaggregation.pvd.draft_runner_sglang import DraftForwardInputs
    from sglang.srt.disaggregation.pvd.sparse_cpu_backend import (
        SparseDecodeBinding,
        make_offline_sparse_backend,
    )
    from sglang.srt.disaggregation.pvd.sparse_payload import (
        SparseKVPayload,
        SparseKVSpec,
    )
    from sglang.srt.disaggregation.pvd.sparse_working_set import CPUSparseWorkingSet
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

    native = runner.attn_backend
    backend = make_offline_sparse_backend(runner)
    prefix, continuation = (1, 4, 13, 7, 22), (19, 27, 31)
    prompt_len = len(prefix)
    config = runner.model.config
    layers, heads, dim = (
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.hidden_size // config.num_attention_heads,
    )
    pool = runner.token_to_kv_pool
    req_pool = runner.req_to_token_pool
    free_before = (
        len(req_pool.free_slots),
        runner.token_to_kv_pool_allocator.available_size(),
    )
    adapter = DraftForwardAdapter(
        runner,
        architecture="LlamaForCausalLM",
        attention_backend="torch_native",
        bytes_per_token=256,
        device="cpu",
    )
    attention_errors = []
    active = {}

    def check_attention(module, args, output):
        if not active:
            return
        q, _, _, batch = args[:4]
        layer, slot = module.layer_id, active["slot"]
        pos = int(batch.positions[0])
        generated_rows = req_pool.req_to_token[slot, prompt_len : pos + 1].long()
        with active["bank"].read() as groups:
            expected = []
            queries = q.reshape(config.num_attention_heads, dim)
            for head in range(config.num_attention_heads):
                kv_head = head // (config.num_attention_heads // heads)
                _, data = groups[(layer, kv_head)]
                keys = torch.cat(
                    (data[0], pool.get_key_buffer(layer)[generated_rows, kv_head])
                )
                vals = torch.cat(
                    (data[1], pool.get_value_buffer(layer)[generated_rows, kv_head])
                )
                expected.append(
                    ((keys @ queries[head]) * module.scaling).softmax(0) @ vals
                )
            expected = torch.stack(expected).reshape_as(output)
        torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-4)
        attention_errors.append(float((output - expected).abs().max()))
        if active.get("inject_failure"):
            raise RuntimeError("injected after actual sparse attention")

    hooks = [
        layer.self_attn.attn.register_forward_hook(check_attention)
        for layer in runner.model.model.layers
    ]

    def run(mode):
        allocator = PrivatePoolAllocator(req_pool, runner.token_to_kv_pool_allocator)
        slot, rows, bank = allocator.alloc_request(), [], None
        budget = TransferBudget(1 << 20, 2)
        try:
            rows = allocator.alloc_kv(prompt_len)
            allocator.write_mapping(slot, 0, rows)
            adapter.forward(
                DraftForwardInputs(
                    "extend",
                    prefix,
                    tuple(range(prompt_len)),
                    (prompt_len,),
                    (slot,),
                    tuple(rows),
                    (0,),
                    (prompt_len,),
                )
            )
            if mode != "native":
                stored = {
                    layer: torch.stack(
                        (
                            pool.get_key_buffer(layer)[rows],
                            pool.get_value_buffer(layer)[rows],
                        )
                    ).clone()
                    for layer in range(layers)
                }
                bank = CPUSparseWorkingSet(
                    request_id="sparse-smoke",
                    incarnation=f"inc-{mode}",
                    entry_transfer_id="fixture",
                    layout_fingerprint="fixture-layout",
                    expected_groups=tuple(
                        (layer, head)
                        for layer in range(layers)
                        for head in range(heads)
                    ),
                    prompt_tokens=prompt_len,
                    head_dim=dim,
                    max_union_tokens=2,
                    budget=budget,
                )

                def stage(boundary):
                    payloads = []
                    try:
                        for layer in range(layers):
                            for head in range(heads):
                                tokens = (
                                    tuple(range(prompt_len))
                                    if boundary == 0
                                    else (
                                        (layer + head) % (prompt_len - 1),
                                        prompt_len - 1,
                                    )
                                )
                                spec = SparseKVSpec(
                                    "sparse-smoke",
                                    f"inc-{mode}",
                                    f"op-{boundary}",
                                    boundary,
                                    "fixture",
                                    "index",
                                    "mapping",
                                    "fixture-layout",
                                    layer,
                                    head,
                                    tokens,
                                )
                                payloads.append(
                                    SparseKVPayload(
                                        spec, stored[layer][:, list(tokens), head]
                                    )
                                )
                        bank.stage(payloads)
                        bank.install(boundary)
                    finally:
                        for payload in payloads:
                            payload.close()

                stage(0)
                # A consumer accidentally reading the ordinary Prompt pool/map
                # now gets poison. Bank copies remain valid and must be used.
                req_pool.req_to_token[slot, :prompt_len] = -1
                for layer in range(layers):
                    pool.get_key_buffer(layer)[rows] = float("nan")
                    pool.get_value_buffer(layer)[rows] = float("nan")
            logits = []
            for step, token in enumerate(continuation):
                if mode == "sparse" and step == 1:
                    stage(1)
                previous_rows = rows[prompt_len:]
                before_generated = [
                    (
                        pool.get_key_buffer(l)[previous_rows].clone(),
                        pool.get_value_buffer(l)[previous_rows].clone(),
                    )
                    for l in range(layers)
                ]
                new_rows = allocator.alloc_kv(1)
                rows.extend(new_rows)
                position = prompt_len + step
                allocator.write_mapping(slot, position, new_rows)
                inputs = DraftForwardInputs(
                    "decode",
                    (token,),
                    (position,),
                    (position + 1,),
                    (slot,),
                    tuple(new_rows),
                )
                if bank is None:
                    output = adapter.forward(inputs)
                else:
                    active.update(
                        bank=bank, slot=slot, inject_failure=mode == "failure"
                    )
                    runner.attn_backend = backend
                    try:
                        with backend.consumer.bind(
                            [
                                SparseDecodeBinding(
                                    slot, "sparse-smoke", f"inc-{mode}", position, bank
                                )
                            ]
                        ):
                            output = adapter.forward(inputs)
                    finally:
                        runner.attn_backend = native
                        active.clear()
                for layer, (before_k, before_v) in enumerate(before_generated):
                    torch.testing.assert_close(
                        pool.get_key_buffer(layer)[previous_rows],
                        before_k,
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        pool.get_value_buffer(layer)[previous_rows],
                        before_v,
                        rtol=0,
                        atol=0,
                    )
                assert torch.isfinite(output).all()
                logits.append(output.clone())
            return logits
        finally:
            runner.attn_backend = native
            active.clear()
            if bank is not None:
                bank.close()
            assert budget.snapshot()["used_staging_bytes"] == 0
            allocator.clear_mapping(slot)
            allocator.free_kv(rows)
            allocator.free_request(slot)

    try:
        dense, full, sparse = run("native"), run("full"), run("sparse")
        checked_attention_count = len(attention_errors)
        try:
            run("failure")
        except RuntimeError as exc:
            assert "injected after actual sparse attention" in str(exc)
        else:
            raise AssertionError("failure injection was not reached")
    finally:
        for hook in hooks:
            hook.remove()
    for expected, actual in zip(dense, full, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(sparse[0], dense[0], atol=2e-5, rtol=2e-4)
    delta = max(
        float((s - d).abs().max()) for s, d in zip(sparse[1:], dense[1:], strict=True)
    )
    assert delta > 1e-4, "fixture cannot detect a sparse selection being ignored"
    assert checked_attention_count == 2 * len(continuation) * layers
    assert len(attention_errors) == checked_attention_count + 1
    assert (
        len(req_pool.free_slots),
        runner.token_to_kv_pool_allocator.available_size(),
    ) == free_before
    assert runner.attn_backend is native
    return {
        "status": "passed",
        "real_model_forwards": adapter.forward_count,
        "full_selection_max_logit_error": max(
            float((a - b).abs().max()) for a, b in zip(dense, full, strict=True)
        ),
        "attention_checks": len(attention_errors),
        "max_attention_error": max(attention_errors),
        "sparse_vs_full_logit_delta": delta,
        "poisoned_prompt_pool_not_read": True,
        "prior_generated_kv_unchanged": True,
        "failed_forward_releases_readers_and_restores_backend": True,
        "gpu_tp_scheduler_memory_savings_validated": False,
    }
