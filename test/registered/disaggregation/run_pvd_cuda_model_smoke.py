"""Strict real CUDA sparse model forward; shares offline tiny-Llama setup.

No RDMA/CAGRA/serving/performance claim. Runs initial full Prompt plus a sparse
refresh and compares every real model layer with independent dense SDPA math.
"""


def validate(runner):
    import threading

    import torch
    from sglang.srt.disaggregation.pvd.cuda_model_attention import (
        CUDAModelPools,
        make_cuda_sparse_backend,
    )
    from sglang.srt.disaggregation.pvd.cuda_runtime_group import CUDARuntimeInstallGroup
    from sglang.srt.disaggregation.pvd.cuda_sparse_attention import (
        CUDASparseAttentionWorkspace,
    )
    from sglang.srt.disaggregation.pvd.cuda_working_set import CUDASparseWorkingSet
    from sglang.srt.disaggregation.pvd.draft_forward_adapter import (
        DraftForwardAdapter,
        PrivatePoolAllocator,
    )
    from sglang.srt.disaggregation.pvd.draft_runner_sglang import DraftForwardInputs
    from sglang.srt.disaggregation.pvd.sparse_payload import (
        SparseKVPayload,
        SparseKVSpec,
    )
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
        ResourceGuard,
        TransferBudget,
    )

    device, dtype = "cuda:0", next(runner.model.parameters()).dtype
    config = runner.model.config
    layers, heads, qheads, dim = (
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.num_attention_heads,
        runner.model_config.head_dim,
    )
    allocator = PrivatePoolAllocator(
        runner.req_to_token_pool, runner.token_to_kv_pool_allocator
    )
    slot, rows = allocator.alloc_request(), allocator.alloc_kv(9)
    allocator.write_mapping(slot, 0, rows)
    adapter = DraftForwardAdapter(
        runner,
        architecture="LlamaForCausalLM",
        attention_backend="torch_native",
        bytes_per_token=layers * heads * dim * 2 * 4,
        device=device,
    )
    lock = threading.Lock()
    with lock:
        adapter.forward(
            DraftForwardInputs(
                "extend",
                (1, 4, 13, 7),
                (0, 1, 2, 3),
                (4,),
                (slot,),
                tuple(rows[:4]),
                (0,),
                (4,),
            )
        )
        torch.cuda.synchronize(device)
    pool = runner.token_to_kv_pool
    prompt = {
        (layer, head): torch.stack(
            (
                pool.get_key_buffer(layer)[rows[:4], head],
                pool.get_value_buffer(layer)[rows[:4], head],
            )
        )
        for layer in range(layers)
        for head in range(heads)
    }
    bank_budget, scratch_budget, output_budget = (
        TransferBudget(16 << 20, 4) for _ in range(3)
    )
    bank = CUDASparseWorkingSet(
        device=device,
        dtype=dtype,
        budget=bank_budget,
        request_id="r",
        incarnation="inc",
        entry_transfer_id="entry",
        layout_fingerprint="layout",
        expected_groups=tuple(prompt),
        prompt_tokens=4,
        head_dim=dim,
        max_union_tokens=2,
    )
    group = CUDARuntimeInstallGroup(
        {0: bank},
        interval=4,
        lead_tokens=1,
        peer_epochs={0: "peer"},
        timeout_seconds=300,
        max_pending_events=8,
        max_pending_bytes=65536,
    )
    peer = group._peers[0]

    def install(count, tokens):
        epoch = group.begin(count)
        parts = [value[:, list(tokens)].contiguous() for value in prompt.values()]
        packed = torch.cat([part.reshape(-1) for part in parts])
        size = parts[0].numel()
        payloads = [
            SparseKVPayload(
                SparseKVSpec(
                    "r",
                    "inc",
                    epoch.operation_id,
                    epoch.target_tokens,
                    "entry",
                    "index",
                    "mapping",
                    "layout",
                    layer,
                    head,
                    tokens,
                ),
                packed[i * size : (i + 1) * size].view(2, len(tokens), dim),
            )
            for i, (layer, head) in enumerate(prompt)
        ]
        source = ResourceGuard(packed, lambda: None)
        group.stage(epoch, 0, payloads, source_guard=source)
        source.request_release()
        assert group.try_install(epoch, {0: epoch.target_tokens})
        assert group.can_decode(epoch.target_tokens)

    install(0, (0, 1, 2, 3))
    workspace = CUDASparseAttentionWorkspace(
        device=device, dtype=dtype, head_dim=dim, chunk_tokens=2, budget=scratch_budget
    )
    backend = make_cuda_sparse_backend(
        runner,
        workspace=workspace,
        execution_lock=lock,
        output_budget=output_budget,
        max_batch_size=1,
    )
    dense_backend = runner.attn_backend
    runner.attn_backend = backend
    errors, output_tokens, release_events = [], [], []

    def retire():
        allocator.clear_mapping(slot)
        allocator.free_kv(rows)
        allocator.free_request(slot)
        torch.cuda.synchronize(device)
        release_events.append(True)

    owner = ResourceGuard(CUDAModelPools(runner.req_to_token_pool, pool), retire)
    native = backend.consumer.forward_decode
    tolerance = 5e-3 if dtype == torch.float16 else 3e-4

    def compare(q, k, v, layer, batch, save_kv_cache=True):
        # Independent oracle: dense SDPA of selected original-position Prompt
        # plus prior generated rows and the current model K/V. No tiled math.
        count = int(batch.positions[0].item()) - 4
        query = q.reshape(qheads, dim)
        current_k, current_v = k.reshape(heads, dim), v.reshape(heads, dim)
        expected = []
        with peer.read(count) as groups:
            for head in range(qheads):
                kvhead = head // (qheads // heads)
                selected = groups[(layer.layer_id, kvhead)][1]
                prior_rows = rows[4 : 4 + count]
                keys = torch.cat(
                    (
                        selected[0],
                        pool.get_key_buffer(layer.layer_id)[prior_rows, kvhead],
                        current_k[kvhead].view(1, dim),
                    )
                )
                values = torch.cat(
                    (
                        selected[1],
                        pool.get_value_buffer(layer.layer_id)[prior_rows, kvhead],
                        current_v[kvhead].view(1, dim),
                    )
                )
                expected.append(
                    torch.nn.functional.scaled_dot_product_attention(
                        query[head].view(1, 1, dim).float(),
                        keys.unsqueeze(0).float(),
                        values.unsqueeze(0).float(),
                        scale=layer.scaling,
                    ).reshape(dim)
                )
        expected = torch.stack(expected)
        actual = native(q, k, v, layer, batch, save_kv_cache).reshape(qheads, dim)
        torch.testing.assert_close(
            actual.float(), expected, rtol=tolerance, atol=tolerance
        )
        errors.append(float((actual.float() - expected).abs().max()))
        return actual.reshape(1, -1)

    backend.consumer.forward_decode = compare
    token = 19
    try:
        for count in range(5):
            if count == 4:
                install(3, (1, 3))
            with group.model_forward(
                backend.consumer, slot=slot, decode_tokens=count, pool_owner=owner
            ):
                logits = adapter.forward(
                    DraftForwardInputs(
                        "decode",
                        (token,),
                        (4 + count,),
                        (5 + count,),
                        (slot,),
                        (rows[4 + count],),
                        (),
                        (),
                    )
                )
                assert torch.isfinite(logits).all()
                next_token = int(logits.argmax().item())
            token = next_token  # Runtime has accepted the completed forward.
            output_tokens.append(token)
            assert output_budget.snapshot()["used_staging_bytes"] == 0
            assert not lock.locked()
        assert len(errors) == layers * 5 and len(output_tokens) == 5
    finally:
        runner.attn_backend = dense_backend
        # UNKNOWN is process-fatal in this standalone smoke: do not invent
        # cleanup safety or refund guarded rows after failed completion.
        if not backend.consumer.snapshot()["quarantine"]:
            owner.request_release()
            workspace.close()
            group.close()
    assert release_events == [True]
    assert all(
        b.snapshot()["used_staging_bytes"] == 0
        for b in (bank_budget, scratch_budget, output_budget)
    )
    return {
        "model_forwards": 5,
        "layer_oracle_checks": len(errors),
        "max_attention_abs_error": max(errors),
        "initial_full_and_sparse_refresh": True,
        "allocator_retirement_completed": True,
        "tokens": output_tokens,
    }


def main(argv=None):
    from run_pvd_cuda_probe_smoke import main as run

    return run(argv, validator=validate, schema="pvd-cuda-sparse-model-v1")


if __name__ == "__main__":
    raise SystemExit(main())
