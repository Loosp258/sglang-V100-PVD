"""Strict V100S draft execution smoke, using the existing CUDA model loader.

Random tiny Llama/Qwen2 weights unless --model-path is explicitly supplied.
The process owns its model and groups; it must never run inside a server.
This validates a private draft forward/cleanup, not production activation,
draft quality, latency overlap or a memory peak bound.
"""

import torch


def validate(runner, *, checkpoint=False):
    from sglang.srt.disaggregation.pvd.draft_forward_adapter import (
        DraftForwardAdapter,
        PrivatePoolAllocator,
    )
    from sglang.srt.disaggregation.pvd.draft_memory import (
        measure_draft_retained_tensors,
    )
    from sglang.srt.disaggregation.pvd.draft_runner_sglang import (
        DraftForwardInputs,
        SGLangDraftRunnerFactory,
    )
    from sglang.srt.disaggregation.pvd.draft_sglang import DraftCapabilities

    request_pool = runner.req_to_token_pool
    kv_allocator = runner.token_to_kv_pool_allocator
    kv_pool = runner.token_to_kv_pool
    before = (len(request_pool.free_slots), kv_allocator.available_size())
    retained = measure_draft_retained_tensors(runner)
    bytes_per_token = sum(
        tensor[0].numel() * tensor.element_size()
        for tensor in kv_pool.k_buffer + kv_pool.v_buffer
    )
    adapter = DraftForwardAdapter(
        runner,
        architecture=type(runner.model).__name__,
        attention_backend="torch_native",
        bytes_per_token=bytes_per_token,
        device="cuda:0",
        transient_bytes_bound=64 << 20,
    )
    factory = SGLangDraftRunnerFactory(
        adapter,
        PrivatePoolAllocator(request_pool, kv_allocator),
        capabilities=DraftCapabilities(
            architectures=(type(runner.model).__name__,),
            attention_backends=("torch_native",),
            max_prefix_tokens=16,
            max_predict_tokens=2,
        ),
        persistent_bytes=retained.total_bytes,
        max_tokens=2,
    )
    prefix = (1, 4, 13, 7)

    def recompute(tokens):
        allocator = PrivatePoolAllocator(request_pool, kv_allocator)
        slot = allocator.alloc_request()
        rows = []
        completed = False
        try:
            rows = allocator.alloc_kv(len(tokens))
            allocator.write_mapping(slot, 0, rows)
            logits = adapter.forward(
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
            )
            result = int(logits.argmax().item())
            adapter.drain()
            completed = True
            return result
        finally:
            # A failed CUDA forward may still be using these rows. This is a
            # standalone process: leave them owned until process exit rather
            # than returning possibly-live storage to another branch.
            if completed:
                allocator.clear_mapping(slot)
                allocator.free_kv(rows)
                allocator.free_request(slot)

    expected_first = recompute(prefix)
    expected_second = recompute(prefix + (expected_first,))
    handle = factory.open(branch_id="cuda-smoke", prefix_tokens=4, max_tokens=2)
    try:
        predicted = tuple(handle.generate(handle.prepare_prefix(prefix), 2))
        assert predicted == (expected_first, expected_second)
        assert torch.cuda.is_available()
    finally:
        handle.release()
        handle.release()
    assert (len(request_pool.free_slots), kv_allocator.available_size()) == before
    assert factory.opened_count == 1
    return {
        "checkpoint_weights": bool(checkpoint),
        "architecture": type(runner.model).__name__,
        "draft_forwards": adapter.forward_count,
        "predicted_tokens": predicted,
        "full_prefix_recompute_matches": True,
        "private_pool_capacity_restored": True,
        "known_retained_tensor_bytes": retained.total_bytes,
        "cuda_peak_memory_or_latency_validated": False,
        "production_pipeline_activated": False,
    }


def main(argv=None):
    from run_pvd_cuda_probe_smoke import main as run

    return run(argv, validator=validate, schema="pvd-cuda-draft-v1")


if __name__ == "__main__":
    raise SystemExit(main())
