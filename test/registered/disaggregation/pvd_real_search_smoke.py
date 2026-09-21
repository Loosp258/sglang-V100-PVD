"""Real model K/Q -> two real V shard apps -> exact independent score oracle.

Uses fixed predicted token ids to isolate the retrieval gate. Draft model
quality and live scheduling are explicitly not covered by this check.
"""

import asyncio
from types import SimpleNamespace

import torch


def validate_real_search(runner):
    from pvd_search_roundtrip import verify_exact_roundtrip
    from sglang.srt.disaggregation.pvd.draft_forward_adapter import (
        DraftForwardAdapter,
        PrivatePoolAllocator,
    )
    from sglang.srt.disaggregation.pvd.draft_runner_sglang import DraftForwardInputs
    from sglang.srt.disaggregation.pvd.prediction import (
        CommittedPrefix,
        DraftConfig,
        FakeDraftProvider,
        PredictionPipeline,
        ProbeConfig,
    )
    from sglang.srt.disaggregation.pvd.target_probe import OfflineLlamaTargetProbe
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

    prefix = CommittedPrefix("real-search", (1, 4, 13, 7, 22), 0, "prefix-1")
    allocator = PrivatePoolAllocator(
        runner.req_to_token_pool, runner.token_to_kv_pool_allocator
    )
    adapter = DraftForwardAdapter(
        runner,
        architecture="LlamaForCausalLM",
        attention_backend="torch_native",
        bytes_per_token=256,
        device="cpu",
    )
    slot, rows = allocator.alloc_request(), []
    try:
        rows = allocator.alloc_kv(len(prefix.tokens))
        allocator.write_mapping(slot, 0, rows)
        adapter.forward(
            DraftForwardInputs(
                "extend",
                prefix.tokens,
                tuple(range(5)),
                (5,),
                (slot,),
                tuple(rows),
                (0,),
                (5,),
            )
        )
        # Repack the real token rows into page-size-2 storage staging. Last
        # token is deliberate poison padding, NOT a model-produced prompt K.
        components = []
        for component in (
            runner.token_to_kv_pool.k_buffer + runner.token_to_kv_pool.v_buffer
        ):
            stored = torch.full((6, *component.shape[1:]), 1e6, dtype=component.dtype)
            stored[:5] = component[rows].detach()
            components.append(stored)
        count = len(components) // 2
        prompt_pool = SimpleNamespace(
            k_buffer=components[:count], v_buffer=components[count:]
        )
    finally:
        allocator.clear_mapping(slot)
        allocator.free_kv(rows)
        allocator.free_request(slot)
    space = "real-search-fixture-target"
    config = ProbeConfig(space, (0, 1), head_count=4)
    budget = TransferBudget(2 << 20, 1)
    probe = OfflineLlamaTargetProbe(
        runner,
        config,
        target_model_id=space,
        max_tokens=16,
        max_predict_tokens=2,
        transient_bytes_bound=1 << 20,
        budget=budget,
    )
    draft = DraftConfig("fixed-prediction-fixture", predict_tokens=2)
    pipeline = PredictionPipeline(
        FakeDraftProvider(draft, tokens=(19, 27)), probe, draft, config
    )
    result = asyncio.run(verify_exact_roundtrip(prompt_pool, prefix, pipeline))
    assert budget.snapshot()["used_staging_bytes"] == 0
    return {
        **result,
        "prompt_k_and_query_q": "real target model",
        "draft": "fixed token fixture, not a loaded draft model",
    }
