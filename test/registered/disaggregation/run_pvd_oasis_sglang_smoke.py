"""One idle GPU: validate paired target execution with real SGLang 7B weights."""

import json
import threading

import torch
import torch.nn.functional as F

from run_pvd_cuda_probe_smoke import main


@torch.inference_mode()
def validate(runner, *, checkpoint=False):
    from sglang.srt.disaggregation.pvd.draft_forward_adapter import DraftForwardAdapter, PrivatePoolAllocator
    from sglang.srt.disaggregation.pvd.draft_runner_sglang import DraftForwardInputs
    from sglang.srt.disaggregation.pvd.oasis_qwen import PromptBank
    from sglang.srt.disaggregation.pvd.oasis_sglang import SGLangQwenPairedDecode
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

    model = runner.model
    allocator = PrivatePoolAllocator(runner.req_to_token_pool, runner.token_to_kv_pool_allocator)
    builder = DraftForwardAdapter(runner, architecture="Qwen2ForCausalLM",
        attention_backend="torch_native", bytes_per_token=28 * 4 * 128 * 2 * 4,
        device="cuda:0")
    prompt = (1, 4, 13, 7, 19, 27, 3, 8)
    slot = allocator.alloc_request()
    rows = allocator.alloc_kv(len(prompt) + 1)
    allocator.write_mapping(slot, 0, rows)
    hooks = []
    try:
        def forward(ids, start):
            batch = builder.build_forward_batch(DraftForwardInputs("extend", tuple(ids),
                tuple(range(start, start + len(ids))), (start + len(ids),), (slot,),
                tuple(rows[start:start + len(ids)]), (start,), (len(ids),)))
            runner.attn_backend.init_forward_metadata(batch)
            with forward_context(ForwardContext(attn_backend=runner.attn_backend)):
                return model.model(batch.input_ids, batch.positions, batch)
        forward(prompt, 0)
        pool = runner.token_to_kv_pool_allocator.get_kvcache()
        banks = [PromptBank(tuple(tuple(range(len(prompt))) for _ in range(4)),
            pool.get_key_buffer(layer)[rows[:len(prompt)]].transpose(0, 1).clone(),
            pool.get_value_buffer(layer)[rows[:len(prompt)]].transpose(0, 1).clone(),
            torch.ones(4, len(prompt), device="cuda", dtype=torch.bool)) for layer in range(28)]
        features = []
        for layer in (1, 13, 24):
            hooks.append(model.model.layers[layer].register_forward_hook(
                lambda module, args, output: features.append((output[0] + output[1]).clone())))
        ordinary = forward((19,), len(prompt))
        ordinary_logits = F.linear(ordinary, model.lm_head.weight)
        ordinary_features = torch.cat(features, dim=-1).unsqueeze(0)
        for hook in hooks:
            hook.remove()
        hooks.clear()
        torch.cuda.synchronize()
        # Bound verification storage: all mapping and allocator rows plus one
        # full actual token from every KV layer; no duplicate model weights.
        before_mapping = runner.req_to_token_pool.req_to_token.clone()
        before_free = runner.token_to_kv_pool_allocator.free_pages.clone()
        before_slots = list(runner.req_to_token_pool.free_slots)
        before_kv = [(pool.get_key_buffer(layer)[rows[-1]].clone(),
            pool.get_value_buffer(layer)[rows[-1]].clone()) for layer in range(28)]
        calls = {"qkv": [], "mlp": []}
        for layer in model.model.layers:
            hooks.append(layer.self_attn.qkv_proj.register_forward_pre_hook(
                lambda module, args: calls["qkv"].append(args[0].shape[0])))
            hooks.append(layer.mlp.register_forward_pre_hook(
                lambda module, args: calls["mlp"].append(args[0].shape[0])))
        published, consumed = [], []
        def bank(layer):
            consumed.append(layer)
            return banks[layer]
        decoder = SGLangQwenPairedDecode(runner, execution_lock=threading.RLock())
        logits, actual_features = decoder.step(19, 901, len(prompt), bank,
            publish=lambda layer, query, resident: published.append(layer))
        assert calls == {"qkv": [2] * 28, "mlp": [2] * 28}
        assert published == consumed == list(range(28))
        other = SGLangQwenPairedDecode(runner, execution_lock=threading.RLock())
        other_logits, other_features = other.step(19, 1719, len(prompt), banks)
        assert torch.equal(logits, other_logits)
        assert torch.equal(actual_features, other_features)
        torch.testing.assert_close(logits, ordinary_logits, rtol=5e-3, atol=0.15)
        torch.testing.assert_close(actual_features, ordinary_features, rtol=5e-3, atol=0.15)
        assert logits.argmax(-1).item() == ordinary_logits.argmax(-1).item()
        assert torch.equal(before_mapping, runner.req_to_token_pool.req_to_token)
        assert torch.equal(before_free, runner.token_to_kv_pool_allocator.free_pages)
        assert before_slots == runner.req_to_token_pool.free_slots
        for layer, (key, value) in enumerate(before_kv):
            assert torch.equal(key, pool.get_key_buffer(layer)[rows[-1]])
            assert torch.equal(value, pool.get_value_buffer(layer)[rows[-1]])
        assert all(len(history) == 1 for history in decoder.generated)
        return {"shared_qkv_and_mlp_calls": 28, "rows_per_call": 2,
            "actual_logit_max_abs_error": float((logits - ordinary_logits).abs().max()),
            "actual_feature_max_abs_error": float((actual_features - ordinary_features).abs().max()),
            "actual_argmax_equal": True, "candidate_isolation_bitwise": True,
            "actual_rows_committed_per_layer": 1, "formal_pools_unchanged": True,
            "per_layer_published_and_consumed": True,
            "formal_scheduler_admission_validated": False}
    finally:
        torch.cuda.synchronize()
        for hook in hooks:
            hook.remove()
        allocator.clear_mapping(slot)
        allocator.free_kv(rows)
        allocator.free_request(slot)
        torch.cuda.synchronize()


if __name__ == "__main__":
    raise SystemExit(main(validator=validate, schema="pvd-oasis-sglang-paired-v1"))
