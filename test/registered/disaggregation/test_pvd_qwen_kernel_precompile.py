"""CPU policy checks only; a real V100S compile needs a CUDA worker."""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch
from sglang.srt.disaggregation.pvd import qwen_kernel_precompile as precompile


def _model_and_pool():
    rotary = NS(use_fallback_kernel=False, rotary_dim=128, is_neox_style=True)
    model = NS(
        model=NS(layers=[NS(self_attn=NS(rotary_emb=rotary))]),
        parameters=lambda: iter((torch.empty(1, dtype=torch.float16),)),
    )
    pool = NS(store_dtype=torch.float16, row_dim=512, same_kv_dim=True)
    return model, pool


def test_precompile_uses_loaded_layout_and_never_calls_model_or_pool(monkeypatch):
    from sglang.jit_kernel import activation, kvcache, rope

    calls = []
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        rope,
        "_jit_fused_rope_module",
        lambda neox, dim, dtype: calls.append(("rope", neox, dim, dtype)),
    )
    monkeypatch.setattr(
        activation,
        "_jit_activation_module",
        lambda dtype: calls.append(("activation", dtype)),
    )
    monkeypatch.setattr(
        kvcache,
        "can_use_store_cache",
        lambda row_bytes: calls.append(("kv_store", row_bytes)) or True,
    )
    model, pool = _model_and_pool()
    precompile.precompile_qwen_decode_kernels(model, pool, device="cuda:0")
    assert calls == [
        ("rope", True, 128, torch.float16),
        ("activation", torch.float16),
        ("kv_store", 1024),
    ]


@pytest.mark.parametrize("fault", ["fallback", "store_dtype", "layout"])
def test_unsupported_layout_refused_before_jit(monkeypatch, fault):
    from sglang.jit_kernel import activation

    model, pool = _model_and_pool()
    if fault == "fallback":
        model.model.layers[0].self_attn.rotary_emb.use_fallback_kernel = True
    elif fault == "store_dtype":
        pool.store_dtype = torch.float32
    else:
        pool.same_kv_dim = False
    monkeypatch.setattr(
        activation,
        "_jit_activation_module",
        lambda *_: pytest.fail("JIT called before layout validation"),
    )
    with pytest.raises(ValueError, match="unsupported"):
        precompile.precompile_qwen_decode_kernels(model, pool, device="cuda:0")


def test_kv_store_jit_refusal_is_not_reported_as_warmed(monkeypatch):
    from sglang.jit_kernel import activation, kvcache, rope

    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(rope, "_jit_fused_rope_module", lambda *_: object())
    monkeypatch.setattr(activation, "_jit_activation_module", lambda *_: object())
    monkeypatch.setattr(kvcache, "can_use_store_cache", lambda *_: False)
    model, pool = _model_and_pool()
    with pytest.raises(ValueError, match="unavailable"):
        precompile.precompile_qwen_decode_kernels(model, pool, device="cuda:0")
