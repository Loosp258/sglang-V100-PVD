"""Compile known Qwen2 Decode JIT modules without touching request or KV data.

This is an opt-in startup optimization. It moves compilation of exact dtype/
layout specializations out of the first PVD request; it is not a model warmup
or evidence that every first-use cost has been removed.
"""

import logging
import time

import torch

logger = logging.getLogger(__name__)


def precompile_qwen_decode_kernels(model, kv_pool, *, device):
    """Compile position, RoPE, activation and KV-store modules for Decode.

    No model forward, pool-row allocation, KV write or request mutation occurs.
    The caller checks the concrete Qwen2 architecture and owns startup failure.
    """
    from sglang.jit_kernel.activation import _jit_activation_module
    from sglang.jit_kernel.clamp_position import _jit_clamp_position_module
    from sglang.jit_kernel.kvcache import can_use_store_cache
    from sglang.jit_kernel.rope import _jit_fused_rope_module

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Qwen Decode precompile requires CUDA")
    layer = model.model.layers[0]
    rotary = layer.self_attn.rotary_emb
    dtype = next(model.parameters()).dtype
    if (
        rotary.use_fallback_kernel
        or rotary.rotary_dim <= 0
        or not isinstance(rotary.is_neox_style, bool)
        or kv_pool.store_dtype != dtype
        or kv_pool.row_dim <= 0
        or not kv_pool.same_kv_dim
    ):
        raise ValueError("unsupported Qwen Decode JIT specialization")
    row_bytes = kv_pool.row_dim * kv_pool.store_dtype.itemsize
    with torch.cuda.device(device):
        for name, compile_module in (
            # ScheduleBatch.prepare_for_extend creates int64 seq_lens, and
            # ForwardBatch.init_new's Decode path clamps that same tensor.
            ("clamp_position", lambda: _jit_clamp_position_module(torch.int64)),
            (
                "rope",
                lambda: _jit_fused_rope_module(
                    rotary.is_neox_style, rotary.rotary_dim, dtype
                ),
            ),
            ("activation", lambda: _jit_activation_module(dtype)),
            ("kv_store", lambda: can_use_store_cache(row_bytes)),
        ):
            started = time.perf_counter()
            result = compile_module()
            if name == "kv_store" and result is not True:
                raise ValueError("Qwen Decode KV-store JIT specialization unavailable")
            logger.info(
                "PVD Qwen Decode JIT precompile: module=%s elapsed_seconds=%.6f",
                name,
                time.perf_counter() - started,
            )
