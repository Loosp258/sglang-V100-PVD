"""Experimental one-token GQA attention kernel for contiguous Prompt KV.

This module is deliberately not wired into serving. It does not own the bank
read lease, generated-row pin, staging allocation, or CUDA completion fence.
The caller must establish those before this prototype can become a backend.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _one_token_gqa(
    Q,
    Prompt,
    GeneratedK,
    GeneratedV,
    Rows,
    Output,
    prompt_len,
    generated_len,
    kv_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    block_tokens: tl.constexpr,
):
    q_head = tl.program_id(0)
    kv_head = q_head // group_size
    token_offsets = tl.arange(0, block_tokens)
    dim_offsets = tl.arange(0, head_dim)
    query = tl.load(Q + q_head * head_dim + dim_offsets).to(tl.float32)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.full((head_dim,), 0.0, tl.float32)

    for start in range(0, prompt_len, block_tokens):
        tokens = start + token_offsets
        offsets = (
            kv_head * 2 * prompt_len * head_dim
            + tokens[:, None] * head_dim
            + dim_offsets[None, :]
        )
        mask = tokens[:, None] < prompt_len
        keys = tl.load(Prompt + offsets, mask=mask, other=0).to(tl.float32)
        values = tl.load(
            Prompt + offsets + prompt_len * head_dim, mask=mask, other=0
        ).to(tl.float32)
        logits = tl.sum(keys * query[None, :], axis=1) * scale
        logits = tl.where(tokens < prompt_len, logits, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(logits, axis=0))
        alpha = tl.exp(maximum - next_maximum)
        weights = tl.exp(logits - next_maximum)
        numerator = numerator * alpha + tl.sum(values * weights[:, None], axis=0)
        denominator = denominator * alpha + tl.sum(weights, axis=0)
        maximum = next_maximum

    for start in range(0, generated_len, block_tokens):
        tokens = start + token_offsets
        valid = tokens < generated_len
        pool_rows = tl.load(Rows + tokens, mask=valid, other=0)
        offsets = (
            pool_rows[:, None] * kv_heads * head_dim
            + kv_head * head_dim
            + dim_offsets[None, :]
        )
        keys = tl.load(GeneratedK + offsets, mask=valid[:, None], other=0).to(
            tl.float32
        )
        values = tl.load(GeneratedV + offsets, mask=valid[:, None], other=0).to(
            tl.float32
        )
        logits = tl.sum(keys * query[None, :], axis=1) * scale
        logits = tl.where(valid, logits, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(logits, axis=0))
        alpha = tl.exp(maximum - next_maximum)
        weights = tl.exp(logits - next_maximum)
        numerator = numerator * alpha + tl.sum(values * weights[:, None], axis=0)
        denominator = denominator * alpha + tl.sum(weights, axis=0)
        maximum = next_maximum

    tl.store(Output + q_head * head_dim + dim_offsets, numerator / denominator)


def one_token_gqa(
    q: torch.Tensor,
    prompt_kv: torch.Tensor,
    generated_k: torch.Tensor,
    generated_v: torch.Tensor,
    generated_rows: torch.Tensor,
    output: torch.Tensor,
    *,
    block_tokens: int = 64,
) -> None:
    """Launch only; the caller retains all buffers until stream completion.

    Prompt layout is [KV-head, K/V, token, head-dim]. Generated K/V use
    [pool-row, KV-head, head-dim] and rows are device-resident indices.
    """
    tensors = (q, prompt_kv, generated_k, generated_v, generated_rows, output)
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all inputs/output must be contiguous CUDA tensors")
    if any(t.device != q.device for t in tensors):
        raise ValueError("all inputs/output must be on the same CUDA device")
    if any(
        t.dtype != torch.float16
        for t in (q, prompt_kv, generated_k, generated_v, output)
    ):
        raise ValueError("Q/K/V/output must be float16")
    if generated_rows.dtype not in (torch.int32, torch.int64):
        raise ValueError("generated_rows must contain int32 or int64 indices")
    if q.ndim != 2 or prompt_kv.ndim != 4 or generated_k.ndim != 3:
        raise ValueError("invalid Q, Prompt, or generated-K rank")
    query_heads, head_dim = q.shape
    kv_heads, components, prompt_len, prompt_dim = prompt_kv.shape
    if (
        components != 2
        or prompt_len < 1
        or prompt_dim != head_dim
        or head_dim not in (64, 128)
        or query_heads < kv_heads
        or query_heads % kv_heads
        or generated_k.shape != generated_v.shape
        or generated_k.shape[1:] != (kv_heads, head_dim)
        or generated_rows.ndim != 1
        or output.shape != q.shape
        or block_tokens not in (8, 16, 32, 64, 128)
    ):
        raise ValueError("unsupported one-token GQA tensor layout")
    _one_token_gqa[(query_heads,)](
        q,
        prompt_kv,
        generated_k,
        generated_v,
        generated_rows,
        output,
        prompt_len,
        generated_rows.numel(),
        kv_heads,
        query_heads // kv_heads,
        head_dim,
        head_dim**-0.5,
        block_tokens,
        num_warps=4,
    )
