"""Experimental one-token GQA attention kernel for contiguous Prompt KV.

This module is deliberately not wired into serving. It does not own the bank
read lease, generated-row pin, staging allocation, or CUDA completion fence.
The caller must establish those before this prototype can become a backend.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _one_token_gqa(
    Q,
    Prompt,
    PromptPtrs,
    PromptLengths,
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
    pointer_table: tl.constexpr,
):
    q_head = tl.program_id(0)
    kv_head = q_head // group_size
    token_offsets = tl.arange(0, block_tokens)
    dim_offsets = tl.arange(0, head_dim)
    query = tl.load(Q + q_head * head_dim + dim_offsets).to(tl.float32)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.full((head_dim,), 0.0, tl.float32)

    if pointer_table:
        local_prompt_len = tl.load(PromptLengths + kv_head)
        prompt_address = tl.load(PromptPtrs + kv_head)
        prompt_base = tl.cast(prompt_address, tl.pointer_type(tl.float16))
    else:
        local_prompt_len = prompt_len
        prompt_base = Prompt + kv_head * 2 * prompt_len * head_dim

    for start in range(0, local_prompt_len, block_tokens):
        tokens = start + token_offsets
        offsets = tokens[:, None] * head_dim + dim_offsets[None, :]
        mask = tokens[:, None] < local_prompt_len
        keys = tl.load(prompt_base + offsets, mask=mask, other=0).to(tl.float32)
        values = tl.load(
            prompt_base + offsets + local_prompt_len * head_dim,
            mask=mask,
            other=0,
        ).to(tl.float32)
        logits = tl.sum(keys * query[None, :], axis=1) * scale
        logits = tl.where(tokens < local_prompt_len, logits, -float("inf"))
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
        prompt_kv,
        generated_rows,
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
        False,
        num_warps=4,
    )


@dataclass(frozen=True)
class PromptPointerView:
    """Pinned-by-caller view over per-head bank tensors; not a read lease.

    The holder must keep the bank's read lease alive through CUDA completion.
    The view retains Python tensor references but cannot prevent bank release.
    """

    sources: tuple[torch.Tensor, ...]
    pointers: torch.Tensor
    lengths: torch.Tensor
    head_dim: int

    @classmethod
    def from_groups(cls, groups, *, layer: int, kv_heads: int, device):
        sources = []
        selected = torch.device(device)
        if selected.type != "cuda" or type(kv_heads) is not int or kv_heads <= 0:
            raise ValueError("explicit CUDA device and positive KV-head count required")
        head_dim = None
        for head in range(kv_heads):
            if (layer, head) not in groups:
                raise ValueError("missing Prompt KV-head group")
            spec, tensor = groups[(layer, head)]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device != selected
                or tensor.dtype != torch.float16
                or tensor.ndim != 3
                or tensor.shape[0] != 2
                or tensor.shape[1] < 1
                or not tensor.is_contiguous()
            ):
                raise ValueError("Prompt group must own contiguous CUDA FP16 K/V")
            if spec is not None and len(spec.token_ids) != tensor.shape[1]:
                raise ValueError("Prompt group token IDs do not match tensor length")
            if head_dim is None:
                head_dim = tensor.shape[2]
            elif head_dim != tensor.shape[2]:
                raise ValueError("Prompt groups have different head dimensions")
            sources.append(tensor)
        if head_dim not in (64, 128):
            raise ValueError("unsupported Prompt head dimension")
        return cls(
            tuple(sources),
            torch.tensor(
                [t.data_ptr() for t in sources], device=selected, dtype=torch.int64
            ),
            torch.tensor(
                [t.shape[1] for t in sources], device=selected, dtype=torch.int32
            ),
            head_dim,
        )


def one_token_gqa_grouped(
    q: torch.Tensor,
    prompt: PromptPointerView,
    generated_k: torch.Tensor,
    generated_v: torch.Tensor,
    generated_rows: torch.Tensor,
    output: torch.Tensor,
    *,
    block_tokens: int = 64,
) -> None:
    """Zero-copy bank-source prototype; caller must hold its read lease."""
    if not isinstance(prompt, PromptPointerView) or not prompt.sources:
        raise ValueError("a live Prompt pointer view is required")
    tensors = (q, generated_k, generated_v, generated_rows, output)
    if any(
        t.device != prompt.pointers.device or not t.is_contiguous() for t in tensors
    ):
        raise ValueError("Q, generated KV, rows, output must be contiguous on bank GPU")
    kv_heads, head_dim = len(prompt.sources), prompt.head_dim
    if (
        q.dtype != torch.float16
        or generated_k.dtype != torch.float16
        or generated_v.dtype != torch.float16
        or output.dtype != torch.float16
        or generated_rows.dtype not in (torch.int32, torch.int64)
        or q.ndim != 2
        or q.shape[1] != head_dim
        or q.shape[0] % kv_heads
        or generated_k.shape != generated_v.shape
        or generated_k.ndim != 3
        or generated_k.shape[1:] != (kv_heads, head_dim)
        or generated_rows.ndim != 1
        or output.shape != q.shape
        or block_tokens not in (8, 16, 32, 64, 128)
    ):
        raise ValueError("unsupported grouped one-token GQA layout")
    _one_token_gqa[(q.shape[0],)](
        q,
        prompt.sources[0],
        prompt.pointers,
        prompt.lengths,
        generated_k,
        generated_v,
        generated_rows,
        output,
        0,
        generated_rows.numel(),
        kv_heads,
        q.shape[0] // kv_heads,
        head_dim,
        head_dim**-0.5,
        block_tokens,
        True,
        num_warps=4,
    )
