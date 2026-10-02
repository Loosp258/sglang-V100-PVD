"""One CUDA launch for uniform component-major Prompt K extraction."""

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["total_rows", "first", "rows", "dim", "heads", "elements"]
)
def _extract(
    source,
    target,
    centered,
    means,
    total_rows,
    first,
    rows,
    dim,
    heads,
    elements,
    CENTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < elements
    head = offset // (rows * dim)
    token, feature = (offset // dim) % rows, offset % dim
    source_offset = (
        ((head // heads) * total_rows + first + token) * heads * dim
        + (head % heads) * dim
        + feature
    )
    value = tl.load(source + source_offset, valid, other=0).to(tl.float32)
    tl.store(target + offset, value, valid)
    if CENTER:
        mean = tl.load(means + head * dim + feature, valid, other=0)
        tl.store(centered + offset, value - mean, valid)


def extract(
    source, *, layers, heads, total_rows, first, rows, dim, means=None, outputs=None
):
    # Allocation and launch use the source device even on a worker whose current
    # CUDA device is its peer rank. The caller charges the owned copy first.
    with torch.cuda.device(source.device):
        if outputs is None:
            target = torch.empty(
                (layers * heads, rows, dim), dtype=torch.float32, device=source.device
            )
            centered = torch.empty_like(target) if means is not None else None
        else:
            target, centered = outputs
        _extract[(triton.cdiv(target.numel(), 256),)](
            source,
            target,
            centered if centered is not None else target,
            means if means is not None else target,
            total_rows,
            first,
            rows,
            dim,
            heads,
            target.numel(),
            CENTER=means is not None,
            BLOCK=256,
            num_warps=4,
        )
    return target, centered
