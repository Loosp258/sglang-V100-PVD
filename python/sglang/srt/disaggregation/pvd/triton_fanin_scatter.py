"""One-kernel reorder for the uniform, whole-V-shard fan-in layout.

The caller owns and fences both GPU buffers. This kernel does not establish
Mooncake completion, publish a bank, or release any receive registration.
"""

import triton
import triton.language as tl


@triton.jit
def _scatter_uniform_rank_packed(
    source,
    destination,
    total_bytes,
    tokens: tl.constexpr,
    parts: tl.constexpr,
    source_token_bytes: tl.constexpr,
    component_source_bytes: tl.constexpr,
    part_source_bytes: tl.constexpr,
    component_destination_bytes: tl.constexpr,
    block_bytes: tl.constexpr,
):
    # A full Prompt receive region can exceed the signed 32-bit byte range.
    dest = tl.program_id(0).to(tl.int64) * block_bytes + tl.arange(0, block_bytes).to(
        tl.int64
    )
    component = dest // component_destination_bytes
    in_component = dest % component_destination_bytes
    token = in_component // (parts * source_token_bytes)
    in_token = in_component % (parts * source_token_bytes)
    part = in_token // source_token_bytes
    in_part = in_token % source_token_bytes
    src = (
        part * part_source_bytes
        + component * component_source_bytes
        + token * source_token_bytes
        + in_part
    )
    byte = tl.load(source + src, mask=dest < total_bytes, other=0)
    tl.store(destination + dest, byte, mask=dest < total_bytes)


def scatter_uniform_rank_packed(
    source, destination, *, components, parts, tokens, width
):
    """Launch only; the owner must retain buffers through CUDA completion."""
    total = destination.numel()
    _scatter_uniform_rank_packed[(triton.cdiv(total, 1024),)](
        source,
        destination,
        total,
        tokens,
        parts,
        width,
        tokens * width,
        components * tokens * width,
        tokens * parts * width,
        1024,
    )
