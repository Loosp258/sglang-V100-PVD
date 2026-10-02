"""Checked contiguous group views and one-launch KV graph input preparation."""

import torch
import triton
import triton.language as tl


def contiguous_batch_view(items):
    """Return only the exact adjacent span of equal contiguous input matrices."""
    first = items[0]
    storage = first.untyped_storage()
    base, count = first.storage_offset(), first.numel()
    if base + count * len(items) > storage.nbytes() // first.element_size():
        return None
    for group, data in enumerate(items):
        if (
            data.shape != first.shape
            or data.device != first.device
            or data.dtype != first.dtype
            or not data.is_contiguous()
            or data.untyped_storage().data_ptr() != storage.data_ptr()
            or data.storage_offset() != base + group * count
        ):
            return None
    return first.as_strided(
        (len(items) * first.shape[0], first.shape[1]),
        (first.shape[1], 1),
        storage_offset=base,
    )


@triton.jit(do_not_specialize=["capacity", "old", "delta", "dim", "elements"])
def _prepare(
    source,
    head_data,
    native_data,
    ids,
    capacity,
    old,
    delta,
    dim,
    elements,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < elements
    head = offset // (delta * dim)
    row, feature = (offset // dim) % delta, offset % dim
    values = tl.load(source + offset, valid, other=0)
    tl.store(head_data + (head * capacity + old + row) * dim + feature, values, valid)
    native_row = 4 * old + (head % 4) * delta + row
    tl.store(
        native_data + (head // 4) * (4 * capacity * dim) + native_row * dim + feature,
        values,
        valid,
    )
    tl.store(ids + head * capacity + old + row, native_row, valid & (feature == 0))


def prewarm(device):
    with torch.cuda.device(device):
        kernel = _prepare.warmup(
            torch.float32,
            torch.float32,
            torch.float32,
            torch.int64,
            2159,
            1792,
            367,
            128,
            2630656,
            BLOCK=256,
            num_warps=4,
            grid=(1,),
        )
        kernel._init_handles()


def prepare(buffers, source, old, delta):
    with torch.cuda.device(buffers.head_data.device):
        _prepare[(triton.cdiv(source.numel(), 256),)](
            source,
            buffers.head_data,
            buffers.native_data,
            buffers.ids,
            buffers.capacity,
            old,
            delta,
            buffers.dim,
            source.numel(),
            BLOCK=256,
            num_warps=4,
        )
