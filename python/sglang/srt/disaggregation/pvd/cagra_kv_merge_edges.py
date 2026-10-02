"""Exact old-row neighbor merge with selective native adjacency writes."""

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "score_head_stride",
        "score_row_stride",
        "capacity",
        "old",
        "delta",
    ]
)
def _merge_edges(
    scores,
    cached_scores,
    cached_ids,
    token_ids,
    mapped,
    graph,
    score_head_stride,
    score_row_stride,
    capacity,
    old,
    delta,
    PRUNE: tl.constexpr,
    RING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    cache = head * capacity * 16 + row * 16
    prior = tl.load(cached_scores + cache + lane, lane < 16, other=-float("inf"))
    prior_ids = tl.load(cached_ids + cache + lane, lane < 16, other=0).to(tl.uint32)
    added = tl.load(
        scores + head * score_head_stride + row * score_row_stride + lane - 16,
        (lane >= 16) & (lane < 16 + delta),
        other=-float("inf"),
    )
    values = tl.where(lane < 16, prior, added)
    ids = tl.where(lane < 16, prior_ids, (old + lane - 16).to(tl.uint32))
    bits = values.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    keys = (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - ids).to(tl.uint64)
    keys = tl.where(lane < 16 + delta, keys, 0)
    improve = True
    if PRUNE:
        new_best = tl.max(tl.where(lane >= 16, keys, 0), 0)
        old_worst = tl.min(
            tl.where(lane < 16, keys, tl.full((BLOCK,), 0xFFFFFFFFFFFFFFFF, tl.uint64)),
            0,
        )
        following = tl.gather(keys, tl.minimum(lane + 1, 15), 0)
        cached_ordered = tl.min(
            tl.where(lane < 15, keys >= following, True).to(tl.int32), 0
        )
        improve = (new_best > old_worst) | (cached_ordered == 0)
    selected_ids = prior_ids
    if improve:
        keys = tl.sort(keys, descending=True)
        selected = (keys >> 32).to(tl.uint32)
        restored = tl.where(
            (selected & 0x80000000) != 0, selected ^ 0x80000000, ~selected
        ).to(tl.float32, bitcast=True)
        selected_ids = 0xFFFFFFFF - keys.to(tl.uint32)
        tl.store(cached_scores + cache + lane, restored, lane < 16)
        tl.store(cached_ids + cache + lane, selected_ids, lane < 16)
    # Old immutable K preserves its previously mapped edges when selected IDs
    # stay unchanged. Appending only changes the ring at row 0 and old - 1.
    exact_width = 14 if RING else 16
    changed = tl.max(
        ((lane < exact_width) & (selected_ids != prior_ids)).to(tl.int32), 0
    )
    if RING:
        changed = changed | (row == 0) | (row == old - 1)
    if changed:
        token = selected_ids.to(tl.int64)
        if RING:
            total = old + delta
            token = tl.where(lane == 14, (row + total - 1) % total, token)
            token = tl.where(lane == 15, (row + 1) % total, token)
        target = tl.load(token_ids + head * capacity + token, lane < 16, other=0)
        source = tl.load(token_ids + head * capacity + row)
        tl.store(mapped + cache + lane, target, lane < 16)
        tl.store(
            graph + (head // 4) * (4 * capacity * 16) + source * 16 + lane,
            target.to(tl.int32),
            lane < 16,
        )


@triton.jit(do_not_specialize=["capacity", "old", "total"])
def _write_new(
    neighbors,
    ids,
    mapped,
    graph,
    capacity,
    old,
    total,
    RING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    head = tl.program_id(1)
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, edge = old + offset // 16, offset % 16
    mask = row < total
    address = head * capacity * 16 + row * 16 + edge
    token = tl.load(neighbors + address, mask, other=0)
    if RING:
        token = tl.where(edge == 14, (row + total - 1) % total, token)
        token = tl.where(edge == 15, (row + 1) % total, token)
    target = tl.load(ids + head * capacity + token, mask, other=0)
    source = tl.load(ids + head * capacity + row, mask, other=0)
    tl.store(mapped + address, target, mask)
    tl.store(
        graph + (head // 4) * (4 * capacity * 16) + source * 16 + edge,
        target.to(tl.int32),
        mask,
    )


def prewarm(device, *, prune, routing_edges):
    with torch.cuda.device(device):
        for block in (32, 64, 128, 256):
            kernel = _merge_edges.warmup(
                torch.float32,
                torch.float32,
                torch.int64,
                torch.int64,
                torch.int64,
                torch.int32,
                262144,
                128,
                2159,
                2048,
                111,
                PRUNE=prune,
                RING=bool(routing_edges),
                BLOCK=block,
                num_warps=4,
                grid=(1, 1),
            )
            kernel._init_handles()
        kernel = _write_new.warmup(
            torch.int64,
            torch.int64,
            torch.int64,
            torch.int32,
            2159,
            2048,
            2159,
            RING=bool(routing_edges),
            BLOCK=256,
            grid=(1, 1),
        )
        kernel._init_handles()


def merge_edges(buffers, scores, old, delta):
    with torch.cuda.device(buffers.head_data.device):
        _merge_edges[(old, buffers.heads)](
            scores,
            buffers.top_scores,
            buffers.neighbors,
            buffers.ids,
            buffers.mapped,
            buffers.native_graph,
            scores.stride(0),
            scores.stride(1),
            buffers.capacity,
            old,
            delta,
            PRUNE=buffers.small_tail_prune,
            RING=bool(buffers.routing_edges),
            BLOCK=triton.next_power_of_2(16 + delta),
            num_warps=4,
        )


def write_new(buffers, old, total):
    with torch.cuda.device(buffers.head_data.device):
        _write_new[(triton.cdiv((total - old) * 16, 256), buffers.heads)](
            buffers.neighbors,
            buffers.ids,
            buffers.mapped,
            buffers.native_graph,
            buffers.capacity,
            old,
            total,
            RING=bool(buffers.routing_edges),
            BLOCK=256,
        )
