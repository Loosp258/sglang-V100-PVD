"""Opt-in CUDA kernels for exact small-tail KV graph maintenance.

The caller owns and budgets every tensor. Kernels use the current Torch stream;
the backend must prove completion before publishing or releasing these buffers.
"""

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "score_head_stride",
        "score_row_stride",
        "capacity",
        "begin",
        "old",
        "delta",
    ]
)
def _merge_neighbors(
    scores,
    cached_scores,
    cached_ids,
    score_head_stride,
    score_row_stride,
    capacity,
    begin,
    old,
    delta,
    PRUNE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    cache = head * capacity * 16 + (begin + row) * 16
    prior = tl.load(cached_scores + cache + lane, lane < 16, other=-float("inf"))
    added = tl.load(
        scores + head * score_head_stride + row * score_row_stride + lane - 16,
        (lane >= 16) & (lane < 16 + delta),
        other=-float("inf"),
    )
    values = tl.where(lane < 16, prior, added)
    ids = tl.where(
        lane < 16,
        tl.load(cached_ids + cache + lane, lane < 16, other=0).to(tl.uint32),
        (old + lane - 16).to(tl.uint32),
    )
    # Encode float order and token ID in one sortable key. Equal scores prefer
    # the smaller token ID.
    bits = values.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    keys = (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - ids).to(tl.uint64)
    keys = tl.where(lane < 16 + delta, keys, 0)
    # Compare the same total-order keys used by the full sort, including ID
    # tie breaks and signed zero. No new key can enter Top-16 when its maximum
    # is below the weakest cached key; cached values/IDs then stay untouched.
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
        # Initial torch.topk can return equal-score IDs in arbitrary order.
        # Canonicalize those rows once, matching the unpruned sort exactly.
        improve = (new_best > old_worst) | (cached_ordered == 0)
    if improve:
        keys = tl.sort(keys, descending=True)
        selected = (keys >> 32).to(tl.uint32)
        restored = tl.where(
            (selected & 0x80000000) != 0,
            selected ^ 0x80000000,
            ~selected,
        ).to(tl.float32, bitcast=True)
        tl.store(cached_scores + cache + lane, restored, lane < 16)
        tl.store(cached_ids + cache + lane, 0xFFFFFFFF - keys.to(tl.uint32), lane < 16)


@triton.jit(do_not_specialize=["capacity", "total"])
def _write_graph(
    neighbors,
    ids,
    mapped,
    graph,
    capacity,
    total,
    RING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    head = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, edge = offsets // 16, offsets % 16
    mask = row < total
    token = tl.load(neighbors + head * capacity * 16 + offsets, mask, other=0)
    if RING:
        token = tl.where(edge == 14, (row + total - 1) % total, token)
        token = tl.where(edge == 15, (row + 1) % total, token)
    target = tl.load(ids + head * capacity + token, mask, other=0)
    source = tl.load(ids + head * capacity + row, mask, other=0)
    tl.store(mapped + head * capacity * 16 + offsets, target, mask)
    tl.store(
        graph + (head // 4) * (4 * capacity * 16) + source * 16 + edge,
        target.to(tl.int32),
        mask,
    )


def prewarm(device, *, max_rows, routing_edges, prune=False):
    """Compile/load all enabled tail widths before accepting requests.

    Dtype arguments become Triton's MockTensor descriptors. No data tensors or
    device workspace are allocated and no kernel is launched by warmup().
    Runtime sizes remain unspecialized so new Prompt lengths reuse the kernels.
    """
    with torch.cuda.device(device):
        block = 32
        while block <= triton.next_power_of_2(16 + max_rows):
            kernel = _merge_neighbors.warmup(
                torch.float32,
                torch.float32,
                torch.int64,
                16384,
                127,
                2047,
                3,
                1536,
                127,
                PRUNE=prune,
                BLOCK=block,
                num_warps=4,
                grid=(1, 1),
            )
            # warmup compiles; initialize also resolves the CUDA module/function
            # handles, avoiding their first-use load on the request thread.
            kernel._init_handles()
            block *= 2
        kernel = _write_graph.warmup(
            torch.int64,
            torch.int64,
            torch.int64,
            torch.int32,
            2047,
            2047,
            RING=bool(routing_edges),
            BLOCK=256,
            grid=(1, 1),
        )
        kernel._init_handles()


def merge_neighbors(buffers, scores, begin, old, delta):
    # V's worker threads may retain a different current device. Torch operators
    # guard their tensor device internally; a Triton launch needs our guard.
    with torch.cuda.device(buffers.head_data.device):
        _merge_neighbors[(scores.shape[1], buffers.heads)](
            scores,
            buffers.top_scores,
            buffers.neighbors,
            scores.stride(0),
            scores.stride(1),
            buffers.capacity,
            begin,
            old,
            delta,
            PRUNE=getattr(buffers, "small_tail_prune", False),
            BLOCK=triton.next_power_of_2(16 + delta),
            num_warps=4,
        )


def write_graph(buffers, total):
    with torch.cuda.device(buffers.head_data.device):
        _write_graph[(triton.cdiv(total * 16, 256), buffers.heads)](
            buffers.neighbors,
            buffers.ids,
            buffers.mapped,
            buffers.native_graph,
            buffers.capacity,
            total,
            RING=bool(buffers.routing_edges),
            BLOCK=256,
        )
