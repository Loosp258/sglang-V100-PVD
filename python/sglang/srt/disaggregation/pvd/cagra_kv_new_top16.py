"""Exact new-row Top-16 over existing FP32 scores, with self masking.

Each tile contributes its exact sixteen largest score/ID keys. The global
sixteen must be in their union; no similarity scores are recomputed.
"""

import threading

import torch
import triton
import triton.language as tl

_LOCK = threading.Lock()
_WARMED = set()


@triton.jit
def _keys(values, tokens, valid):
    bits = values.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    keys = (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - tokens.to(tl.uint32)).to(
        tl.uint64
    )
    return tl.where(valid, keys, 0)


@triton.jit
def _store(keys, output_scores, output_ids, offset):
    lane = tl.arange(0, 16)
    ordered = (keys >> 32).to(tl.uint32)
    bits = tl.where((ordered & 0x80000000) != 0, ordered ^ 0x80000000, ~ordered)
    tl.store(output_scores + offset + lane, bits.to(tl.float32, bitcast=True))
    tl.store(output_ids + offset + lane, 0xFFFFFFFF - keys.to(tl.uint32))


@triton.jit(do_not_specialize=["hs", "rs", "capacity", "begin", "width"])
def _direct(
    scores,
    output_scores,
    output_ids,
    hs,
    rs,
    capacity,
    begin,
    width,
    BLOCK: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    values = tl.load(
        scores + head * hs + row * rs + lane, lane < width, other=-float("inf")
    )
    keys = _keys(values, lane, (lane < width) & (lane != begin + row))
    _store(
        tl.topk(keys, 16),
        output_scores,
        output_ids,
        (head * capacity + begin + row) * 16,
    )


@triton.jit(do_not_specialize=["hs", "rs", "rows", "tiles", "begin", "width"])
def _partial(scores, workspace, hs, rs, rows, tiles, begin, width, TILE: tl.constexpr):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    lane = tl.arange(0, TILE)
    tokens = tile * TILE + lane
    values = tl.load(
        scores + head * hs + row * rs + tokens, tokens < width, other=-float("inf")
    )
    keys = _keys(values, tokens, (tokens < width) & (tokens != begin + row))
    selected = tl.topk(keys, 16)
    tl.store(
        workspace + ((head * rows + row) * tiles + tile) * 16 + tl.arange(0, 16),
        selected,
    )


@triton.jit(do_not_specialize=["rows", "tiles", "capacity", "begin", "active"])
def _finish(
    workspace,
    output_scores,
    output_ids,
    rows,
    tiles,
    capacity,
    begin,
    active,
    BLOCK: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    keys = tl.load(
        workspace + (head * rows + row) * tiles * 16 + lane, lane < active * 16, other=0
    )
    _store(
        tl.topk(keys, 16),
        output_scores,
        output_ids,
        (head * capacity + begin + row) * 16,
    )


def prewarm(device, *, tile=256, max_rows=8192):
    with _LOCK, torch.cuda.device(device):
        key = (torch.cuda.current_device(), tile, max_rows)
        if key in _WARMED:
            return
        if tile:
            kernel = _partial.warmup(
                torch.float32,
                torch.uint64,
                16384,
                8192,
                128,
                32,
                2048,
                2159,
                TILE=tile,
                num_warps=4,
                grid=(1, 1, 1),
            )
            kernel._init_handles()
        block = 16 if tile else 32
        limit = (
            triton.next_power_of_2(triton.cdiv(max_rows, tile) * 16)
            if tile
            else triton.next_power_of_2(max_rows)
        )
        while block <= limit:
            if tile:
                kernel = _finish.warmup(
                    torch.uint64,
                    torch.float32,
                    torch.int64,
                    128,
                    32,
                    8192,
                    2048,
                    9,
                    BLOCK=block,
                    num_warps=4,
                    grid=(1, 1),
                )
            else:
                kernel = _direct.warmup(
                    torch.float32,
                    torch.float32,
                    torch.int64,
                    16384,
                    8192,
                    8192,
                    2048,
                    2159,
                    BLOCK=block,
                    num_warps=4,
                    grid=(1, 1),
                )
            kernel._init_handles()
            block *= 2
        _WARMED.add(key)


def select(buffers, scores, begin, *, tile=256):
    rows, width = scores.shape[1:]
    with torch.cuda.device(scores.device):
        if not tile:
            _direct[(rows, buffers.heads)](
                scores,
                buffers.top_scores,
                buffers.neighbors,
                scores.stride(0),
                scores.stride(1),
                buffers.capacity,
                begin,
                width,
                BLOCK=triton.next_power_of_2(width),
                num_warps=4,
            )
            return
        workspace = buffers.new_top16_workspace
        tiles = workspace.shape[2]
        active = triton.cdiv(width, tile)
        _partial[(rows, buffers.heads, active)](
            scores,
            workspace,
            scores.stride(0),
            scores.stride(1),
            workspace.shape[1],
            tiles,
            begin,
            width,
            TILE=tile,
            num_warps=4,
        )
        _finish[(rows, buffers.heads)](
            workspace,
            buffers.top_scores,
            buffers.neighbors,
            workspace.shape[1],
            tiles,
            buffers.capacity,
            begin,
            active,
            BLOCK=triton.next_power_of_2(active * 16),
            num_warps=4,
        )
