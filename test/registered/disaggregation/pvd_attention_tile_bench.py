"""Isolated one-layer online-attention latency baseline for the V100S path.

The benchmark excludes model forward, index search, RDMA, tensor allocation,
and the full-softmax oracle. Run the parity gate first for each selected length.
It measures the current Python/Torch implementation, not a fused kernel.
"""

import argparse
import json
import math
import statistics
import time

import torch
from pvd_attention_tile_parity import run_case
from sglang.srt.disaggregation.pvd import cuda_sparse_attention as attention
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping


def benchmark(length: int, warmups: int, repeats: int) -> dict:
    run_case(length)
    generator = torch.Generator().manual_seed(3000 + length)
    device = "cuda:0"
    q = torch.randn(28, 128, generator=generator).to(device, torch.float16)
    prompt_kv = torch.randn(4, 2, length, 128, generator=generator).to(
        device, torch.float16
    )
    generated_k = torch.randn(32, 4, 128, generator=generator).to(device, torch.float16)
    generated_v = torch.randn(32, 4, 128, generator=generator).to(device, torch.float16)
    rows = (19, 3, 17, 11)
    prompt = {(0, head): (None, prompt_kv[head]) for head in range(4)}
    mapping = QueryHeadMapping(28, 4)
    torch.cuda.synchronize(device)
    results = {}
    for chunk in (8, 64):
        output = torch.empty_like(q)
        data = attention.AttentionBuffers(q, generated_k, generated_v, output, rows)
        scratch = torch.empty(
            attention.scratch_elements(chunk, 128), device=device, dtype=torch.float32
        )
        for _ in range(warmups):
            attention._stream_attention(
                prompt, data, mapping, 0, 1 / math.sqrt(128), scratch, chunk, 128
            )
        torch.cuda.synchronize(device)
        wall_ms = []
        event_ms = []
        for _ in range(repeats):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            wall_start = time.perf_counter()
            start.record()
            attention._stream_attention(
                prompt, data, mapping, 0, 1 / math.sqrt(128), scratch, chunk, 128
            )
            end.record()
            end.synchronize()
            wall_ms.append((time.perf_counter() - wall_start) * 1000)
            event_ms.append(start.elapsed_time(end))
        results[str(chunk)] = {
            "wall_ms": wall_ms,
            "wall_median_ms": statistics.median(wall_ms),
            "cuda_event_ms": event_ms,
            "cuda_event_median_ms": statistics.median(event_ms),
        }
    return {
        "prompt_tokens": length,
        "warmups": warmups,
        "repeats": repeats,
        "tiles": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, action="append")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    lengths = args.length or [128, 511, 1923]
    if not torch.cuda.is_available():
        parser.error("a real CUDA device is required")
    if any(n <= 0 for n in lengths) or args.warmups < 0 or args.repeats <= 0:
        parser.error("lengths/repeats must be positive and warmups nonnegative")
    print(
        json.dumps(
            {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
            sort_keys=True,
        ),
        flush=True,
    )
    for length in lengths:
        print(
            json.dumps(benchmark(length, args.warmups, args.repeats), sort_keys=True),
            flush=True,
        )


if __name__ == "__main__":
    main()
