"""Standalone V100S microprobe for the opt-in bounded SDPA math core.

This does not activate a serving backend. It measures one synthetic layer and
observes CUDA allocation peaks, but does not prove peak bounds under serving.
"""

import argparse
import json
import math
import statistics
import time

import torch
from sglang.srt.disaggregation.pvd.cuda_sparse_attention import (
    AttentionBuffers,
    _sdpa_attention,
    _stream_attention,
    scratch_elements,
    sdpa_workspace_bytes,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping


def _measure(call, *, warmups, repeats):
    for _ in range(warmups):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--generated-tokens", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if (
        not 1 <= args.prompt_tokens <= 256
        or not 1 <= args.generated_tokens <= 128
        or not 1 <= args.repeats <= 20
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability(0) != (7, 0)
    ):
        parser.error(
            "bounded prompt/generated/repeats and a V100S-class SM70 GPU required"
        )

    torch.manual_seed(7)
    device, dim, kv_heads, q_heads = "cuda:0", 128, 4, 28
    scale = 1 / math.sqrt(dim)
    mapping = QueryHeadMapping(q_heads, kv_heads)
    q = torch.randn((q_heads, dim), device=device, dtype=torch.float16)
    prompt_k = torch.randn(
        (kv_heads, args.prompt_tokens, dim), device=device, dtype=torch.float16
    )
    prompt_v = torch.randn_like(prompt_k)
    pool_k = torch.randn((256, kv_heads, dim), device=device, dtype=torch.float16)
    pool_v = torch.randn_like(pool_k)
    row_tuple = tuple(range(1, args.generated_tokens + 1))
    groups = {
        (0, head): (None, torch.stack((prompt_k[head], prompt_v[head])))
        for head in range(kv_heads)
    }
    reference = torch.empty_like(q)
    candidate = torch.empty_like(q)
    buffers = AttentionBuffers(q, pool_k, pool_v, reference, row_tuple)
    scratch = torch.empty(scratch_elements(8, dim), device=device, dtype=torch.float32)
    max_tokens = args.prompt_tokens + args.generated_tokens
    keys = torch.empty((kv_heads, max_tokens, dim), device=device, dtype=torch.float16)
    values = torch.empty_like(keys)
    candidate_buffers = AttentionBuffers(q, pool_k, pool_v, candidate, row_tuple)

    def online():
        _stream_attention(groups, buffers, mapping, 0, scale, scratch, 8, dim)

    def sdpa():
        _sdpa_attention(
            groups, candidate_buffers, mapping, 0, scale, keys, values, max_tokens
        )

    try:
        online()
        sdpa()
        torch.cuda.synchronize()
        max_error = (reference.float() - candidate.float()).abs().max().item()
        if max_error > 0.02:
            raise ValueError(f"SDPA candidate differs from reference: {max_error}")
        online_seconds = _measure(online, warmups=1, repeats=args.repeats)
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        sdpa_seconds = _measure(sdpa, warmups=1, repeats=args.repeats)
        peak_extra = torch.cuda.max_memory_allocated() - allocated_before
    except Exception as exc:
        parser.exit(1, f"SDPA candidate probe failed: {exc}\n")
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "prompt_tokens": args.prompt_tokens,
                "generated_tokens": args.generated_tokens,
                "max_abs_error": max_error,
                "sdpa_reserved_bytes_estimate": sdpa_workspace_bytes(
                    max_tokens, kv_heads, q_heads, dim, torch.float16
                ),
                "sdpa_observed_peak_extra_bytes": peak_extra,
                "online_one_layer_median_seconds": online_seconds,
                "sdpa_one_layer_median_seconds": sdpa_seconds,
                "serving_validated": False,
                "memory_budget_validated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
