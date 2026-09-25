"""Real-CUDA numerical and one-layer timing gate for the fused prototype."""

import json
import math
import statistics
import time

import torch
from pvd_attention_tile_parity import PROMPT_LENGTHS
from sglang.srt.disaggregation.pvd.triton_sparse_attention import one_token_gqa


def check_case(prompt_tokens: int, *, large_logits: bool = False) -> dict:
    generator = torch.Generator().manual_seed(4000 + prompt_tokens)
    device = "cuda:0"
    q = torch.randn(28, 128, generator=generator).to(device, torch.float16)
    if large_logits:
        q.mul_(30)
    prompt = torch.randn(4, 2, prompt_tokens, 128, generator=generator).to(
        device, torch.float16
    )
    generated_k = torch.randn(32, 4, 128, generator=generator).to(device, torch.float16)
    generated_v = torch.randn(32, 4, 128, generator=generator).to(device, torch.float16)
    rows = torch.tensor((19, 3, 17, 11), device=device, dtype=torch.int64)
    expected = torch.empty_like(q)
    for head in range(28):
        kv_head = head // 7
        keys = torch.cat(
            (prompt[kv_head, 0], generated_k[rows, kv_head]), dim=0
        ).float()
        values = torch.cat(
            (prompt[kv_head, 1], generated_v[rows, kv_head]), dim=0
        ).float()
        scores = torch.mv(keys, q[head].float()) / math.sqrt(128)
        expected[head] = torch.mv(values.T, torch.softmax(scores, dim=0))

    original = (q.clone(), prompt.clone(), generated_k.clone(), generated_v.clone())
    errors = {}
    for tile in (8, 64):
        output = torch.empty_like(q)
        one_token_gqa(
            q, prompt, generated_k, generated_v, rows, output, block_tokens=tile
        )
        torch.cuda.synchronize(device)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, expected, atol=0.025, rtol=0.002)
        errors[str(tile)] = (output.float() - expected.float()).abs().max().item()
    for actual, before in zip(
        (q, prompt, generated_k, generated_v), original, strict=True
    ):
        torch.testing.assert_close(actual, before, atol=0, rtol=0)
    return {
        "prompt_tokens": prompt_tokens,
        "large_logits": large_logits,
        "errors": errors,
    }


def benchmark_long_case() -> dict:
    length = 1923
    device = "cuda:0"
    q = torch.randn(28, 128, device=device, dtype=torch.float16)
    prompt = torch.randn(4, 2, length, 128, device=device, dtype=torch.float16)
    generated_k = torch.randn(32, 4, 128, device=device, dtype=torch.float16)
    generated_v = torch.randn_like(generated_k)
    rows = torch.tensor((19, 3, 17, 11), device=device, dtype=torch.int64)
    output = torch.empty_like(q)
    for _ in range(2):
        one_token_gqa(q, prompt, generated_k, generated_v, rows, output)
    torch.cuda.synchronize(device)
    wall_ms = []
    event_ms = []
    for _ in range(5):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        wall_start = time.perf_counter()
        start.record()
        one_token_gqa(q, prompt, generated_k, generated_v, rows, output)
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - wall_start) * 1000)
        event_ms.append(start.elapsed_time(end))
    return {
        "prompt_tokens": length,
        "wall_ms": wall_ms,
        "wall_median_ms": statistics.median(wall_ms),
        "cuda_event_ms": event_ms,
        "cuda_event_median_ms": statistics.median(event_ms),
    }


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("real CUDA device required")
    print(
        json.dumps(
            {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
            sort_keys=True,
        ),
        flush=True,
    )
    for length in PROMPT_LENGTHS:
        print(json.dumps(check_case(length), sort_keys=True), flush=True)
    print(json.dumps(check_case(65, large_logits=True), sort_keys=True), flush=True)
    print(json.dumps({"benchmark": benchmark_long_case()}, sort_keys=True), flush=True)
