"""Standalone V100S numerical gate for Qwen2.5-7B-shaped online attention.

This checks one layer against an independent full-softmax oracle. It does not
measure end-to-end accuracy, service latency, or the fused-kernel performance.
Run with CUDA_VISIBLE_DEVICES set to the isolated validation GPU.
"""

import json
import math

import torch
from sglang.srt.disaggregation.pvd import cuda_sparse_attention as attention
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping

PROMPT_LENGTHS = (1, 7, 8, 9, 63, 64, 65, 127, 128, 129, 511, 1923)


def run_case(prompt_tokens: int, *, device: str = "cuda:0") -> dict[str, float | int]:
    generator = torch.Generator().manual_seed(2000 + prompt_tokens)
    q = torch.randn(28, 128, generator=generator).to(device, torch.float16)
    prompt_kv = torch.randn(4, 2, prompt_tokens, 128, generator=generator).to(
        device, torch.float16
    )
    generated_k = torch.randn(32, 4, 128, generator=generator).to(device, torch.float16)
    generated_v = torch.randn(32, 4, 128, generator=generator).to(device, torch.float16)
    rows = (19, 3, 17, 11)
    mapping = QueryHeadMapping(28, 4)
    prompt = {(0, head): (None, prompt_kv[head]) for head in range(4)}
    scale = 1 / math.sqrt(128)
    expected = torch.empty_like(q)
    for head in range(28):
        kv_head = mapping.kv_head_for(head)
        keys = torch.cat(
            (prompt_kv[kv_head, 0], generated_k[list(rows), kv_head]), dim=0
        ).float()
        values = torch.cat(
            (prompt_kv[kv_head, 1], generated_v[list(rows), kv_head]), dim=0
        ).float()
        scores = torch.mv(keys, q[head].float()) * scale
        expected[head] = torch.mv(values.T, torch.softmax(scores, dim=0))

    before = (q.clone(), prompt_kv.clone(), generated_k.clone(), generated_v.clone())
    outputs = []
    errors = []
    for chunk in (8, 64):
        output = torch.empty_like(q)
        data = attention.AttentionBuffers(q, generated_k, generated_v, output, rows)
        scratch = torch.empty(
            attention.scratch_elements(chunk, 128), device=device, dtype=torch.float32
        )
        attention._stream_attention(
            prompt, data, mapping, 0, scale, scratch, chunk, 128
        )
        torch.cuda.synchronize(device)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, expected, atol=0.025, rtol=0.002)
        errors.append((output.float() - expected.float()).abs().max().item())
        outputs.append(output.clone())

    torch.testing.assert_close(outputs[0], outputs[1], atol=0.025, rtol=0.002)
    for actual, original in zip(
        (q, prompt_kv, generated_k, generated_v), before, strict=True
    ):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)
    return {
        "prompt_tokens": prompt_tokens,
        "tile8_max_abs_error": errors[0],
        "tile64_max_abs_error": errors[1],
        "tiles_max_abs_difference": (outputs[0].float() - outputs[1].float())
        .abs()
        .max()
        .item(),
    }


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device is required for this hardware gate")
    print(
        json.dumps(
            {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
            sort_keys=True,
        ),
        flush=True,
    )
    for length in PROMPT_LENGTHS:
        print(json.dumps(run_case(length), sort_keys=True), flush=True)
