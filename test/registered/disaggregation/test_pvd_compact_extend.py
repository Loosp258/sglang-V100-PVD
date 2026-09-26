"""PVD-only compact TorchNative EXTEND with a previously stored prefix."""

import pytest
import torch
from sglang.srt.layers.attention import torch_native_backend as native


@pytest.mark.parametrize("prefix_len,extend_len", [(0, 5), (3, 1), (8, 2)])
@pytest.mark.parametrize("sliding_window", [None, 4])
@pytest.mark.parametrize("q_heads,kv_heads", [(2, 2), (4, 2)])
def test_compact_extend_matches_explicit_offset_causal_attention(
    prefix_len, extend_len, sliding_window, q_heads, kv_heads, monkeypatch
):
    torch.manual_seed(2901 + prefix_len + extend_len)
    seq_len, dim = prefix_len + extend_len, 8
    q = torch.randn(extend_len, q_heads, dim)
    k = torch.randn(seq_len + 1, kv_heads, dim)
    v = torch.randn(seq_len + 1, kv_heads, dim)
    mapping = torch.zeros((2, seq_len), dtype=torch.int64)
    mapping[1] = torch.arange(1, seq_len + 1)
    seen_query_rows = []
    real_sdpa = native.scaled_dot_product_attention

    def record_query_rows(query, *args, **kwargs):
        seen_query_rows.append(query.shape[-2])
        return real_sdpa(query, *args, **kwargs)

    monkeypatch.setattr(native, "scaled_dot_product_attention", record_query_rows)
    backend = native.TorchNativeAttnBackend.__new__(native.TorchNativeAttnBackend)
    output = torch.empty_like(q)
    backend._run_sdpa_forward_extend(
        q,
        output,
        k,
        v,
        mapping,
        torch.tensor([1]),
        torch.tensor([seq_len]),
        torch.tensor([prefix_len]),
        torch.tensor([extend_len]),
        scaling=dim**-0.5,
        enable_gqa=q_heads != kv_heads,
        causal=True,
        sliding_window_size=sliding_window,
        compact_queries=True,
    )
    assert seen_query_rows == [extend_len]

    grouped_k = k[1 : seq_len + 1].repeat_interleave(q_heads // kv_heads, dim=1)
    grouped_v = v[1 : seq_len + 1].repeat_interleave(q_heads // kv_heads, dim=1)
    scores = torch.einsum("thd,shd->ths", q, grouped_k) * dim**-0.5
    qpos = torch.arange(prefix_len, seq_len).unsqueeze(1)
    kpos = torch.arange(seq_len).unsqueeze(0)
    visible = kpos <= qpos
    if sliding_window is not None:
        visible &= kpos >= qpos - sliding_window
    expected = torch.einsum(
        "ths,shd->thd",
        scores.masked_fill(~visible.unsqueeze(1), float("-inf")).softmax(-1),
        grouped_v,
    )
    torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-5)


def test_compact_extend_removes_redundant_prefix_queries(monkeypatch):
    original = native.scaled_dot_product_attention
    seen = []

    def record(query, *args, **kwargs):
        seen.append(query.shape[-2])
        return original(query, *args, **kwargs)

    monkeypatch.setattr(native, "scaled_dot_product_attention", record)
    backend = native.TorchNativeAttnBackend.__new__(native.TorchNativeAttnBackend)
    q = torch.randn(2, 2, 8)
    k = torch.randn(9, 2, 8)
    v = torch.randn(9, 2, 8)
    mapping = torch.arange(1, 9).reshape(1, 8)
    common = (
        q,
        torch.empty_like(q),
        k,
        v,
        mapping,
        torch.tensor([0]),
        torch.tensor([8]),
        torch.tensor([6]),
        torch.tensor([2]),
    )
    backend._run_sdpa_forward_extend(*common, causal=True)
    backend._run_sdpa_forward_extend(*common, causal=True, compact_queries=True)
    assert seen == [8, 2]
