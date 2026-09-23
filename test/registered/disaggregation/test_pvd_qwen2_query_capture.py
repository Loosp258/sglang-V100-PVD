"""Qwen2 must hand the batch-owned collector post-RoPE target Q."""

from types import SimpleNamespace

import torch
from sglang.srt.models.qwen2 import Qwen2Attention


def test_qwen2_capture_is_after_rope_and_before_attention():
    events = []
    positions = torch.tensor([2, 3])
    qkv = torch.tensor([[1.0, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

    class Attention:
        layer_id = 4

        def __call__(self, q, k, v, batch):
            events.append(("attention", q.clone()))
            return q

    class Capture:
        def capture(self, layer, seen_positions, q):
            assert layer == 4
            torch.testing.assert_close(seen_positions, positions)
            events.append(("capture", q.clone()))

    fake = SimpleNamespace(
        qkv_proj=lambda hidden: (qkv, None),
        q_size=2,
        kv_size=2,
        rotary_emb=lambda pos, q, k: (q + 10, k + 20),
        attn=Attention(),
        o_proj=lambda output: (output, None),
    )
    batch = SimpleNamespace(pvd_query_capture=Capture())
    output = Qwen2Attention.forward(fake, positions, torch.empty(2, 2), batch)
    assert [event for event, _ in events] == ["capture", "attention"]
    torch.testing.assert_close(events[0][1], qkv[:, :2] + 10)
    torch.testing.assert_close(output, events[1][1])


def test_qwen2_without_collector_keeps_normal_attention_order():
    calls = []

    class Attention:
        layer_id = 0

        def __call__(self, q, k, v, batch):
            calls.append("attention")
            return q

    fake = SimpleNamespace(
        qkv_proj=lambda hidden: (torch.ones(1, 6), None),
        q_size=2,
        kv_size=2,
        rotary_emb=lambda pos, q, k: (q, k),
        attn=Attention(),
        o_proj=lambda output: (output, None),
    )
    Qwen2Attention.forward(
        fake,
        torch.tensor([0]),
        torch.empty(1, 2),
        SimpleNamespace(pvd_query_capture=None),
    )
    assert calls == ["attention"]
