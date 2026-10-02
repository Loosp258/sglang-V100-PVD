"""Paired causal/history equivalence with real CPU SDPA; CUDA is a separate gate."""
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.pvd.oasis_attention import PairedLayerAttention
from sglang.srt.disaggregation.pvd.oasis_attention_workspace import PairedAttentionWorkspace


def workspace(**options):
    return PairedAttentionWorkspace(device='cpu', dtype=torch.float32, q_heads=4,
        kv_heads=2, head_dim=8, max_bank_rows=5, max_history=4, **options)


def bank(width):
    valid = torch.ones((2, width), dtype=torch.bool)
    valid[0, -1] = False
    return SimpleNamespace(keys=torch.randn(2, width, 8), values=torch.randn(2, width, 8),
                           valid=valid, completion=None)


def test_variable_spans_match_original_paired_attention_and_actual_only_history():
    torch.manual_seed(8542)
    owner = workspace()
    storage = owner.k.data_ptr(), owner.expanded_k.data_ptr(), owner.mask.data_ptr()
    histories = [[[]], [[]]]
    for width in (5, 2, 4, 3):
        resident = bank(width)
        q, k, v = torch.randn(2, 4, 8), torch.randn(2, 2, 8), torch.randn(2, 2, 8)
        contexts = [PairedLayerAttention(history, [resident], q_heads=4, kv_heads=2,
            head_dim=8, feature_layers=(0,), workspace=owner if mode else None)
            for mode, history in enumerate(histories)]
        outputs = [context.attention(0, q, k, v) for context in contexts]
        assert torch.equal(*outputs)
        for context in contexts:
            context.after_layer(0, torch.ones(2, 32))
            context.commit()
        for expected, actual in zip(histories[0][0], histories[1][0], strict=True):
            assert torch.equal(expected[0], actual[0]) and torch.equal(expected[1], actual[1])
            assert actual[0].shape == actual[1].shape == (2, 1, 8)
        assert storage == (owner.k.data_ptr(), owner.expanded_k.data_ptr(), owner.mask.data_ptr())
    owner.close()
    assert owner.closed and owner.k is owner.expanded_k is owner.mask is None


def test_current_branch_never_sees_predicted_kv_in_reused_mask():
    torch.manual_seed(18)
    owner = workspace()
    resident = bank(3)
    q, k, v = torch.randn(2, 4, 8), torch.randn(2, 2, 8), torch.randn(2, 2, 8)
    first = owner.attention(q, k, v, resident, [])
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[1].fill_(83)
    changed_v[1].fill_(-17)
    second = owner.attention(q, changed_k, changed_v, resident, [])
    assert torch.equal(first[0], second[0]) and not torch.equal(first[1], second[1])
    owner.close()


def test_budget_rejected_before_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('unadmitted attention allocation')
    monkeypatch.setattr(torch, 'zeros', forbidden)
    with pytest.raises(ValueError, match='scratch budget'):
        workspace(max_bytes=1)


def test_oversized_bank_and_closed_reuse_rejected():
    owner = workspace()
    q, k, v = torch.zeros(2, 4, 8), torch.zeros(2, 2, 8), torch.zeros(2, 2, 8)
    with pytest.raises(ValueError, match='scope'):
        owner.attention(q, k, v, bank(6), [])
    owner.close()
    with pytest.raises(RuntimeError, match='closed'):
        owner.attention(q, k, v, bank(3), [])


def test_capture_requires_explicit_cuda_and_no_silent_fallback():
    with pytest.raises(ValueError, match='CUDA graph'):
        workspace(graph=True)
    owner = workspace()
    with pytest.raises(RuntimeError, match='graph diagnostic'):
        owner.prime((4,))
    owner.close()
