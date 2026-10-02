"""Shared paired attention: causal rows, atomic commit and completion proof."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

path = Path(__file__).resolve().parents[3] / "python/sglang/srt/disaggregation/pvd/oasis_attention.py"
spec = importlib.util.spec_from_file_location("oasis_attention_test", path)
oasis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oasis)


class PairedAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8128)
        self.history = [[], []]
        self.banks = [SimpleNamespace(keys=torch.randn(2, 5, 3),
            values=torch.randn(2, 5, 3), valid=torch.ones(2, 5, dtype=torch.bool),
            completion=None) for _ in range(2)]
        self.q, self.k, self.v = torch.randn(2, 4, 3), torch.randn(2, 2, 3), torch.randn(2, 2, 3)

    def context(self, **kwargs):
        return oasis.PairedLayerAttention(self.history, self.banks, q_heads=4,
            kv_heads=2, head_dim=3, feature_layers=(0, 1), **kwargs)

    def test_actual_row_matches_single_token_and_ignores_candidate(self):
        context = self.context()
        actual = context.attention(0, self.q, self.k, self.v)[0]
        bank = self.banks[0]
        keys = torch.cat([bank.keys, self.k[:1].transpose(0, 1)], dim=1)
        values = torch.cat([bank.values, self.v[:1].transpose(0, 1)], dim=1)
        expected = F.scaled_dot_product_attention(self.q[:1].transpose(0, 1)[None],
            keys.repeat_interleave(2, dim=0)[None], values.repeat_interleave(2, dim=0)[None])
        torch.testing.assert_close(actual, expected.reshape(-1))
        q, k, v = self.q.clone(), self.k.clone(), self.v.clone()
        q[1], k[1], v[1] = 999, -999, 777
        other = self.context().attention(0, q, k, v)
        self.assertTrue(torch.equal(actual, other[0]))
        self.assertFalse(torch.equal(context.pending[0][0], self.k[1, :, None, :]))

    def test_publish_is_per_layer_before_attention_and_commits_only_complete_actual(self):
        calls = []
        def publish(layer, q, bank):
            calls.append(layer)
            self.assertEqual(len(context.pending), layer)
            self.assertTrue(torch.equal(q, self.q[1]))
        context = self.context(publish=publish)
        for layer in range(2):
            context.attention(layer, self.q, self.k, self.v)
            context.after_layer(layer, torch.ones(2, 12) * layer)
        self.assertEqual(self.history, [[], []])
        context.commit()
        self.assertEqual(calls, [0, 1])
        for history in self.history:
            self.assertEqual(len(history), 1)
            self.assertTrue(torch.equal(history[0][0], self.k[0, :, None, :]))
        self.assertEqual(context.features[0].shape, (1, 12))
        with self.assertRaises(RuntimeError):
            context.commit()

    def test_partial_failure_commits_no_layer(self):
        context = self.context()
        context.attention(0, self.q, self.k, self.v)
        context.after_layer(0, torch.zeros(2, 12))
        with self.assertRaises(RuntimeError):
            context.commit()
        self.assertEqual(self.history, [[], []])
        owner = oasis.PairedForwardOwner("cpu")
        owner.begin(context)
        owner.complete(commit=False)
        self.assertIsNone(owner.active)

    def test_unknown_fence_retains_banks_and_actual_pending_rows(self):
        def failed():
            raise RuntimeError("completion unknown")
        context = self.context()
        context.attention(0, self.q, self.k, self.v)
        owner = oasis.PairedForwardOwner("cpu", failed)
        owner.begin(context)
        with self.assertRaisesRegex(RuntimeError, "completion unknown"):
            owner.complete(commit=False)
        self.assertIs(owner.active, context)
        self.assertTrue(any(owner is self.banks[0] for owner in context.owners))
        self.assertTrue(owner.quarantined)
        self.assertEqual(self.history, [[], []])
        with self.assertRaises(RuntimeError):
            owner.begin(self.context())

    def test_layer_provider_is_consumed_in_order_without_all_layer_barrier(self):
        visits = []
        context = self.context()
        context.banks = lambda layer: (visits.append(layer), self.banks[layer])[1]
        context.attention(0, self.q, self.k, self.v)
        self.assertEqual(visits, [0])
        with self.assertRaises(RuntimeError):
            context.attention(0, self.q, self.k, self.v)
        self.assertEqual(visits, [0])
        context.attention(1, self.q, self.k, self.v)
        self.assertEqual(visits, [0, 1])

    def test_wrong_bank_is_rejected_before_publish(self):
        context = self.context(publish=lambda *a: self.fail("invalid bank published"))
        self.banks[0].valid = torch.ones(2, 5)
        with self.assertRaises(ValueError):
            context.attention(0, self.q, self.k, self.v)
        self.assertEqual(context.pending, [])


if __name__ == "__main__":
    unittest.main()
