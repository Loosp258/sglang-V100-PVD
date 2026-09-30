"""Learned target-Q routing must preserve request identity and absolute RoPE."""

import threading
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.pvd.joint_draft_q import (
    JointDraftQPipeline, apply_target_rope, build_joint_startup,
)
from sglang.srt.disaggregation.pvd.prediction import (
    CommittedPrefix, DraftConfig, DraftPrediction, PredictionConfigError, ProbeConfig,
)


class TestJointDraftQRouting(unittest.TestCase):
    def test_layer_head_and_future_position_mapping(self):
        prefix = CommittedPrefix("request-A", (5, 7, 9), 0, "epoch:0")
        rows = torch.arange(2 * 28 * 28 * 128).reshape(2, 28, 28, 128).float()
        provider = SimpleNamespace(predict_q=lambda p, n: (
            DraftPrediction(p.request_id, p.version, (11, 12)), rows))
        pipeline = JointDraftQPipeline(provider, SimpleNamespace(),
            DraftConfig("draft", device="cuda:0", predict_tokens=2),
            ProbeConfig("target", tuple(range(28)), head_count=28), threading.RLock())
        with self.assertRaises(PredictionConfigError):
            pipeline.run(prefix)
        pipeline._scope_active = True
        result = pipeline.run(prefix)
        self.assertEqual(len(result), 28)
        for layer, query in enumerate(result):
            self.assertEqual((query.request_id, query.prefix_version, query.positions),
                             ("request-A", "epoch:0", (3, 4)))
            self.assertEqual((query.vector_space, query.head_count, query.positional_encoding),
                             ("target", 28, "rope_applied"))
            torch.testing.assert_close(query.vectors, rows[:, layer])

    def test_absolute_rope_keeps_norm_and_uses_future_offset(self):
        query = torch.randn(28, 28, 128)
        torch.testing.assert_close(apply_target_rope(query, 0), query)
        future = apply_target_rope(query, 2155)
        torch.testing.assert_close(future.norm(dim=-1), query.norm(dim=-1))
        self.assertFalse(torch.allclose(future, query))

    def test_other_target_geometry_rejected_before_loading(self):
        runner = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(
            num_hidden_layers=61, num_attention_heads=28, num_key_value_heads=4,
            hidden_size=3584, rope_theta=1_000_000)))
        with self.assertRaises(PredictionConfigError):
            build_joint_startup(runner, checkpoint="absent", draft_model_path="absent",
                target_model_id="wrong", placement=None, execution_lock=None,
                vocabulary=None, max_prefix_tokens=2200, predict_tokens=8,
                target_scratch_budget=None, probe_transient_bytes_bound=0,
                prefix_budget=None)


if __name__ == "__main__":
    unittest.main()
