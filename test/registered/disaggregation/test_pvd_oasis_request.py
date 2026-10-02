"""Actual commits, per-layer demand and closure without a model dependency."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

base = Path(__file__).resolve().parents[3] / "python/sglang/srt/disaggregation/pvd"
for name in ("oasis_pipeline", "oasis_request"):
    qualified = "sglang.srt.disaggregation.pvd." + name
    spec = importlib.util.spec_from_file_location(qualified, base / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
request = sys.modules["sglang.srt.disaggregation.pvd.oasis_request"]
pipeline = sys.modules["sglang.srt.disaggregation.pvd.oasis_pipeline"]


class RequestTest(unittest.TestCase):
    def make(self, *, foreign=False, fail_layer=False):
        visits, fetched, draft = [], [], []
        class Decoder:
            layers = 2
            owner = SimpleNamespace(active=None, quarantined=False)
            def step(self, actual, predicted, position, banks, *, publish):
                for layer in range(2):
                    bank = banks(layer)
                    visits.append((position, layer, bank))
                    publish(layer, (predicted, layer), bank)
                    if fail_layer:
                        raise RuntimeError("target failed")
                return "logits", (actual, position)
        def predict(current, features):
            draft.append((current, features))
            return current + 10
        def transport(query, bank):
            def fetch(ticket):
                fetched.append((ticket, query, bank))
                if foreign:
                    ticket = pipeline.LayerTicket(ticket.request_id, "stale", ticket.step, ticket.layer)
                return pipeline.LayerReply(ticket, query)
            return fetch
        owner = request.OasisRequestDecoder("r", "epoch", decoder=Decoder(),
            initial_banks=("initial0", "initial1"), predict_one=predict,
            fetch_layer=transport, current_token=7, position=8, max_steps=3)
        return owner, visits, fetched, draft

    def test_future_banks_consumed_only_by_next_step_and_actual_tokens_acknowledged(self):
        owner, visits, fetched, draft = self.make()
        try:
            self.assertEqual(owner.forward(7, 8), "logits")
            self.assertEqual(visits, [(8, 0, "initial0"), (8, 1, "initial1")])
            self.assertEqual(owner.pipeline.trace, [])
            with self.assertRaises(RuntimeError):
                owner.forward(7, 8)
            owner.actual_committed(19)
            self.assertEqual(owner.forward(19, 9), "logits")
            self.assertEqual(visits[2:], [(9, 0, (17, 0)), (9, 1, (17, 1))])
            self.assertEqual(draft, [(7, None), (19, (7, 8))])
            self.assertEqual(len(owner.pipeline.trace), 2)
            self.assertEqual(owner.current_token, 19)
        finally:
            owner.close()

    def test_stale_reply_cannot_become_next_layer_bank(self):
        owner, visits, _, _ = self.make(foreign=True)
        owner.forward(7, 8)
        owner.actual_committed(19)
        with self.assertRaisesRegex(RuntimeError, "stale or foreign"):
            owner.forward(19, 9)
        self.assertEqual(len(visits), 2)
        self.assertEqual(owner.state, "failed")
        self.assertTrue(owner.close())

    def test_failed_target_does_not_advance_actual_step(self):
        owner, _, _, _ = self.make(fail_layer=True)
        with self.assertRaisesRegex(RuntimeError, "target failed"):
            owner.forward(7, 8)
        self.assertEqual(owner.step, 0)
        with self.assertRaises(RuntimeError):
            owner.actual_committed(19)
        owner.close()

    def test_close_refuses_unknown_target_completion_and_bound_prevents_replay(self):
        owner, _, _, _ = self.make()
        owner.decoder.owner.quarantined = True
        with self.assertRaises(RuntimeError):
            owner.close()
        self.assertNotEqual(owner.state, "closed")
        owner.decoder.owner.quarantined = False
        owner.max_steps = 1
        owner.forward(7, 8)
        owner.actual_committed(19)
        with self.assertRaises(RuntimeError):
            owner.forward(19, 9)
        owner.close()


if __name__ == "__main__":
    unittest.main()
