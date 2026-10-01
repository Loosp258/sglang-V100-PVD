"""Causal scheduling and ownership checks without a CUDA dependency."""
import importlib.util
from pathlib import Path
import sys
from threading import Event, Thread
import unittest

_path = Path(__file__).resolve().parents[3] / "python/sglang/srt/disaggregation/pvd/oasis_pipeline.py"
_spec = importlib.util.spec_from_file_location("oasis_pipeline", _path)
oasis = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = oasis
_spec.loader.exec_module(oasis)


class OasisPipelineTest(unittest.TestCase):
    def test_only_consumed_layer_waits_and_close_drains(self):
        blocked, released, finished = Event(), Event(), Event()
        pipeline = oasis.LayerLookahead("request", "epoch", layers=2, workers=2)

        def delayed(ticket):
            blocked.set()
            if not released.wait(2):
                raise TimeoutError("test failed to release transfer")
            finished.set()
            return oasis.LayerReply(ticket, "slow")

        pipeline.publish(0, 1, delayed)
        self.assertTrue(blocked.wait(2))
        pipeline.publish(0, 0, lambda t: oasis.LayerReply(t, "ready"))
        self.assertEqual(pipeline.consume(0, 0), "ready")
        self.assertFalse(finished.is_set())
        released.set()
        self.assertEqual(pipeline.consume(0, 1), "slow")
        self.assertEqual(pipeline.close(), ())
        self.assertTrue(finished.is_set())
        with self.assertRaises(RuntimeError):
            pipeline.publish(1, 0, delayed)

    def test_foreign_reply_and_replay_are_rejected(self):
        with oasis.LayerLookahead("r", "generation", layers=1) as pipeline:
            foreign = oasis.LayerTicket("r", "old-generation", 0, 0)
            pipeline.publish(0, 0, lambda t: oasis.LayerReply(foreign, "bad"))
            with self.assertRaises(RuntimeError):
                pipeline.consume(0, 0)
            with self.assertRaises(RuntimeError):
                pipeline.publish(0, 0, lambda t: oasis.LayerReply(t, "replay"))
        with oasis.LayerLookahead("r", "generation", layers=1) as pipeline:
            with self.assertRaises(RuntimeError):
                pipeline.publish(1, 0, lambda t: oasis.LayerReply(t, "gap"))

    def test_close_retains_running_callback_until_terminal(self):
        started, release, drained = Event(), Event(), Event()
        pipeline = oasis.LayerLookahead("r", "generation", layers=1)
        def transfer(ticket):
            started.set()
            if not release.wait(2):
                raise TimeoutError("test transfer never released")
            drained.set()
            return oasis.LayerReply(ticket, object())
        pipeline.publish(0, 0, transfer)
        self.assertTrue(started.wait(2))
        closed = Event()
        closer = Thread(target=lambda: (pipeline.close(), closed.set()))
        closer.start()
        self.assertFalse(closed.wait(0.02))
        self.assertFalse(drained.is_set())
        release.set()
        closer.join(2)
        self.assertTrue(closed.is_set())
        self.assertTrue(drained.is_set())

    def test_resident_reuse_and_new_row_cap(self):
        self.assertEqual(oasis.select_resident([7, 8, 2, 9, 2], [2, 3, 4],
            capacity=3, max_new=1), (7, 2, 3))
        self.assertEqual(oasis.select_resident([7, 8], [2, 3, 4],
            capacity=3, max_new=0), (2, 3, 4))


if __name__ == "__main__":
    unittest.main()
