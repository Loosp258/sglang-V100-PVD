"""CPU-only qualification of diagnostic identity, spawn and retirement hooks.

These tests do not validate CUDA timings, target/EAGLE output fidelity, native
transport or a performance gain; those are strict gates of the actual replay.
"""

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import pvd_oasis_no_wait_replay as diagnostic

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


class FakeCapture:
    def __init__(self, req, events, session):
        self.req, self.events, self.session = req, events, session
        self.case = 99401
        self.closed = False
    def finish(self):
        assert self.req.req_pool_idx is None
        self.events.append("replay")
        self.closed = True


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        ARTIFACTS.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="ready-cpu-", dir=ARTIFACTS)
        self.directory = Path(self.temp.name).resolve()
        assert self.directory.is_relative_to(ARTIFACTS)
    def tearDown(self):
        self.temp.cleanup()

    def config(self):
        return dict(capture_directory=str(self.directory / "capture"), cases=[99401, 99402],
            expected_prompt_tokens=2159, output_tokens=16, workers=2,
            warmup_replays=2, measured_replays=3, diagnostic_gpu_budget_bytes=256 << 20,
            expected_prompt_ids_sha256={"99401": "1" * 64, "99402": "2" * 64})

    def test_prompt_hash_is_exact_compact_json(self):
        expected = hashlib.sha256(b"[1,20,300]").hexdigest()
        self.assertEqual(diagnostic.prompt_ids_sha256((1, 20, 300)), expected)
        self.assertNotEqual(diagnostic.prompt_ids_sha256((1, 20, 301)), expected)

    def test_config_identity_and_output_bounds(self):
        config = self.config()
        path = self.directory / "config.json"
        diagnostic.save_json_atomic(path, config)
        self.assertEqual(diagnostic.load_config(path)["workers"], 2)
        for name, changed in [("workers", 4), ("output_tokens", 15),
                              ("warmup_replays", 1), ("capture_directory", str(ROOT.parent))]:
            bad = dict(config, **{name: changed})
            diagnostic.save_json_atomic(path, bad)
            with self.assertRaises(ValueError):
                diagnostic.load_config(path)

    def test_unlisted_and_warmup_prompt_identities_are_excluded(self):
        state = diagnostic.Controller.__new__(diagnostic.Controller)
        state.config = {"expected_prompt_ids_sha256": {"99401": diagnostic.prompt_ids_sha256([1, 2]),
                                                       "99402": diagnostic.prompt_ids_sha256([3, 4])}}
        self.assertEqual(state.selected_case(SimpleNamespace(origin_input_ids=[1, 2])), 99401)
        self.assertIsNone(state.selected_case(SimpleNamespace(origin_input_ids=[99991])))

    def fixture(self):
        events, req = [], SimpleNamespace(rid="real-request", req_pool_idx=3)
        session = SimpleNamespace(_close_future=None)
        capture = FakeCapture(req, events, session)
        state = diagnostic.Controller.__new__(diagnostic.Controller)
        state.captures, state.pending, state.completed = {req.rid: capture}, [], set()
        diagnostic._QUARANTINE.append(capture)
        self.addCleanup(lambda: diagnostic._QUARANTINE.remove(capture)
                        if capture in diagnostic._QUARANTINE else None)
        return state, capture, req, events

    def scheduler_module(self, events):
        module = ModuleType("sglang.srt.disaggregation.pvd.oasis_scheduler")
        class Binding:
            deferred = False
            def forward(self, batch):
                return "forward"
            def release(self, req):
                events.append("native-release")
                return self.deferred
        module.OasisSchedulerBinding = Binding
        return module

    def test_release_queues_without_replaying_or_releasing_charge(self):
        state, capture, req, events = self.fixture()
        module = self.scheduler_module(events)
        with patch.object(diagnostic, "controller", return_value=state):
            diagnostic.patch_module(module)
            self.assertFalse(module.OasisSchedulerBinding().release(req))
        self.assertEqual(events, ["native-release"])
        self.assertEqual(state.pending, [capture])
        self.assertFalse(capture.closed)

    def test_deferred_release_never_queues_early_replay(self):
        state, capture, req, events = self.fixture()
        module = self.scheduler_module(events)
        with patch.object(diagnostic, "controller", return_value=state):
            diagnostic.patch_module(module)
            binding = module.OasisSchedulerBinding()
            binding.deferred = True
            self.assertTrue(binding.release(req))
        self.assertEqual(state.pending, [])
        self.assertFalse(capture.closed)

    def test_pending_waits_for_both_formal_free_and_session_submission(self):
        state, capture, req, events = self.fixture()
        state.pending.append(capture)
        state.finish_pending()
        req.req_pool_idx = None
        state.finish_pending()
        self.assertEqual(events, [])
        capture.session._close_future = object()
        state.finish_pending()
        self.assertEqual(events, ["replay"])
        self.assertEqual(state.pending, [])
        self.assertEqual(state.completed, {99401})

    def test_public_result_hook_finishes_only_after_entire_method(self):
        state, capture, req, events = self.fixture()
        state.pending.append(capture)
        module = ModuleType("sglang.srt.managers.scheduler_components.batch_result_processor")
        class Processor:
            def process_batch_result_decode(self, batch, result):
                events.extend(["actual-commit", "formal-free-group-end", "native-close"])
                req.req_pool_idx, capture.session._close_future = None, object()
                return "result"
        module.SchedulerBatchResultProcessor = Processor
        with patch.object(diagnostic, "controller", return_value=state):
            diagnostic.patch_module(module)
            self.assertEqual(module.SchedulerBatchResultProcessor().process_batch_result_decode(None, None), "result")
        self.assertEqual(events[-1], "replay")
        self.assertEqual(events[:3], ["actual-commit", "formal-free-group-end", "native-close"])

    def test_hook_classes_and_methods_exist_in_real_serving_sources(self):
        targets = (
            ('python/sglang/srt/managers/scheduler_components/batch_result_processor.py',
             'SchedulerBatchResultProcessor', ('process_batch_result_decode',)),
            ('python/sglang/srt/disaggregation/pvd/decode_refresh.py',
             'PVDDecodeRefresher', ('cleanup_finished',)),
            ('python/sglang/srt/disaggregation/pvd/oasis_scheduler.py',
             'OasisSchedulerBinding', ('forward', 'release')),
            ('python/sglang/srt/disaggregation/pvd/oasis_startup.py',
             'OasisResources', ('prepare',)))
        for path, name, methods in targets:
            tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
            classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
            self.assertIn(name, classes)
            actual = {node.name for node in classes[name].body if isinstance(node, ast.FunctionDef)}
            self.assertTrue(set(methods).issubset(actual), (path, name, methods))
            self.assertIn('module.' + name + '.', (ROOT / 'benchmark/pvd_oasis_no_wait_replay.py').read_text())

    def test_normal_next_iteration_cleanup_is_a_second_safe_hook(self):
        state, capture, req, events = self.fixture()
        module = ModuleType("sglang.srt.disaggregation.pvd.decode_refresh")
        class Refresher:
            def cleanup_finished(self):
                events.append("cleanup-native-close")
                state.pending.append(capture)
                req.req_pool_idx, capture.session._close_future = None, object()
        module.PVDDecodeRefresher = Refresher
        with patch.object(diagnostic, "controller", return_value=state):
            diagnostic.patch_module(module)
            module.PVDDecodeRefresher().cleanup_finished()
        self.assertEqual(events, ["cleanup-native-close", "replay"])

    def test_failure_evidence_is_atomic_and_retains_diagnostic_charge(self):
        state, capture, req, events = self.fixture()
        state.config = {"capture_directory": self.directory / "capture"}
        capture.steps, capture.replay_failures = [], []
        capture.budget = SimpleNamespace(snapshot=lambda: {"used_staging_bytes": 256 << 20})
        state.error(capture, RuntimeError("terminal proof unknown"))
        path = self.directory / "capture/99401/error.json"
        value = json.loads(path.read_text())
        self.assertEqual(value["status"], "failed")
        self.assertTrue(value["diagnostic_charge_retained"])
        self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_sitecustomize_installs_independently_in_spawned_interpreter(self):
        hook = self.directory / "hooks"
        hook.mkdir()
        (hook / "sitecustomize.py").write_bytes((ROOT / "benchmark/pvd_oasis_no_wait_sitecustomize.py").read_bytes())
        script = self.directory / "spawn_probe.py"
        script.write_text("import json, multiprocessing, os, sys\n"
            "import pvd_oasis_no_wait_replay as d\n"
            "def check():\n"
            "    assert d._INSTALLED and 'torch' not in sys.modules\n"
            "    print(json.dumps({'spawn_installed': True, 'pid': os.getpid()}), flush=True)\n"
            "if __name__ == '__main__':\n"
            "    check()\n"
            "    child=multiprocessing.get_context('spawn').Process(target=check)\n"
            "    child.start(); child.join(15)\n"
            "    assert child.exitcode == 0\n", encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(hook), str(ROOT / "benchmark")]),
                   PVD_OASIS_REPLAY_CONFIG=str(self.directory / "not-read-until-serving.json"),
                   PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", str(script)], capture_output=True,
                                text=True, env=env, timeout=25)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row["pid"] for row in rows}), 2)
        self.assertIn('"hook_installed": true', result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
