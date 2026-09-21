"""Probe CLI/reporting tests only; no fake success is hardware evidence."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def probe_module():
    path = Path(__file__).resolve().parents[3] / "scripts/pvd/check_cagra.py"
    spec = importlib.util.spec_from_file_location("pvd_cagra_probe_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_defaults_collect_only_and_keep_v100s_smoke_target(probe_module):
    args = probe_module.parser().parse_args([])
    probe_module.validate(args)
    assert args.expected_gpu == "V100S"
    assert args.metric == "inner_product"
    assert args.mode == "inventory"


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--rows", "999999"),
        ("--rows", "100"),
        ("--queries", "0"),
        ("--dim", "4096"),
        ("--k", "100"),
        ("--device", "-1"),
        ("--expected-gpu", " "),
        ("--min-recall", "0"),
        ("--min-recall", "nan"),
    ],
)
def test_invalid_settings_fail_before_gpu_work(
    probe_module, monkeypatch, capsys, flag, value
):
    def unexpected():
        pytest.fail("invalid arguments must not reach inventory")

    monkeypatch.setattr(probe_module, "inventory", unexpected)
    assert probe_module.main([flag, value]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert result["stage"] == "validate_args"


def test_missing_dependency_is_failure_not_skip_or_fallback(
    probe_module, monkeypatch, capsys
):
    monkeypatch.setattr(probe_module, "inventory", lambda: {"test_only": True})

    def missing(args, report):
        report["stage"] = "import_cupy"
        raise ModuleNotFoundError("No module named 'cupy'")

    monkeypatch.setattr(probe_module, "probe", missing)
    assert probe_module.main(["--mode", "smoke"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["stage"] == "import_cupy"
    assert report["error"]["type"] == "ModuleNotFoundError"
    assert report["config"]["metric"] == "inner_product"
    assert report["cagra_test"] == "failed"


@pytest.mark.parametrize("argv", [[], ["--mode", "inventory"]])
def test_inventory_needs_no_gpu_or_cupy(probe_module, monkeypatch, capsys, argv):
    environment = {"packages": {}, "nvidia_smi": {"error": "not installed"}}
    monkeypatch.setattr(probe_module, "inventory", lambda: environment)

    def forbidden(*args):
        pytest.fail("inventory must not import or execute the GPU probe")

    monkeypatch.setattr(probe_module, "probe", forbidden)
    assert probe_module.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "pvd_cagra_probe_v2"
    assert report["status"] == "collected"
    assert report["cagra_test"] == "not_run"
    assert report["environment"] == environment
    assert "build_ms" not in report


def test_inventory_reports_absent_nvidia_smi(probe_module, monkeypatch):
    monkeypatch.setattr(probe_module.importlib.metadata, "distributions", lambda: [])

    def absent(*args, **kwargs):
        raise FileNotFoundError("no NVIDIA utility")

    monkeypatch.setattr(probe_module.subprocess, "run", absent)
    result = probe_module.inventory()
    assert result["packages"] == {}
    assert result["nvidia_smi"] == {"error": "no NVIDIA utility"}


def test_explicit_smoke_runs_probe_and_reports_success(
    probe_module, monkeypatch, capsys
):
    # Only checks CLI dispatch/reporting, NOT an actual CAGRA test.
    monkeypatch.setattr(probe_module, "inventory", lambda: {"test_only": True})
    calls = []

    def stub(args, report):
        calls.append(args.mode)
        report["stage"] = "complete"

    monkeypatch.setattr(probe_module, "probe", stub)
    assert probe_module.main(["--mode", "smoke"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert calls == ["smoke"]
    assert report["status"] == report["cagra_test"] == "passed"


@pytest.mark.parametrize("metric", ["inner_product", "sqeuclidean"])
def test_real_metric_oracle_checks_selected_scores_and_exact_recall(
    probe_module, metric
):
    data = np.array([[1, 0], [0, 2], [3, 0], [-1, 0]], dtype=np.float32)
    queries = np.array([[2, 0]], dtype=np.float32)
    ids = np.array([[2, 0]], dtype=np.uint32)
    scores = np.array(
        [[6, 2]] if metric == "inner_product" else [[1, 1]], dtype=np.float32
    )
    evidence = probe_module.validate_search_results(
        data,
        queries,
        ids,
        scores,
        metric=metric,
        k=2,
        min_recall=1,
    )
    assert evidence["synthetic_recall_at_k"] == 1
    assert evidence["score_max_abs_error"] == 0


@pytest.mark.parametrize(
    "mode",
    ["sign", "wrong_metric", "wrong_id_score", "nan", "duplicate", "range", "recall"],
)
def test_correct_ids_alone_are_not_acceptance(probe_module, mode):
    data = np.array([[1, 0], [0, 2], [3, 0], [-1, 0]], dtype=np.float32)
    queries = np.array([[2, 0]], dtype=np.float32)
    ids = np.array([[2, 0]], dtype=np.int64)
    scores = np.array([[6, 2]], dtype=np.float32)
    if mode == "sign":
        scores *= -1
    elif mode == "wrong_metric":
        scores[:] = 1  # L2 values are wrong for the requested inner product
    elif mode == "wrong_id_score":
        scores[:] = scores[:, ::-1]
    elif mode == "nan":
        scores[0, 0] = np.nan
    elif mode == "duplicate":
        ids[:] = 2
    elif mode == "range":
        ids[0, 0] = 4
    else:
        ids[:] = [1, 3]
        scores[:] = [0, -2]  # numerically correct scores but wrong neighbors
    with pytest.raises(RuntimeError):
        probe_module.validate_search_results(
            data,
            queries,
            ids,
            scores,
            metric="inner_product",
            k=2,
            min_recall=1,
        )


def test_tied_exact_neighbors_are_not_false_recall_failures(probe_module):
    data = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32)
    queries = np.array([[2, 0]], dtype=np.float32)
    evidence = probe_module.validate_search_results(
        data,
        queries,
        np.array([[2, 1]]),
        np.array([[2.0, 2.0]]),
        metric="inner_product",
        k=2,
        min_recall=1,
    )
    assert evidence["synthetic_recall_at_k"] == 1
