"""Probe CLI/reporting tests only; no fake success is hardware evidence."""

import importlib.util
import json
from pathlib import Path

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
