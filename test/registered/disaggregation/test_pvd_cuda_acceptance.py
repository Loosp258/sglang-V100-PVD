"""Validate refusal/evidence handling, not CUDA execution."""

import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import run_pvd_cuda_acceptance as acceptance


def evidence():
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for module, names in acceptance.CASES.items():
        for name in names:
            ET.SubElement(
                suite,
                "testcase",
                classname=f"registered.disaggregation.{module}",
                name=name,
            )
    return root


def xml(root):
    return ET.tostring(root, encoding="unicode")


def test_exact_eight_cases_are_required():
    cases = acceptance.validate_junit(xml(evidence()), returncode=0)
    assert len(cases) == 8


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "unknown",
        "skip",
        "failure",
        "error",
        "empty",
        "malformed",
        "exit",
        "root",
    ],
)
def test_weak_or_failed_evidence_is_refused(mutation):
    root = evidence()
    suite = root.find("testsuite")
    if mutation == "missing":
        suite.remove(suite[0])
    elif mutation == "duplicate":
        suite.append(suite[0])
    elif mutation == "unknown":
        suite[0].set("name", "test_cpu_double")
    elif mutation in ("skip", "failure", "error"):
        ET.SubElement(suite[0], "skipped" if mutation == "skip" else mutation)
    elif mutation == "empty":
        suite.clear()
    elif mutation == "root":
        root.tag = "wrong"
    with pytest.raises(acceptance.CUDAAcceptanceError):
        acceptance.validate_junit(
            "bad XML" if mutation == "malformed" else xml(root),
            returncode=1 if mutation == "exit" else 0,
        )


def test_no_device_is_blocked_and_never_runs_pytest(monkeypatch, capsys):
    def blocked(_):
        raise acceptance.CUDAAcceptanceError("CUDA unavailable")

    def refuse(*args, **kwargs):
        raise AssertionError("must not start tests without CUDA")

    monkeypatch.setattr(acceptance, "inventory", blocked)
    monkeypatch.setattr(subprocess, "run", refuse)
    assert acceptance.main([]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked"
    assert not report["production_gpu_rdma_validated"]


@pytest.mark.parametrize("outcome", ["passed", "skipped", "timeout", "no_xml", "exit"])
def test_runner_contract_and_machine_readable_status(monkeypatch, capsys, outcome):
    monkeypatch.setattr(acceptance, "inventory", lambda _: {"test_double": True})
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k nothing")
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")

    def run(command, **kwargs):
        assert kwargs["timeout"] == 71
        assert "PYTEST_ADDOPTS" not in kwargs["env"]
        assert "PYTHONOPTIMIZE" not in kwargs["env"]
        assert kwargs["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        nodes = [part for part in command if "::" in part]
        assert len(nodes) == 8 and len(set(nodes)) == 8
        assert Path(kwargs["cwd"]).joinpath("python", "sglang").is_dir()
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 71)
        if outcome != "no_xml":
            path = next(
                part.split("=", 1)[1]
                for part in command
                if part.startswith("--junitxml=")
            )
            root = evidence()
            if outcome == "skipped":
                ET.SubElement(root.find(".//testcase"), "skipped")
            Path(path).write_text(xml(root), encoding="utf-8")
        return subprocess.CompletedProcess(
            command, 1 if outcome == "exit" else 0, "output", "errors"
        )

    monkeypatch.setattr(subprocess, "run", run)
    assert acceptance.main(["--timeout-seconds", "71"]) == (
        0 if outcome == "passed" else 1
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == ("passed" if outcome == "passed" else "failed")
    assert all(
        report[key] is False
        for key in (
            "production_gpu_rdma_validated",
            "cagra_validated",
            "model_forward_validated",
            "performance_validated",
        )
    )
