"""Real spawn processes with per-rank tensor ownership; CPU-only evidence."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "ranks,fault", [(2, "none"), (4, "none"), (2, "install"), (2, "exit")]
)
def test_independent_rank_banks_over_bounded_control_messages(ranks, fault):
    script = Path(__file__).with_name("run_pvd_rank_install_cpu_smoke.py")
    result = subprocess.run(
        [sys.executable, str(script), "--ranks", str(ranks), "--fault", fault],
        capture_output=True,
        text=True,
        timeout=150,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["status"] == "passed"
    assert evidence["independent_processes"] == ranks
    assert evidence["reader_drain_and_global_resume_gate"]
    assert evidence["partial_failure_blocks_resume"] == (fault != "none")
    assert evidence["live_rank_budgets_restored"] == ranks - (fault == "exit")
    assert not evidence["real_model_tp_gpu_rdma_scheduler_validated"]


@pytest.mark.parametrize(
    "fault", ["none", "install", "exit", "lost-resume", "lost-prepared"]
)
def test_owner_runtime_drives_real_rank_processes_and_failures(fault):
    script = Path(__file__).with_name("run_pvd_rank_install_cpu_smoke.py")
    result = subprocess.run(
        [sys.executable, str(script), "--runtime", "--ranks", "2", "--fault", fault],
        capture_output=True,
        text=True,
        timeout=150,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["status"] == "passed" and evidence["runtime"]
    assert evidence["independent_processes"] == 2
    assert evidence["live_rank_budgets_restored"] == 2 - (fault == "exit")
    assert evidence["all_bound_peers_notified_on_failure"] == (fault != "none")
    assert not evidence["resource_cleanup_proven_by_control"]
    assert not evidence["real_model_tp_gpu_rdma_scheduler_validated"]
