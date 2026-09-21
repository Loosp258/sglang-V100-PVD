"""Evidence validation is not model execution; the CLI runs the real model gate."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pvd_rank_model_acceptance import (
    FAULT_CHECKS,
    FRAME,
    SCHEMA,
    AcceptanceError,
    parse_report,
    smoke_options,
    validate_report,
)
from run_pvd_rank_model_acceptance import main


def evidence(run_id="test-run", fault="none"):
    loop = dict.fromkeys(
        (
            "wait_all_at_boundary",
            "reordered_results_and_membership",
            "cancelled_member_output_discarded",
            "real_batch_failure_commits_nothing",
            "reusable_batch_executor",
            "real_req_schedule_batch_result_processor",
            "rank_runtime_bound_to_model_banks",
            "real_req_retraction_and_length_limit",
            "independent_real_draft",
            "request_local_refresh_driver",
            "target_state_rng_unchanged_during_draft",
            "private_backing_storage_verified",
            "draft_pool_capacity_restored",
            "committed_boundary_fallback_did_not_call_draft",
        ),
        True,
    )
    loop.update(
        fault_evidence={
            "mode": fault,
            "cleanup_verified": True,
            **dict.fromkeys(FAULT_CHECKS[fault], True),
        },
        status="passed",
        production_scheduler_gpu_rdma_validated=False,
        model_quality_gpu_latency_validated=False,
        attention_checks=21,
        max_attention_error=3e-7,
        draft_forwards=2,
        draft_retained_tensor_bytes=39712,
        draft_predictions=[["old", [13, 13]]],
        batch_sizes=[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1],
        committed_d_tokens={"old": 9, "new": 2, "third": 0, "length-limit": 1},
        sparse_delivery_evidence={
            "http_shard_deliveries": 4,
            "selected_kv_bytes": 1600,
            "no_local_pack_callback": True,
            "acknowledged_after_all_rank_install": True,
            "receive_budget_restored": True,
            "transport": "fake in-process byte copy; real localhost HTTP control",
        },
    )
    if fault == "install":
        loop.update(
            committed_d_tokens={"old": 4, "new": 2},
            batch_sizes=[1, 1, 1, 2, 1],
            committed_boundary_fallback_did_not_call_draft=False,
        )
    return {
        "acceptance_schema": SCHEMA,
        "acceptance_run_id": run_id,
        "status": "passed",
        "device": "cpu",
        "dtype": "float32",
        "backend": "TorchNativeAttnBackend",
        "forward_count": 33,
        "comparisons": 6,
        "max_abs_error": 8e-7,
        "corrupt_mapping_min_logit_delta": 0.9,
        "real_handle_success_early_release_reuse": True,
        "pool_capacity_restored": True,
        "gpu_rdma_latency_validated": False,
        "rank_runtime_loop_evidence": loop,
    }


def frame(report):
    return "unrelated library log\n" + FRAME + json.dumps(report) + "\n"


def test_complete_report_has_one_matching_run_and_explicit_cpu_scope():
    report = evidence()
    assert parse_report(frame(report), returncode=0, run_id="test-run") == report


@pytest.mark.parametrize(
    "field",
    [
        "rank_runtime_bound_to_model_banks",
        "independent_real_draft",
        "real_req_schedule_batch_result_processor",
        "request_local_refresh_driver",
        "target_state_rng_unchanged_during_draft",
        "private_backing_storage_verified",
        "draft_pool_capacity_restored",
        "committed_boundary_fallback_did_not_call_draft",
        "real_batch_failure_commits_nothing",
        "wait_all_at_boundary",
    ],
)
@pytest.mark.parametrize("value", [None, False, 1])
def test_no_missing_fabricated_or_truthy_only_execution_evidence(field, value):
    report = evidence()
    report["rank_runtime_loop_evidence"][field] = value
    with pytest.raises(AcceptanceError, match=field):
        validate_report(report, "test-run")


@pytest.mark.parametrize(
    "failure",
    [
        "absent_loop",
        "old_run",
        "hardware_claim",
        "numeric_bool",
        "nan",
        "inf",
        "tolerance",
        "no_canary",
        "wrong_counts",
        "no_delivery",
        "uncharged_receive",
        "no_draft",
        "wrong_transport",
        "wrong_batch",
        "extra_frame",
        "no_frame",
        "exit_failure",
        "invalid_json",
        "duplicate_key",
    ],
)
def test_partial_or_invalid_reports_cannot_pass(failure):
    report = evidence()
    loop = report["rank_runtime_loop_evidence"]
    if failure == "absent_loop":
        report["rank_runtime_loop_evidence"] = None
    elif failure == "old_run":
        report["acceptance_run_id"] = "old"
    elif failure == "hardware_claim":
        report["gpu_rdma_latency_validated"] = True
    elif failure == "numeric_bool":
        loop["attention_checks"] = True
    elif failure in ("nan", "inf", "tolerance"):
        loop["max_attention_error"] = 1.0 if failure == "tolerance" else float(failure)
    elif failure == "no_canary":
        report["corrupt_mapping_min_logit_delta"] = 0
    elif failure == "wrong_counts":
        loop["committed_d_tokens"]["new"] = 9
    elif failure == "no_delivery":
        loop["sparse_delivery_evidence"] = None
    elif failure == "uncharged_receive":
        loop["sparse_delivery_evidence"]["receive_budget_restored"] = False
    elif failure == "no_draft":
        loop["draft_predictions"] = []
    elif failure == "wrong_transport":
        loop["sparse_delivery_evidence"]["transport"] = "RDMA"
    elif failure == "wrong_batch":
        loop["batch_sizes"] = [1] * 11
    raw = frame(report)
    if failure == "extra_frame":
        raw += frame(report)
    elif failure == "no_frame":
        raw = json.dumps(report)
    elif failure == "invalid_json":
        raw = FRAME + "{broken"
    elif failure == "duplicate_key":
        raw = raw.replace(
            '"status": "passed"', '"status": "failed", "status": "passed"', 1
        )
    with pytest.raises(AcceptanceError):
        parse_report(
            raw, returncode=1 if failure == "exit_failure" else 0, run_id="test-run"
        )


def test_smoke_rejects_misspelled_flags_instead_of_running_only_baseline():
    with pytest.raises(SystemExit) as exc:
        smoke_options(["--rank-runtime-looop"])
    assert exc.value.code == 2
    assert smoke_options(["--rank-runtime-loop"]).rank_runtime_loop


def test_optimized_python_cannot_report_assertion_based_checks_as_passed():
    script = Path(__file__).with_name("run_pvd_rank_model_acceptance.py")
    child = subprocess.run(
        [sys.executable, "-O", str(script)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert child.returncode == 2 and "assertions must be enabled" in child.stderr
    assert not child.stdout


def test_cli_runs_required_branch_with_this_interpreter_and_nonce(monkeypatch, capsys):
    def run(command, **kwargs):
        assert command[2] == "--rank-runtime-loop"
        assert kwargs["timeout"] == 75 and kwargs["check"] is False
        assert "PYTHONOPTIMIZE" not in kwargs["env"]
        return SimpleNamespace(
            returncode=0, stdout=frame(evidence(command[-1])), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", run)
    assert main(["--timeout-seconds", "75"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "passed" and not output["production_gpu_rdma_validated"]


@pytest.mark.parametrize("failure", ["timeout", "missing_import", "missing_evidence"])
def test_cli_failures_are_nonzero_never_skip_or_success(monkeypatch, capsys, failure):
    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(
            returncode=1 if failure == "missing_import" else 0,
            stdout="",
            stderr="import unavailable",
        )

    monkeypatch.setattr(subprocess, "run", run)
    assert main([]) == 1
    captured = capsys.readouterr()
    assert not captured.out and "acceptance failed" in captured.err


@pytest.mark.parametrize("fault", ["lost-resume", "install", "cleanup"])
def test_fault_report_requires_requested_scenario_and_cleanup(fault):
    report = evidence(fault=fault)
    assert validate_report(report, "test-run", fault=fault) is report
    with pytest.raises(AcceptanceError, match="requested fault mode"):
        validate_report(report, "test-run", fault="none")
    report["rank_runtime_loop_evidence"]["fault_evidence"]["cleanup_verified"] = False
    with pytest.raises(AcceptanceError, match="cleanup_verified"):
        validate_report(report, "test-run", fault=fault)


@pytest.mark.parametrize(
    "fault,field",
    [(fault, field) for fault, fields in FAULT_CHECKS.items() for field in fields],
)
def test_missing_fault_observation_never_counts_as_pass(fault, field):
    report = evidence(fault=fault)
    del report["rank_runtime_loop_evidence"]["fault_evidence"][field]
    with pytest.raises(AcceptanceError, match=field):
        validate_report(report, "test-run", fault=fault)


def test_install_fault_cannot_claim_unexecuted_boundary_fallback():
    report = evidence(fault="install")
    report["rank_runtime_loop_evidence"][
        "committed_boundary_fallback_did_not_call_draft"
    ] = True
    with pytest.raises(AcceptanceError, match="not executed"):
        validate_report(report, "test-run", fault="install")


def test_all_fault_cli_runs_every_case_in_fresh_child_with_unique_id(
    monkeypatch, capsys
):
    calls = []

    def run(command, **kwargs):
        fault, run_id = command[4], command[-1]
        calls.append((fault, run_id))
        return SimpleNamespace(
            returncode=0, stdout=frame(evidence(run_id, fault)), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", run)
    assert main(["--fault", "all"]) == 0
    assert [fault for fault, _ in calls] == list(FAULT_CHECKS)
    assert len({run_id for _, run_id in calls}) == 4
    assert set(json.loads(capsys.readouterr().out)["evidence_by_fault"]) == set(
        FAULT_CHECKS
    )


def test_all_fault_cli_cannot_hide_a_failed_middle_case(monkeypatch, capsys):
    calls = []

    def run(command, **kwargs):
        fault, run_id = command[4], command[-1]
        calls.append(fault)
        return SimpleNamespace(
            returncode=1 if fault == "install" else 0,
            stdout=frame(evidence(run_id, fault)),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", run)
    assert main(["--fault", "all"]) == 1
    assert calls == ["none", "lost-resume", "install"]
    assert not capsys.readouterr().out


def test_fault_flag_cannot_run_an_unrelated_smoke_branch():
    with pytest.raises(SystemExit) as exc:
        smoke_options(["--rank-runtime-fault", "install"])
    assert exc.value.code == 2
