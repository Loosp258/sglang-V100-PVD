"""Standard-library-only validation of the explicit CPU model acceptance report."""

import argparse
import json
import math

FRAME = "PVD_CPU_SMOKE_RESULT="
SCHEMA = "pvd-rank-model-cpu-v1"
SMOKE_FLAGS = (
    "probe",
    "search",
    "sparse-decode",
    "controlled-decode",
    "batch-decode",
    "scheduled-decode",
    "real-draft-loop",
    "wire-sparse-loop",
    "rank-runtime-decode",
    "rank-runtime-loop",
)


def smoke_options(argv=None):
    if not __debug__:
        raise RuntimeError("CPU acceptance requires enabled Python assertions")
    parser = argparse.ArgumentParser(description="Strict opt-in real CPU model smoke")
    for flag in SMOKE_FLAGS:
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument("--acceptance-run-id")
    return parser.parse_args(argv)


class AcceptanceError(ValueError):
    pass


def _require(condition, field):
    if not condition:
        raise AcceptanceError("missing or invalid acceptance evidence: " + field)


def _object(value, field):
    _require(type(value) is dict, field)
    return value


def _true(data, fields):
    for field in fields:
        _require(data.get(field) is True, field)


def _positive(data, field):
    value = data.get(field)
    _require(type(value) is int and value > 0, field)


def _error(data, field):
    value = data.get(field)
    _require(
        type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 2e-5,
        field,
    )


def validate_report(report, run_id):
    _object(report, "report")
    _require(type(run_id) is str and bool(run_id), "expected run id")
    for key, expected in (
        ("acceptance_schema", SCHEMA),
        ("acceptance_run_id", run_id),
        ("status", "passed"),
        ("device", "cpu"),
        ("dtype", "float32"),
        ("backend", "TorchNativeAttnBackend"),
    ):
        _require(report.get(key) == expected, key)
    _require(report.get("gpu_rdma_latency_validated") is False, "CPU-only scope")
    _true(report, ("real_handle_success_early_release_reuse", "pool_capacity_restored"))
    for field in ("forward_count", "comparisons"):
        _positive(report, field)
    _error(report, "max_abs_error")
    delta = report.get("corrupt_mapping_min_logit_delta")
    _require(
        type(delta) in (int, float) and math.isfinite(delta) and delta > 1e-5,
        "mapping canary",
    )
    loop = _object(
        report.get("rank_runtime_loop_evidence"), "rank_runtime_loop_evidence"
    )
    _require(loop.get("status") == "passed", "loop status")
    _true(
        loop,
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
    )
    for field in (
        "production_scheduler_gpu_rdma_validated",
        "model_quality_gpu_latency_validated",
    ):
        _require(loop.get(field) is False, field)
    for field in ("attention_checks", "draft_forwards", "draft_retained_tensor_bytes"):
        _positive(loop, field)
    _error(loop, "max_attention_error")
    counts = _object(loop.get("committed_d_tokens"), "committed_d_tokens")
    expected = {"old": 9, "new": 2, "third": 0, "length-limit": 1}
    _require(
        counts == expected and all(type(v) is int for v in counts.values()),
        "independent committed counts",
    )
    sizes = loop.get("batch_sizes")
    _require(
        type(sizes) is list
        and len(sizes) == 11
        and all(type(n) is int and n in (1, 2) for n in sizes)
        and set(sizes) == {1, 2},
        "actual single/multi-request forwards",
    )
    predictions = loop.get("draft_predictions")
    _require(
        type(predictions) is list and len(predictions) == 1, "one predictive draft call"
    )
    prediction = predictions[0]
    _require(
        type(prediction) is list and len(prediction) == 2 and prediction[0] == "old",
        "draft request identity",
    )
    _require(
        type(prediction[1]) is list
        and len(prediction[1]) == 2
        and all(type(t) is int and t >= 0 for t in prediction[1]),
        "predicted ids",
    )
    _require(loop["draft_forwards"] == 2, "draft prefill/continuation")
    delivery = _object(loop.get("sparse_delivery_evidence"), "sparse_delivery_evidence")
    _require(
        type(delivery.get("http_shard_deliveries")) is int
        and delivery["http_shard_deliveries"] == 4,
        "four shard deliveries",
    )
    _positive(delivery, "selected_kv_bytes")
    _true(
        delivery,
        (
            "no_local_pack_callback",
            "acknowledged_after_all_rank_install",
            "receive_budget_restored",
        ),
    )
    _require(
        delivery.get("transport")
        == "fake in-process byte copy; real localhost HTTP control",
        "explicit payload transport",
    )
    return report


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value):
    raise AcceptanceError("nonfinite JSON constant: " + value)


def parse_report(stdout, *, returncode, run_id):
    _require(type(returncode) is int and returncode == 0, "child exit code")
    _require(type(stdout) is str, "stdout")
    frames = [
        line[len(FRAME) :] for line in stdout.splitlines() if line.startswith(FRAME)
    ]
    _require(
        len(frames) == 1 and len(frames[0]) <= 1_000_000, "one bounded result frame"
    )
    try:
        report = json.loads(
            frames[0], object_pairs_hook=_unique, parse_constant=_nonfinite
        )
    except (ValueError, RecursionError) as exc:
        raise AcceptanceError("invalid result JSON") from exc
    return validate_report(report, run_id)
