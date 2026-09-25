import dataclasses
import json
from pathlib import Path

import pytest
from sglang.srt.disaggregation.pvd.cuda_serving_limits import (
    CUDAServingLimits,
    load_cuda_serving_limits,
)


def valid_config():
    return {
        "max_sequence_tokens": 8192,
        "lead_tokens": 2,
        "attention_chunk_tokens": 128,
        "request_timeout_seconds": 30.0,
        "poll_interval_seconds": 0.01,
        "max_pending_events": 64,
        "max_pending_bytes": 1 << 20,
        "draft_transient_bytes_bound": 0,
        "probe_transient_bytes_bound": 4096,
        "target_scratch_max_reservations": 8,
        "bank_max_reservations": 4,
    }


def write_config(tmp_path, value):
    path = tmp_path / "cuda-serving.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def load(path, *, refresh_interval=16, predict_tokens=4):
    return load_cuda_serving_limits(
        path, refresh_interval=refresh_interval, predict_tokens=predict_tokens
    )


def test_loads_exact_config_as_immutable_dataclass(tmp_path):
    config = valid_config()
    limits = load(write_config(tmp_path, config))

    assert isinstance(limits, CUDAServingLimits)
    assert dataclasses.is_dataclass(limits)
    assert limits.max_sequence_tokens == 8192
    assert limits.lead_tokens == 2
    assert limits.request_timeout_seconds == 30.0
    assert limits.poll_interval_seconds == 0.01
    assert limits.draft_transient_bytes_bound == 0
    assert limits.attention_impl == "online"
    with pytest.raises(dataclasses.FrozenInstanceError):
        limits.lead_tokens = 3


def test_v100s_chunk64_experiment_only_changes_attention_tile():
    directory = Path(__file__).parent
    baseline = json.loads(
        (directory / "pvd_qwen_v100s_serving_limits.json").read_text()
    )
    alternative = json.loads(
        (directory / "pvd_qwen_v100s_serving_limits_chunk64.json").read_text()
    )
    assert baseline["attention_chunk_tokens"] == 8
    assert alternative == {**baseline, "attention_chunk_tokens": 64}
    limits = load_cuda_serving_limits(
        directory / "pvd_qwen_v100s_serving_limits_chunk64.json",
        refresh_interval=4,
        predict_tokens=2,
    )
    assert limits.attention_chunk_tokens == 64


def test_v100s_sdpa_lead3_experiment_keeps_four_token_cadence():
    directory = Path(__file__).parent
    baseline = json.loads(
        (directory / "pvd_qwen_v100s_serving_limits_sdpa.json").read_text()
    )
    alternative = json.loads(
        (directory / "pvd_qwen_v100s_serving_limits_sdpa_lead3.json").read_text()
    )
    assert alternative == {**baseline, "lead_tokens": 3}
    limits = load_cuda_serving_limits(
        directory / "pvd_qwen_v100s_serving_limits_sdpa_lead3.json",
        refresh_interval=4,
        predict_tokens=3,
    )
    assert limits.lead_tokens == 3


def test_v100s_sdpa_m8_lead6_experiment_extends_only_prefetch_window():
    directory = Path(__file__).parent
    baseline = json.loads(
        (directory / "pvd_qwen_v100s_serving_limits_sdpa.json").read_text()
    )
    alternative = json.loads(
        (directory / "pvd_qwen_v100s_serving_limits_sdpa_m8_lead6.json").read_text()
    )
    assert alternative == {**baseline, "lead_tokens": 6}
    with pytest.raises(ValueError, match="less than refresh_interval"):
        load_cuda_serving_limits(
            directory / "pvd_qwen_v100s_serving_limits_sdpa_m8_lead6.json",
            refresh_interval=4,
            predict_tokens=6,
        )
    limits = load_cuda_serving_limits(
        directory / "pvd_qwen_v100s_serving_limits_sdpa_m8_lead6.json",
        refresh_interval=8,
        predict_tokens=6,
    )
    assert limits.lead_tokens == 6


def test_bounded_sdpa_requires_opt_in_and_short_context(tmp_path):
    config = valid_config()
    config["max_sequence_tokens"] = 128
    config["attention_impl"] = "sdpa_bounded"
    assert load(write_config(tmp_path, config)).attention_impl == "sdpa_bounded"
    config["max_sequence_tokens"] = 8192
    with pytest.raises(ValueError, match="<= 256"):
        load(write_config(tmp_path, config))
    config["attention_impl"] = "unknown"
    with pytest.raises(ValueError, match="attention_impl"):
        load(write_config(tmp_path, config))


def test_v100s_sdpa_fixture_changes_only_attention_implementation():
    directory = Path(__file__).parent
    baseline = json.loads(
        (directory / "pvd_qwen_v100s_serving_limits.json").read_text()
    )
    candidate_path = directory / "pvd_qwen_v100s_serving_limits_sdpa.json"
    assert json.loads(candidate_path.read_text()) == {
        **baseline,
        "attention_impl": "sdpa_bounded",
    }
    assert (
        load_cuda_serving_limits(
            candidate_path, refresh_interval=4, predict_tokens=2
        ).attention_impl
        == "sdpa_bounded"
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda config: config.pop("lead_tokens"), "missing keys"),
        (lambda config: config.update(unexpected=1), "unknown keys"),
    ],
)
def test_rejects_missing_or_unknown_keys(tmp_path, mutate, message):
    config = valid_config()
    mutate(config)
    with pytest.raises(ValueError, match=message):
        load(write_config(tmp_path, config))


@pytest.mark.parametrize(
    "raw, message",
    [
        ("[]", "root must be an object"),
        ('{"lead_tokens": 1, "lead_tokens": 2}', "duplicate JSON key"),
        ('{"duration": NaN}', "non-finite JSON number"),
        ("{", "could not read"),
    ],
)
def test_rejects_non_object_duplicate_nonstandard_and_malformed_json(
    tmp_path, raw, message
):
    path = tmp_path / "cuda-serving.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load(path)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("max_sequence_tokens", True, "positive integer"),
        ("max_sequence_tokens", 1.0, "positive integer"),
        ("lead_tokens", 0, "positive integer"),
        ("attention_chunk_tokens", -1, "positive integer"),
        ("max_pending_events", False, "positive integer"),
        ("max_pending_bytes", 0, "positive integer"),
        ("draft_transient_bytes_bound", -1, "nonnegative integer"),
        ("probe_transient_bytes_bound", 1.5, "nonnegative integer"),
        ("target_scratch_max_reservations", True, "positive integer"),
        ("bank_max_reservations", 0, "positive integer"),
        ("request_timeout_seconds", True, "finite positive JSON number"),
        ("request_timeout_seconds", 0, "finite positive JSON number"),
        ("request_timeout_seconds", -1.0, "finite positive JSON number"),
        ("request_timeout_seconds", float("inf"), "non-finite JSON number"),
        ("poll_interval_seconds", float("nan"), "non-finite JSON number"),
        ("poll_interval_seconds", "0.1", "finite positive JSON number"),
    ],
)
def test_rejects_values_with_wrong_types_or_bounds(tmp_path, field, value, message):
    config = valid_config()
    config[field] = value
    with pytest.raises(ValueError, match=message):
        load(write_config(tmp_path, config))


@pytest.mark.parametrize(
    "config_updates, context, message",
    [
        ({"lead_tokens": 16}, {}, "less than refresh_interval"),
        ({"lead_tokens": 5}, {"predict_tokens": 4}, "must not exceed predict_tokens"),
        (
            {"max_sequence_tokens": 4},
            {"predict_tokens": 4},
            "must exceed predict_tokens",
        ),
    ],
)
def test_checks_relationships_to_pvd_runtime_settings(
    tmp_path, config_updates, context, message
):
    config = valid_config()
    config.update(config_updates)
    with pytest.raises(ValueError, match=message):
        load(write_config(tmp_path, config), **context)


@pytest.mark.parametrize(
    "context, message",
    [
        ({"refresh_interval": True}, "refresh_interval must be a positive integer"),
        ({"refresh_interval": 1}, "refresh_interval must be at least 2"),
        ({"predict_tokens": 0}, "predict_tokens must be a positive integer"),
    ],
)
def test_validates_context_values(tmp_path, context, message):
    with pytest.raises(ValueError, match=message):
        load(write_config(tmp_path, valid_config()), **context)


@pytest.mark.parametrize("path", [None, 1, object()])
def test_requires_filesystem_path(path):
    with pytest.raises(TypeError, match="filesystem path"):
        load_cuda_serving_limits(path, refresh_interval=16, predict_tokens=4)


def test_requires_existing_file_not_directory(tmp_path):
    with pytest.raises(ValueError, match="is not a file"):
        load(tmp_path)
    with pytest.raises(ValueError, match="is not a file"):
        load(tmp_path / "missing.json")


def test_accepts_pathlike_and_integer_json_seconds(tmp_path):
    config = valid_config()
    config["request_timeout_seconds"] = 30
    config["poll_interval_seconds"] = 1

    limits = load(write_config(tmp_path, config))

    assert limits.request_timeout_seconds == 30.0
    assert type(limits.request_timeout_seconds) is float
    assert limits.poll_interval_seconds == 1.0
    assert type(limits.poll_interval_seconds) is float


def test_qwen_v100s_example_matches_four_token_refresh():
    example = Path(__file__).with_name("pvd_qwen_v100s_serving_limits.json")
    limits = load_cuda_serving_limits(example, refresh_interval=4, predict_tokens=2)
    assert limits.max_sequence_tokens == 128
    assert limits.lead_tokens == 2
