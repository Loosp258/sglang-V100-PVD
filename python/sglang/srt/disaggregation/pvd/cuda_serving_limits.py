"""Strict, hardware-independent configuration for opt-in CUDA PVD serving."""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

_INTEGER_FIELDS = (
    "max_sequence_tokens",
    "lead_tokens",
    "attention_chunk_tokens",
    "max_pending_events",
    "max_pending_bytes",
    "target_scratch_max_reservations",
    "bank_max_reservations",
)
_NONNEGATIVE_INTEGER_FIELDS = (
    "draft_transient_bytes_bound",
    "probe_transient_bytes_bound",
)
_DURATION_FIELDS = ("request_timeout_seconds", "poll_interval_seconds")
_FIELDS = frozenset(_INTEGER_FIELDS + _NONNEGATIVE_INTEGER_FIELDS + _DURATION_FIELDS)
_OPTIONAL_FIELDS = frozenset(("attention_impl",))


@dataclass(frozen=True)
class CUDAServingLimits:
    """Immutable validated bounds consumed by the CUDA serving assembly."""

    max_sequence_tokens: int
    lead_tokens: int
    attention_chunk_tokens: int
    request_timeout_seconds: float
    poll_interval_seconds: float
    max_pending_events: int
    max_pending_bytes: int
    draft_transient_bytes_bound: int
    probe_transient_bytes_bound: int
    target_scratch_max_reservations: int
    bank_max_reservations: int
    attention_impl: str = "online"


def _reject_constant(value):
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _positive_context_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_json_object(path):
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("CUDA serving limits must be loaded from a filesystem path")
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"CUDA serving limits path is not a file: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read CUDA serving limits JSON: {exc}") from exc
    if type(value) is not dict:
        raise ValueError("CUDA serving limits JSON root must be an object")
    return value


def load_cuda_serving_limits(path, *, refresh_interval, predict_tokens):
    """Load strict bounds and validate relationships to active PVD settings.

    ``max_sequence_tokens`` caps the full Prompt-plus-generated sequence. The
    draft runner receives the remaining prefix budget after ``predict_tokens``
    have been reserved for its continuation.
    """

    refresh_interval = _positive_context_integer(refresh_interval, "refresh_interval")
    predict_tokens = _positive_context_integer(predict_tokens, "predict_tokens")
    if refresh_interval < 2:
        raise ValueError("refresh_interval must be at least 2")

    config = _read_json_object(path)
    actual_fields = set(config)
    missing = _FIELDS - actual_fields
    extra = actual_fields - _FIELDS - _OPTIONAL_FIELDS
    if missing or extra:
        details = []
        if missing:
            details.append("missing keys: " + ", ".join(sorted(missing)))
        if extra:
            details.append("unknown keys: " + ", ".join(sorted(extra)))
        raise ValueError(
            "invalid CUDA serving limits keys (" + "; ".join(details) + ")"
        )

    values = {}
    for name in _INTEGER_FIELDS:
        value = config[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        values[name] = value

    for name in _NONNEGATIVE_INTEGER_FIELDS:
        value = config[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        values[name] = value

    for name in _DURATION_FIELDS:
        value = config[name]
        if type(value) not in (int, float):
            raise ValueError(f"{name} must be a finite positive JSON number")
        try:
            value = float(value)
        except OverflowError as exc:
            raise ValueError(f"{name} must be a finite positive JSON number") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive JSON number")
        values[name] = value

    if values["lead_tokens"] >= refresh_interval:
        raise ValueError("lead_tokens must be less than refresh_interval")
    if values["lead_tokens"] > predict_tokens:
        raise ValueError("lead_tokens must not exceed predict_tokens")
    if values["max_sequence_tokens"] <= predict_tokens:
        raise ValueError("max_sequence_tokens must exceed predict_tokens")

    attention_impl = config.get("attention_impl", "online")
    if (
        attention_impl not in ("online", "sdpa_bounded", "triton_grouped")
        or type(attention_impl) is not str
    ):
        raise ValueError(
            "attention_impl must be online, sdpa_bounded or triton_grouped"
        )
    if attention_impl == "sdpa_bounded" and values["max_sequence_tokens"] > 256:
        raise ValueError("sdpa_bounded requires max_sequence_tokens <= 256")
    if attention_impl == "triton_grouped" and values["attention_chunk_tokens"] not in (
        8,
        16,
        32,
        64,
        128,
    ):
        raise ValueError("triton_grouped requires tile size 8, 16, 32, 64 or 128")
    values["attention_impl"] = attention_impl

    return CUDAServingLimits(**values)
