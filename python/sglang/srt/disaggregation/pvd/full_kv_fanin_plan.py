"""Validate full-KV fan-in wire plans before reserving or submitting writes."""

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping

from sglang.srt.disaggregation.pvd.protocol import (
    KVEntryKey,
    KVLayoutSignature,
    ProtocolValidationError,
    RemoteRegionDescriptor,
)
from sglang.srt.disaggregation.pvd.sharding import (
    packed_fanin_transfer_slices,
    source_shard_intersections,
)

FULL_KV_FANIN_PROTOCOL = "pvd-full-kv-fanin-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def plan_fingerprint(manifest):
    return hashlib.sha256(canonical(manifest).encode()).hexdigest()


@dataclass(frozen=True)
class FanInPlan:
    key: KVEntryKey
    delivery_id: str
    destination: RemoteRegionDescriptor
    storage: KVLayoutSignature
    compute: KVLayoutSignature
    token_count: int
    writers: Mapping
    fingerprint: str


def validate_fanin_plan(value, *, max_slices):
    fields = {
        "protocol",
        "key",
        "delivery_id",
        "destination",
        "storage_layout",
        "compute_layout",
        "token_count",
        "writers",
        "plan_fingerprint",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ProtocolValidationError("complete full-KV fan-in plan required")
    if type(max_slices) is not int or max_slices <= 0:
        raise ProtocolValidationError("positive fan-in slice bound required")
    raw = copy.deepcopy(dict(value))
    fingerprint = raw.pop("plan_fingerprint")
    try:
        if raw["protocol"] != FULL_KV_FANIN_PROTOCOL or fingerprint != plan_fingerprint(
            raw
        ):
            raise ProtocolValidationError("fan-in plan fingerprint/protocol mismatch")
        key_value = raw["key"]
        if not isinstance(key_value, dict) or set(key_value) != {
            "model_instance_id",
            "req_id",
            "transfer_id",
        }:
            raise ProtocolValidationError("exact Entry identity required")
        key = KVEntryKey(**key_value)
        if any(not isinstance(v, str) or not v.strip() for v in key_value.values()):
            raise ProtocolValidationError("non-empty Entry identity required")
        storage = KVLayoutSignature.from_dict(raw["storage_layout"])
        compute = KVLayoutSignature.from_dict(raw["compute_layout"])
        destination = RemoteRegionDescriptor.from_dict(raw["destination"])
        # Reject from_dict coercions and ignored fields, including 1.0 / True.
        for name, parsed in (
            ("storage_layout", storage),
            ("compute_layout", compute),
            ("destination", destination),
        ):
            if canonical(raw[name]) != canonical(parsed.to_dict()):
                raise ProtocolValidationError("non-canonical fan-in layout/descriptor")
        delivery_id, tokens = raw["delivery_id"], raw["token_count"]
        if (
            not isinstance(delivery_id, str)
            or not delivery_id.strip()
            or type(tokens) is not int
            or tokens <= 0
        ):
            raise ProtocolValidationError("bounded delivery identity/tokens required")
        parts = source_shard_intersections(storage, compute, destination.rank)
        if (
            len(parts) * len(compute.extra["component_bytes_per_token"]) * tokens
            > max_slices
        ):
            raise ProtocolValidationError("fan-in plan exceeds slice bound")
        if (
            destination.length
            != sum(compute.extra["component_bytes_per_token"]) * tokens
        ):
            raise ProtocolValidationError("fan-in destination size mismatch")
        writers = packed_fanin_transfer_slices(
            storage, compute, compute_rank=destination.rank, token_count=tokens
        )
        expected = {
            str(rank): [asdict(p) for p in slices] for rank, slices in writers.items()
        }
        if canonical(raw["writers"]) != canonical(expected):
            raise ProtocolValidationError(
                "fan-in ranges differ from layout-derived plan"
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolValidationError(f"invalid fan-in plan: {exc}") from exc
    return FanInPlan(
        key,
        delivery_id,
        destination,
        storage,
        compute,
        tokens,
        MappingProxyType({r: tuple(p) for r, p in writers.items()}),
        fingerprint,
    )
