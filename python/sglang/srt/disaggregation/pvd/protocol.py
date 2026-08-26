"""Versioned control-plane protocol for PVD disaggregation.

Only descriptors and request metadata belong here. KV tensor bytes are never
serialized by this module; they travel through a transfer backend.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


PVD_PROTOCOL_VERSION = 2


class ProtocolValidationError(ValueError):
    """Raised before memory allocation or transfer for an invalid message."""


def _require_non_empty(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ProtocolValidationError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class KVEntryKey:
    """Globally unique identity of one immutable prompt-KV snapshot."""

    model_instance_id: str
    req_id: str
    transfer_id: str

    def __post_init__(self) -> None:
        _require_non_empty("model_instance_id", self.model_instance_id)
        _require_non_empty("req_id", self.req_id)
        _require_non_empty("transfer_id", self.transfer_id)

    @staticmethod
    def new(model_instance_id: str, req_id: str) -> "KVEntryKey":
        return KVEntryKey(
            model_instance_id=model_instance_id,
            req_id=req_id,
            transfer_id=uuid.uuid4().hex,
        )

    def to_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KVEntryKey":
        return cls(
            model_instance_id=str(value["model_instance_id"]),
            req_id=str(value["req_id"]),
            transfer_id=str(value["transfer_id"]),
        )


@dataclass(frozen=True)
class KVLayoutSignature:
    """Fields that must match on P, every V shard and D before RDMA starts."""

    model_id: str
    model_revision: str
    kv_dtype: str
    page_size: int
    num_layers: int
    total_kv_heads: int
    kv_heads_per_rank: int
    head_dim: int
    tp_size: int
    pp_size: int
    tensor_layout: str
    protocol_version: int = PVD_PROTOCOL_VERSION
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty("model_id", self.model_id)
        _require_non_empty("kv_dtype", self.kv_dtype)
        _require_non_empty("tensor_layout", self.tensor_layout)
        for name in (
            "page_size",
            "num_layers",
            "total_kv_heads",
            "kv_heads_per_rank",
            "head_dim",
            "tp_size",
            "pp_size",
        ):
            if getattr(self, name) <= 0:
                raise ProtocolValidationError(f"{name} must be positive")
        if self.protocol_version != PVD_PROTOCOL_VERSION:
            raise ProtocolValidationError(
                f"unsupported PVD protocol version {self.protocol_version}; "
                f"expected {PVD_PROTOCOL_VERSION}"
            )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        result = dataclasses.asdict(self)
        result["extra"] = dict(self.extra)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KVLayoutSignature":
        return cls(
            model_id=str(value["model_id"]),
            model_revision=str(value.get("model_revision", "")),
            kv_dtype=str(value["kv_dtype"]),
            page_size=int(value["page_size"]),
            num_layers=int(value["num_layers"]),
            total_kv_heads=int(value["total_kv_heads"]),
            kv_heads_per_rank=int(value["kv_heads_per_rank"]),
            head_dim=int(value["head_dim"]),
            tp_size=int(value["tp_size"]),
            pp_size=int(value["pp_size"]),
            tensor_layout=str(value["tensor_layout"]),
            protocol_version=int(value.get("protocol_version", PVD_PROTOCOL_VERSION)),
            extra=dict(value.get("extra", {})),
        )


@dataclass(frozen=True)
class KVShardManifest:
    """Expected byte layout for one rank-local KV shard."""

    rank: int
    rail: str
    expected_bytes: int
    page_count: int
    last_page_valid_tokens: int
    layer_start: int
    layer_end: int

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ProtocolValidationError("rank must be non-negative")
        _require_non_empty("rail", self.rail)
        if self.expected_bytes <= 0:
            raise ProtocolValidationError("expected_bytes must be positive")
        if self.page_count <= 0:
            raise ProtocolValidationError("page_count must be positive")
        if self.last_page_valid_tokens <= 0:
            raise ProtocolValidationError("last_page_valid_tokens must be positive")
        if self.layer_start < 0 or self.layer_end <= self.layer_start:
            raise ProtocolValidationError("invalid shard layer range")

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KVShardManifest":
        return cls(
            rank=int(value["rank"]),
            rail=str(value["rail"]),
            expected_bytes=int(value["expected_bytes"]),
            page_count=int(value["page_count"]),
            last_page_valid_tokens=int(value["last_page_valid_tokens"]),
            layer_start=int(value["layer_start"]),
            layer_end=int(value["layer_end"]),
        )


@dataclass(frozen=True)
class FirstTokenMetadata:
    """Small P-produced metadata stored atomically with the KV entry."""

    output_token_id: int
    cached_tokens: int = 0
    cached_tokens_device: int = 0
    cached_tokens_host: int = 0
    cached_tokens_storage: int = 0
    output_token_logprob: Optional[float] = None
    output_token_logprob_index: Optional[int] = None
    output_top_logprobs_values: Optional[List[float]] = None
    output_top_logprobs_indices: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FirstTokenMetadata":
        return cls(
            output_token_id=int(value["output_token_id"]),
            cached_tokens=int(value.get("cached_tokens", 0)),
            cached_tokens_device=int(value.get("cached_tokens_device", 0)),
            cached_tokens_host=int(value.get("cached_tokens_host", 0)),
            cached_tokens_storage=int(value.get("cached_tokens_storage", 0)),
            output_token_logprob=(
                float(value["output_token_logprob"])
                if value.get("output_token_logprob") is not None
                else None
            ),
            output_token_logprob_index=(
                int(value["output_token_logprob_index"])
                if value.get("output_token_logprob_index") is not None
                else None
            ),
            output_top_logprobs_values=(
                [float(item) for item in value["output_top_logprobs_values"]]
                if value.get("output_top_logprobs_values") is not None
                else None
            ),
            output_top_logprobs_indices=(
                [int(item) for item in value["output_top_logprobs_indices"]]
                if value.get("output_top_logprobs_indices") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class KVEntryManifest:
    key: KVEntryKey
    layout: KVLayoutSignature
    prompt_token_count: int
    shards: List[KVShardManifest]

    def __post_init__(self) -> None:
        if self.prompt_token_count <= 0:
            raise ProtocolValidationError("prompt_token_count must be positive")
        if len(self.shards) != self.layout.tp_size:
            raise ProtocolValidationError(
                f"expected {self.layout.tp_size} shard manifests, got {len(self.shards)}"
            )
        ranks = [shard.rank for shard in self.shards]
        if sorted(ranks) != list(range(self.layout.tp_size)):
            raise ProtocolValidationError(
                f"shard ranks must be exactly [0, {self.layout.tp_size}), got {ranks}"
            )
        expected_last = self.prompt_token_count % self.layout.page_size
        expected_last = expected_last or self.layout.page_size
        for shard in self.shards:
            if shard.last_page_valid_tokens != expected_last:
                raise ProtocolValidationError(
                    "last_page_valid_tokens does not match prompt length/page size"
                )

    def shard(self, rank: int) -> KVShardManifest:
        try:
            return next(shard for shard in self.shards if shard.rank == rank)
        except StopIteration as exc:
            raise ProtocolValidationError(f"manifest has no shard for rank {rank}") from exc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "layout": self.layout.to_dict(),
            "layout_fingerprint": self.layout.fingerprint,
            "prompt_token_count": self.prompt_token_count,
            "shards": [shard.to_dict() for shard in self.shards],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KVEntryManifest":
        manifest = cls(
            key=KVEntryKey.from_dict(value["key"]),
            layout=KVLayoutSignature.from_dict(value["layout"]),
            prompt_token_count=int(value["prompt_token_count"]),
            shards=[KVShardManifest.from_dict(item) for item in value["shards"]],
        )
        supplied = value.get("layout_fingerprint")
        if supplied is not None and supplied != manifest.layout.fingerprint:
            raise ProtocolValidationError("layout fingerprint does not match layout")
        return manifest


@dataclass(frozen=True)
class RemoteRegionDescriptor:
    """Backend-neutral descriptor for an already registered remote region."""

    endpoint: str
    region_id: str
    address: int
    length: int
    device: str
    rank: int
    rail: str
    backend_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty("endpoint", self.endpoint)
        _require_non_empty("region_id", self.region_id)
        _require_non_empty("device", self.device)
        _require_non_empty("rail", self.rail)
        if self.address < 0 or self.length <= 0 or self.rank < 0:
            raise ProtocolValidationError("invalid remote region descriptor")

    def to_dict(self) -> Dict[str, Any]:
        result = dataclasses.asdict(self)
        result["backend_metadata"] = dict(self.backend_metadata)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RemoteRegionDescriptor":
        return cls(
            endpoint=str(value["endpoint"]),
            region_id=str(value["region_id"]),
            address=int(value["address"]),
            length=int(value["length"]),
            device=str(value["device"]),
            rank=int(value["rank"]),
            rail=str(value["rail"]),
            backend_metadata=dict(value.get("backend_metadata", {})),
        )
