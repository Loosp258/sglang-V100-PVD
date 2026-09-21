"""Versioned sparse PUT byte contract. A manifest is NOT a write permission.

Each delivery concatenates paired K/V groups in the declared order, without
padding or raw addresses. The receiver must separately own a lifecycle-v1
destination and wait for actual terminal completion before reading these bytes.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, fields

import torch
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVPayload, SparseKVSpec

SPARSE_DELIVERY_KEY = "pvd_sparse_delivery"
SPARSE_DELIVERY_PROTOCOL = "pvd-sparse-kv-v1"
MAX_GROUPS = 4096
MAX_TOKEN_REFERENCES = 1 << 20
_DTYPES = {"torch.float16": 2, "torch.bfloat16": 2, "torch.float32": 4}
_SPEC_FIELDS = frozenset(f.name for f in fields(SparseKVSpec))


@dataclass(frozen=True)
class SparseDeliveryManifest:
    specs: tuple[SparseKVSpec, ...]
    dtype: str
    head_dim: int

    def __post_init__(self):
        if (
            self.dtype not in _DTYPES
            or type(self.head_dim) is not int
            or not 0 < self.head_dim <= 4096
        ):
            raise ValueError("unsupported sparse dtype/head dimension")
        if not isinstance(self.specs, tuple) or not 0 < len(self.specs) <= MAX_GROUPS:
            raise ValueError("bounded nonempty immutable sparse group list required")
        if any(not isinstance(s, SparseKVSpec) for s in self.specs):
            raise ValueError("explicit SparseKVSpec groups required")
        first = self.specs[0]
        common = _SPEC_FIELDS - {"layer", "kv_head", "token_ids"}
        if any(
            any(getattr(s, f) != getattr(first, f) for f in common) for s in self.specs
        ):
            raise ValueError(
                "all sparse groups must share one request/operation/Entry/version"
            )
        if len({(s.layer, s.kv_head) for s in self.specs}) != len(self.specs):
            raise ValueError("duplicate sparse layer/KV-head group")
        if sum(len(s.token_ids) for s in self.specs) > MAX_TOKEN_REFERENCES:
            raise ValueError("sparse delivery token-reference limit exceeded")

    @property
    def nbytes(self):
        return sum(self.group_bytes(s) for s in self.specs)

    def group_bytes(self, spec):
        return 2 * len(spec.token_ids) * self.head_dim * _DTYPES[self.dtype]

    @property
    def fingerprint(self):
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self):
        specs = []
        for spec in self.specs:
            item = asdict(spec)
            item["token_ids"] = list(spec.token_ids)
            specs.append(item)
        return {
            "protocol": SPARSE_DELIVERY_PROTOCOL,
            "dtype": self.dtype,
            "head_dim": self.head_dim,
            "specs": specs,
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {
            "protocol",
            "dtype",
            "head_dim",
            "specs",
        }:
            raise ValueError("sparse manifest requires exactly its protocol fields")
        if value["protocol"] != SPARSE_DELIVERY_PROTOCOL:
            raise ValueError("unsupported sparse delivery protocol")
        groups = value["specs"]
        if not isinstance(groups, list) or not 0 < len(groups) <= MAX_GROUPS:
            raise ValueError("bounded nonempty wire group list required")
        specs, total = [], 0
        for item in groups:
            if not isinstance(item, dict) or set(item) != _SPEC_FIELDS:
                raise ValueError("sparse spec requires exactly its identity fields")
            ids = item["token_ids"]
            if not isinstance(ids, list):
                raise ValueError("wire token ids must be a list")  # noqa: TRY004
            total += len(ids)
            if total > MAX_TOKEN_REFERENCES:
                raise ValueError("sparse delivery token-reference limit exceeded")
            specs.append(SparseKVSpec(**{**item, "token_ids": tuple(ids)}))
        return cls(tuple(specs), value["dtype"], value["head_dim"])

    def payload_views(self, buffer):
        """Views only: caller must pin/fence the destination, then copy/install.

        This method establishes byte shape, never native completion or lifetime.
        It neither releases the registration nor authorizes attention reads.
        """
        if (
            not isinstance(buffer, torch.Tensor)
            or buffer.dtype != torch.uint8
            or buffer.ndim != 1
            or not buffer.is_contiguous()
            or buffer.numel() != self.nbytes
        ):
            raise ValueError("sparse payload needs exact contiguous byte extent")
        dtype = getattr(torch, self.dtype.split(".")[-1])
        result, offset = [], 0
        for spec in self.specs:
            length = self.group_bytes(spec)
            view = (
                buffer[offset : offset + length]
                .view(dtype)
                .reshape(2, len(spec.token_ids), self.head_dim)
            )
            result.append(SparseKVPayload(spec, view))
            offset += length
        return tuple(result)
