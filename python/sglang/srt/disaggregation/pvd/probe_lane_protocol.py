"""Bounded identity contract for a future isolated D-side prediction process.

This is not an IPC transport or serving integration. The future transport
must authenticate its peer/model at startup before accepting these messages.
"""

from __future__ import annotations

import hashlib
import math
import re
import secrets
import struct
import time
from dataclasses import dataclass

import torch
from sglang.srt.disaggregation.pvd.prediction import QueryVectors
from sglang.srt.disaggregation.pvd.probe_search import ProbeWindow
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ProbeLaneProtocolError(ValueError):
    """A stale, oversized or misidentified prediction-lane message."""


def _digest_tokens(tokens: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        if type(token) is not int or not 0 <= token <= 0xFFFFFFFF:
            raise ProbeLaneProtocolError("prefix token is outside uint32 range")
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


@dataclass(frozen=True)
class ProbeLaneTicket:
    """One immutable capture request; never contains a live Req or GPU address."""

    window: ProbeWindow
    target_model_id: str
    weights_sha256: str
    tokenizer_sha256: str
    layers: tuple[int, ...]
    head_start: int
    head_count: int
    head_dim: int
    max_reply_bytes: int
    deadline_monotonic: float
    prefix_digest: str
    nonce: str

    @classmethod
    def issue(
        cls,
        window: ProbeWindow,
        *,
        target_model_id: str,
        weights_sha256: str,
        tokenizer_sha256: str,
        layers: tuple[int, ...],
        head_start: int,
        head_count: int,
        head_dim: int,
        max_reply_bytes: int,
        deadline_monotonic: float,
    ) -> ProbeLaneTicket:
        if not isinstance(window, ProbeWindow) or not window.prefix.tokens:
            raise ProbeLaneProtocolError("nonempty immutable ProbeWindow required")
        if (
            not window.incarnation
            or not window.operation_id
            or not window.entry_transfer_id
            or window.query_source not in ("predicted", "committed")
            or not window.query_positions
            or len(window.query_positions) > 64
            or tuple(sorted(set(window.query_positions))) != window.query_positions
        ):
            raise ProbeLaneProtocolError("invalid window identity or positions")
        if not isinstance(target_model_id, str) or not target_model_id.strip():
            raise ProbeLaneProtocolError("explicit target model identity required")
        if any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in (weights_sha256, tokenizer_sha256)
        ):
            raise ProbeLaneProtocolError("exact weight/tokenizer digests required")
        if (
            not isinstance(layers, tuple)
            or not layers
            or any(type(layer) is not int or layer < 0 for layer in layers)
            or tuple(sorted(set(layers))) != layers
            or type(head_start) is not int
            or head_start < 0
            or any(
                type(value) is not int or value <= 0
                for value in (head_count, head_dim, max_reply_bytes)
            )
        ):
            raise ProbeLaneProtocolError("invalid layer/head/reply bounds")
        required_bytes = (
            len(layers) * len(window.query_positions) * head_count * head_dim * 4
        )
        if required_bytes > max_reply_bytes:
            raise ProbeLaneProtocolError("Q reply exceeds admitted byte bound")
        if (
            isinstance(deadline_monotonic, bool)
            or not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(deadline_monotonic)
            or deadline_monotonic <= time.monotonic()
        ):
            raise ProbeLaneProtocolError("future finite deadline required")
        if window.query_source == "committed" and any(
            position >= len(window.prefix.tokens) for position in window.query_positions
        ):
            raise ProbeLaneProtocolError("committed Q must be inside the prefix")
        if window.query_source == "predicted" and any(
            position < len(window.prefix.tokens) for position in window.query_positions
        ):
            raise ProbeLaneProtocolError("predicted Q must follow the prefix")
        return cls(
            window,
            target_model_id,
            weights_sha256,
            tokenizer_sha256,
            layers,
            head_start,
            head_count,
            head_dim,
            max_reply_bytes,
            float(deadline_monotonic),
            _digest_tokens(window.prefix.tokens),
            secrets.token_hex(16),
        )


@dataclass(frozen=True)
class ProbeLaneReply:
    """CPU Q rows; verification copies them before publication to search."""

    nonce: str
    operation_id: str
    prefix_digest: str
    weights_sha256: str
    tokenizer_sha256: str
    queries: tuple[QueryVectors, ...]


def verify_reply(
    ticket: ProbeLaneTicket, reply: ProbeLaneReply, *, now: float | None = None
) -> tuple[QueryVectors, ...]:
    """Reject stale/foreign/partial Q before any V search can observe it."""
    if not isinstance(ticket, ProbeLaneTicket) or not isinstance(reply, ProbeLaneReply):
        raise ProbeLaneProtocolError("exact ticket and reply types required")
    observed = time.monotonic() if now is None else now
    if (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or not math.isfinite(observed)
        or observed >= ticket.deadline_monotonic
    ):
        raise ProbeLaneProtocolError("probe reply missed its deadline")
    if (
        reply.nonce != ticket.nonce
        or reply.operation_id != ticket.window.operation_id
        or reply.prefix_digest != ticket.prefix_digest
        or reply.weights_sha256 != ticket.weights_sha256
        or reply.tokenizer_sha256 != ticket.tokenizer_sha256
    ):
        raise ProbeLaneProtocolError("probe reply belongs to another ticket/model")
    if not isinstance(reply.queries, tuple) or len(reply.queries) != len(ticket.layers):
        raise ProbeLaneProtocolError("probe reply omitted or added a layer")
    prefix = ticket.window.prefix
    positions = ticket.window.query_positions
    owned = []
    for layer, query in zip(ticket.layers, reply.queries, strict=True):
        if (
            not isinstance(query, QueryVectors)
            or query.layer != layer
            or query.vector_space != ticket.target_model_id
            or query.version != f"{prefix.version}:probe:{ticket.nonce}"
            or query.request_id != prefix.request_id
            or query.prefix_version != prefix.version
            or query.positional_encoding != ROPE_APPLIED
            or query.positions != positions
            or query.valid_length != len(positions)
            or query.head_start != ticket.head_start
            or query.head_count != ticket.head_count
        ):
            raise ProbeLaneProtocolError("probe Q identity or coverage mismatch")
        vectors = query.vectors
        if (
            not isinstance(vectors, torch.Tensor)
            or vectors.device.type != "cpu"
            or vectors.dtype != torch.float32
            or tuple(vectors.shape)
            != (len(positions), ticket.head_count, ticket.head_dim)
            or not torch.isfinite(vectors).all()
        ):
            raise ProbeLaneProtocolError("probe Q rows are malformed or nonfinite")
        owned.append(
            QueryVectors(
                query.vector_space,
                query.version,
                query.layer,
                query.head_start,
                query.head_count,
                query.positions,
                query.valid_length,
                vectors.contiguous().clone(),
                prefix_version=query.prefix_version,
                positional_encoding=query.positional_encoding,
                request_id=query.request_id,
            )
        )
    if sum(query.vectors.numel() * 4 for query in owned) > ticket.max_reply_bytes:
        raise ProbeLaneProtocolError("probe reply exceeds admitted byte bound")
    return tuple(owned)
