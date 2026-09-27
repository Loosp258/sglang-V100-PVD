"""Bounded local-lane frames: JSON metadata plus raw FP32 Q rows.

The codec authenticates nothing. Only a same-UID Unix socket with verified
sidecar startup identity may use it. No serving route imports this module.
"""

from __future__ import annotations

import dataclasses
import json
import re
import struct

import torch
from sglang.srt.disaggregation.pvd.prediction import CommittedPrefix, QueryVectors
from sglang.srt.disaggregation.pvd.probe_lane_protocol import (
    MAX_LANE_REPLY_BYTES,
    ProbeLaneProtocolError,
    ProbeLaneReply,
    ProbeLaneTicket,
    verify_reply,
)
from sglang.srt.disaggregation.pvd.probe_search import ProbeWindow

MAX_TICKET_FRAME_BYTES = 256 * 1024
MAX_REPLY_FRAME_OVERHEAD = 64 * 1024
_NONCE = re.compile(r"[0-9a-f]{32}\Z")


def _fields(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ProbeLaneProtocolError(f"{name} has unexpected fields")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProbeLaneProtocolError("duplicate probe lane JSON field")
        result[key] = value
    return result


def _reject_constant(value):
    raise ProbeLaneProtocolError(f"nonfinite probe lane JSON constant {value}")


def _json_bytes(data):
    try:
        return json.dumps(
            data, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProbeLaneProtocolError("unserializable probe lane metadata") from exc


def _parse_json(frame: bytes, *, limit: int):
    if type(frame) is not bytes or not 0 < len(frame) <= limit:
        raise ProbeLaneProtocolError("probe lane frame exceeds its byte bound")
    try:
        return json.loads(
            frame.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except ProbeLaneProtocolError:
        raise
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise ProbeLaneProtocolError("malformed probe lane JSON") from exc


def encode_ticket(ticket: ProbeLaneTicket) -> bytes:
    if not isinstance(ticket, ProbeLaneTicket):
        raise ProbeLaneProtocolError("exact probe ticket required")
    window = ticket.window
    frame = _json_bytes(
        {
            "schema": 1,
            "window": {
                "incarnation": window.incarnation,
                "operation_id": window.operation_id,
                "entry_transfer_id": window.entry_transfer_id,
                "request_id": window.prefix.request_id,
                "tokens": window.prefix.tokens,
                "committed_position": window.prefix.committed_position,
                "prefix_version": window.prefix.version,
                "target_tokens": window.target_tokens,
                "query_positions": window.query_positions,
                "query_source": window.query_source,
            },
            "target_model_id": ticket.target_model_id,
            "weights_sha256": ticket.weights_sha256,
            "tokenizer_sha256": ticket.tokenizer_sha256,
            "layers": ticket.layers,
            "head_start": ticket.head_start,
            "head_count": ticket.head_count,
            "head_dim": ticket.head_dim,
            "max_reply_bytes": ticket.max_reply_bytes,
            "deadline_monotonic": ticket.deadline_monotonic,
            "prefix_digest": ticket.prefix_digest,
            "nonce": ticket.nonce,
        }
    )
    if len(frame) > MAX_TICKET_FRAME_BYTES:
        raise ProbeLaneProtocolError("probe ticket exceeds frame bound")
    return frame


def decode_ticket(
    frame: bytes,
    *,
    target_model_id: str,
    weights_sha256: str,
    tokenizer_sha256: str,
) -> ProbeLaneTicket:
    """Sidecar admits only its pinned model/tokenizer and a fresh ticket."""
    data = _parse_json(frame, limit=MAX_TICKET_FRAME_BYTES)
    _fields(
        data,
        (
            "schema",
            "window",
            "target_model_id",
            "weights_sha256",
            "tokenizer_sha256",
            "layers",
            "head_start",
            "head_count",
            "head_dim",
            "max_reply_bytes",
            "deadline_monotonic",
            "prefix_digest",
            "nonce",
        ),
        "ticket",
    )
    if data["schema"] != 1 or type(data["schema"]) is not int:
        raise ProbeLaneProtocolError("unsupported probe ticket schema")
    if (
        data["target_model_id"] != target_model_id
        or data["weights_sha256"] != weights_sha256
        or data["tokenizer_sha256"] != tokenizer_sha256
    ):
        raise ProbeLaneProtocolError("sidecar model/tokenizer identity differs")
    item = data["window"]
    _fields(
        item,
        (
            "incarnation",
            "operation_id",
            "entry_transfer_id",
            "request_id",
            "tokens",
            "committed_position",
            "prefix_version",
            "target_tokens",
            "query_positions",
            "query_source",
        ),
        "window",
    )
    if type(item["tokens"]) is not list or type(item["query_positions"]) is not list:
        raise ProbeLaneProtocolError("wire token/position lists required")
    if type(data["layers"]) is not list:
        raise ProbeLaneProtocolError("wire layer list required")
    try:
        prefix = CommittedPrefix(
            item["request_id"],
            tuple(item["tokens"]),
            item["committed_position"],
            item["prefix_version"],
        )
        window = ProbeWindow(
            item["incarnation"],
            item["operation_id"],
            item["entry_transfer_id"],
            prefix,
            item["target_tokens"],
            tuple(item["query_positions"]),
            item["query_source"],
        )
        issued = ProbeLaneTicket.issue(
            window,
            target_model_id=target_model_id,
            weights_sha256=weights_sha256,
            tokenizer_sha256=tokenizer_sha256,
            layers=tuple(data["layers"]),
            head_start=data["head_start"],
            head_count=data["head_count"],
            head_dim=data["head_dim"],
            max_reply_bytes=data["max_reply_bytes"],
            deadline_monotonic=data["deadline_monotonic"],
        )
    except ProbeLaneProtocolError:
        raise
    except (TypeError, ValueError) as exc:
        raise ProbeLaneProtocolError("invalid probe ticket contents") from exc
    if (
        type(data["nonce"]) is not str
        or _NONCE.fullmatch(data["nonce"]) is None
        or data["prefix_digest"] != issued.prefix_digest
    ):
        raise ProbeLaneProtocolError("ticket nonce or prefix digest changed")
    return dataclasses.replace(issued, nonce=data["nonce"])


def encode_reply(ticket: ProbeLaneTicket, reply: ProbeLaneReply) -> bytes:
    queries = verify_reply(ticket, reply)
    header = _json_bytes(
        {
            "schema": 1,
            "nonce": reply.nonce,
            "operation_id": reply.operation_id,
            "prefix_digest": reply.prefix_digest,
            "weights_sha256": reply.weights_sha256,
            "tokenizer_sha256": reply.tokenizer_sha256,
            "queries": [
                {
                    "layer": query.layer,
                    "vector_space": query.vector_space,
                    "version": query.version,
                    "request_id": query.request_id,
                    "prefix_version": query.prefix_version,
                    "positional_encoding": query.positional_encoding,
                    "positions": query.positions,
                    "valid_length": query.valid_length,
                    "head_start": query.head_start,
                    "head_count": query.head_count,
                }
                for query in queries
            ],
        }
    )
    if len(header) > MAX_REPLY_FRAME_OVERHEAD - 4:
        raise ProbeLaneProtocolError("probe reply header exceeds frame bound")
    frame = (
        struct.pack(">I", len(header))
        + header
        + b"".join(query.vectors.numpy().tobytes() for query in queries)
    )
    if len(frame) > ticket.max_reply_bytes + MAX_REPLY_FRAME_OVERHEAD:
        raise ProbeLaneProtocolError("probe reply exceeds frame bound")
    return frame


def decode_reply(ticket: ProbeLaneTicket, frame: bytes) -> tuple[QueryVectors, ...]:
    """D copies fully verified host Q; never publishes a partial frame."""
    limit = min(MAX_LANE_REPLY_BYTES, ticket.max_reply_bytes) + MAX_REPLY_FRAME_OVERHEAD
    if type(frame) is not bytes or not 4 < len(frame) <= limit:
        raise ProbeLaneProtocolError("probe lane frame exceeds its byte bound")
    header_length = struct.unpack(">I", frame[:4])[0]
    if not 0 < header_length <= MAX_REPLY_FRAME_OVERHEAD - 4:
        raise ProbeLaneProtocolError("probe reply header exceeds frame bound")
    header_end = 4 + header_length
    if header_end > len(frame):
        raise ProbeLaneProtocolError("truncated probe reply header")
    data = _parse_json(frame[4:header_end], limit=MAX_REPLY_FRAME_OVERHEAD - 4)
    _fields(
        data,
        (
            "schema",
            "nonce",
            "operation_id",
            "prefix_digest",
            "weights_sha256",
            "tokenizer_sha256",
            "queries",
        ),
        "reply",
    )
    if data["schema"] != 1 or type(data["schema"]) is not int:
        raise ProbeLaneProtocolError("unsupported probe reply schema")
    if type(data["queries"]) is not list or len(data["queries"]) != len(ticket.layers):
        raise ProbeLaneProtocolError("probe reply omitted or added a layer")
    row_bytes = (
        len(ticket.window.query_positions) * ticket.head_count * ticket.head_dim * 4
    )
    raw_rows = frame[header_end:]
    if len(raw_rows) != row_bytes * len(ticket.layers):
        raise ProbeLaneProtocolError("probe Q binary row length differs")
    queries = []
    for index, item in enumerate(data["queries"]):
        _fields(
            item,
            (
                "layer",
                "vector_space",
                "version",
                "request_id",
                "prefix_version",
                "positional_encoding",
                "positions",
                "valid_length",
                "head_start",
                "head_count",
            ),
            "query",
        )
        if type(item["positions"]) is not list:
            raise ProbeLaneProtocolError("wire query positions must be a list")
        try:
            chunk = raw_rows[index * row_bytes : (index + 1) * row_bytes]
            vectors = torch.frombuffer(bytearray(chunk), dtype=torch.float32)
            vectors = vectors.reshape(
                len(ticket.window.query_positions), ticket.head_count, ticket.head_dim
            )
            queries.append(
                QueryVectors(
                    item["vector_space"],
                    item["version"],
                    item["layer"],
                    item["head_start"],
                    item["head_count"],
                    tuple(item["positions"]),
                    item["valid_length"],
                    vectors,
                    prefix_version=item["prefix_version"],
                    positional_encoding=item["positional_encoding"],
                    request_id=item["request_id"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise ProbeLaneProtocolError("malformed probe Q metadata") from exc
    reply = ProbeLaneReply(
        data["nonce"],
        data["operation_id"],
        data["prefix_digest"],
        data["weights_sha256"],
        data["tokenizer_sha256"],
        tuple(queries),
    )
    return verify_reply(ticket, reply)
