"""Bounded wire codec against malformed and cross-request frames."""

import dataclasses
import json
import struct

import pytest
import torch
from sglang.srt.disaggregation.pvd.probe_lane_protocol import ProbeLaneProtocolError
from sglang.srt.disaggregation.pvd.probe_lane_wire import (
    MAX_TICKET_FRAME_BYTES,
    decode_reply,
    decode_ticket,
    encode_reply,
    encode_ticket,
)
from test_pvd_probe_lane_protocol import reply_for, ticket


def decode_bound(frame, bound):
    return decode_ticket(
        frame,
        target_model_id=bound.target_model_id,
        weights_sha256=bound.weights_sha256,
        tokenizer_sha256=bound.tokenizer_sha256,
    )


def mutate_ticket(frame, edit):
    data = json.loads(frame)
    edit(data)
    return json.dumps(data, separators=(",", ":")).encode()


def mutate_reply(frame, edit):
    header_length = struct.unpack(">I", frame[:4])[0]
    data = json.loads(frame[4 : 4 + header_length])
    edit(data)
    header = json.dumps(data, separators=(",", ":")).encode()
    return struct.pack(">I", len(header)) + header + frame[4 + header_length :]


def test_ticket_and_reply_round_trip_owns_independent_q_rows():
    issued = ticket()
    decoded = decode_bound(encode_ticket(issued), issued)
    assert decoded == issued
    reply = reply_for(decoded)
    rows = decode_reply(decoded, encode_reply(decoded, reply))
    torch.testing.assert_close(rows[0].vectors, reply.queries[0].vectors)
    reply.queries[0].vectors.zero_()
    assert rows[0].vectors[0, 0, 0] == 1


def test_ticket_frame_bound_and_model_pin_are_enforced():
    issued = ticket()
    with pytest.raises(ProbeLaneProtocolError, match="frame.*byte bound"):
        decode_bound(b"x" * (MAX_TICKET_FRAME_BYTES + 1), issued)
    with pytest.raises(ProbeLaneProtocolError, match="model/tokenizer"):
        decode_ticket(
            encode_ticket(issued),
            target_model_id="other",
            weights_sha256=issued.weights_sha256,
            tokenizer_sha256=issued.tokenizer_sha256,
        )


def test_ticket_rejects_duplicate_json_keys_and_nonfinite_deadline():
    issued = ticket()
    frame = encode_ticket(issued)
    duplicate = frame[:-1] + b',"nonce":"duplicate"}'
    with pytest.raises(ProbeLaneProtocolError, match="duplicate"):
        decode_bound(duplicate, issued)
    data = json.loads(frame)
    data["deadline_monotonic"] = float("nan")
    nonfinite = json.dumps(data).encode()
    with pytest.raises(ProbeLaneProtocolError, match="nonfinite"):
        decode_bound(nonfinite, issued)


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda data: data.update(nonce="stale"), "nonce"),
        (lambda data: data.update(prefix_digest="0" * 64), "digest"),
        (lambda data: data["window"].update(tokens=[1, 2, 99]), "digest"),
        (lambda data: data["window"].update(query_positions=[2]), "follow"),
        (lambda data: data.update(schema=2), "schema"),
        (lambda data: data.update(extra=1), "unexpected fields"),
    ],
)
def test_ticket_rejects_stale_tampered_or_unknown_fields(edit, match):
    issued = ticket()
    with pytest.raises(ProbeLaneProtocolError, match=match):
        decode_bound(mutate_ticket(encode_ticket(issued), edit), issued)


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda data: data.update(nonce="other"), "another ticket"),
        (lambda data: data["queries"].pop(), "omitted"),
        (
            lambda data: data["queries"][0].update(positional_encoding="pre-rope"),
            "identity or coverage",
        ),
        (lambda data: data.update(extra=1), "unexpected fields"),
    ],
)
def test_reply_rejects_partial_or_foreign_frames(edit, match):
    issued = ticket()
    reply = encode_reply(issued, reply_for(issued))
    with pytest.raises(ProbeLaneProtocolError, match=match):
        decode_reply(issued, mutate_reply(reply, edit))


def test_reply_rejects_truncated_binary_q_rows():
    issued = ticket()
    with pytest.raises(ProbeLaneProtocolError, match="binary row"):
        decode_reply(issued, encode_reply(issued, reply_for(issued))[:-1])


def test_reply_rejects_truncated_header_and_oversized_header_length():
    issued = ticket()
    with pytest.raises(ProbeLaneProtocolError, match="truncated"):
        decode_reply(issued, struct.pack(">I", 12) + b"{}")
    with pytest.raises(ProbeLaneProtocolError, match="header exceeds"):
        decode_reply(issued, struct.pack(">I", 70000) + b"{}")


def test_reply_frame_bound_precedes_unpacking():
    issued = ticket()
    with pytest.raises(ProbeLaneProtocolError, match="frame.*byte bound"):
        decode_reply(issued, b"x" * 70000)


def test_reply_rejects_cross_ticket_even_with_valid_binary_rows():
    issued = ticket()
    second = ticket()
    frame = encode_reply(issued, reply_for(issued))
    with pytest.raises(ProbeLaneProtocolError, match="another ticket"):
        decode_reply(second, frame)


def test_sidecar_cannot_encode_mismatched_reply():
    issued = ticket()
    response = reply_for(issued)
    with pytest.raises(ProbeLaneProtocolError, match="another ticket"):
        encode_reply(issued, dataclasses.replace(response, operation_id="other"))
