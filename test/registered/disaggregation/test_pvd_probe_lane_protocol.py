"""Off-owner Q identity/byte admission, without claiming a sidecar exists."""

import dataclasses
import time

import pytest
import torch
from sglang.srt.disaggregation.pvd.prediction import CommittedPrefix, QueryVectors
from sglang.srt.disaggregation.pvd.probe_lane_protocol import (
    ProbeLaneProtocolError,
    ProbeLaneReply,
    ProbeLaneTicket,
    verify_reply,
)
from sglang.srt.disaggregation.pvd.probe_search import ProbeWindow
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED


def ticket(*, max_reply_bytes=64, query_source="predicted"):
    prefix = CommittedPrefix("request", (1, 2, 3), 0, "prefix-v1")
    positions = (4,) if query_source == "predicted" else (2,)
    window = ProbeWindow(
        "incarnation", "operation", "entry", prefix, 4, positions, query_source
    )
    return ProbeLaneTicket.issue(
        window,
        target_model_id="target-checkpoint",
        weights_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        layers=(0, 1),
        head_start=2,
        head_count=2,
        head_dim=4,
        max_reply_bytes=max_reply_bytes,
        deadline_monotonic=time.monotonic() + 30,
    )


def reply_for(bound):
    queries = tuple(
        QueryVectors(
            "target-checkpoint",
            f"{bound.window.prefix.version}:probe:{bound.nonce}",
            layer,
            2,
            2,
            bound.window.query_positions,
            1,
            torch.full((1, 2, 4), float(layer + 1)),
            prefix_version=bound.window.prefix.version,
            positional_encoding=ROPE_APPLIED,
            request_id=bound.window.prefix.request_id,
        )
        for layer in bound.layers
    )
    return ProbeLaneReply(
        bound.nonce,
        bound.window.operation_id,
        bound.prefix_digest,
        bound.weights_sha256,
        bound.tokenizer_sha256,
        queries,
    )


def test_reply_accepts_exact_post_rope_target_q_and_owns_copy():
    bound = ticket()
    reply = reply_for(bound)
    owned = verify_reply(bound, reply)
    assert tuple(q.layer for q in owned) == (0, 1)
    reply.queries[0].vectors.zero_()
    assert owned[0].vectors[0, 0, 0] == 1
    assert owned[0].vectors.data_ptr() != reply.queries[0].vectors.data_ptr()


@pytest.mark.parametrize(
    "field,value",
    [
        ("nonce", "old"),
        ("operation_id", "other"),
        ("prefix_digest", "0" * 64),
        ("weights_sha256", "c" * 64),
        ("tokenizer_sha256", "d" * 64),
    ],
)
def test_reply_rejects_foreign_ticket_or_model(field, value):
    bound = ticket()
    with pytest.raises(ProbeLaneProtocolError, match="another ticket"):
        verify_reply(bound, dataclasses.replace(reply_for(bound), **{field: value}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("vector_space", "draft-model"),
        ("request_id", "other-request"),
        ("prefix_version", "old-prefix"),
        ("positional_encoding", "pre-rope"),
        ("positions", (5,)),
        ("valid_length", 0),
        ("head_start", 1),
        ("head_count", 1),
        ("version", "unbound-query"),
    ],
)
def test_reply_rejects_wrong_q_identity_or_coverage(field, value):
    bound = ticket()
    reply = reply_for(bound)
    if field == "valid_length":
        # QueryVectors itself refuses zero; a partial but legal response is
        # represented by an extra position with only the first marked valid.
        value = 1
        replacement = dataclasses.replace(
            reply.queries[0], positions=(4, 5), valid_length=value
        )
    else:
        if field == "head_count":
            value = 1
        replacement = dataclasses.replace(reply.queries[0], **{field: value})
    with pytest.raises(ProbeLaneProtocolError, match="identity or coverage"):
        verify_reply(
            bound, dataclasses.replace(reply, queries=(replacement, *reply.queries[1:]))
        )


def test_reply_rejects_missing_layer_nonfinite_or_wrong_device_dtype():
    bound = ticket()
    reply = reply_for(bound)
    with pytest.raises(ProbeLaneProtocolError, match="omitted"):
        verify_reply(bound, dataclasses.replace(reply, queries=reply.queries[:1]))
    for vectors in (
        torch.full((1, 2, 4), float("nan")),
        torch.ones((1, 2, 4), dtype=torch.float16),
        torch.ones((2, 2, 4)),
    ):
        bad = dataclasses.replace(reply.queries[0], vectors=vectors)
        with pytest.raises(ProbeLaneProtocolError, match="malformed"):
            verify_reply(
                bound, dataclasses.replace(reply, queries=(bad, *reply.queries[1:]))
            )


def test_reply_deadline_and_ticket_budget_fail_closed():
    bound = ticket()
    with pytest.raises(ProbeLaneProtocolError, match="deadline"):
        verify_reply(bound, reply_for(bound), now=bound.deadline_monotonic)
    with pytest.raises(ProbeLaneProtocolError, match="byte bound"):
        ticket(max_reply_bytes=63)


def test_ticket_rejects_wrong_query_source_position_and_unpinned_model():
    committed = ticket(query_source="committed")
    assert verify_reply(committed, reply_for(committed))
    wrong = dataclasses.replace(
        committed.window, query_source="predicted", query_positions=(2,)
    )
    with pytest.raises(ProbeLaneProtocolError, match="follow the prefix"):
        ProbeLaneTicket.issue(
            wrong,
            target_model_id="target-checkpoint",
            weights_sha256="a" * 64,
            tokenizer_sha256="b" * 64,
            layers=(0, 1),
            head_start=2,
            head_count=2,
            head_dim=4,
            max_reply_bytes=64,
            deadline_monotonic=time.monotonic() + 30,
        )
    with pytest.raises(ProbeLaneProtocolError, match="digests"):
        ProbeLaneTicket.issue(
            committed.window,
            target_model_id="target-checkpoint",
            weights_sha256="unverified",
            tokenizer_sha256="b" * 64,
            layers=(0, 1),
            head_start=2,
            head_count=2,
            head_dim=4,
            max_reply_bytes=64,
            deadline_monotonic=time.monotonic() + 30,
        )
