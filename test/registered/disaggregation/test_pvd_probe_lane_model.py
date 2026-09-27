"""Pure selection tests for the isolated real-CUDA handler's reply shape."""

import dataclasses
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.prediction import QueryVectors
from sglang.srt.disaggregation.pvd.probe_lane_model import (
    ProbeLaneCUDAHandler,
    _materialize_reply,
)
from sglang.srt.disaggregation.pvd.probe_lane_protocol import (
    ProbeLaneProtocolError,
    verify_reply,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED
from test_pvd_probe_lane_protocol import ticket


def _full_query(bound, layer):
    # Probe has positions 3 and 4 and four Q heads. Ticket asks only position
    # 4, heads 2 and 3; unrelated positions/heads may never reach D.
    rows = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    return QueryVectors(
        bound.target_model_id,
        "private-target-probe",
        layer,
        0,
        4,
        (3, 4),
        2,
        rows + layer * 100,
        prefix_version=bound.window.prefix.version,
        positional_encoding=ROPE_APPLIED,
        request_id=bound.window.prefix.request_id,
    )


def test_handler_reply_selects_only_ticket_positions_heads_and_layers():
    bound = ticket()
    source = tuple(_full_query(bound, layer) for layer in bound.layers)
    reply = _materialize_reply(bound, source)
    verified = verify_reply(bound, reply)
    assert len(verified) == 2
    assert tuple(verified[0].vectors.shape) == (1, 2, 4)
    torch.testing.assert_close(verified[0].vectors[0], source[0].vectors[1, 2:4])
    source[0].vectors.zero_()
    assert verified[0].vectors[0, 0, 0] == 24


def test_handler_reply_rejects_missing_position_and_foreign_layer():
    bound = ticket()
    source = tuple(_full_query(bound, layer) for layer in bound.layers)
    missing = dataclasses.replace(source[0], positions=(2, 3))
    with pytest.raises(ProbeLaneProtocolError, match="positions"):
        _materialize_reply(bound, (missing, source[1]))
    with pytest.raises(ProbeLaneProtocolError, match="layer coverage"):
        _materialize_reply(bound, (source[0], source[0]))


def test_sidecar_prefix_cache_is_bound_to_incarnation_and_reaped():
    class Probe:
        prefix_budget = object()

        def __init__(self):
            self.live = None
            self.events = []

        def register_cached_request(self, req):
            assert self.live is None
            self.live = req
            self.events.append(("register", req))

        def retire_cached_request(self, req):
            assert self.live is req
            self.live = None
            self.events.append(("retire", req))

    probe = Probe()
    draft_events = []
    provider = SimpleNamespace(
        factory=SimpleNamespace(prefix_cache_enabled=True),
        set_sidecar_cache_identity=lambda *identity: draft_events.append(
            ("bind", identity)
        ),
        retire_sidecar_cache=lambda: draft_events.append(("retire", None)),
    )
    handler = ProbeLaneCUDAHandler.__new__(ProbeLaneCUDAHandler)
    handler.pipeline = SimpleNamespace(probe=probe, provider=provider)
    handler.owner_thread = threading.get_ident()
    handler._cached_req = None
    handler._cached_incarnation = None
    handler._cached_used_at = None
    first = SimpleNamespace(prefix=SimpleNamespace(request_id="rid"), incarnation="a")
    second = SimpleNamespace(prefix=first.prefix, incarnation="b")

    handler._prepare_cache(first)
    first_req = probe.live
    handler._prepare_cache(first)
    assert probe.live is first_req
    assert len(probe.events) == 1
    assert draft_events == [("bind", ("rid", "a"))]

    handler._prepare_cache(second)
    assert probe.events[:2] == [("register", first_req), ("retire", first_req)]
    assert probe.live is not first_req
    assert handler._cached_incarnation == ("rid", "b")
    assert draft_events == [
        ("bind", ("rid", "a")),
        ("retire", None),
        ("bind", ("rid", "b")),
    ]

    handler._cached_used_at = time.monotonic() - 10
    handler.retire_idle_cache()
    assert probe.live is None
    assert handler._cached_incarnation is None
    handler.close()
    assert len(probe.events) == 4
    assert draft_events[-1] == ("retire", None)


def test_sidecar_idle_ttl_starts_after_long_handler_work(monkeypatch):
    bound = ticket()
    source = tuple(_full_query(bound, layer) for layer in bound.layers)
    handler = ProbeLaneCUDAHandler.__new__(ProbeLaneCUDAHandler)
    handler.owner_thread = threading.get_ident()
    handler.weights_sha256 = bound.weights_sha256
    handler.tokenizer_sha256 = bound.tokenizer_sha256
    handler.device = torch.device("cpu")
    handler.pipeline = SimpleNamespace(
        probe_config=SimpleNamespace(
            target_model_id=bound.target_model_id,
            layers=bound.layers,
            head_start=bound.head_start,
            head_count=bound.head_count,
        ),
        query_branch=lambda prefix: nullcontext(source),
    )
    handler._cached_incarnation = ("request", "incarnation")
    handler._cached_used_at = time.monotonic() - 10
    handler._prepare_cache = lambda window: None
    retired = []
    handler.close = lambda: retired.append(True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    before = time.monotonic()
    verify_reply(bound, handler(bound))
    assert handler._cached_used_at >= before
    handler.retire_idle_cache()
    assert not retired
