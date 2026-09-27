"""D derives exact sidecar Q coverage from trusted V routes."""

import dataclasses
import time

import pytest
from sglang.srt.disaggregation.pvd.probe_lane_identity import (
    ProbeLaneCheckpointIdentity,
)
from sglang.srt.disaggregation.pvd.probe_lane_protocol import ProbeLaneProtocolError
from sglang.srt.disaggregation.pvd.probe_lane_routing import (
    issue_routed_probe_ticket,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from test_pvd_probe_search import setup


def _checkpoint():
    return ProbeLaneCheckpointIdentity("a" * 64, "b" * 64, 100, 10)


def test_ticket_coverage_follows_trusted_route_and_checkpoint():
    _, _, _, window, _, _, route = setup()
    bound = issue_routed_probe_ticket(
        window,
        (route,),
        QueryHeadMapping(1, 1),
        _checkpoint(),
        deadline_monotonic=time.monotonic() + 30,
    )
    assert bound.window is window
    assert bound.target_model_id == route.identity.vector_space
    assert bound.layers == (route.identity.layer,)
    assert (bound.head_start, bound.head_count) == (0, 1)
    assert bound.max_reply_bytes == route.scope.head_dim * 4
    assert bound.weights_sha256 == "a" * 64


def test_ticket_refuses_duplicates_head_holes_and_mapping_mismatch():
    _, _, _, window, _, _, route = setup()
    deadline = time.monotonic() + 30
    with pytest.raises(ProbeLaneProtocolError, match="rectangular"):
        issue_routed_probe_ticket(
            window,
            (route, route),
            QueryHeadMapping(1, 1),
            _checkpoint(),
            deadline_monotonic=deadline,
        )
    with pytest.raises(ProbeLaneProtocolError, match="rectangular"):
        issue_routed_probe_ticket(
            window,
            (route, dataclasses.replace(route, query_head=2)),
            QueryHeadMapping(3, 1),
            _checkpoint(),
            deadline_monotonic=deadline,
        )
    with pytest.raises(ProbeLaneProtocolError, match="mapping"):
        issue_routed_probe_ticket(
            window,
            (dataclasses.replace(route, query_head=1),),
            QueryHeadMapping(2, 2),
            _checkpoint(),
            deadline_monotonic=deadline,
        )
