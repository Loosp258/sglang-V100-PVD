"""Derive one bounded off-owner Q ticket from D's trusted V search routes."""

from __future__ import annotations

from sglang.srt.disaggregation.pvd.probe_lane_identity import (
    ProbeLaneCheckpointIdentity,
)
from sglang.srt.disaggregation.pvd.probe_lane_protocol import (
    ProbeLaneProtocolError,
    ProbeLaneTicket,
)
from sglang.srt.disaggregation.pvd.probe_search import ProbeSearchRoute, ProbeWindow
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED, QueryHeadMapping


def issue_routed_probe_ticket(
    window: ProbeWindow,
    routes: tuple[ProbeSearchRoute, ...],
    mapping: QueryHeadMapping,
    checkpoint: ProbeLaneCheckpointIdentity,
    *,
    deadline_monotonic: float,
) -> ProbeLaneTicket:
    """Require a rectangular layer × Q-head route set, no hidden head holes."""
    if (
        not isinstance(window, ProbeWindow)
        or not isinstance(routes, tuple)
        or not routes
        or any(not isinstance(route, ProbeSearchRoute) for route in routes)
        or not isinstance(mapping, QueryHeadMapping)
        or not isinstance(checkpoint, ProbeLaneCheckpointIdentity)
    ):
        raise ProbeLaneProtocolError("explicit window, routes and checkpoint required")
    layers = tuple(sorted({route.identity.layer for route in routes}))
    heads = tuple(sorted({route.query_head for route in routes}))
    if heads != tuple(range(heads[0], heads[-1] + 1)) or len(routes) != len(
        layers
    ) * len(heads):
        raise ProbeLaneProtocolError("routed Q coverage must be rectangular")
    dimension = routes[0].scope.head_dim
    target_model_id = routes[0].identity.vector_space
    observed = set()
    for route in routes:
        identity = route.identity
        key = (identity.layer, route.query_head)
        if (
            key in observed
            or identity.entry_transfer_id != window.entry_transfer_id
            or identity.vector_space != target_model_id
            or identity.positional_encoding != ROPE_APPLIED
            or route.scope.head_dim != dimension
            or mapping.kv_head_for(route.query_head) != identity.kv_head
        ):
            raise ProbeLaneProtocolError("routed Q identity or mapping differs")
        observed.add(key)
    required_bytes = (
        len(layers) * len(window.query_positions) * len(heads) * dimension * 4
    )
    return ProbeLaneTicket.issue(
        window,
        target_model_id=target_model_id,
        weights_sha256=checkpoint.weights_sha256,
        tokenizer_sha256=checkpoint.tokenizer_sha256,
        layers=layers,
        head_start=heads[0],
        head_count=len(heads),
        head_dim=dimension,
        max_reply_bytes=required_bytes,
        deadline_monotonic=deadline_monotonic,
    )
