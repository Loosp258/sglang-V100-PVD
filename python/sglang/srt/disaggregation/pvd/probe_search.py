"""Standalone CPU probe -> single-shard retrieval, NOT a serving integration.

One owner thread/event loop owns each session. Prediction runs synchronously
on that thread; only HTTP yields. A selected logical result is NOT KV_READY,
an RDMA grant, an installation, or permission to advance a PrefetchClock.
No live request, writable committed KV or sampler is passed to providers.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace

import torch
from sglang.srt.disaggregation.pvd.prediction import (
    CommittedPrefix,
    PredictionConfigError,
    PredictionPipeline,
)
from sglang.srt.disaggregation.pvd.prompt_index import SearchRequestIdentity
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED, QueryHeadMapping
from sglang.srt.disaggregation.pvd.search_client import (
    PVDShardSearchClient,
    SearchScope,
    ShardSearchResult,
)


class StaleProbeSearch(ValueError):
    """The operation no longer belongs to a live request/window."""


def _text(name, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _count(name, value):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class ProbeWindow:
    """All identities are separate; positions are absolute sequence positions.

    target_tokens counts committed D tokens, excluding P's first token. It is
    not a sequence position. query_positions explicitly selects predicted Q
    positions; this module invents no last-token or cross-position policy.
    """

    incarnation: str
    operation_id: str
    entry_transfer_id: str
    prefix: CommittedPrefix
    target_tokens: int
    query_positions: tuple[int, ...]


@dataclass(frozen=True)
class ProbeSearchRoute:
    """One global Q head -> one global KV head, no head/layer merging."""

    query_head: int
    identity: SearchRequestIdentity
    scope: SearchScope
    top_k: int

    def __post_init__(self):
        _count("query_head", self.query_head)
        if not isinstance(self.identity, SearchRequestIdentity) or not isinstance(
            self.scope, SearchScope
        ):
            raise TypeError("route requires explicit identity and trusted scope")
        if type(self.top_k) is not int or not 1 <= self.top_k <= min(
            512, self.scope.prompt_tokens
        ):
            raise ValueError("route top_k exceeds the protocol or prompt limit")


@dataclass(frozen=True)
class PreparedProbeQuery:
    route: ProbeSearchRoute
    query_version: str
    rows: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class PreparedProbeSearch:
    window: ProbeWindow
    queries: tuple[PreparedProbeQuery, ...]


@dataclass(frozen=True)
class ProbeSelection:
    window: ProbeWindow
    queries: tuple[PreparedProbeQuery, ...]
    selections: tuple[ShardSearchResult, ...]


class ProbeSearchSession:
    """One request incarnation; explicit invalidation, no hidden retries.

    Normal committed progress toward the boundary does not invalidate the
    original prediction snapshot (approximation is allowed). Call invalidate
    for prefix replacement and replace_entry for Entry changes. A retraction
    that decreases the committed count requires a new session incarnation.
    Calls must be serialized on the owner thread; not a thread-safe registry.
    """

    def __init__(self, request_id: str, entry_transfer_id: str):
        _text("request_id", request_id)
        _text("entry_transfer_id", entry_transfer_id)
        self.request_id = request_id
        self.entry_transfer_id = entry_transfer_id
        self.incarnation = uuid.uuid4().hex
        self._closed = False
        self._observed = 0
        self._pending = None
        self._prepared = None
        self._ready = None
        self._searching = None

    def _match(self, window):
        if self._closed or self._pending is not window:
            raise StaleProbeSearch("closed, stale or foreign probe window")

    def begin(
        self,
        prefix: CommittedPrefix,
        *,
        target_tokens: int,
        query_positions: tuple[int, ...],
    ) -> ProbeWindow:
        if self._closed:
            raise StaleProbeSearch("request is closed")
        if self._pending is not None or self._searching is not None:
            raise ValueError("one outstanding probe/search window per request")
        if (
            not isinstance(prefix, CommittedPrefix)
            or prefix.request_id != self.request_id
        ):
            raise ValueError("prefix belongs to another request")
        _count("target_tokens", target_tokens)
        if prefix.committed_position < self._observed:
            raise StaleProbeSearch("committed token count regressed")
        if target_tokens <= prefix.committed_position:
            raise ValueError(
                "a periodic probe needs a future target boundary; bootstrap is separate"
            )
        if (
            not isinstance(query_positions, tuple)
            or not 1 <= len(query_positions) <= 64
            or any(
                type(p) is not int or p < len(prefix.tokens) for p in query_positions
            )
            or tuple(sorted(set(query_positions))) != query_positions
        ):
            raise ValueError(
                "query positions must be 1..64 distinct ascending predicted positions"
            )
        self._observed = prefix.committed_position
        window = ProbeWindow(
            self.incarnation,
            uuid.uuid4().hex,
            self.entry_transfer_id,
            prefix,
            target_tokens,
            query_positions,
        )
        self._pending = window
        return window

    def observe(self, committed_tokens: int):
        """Notification only: never advance real Decode or a refresh clock."""
        if self._closed:
            raise StaleProbeSearch("request is closed")
        _count("committed_tokens", committed_tokens)
        if committed_tokens < self._observed:
            raise ValueError(
                "committed token count regressed; explicitly invalidate on retraction"
            )
        self._observed = committed_tokens
        if self._pending is not None and committed_tokens > self._pending.target_tokens:
            self.invalidate()
            raise StaleProbeSearch("probe window expired past its target boundary")

    def invalidate(self):
        """Discard local results; never assume native/GPU work was cancelled."""
        self._pending = self._prepared = self._ready = None

    def replace_entry(self, entry_transfer_id: str):
        if self._closed:
            raise StaleProbeSearch("request is closed")
        _text("entry_transfer_id", entry_transfer_id)
        self.invalidate()
        self.entry_transfer_id = entry_transfer_id

    def close(self):
        self.invalidate()
        self._closed = True

    def prepare(
        self,
        window: ProbeWindow,
        pipeline: PredictionPipeline,
        *,
        routes: tuple[ProbeSearchRoute, ...],
        head_mapping: QueryHeadMapping,
    ) -> PreparedProbeSearch:
        """Validate and snapshot host queries, releasing branch scratch first.

        CPU foundation only. Concrete adapters must reserve temporary budgets
        in their branch scopes. No claim is made about CUDA RNG, GPU workspace,
        model fidelity or isolation of adapters that retain live external state.
        """
        self._match(window)
        if self._prepared is not None:
            raise ValueError("window was already prepared")
        try:
            if not isinstance(pipeline, PredictionPipeline) or not isinstance(
                head_mapping, QueryHeadMapping
            ):
                raise TypeError("explicit pipeline and query-head mapping required")
            if torch.device(pipeline.draft_config.device).type != "cpu":
                raise PredictionConfigError(
                    "this standalone bridge only validates CPU execution"
                )
            if not isinstance(routes, tuple) or not 1 <= len(routes) <= 64:
                raise ValueError("provide 1..64 explicit single-shard routes")
            keys = set()
            for route in routes:
                if not isinstance(route, ProbeSearchRoute):
                    raise TypeError("explicit ProbeSearchRoute required")
                identity = route.identity
                if identity.entry_transfer_id != window.entry_transfer_id:
                    raise ValueError("route points to another Entry")
                if identity.vector_space != pipeline.probe_config.target_model_id:
                    raise ValueError("route vector space differs from target model")
                if identity.positional_encoding != ROPE_APPLIED:
                    raise ValueError("the predictive contract requires post-RoPE Q")
                if head_mapping.kv_head_for(route.query_head) != identity.kv_head:
                    raise ValueError("query-head to KV-head mapping mismatch")
                key = (identity.layer, route.query_head)
                if key in keys:
                    raise ValueError("duplicate layer/query-head route")
                keys.add(key)
            prepared = []
            with pipeline.query_branch(window.prefix) as queries:
                by_layer = {q.layer: q for q in queries}
                for route in routes:
                    query = by_layer.get(route.identity.layer)
                    if query is None:
                        raise ValueError("probe omitted a routed layer")
                    if query.prefix_version != window.prefix.version:
                        raise ValueError("probe query lacks matching prefix identity")
                    if query.request_id != window.prefix.request_id:
                        raise ValueError("probe query lacks matching request identity")
                    if query.positional_encoding != route.identity.positional_encoding:
                        raise ValueError("probe positional encoding mismatch")
                    tensor = query.vectors
                    if (
                        not isinstance(tensor, torch.Tensor)
                        or tensor.device.type != "cpu"
                        or not tensor.is_floating_point()
                        or tuple(tensor.shape)
                        != (
                            len(query.positions),
                            query.head_count,
                            route.scope.head_dim,
                        )
                    ):
                        raise ValueError(
                            "probe requires CPU float Q [positions, query_heads, head_dim]"
                        )
                    local_head = route.query_head - query.head_start
                    if not 0 <= local_head < query.head_count:
                        raise ValueError("routed query head is missing from probe")
                    valid = query.positions[: query.valid_length]
                    if any(p not in valid for p in window.query_positions):
                        raise ValueError(
                            "requested Q position was not produced (or is padding)"
                        )
                    rows = []
                    for position in window.query_positions:
                        row = (
                            tensor[valid.index(position), local_head]
                            .detach()
                            .to(dtype=torch.float32)
                        )
                        if not torch.isfinite(row).all():
                            raise ValueError(
                                "probe query must contain finite float32 values"
                            )
                        rows.append(tuple(row.tolist()))
                    prepared.append(
                        PreparedProbeQuery(route, query.version, tuple(rows))
                    )
            self._match(window)
            result = PreparedProbeSearch(window, tuple(prepared))
            self._prepared = result
            return result
        except BaseException:
            if self._pending is window:
                self.invalidate()
            raise

    async def search(self, prepared: PreparedProbeSearch, client: PVDShardSearchClient):
        """Publish only a complete same-version set; never mark KV ready."""
        self._match(prepared.window)
        if (
            self._prepared is not prepared
            or self._searching is not None
            or self._ready is not None
        ):
            raise ValueError("search requires this session's unused prepared operation")
        window = prepared.window
        self._searching = window
        try:
            results = []
            versions = None
            for query in prepared.queries:
                self._match(window)
                identity = query.route.identity
                if versions is not None:
                    if identity.expected_index_version not in (
                        None,
                        versions[0],
                    ) or identity.expected_id_mapping_version not in (
                        None,
                        versions[1],
                    ):
                        raise ValueError("routes pin inconsistent index versions")
                    identity = replace(
                        identity,
                        expected_index_version=versions[0],
                        expected_id_mapping_version=versions[1],
                    )
                reply = await client.search(
                    identity,
                    queries=query.rows,
                    top_k=query.route.top_k,
                    scope=query.route.scope,
                )
                self._match(window)
                if reply.identity != identity:
                    raise ValueError(
                        "search reply identity differs from the submitted query"
                    )
                current_versions = (reply.index_version, reply.id_mapping_version)
                if versions is not None and current_versions != versions:
                    raise ValueError("index changed within a probe window")
                versions = current_versions
                results.append(reply)
            self._ready = ProbeSelection(window, prepared.queries, tuple(results))
        except BaseException:
            if self._pending is window:
                self.invalidate()
            raise
        finally:
            self._searching = None

    def take_selection(self, window: ProbeWindow) -> ProbeSelection:
        """Consume logical choices once, not install KV. Recheck before use.

        Future transfer/installation must independently validate the attached
        incarnation, Entry, versions and target window at its own boundary.
        """
        self._match(window)
        if self._ready is None:
            raise ValueError("search selection is not ready")
        result = self._ready
        self.invalidate()
        return result
