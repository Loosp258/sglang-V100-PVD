"""Standalone CPU probe -> single-shard retrieval, NOT a serving integration.

One owner thread/event loop owns each session. Prediction runs synchronously
on that thread; only HTTP yields. A selected logical result is NOT KV_READY,
an RDMA grant, an installation, or permission to advance a PrefetchClock.
No live request, writable committed KV or sampler is passed to providers.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from contextlib import nullcontext
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
    SearchRefused,
    SearchScope,
    ShardSearchResult,
)


class StaleProbeSearch(ValueError):
    """The operation no longer belongs to a live request/window."""


# Bound materialized query rows rather than route count alone. A real model can
# have hundreds of layer/Q-head routes even when it probes only one position.
MAX_PREPARED_QUERY_ROWS = 4096
# Search RPCs are per layer/KV head. Keep network overlap bounded; the V
# backend may still serialize GPU work, and its scratch budget remains binding.
MAX_CONCURRENT_SHARD_SEARCHES = 8


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
    positions, or in-prefix Q when query_source is committed at a late start.
    This module invents no last-token or cross-position policy.
    """

    incarnation: str
    operation_id: str
    entry_transfer_id: str
    prefix: CommittedPrefix
    target_tokens: int
    query_positions: tuple[int, ...]
    query_source: str = "predicted"


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
    # Each response is the union for these complete query indices, not a
    # fabricated per-head response. None retains legacy one-query/one-result.
    query_groups: tuple[tuple[int, ...], ...] | None = None


def _group_search_queries(queries, source_scopes):
    """Coalesce compatible rows, keeping every complete Q-head's provenance.

    A prepared query has at most 64 rows. Chunk only between queries so one
    head never loses the identity of the positions it contributed.
    """
    grouped = {}
    for index, (query, source) in enumerate(zip(queries, source_scopes, strict=True)):
        route = query.route
        key = (source, route.identity, route.scope, route.top_k, query.query_version)
        if not 1 <= len(query.rows) <= 64:
            raise ValueError("prepared query must contain 1..64 rows")
        grouped.setdefault(key, []).append(index)
    chunks = []
    for indices in grouped.values():
        chunk, count = [], 0
        for index in indices:
            size = len(queries[index].rows)
            if count + size > 64:
                chunks.append(tuple(chunk))
                chunk, count = [], 0
            chunk.append(index)
            count += size
        chunks.append(tuple(chunk))
    return tuple(chunks)


class ProbeSearchSession:
    """One request incarnation; explicit invalidation, no hidden retries.

    Normal committed progress toward the boundary does not invalidate the
    original prediction snapshot (approximation is allowed). Call invalidate
    for prefix replacement and replace_entry for Entry changes. A retraction
    that decreases the committed count requires a new session incarnation.
    Calls must be serialized on the owner thread; not a thread-safe registry.
    """

    def __init__(self, request_id: str, entry_transfer_id: str, *, incarnation=None):
        _text("request_id", request_id)
        _text("entry_transfer_id", entry_transfer_id)
        self.request_id = request_id
        self.entry_transfer_id = entry_transfer_id
        if incarnation is not None:
            _text("incarnation", incarnation)
        self.incarnation = incarnation if incarnation is not None else uuid.uuid4().hex
        self._controlled = incarnation is not None
        self._last_controlled_round = -1
        self._children = ()
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
        install_epoch=None,
        query_source="predicted",
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
        if self._controlled:
            from sglang.srt.disaggregation.pvd.sparse_install import InstallEpoch

            if not isinstance(install_epoch, InstallEpoch) or (
                install_epoch.request_id,
                install_epoch.incarnation,
                install_epoch.entry_transfer_id,
                install_epoch.target_tokens,
            ) != (
                self.request_id,
                self.incarnation,
                self.entry_transfer_id,
                target_tokens,
            ):
                raise StaleProbeSearch(
                    "controlled probe requires its exact installation epoch"
                )
            if install_epoch.round <= self._last_controlled_round:
                raise StaleProbeSearch("controlled probe epoch cannot be replayed")
        elif install_epoch is not None:
            raise ValueError("standalone session cannot adopt an installation epoch")
        if prefix.committed_position < self._observed:
            raise StaleProbeSearch("committed token count regressed")
        if query_source not in ("predicted", "committed"):
            raise ValueError("unknown query source")
        committed = query_source == "committed"
        if committed and (
            not self._controlled or target_tokens != prefix.committed_position
        ):
            raise ValueError(
                "committed-prefix capture requires a controlled boundary start"
            )
        if not committed and target_tokens <= prefix.committed_position:
            raise ValueError(
                "a periodic probe needs a future target boundary; bootstrap is separate"
            )
        if (
            not isinstance(query_positions, tuple)
            or not 1 <= len(query_positions) <= 64
            or any(
                type(p) is not int
                or (
                    not 0 <= p < len(prefix.tokens)
                    if committed
                    else p < len(prefix.tokens)
                )
                for p in query_positions
            )
            or tuple(sorted(set(query_positions))) != query_positions
        ):
            raise ValueError(
                "query positions must be 1..64 distinct ascending positions within the declared query source"
            )
        self._observed = prefix.committed_position
        window = ProbeWindow(
            self.incarnation,
            install_epoch.operation_id if self._controlled else uuid.uuid4().hex,
            self.entry_transfer_id,
            prefix,
            target_tokens,
            query_positions,
            query_source,
        )
        self._pending = window
        if self._controlled:
            self._last_controlled_round = install_epoch.round
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
        for child in self._children:
            child.close()
        self._children = ()
        self._pending = self._prepared = self._ready = None

    def replace_entry(self, entry_transfer_id: str):
        if self._closed:
            raise StaleProbeSearch("request is closed")
        if self._controlled:
            raise ValueError(
                "controlled Entry replacement requires a new request controller"
            )
        _text("entry_transfer_id", entry_transfer_id)
        self.invalidate()
        self.entry_transfer_id = entry_transfer_id

    def close(self):
        self.invalidate()
        self._closed = True

    def _validate_pipeline(self, pipeline):
        if torch.device(pipeline.draft_config.device).type != "cpu":
            raise PredictionConfigError(
                "this standalone bridge only validates CPU execution"
            )

    def _query_device(self, tensor):
        return tensor.device.type == "cpu"

    def _query_rows(self, tensor, indices, head, pipeline):
        rows = []
        for index in indices:
            row = tensor[index, head].detach().to(dtype=torch.float32)
            if not torch.isfinite(row).all():
                raise ValueError("probe query must contain finite float32 values")
            rows.append(tuple(row.tolist()))
        return tuple(rows)

    def _prepare_scope(self, routes, window):
        return nullcontext()

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
            self._validate_pipeline(pipeline)
            if (
                not isinstance(routes, tuple)
                or not routes
                or len(routes) * len(window.query_positions) > MAX_PREPARED_QUERY_ROWS
            ):
                raise ValueError(
                    "explicit single-shard routes exceed the prepared-query row bound"
                )
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
            branch = (
                pipeline.committed_query_branch(window.prefix, window.query_positions)
                if window.query_source == "committed"
                else pipeline.query_branch(window.prefix)
            )
            with self._prepare_scope(routes, window), branch as queries:
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
                        or not self._query_device(tensor)
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
                    rows = self._query_rows(
                        tensor,
                        tuple(valid.index(p) for p in window.query_positions),
                        local_head,
                        pipeline,
                    )
                    prepared.append(PreparedProbeQuery(route, query.version, rows))
            self._match(window)
            result = PreparedProbeSearch(window, tuple(prepared))
            self._prepared = result
            return result
        except BaseException:
            if self._pending is window:
                self.invalidate()
            raise

    async def search(
        self,
        prepared: PreparedProbeSearch,
        client: PVDShardSearchClient,
        *,
        index_ready_wait_seconds: float = 0.0,
    ):
        """Publish only a complete same-version set; never mark KV ready."""
        if (
            type(index_ready_wait_seconds) not in (int, float)
            or not math.isfinite(index_ready_wait_seconds)
            or index_ready_wait_seconds < 0
        ):
            raise ValueError("index ready wait must be finite and non-negative")
        self._match(prepared.window)
        if (
            self._prepared is not prepared
            or self._searching is not None
            or self._ready is not None
            or self._children
        ):
            raise ValueError("search requires this session's unused prepared operation")
        window = prepared.window
        ready_deadline = time.monotonic() + index_ready_wait_seconds
        self._searching = window
        try:
            # D compute rank is not a V version namespace: TP1 D can search
            # several V shards. Route identity is trusted local configuration,
            # never a server-provided key. Legacy single-shard clients retain
            # exactly their old one-version-per-window behaviour.
            from sglang.srt.disaggregation.pvd.search_routing import (
                RoutedShardSearchClient,
            )

            source_scopes = tuple(
                client.version_scope(query.route.identity)
                if isinstance(client, RoutedShardSearchClient)
                else 0
                for query in prepared.queries
            )
            query_groups = _group_search_queries(prepared.queries, source_scopes)
            source_versions = {}

            async def search_group(group_index, versions):
                members = query_groups[group_index]
                query = prepared.queries[members[0]]
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
                query_rows = tuple(
                    row for index in members for row in prepared.queries[index].rows
                )
                while True:
                    self._match(window)
                    try:
                        # A readiness wait bounds the whole HTTP attempt, not
                        # only the sleeps between retryable refusals. Otherwise
                        # a slow reply can publish a selection after expiry.
                        remaining = ready_deadline - time.monotonic()
                        if index_ready_wait_seconds and remaining <= 0:
                            raise TimeoutError("V index readiness deadline expired")
                        search = client.search(
                            identity,
                            queries=query_rows,
                            top_k=query.route.top_k,
                            scope=query.route.scope,
                        )
                        if index_ready_wait_seconds:
                            try:
                                reply = await asyncio.wait_for(
                                    search, timeout=remaining
                                )
                            except asyncio.TimeoutError as exc:
                                raise TimeoutError(
                                    "V index readiness deadline expired"
                                ) from exc
                        else:
                            reply = await search
                    except SearchRefused as exc:
                        remaining = ready_deadline - time.monotonic()
                        if not exc.retryable or remaining <= 0:
                            raise
                        # Same immutable Q/Entry/route; no new draft or probe.
                        await asyncio.sleep(min(0.2, remaining))
                    else:
                        if (
                            index_ready_wait_seconds
                            and time.monotonic() >= ready_deadline
                        ):
                            raise TimeoutError("V index readiness deadline expired")
                        break
                self._match(window)
                if reply.identity != identity:
                    raise ValueError(
                        "search reply identity differs from the submitted query"
                    )
                current_versions = (reply.index_version, reply.id_mapping_version)
                if versions is not None and current_versions != versions:
                    raise ValueError("index changed within a probe window")
                return reply

            results = [None] * len(query_groups)
            # The first search to each selected V source establishes its
            # version before any other request to that source is submitted.
            # This preserves the old pin-before-HTTP contract even when the
            # remaining independent layer/head queries overlap.
            for index, members in enumerate(query_groups):
                source = source_scopes[members[0]]
                if source not in source_versions:
                    reply = await search_group(index, None)
                    results[index] = reply
                    source_versions[source] = (
                        reply.index_version,
                        reply.id_mapping_version,
                    )

            semaphore = asyncio.Semaphore(MAX_CONCURRENT_SHARD_SEARCHES)

            async def search_pinned(index, versions):
                async with semaphore:
                    return await search_group(index, versions)

            tasks = {
                index: asyncio.create_task(
                    search_pinned(index, source_versions[source_scopes[members[0]]])
                )
                for index, members in enumerate(query_groups)
                if results[index] is None
            }
            try:
                if tasks:
                    replies = await asyncio.gather(*tasks.values())
                    for index, reply in zip(tasks, replies, strict=True):
                        results[index] = reply
            finally:
                for task in tasks.values():
                    if not task.done():
                        task.cancel()
                if tasks:
                    await asyncio.gather(*tasks.values(), return_exceptions=True)
            self._match(window)
            self._ready = ProbeSelection(
                window, prepared.queries, tuple(results), query_groups
            )
        except BaseException:
            if self._pending is window:
                self.invalidate()
            raise
        finally:
            self._searching = None

    def fork_prepared(self, prepared, partitions):
        """Partition this actual capture BEFORE search; never relabel replies.

        One draft/probe invocation supplies all shards. Each child keeps the
        exact same immutable window, but pins its own shard's index version.
        Parent invalidation closes every child, including in-flight searches.
        Partition values are indices into prepared.queries, covering each once.
        Aggregate route-position rows remain bounded by prepare(); not a serving API.
        """
        self._match(prepared.window)
        if (
            not self._controlled
            or self._prepared is not prepared
            or self._children
            or self._searching is not None
            or self._ready is not None
        ):
            raise ValueError("fork requires this controlled session's unused capture")
        parts = {rank: tuple(indices) for rank, indices in partitions.items()}
        flat = [i for indices in parts.values() for i in indices]
        if (
            not parts
            or any(type(r) is not int or r < 0 for r in parts)
            or any(not indices for indices in parts.values())
            or any(type(i) is not int for i in flat)
            or sorted(flat) != list(range(len(prepared.queries)))
        ):
            raise ValueError("shard partitions must cover every query exactly once")
        result = {}
        for rank, indices in parts.items():
            child = ProbeSearchSession(
                self.request_id, self.entry_transfer_id, incarnation=self.incarnation
            )
            child._pending = prepared.window
            child._observed = self._observed
            child._last_controlled_round = self._last_controlled_round
            child_prepared = PreparedProbeSearch(
                prepared.window, tuple(prepared.queries[i] for i in indices)
            )
            child._prepared = child_prepared
            result[rank] = (child, child_prepared)
        self._children = tuple(child for child, _ in result.values())
        return result

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
