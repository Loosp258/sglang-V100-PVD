"""Bounded single-shard search client for future D-side probe integration.

This returns logical selections only. It never fetches KV, updates a refresh
clock, selects a V group, or mutates committed Decode state. The caller must
supply the selected shard endpoint and its trusted Prompt layout explicitly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import aiohttp
import orjson
from sglang.srt.disaggregation.pvd.prompt_index import SearchRequestIdentity

SEARCH_PROTOCOL = "pvd.search.v1"
SEARCH_BATCH_PROTOCOL = "pvd.search.batch.v1"
logger = logging.getLogger(__name__)


class ShardSearchError(RuntimeError):
    """No result may be installed after this error."""


class SearchReplyError(ShardSearchError):
    """Malformed, stale or mismatched remote reply; do not retry blindly."""


class SearchRefused(ShardSearchError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"V search refused ({status}, {code}): {message}")
        self.status = status
        self.code = code
        # No hidden retries: the request's scheduler decides whether to wait.
        self.retryable = code in ("index_not_ready", "index_capacity")


class SearchTransportError(ShardSearchError):
    pass


def _positive(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class SearchScope:
    """Trusted request-local bounds, never inferred from a V reply."""

    prompt_tokens: int
    page_size: int
    head_dim: int
    metric: str

    def __post_init__(self):
        for field in ("prompt_tokens", "page_size", "head_dim"):
            _positive(field, getattr(self, field))
        if self.metric not in ("ip", "l2"):
            raise ValueError("unsupported search metric")


@dataclass(frozen=True)
class ShardSearchResult:
    identity: SearchRequestIdentity
    index_version: str
    id_mapping_version: str
    token_ids: tuple[int, ...]
    page_ids: tuple[int, ...]
    scores: tuple[float, ...]
    metric: str
    validated: tuple[str, ...]


class PVDShardSearchClient:
    def __init__(
        self,
        base_url: str,
        *,
        session: aiohttp.ClientSession | None = None,
        background_loop: asyncio.AbstractEventLoop | None = None,
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 2 * 1024 * 1024,
    ):
        if not isinstance(base_url, str) or not base_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("an explicit HTTP(S) V shard endpoint is required")
        if not _finite_number(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if background_loop is not None and (
            session is not None
            or not isinstance(background_loop, asyncio.AbstractEventLoop)
            or not background_loop.is_running()
            or background_loop.is_closed()
        ):
            raise ValueError("background search requires a running owned I/O loop")
        if background_loop is not None:
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            if current_loop is background_loop:
                raise ValueError("background search loop must differ from owner loop")
        self.base_url = base_url.rstrip("/")
        self._session = session
        self._owns_session = session is None
        self._background_loop = background_loop
        self._background_inflight = set()
        self._background_close_future = None
        self._closed = False
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._max_response_bytes = _positive("max_response_bytes", max_response_bytes)

    async def close(self):
        self._closed = True
        if self._background_loop is not None:
            # A cancelled owner await does not cancel a read-only V search.
            # Drain the actual RPC before closing its loop-affine session.
            if not self._background_loop.is_running():
                raise SearchTransportError("V search I/O loop stopped before close")
            pending = tuple(self._background_inflight)
            if pending:
                await asyncio.gather(
                    *(
                        asyncio.shield(asyncio.wrap_future(future))
                        for future in pending
                    ),
                    return_exceptions=True,
                )
            self._background_inflight.clear()
            if self._session is not None and self._owns_session:
                if self._background_close_future is None:
                    coroutine = self._session.close()
                    try:
                        self._background_close_future = (
                            asyncio.run_coroutine_threadsafe(
                                coroutine, self._background_loop
                            )
                        )
                    except BaseException:
                        coroutine.close()
                        raise
                await asyncio.shield(asyncio.wrap_future(self._background_close_future))
            self._session = None
            return
        if self._session is not None and self._owns_session:
            await self._session.close()
        self._session = None

    def _prepare_search(
        self,
        identity: SearchRequestIdentity,
        *,
        queries: Sequence[Sequence[float]],
        top_k: int,
        scope: SearchScope,
    ):
        if self._closed:
            raise ShardSearchError("search client is closed")
        if not isinstance(identity, SearchRequestIdentity) or not isinstance(
            scope, SearchScope
        ):
            raise TypeError("explicit search identity and scope are required")
        _positive("top_k", top_k)
        if top_k > min(512, scope.prompt_tokens):
            raise ValueError("top_k exceeds the request or protocol limit")
        if not isinstance(queries, (list, tuple)) or not 1 <= len(queries) <= 64:
            raise ValueError("queries must contain 1..64 host vectors")
        snapshot = []
        for row in queries:
            if not isinstance(row, (list, tuple)) or len(row) != scope.head_dim:
                raise ValueError("query dimensions disagree with the trusted layout")
            if any(
                not _finite_number(v) or abs(v) > 3.4028234663852886e38 for v in row
            ):
                raise ValueError("query values must be finite numbers")
            snapshot.append(list(row))
        search_id = uuid.uuid4().hex
        payload = {
            "search_protocol": SEARCH_PROTOCOL,
            "search_id": search_id,
            "transfer_id": identity.entry_transfer_id,
            "vector_space": identity.vector_space,
            "positional_encoding": identity.positional_encoding,
            "layer": identity.layer,
            "kv_head": identity.kv_head,
            "queries": snapshot,
            "top_k": top_k,
        }
        for name in ("expected_index_version", "expected_id_mapping_version"):
            value = getattr(identity, name)
            if value is not None:
                payload[name] = value
        return payload, search_id, len(snapshot)

    async def _post_json(self, path, payload, *, encoded_payload=None):
        if self._background_loop is None:
            return await self._post_json_on_loop(
                path, payload, encoded_payload=encoded_payload
            )
        if not self._background_loop.is_running():
            raise SearchTransportError("V search I/O loop stopped")
        coroutine = self._post_json_on_loop(
            path, payload, encoded_payload=encoded_payload
        )
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, self._background_loop)
        except BaseException:
            coroutine.close()
            raise
        self._background_inflight.add(future)
        try:
            # Shielding matters: cancelling the scheduler-side refresh must
            # not make the proxy look drained before aiohttp has unwound.
            return await asyncio.shield(asyncio.wrap_future(future))
        finally:
            if future.done():
                self._background_inflight.discard(future)

    async def _post_json_on_loop(self, path, payload, *, encoded_payload=None):
        timeline_started = (
            time.monotonic()
            if os.environ.get("PVD_PROFILE_REFRESH_TIMELINE") == "1"
            else None
        )
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        # Batched Q rows can be hundreds of KiB. Their wire bytes are already
        # encoded for the size bound; send those exact bytes instead of asking
        # aiohttp to serialize the same floats a second time.
        body = (
            {"json": payload}
            if encoded_payload is None
            else {
                "data": encoded_payload,
                "headers": {"Content-Type": "application/json"},
            }
        )
        try:
            async with self._session.post(
                f"{self.base_url}{path}",
                **body,
                timeout=self._timeout,
                allow_redirects=False,
            ) as response:
                chunks, size = [], 0
                async for chunk in response.content.iter_chunked(65536):
                    size += len(chunk)
                    if size > self._max_response_bytes:
                        raise SearchReplyError("V search reply exceeds byte limit")
                    chunks.append(chunk)
                try:
                    body = json.loads(b"".join(chunks))
                except (ValueError, UnicodeDecodeError) as exc:
                    raise SearchReplyError("V search returned invalid JSON") from exc
                if not isinstance(body, dict):
                    raise SearchReplyError("V search reply must be an object")
                if response.status != 200:
                    code = body.get("code", "request_refused")
                    # Only the defined status/code pairs permit a retry.
                    if (response.status, code) not in (
                        (400, "index_not_ready"),
                        (507, "index_capacity"),
                    ):
                        code = "request_refused"
                    raise SearchRefused(
                        response.status, code, str(body.get("error", ""))
                    )
        except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
            raise SearchTransportError(
                f"V shard search transport failed: {exc}"
            ) from exc
        if timeline_started is not None:
            ended = time.monotonic()
            try:
                logger.info(
                    "PVD timeline event=search_http path=%s t_start=%.6f "
                    "t_end=%.6f seconds=%.6f",
                    path,
                    timeline_started,
                    ended,
                    ended - timeline_started,
                )
            except Exception:
                pass  # An HTTP result stays valid if diagnostics fail.
        return body

    async def search(
        self,
        identity: SearchRequestIdentity,
        *,
        queries: Sequence[Sequence[float]],
        top_k: int,
        scope: SearchScope,
    ) -> ShardSearchResult:
        payload, search_id, query_count = self._prepare_search(
            identity, queries=queries, top_k=top_k, scope=scope
        )
        body = await self._post_json("/internal/v1/indexes/search", payload)
        return self._validate_reply(
            body, identity, scope, search_id, query_count, top_k
        )

    async def search_many(self, requests):
        """One bounded pinned RPC; each result remains independently checked."""
        profile = os.environ.get("PVD_PROFILE_D_SEARCH_BATCH") == "1"
        started = time.perf_counter() if profile else 0.0
        if not isinstance(requests, (list, tuple)) or not 1 <= len(requests) <= 32:
            raise ValueError("search batch requires 1..32 requests")
        prepared = []
        for request in requests:
            if not isinstance(request, (list, tuple)) or len(request) != 4:
                raise TypeError(
                    "batch request must carry identity, queries, top_k, scope"
                )
            identity, queries, top_k, scope = request
            payload, search_id, count = self._prepare_search(
                identity, queries=queries, top_k=top_k, scope=scope
            )
            if (
                identity.expected_index_version is None
                or identity.expected_id_mapping_version is None
            ):
                raise ValueError("batch search requires both version pins")
            prepared.append((identity, scope, top_k, payload, search_id, count))
        first = prepared[0][0]
        if any(
            (
                identity.entry_transfer_id,
                identity.vector_space,
                identity.positional_encoding,
                identity.expected_index_version,
                identity.expected_id_mapping_version,
            )
            != (
                first.entry_transfer_id,
                first.vector_space,
                first.positional_encoding,
                first.expected_index_version,
                first.expected_id_mapping_version,
            )
            for identity, *_ in prepared[1:]
        ):
            raise ValueError("batch searches require one pinned Entry/space/version")
        if (
            sum(item[5] for item in prepared) > 512
            or sum(item[5] * item[2] for item in prepared) > 16384
        ):
            raise ValueError("search batch row or result bound exceeded")
        batch_id = uuid.uuid4().hex
        payload = {
            "batch_protocol": SEARCH_BATCH_PROTOCOL,
            "batch_id": batch_id,
            "items": [item[3] for item in prepared],
        }
        if profile:
            prepared_at = time.perf_counter()
        encoded = orjson.dumps(payload)
        if len(encoded) > 3 * 1024 * 1024:
            raise ValueError("search batch request exceeds 3 MiB")
        if profile:
            encoded_at = time.perf_counter()
        body = await self._post_json(
            "/internal/v1/indexes/search-batch", payload, encoded_payload=encoded
        )
        if profile:
            replied_at = time.perf_counter()
        results = body.get("results")
        if (
            body.get("batch_protocol") != SEARCH_BATCH_PROTOCOL
            or body.get("batch_id") != batch_id
            or not isinstance(results, list)
            or len(results) != len(prepared)
        ):
            raise SearchReplyError("search batch envelope mismatch")
        validated = tuple(
            self._validate_reply(reply, identity, scope, search_id, count, top_k)
            for reply, (identity, scope, top_k, _, search_id, count) in zip(
                results, prepared, strict=True
            )
        )
        if profile:
            logger.info(
                "PVD D search-batch items=%d query_rows=%d bytes=%d "
                "prepare_ms=%.3f encode_ms=%.3f http_ms=%.3f "
                "validate_ms=%.3f total_ms=%.3f",
                len(prepared),
                sum(item[5] for item in prepared),
                len(encoded),
                (prepared_at - started) * 1000,
                (encoded_at - prepared_at) * 1000,
                (replied_at - encoded_at) * 1000,
                (time.perf_counter() - replied_at) * 1000,
                (time.perf_counter() - started) * 1000,
            )
        return validated

    @staticmethod
    def _validate_reply(body, identity, scope, search_id, query_count, top_k):
        if not isinstance(body, dict):
            raise SearchReplyError("search reply must be an object")
        expected = {
            "search_protocol": SEARCH_PROTOCOL,
            "search_id": search_id,
            "transfer_id": identity.entry_transfer_id,
            "vector_space": identity.vector_space,
            "positional_encoding": identity.positional_encoding,
            "layer": identity.layer,
            "kv_head": identity.kv_head,
            "metric": scope.metric,
        }
        for field, value in expected.items():
            if type(body.get(field)) is not type(value) or body[field] != value:
                raise SearchReplyError(f"search reply {field} mismatch")
        for field, pin in (
            ("index_version", identity.expected_index_version),
            ("id_mapping_version", identity.expected_id_mapping_version),
        ):
            value = body.get(field)
            if (
                not isinstance(value, str)
                or not value.strip()
                or (pin is not None and value != pin)
            ):
                raise SearchReplyError(f"search reply {field} mismatch")
        checked = body.get("validated")
        required = {
            "vector_space",
            "entry_transfer_id",
            "positional_encoding",
            "layer",
            "kv_head",
        }
        if identity.expected_index_version is not None:
            required.add("index_version")
        if identity.expected_id_mapping_version is not None:
            required.add("id_mapping_version")
        if (
            not isinstance(checked, list)
            or any(not isinstance(v, str) for v in checked)
            or len(checked) != len(set(checked))
            or set(checked) != required
        ):
            raise SearchReplyError("search reply validation claims mismatch")
        tokens, pages, scores = (
            body.get(k) for k in ("token_ids", "page_ids", "scores")
        )
        if (
            not isinstance(tokens, list)
            or not 1 <= len(tokens) <= min(scope.prompt_tokens, query_count * top_k)
            or any(
                type(t) is not int or not 0 <= t < scope.prompt_tokens for t in tokens
            )
            or len(tokens) != len(set(tokens))
        ):
            raise SearchReplyError(
                "search reply token selection is out of bounds or duplicated"
            )
        if (
            not isinstance(pages, list)
            or any(type(p) is not int for p in pages)
            or pages != sorted({t // scope.page_size for t in tokens})
        ):
            raise SearchReplyError("search reply page mapping mismatch")
        if (
            not isinstance(scores, list)
            or len(scores) != len(tokens)
            or any(not _finite_number(s) for s in scores)
        ):
            raise SearchReplyError("search reply scores are invalid")
        if scope.metric == "l2" and any(s > 0 for s in scores):
            raise SearchReplyError("L2 similarity must be non-positive")
        return ShardSearchResult(
            identity,
            body["index_version"],
            body["id_mapping_version"],
            tuple(tokens),
            tuple(pages),
            tuple(scores),
            scope.metric,
            tuple(checked),
        )
