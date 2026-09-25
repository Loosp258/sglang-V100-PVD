"""HTTP control plane for the rank-sharded PVD vector node.

The data plane never goes through HTTP.  These endpoints only exchange immutable
entry manifests, registered-memory descriptors, lifecycle transitions and
delivery acknowledgements.  P->V and V->D KV bytes are moved by TransferEngine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from typing import Any, Dict, Mapping, Optional

import aiohttp
import torch
from aiohttp import web
from sglang.srt.disaggregation.pvd.coordinator import (
    CoordinatorError,
    LocalShardClient,
    ShardClient,
    VectorCoordinator,
)
from sglang.srt.disaggregation.pvd.index_search import IndexNotReadyError
from sglang.srt.disaggregation.pvd.protocol import (
    FirstTokenMetadata,
    KVEntryKey,
    KVEntryManifest,
    ProtocolValidationError,
    RemoteRegionDescriptor,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.request_state import InvalidStateTransition
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferCapacityError,
    TransportState,
)
from sglang.srt.disaggregation.pvd.vector_store import (
    EntryConflictError,
    EntryNotFoundError,
    ResourceExhaustedError,
    VectorKVStore,
)

logger = logging.getLogger(__name__)


def _json_error(message: str, status: int) -> web.Response:
    return web.json_response({"error": message}, status=status)


@web.middleware
async def pvd_error_middleware(request: web.Request, handler):
    try:
        return await handler(request)
    except IndexNotReadyError as exc:
        return web.json_response(
            {"error": str(exc), "code": "index_not_ready"}, status=400
        )
    except TransferCapacityError as exc:
        return web.json_response(
            {"error": str(exc), "code": "index_capacity"}, status=507
        )
    except ProtocolValidationError as exc:
        return _json_error(str(exc), 400)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return _json_error(f"invalid request: {exc}", 400)
    except EntryNotFoundError as exc:
        return _json_error(str(exc), 404)
    except (EntryConflictError, InvalidStateTransition) as exc:
        return _json_error(str(exc), 409)
    except ResourceExhaustedError as exc:
        return _json_error(str(exc), 507)
    except CoordinatorError as exc:
        return _json_error(str(exc), 409)


async def _payload(request: web.Request) -> Dict[str, Any]:
    value = await request.json()
    if not isinstance(value, dict):
        raise TypeError("JSON body must be an object")
    return value


def _key(value: Mapping[str, Any]) -> KVEntryKey:
    return KVEntryKey.from_dict(value["key"])


def _transport_state(value: object) -> TransportState:
    """Parse a transport state name strictly.

    An unrecognised or non-string value is rejected rather than coerced: a
    malformed report must never be read as a terminal state.
    """
    if not isinstance(value, str):
        raise ValueError("transport state must be a string")
    try:
        return TransportState(value)
    except ValueError as exc:
        raise ValueError(f"unknown transport state {value!r}") from exc


def _closed_flag(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("closed must be a boolean")
    return value


class HttpShardClient(ShardClient):
    """Rank-0 client for the other rank's private control service."""

    def __init__(
        self,
        rank: int,
        base_url: str,
        *,
        session: Optional[aiohttp.ClientSession] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.rank = rank
        self.base_url = base_url.rstrip("/")
        self._session = session
        self._owns_session = session is None
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._closed = False

    async def _request(self, method: str, path: str, payload=None) -> Mapping:
        if self._closed:
            raise CoordinatorError("V shard control client is closed")
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        async with self._session.request(
            method, f"{self.base_url}{path}", json=payload
        ) as response:
            body = await response.json()
            if response.status >= 400:
                raise CoordinatorError(
                    f"V rank {self.rank} returned {response.status}: "
                    f"{body.get('error', body)}"
                )
            return body

    async def create_entry(
        self, manifest: KVEntryManifest, *, uploader_epoch: Optional[str] = None
    ) -> Mapping:
        payload = {"manifest": manifest.to_dict()}
        if uploader_epoch is not None:
            payload["uploader_epoch"] = uploader_epoch
        return await self._request("POST", "/internal/v1/entries", payload)

    async def begin_p_write(
        self, key: KVEntryKey, identity: Optional[WriteIdentity] = None
    ) -> Mapping:
        payload = {"key": key.to_dict()}
        if identity is not None:
            payload["identity"] = identity.to_dict()
        return await self._request("POST", "/internal/v1/entries/begin", payload)

    async def sync_upload(
        self, identity: WriteIdentity, state: TransportState, closed: bool
    ) -> Mapping:
        return await self._request(
            "POST",
            "/internal/v1/uploads/sync",
            {
                "identity": identity.to_dict(),
                "state": state.value,
                "closed": bool(closed),
            },
        )

    async def commit_p_write(self, key: KVEntryKey, received_bytes: int) -> Mapping:
        return await self._request(
            "POST",
            "/internal/v1/entries/commit",
            {"key": key.to_dict(), "received_bytes": received_bytes},
        )

    async def reserve_delivery(
        self,
        key: KVEntryKey,
        delivery_id: str,
        destination: RemoteRegionDescriptor,
    ) -> Mapping:
        return await self._request(
            "POST",
            "/internal/v1/deliveries",
            {
                "key": key.to_dict(),
                "delivery_id": delivery_id,
                "destination": destination.to_dict(),
            },
        )

    async def start_delivery(self, key: KVEntryKey, delivery_id: str) -> Mapping:
        return await self._request(
            "POST",
            "/internal/v1/deliveries/start",
            {"key": key.to_dict(), "delivery_id": delivery_id},
        )

    async def reserve_fanin_delivery(
        self, manifest, *, expected_sender_epoch=None
    ) -> Mapping:
        return await self._request(
            "POST",
            "/internal/v1/fanin/reserve",
            {"manifest": manifest, "expected_sender_epoch": expected_sender_epoch},
        )

    async def fence_fanin_delivery(self, manifest, identity: WriteIdentity) -> Mapping:
        return await self._request(
            "POST",
            "/internal/v1/fanin/fence",
            {"manifest": manifest, "identity": identity.to_dict()},
        )

    async def poll_delivery(self, key: KVEntryKey, delivery_id: str) -> Mapping:
        return await self._request(
            "POST",
            "/internal/v1/deliveries/poll",
            {"key": key.to_dict(), "delivery_id": delivery_id},
        )

    async def ack_delivery(self, key: KVEntryKey, delivery_id: str) -> Mapping:
        return await self._request(
            "POST",
            "/internal/v1/deliveries/ack",
            {"key": key.to_dict(), "delivery_id": delivery_id},
        )

    async def cancel_delivery(
        self, key: KVEntryKey, delivery_id: str, reason: str
    ) -> Mapping:
        return await self._request(
            "POST",
            "/internal/v1/deliveries/cancel",
            {
                "key": key.to_dict(),
                "delivery_id": delivery_id,
                "reason": reason,
            },
        )

    async def cancel_entry(self, key: KVEntryKey, reason: str) -> None:
        await self._request(
            "POST",
            "/internal/v1/entries/cancel",
            {"key": key.to_dict(), "reason": reason},
        )

    async def fence_delivery(self, identity: WriteIdentity) -> Mapping:
        return await self._request(
            "POST",
            "/internal/v1/deliveries/fence",
            {"identity": identity.to_dict()},
        )

    async def release_entry(self, key: KVEntryKey) -> None:
        await self._request(
            "POST", "/internal/v1/entries/release", {"key": key.to_dict()}
        )

    async def health(self) -> Mapping:
        return await self._request("GET", "/internal/health")

    async def close(self) -> None:
        self._closed = True
        if self._session is not None and self._owns_session:
            await self._session.close()
        self._session = None


def _optional_text(data: Mapping[str, Any], field: str) -> Optional[str]:
    """A version pin the caller may or may not have. Absent stays absent.

    Returning ``None`` for a missing field is the whole point: the manager
    reports which pins it compared, and a pin invented here would be compared
    against the value it was invented from.
    """
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string when supplied")
    return value


def _stage_ms(timings: Mapping[str, float]) -> Dict[str, float]:
    """Keep diagnostic values bounded and independent of request contents."""
    return {name: round(seconds * 1000, 3) for name, seconds in timings.items()}


def _run_search(index, identity, queries, top_k: int, timings=None, queued_at=0.0):
    """Build the query tensor and search, both off the HTTP event loop.

    Materialising a 64 x head_dim tensor is small but not free, and the search
    itself is not: neither belongs on the loop that is also answering
    delivery polls.
    """
    if timings is not None:
        timings["thread_wait"] = time.perf_counter() - queued_at
        started = time.perf_counter()
    tensor = torch.tensor(queries, dtype=torch.float32)
    if timings is not None:
        timings["query_tensor"] = time.perf_counter() - started
    if timings is None:
        return index.search(identity, queries=tensor, top_k=top_k)
    return index.search(identity, queries=tensor, top_k=top_k, timings=timings)


def create_shard_app(
    store: VectorKVStore, *, preflight: Optional[Mapping[str, Any]] = None
) -> web.Application:
    # Batched per-head queries can exceed aiohttp's 1 MiB default. The batch
    # endpoint enforces its own item/row/response limits before indexing.
    app = web.Application(
        middlewares=[pvd_error_middleware], client_max_size=4 * 1024 * 1024
    )

    async def create_entry(request):
        data = await _payload(request)
        uploader_epoch = data.get("uploader_epoch")
        if uploader_epoch is not None and not isinstance(uploader_epoch, str):
            raise ValueError("uploader_epoch must be a string")
        return web.json_response(
            store.create_entry(
                KVEntryManifest.from_dict(data["manifest"]),
                uploader_epoch=uploader_epoch,
            ).to_dict()
        )

    async def begin_entry(request):
        data = await _payload(request)
        identity = data.get("identity")
        return web.json_response(
            store.begin_p_write(
                _key(data),
                WriteIdentity.from_dict(identity) if identity is not None else None,
            ).to_dict()
        )

    async def sync_upload(request):
        data = await _payload(request)
        result = await asyncio.to_thread(
            store.sync_upload,
            WriteIdentity.from_dict(data["identity"]),
            _transport_state(data["state"]),
            _closed_flag(data["closed"]),
        )
        return web.json_response(result)

    async def commit_entry(request):
        data = await _payload(request)
        return web.json_response(
            store.commit_p_write(_key(data), int(data["received_bytes"])).to_dict()
        )

    async def reserve_delivery(request):
        data = await _payload(request)
        return web.json_response(
            store.reserve_delivery(
                _key(data),
                str(data["delivery_id"]),
                RemoteRegionDescriptor.from_dict(data["destination"]),
            ).to_dict()
        )

    async def reserve_fanin(request):
        data = await _payload(request)
        result = await asyncio.to_thread(
            store.reserve_fanin_delivery,
            data["manifest"],
            expected_sender_epoch=data.get("expected_sender_epoch"),
        )
        return web.json_response(result.to_dict())

    async def fence_fanin(request):
        data = await _payload(request)
        result = await asyncio.to_thread(
            store.fence_fanin_delivery,
            data["manifest"],
            WriteIdentity.from_dict(data["identity"]),
        )
        return web.json_response(result)

    async def start_delivery(request):
        data = await _payload(request)
        result = await asyncio.to_thread(
            store.start_delivery, _key(data), str(data["delivery_id"])
        )
        return web.json_response(result.to_dict())

    async def poll_delivery(request):
        data = await _payload(request)
        result = await asyncio.to_thread(
            store.poll_delivery, _key(data), str(data["delivery_id"])
        )
        return web.json_response(result.to_dict())

    async def ack_delivery(request):
        data = await _payload(request)
        return web.json_response(
            store.ack_delivery(_key(data), str(data["delivery_id"])).to_dict()
        )

    async def cancel_delivery(request):
        data = await _payload(request)
        result = await asyncio.to_thread(
            store.cancel_delivery,
            _key(data),
            str(data["delivery_id"]),
            str(data.get("reason", "")),
        )
        return web.json_response(result.to_dict())

    async def cancel_entry(request):
        data = await _payload(request)
        await asyncio.to_thread(
            store.cancel_entry, _key(data), str(data.get("reason", ""))
        )
        return web.json_response({"ok": True})

    async def fence_delivery(request):
        data = await _payload(request)
        result = await LocalShardClient(store).fence_delivery(
            WriteIdentity.from_dict(data["identity"])
        )
        return web.json_response(result)

    async def release_entry(request):
        data = await _payload(request)
        await asyncio.to_thread(store.release_entry, _key(data))
        return web.json_response({"ok": True})

    # Bounds on one retrieval request, so a caller cannot ask for an
    # unbounded response. Section 8 requires response-byte limits.
    max_queries = 64
    max_top_k = 512

    def _require_prompt_index():
        if store.prompt_index is None:
            raise ValueError(
                "this V rank has no prompt index; start it with "
                "--prompt-index-vector-space to enable retrieval"
            )
        return store.prompt_index

    async def progress_indexes(_request):
        """Drive one bounded round of index builds. Never fails an Entry."""
        if store.prompt_index is None:
            return web.json_response({"enabled": False})
        result = await asyncio.to_thread(store.progress_prompt_indexes)
        return web.json_response({"enabled": True, **result})

    async def index_snapshot(_request):
        if store.prompt_index is None:
            return web.json_response({"enabled": False})
        return web.json_response(
            {"enabled": True, **(await asyncio.to_thread(store.prompt_index.snapshot))}
        )

    async def _search_data(data, *, timings=None):
        """Search one (layer, KV head) and return logical token/page ids.

        The request carries its own identity: which model's vector space the
        Q comes from and under which positional-encoding semantics, which
        Entry, layer and KV head, and optionally the index and id-mapping
        versions it was built against. None of that is filled in from this
        rank's configuration -- a same-shaped query from another model must
        be refused here, and it cannot be if the shard supplies the answer it
        is about to check.
        """
        from sglang.srt.disaggregation.pvd.prompt_index import SearchRequestIdentity
        from sglang.srt.disaggregation.pvd.search_client import SEARCH_PROTOCOL

        started = time.perf_counter() if timings is not None else 0.0
        index = _require_prompt_index()
        search_id = data.get("search_id")
        if "search_id" in data or "search_protocol" in data:
            if data.get("search_protocol") != SEARCH_PROTOCOL:
                raise ValueError("unsupported search protocol")
            if not isinstance(search_id, str) or not 1 <= len(search_id) <= 128:
                raise ValueError("search_id must be a non-empty bounded string")
        for field in ("vector_space", "positional_encoding", "transfer_id"):
            value = data.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"a search request must state its {field}; this rank will "
                    "not supply the identity it is meant to verify"
                )
        queries = data.get("queries")
        if not isinstance(queries, list) or not queries:
            raise ValueError("queries must be a non-empty list of vectors")
        if len(queries) > max_queries:
            raise ValueError(f"at most {max_queries} queries per request")
        if not all(isinstance(row, list) and row for row in queries):
            raise ValueError("queries must be equal-length lists of numbers")
        if len({len(row) for row in queries}) != 1:
            raise ValueError("queries must be equal-length lists of numbers")
        if any(
            type(value) not in (int, float)
            or abs(value) > torch.finfo(torch.float32).max
            or not math.isfinite(value)
            for row in queries
            for value in row
        ):
            raise ValueError("queries must contain finite float32 numbers")
        top_k = data.get("top_k", 1)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if top_k > max_top_k:
            raise ValueError(f"top_k must not exceed {max_top_k}")
        identity = SearchRequestIdentity(
            vector_space=str(data["vector_space"]),
            positional_encoding=str(data["positional_encoding"]),
            entry_transfer_id=str(data["transfer_id"]),
            layer=data["layer"],
            kv_head=data["kv_head"],
            expected_index_version=_optional_text(data, "expected_index_version"),
            expected_id_mapping_version=_optional_text(
                data, "expected_id_mapping_version"
            ),
        )
        if timings is not None:
            timings["request_validate"] = time.perf_counter() - started
        # The query is built on the host because that is where JSON numbers
        # arrive; the index manager places it on its backend's device only
        # after identity checks and the search-scratch budget reservation. Both the
        # tensor build and the search run off the event loop.
        queued_at = time.perf_counter() if timings is not None else 0.0
        result = await asyncio.to_thread(
            _run_search, index, identity, queries, top_k, timings, queued_at
        )
        if timings is not None:
            timings["item_total"] = time.perf_counter() - started
        selection = result.selection
        return {
            "search_protocol": SEARCH_PROTOCOL,
            "search_id": search_id,
            "transfer_id": str(data["transfer_id"]),
            "vector_space": identity.vector_space,
            "positional_encoding": identity.positional_encoding,
            "layer": selection.layer,
            "kv_head": selection.kv_head,
            "token_ids": list(selection.token_ids),
            "page_ids": list(selection.page_ids),
            "scores": [float(score) for score in selection.scores],
            "metric": selection.metric,
            "id_mapping_version": result.id_mapping_version,
            "index_version": result.index_version,
            # Exactly which identity comparisons were made. A caller that
            # pinned no version will not see one named here, rather than
            # being told its request was "validated".
            "validated": list(result.validated),
        }

    async def search_index(request):
        timings = {} if os.environ.get("PVD_PROFILE_V_SEARCH") == "1" else None
        result = await _search_data(await _payload(request), timings=timings)
        if timings is not None:
            logger.info("PVD V search stage_ms=%s", _stage_ms(timings))
        return web.json_response(result)

    async def search_index_batch(request):
        from sglang.srt.disaggregation.pvd.search_client import SEARCH_BATCH_PROTOCOL

        data = await _payload(request)
        if data.get("batch_protocol") != SEARCH_BATCH_PROTOCOL:
            raise ValueError("unsupported search batch protocol")
        batch_id = data.get("batch_id")
        if not isinstance(batch_id, str) or not 1 <= len(batch_id) <= 128:
            raise ValueError("batch_id must be a non-empty bounded string")
        items = data.get("items")
        if not isinstance(items, list) or not 1 <= len(items) <= 32:
            raise ValueError("search batch requires 1..32 items")
        if any(not isinstance(item, dict) for item in items):
            raise ValueError("search batch items must be objects")
        if any(
            item.get("search_protocol") != "pvd.search.v1"
            or not isinstance(item.get("search_id"), str)
            or not 1 <= len(item["search_id"]) <= 128
            or type(item.get("top_k")) is not int
            or not 1 <= item["top_k"] <= 512
            or not isinstance(item.get("queries"), list)
            or not 1 <= len(item["queries"]) <= 64
            or any(
                not isinstance(item.get(name), str) or not item[name].strip()
                for name in ("expected_index_version", "expected_id_mapping_version")
            )
            for item in items
        ):
            raise ValueError("batch items require bounded pinned search identities")
        if sum(len(item["queries"]) for item in items) > 512:
            raise ValueError("search batch exceeds 512 query rows")
        if sum(len(item["queries"]) * item["top_k"] for item in items) > 16384:
            raise ValueError("search batch exceeds result-token bound")
        search_ids = [item.get("search_id") for item in items]
        if len(set(search_ids)) != len(items):
            raise ValueError("duplicate search_id in batch")
        # One Entry, vector space and version namespace per batch; otherwise a
        # caller could disguise unrelated work under one admission bound.
        shared = (
            "transfer_id",
            "vector_space",
            "positional_encoding",
            "expected_index_version",
            "expected_id_mapping_version",
        )
        first = tuple(items[0].get(name) for name in shared)
        if any(tuple(item.get(name) for name in shared) != first for item in items[1:]):
            raise ValueError("batch items must share Entry, space and version pins")
        profile = os.environ.get("PVD_PROFILE_V_SEARCH") == "1"
        batch_started = time.perf_counter() if profile else 0.0
        stages = [] if profile else None
        results = []
        for item in items:
            timings = {} if profile else None
            results.append(await _search_data(item, timings=timings))
            if stages is not None:
                stages.append(timings)
        if stages is not None:
            totals = {
                name: sum(item.get(name, 0.0) for item in stages) for name in stages[0]
            }
            totals["batch_total"] = time.perf_counter() - batch_started
            logger.info(
                "PVD V search-batch items=%d query_rows=%d stage_ms=%s",
                len(items),
                sum(len(item["queries"]) for item in items),
                _stage_ms(totals),
            )
        reply = {
            "batch_protocol": SEARCH_BATCH_PROTOCOL,
            "batch_id": batch_id,
            "results": results,
        }
        if (
            len(json.dumps(reply, separators=(",", ":")).encode("utf-8"))
            > 2 * 1024 * 1024
        ):
            raise ValueError("search batch response exceeds 2 MiB")
        return web.json_response(reply)

    async def health(_request):
        snapshot = await asyncio.to_thread(store.snapshot)
        snapshot["preflight"] = dict(preflight or {})
        return web.json_response(snapshot)

    app.add_routes(
        [
            web.post("/internal/v1/entries", create_entry),
            web.post("/internal/v1/entries/begin", begin_entry),
            web.post("/internal/v1/entries/commit", commit_entry),
            web.post("/internal/v1/uploads/sync", sync_upload),
            web.post("/internal/v1/deliveries", reserve_delivery),
            web.post("/internal/v1/fanin/reserve", reserve_fanin),
            web.post("/internal/v1/fanin/fence", fence_fanin),
            web.post("/internal/v1/deliveries/start", start_delivery),
            web.post("/internal/v1/deliveries/poll", poll_delivery),
            web.post("/internal/v1/deliveries/ack", ack_delivery),
            web.post("/internal/v1/deliveries/cancel", cancel_delivery),
            web.post("/internal/v1/deliveries/fence", fence_delivery),
            web.post("/internal/v1/entries/cancel", cancel_entry),
            web.post("/internal/v1/entries/release", release_entry),
            web.post("/internal/v1/indexes/progress", progress_indexes),
            web.post("/internal/v1/indexes/search", search_index),
            web.post("/internal/v1/indexes/search-batch", search_index_batch),
            web.get("/internal/v1/indexes", index_snapshot),
            web.get("/internal/health", health),
        ]
    )
    return app


def create_coordinator_app(coordinator: VectorCoordinator) -> web.Application:
    # Router admissions carry original long prompts; the default 1 MiB would
    # reject them before P can tokenize. KV tensor bytes still never use HTTP.
    app = web.Application(
        middlewares=[pvd_error_middleware], client_max_size=64 * 1024 * 1024
    )

    async def fanin(request):
        if coordinator.fanin is None:
            raise CoordinatorError("full-KV fan-in coordinator is disabled")
        data = await _payload(request)
        operation = request.match_info["operation"]
        if operation == "preflight":
            health = await coordinator.health()
            return web.json_response(
                {
                    "full_kv_fanin": health["full_kv_fanin"],
                    "shards": [
                        {
                            k: shard.get(k)
                            for k in (
                                "rank",
                                "worker_epoch",
                                "rail",
                                "ready",
                                "full_kv_fanin",
                            )
                        }
                        for shard in health["shards"]
                    ],
                }
            )
        if operation in {"reserve", "fence"}:
            result = await getattr(coordinator.fanin, operation)(
                data["manifest"], data["source_epochs"]
            )
        elif operation in {"start", "poll", "ack"}:
            result = await getattr(coordinator.fanin, operation)(
                data["delivery_id"], data["destination_rank"]
            )
        else:
            raise ValueError("unknown fan-in operation")
        return web.json_response(result)

    async def admit_request(request):
        return web.json_response(
            await coordinator.admit_request(await _payload(request))
        )

    async def retrieve(request):
        data = await _payload(request)
        if not isinstance(data["sequences"], list):
            raise ValueError("sequences must be a list")
        return web.json_response(
            {"results": await coordinator.retrieve(data["sequences"])}
        )

    async def fence_retrieval(request):
        data = await _payload(request)
        if not isinstance(data["identities"], list):
            raise ValueError("identities must be a list")
        return web.json_response(
            await coordinator.fence_retrieval(data["delivery_id"], data["identities"])
        )

    async def renew_consumer(request):
        data = await _payload(request)
        return web.json_response(
            await coordinator.renew_consumer(_key(data), data["consumer_id"])
        )

    async def release_consumer(request):
        data = await _payload(request)
        return web.json_response(
            await coordinator.release_consumer(_key(data), data["consumer_id"])
        )

    async def create_entry(request):
        data = await _payload(request)
        uploader_epoch = data.get("uploader_epoch")
        if uploader_epoch is not None and not isinstance(uploader_epoch, str):
            raise ValueError("uploader_epoch must be a string")
        result = await coordinator.create_entry(
            KVEntryManifest.from_dict(data["manifest"]),
            uploader_epoch=uploader_epoch,
            uploader_epochs=data.get("uploader_epochs"),
        )
        return web.json_response(result.to_dict())

    async def selected_shard_routes(request):
        data = await _payload(request)
        return web.json_response(await coordinator.selected_shard_routes(_key(data)))

    async def sync_upload(request):
        data = await _payload(request)
        return web.json_response(
            await coordinator.sync_upload(
                WriteIdentity.from_dict(data["identity"]),
                _transport_state(data["state"]),
                _closed_flag(data["closed"]),
            )
        )

    async def commit_entry(request):
        data = await _payload(request)
        token = data.get("first_token")
        result = await coordinator.commit_shard(
            _key(data),
            int(data["rank"]),
            int(data["received_bytes"]),
            FirstTokenMetadata.from_dict(token) if token is not None else None,
        )
        return web.json_response(result.to_dict())

    async def select(request):
        data = await _payload(request)
        results = await coordinator.select(
            [KVEntryKey.from_dict(value) for value in data["keys"]]
        )
        return web.json_response({"results": results})

    async def reserve_delivery(request):
        data = await _payload(request)
        result = await coordinator.reserve_delivery(
            key=_key(data),
            delivery_id=str(data["delivery_id"]),
            destinations={
                int(rank): RemoteRegionDescriptor.from_dict(descriptor)
                for rank, descriptor in data["destinations"].items()
            },
        )
        return web.json_response(result.to_dict())

    async def start_delivery(request):
        data = await _payload(request)
        result = await coordinator.start_delivery(str(data["delivery_id"]))
        return web.json_response(result.to_dict())

    async def poll_delivery(request):
        data = await _payload(request)
        result = await coordinator.poll_delivery(str(data["delivery_id"]))
        return web.json_response(result.to_dict())

    async def ack_delivery(request):
        data = await _payload(request)
        result = await coordinator.ack_delivery(str(data["delivery_id"]))
        return web.json_response(result.to_dict())

    async def cancel_delivery(request):
        data = await _payload(request)
        result = await coordinator.cancel_delivery(
            str(data["delivery_id"]), str(data.get("reason", ""))
        )
        return web.json_response(result.to_dict())

    async def cancel_entry(request):
        data = await _payload(request)
        result = await coordinator.cancel_entry(_key(data), str(data.get("reason", "")))
        return web.json_response(result.to_dict())

    async def release_entry(request):
        data = await _payload(request)
        result = await coordinator.release_entry(_key(data))
        return web.json_response(result.to_dict())

    async def health(_request):
        return web.json_response(await coordinator.health())

    app.add_routes(
        [
            web.post("/v1/requests", admit_request),
            web.post("/v1/fanin/{operation}", fanin),
            web.post("/v1/retrieve", retrieve),
            web.post("/v1/retrievals/fence", fence_retrieval),
            web.post("/v1/consumers/renew", renew_consumer),
            web.post("/v1/consumers/release", release_consumer),
            web.post("/v1/entries", create_entry),
            web.post("/v1/entries/routes", selected_shard_routes),
            web.post("/v1/entries/commit", commit_entry),
            web.post("/v1/uploads/sync", sync_upload),
            web.post("/v1/select", select),
            web.post("/v1/deliveries", reserve_delivery),
            web.post("/v1/deliveries/start", start_delivery),
            web.post("/v1/deliveries/poll", poll_delivery),
            web.post("/v1/deliveries/ack", ack_delivery),
            web.post("/v1/deliveries/cancel", cancel_delivery),
            web.post("/v1/entries/cancel", cancel_entry),
            web.post("/v1/entries/release", release_entry),
            web.get("/health", health),
        ]
    )
    return app
