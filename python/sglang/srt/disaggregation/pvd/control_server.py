"""HTTP control plane for the rank-sharded PVD vector node.

The data plane never goes through HTTP.  These endpoints only exchange immutable
entry manifests, registered-memory descriptors, lifecycle transitions and
delivery acknowledgements.  P->V and V->D KV bytes are moved by TransferEngine.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Mapping, Optional

import aiohttp
from aiohttp import web

from sglang.srt.disaggregation.pvd.coordinator import (
    CoordinatorError,
    ShardClient,
    VectorCoordinator,
)
from sglang.srt.disaggregation.pvd.protocol import (
    FirstTokenMetadata,
    KVEntryKey,
    KVEntryManifest,
    ProtocolValidationError,
    RemoteRegionDescriptor,
)
from sglang.srt.disaggregation.pvd.request_state import InvalidStateTransition
from sglang.srt.disaggregation.pvd.vector_store import (
    EntryConflictError,
    EntryNotFoundError,
    ResourceExhaustedError,
    VectorKVStore,
)


def _json_error(message: str, status: int) -> web.Response:
    return web.json_response({"error": message}, status=status)


@web.middleware
async def pvd_error_middleware(request: web.Request, handler):
    try:
        return await handler(request)
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

    async def _request(self, method: str, path: str, payload=None) -> Mapping:
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

    async def create_entry(self, manifest: KVEntryManifest) -> Mapping:
        return await self._request(
            "POST", "/internal/v1/entries", {"manifest": manifest.to_dict()}
        )

    async def begin_p_write(self, key: KVEntryKey) -> Mapping:
        return await self._request(
            "POST", "/internal/v1/entries/begin", {"key": key.to_dict()}
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

    async def release_entry(self, key: KVEntryKey) -> None:
        await self._request(
            "POST", "/internal/v1/entries/release", {"key": key.to_dict()}
        )

    async def health(self) -> Mapping:
        return await self._request("GET", "/internal/health")

    async def close(self) -> None:
        if self._session is not None and self._owns_session:
            await self._session.close()
        self._session = None


def create_shard_app(
    store: VectorKVStore, *, preflight: Optional[Mapping[str, Any]] = None
) -> web.Application:
    app = web.Application(middlewares=[pvd_error_middleware])

    async def create_entry(request):
        data = await _payload(request)
        return web.json_response(
            store.create_entry(KVEntryManifest.from_dict(data["manifest"])).to_dict()
        )

    async def begin_entry(request):
        data = await _payload(request)
        return web.json_response(store.begin_p_write(_key(data)).to_dict())

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

    async def start_delivery(request):
        data = await _payload(request)
        result = await asyncio.to_thread(
            store.start_delivery, _key(data), str(data["delivery_id"])
        )
        return web.json_response(result.to_dict())

    async def ack_delivery(request):
        data = await _payload(request)
        return web.json_response(
            store.ack_delivery(_key(data), str(data["delivery_id"])).to_dict()
        )

    async def cancel_delivery(request):
        data = await _payload(request)
        return web.json_response(
            store.cancel_delivery(
                _key(data), str(data["delivery_id"]), str(data.get("reason", ""))
            ).to_dict()
        )

    async def cancel_entry(request):
        data = await _payload(request)
        store.cancel_entry(_key(data), str(data.get("reason", "")))
        return web.json_response({"ok": True})

    async def release_entry(request):
        data = await _payload(request)
        store.release_entry(_key(data))
        return web.json_response({"ok": True})

    async def health(_request):
        snapshot = store.snapshot()
        snapshot["preflight"] = dict(preflight or {})
        return web.json_response(snapshot)

    app.add_routes(
        [
            web.post("/internal/v1/entries", create_entry),
            web.post("/internal/v1/entries/begin", begin_entry),
            web.post("/internal/v1/entries/commit", commit_entry),
            web.post("/internal/v1/deliveries", reserve_delivery),
            web.post("/internal/v1/deliveries/start", start_delivery),
            web.post("/internal/v1/deliveries/ack", ack_delivery),
            web.post("/internal/v1/deliveries/cancel", cancel_delivery),
            web.post("/internal/v1/entries/cancel", cancel_entry),
            web.post("/internal/v1/entries/release", release_entry),
            web.get("/internal/health", health),
        ]
    )
    return app


def create_coordinator_app(coordinator: VectorCoordinator) -> web.Application:
    app = web.Application(middlewares=[pvd_error_middleware])

    async def create_entry(request):
        data = await _payload(request)
        result = await coordinator.create_entry(
            KVEntryManifest.from_dict(data["manifest"])
        )
        return web.json_response(result.to_dict())

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
        result = await coordinator.cancel_entry(
            _key(data), str(data.get("reason", ""))
        )
        return web.json_response(result.to_dict())

    async def release_entry(request):
        data = await _payload(request)
        result = await coordinator.release_entry(_key(data))
        return web.json_response(result.to_dict())

    async def health(_request):
        return web.json_response(await coordinator.health())

    app.add_routes(
        [
            web.post("/v1/entries", create_entry),
            web.post("/v1/entries/commit", commit_entry),
            web.post("/v1/select", select),
            web.post("/v1/deliveries", reserve_delivery),
            web.post("/v1/deliveries/start", start_delivery),
            web.post("/v1/deliveries/ack", ack_delivery),
            web.post("/v1/deliveries/cancel", cancel_delivery),
            web.post("/v1/entries/cancel", cancel_entry),
            web.post("/v1/entries/release", release_entry),
            web.get("/health", health),
        ]
    )
    return app
