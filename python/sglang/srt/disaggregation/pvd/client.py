"""Async client for the V rank-0 coordinator control API."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import aiohttp

from sglang.srt.disaggregation.pvd.protocol import (
    FirstTokenMetadata,
    KVEntryKey,
    KVEntryManifest,
    RemoteRegionDescriptor,
)


class PVDControlPlaneError(RuntimeError):
    pass


class PVDCoordinatorClient:
    def __init__(
        self,
        base_url: str,
        *,
        session: Optional[aiohttp.ClientSession] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = session
        self._owns_session = session is None
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def _request(self, path: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        async with self._session.post(
            f"{self.base_url}{path}", json=dict(payload)
        ) as response:
            try:
                body = await response.json()
            except (aiohttp.ContentTypeError, ValueError) as exc:
                text = await response.text()
                raise PVDControlPlaneError(
                    f"V coordinator returned non-JSON status={response.status}: {text}"
                ) from exc
            if response.status >= 400:
                raise PVDControlPlaneError(
                    f"V coordinator returned {response.status}: {body.get('error', body)}"
                )
            return body

    async def create_entry(self, manifest: KVEntryManifest) -> Dict[str, Any]:
        return await self._request("/v1/entries", {"manifest": manifest.to_dict()})

    async def commit_shard(
        self,
        key: KVEntryKey,
        rank: int,
        received_bytes: int,
        first_token: Optional[FirstTokenMetadata] = None,
    ) -> Dict[str, Any]:
        payload = {
            "key": key.to_dict(),
            "rank": rank,
            "received_bytes": received_bytes,
        }
        if first_token is not None:
            payload["first_token"] = first_token.to_dict()
        return await self._request("/v1/entries/commit", payload)

    async def select(self, keys: Iterable[KVEntryKey]) -> Dict[str, Any]:
        return await self._request(
            "/v1/select", {"keys": [key.to_dict() for key in keys]}
        )

    async def reserve_delivery(
        self,
        key: KVEntryKey,
        delivery_id: str,
        destinations: Mapping[int, RemoteRegionDescriptor],
    ) -> Dict[str, Any]:
        return await self._request(
            "/v1/deliveries",
            {
                "key": key.to_dict(),
                "delivery_id": delivery_id,
                "destinations": {
                    str(rank): descriptor.to_dict()
                    for rank, descriptor in destinations.items()
                },
            },
        )

    async def start_delivery(self, delivery_id: str) -> Dict[str, Any]:
        return await self._request(
            "/v1/deliveries/start", {"delivery_id": delivery_id}
        )

    async def ack_delivery(self, delivery_id: str) -> Dict[str, Any]:
        return await self._request(
            "/v1/deliveries/ack", {"delivery_id": delivery_id}
        )

    async def cancel_delivery(
        self, delivery_id: str, reason: str
    ) -> Dict[str, Any]:
        return await self._request(
            "/v1/deliveries/cancel",
            {"delivery_id": delivery_id, "reason": reason},
        )

    async def release_entry(self, key: KVEntryKey) -> Dict[str, Any]:
        return await self._request(
            "/v1/entries/release", {"key": key.to_dict()}
        )

    async def cancel_entry(self, key: KVEntryKey, reason: str) -> Dict[str, Any]:
        return await self._request(
            "/v1/entries/cancel", {"key": key.to_dict(), "reason": reason}
        )

    async def close(self) -> None:
        if self._session is not None and self._owns_session:
            await self._session.close()
        self._session = None
