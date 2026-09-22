"""Async client for the V rank-0 coordinator control API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import aiohttp
from sglang.srt.disaggregation.pvd.protocol import (
    FirstTokenMetadata,
    KVEntryKey,
    KVEntryManifest,
    RemoteRegionDescriptor,
    WriteIdentity,
    normalize_uploader_epochs,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransportState


class PVDControlPlaneError(RuntimeError):
    pass


@dataclass(frozen=True)
class PVDSelectedShardRoute:
    rank: int
    url: str
    sender_epoch: str
    rail: str


@dataclass(frozen=True)
class PVDSelectedShardRoutes:
    manifest: KVEntryManifest
    shards: tuple[PVDSelectedShardRoute, ...]


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

    async def create_entry(
        self,
        manifest: KVEntryManifest,
        *,
        uploader_epoch: Optional[str] = None,
        uploader_epochs: Optional[Mapping[int, str]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"manifest": manifest.to_dict()}
        epochs = normalize_uploader_epochs(
            (s.rank for s in manifest.shards),
            uploader_epoch=uploader_epoch,
            uploader_epochs=uploader_epochs,
        )
        if uploader_epoch is not None:
            payload["uploader_epoch"] = uploader_epoch
        elif uploader_epochs is not None:
            payload["uploader_epochs"] = {
                str(rank): epoch for rank, epoch in epochs.items()
            }
        return await self._request("/v1/entries", payload)

    async def selected_shard_routes(self, key: KVEntryKey) -> PVDSelectedShardRoutes:
        """Discover the selected V group's shard URLs and current epochs."""
        if not isinstance(key, KVEntryKey):
            raise ValueError("explicit selected Entry key required")
        reply = await self._request("/v1/entries/routes", {"key": key.to_dict()})
        try:
            manifest = KVEntryManifest.from_dict(reply["manifest"])
            sources = reply["shards"]
            if (
                manifest.key != key
                or manifest.layout.tp_size != 2
                or not isinstance(sources, list)
                or len(sources) != 2
            ):
                raise ValueError("selected Entry or source count changed")
            routes = []
            for source in sources:
                rank = source["rank"]
                url = source["url"]
                epoch = source["sender_epoch"]
                rail = source["rail"]
                if (
                    type(rank) is not int
                    or rank not in (0, 1)
                    or not isinstance(url, str)
                    or not url.startswith(("http://", "https://"))
                    or not isinstance(epoch, str)
                    or not epoch.strip()
                    or rail != manifest.shard(rank).rail
                ):
                    raise ValueError("invalid selected shard route")
                routes.append(PVDSelectedShardRoute(rank, url.rstrip("/"), epoch, rail))
            if [route.rank for route in routes] != [0, 1] or (
                routes[0].url == routes[1].url
            ):
                raise ValueError("shard ranks or URLs are not distinct and ordered")
        except (KeyError, TypeError, ValueError) as exc:
            raise PVDControlPlaneError(f"invalid selected shard routes: {exc}") from exc
        return PVDSelectedShardRoutes(manifest, tuple(routes))

    async def sync_upload(
        self,
        identity: WriteIdentity,
        state: TransportState,
        closed: bool,
    ) -> Dict[str, Any]:
        """Report this upload's transport state and collect V's close request.

        The reply is validated here as well as on V: a reply that does not echo
        the exact identity, or that omits typed flags, is not allowed to look
        like a terminal acknowledgement to the caller.
        """
        if not isinstance(identity, WriteIdentity):
            raise ValueError("identity must be a WriteIdentity")
        if not isinstance(state, TransportState):
            raise ValueError("state must be a TransportState")
        if type(closed) is not bool:
            raise ValueError("closed must be a boolean")
        reply = await self._request(
            "/v1/uploads/sync",
            {
                "identity": identity.to_dict(),
                "state": state.value,
                "closed": closed,
            },
        )
        try:
            if not isinstance(reply, Mapping):
                raise ValueError("reply must be an object")
            if WriteIdentity.from_dict(reply["identity"]) != identity:
                raise ValueError("reply identity does not match")
            if (
                type(reply.get("close_requested")) is not bool
                or type(reply.get("terminal_ack")) is not bool
            ):
                raise ValueError("reply is missing typed lifecycle flags")
        except (KeyError, ValueError) as exc:
            raise PVDControlPlaneError(f"invalid upload sync reply: {exc}") from exc
        return reply

    async def admit_request(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        return await self._request("/v1/requests", request)

    async def retrieve(self, sequences) -> Dict[str, Any]:
        return await self._request("/v1/retrieve", {"sequences": sequences})

    async def fence_retrieval(
        self, delivery_id: str, identities: list[dict]
    ) -> Dict[str, Any]:
        if not isinstance(delivery_id, str) or not delivery_id.strip():
            raise ValueError("delivery_id must be non-empty")
        expected = self._fence_identities(identities)
        if any(
            identity.transfer_id != f"{delivery_id}:d{rank}"
            for rank, identity in expected.items()
        ):
            raise ValueError("write identity does not belong to delivery")
        reply = await self._request(
            "/v1/retrievals/fence",
            {
                "delivery_id": delivery_id,
                "identities": [identity.to_dict() for identity in expected.values()],
            },
        )
        try:
            if (
                not isinstance(reply, Mapping)
                or reply.get("delivery_id") != delivery_id
                or type(reply.get("fenced")) is not bool
                or self._fence_identities(reply.get("identities")) != expected
            ):
                raise ValueError("reply does not match expected write identities")
        except ValueError as exc:
            raise PVDControlPlaneError(f"invalid retrieval fence reply: {exc}") from exc
        return reply

    @staticmethod
    def _fence_identities(values) -> Dict[int, WriteIdentity]:
        if not isinstance(values, list) or not values:
            raise ValueError("fence identities must be a non-empty list")
        identities = [WriteIdentity.from_dict(value) for value in values]
        by_rank = {identity.shard_rank: identity for identity in identities}
        if len(by_rank) != len(identities):
            raise ValueError("fence identities have duplicate ranks")
        return by_rank

    async def renew_consumer(self, key: KVEntryKey, consumer_id: str) -> Dict[str, Any]:
        return await self._request(
            "/v1/consumers/renew", {"key": key.to_dict(), "consumer_id": consumer_id}
        )

    async def release_consumer(
        self, key: KVEntryKey, consumer_id: str
    ) -> Dict[str, Any]:
        return await self._request(
            "/v1/consumers/release", {"key": key.to_dict(), "consumer_id": consumer_id}
        )

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
        return await self._request("/v1/deliveries/start", {"delivery_id": delivery_id})

    async def poll_delivery(self, delivery_id: str) -> Dict[str, Any]:
        return await self._request("/v1/deliveries/poll", {"delivery_id": delivery_id})

    async def ack_delivery(self, delivery_id: str) -> Dict[str, Any]:
        return await self._request("/v1/deliveries/ack", {"delivery_id": delivery_id})

    async def cancel_delivery(self, delivery_id: str, reason: str) -> Dict[str, Any]:
        return await self._request(
            "/v1/deliveries/cancel",
            {"delivery_id": delivery_id, "reason": reason},
        )

    async def release_entry(self, key: KVEntryKey) -> Dict[str, Any]:
        return await self._request("/v1/entries/release", {"key": key.to_dict()})

    async def cancel_entry(self, key: KVEntryKey, reason: str) -> Dict[str, Any]:
        return await self._request(
            "/v1/entries/cancel", {"key": key.to_dict(), "reason": reason}
        )

    async def close(self) -> None:
        if self._session is not None and self._owns_session:
            await self._session.close()
        self._session = None
