"""Rank-0 control coordinator for a sharded PVD vector worker group."""

from __future__ import annotations

import abc
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

from sglang.srt.disaggregation.pvd.metrics import PVDMetrics
from sglang.srt.disaggregation.pvd.protocol import (
    FirstTokenMetadata,
    KVEntryKey,
    KVEntryManifest,
    RemoteRegionDescriptor,
)
from sglang.srt.disaggregation.pvd.request_state import (
    DELIVERY_TERMINAL_STATES,
    DeliveryState,
    EntryShardState,
    EntryState,
    transition,
)
from sglang.srt.disaggregation.pvd.selector import PassThroughSelector
from sglang.srt.disaggregation.pvd.vector_store import VectorKVStore


class CoordinatorError(RuntimeError):
    pass


class ShardClient(abc.ABC):
    rank: int

    @abc.abstractmethod
    async def create_entry(self, manifest: KVEntryManifest) -> Mapping: ...

    @abc.abstractmethod
    async def begin_p_write(self, key: KVEntryKey) -> Mapping: ...

    @abc.abstractmethod
    async def commit_p_write(self, key: KVEntryKey, received_bytes: int) -> Mapping: ...

    @abc.abstractmethod
    async def reserve_delivery(
        self,
        key: KVEntryKey,
        delivery_id: str,
        destination: RemoteRegionDescriptor,
    ) -> Mapping: ...

    @abc.abstractmethod
    async def start_delivery(self, key: KVEntryKey, delivery_id: str) -> Mapping: ...

    @abc.abstractmethod
    async def ack_delivery(self, key: KVEntryKey, delivery_id: str) -> Mapping: ...

    @abc.abstractmethod
    async def cancel_delivery(
        self, key: KVEntryKey, delivery_id: str, reason: str
    ) -> Mapping: ...

    @abc.abstractmethod
    async def cancel_entry(self, key: KVEntryKey, reason: str) -> None: ...

    @abc.abstractmethod
    async def release_entry(self, key: KVEntryKey) -> None: ...

    @abc.abstractmethod
    async def health(self) -> Mapping: ...


class LocalShardClient(ShardClient):
    def __init__(self, store: VectorKVStore, *, preflight: Optional[Mapping] = None) -> None:
        self.store = store
        self.rank = store.rank
        self.preflight = dict(preflight or {})

    async def create_entry(self, manifest: KVEntryManifest) -> Mapping:
        return self.store.create_entry(manifest).to_dict()

    async def begin_p_write(self, key: KVEntryKey) -> Mapping:
        return self.store.begin_p_write(key).to_dict()

    async def commit_p_write(self, key: KVEntryKey, received_bytes: int) -> Mapping:
        return self.store.commit_p_write(key, received_bytes).to_dict()

    async def reserve_delivery(
        self,
        key: KVEntryKey,
        delivery_id: str,
        destination: RemoteRegionDescriptor,
    ) -> Mapping:
        return self.store.reserve_delivery(key, delivery_id, destination).to_dict()

    async def start_delivery(self, key: KVEntryKey, delivery_id: str) -> Mapping:
        result = await asyncio.to_thread(self.store.start_delivery, key, delivery_id)
        return result.to_dict()

    async def ack_delivery(self, key: KVEntryKey, delivery_id: str) -> Mapping:
        return self.store.ack_delivery(key, delivery_id).to_dict()

    async def cancel_delivery(
        self, key: KVEntryKey, delivery_id: str, reason: str
    ) -> Mapping:
        return self.store.cancel_delivery(key, delivery_id, reason).to_dict()

    async def cancel_entry(self, key: KVEntryKey, reason: str) -> None:
        self.store.cancel_entry(key, reason)

    async def release_entry(self, key: KVEntryKey) -> None:
        self.store.release_entry(key)

    async def health(self) -> Mapping:
        snapshot = self.store.snapshot()
        snapshot["preflight"] = self.preflight
        return snapshot


@dataclass
class EntryRecord:
    manifest: KVEntryManifest
    state: EntryState
    created_at: float
    expires_at: float
    shard_states: Dict[int, EntryShardState] = field(default_factory=dict)
    target_regions: Dict[int, RemoteRegionDescriptor] = field(default_factory=dict)
    first_token: Optional[FirstTokenMetadata] = None
    active_delivery_count: int = 0
    error: Optional[str] = None

    def to_dict(self):
        return {
            "manifest": self.manifest.to_dict(),
            "state": self.state.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "shard_states": {
                str(rank): state.value for rank, state in self.shard_states.items()
            },
            "target_regions": {
                str(rank): descriptor.to_dict()
                for rank, descriptor in self.target_regions.items()
            },
            "first_token": self.first_token.to_dict() if self.first_token else None,
            "active_delivery_count": self.active_delivery_count,
            "error": self.error,
        }


@dataclass
class DeliveryRecord:
    delivery_id: str
    entry_key: KVEntryKey
    destinations: Dict[int, RemoteRegionDescriptor]
    state: DeliveryState
    created_at: float
    deadline: float
    shard_states: Dict[int, DeliveryState] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self):
        return {
            "delivery_id": self.delivery_id,
            "entry_key": self.entry_key.to_dict(),
            "destinations": {
                str(rank): descriptor.to_dict()
                for rank, descriptor in self.destinations.items()
            },
            "state": self.state.value,
            "created_at": self.created_at,
            "deadline": self.deadline,
            "shard_states": {
                str(rank): state.value for rank, state in self.shard_states.items()
            },
            "error": self.error,
        }


class VectorCoordinator:
    """Atomic group-level view over exactly two rank-local V stores."""

    def __init__(
        self,
        shard_clients: List[ShardClient],
        *,
        entry_ttl_secs: float = 300.0,
        delivery_timeout_secs: float = 300.0,
        metrics: Optional[PVDMetrics] = None,
    ) -> None:
        ranks = sorted(client.rank for client in shard_clients)
        if ranks != [0, 1]:
            raise ValueError(f"PVD v1 requires V shard ranks [0, 1], got {ranks}")
        self.shards = {client.rank: client for client in shard_clients}
        self.metrics = metrics or PVDMetrics()
        self.entry_ttl_secs = entry_ttl_secs
        self.delivery_timeout_secs = delivery_timeout_secs
        self.entries: Dict[KVEntryKey, EntryRecord] = {}
        self.deliveries: Dict[str, DeliveryRecord] = {}
        self._lock = asyncio.Lock()
        self._entry_create_locks: Dict[KVEntryKey, asyncio.Lock] = {}
        self._delivery_reserve_locks: Dict[str, asyncio.Lock] = {}
        self._delivery_start_locks: Dict[str, asyncio.Lock] = {}
        self.selector = PassThroughSelector(self.entry_state)

    def entry_state(self, key: KVEntryKey) -> Optional[EntryState]:
        entry = self.entries.get(key)
        return entry.state if entry is not None else None

    async def create_entry(self, manifest: KVEntryManifest) -> EntryRecord:
        async with self._lock:
            create_lock = self._entry_create_locks.setdefault(
                manifest.key, asyncio.Lock()
            )
        async with create_lock:
            return await self._create_entry_once(manifest)

    async def _create_entry_once(self, manifest: KVEntryManifest) -> EntryRecord:
        async with self._lock:
            existing = self.entries.get(manifest.key)
            if existing is not None:
                if existing.manifest != manifest:
                    raise CoordinatorError(
                        "entry key already exists with a different global manifest"
                    )
                return existing
            now = time.monotonic()
            record = EntryRecord(
                manifest=manifest,
                state=EntryState.CREATED,
                created_at=now,
                expires_at=now + self.entry_ttl_secs,
            )
            record.state = transition(record.state, EntryState.ALLOCATING)
            self.entries[manifest.key] = record

        try:
            created = await asyncio.gather(
                *(self.shards[rank].create_entry(manifest) for rank in (0, 1))
            )
            begun = await asyncio.gather(
                *(self.shards[rank].begin_p_write(manifest.key) for rank in (0, 1))
            )
        except Exception as exc:
            await self.cancel_entry(manifest.key, f"shard allocation failed: {exc}")
            raise

        async with self._lock:
            record = self.entries[manifest.key]
            record.shard_states = {
                rank: EntryShardState(begun[rank]["state"]) for rank in (0, 1)
            }
            record.target_regions = {
                rank: RemoteRegionDescriptor.from_dict(created[rank]["target_region"])
                for rank in (0, 1)
            }
            record.state = transition(record.state, EntryState.P_WRITING)
            self.metrics.increment("coordinator_entries_created")
            return record

    async def commit_shard(
        self,
        key: KVEntryKey,
        rank: int,
        received_bytes: int,
        first_token: Optional[FirstTokenMetadata] = None,
    ) -> EntryRecord:
        if rank not in self.shards:
            raise CoordinatorError(f"unknown V shard rank {rank}")
        if first_token is not None and rank != 0:
            raise CoordinatorError("first-token metadata must be committed by rank 0")
        result = await self.shards[rank].commit_p_write(key, received_bytes)
        async with self._lock:
            entry = self.entries[key]
            entry.shard_states[rank] = EntryShardState(result["state"])
            if first_token is not None:
                if entry.first_token is not None and entry.first_token != first_token:
                    entry.state = transition(entry.state, EntryState.FAILED)
                    entry.error = "first-token metadata mismatch across shard commits"
                    raise CoordinatorError(entry.error)
                entry.first_token = first_token
            if all(
                entry.shard_states.get(shard_rank) == EntryShardState.STORED
                for shard_rank in (0, 1)
            ):
                if entry.first_token is None:
                    entry.state = transition(entry.state, EntryState.FAILED)
                    entry.error = (
                        "rank-0 commit must include first-token metadata before STORED"
                    )
                    raise CoordinatorError(
                        "rank-0 commit must include first-token metadata before STORED"
                    )
                entry.state = transition(entry.state, EntryState.STORED)
                entry.expires_at = time.monotonic() + self.entry_ttl_secs
                self.metrics.increment("coordinator_entries_stored")
                for delivery in self.deliveries.values():
                    if (
                        delivery.entry_key == key
                        and delivery.state == DeliveryState.WAITING_SOURCE
                    ):
                        delivery.state = transition(
                            delivery.state, DeliveryState.D_RESERVED
                        )
            return entry

    async def select(self, keys: List[KVEntryKey]):
        async with self._lock:
            payloads = []
            for result in self.selector.select(keys):
                payload = result.to_dict()
                entry = self.entries.get(result.key)
                if entry is not None:
                    payload["entry"] = entry.to_dict()
                payloads.append(payload)
            return payloads

    async def reserve_delivery(
        self,
        *,
        key: KVEntryKey,
        delivery_id: str,
        destinations: Dict[int, RemoteRegionDescriptor],
    ) -> DeliveryRecord:
        async with self._lock:
            reserve_lock = self._delivery_reserve_locks.setdefault(
                delivery_id, asyncio.Lock()
            )
        async with reserve_lock:
            return await self._reserve_delivery_once(
                key=key,
                delivery_id=delivery_id,
                destinations=destinations,
            )

    async def _reserve_delivery_once(
        self,
        *,
        key: KVEntryKey,
        delivery_id: str,
        destinations: Dict[int, RemoteRegionDescriptor],
    ) -> DeliveryRecord:
        if sorted(destinations) != [0, 1]:
            raise CoordinatorError("delivery must provide D destinations for ranks 0 and 1")
        async with self._lock:
            entry = self.entries.get(key)
            if entry is None:
                raise CoordinatorError("cannot reserve delivery for an unknown entry")
            existing = self.deliveries.get(delivery_id)
            if existing is not None:
                if existing.entry_key != key or existing.destinations != destinations:
                    raise CoordinatorError(
                        "delivery id already exists with different parameters"
                    )
                return existing
            state = (
                DeliveryState.D_RESERVED
                if entry.state == EntryState.STORED
                else DeliveryState.WAITING_SOURCE
            )
            now = time.monotonic()
            record = DeliveryRecord(
                delivery_id=delivery_id,
                entry_key=key,
                destinations=destinations,
                state=state,
                created_at=now,
                deadline=now + self.delivery_timeout_secs,
            )
            self.deliveries[delivery_id] = record
            entry.active_delivery_count += 1

        try:
            results = await asyncio.gather(
                *(
                    self.shards[rank].reserve_delivery(
                        key, delivery_id, destinations[rank]
                    )
                    for rank in (0, 1)
                )
            )
        except Exception as exc:
            await self.cancel_delivery(delivery_id, f"reserve failed: {exc}")
            raise
        async with self._lock:
            record.shard_states = {
                rank: DeliveryState(results[rank]["state"]) for rank in (0, 1)
            }
            self.metrics.increment("coordinator_deliveries_created")
            return record

    async def start_delivery(self, delivery_id: str) -> DeliveryRecord:
        async with self._lock:
            start_lock = self._delivery_start_locks.setdefault(
                delivery_id, asyncio.Lock()
            )
        async with start_lock:
            return await self._start_delivery_once(delivery_id)

    async def _start_delivery_once(self, delivery_id: str) -> DeliveryRecord:
        async with self._lock:
            delivery = self.deliveries[delivery_id]
            entry = self.entries[delivery.entry_key]
            if delivery.state == DeliveryState.DELIVERED:
                return delivery
            if entry.state != EntryState.STORED:
                return delivery
            if delivery.state == DeliveryState.WAITING_SOURCE:
                delivery.state = transition(delivery.state, DeliveryState.D_RESERVED)
            delivery.state = transition(delivery.state, DeliveryState.V_WRITING)

        results = await asyncio.gather(
            *(
                self.shards[rank].start_delivery(
                    delivery.entry_key, delivery.delivery_id
                )
                for rank in (0, 1)
            ),
            return_exceptions=True,
        )
        async with self._lock:
            failures = []
            for rank, result in enumerate(results):
                if isinstance(result, Exception):
                    failures.append(f"rank {rank}: {result}")
                else:
                    state = DeliveryState(result["state"])
                    delivery.shard_states[rank] = state
                    if state == DeliveryState.FAILED:
                        failures.append(f"rank {rank}: {result.get('error')}")
            if failures:
                await asyncio.gather(
                    *(
                        self.shards[rank].cancel_delivery(
                            delivery.entry_key,
                            delivery.delivery_id,
                            "; ".join(failures),
                        )
                        for rank in (0, 1)
                    ),
                    return_exceptions=True,
                )
                delivery.state = transition(delivery.state, DeliveryState.FAILED)
                delivery.error = "; ".join(failures)
                self.entries[delivery.entry_key].active_delivery_count -= 1
                self.metrics.increment("coordinator_delivery_failures")
            elif all(
                delivery.shard_states.get(rank) == DeliveryState.DELIVERED
                for rank in (0, 1)
            ):
                delivery.state = transition(delivery.state, DeliveryState.DELIVERED)
                self.metrics.increment("coordinator_deliveries_completed")
            return delivery

    async def ack_delivery(self, delivery_id: str) -> DeliveryRecord:
        async with self._lock:
            delivery_lock = self._delivery_start_locks.setdefault(
                delivery_id, asyncio.Lock()
            )
        async with delivery_lock:
            return await self._ack_delivery_once(delivery_id)

    async def _ack_delivery_once(self, delivery_id: str) -> DeliveryRecord:
        async with self._lock:
            delivery = self.deliveries[delivery_id]
            if delivery.state == DeliveryState.RELEASED:
                return delivery
            key = delivery.entry_key
        await asyncio.gather(
            *(self.shards[rank].ack_delivery(key, delivery_id) for rank in (0, 1))
        )
        async with self._lock:
            delivery.state = transition(delivery.state, DeliveryState.ACKED)
            delivery.state = transition(delivery.state, DeliveryState.RELEASED)
            self.entries[key].active_delivery_count -= 1
            self.metrics.increment("coordinator_deliveries_acked")
            return delivery

    async def cancel_delivery(self, delivery_id: str, reason: str) -> DeliveryRecord:
        async with self._lock:
            delivery_lock = self._delivery_start_locks.setdefault(
                delivery_id, asyncio.Lock()
            )
        async with delivery_lock:
            return await self._cancel_delivery_once(delivery_id, reason)

    async def _cancel_delivery_once(
        self, delivery_id: str, reason: str
    ) -> DeliveryRecord:
        async with self._lock:
            delivery = self.deliveries[delivery_id]
            if delivery.state in DELIVERY_TERMINAL_STATES:
                return delivery
            key = delivery.entry_key
        await asyncio.gather(
            *(
                self.shards[rank].cancel_delivery(key, delivery_id, reason)
                for rank in (0, 1)
            ),
            return_exceptions=True,
        )
        async with self._lock:
            if delivery.state not in DELIVERY_TERMINAL_STATES:
                delivery.state = transition(delivery.state, DeliveryState.CANCELLED)
                delivery.error = reason
                self.entries[key].active_delivery_count -= 1
                self.metrics.increment("coordinator_deliveries_cancelled")
            return delivery

    async def cancel_entry(self, key: KVEntryKey, reason: str) -> EntryRecord:
        async with self._lock:
            entry = self.entries[key]
            delivery_ids = [
                delivery_id
                for delivery_id, delivery in self.deliveries.items()
                if delivery.entry_key == key
                and delivery.state not in DELIVERY_TERMINAL_STATES
            ]
        for delivery_id in delivery_ids:
            await self.cancel_delivery(delivery_id, reason)
        await asyncio.gather(
            *(self.shards[rank].cancel_entry(key, reason) for rank in (0, 1)),
            return_exceptions=True,
        )
        async with self._lock:
            if entry.state not in (
                EntryState.RELEASED,
                EntryState.FAILED,
                EntryState.CANCELLED,
                EntryState.EXPIRED,
            ):
                entry.state = transition(entry.state, EntryState.CANCELLED)
            entry.error = reason
            self.metrics.increment("coordinator_entries_cancelled")
            return entry

    async def release_entry(self, key: KVEntryKey) -> EntryRecord:
        async with self._lock:
            entry = self.entries[key]
            if entry.active_delivery_count:
                raise CoordinatorError(
                    f"entry has {entry.active_delivery_count} active deliveries"
                )
            if entry.state == EntryState.RELEASED:
                return entry
            entry.state = transition(entry.state, EntryState.RELEASING)
        await asyncio.gather(
            *(self.shards[rank].release_entry(key) for rank in (0, 1))
        )
        async with self._lock:
            entry.state = transition(entry.state, EntryState.RELEASED)
            self.metrics.increment("coordinator_entries_released")
            return entry

    async def health(self):
        shard_health = await asyncio.gather(
            *(self.shards[rank].health() for rank in (0, 1)),
            return_exceptions=True,
        )
        return {
            "healthy": all(not isinstance(item, Exception) for item in shard_health),
            "role": "pvd-vector-coordinator",
            "world_size": 2,
            "shards": [
                {"error": str(item)} if isinstance(item, Exception) else item
                for item in shard_health
            ],
            "metrics": self.metrics.snapshot(),
        }

    async def reap_expired(self, now: Optional[float] = None) -> Dict[str, int]:
        """Expire coordinator records first, then release both V shards atomically."""
        now = time.monotonic() if now is None else now
        async with self._lock:
            delivery_ids = [
                delivery_id
                for delivery_id, delivery in self.deliveries.items()
                if delivery.state not in DELIVERY_TERMINAL_STATES
                and delivery.deadline <= now
            ]
        for delivery_id in delivery_ids:
            await self.cancel_delivery(delivery_id, "delivery timeout")

        async with self._lock:
            releasable_keys = [
                key
                for key, entry in self.entries.items()
                if entry.active_delivery_count == 0
                and entry.expires_at <= now
                and entry.state == EntryState.STORED
            ]
            cancellable_keys = [
                key
                for key, entry in self.entries.items()
                if entry.active_delivery_count == 0
                and entry.expires_at <= now
                and entry.state
                not in (
                    EntryState.STORED,
                    EntryState.RELEASED,
                    EntryState.FAILED,
                    EntryState.CANCELLED,
                    EntryState.EXPIRED,
                )
            ]
        released = 0
        for key in releasable_keys:
            try:
                await self.release_entry(key)
                released += 1
            except Exception:
                self.metrics.increment("coordinator_reap_failures")
        for key in cancellable_keys:
            try:
                await self.cancel_entry(key, "entry creation timeout")
                released += 1
            except Exception:
                self.metrics.increment("coordinator_reap_failures")
        if delivery_ids:
            self.metrics.increment("coordinator_deliveries_expired", len(delivery_ids))
        if released:
            self.metrics.increment("coordinator_entries_expired", released)
        return {"entries": released, "deliveries": len(delivery_ids)}
