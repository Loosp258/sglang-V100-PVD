"""Rank-0 control coordinator for a sharded PVD vector worker group."""

from __future__ import annotations

import abc
import asyncio
import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

from sglang.srt.disaggregation.pvd.metrics import PVDMetrics
from sglang.srt.disaggregation.pvd.protocol import (
    PVD_GENERATION_METADATA_KEY,
    PVD_RECEIVER_EPOCH_METADATA_KEY,
    FirstTokenMetadata,
    KVEntryKey,
    KVEntryManifest,
    RemoteRegionDescriptor,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.request_state import (
    DELIVERY_TERMINAL_STATES,
    DeliveryState,
    EntryShardState,
    EntryState,
    transition,
)
from sglang.srt.disaggregation.pvd.retrieval import RetrievalRequest
from sglang.srt.disaggregation.pvd.selector import PassThroughSelector
from sglang.srt.disaggregation.pvd.sharding import (
    layout_from_destination,
    source_rank_and_head_offset,
)
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
    async def poll_delivery(self, key: KVEntryKey, delivery_id: str) -> Mapping: ...

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

    @abc.abstractmethod
    async def fence_delivery(self, identity: WriteIdentity) -> Mapping: ...


class LocalShardClient(ShardClient):
    def __init__(
        self, store: VectorKVStore, *, preflight: Optional[Mapping] = None
    ) -> None:
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

    async def poll_delivery(self, key: KVEntryKey, delivery_id: str) -> Mapping:
        result = await asyncio.to_thread(self.store.poll_delivery, key, delivery_id)
        return result.to_dict()

    async def ack_delivery(self, key: KVEntryKey, delivery_id: str) -> Mapping:
        return self.store.ack_delivery(key, delivery_id).to_dict()

    async def cancel_delivery(
        self, key: KVEntryKey, delivery_id: str, reason: str
    ) -> Mapping:
        result = await asyncio.to_thread(
            self.store.cancel_delivery, key, delivery_id, reason
        )
        return result.to_dict()

    async def cancel_entry(self, key: KVEntryKey, reason: str) -> None:
        await asyncio.to_thread(self.store.cancel_entry, key, reason)

    async def fence_delivery(self, identity: WriteIdentity) -> Mapping:
        if isinstance(self.store, VectorKVStore):
            return await asyncio.to_thread(self.store.fence_write, identity)
        # Non-store legacy/test adapters cannot manufacture lifecycle proof by
        # echoing an ID-only cancellation reply. Keep that bridge fail-closed.
        return {
            **identity.to_dict(),
            "fenced": False,
            "reason": "transport_terminal_unverified",
        }

    async def release_entry(self, key: KVEntryKey) -> None:
        await asyncio.to_thread(self.store.release_entry, key)

    async def health(self) -> Mapping:
        snapshot = await asyncio.to_thread(self.store.snapshot)
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

    consumer_leases: Dict[str, float] = field(default_factory=dict)

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
    source_shards: Dict[int, int]
    state: DeliveryState
    created_at: float
    deadline: float
    write_identities: Dict[int, WriteIdentity] = field(default_factory=dict)
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
            "source_shards": {
                str(rank): source_rank
                for rank, source_rank in self.source_shards.items()
            },
            "write_identities": {
                str(rank): identity.to_dict()
                for rank, identity in self.write_identities.items()
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
            raise ValueError(f"PVD requires V storage shard ranks [0, 1], got {ranks}")
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
        self.admissions = {}
        self._retrieval_locks = {}
        self._fenced_retrievals = set()

    async def fence_retrieval(
        self, delivery_id: str, identities: List[Mapping]
    ) -> Mapping:
        """Drain active writes and reject delayed/retried starts for this ID."""
        if not isinstance(delivery_id, str) or not delivery_id.strip():
            raise ValueError("delivery_id must be non-empty")
        if not isinstance(identities, list) or not identities:
            raise ValueError("write identities must be a non-empty list")
        try:
            supplied = [WriteIdentity.from_dict(value) for value in identities]
        except ValueError as exc:
            raise CoordinatorError(f"invalid write identity: {exc}") from exc
        supplied_ranks = [identity.shard_rank for identity in supplied]
        if len(set(supplied_ranks)) != len(supplied_ranks):
            raise CoordinatorError("write identity set has duplicate shard ranks")
        lock = self._retrieval_locks.setdefault(delivery_id, asyncio.Lock())
        async with lock:
            delivery = self.deliveries.get(delivery_id)
            if delivery is None:
                # Block a delayed reservation, but never infer NOT_SUBMITTED
                # merely from the absence of a coordinator record.
                self._fenced_retrievals.add(delivery_id)
                raise CoordinatorError(
                    "write identity cannot be confirmed for an unknown delivery"
                )
            ranks = sorted(delivery.destinations)
            expected = delivery.write_identities
            if sorted(expected) != ranks or sorted(supplied_ranks) != ranks:
                raise CoordinatorError("write identity set does not match destinations")
            supplied_by_rank = {identity.shard_rank: identity for identity in supplied}
            for rank in ranks:
                identity = expected[rank]
                if identity.key != delivery.entry_key:
                    raise CoordinatorError("stored write identity has a mismatched key")
                if identity.transfer_id != self._subdelivery_id(delivery_id, rank):
                    raise CoordinatorError(
                        "stored write identity has a mismatched transfer ID"
                    )
                try:
                    identity.validate_destination(delivery.destinations[rank])
                except ValueError as exc:
                    raise CoordinatorError(
                        f"stored write identity does not match destination: {exc}"
                    ) from exc
                if supplied_by_rank[rank] != identity:
                    raise CoordinatorError(
                        "supplied write identity does not match stored authorization"
                    )
            self._fenced_retrievals.add(delivery_id)
            replies = await asyncio.gather(
                *(
                    self.shards[delivery.source_shards[rank]].fence_delivery(
                        expected[rank]
                    )
                    for rank in ranks
                ),
                return_exceptions=True,
            )
            all_fenced = True
            for rank, reply in zip(ranks, replies):
                if isinstance(reply, BaseException):
                    raise CoordinatorError(f"V shard fence unconfirmed: {reply}")
                if (
                    not isinstance(reply, Mapping)
                    or type(reply.get("fenced")) is not bool
                ):
                    raise CoordinatorError(
                        "V shard returned an invalid identity fence acknowledgement"
                    )
                try:
                    reply_identity = WriteIdentity.from_dict(
                        {name: reply[name] for name in expected[rank].to_dict()}
                    )
                except (KeyError, ValueError, TypeError) as exc:
                    raise CoordinatorError(
                        "V shard returned an invalid identity fence acknowledgement"
                    ) from exc
                if reply_identity != expected[rank]:
                    raise CoordinatorError(
                        "V shard returned a mismatched write identity"
                    )
                all_fenced = all_fenced and reply["fenced"]
            if all_fenced:
                await self.cancel_delivery(delivery_id, "Decode fenced retrieval")
        return {
            "delivery_id": delivery_id,
            "identities": [expected[rank].to_dict() for rank in ranks],
            "fenced": all_fenced,
        }

    async def admit_request(self, request: Mapping) -> Mapping:
        """Register Router dispatch; precise allocation follows P's manifest."""
        fields = [
            request[name]
            for name in ("pvd_transfer_id", "pvd_delivery_id", "pvd_vector_group_id")
        ]
        fields = [v if isinstance(v, list) else [v] for v in fields]
        if not fields[0] or len({len(v) for v in fields}) != 1:
            raise ValueError(
                "PVD admission identity arrays must have equal nonzero length"
            )
        identities = list(zip(*fields))
        if any(
            not isinstance(v, str) or not v.strip() for row in identities for v in row
        ):
            raise ValueError("PVD admission identities must be non-empty strings")
        if len({row[0] for row in identities}) != len(identities):
            raise ValueError("duplicate transfer ID in PVD admission")
        async with self._lock:
            for transfer_id, delivery_id, group_id in identities:
                previous = self.admissions.get(transfer_id)
                if previous and previous[0] != (delivery_id, group_id):
                    raise CoordinatorError("conflicting Router admission identity")
            for transfer_id, delivery_id, group_id in identities:
                self.admissions[transfer_id] = (
                    (delivery_id, group_id),
                    time.monotonic() + self.entry_ttl_secs,
                )
        return {"accepted": [row[0] for row in identities]}

    async def renew_consumer(self, key: KVEntryKey, consumer_id: str) -> Mapping:
        if not isinstance(consumer_id, str) or not consumer_id.strip():
            raise ValueError("consumer ID must be non-empty")
        async with self._lock:
            entry = self.entries[key]
            if entry.state != EntryState.STORED:
                raise CoordinatorError("consumer requires a STORED Entry")
            duration = max(3.0, self.entry_ttl_secs)
            entry.consumer_leases[consumer_id] = time.monotonic() + duration
            return {"renew_after_seconds": duration / 3}

    async def release_consumer(self, key: KVEntryKey, consumer_id: str) -> Mapping:
        async with self._lock:
            entry = self.entries[key]
            entry.consumer_leases.pop(consumer_id, None)
            entry.expires_at = time.monotonic() + self.entry_ttl_secs
        return {"ok": True}

    async def retrieve(self, sequences: List[Mapping]) -> List[Mapping]:
        # Parse the whole batch before initiating any remote writes.
        requests = [RetrievalRequest.from_dict(value) for value in sequences]
        if len({r.delivery_id for r in requests}) != len(requests):
            raise ValueError("duplicate delivery ID in retrieval batch")

        async def one(request):
            result = {
                "sequence_id": request.sequence_id,
                "delivery_id": request.delivery_id,
                "selection": request.selection,
                "key": request.key.to_dict(),
            }
            try:
                if request.delivery_id in self._fenced_retrievals:
                    raise CoordinatorError("retrieval has been fenced")
                entry = self.entries[request.key]
                if entry.state != EntryState.STORED:
                    raise CoordinatorError("retrieval requires KV_READY/STORED Entry")
                await self.reserve_delivery(
                    key=request.key,
                    delivery_id=request.delivery_id,
                    destinations=request.destinations,
                )
                delivered = await self.start_delivery(request.delivery_id)
                result.update(
                    state=delivered.state.value,
                    write_identities={
                        str(rank): identity.to_dict()
                        for rank, identity in delivered.write_identities.items()
                    },
                    error=delivered.error,
                    token_ranges=[[0, entry.manifest.prompt_token_count]],
                )
            except Exception as exc:
                result.update(state="failed", error=str(exc))
            return result

        # Reserve/start have their own idempotency locks and each rechecks the
        # closed gate. Do not hold the fence lock over submission RPCs: fence
        # must be able to close a worker gate while native submission is slow.
        return await asyncio.gather(*(one(request) for request in requests))

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
        # Freeze caller-owned maps (including nested layout metadata) before
        # the first await. Retries must compare against the original descriptor.
        destinations = copy.deepcopy(destinations)
        async with self._lock:
            if delivery_id in self._fenced_retrievals:
                raise CoordinatorError("retrieval has been fenced")
            entry = self.entries.get(key)
            if entry is None:
                raise CoordinatorError("cannot reserve delivery for an unknown entry")
            destination_ranks = sorted(destinations)
            if not destination_ranks or destination_ranks != list(
                range(len(destination_ranks))
            ):
                raise CoordinatorError(
                    "delivery D destination ranks must be contiguous from rank 0"
                )
            source_shards: Dict[int, int] = {}
            for rank, destination in destinations.items():
                if destination.rank != rank:
                    raise CoordinatorError(
                        f"destination map key {rank} does not match descriptor rank "
                        f"{destination.rank}"
                    )
                if "pvd_layout" in destination.backend_metadata:
                    compute_layout = layout_from_destination(destination)
                    if compute_layout.tp_size != len(destinations):
                        raise CoordinatorError(
                            f"D layout TP={compute_layout.tp_size} does not match "
                            f"{len(destinations)} destinations"
                        )
                    source_rank, _ = source_rank_and_head_offset(
                        entry.manifest.layout, compute_layout, rank
                    )
                    source_shards[rank] = source_rank
                else:
                    if destination_ranks != [0, 1]:
                        raise CoordinatorError(
                            "heterogeneous delivery destinations require pvd_layout metadata"
                        )
                    source_shards[rank] = rank
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
                source_shards=source_shards,
                state=state,
                created_at=now,
                deadline=now + self.delivery_timeout_secs,
            )
            self.deliveries[delivery_id] = record
            entry.active_delivery_count += 1

        try:
            results = await asyncio.gather(
                *(
                    self.shards[source_shards[rank]].reserve_delivery(
                        key,
                        self._subdelivery_id(delivery_id, rank),
                        copy.deepcopy(destinations[rank]),
                    )
                    for rank in destination_ranks
                )
            )
            identity_values = [result.get("write_identity") for result in results]
            requires_identity = any(
                PVD_RECEIVER_EPOCH_METADATA_KEY in destination.backend_metadata
                or PVD_GENERATION_METADATA_KEY in destination.backend_metadata
                for destination in destinations.values()
            )
            if requires_identity or any(value is not None for value in identity_values):
                if any(value is None for value in identity_values):
                    raise CoordinatorError(
                        "V shards returned an incomplete write identity set"
                    )
                write_identities = {}
                for rank, result, value in zip(
                    destination_ranks, results, identity_values
                ):
                    try:
                        returned_destination = RemoteRegionDescriptor.from_dict(
                            result["destination"]
                        )
                        identity = WriteIdentity.from_dict(value)
                    except (KeyError, TypeError, ValueError) as exc:
                        raise CoordinatorError(
                            f"V shard {rank} returned an invalid write identity: {exc}"
                        ) from exc
                    if returned_destination != destinations[rank]:
                        raise CoordinatorError(
                            f"V shard {rank} substituted the authorized destination"
                        )
                    if identity.key != key:
                        raise CoordinatorError(
                            f"V shard {rank} returned a write identity with the wrong key"
                        )
                    if identity.transfer_id != self._subdelivery_id(delivery_id, rank):
                        raise CoordinatorError(
                            f"V shard {rank} returned a mismatched write transfer ID"
                        )
                    try:
                        identity.validate_destination(destinations[rank])
                    except ValueError as exc:
                        raise CoordinatorError(
                            f"V shard {rank} write identity does not match destination: {exc}"
                        ) from exc
                    write_identities[rank] = identity
                record.write_identities = write_identities
        except Exception as exc:
            await self.cancel_delivery(delivery_id, f"reserve failed: {exc}")
            raise
        async with self._lock:
            record.shard_states = {
                rank: DeliveryState(result["state"])
                for rank, result in zip(destination_ranks, results)
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
            if delivery_id in self._fenced_retrievals:
                raise CoordinatorError("retrieval has been fenced")
            delivery = self.deliveries[delivery_id]
            entry = self.entries[delivery.entry_key]
            if delivery.state == DeliveryState.DELIVERED:
                return delivery
            if entry.state != EntryState.STORED:
                return delivery
            if delivery.state == DeliveryState.WAITING_SOURCE:
                delivery.state = transition(delivery.state, DeliveryState.D_RESERVED)
            delivery.state = transition(delivery.state, DeliveryState.V_WRITING)

        destination_ranks = sorted(delivery.destinations)
        results = await asyncio.gather(
            *(
                self.shards[delivery.source_shards[rank]].start_delivery(
                    delivery.entry_key,
                    self._subdelivery_id(delivery.delivery_id, rank),
                )
                for rank in destination_ranks
            ),
            return_exceptions=True,
        )
        return await self._collect_delivery_results(
            delivery, destination_ranks, results
        )

    async def poll_delivery(self, delivery_id: str) -> DeliveryRecord:
        async with self._lock:
            delivery = self.deliveries[delivery_id]
            ranks = sorted(delivery.destinations)
        # Poll even logically cancelled deliveries: native resources may still
        # be draining. No coordinator business lock is held across shard RPCs.
        results = await asyncio.gather(
            *(
                self.shards[delivery.source_shards[rank]].poll_delivery(
                    delivery.entry_key, self._subdelivery_id(delivery_id, rank)
                )
                for rank in ranks
            ),
            return_exceptions=True,
        )
        return await self._collect_delivery_results(delivery, ranks, results)

    async def _collect_delivery_results(self, delivery, ranks, results):
        failures = []
        async with self._lock:
            for rank, result in zip(ranks, results):
                if isinstance(result, BaseException):
                    failures.append(f"rank {rank}: {result}")
                    continue
                try:
                    state = DeliveryState(result["state"])
                except (KeyError, ValueError, TypeError):
                    failures.append(f"rank {rank}: malformed delivery reply")
                    continue
                delivery.shard_states[rank] = state
                if (
                    state in DELIVERY_TERMINAL_STATES
                    and state != DeliveryState.RELEASED
                ):
                    failures.append(
                        f"rank {rank}: {result.get('error') or state.value}"
                    )
            if delivery.state in DELIVERY_TERMINAL_STATES:
                return delivery
            if failures:
                delivery.state = transition(delivery.state, DeliveryState.FAILED)
                delivery.error = "; ".join(failures)
                self.entries[delivery.entry_key].active_delivery_count -= 1
                self.metrics.increment("coordinator_delivery_failures")
            elif delivery.state == DeliveryState.V_WRITING and all(
                delivery.shard_states.get(rank) == DeliveryState.DELIVERED
                for rank in ranks
            ):
                delivery.state = transition(delivery.state, DeliveryState.DELIVERED)
                self.metrics.increment("coordinator_deliveries_completed")
        if failures:
            await asyncio.gather(
                *(
                    self.shards[delivery.source_shards[rank]].cancel_delivery(
                        delivery.entry_key,
                        self._subdelivery_id(delivery.delivery_id, rank),
                        delivery.error,
                    )
                    for rank in ranks
                ),
                return_exceptions=True,
            )
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
            *(
                self.shards[delivery.source_shards[rank]].ack_delivery(
                    key, self._subdelivery_id(delivery_id, rank)
                )
                for rank in sorted(delivery.destinations)
            )
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
                self.shards[delivery.source_shards[rank]].cancel_delivery(
                    key, self._subdelivery_id(delivery_id, rank), reason
                )
                for rank in sorted(delivery.destinations)
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

    @staticmethod
    def _subdelivery_id(delivery_id: str, destination_rank: int) -> str:
        return f"{delivery_id}:d{destination_rank}"

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
            if any(
                deadline > time.monotonic()
                for deadline in entry.consumer_leases.values()
            ):
                raise CoordinatorError("entry has active consumer leases")
            if entry.active_delivery_count:
                raise CoordinatorError(
                    f"entry has {entry.active_delivery_count} active deliveries"
                )
            if entry.state == EntryState.RELEASED:
                return entry
            entry.state = transition(entry.state, EntryState.RELEASING)
        await asyncio.gather(*(self.shards[rank].release_entry(key) for rank in (0, 1)))
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
            "pvd_framework_version": 3,
            "capabilities": [
                "request_admission",
                "full_prompt_retrieval",
                "consumer_leases",
                "retrieval_fencing",
            ],
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
            self.admissions = {
                key: value for key, value in self.admissions.items() if value[1] > now
            }
            for entry in self.entries.values():
                entry.consumer_leases = {
                    key: deadline
                    for key, deadline in entry.consumer_leases.items()
                    if deadline > now
                }
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
                and not entry.consumer_leases
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
