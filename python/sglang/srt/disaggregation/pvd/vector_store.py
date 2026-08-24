"""Rank-local GPU KV page store used by the PVD vector worker group."""

from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.disaggregation.pvd.metrics import PVDMetrics
from sglang.srt.disaggregation.pvd.protocol import (
    KVEntryKey,
    KVEntryManifest,
    KVShardManifest,
    RemoteRegionDescriptor,
)
from sglang.srt.disaggregation.pvd.request_state import (
    DELIVERY_TERMINAL_STATES,
    DeliveryState,
    EntryShardState,
    transition,
)
from sglang.srt.disaggregation.pvd.transfer_engine import (
    MemorySlice,
    RegisteredMemory,
    TransferEngine,
    TransferHandle,
    TransferStatus,
    descriptor_with_slice,
)


class ResourceExhaustedError(RuntimeError):
    pass


class EntryNotFoundError(KeyError):
    pass


class EntryConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class PageAllocation:
    start_page: int
    page_count: int

    @property
    def page_indices(self) -> List[int]:
        return list(range(self.start_page, self.start_page + self.page_count))


class ContiguousPageAllocator:
    """Thread-safe first-fit allocator with deterministic range coalescing."""

    def __init__(self, total_pages: int) -> None:
        if total_pages <= 0:
            raise ValueError("total_pages must be positive")
        self.total_pages = total_pages
        self._free_ranges: List[Tuple[int, int]] = [(0, total_pages)]
        self._allocated_pages = 0
        self._lock = threading.Lock()

    @property
    def available_pages(self) -> int:
        with self._lock:
            return self.total_pages - self._allocated_pages

    @property
    def allocated_pages(self) -> int:
        with self._lock:
            return self._allocated_pages

    def allocate(self, page_count: int) -> PageAllocation:
        if page_count <= 0:
            raise ValueError("page_count must be positive")
        with self._lock:
            for index, (start, count) in enumerate(self._free_ranges):
                if count < page_count:
                    continue
                allocation = PageAllocation(start, page_count)
                if count == page_count:
                    self._free_ranges.pop(index)
                else:
                    self._free_ranges[index] = (start + page_count, count - page_count)
                self._allocated_pages += page_count
                return allocation
        raise ResourceExhaustedError(
            f"vector GPU pool has no contiguous run of {page_count} pages"
        )

    def free(self, allocation: PageAllocation) -> None:
        with self._lock:
            start = allocation.start_page
            count = allocation.page_count
            if start < 0 or start + count > self.total_pages:
                raise ValueError("allocation is outside this pool")
            self._free_ranges.append((start, count))
            self._free_ranges.sort()
            merged: List[Tuple[int, int]] = []
            for range_start, range_count in self._free_ranges:
                if merged and merged[-1][0] + merged[-1][1] == range_start:
                    prior_start, prior_count = merged[-1]
                    merged[-1] = (prior_start, prior_count + range_count)
                else:
                    merged.append((range_start, range_count))
            self._free_ranges = merged
            self._allocated_pages -= count
            if self._allocated_pages < 0:
                raise RuntimeError("page allocator double free detected")


@dataclass
class DeliveryShardRecord:
    delivery_id: str
    entry_key: KVEntryKey
    destination: RemoteRegionDescriptor
    state: DeliveryState
    created_at: float
    deadline: float
    transfer_handle: Optional[TransferHandle] = None
    error: Optional[str] = None

    def to_dict(self):
        return {
            "delivery_id": self.delivery_id,
            "entry_key": self.entry_key.to_dict(),
            "destination": self.destination.to_dict(),
            "state": self.state.value,
            "created_at": self.created_at,
            "deadline": self.deadline,
            "error": self.error,
        }


@dataclass
class EntryShardRecord:
    key: KVEntryKey
    layout_fingerprint: str
    manifest: KVShardManifest
    allocation: PageAllocation
    target_region: RemoteRegionDescriptor
    state: EntryShardState
    created_at: float
    expires_at: float
    stored_at: Optional[float] = None
    received_bytes: int = 0
    active_delivery_count: int = 0
    resources_released: bool = False
    error: Optional[str] = None
    deliveries: Dict[str, DeliveryShardRecord] = field(default_factory=dict)

    def to_dict(self):
        return {
            "key": self.key.to_dict(),
            "layout_fingerprint": self.layout_fingerprint,
            "manifest": self.manifest.to_dict(),
            "page_indices": self.allocation.page_indices,
            "target_region": self.target_region.to_dict(),
            "state": self.state.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "stored_at": self.stored_at,
            "received_bytes": self.received_bytes,
            "active_delivery_count": self.active_delivery_count,
            "resources_released": self.resources_released,
            "error": self.error,
        }


class VectorKVStore:
    """One V rank's registered GPU pool and immutable prompt-KV entries."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        rail: str,
        device: str,
        total_pages: int,
        page_bytes: int,
        endpoint: str,
        transfer_engine: TransferEngine,
        entry_ttl_secs: float = 300.0,
        delivery_timeout_secs: float = 300.0,
        allow_cpu_for_tests: bool = False,
        metrics: Optional[PVDMetrics] = None,
    ) -> None:
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank {rank} is outside world size {world_size}")
        if world_size != 2:
            raise ValueError("PVD v1 requires exactly two V ranks")
        if page_bytes <= 0:
            raise ValueError("page_bytes must be positive")
        if device == "cpu" and not allow_cpu_for_tests:
            raise RuntimeError("PVD V storage cannot silently fall back to CPU")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device} requested but CUDA is unavailable")

        self.rank = rank
        self.world_size = world_size
        self.rail = rail
        self.device = device
        self.page_bytes = page_bytes
        self.endpoint = endpoint
        self.transfer_engine = transfer_engine
        self.entry_ttl_secs = entry_ttl_secs
        self.delivery_timeout_secs = delivery_timeout_secs
        self.metrics = metrics or PVDMetrics()
        self.allocator = ContiguousPageAllocator(total_pages)
        self.pool = torch.empty(total_pages * page_bytes, dtype=torch.uint8, device=device)
        self.registration: RegisteredMemory = transfer_engine.register_memory(
            self.pool,
            endpoint=endpoint,
            rank=rank,
            rail=rail,
            metadata={"role": "vector", "page_bytes": page_bytes},
        )
        self.entries: Dict[KVEntryKey, EntryShardRecord] = {}
        self._lock = threading.RLock()
        self._refresh_metrics()

    def _refresh_metrics(self) -> None:
        self.metrics.set_gauge("vector_allocated_pages", self.allocator.allocated_pages)
        self.metrics.set_gauge("vector_available_pages", self.allocator.available_pages)
        self.metrics.set_gauge("vector_entries", len(self.entries))

    def _entry(self, key: KVEntryKey) -> EntryShardRecord:
        entry = self.entries.get(key)
        if entry is None:
            raise EntryNotFoundError(key)
        return entry

    def create_entry(self, manifest: KVEntryManifest) -> EntryShardRecord:
        shard = manifest.shard(self.rank)
        if manifest.layout.tp_size != self.world_size:
            raise EntryConflictError(
                f"layout TP={manifest.layout.tp_size} does not match V world size={self.world_size}"
            )
        if shard.rail != self.rail:
            raise EntryConflictError(
                f"rank {self.rank} requires rail {self.rail}, manifest requested {shard.rail}"
            )
        capacity = shard.page_count * self.page_bytes
        if shard.expected_bytes > capacity:
            raise EntryConflictError(
                f"expected KV bytes {shard.expected_bytes} exceed allocated "
                f"page capacity {capacity}"
            )

        with self._lock:
            existing = self.entries.get(manifest.key)
            if existing is not None:
                if (
                    existing.layout_fingerprint != manifest.layout.fingerprint
                    or existing.manifest != shard
                ):
                    raise EntryConflictError("entry key already exists with a different manifest")
                return existing

            allocation = self.allocator.allocate(shard.page_count)
            offset = allocation.start_page * self.page_bytes
            target_region = descriptor_with_slice(
                self.registration, offset=offset, length=shard.expected_bytes
            )
            now = time.monotonic()
            record = EntryShardRecord(
                key=manifest.key,
                layout_fingerprint=manifest.layout.fingerprint,
                manifest=shard,
                allocation=allocation,
                target_region=target_region,
                state=EntryShardState.ALLOCATED,
                created_at=now,
                expires_at=now + self.entry_ttl_secs,
            )
            self.entries[manifest.key] = record
            self.metrics.increment("vector_entries_created")
            self._refresh_metrics()
            return record

    def begin_p_write(self, key: KVEntryKey) -> EntryShardRecord:
        with self._lock:
            entry = self._entry(key)
            entry.state = transition(entry.state, EntryShardState.P_WRITING)
            return entry

    def commit_p_write(self, key: KVEntryKey, received_bytes: int) -> EntryShardRecord:
        with self._lock:
            entry = self._entry(key)
            if entry.state == EntryShardState.STORED:
                if entry.received_bytes != received_bytes:
                    raise EntryConflictError("idempotent commit has a different byte count")
                return entry
            if received_bytes != entry.manifest.expected_bytes:
                self._fail_entry_locked(
                    entry,
                    f"received {received_bytes} KV bytes, expected {entry.manifest.expected_bytes}",
                )
                raise EntryConflictError(entry.error)
            entry.state = transition(entry.state, EntryShardState.STORED)
            entry.received_bytes = received_bytes
            entry.stored_at = time.monotonic()
            entry.expires_at = entry.stored_at + self.entry_ttl_secs
            for delivery in entry.deliveries.values():
                if delivery.state == DeliveryState.WAITING_SOURCE:
                    delivery.state = transition(delivery.state, DeliveryState.D_RESERVED)
            self.metrics.increment("vector_p_to_v_bytes", received_bytes)
            self.metrics.increment("vector_entries_stored")
            return entry

    def reserve_delivery(
        self,
        key: KVEntryKey,
        delivery_id: str,
        destination: RemoteRegionDescriptor,
    ) -> DeliveryShardRecord:
        if destination.rank != self.rank:
            raise EntryConflictError(
                f"V rank {self.rank} cannot deliver to D rank {destination.rank}"
            )
        if destination.rail != self.rail:
            raise EntryConflictError(
                f"rank {self.rank} requires rail {self.rail}, destination uses {destination.rail}"
            )
        with self._lock:
            entry = self._entry(key)
            if entry.resources_released:
                raise EntryConflictError("entry resources have already been released")
            existing = entry.deliveries.get(delivery_id)
            if existing is not None:
                if existing.destination != destination:
                    raise EntryConflictError(
                        "delivery id already exists with a different destination"
                    )
                return existing
            now = time.monotonic()
            state = (
                DeliveryState.D_RESERVED
                if entry.state == EntryShardState.STORED
                else DeliveryState.WAITING_SOURCE
            )
            delivery = DeliveryShardRecord(
                delivery_id=delivery_id,
                entry_key=key,
                destination=destination,
                state=state,
                created_at=now,
                deadline=now + self.delivery_timeout_secs,
            )
            entry.deliveries[delivery_id] = delivery
            entry.active_delivery_count += 1
            self.metrics.increment("vector_deliveries_created")
            return delivery

    def start_delivery(self, key: KVEntryKey, delivery_id: str) -> DeliveryShardRecord:
        with self._lock:
            entry = self._entry(key)
            delivery = entry.deliveries[delivery_id]
            if delivery.state == DeliveryState.DELIVERED:
                return delivery
            if delivery.state == DeliveryState.WAITING_SOURCE:
                if entry.state != EntryShardState.STORED:
                    return delivery
                delivery.state = transition(delivery.state, DeliveryState.D_RESERVED)
            delivery.state = transition(delivery.state, DeliveryState.V_WRITING)

            local_offset = entry.allocation.start_page * self.page_bytes
            local = MemorySlice(
                registration=self.registration,
                offset=local_offset,
                length=entry.manifest.expected_bytes,
            )
            handle = self.transfer_engine.submit_put(
                local, delivery.destination, remote_offset=0
            )
            delivery.transfer_handle = handle
            status = self.transfer_engine.poll(handle)
            if status == TransferStatus.SUCCESS:
                delivery.state = transition(delivery.state, DeliveryState.DELIVERED)
                self.metrics.increment("vector_v_to_d_bytes", handle.transferred_bytes)
                self.metrics.increment("vector_deliveries_completed")
            elif status in (TransferStatus.FAILED, TransferStatus.CANCELLED):
                delivery.state = transition(delivery.state, DeliveryState.FAILED)
                delivery.error = handle.error or status.value
                entry.active_delivery_count -= 1
                self.metrics.increment("vector_delivery_failures")
            return delivery

    def poll_delivery(self, key: KVEntryKey, delivery_id: str) -> DeliveryShardRecord:
        with self._lock:
            entry = self._entry(key)
            delivery = entry.deliveries[delivery_id]
            handle = delivery.transfer_handle
            if delivery.state != DeliveryState.V_WRITING or handle is None:
                return delivery
            status = self.transfer_engine.poll(handle)
            if status == TransferStatus.SUCCESS:
                delivery.state = transition(delivery.state, DeliveryState.DELIVERED)
                self.metrics.increment("vector_v_to_d_bytes", handle.transferred_bytes)
                self.metrics.increment("vector_deliveries_completed")
            elif status in (TransferStatus.FAILED, TransferStatus.CANCELLED):
                delivery.state = transition(delivery.state, DeliveryState.FAILED)
                delivery.error = handle.error or status.value
                entry.active_delivery_count -= 1
                self.metrics.increment("vector_delivery_failures")
            return delivery

    def ack_delivery(self, key: KVEntryKey, delivery_id: str) -> DeliveryShardRecord:
        with self._lock:
            entry = self._entry(key)
            delivery = entry.deliveries[delivery_id]
            if delivery.state == DeliveryState.RELEASED:
                return delivery
            delivery.state = transition(delivery.state, DeliveryState.ACKED)
            delivery.state = transition(delivery.state, DeliveryState.RELEASED)
            entry.active_delivery_count -= 1
            self.metrics.increment("vector_deliveries_acked")
            return delivery

    def cancel_delivery(
        self, key: KVEntryKey, delivery_id: str, reason: str
    ) -> DeliveryShardRecord:
        with self._lock:
            entry = self._entry(key)
            delivery = entry.deliveries[delivery_id]
            if delivery.state in DELIVERY_TERMINAL_STATES:
                return delivery
            if delivery.transfer_handle is not None:
                self.transfer_engine.abort(delivery.transfer_handle)
            delivery.state = transition(delivery.state, DeliveryState.CANCELLED)
            delivery.error = reason
            entry.active_delivery_count -= 1
            self.metrics.increment("vector_deliveries_cancelled")
            return delivery

    def release_entry(self, key: KVEntryKey) -> None:
        with self._lock:
            entry = self._entry(key)
            if entry.active_delivery_count:
                raise EntryConflictError(
                    f"entry has {entry.active_delivery_count} active deliveries"
                )
            if entry.resources_released:
                return
            if entry.state == EntryShardState.STORED:
                entry.state = transition(entry.state, EntryShardState.RELEASING)
                self._release_resources_locked(entry)
                entry.state = transition(entry.state, EntryShardState.RELEASED)
            else:
                self._release_resources_locked(entry)
            self.metrics.increment("vector_entries_released")

    def cancel_entry(self, key: KVEntryKey, reason: str) -> None:
        with self._lock:
            entry = self._entry(key)
            for delivery_id in list(entry.deliveries):
                delivery = entry.deliveries[delivery_id]
                if delivery.state not in DELIVERY_TERMINAL_STATES:
                    self.cancel_delivery(key, delivery_id, reason)
            if entry.state not in (
                EntryShardState.RELEASED,
                EntryShardState.FAILED,
                EntryShardState.CANCELLED,
                EntryShardState.EXPIRED,
            ):
                entry.state = transition(entry.state, EntryShardState.CANCELLED)
            entry.error = reason
            self._release_resources_locked(entry)
            self.metrics.increment("vector_entries_cancelled")

    def _fail_entry_locked(self, entry: EntryShardRecord, reason: str) -> None:
        if entry.state not in (
            EntryShardState.RELEASED,
            EntryShardState.FAILED,
            EntryShardState.CANCELLED,
            EntryShardState.EXPIRED,
        ):
            entry.state = transition(entry.state, EntryShardState.FAILED)
        entry.error = reason
        self._release_resources_locked(entry)
        self.metrics.increment("vector_entry_failures")

    def _release_resources_locked(self, entry: EntryShardRecord) -> None:
        if entry.resources_released:
            return
        self.allocator.free(entry.allocation)
        entry.resources_released = True
        self._refresh_metrics()

    def reap_expired(
        self, now: Optional[float] = None, *, reap_entries: bool = True
    ) -> Dict[str, int]:
        now = time.monotonic() if now is None else now
        expired_deliveries = 0
        expired_entries = 0
        with self._lock:
            for entry in self.entries.values():
                for delivery in entry.deliveries.values():
                    if (
                        delivery.state not in DELIVERY_TERMINAL_STATES
                        and delivery.deadline <= now
                    ):
                        if delivery.transfer_handle is not None:
                            self.transfer_engine.abort(delivery.transfer_handle)
                        delivery.state = transition(delivery.state, DeliveryState.EXPIRED)
                        delivery.error = "delivery timeout"
                        entry.active_delivery_count -= 1
                        expired_deliveries += 1
                if (
                    reap_entries
                    and not entry.resources_released
                    and entry.active_delivery_count == 0
                    and entry.expires_at <= now
                    and entry.state
                    not in (
                        EntryShardState.RELEASED,
                        EntryShardState.FAILED,
                        EntryShardState.CANCELLED,
                        EntryShardState.EXPIRED,
                    )
                ):
                    entry.state = transition(entry.state, EntryShardState.EXPIRED)
                    entry.error = "entry TTL expired"
                    self._release_resources_locked(entry)
                    expired_entries += 1
        if expired_deliveries:
            self.metrics.increment("vector_deliveries_expired", expired_deliveries)
        if expired_entries:
            self.metrics.increment("vector_entries_expired", expired_entries)
        return {"entries": expired_entries, "deliveries": expired_deliveries}

    def state_for(self, key: KVEntryKey) -> Optional[EntryShardState]:
        with self._lock:
            entry = self.entries.get(key)
            return entry.state if entry is not None else None

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            self._refresh_metrics()
            return {
                "rank": self.rank,
                "world_size": self.world_size,
                "rail": self.rail,
                "device": self.device,
                "page_bytes": self.page_bytes,
                "total_pages": self.allocator.total_pages,
                "available_pages": self.allocator.available_pages,
                "entries": [entry.to_dict() for entry in self.entries.values()],
                "transport": self.transfer_engine.health(),
                "metrics": self.metrics.snapshot(),
            }

    def close(self) -> None:
        with self._lock:
            for entry in self.entries.values():
                if not entry.resources_released:
                    self._release_resources_locked(entry)
            self.transfer_engine.release_memory(self.registration)
