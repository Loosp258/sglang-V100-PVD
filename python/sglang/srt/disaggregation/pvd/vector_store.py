"""Rank-local GPU KV page store used by the PVD vector worker group."""

from __future__ import annotations

import copy
import logging
import threading
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import torch
from sglang.srt.disaggregation.pvd.full_kv_fanin_plan import (
    FULL_KV_FANIN_PROTOCOL,
    RANK_PACKED_FULL_KV_FANIN_PROTOCOL,
    validate_fanin_plan,
)
from sglang.srt.disaggregation.pvd.full_kv_fanin_writer import FullKVFanInWriter
from sglang.srt.disaggregation.pvd.metrics import PVDMetrics
from sglang.srt.disaggregation.pvd.protocol import (
    PVD_GENERATION_METADATA_KEY,
    PVD_RECEIVER_EPOCH_METADATA_KEY,
    PVD_TRANSFER_LIFECYCLE_PROTOCOL,
    KVEntryKey,
    KVEntryManifest,
    KVLayoutSignature,
    KVShardManifest,
    RemoteRegionDescriptor,
    WriteIdentity,
    upload_transfer_id,
)
from sglang.srt.disaggregation.pvd.request_state import (
    DELIVERY_TERMINAL_STATES,
    ENTRY_SHARD_TERMINAL_STATES,
    DeliveryState,
    EntryShardState,
    transition,
)
from sglang.srt.disaggregation.pvd.sharding import (
    layout_from_destination,
    source_rank_and_head_offset,
)
from sglang.srt.disaggregation.pvd.sparse_copy import copy_sparse_kv_into
from sglang.srt.disaggregation.pvd.sparse_delivery import (
    SPARSE_DELIVERY_KEY,
    SparseDeliveryManifest,
)
from sglang.srt.disaggregation.pvd.sparse_pack_plan import SparsePackCompletionUnknown
from sglang.srt.disaggregation.pvd.transfer_authorization import WriteAuthorization
from sglang.srt.disaggregation.pvd.transfer_engine import (
    MemorySlice,
    RegisteredMemory,
    TransferEngine,
    TransferHandle,
    TransferStatus,
    descriptor_with_slice,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    GuardUnpinOutcome,
    ResourceGuard,
    TransportState,
    budget_of,
)
from sglang.srt.disaggregation.pvd.transfer_progress import PVD_TRANSFER_CAPABILITY

logger = logging.getLogger(__name__)

# Owner token for the pre-lifecycle upload pin.  Legacy callers that create an
# Entry without an uploader epoch keep the old behaviour: the pin is taken when
# the destination descriptor is published and dropped by a successful
# commit_p_write.  That report is a completion claim, not transport-terminal
# proof, so it can never be used by a lifecycle-v1 upload.
_LEGACY_UPLOAD_OWNER = "upload"

_UPLOAD_TERMINAL_STATES = frozenset(
    {
        TransportState.NOT_SUBMITTED,
        TransportState.TERMINAL_SUCCESS,
        TransportState.TERMINAL_FAILED,
    }
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

    @property
    def largest_contiguous_free_pages(self) -> int:
        with self._lock:
            return max((count for _, count in self._free_ranges), default=0)

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
    source_guard: Optional[ResourceGuard] = field(default=None, repr=False)
    authorization: Optional[WriteAuthorization] = field(default=None, repr=False)
    staging_guard: Optional[ResourceGuard] = field(default=None, repr=False)
    owner: str = field(default_factory=lambda: uuid.uuid4().hex, repr=False)
    submitting: bool = False
    authorization_begun: bool = False
    local_terminal: Optional[TransportState] = None
    progress_settled: bool = field(default=False, repr=False)
    progress_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    sparse_manifest: Optional[SparseDeliveryManifest] = None
    # Retained on unknown CUDA completion; cancellation must not drop the lease.
    packing_index_lease: Optional[ExitStack] = field(default=None, repr=False)
    packing_workspace: Optional[object] = field(default=None, repr=False)
    fanin_writer: Optional[FullKVFanInWriter] = field(default=None, repr=False)

    def to_dict(self):
        result = {
            "delivery_id": self.delivery_id,
            "entry_key": self.entry_key.to_dict(),
            "destination": self.destination.to_dict(),
            "state": self.state.value,
            "created_at": self.created_at,
            "deadline": self.deadline,
            "error": self.error,
            "sparse_fingerprint": (
                self.sparse_manifest.fingerprint if self.sparse_manifest else None
            ),
            "packing_index_lease_held": self.packing_index_lease is not None,
            "packing_workspace_held": self.packing_workspace is not None,
            "write_identity": (
                self.authorization.identity.to_dict() if self.authorization else None
            ),
            # Observation only: unlike fence_write(), this does not cancel the
            # Delivery. D needs terminal proof before reading, then can ACK it.
            "write_fence": (
                self.authorization.fence(self.authorization.identity)
                if self.authorization
                else None
            ),
            "transferred_bytes": (
                self.transfer_handle.transferred_bytes if self.transfer_handle else 0
            ),
            "transport_state": (
                self.transfer_handle.transport_state.value
                if self.transfer_handle
                else self.local_terminal.value
                if self.local_terminal
                else "preparing"
            ),
        }
        if self.fanin_writer is not None:
            proof = self.fanin_writer.snapshot()
            result.update(
                fanin_proof=proof,
                transferred_bytes=proof["transferred_bytes"],
                transport_state=proof["transport_state"],
            )
        return result


@dataclass
class EntryShardRecord:
    key: KVEntryKey
    layout_fingerprint: str
    layout: KVLayoutSignature
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
    allocation_guard: Optional[ResourceGuard] = field(default=None, repr=False)
    upload_pending: bool = True
    release_requested: bool = False
    pool_owner: str = field(default_factory=lambda: uuid.uuid4().hex, repr=False)
    # Lifecycle-v1 upload state.  ``upload_authorization`` holds the only pin
    # that a lifecycle upload may drop, and only through a matching closed
    # terminal report.  ``upload_close_requested`` is a request to the sender,
    # never a release.
    upload_identity: Optional[WriteIdentity] = field(default=None, repr=False)
    upload_authorization: Optional[WriteAuthorization] = field(default=None, repr=False)
    upload_close_requested: bool = False
    upload_begun: bool = False
    upload_terminal: Optional[TransportState] = None
    upload_committed_bytes: Optional[int] = None

    def to_dict(self):
        return {
            "key": self.key.to_dict(),
            "layout_fingerprint": self.layout_fingerprint,
            "layout": self.layout.to_dict(),
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
            "upload_pending": self.upload_pending,
            "release_requested": self.release_requested,
            "upload_identity": (
                self.upload_identity.to_dict() if self.upload_identity else None
            ),
            "upload_close_requested": self.upload_close_requested,
            "upload_terminal": (
                self.upload_terminal.value if self.upload_terminal else None
            ),
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
        prompt_index: Optional[Any] = None,
        max_entry_records: int = 8192,
        max_delivery_records: int = 65536,
        max_legacy_absent_fences: int = 4096,
        max_absent_write_fences: int = 4096,
        max_absent_entry_cancellations: int = 4096,
        allow_cuda_sparse_packing: bool = False,
        fused_cuda_sparse_packing: bool = False,
        full_kv_fanin_max_slices: Optional[int] = None,
        full_kv_fanin_max_inflight: Optional[int] = None,
        full_kv_fanin_native_batch: bool = False,
    ) -> None:
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank {rank} is outside world size {world_size}")
        if world_size != 2:
            raise ValueError("PVD requires exactly two V storage ranks")
        if page_bytes <= 0:
            raise ValueError("page_bytes must be positive")
        fanin_limits = (full_kv_fanin_max_slices, full_kv_fanin_max_inflight)
        if fanin_limits != (None, None) and any(
            type(v) is not int or v <= 0 for v in fanin_limits
        ):
            raise ValueError("both full-KV fan-in bounds must be positive integers")
        self._fanin_max_slices, self._fanin_max_inflight = fanin_limits
        if type(full_kv_fanin_native_batch) is not bool or (
            full_kv_fanin_native_batch and fanin_limits == (None, None)
        ):
            raise ValueError("native fan-in batch requires configured fan-in bounds")
        if full_kv_fanin_native_batch:
            if not callable(getattr(transfer_engine, "submit_batch_put", None)):
                raise ValueError("transfer engine has no native batch PUT")
            require_batch = getattr(transfer_engine, "require_native_batch", None)
            if callable(require_batch):
                require_batch()
        self._fanin_native_batch = full_kv_fanin_native_batch
        if type(max_entry_records) is not int or max_entry_records <= 0:
            raise ValueError("max_entry_records must be a positive integer")
        if type(max_delivery_records) is not int or max_delivery_records <= 0:
            raise ValueError("max_delivery_records must be a positive integer")
        if type(max_legacy_absent_fences) is not int or max_legacy_absent_fences <= 0:
            raise ValueError("max_legacy_absent_fences must be a positive integer")
        if type(max_absent_write_fences) is not int or max_absent_write_fences <= 0:
            raise ValueError("max_absent_write_fences must be a positive integer")
        if (
            type(max_absent_entry_cancellations) is not int
            or max_absent_entry_cancellations <= 0
        ):
            raise ValueError("max_absent_entry_cancellations must be positive")
        if type(allow_cuda_sparse_packing) is not bool:
            raise ValueError("allow_cuda_sparse_packing must be a boolean")
        if type(fused_cuda_sparse_packing) is not bool or (
            fused_cuda_sparse_packing and not allow_cuda_sparse_packing
        ):
            raise ValueError("fused CUDA packing requires CUDA sparse packing")
        if allow_cuda_sparse_packing and (
            not device.startswith("cuda")
            or prompt_index is None
            or budget_of(transfer_engine) is None
        ):
            raise ValueError("CUDA sparse packing requires CUDA, an index and a budget")
        if device == "cpu" and not allow_cpu_for_tests:
            raise RuntimeError("PVD V storage cannot silently fall back to CPU")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {device} requested but CUDA is unavailable"
            )

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
        self.pool = torch.empty(
            total_pages * page_bytes, dtype=torch.uint8, device=device
        )
        self.registration: RegisteredMemory = transfer_engine.register_memory(
            self.pool,
            endpoint=endpoint,
            rank=rank,
            rail=rail,
            metadata={"role": "vector", "page_bytes": page_bytes},
        )
        # Optional retrieval index over stored prompts. None by default, so a
        # store built without one behaves exactly as before. Full-Prompt
        # delivery stays index-independent; explicit sparse delivery leases it.
        self.prompt_index = prompt_index
        self._quarantined_index_sources = []
        self.allow_cuda_sparse_packing = allow_cuda_sparse_packing
        self.fused_cuda_sparse_packing = fused_cuda_sparse_packing
        self.entries: Dict[KVEntryKey, EntryShardRecord] = {}
        # A worker epoch retains all keys for replay refusal. Refuse new keys
        # at the bound rather than silently evicting a live protocol fence.
        self._max_entry_records = max_entry_records
        self._max_delivery_records = max_delivery_records
        self._delivery_records = 0
        self._max_legacy_absent_fences = max_legacy_absent_fences
        self._legacy_absent_fences = set()
        # Historical records remain addressable for replay rejection. Only
        # allocations that can still expire, build an index or drain a write
        # belong to the maintenance scan.
        self._live_entries: Dict[KVEntryKey, EntryShardRecord] = {}
        # Keep replay tombstones in entries, but do not revisit already
        # released allocations on every maintenance tick.
        self._release_pending: Dict[KVEntryKey, EntryShardRecord] = {}
        # Tombstones stay in entries for replay/fence proof. Only records that
        # can still change transport or cleanup state need background progress.
        self._active_progress: Dict[
            Tuple[KVEntryKey, str], Tuple[EntryShardRecord, DeliveryShardRecord]
        ] = {}
        self.worker_epoch = uuid.uuid4().hex
        self._closed = False
        self._isolated_reason = None
        self._pool_guard = ResourceGuard(
            self.registration,
            lambda: self.transfer_engine.release_memory(self.registration),
        )
        self._fenced_deliveries = set()
        # Lost reserve requests can arrive after D asks to close its receiver.
        # Keep full identities until this worker epoch ends. Never TTL/LRU-evict
        # a returned fence: delayed reserve/start must still be denied. At the
        # bound refuse NEW absence proofs; existing proofs remain retryable.
        self._absent_write_fences = {}
        self._max_absent_write_fences = max_absent_write_fences
        # A create may fail on this rank after the peer has allocated. The
        # coordinator still sends cancel to both ranks. Fence an absent key
        # for this worker epoch so a delayed create cannot allocate afterward.
        self._absent_entry_cancellations = set()
        self._max_absent_entry_cancellations = max_absent_entry_cancellations
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

    def create_entry(
        self,
        manifest: KVEntryManifest,
        *,
        uploader_epoch: Optional[str] = None,
    ) -> EntryShardRecord:
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
            if self._closed or self._isolated_reason:
                raise EntryConflictError("V store is closed or isolated")
            if manifest.key in self._absent_entry_cancellations:
                raise EntryConflictError("entry was cancelled before allocation")
            existing = self.entries.get(manifest.key)
            if existing is not None:
                if (
                    existing.layout_fingerprint != manifest.layout.fingerprint
                    or existing.manifest != shard
                ):
                    raise EntryConflictError(
                        "entry key already exists with a different manifest"
                    )
                # An Entry has exactly one uploader incarnation. A second P
                # incarnation, or a legacy caller arriving after lifecycle
                # metadata exists, must not inherit the first one's pin.
                existing_epoch = (
                    existing.upload_identity.sender_epoch
                    if existing.upload_identity
                    else None
                )
                if existing_epoch != uploader_epoch:
                    raise EntryConflictError(
                        "entry key already exists with a different uploader epoch"
                    )
                return existing

            if len(self.entries) >= self._max_entry_records:
                raise ResourceExhaustedError("V Entry record capacity exceeded")

            allocation = self.allocator.allocate(shard.page_count)
            offset = allocation.start_page * self.page_bytes
            target_region = descriptor_with_slice(
                self.registration, offset=offset, length=shard.expected_bytes
            )
            target_region = replace(
                target_region,
                backend_metadata={
                    **target_region.backend_metadata,
                    PVD_RECEIVER_EPOCH_METADATA_KEY: self.worker_epoch,
                    PVD_GENERATION_METADATA_KEY: uuid.uuid4().hex,
                },
            )
            now = time.monotonic()
            record = EntryShardRecord(
                key=manifest.key,
                layout_fingerprint=manifest.layout.fingerprint,
                layout=manifest.layout,
                manifest=shard,
                allocation=allocation,
                target_region=target_region,
                state=EntryShardState.ALLOCATED,
                created_at=now,
                expires_at=now + self.entry_ttl_secs,
            )
            record.allocation_guard = ResourceGuard(
                allocation, lambda: self._free_allocation(record)
            )
            # Publishing a destination grants a potential remote writer. Never
            # infer that no write exists merely because begin/commit is absent.
            # The pin is taken here, before the descriptor leaves this method.
            if uploader_epoch is None:
                record.allocation_guard.pin(_LEGACY_UPLOAD_OWNER)
            else:
                identity = WriteIdentity(
                    protocol=PVD_TRANSFER_LIFECYCLE_PROTOCOL,
                    sender_epoch=uploader_epoch,
                    receiver_epoch=self.worker_epoch,
                    transfer_id=upload_transfer_id(manifest.key, self.rank),
                    region_id=target_region.region_id,
                    generation=target_region.backend_metadata[
                        PVD_GENERATION_METADATA_KEY
                    ],
                    shard_rank=target_region.rank,
                    key=manifest.key,
                )
                identity.validate_destination(target_region)
                record.upload_identity = identity
                # Constructing the authorization is what pins the allocation in
                # lifecycle mode; no separate legacy owner token is taken, so
                # commit_p_write alone can never drop it.
                record.upload_authorization = WriteAuthorization(
                    identity, record.allocation_guard
                )
            self._pool_guard.pin(record.pool_owner)
            self.entries[manifest.key] = record
            self._live_entries[manifest.key] = record
            self.metrics.increment("vector_entries_created")
            self._refresh_metrics()
            return record

    def begin_p_write(
        self,
        key: KVEntryKey,
        identity: Optional[WriteIdentity] = None,
    ) -> EntryShardRecord:
        with self._lock:
            if self._closed:
                raise EntryConflictError("V store is closed")
            entry = self._entry(key)
            if entry.release_requested:
                raise EntryConflictError("entry release requested")
            authorization = entry.upload_authorization
            if authorization is not None:
                # Lifecycle metadata exists, so there is no legacy fallback.
                if identity is None:
                    raise EntryConflictError(
                        "upload requires a lifecycle write identity"
                    )
                if identity != entry.upload_identity:
                    raise EntryConflictError(
                        "upload write identity does not match this entry"
                    )
                if entry.upload_close_requested:
                    raise EntryConflictError("upload authorization is closing")
                # One-shot gate: a retry, or a second sender, is rejected here.
                authorization.begin(identity)
                entry.upload_begun = True
            elif identity is not None:
                raise EntryConflictError("entry has no lifecycle upload authorization")
            entry.state = transition(entry.state, EntryShardState.P_WRITING)
            return entry

    def commit_p_write(self, key: KVEntryKey, received_bytes: int) -> EntryShardRecord:
        legacy_unpin = False
        with self._lock:
            entry = self._entry(key)
            if entry.release_requested:
                raise EntryConflictError("entry release requested")
            if entry.state == EntryShardState.STORED:
                if entry.received_bytes != received_bytes:
                    raise EntryConflictError(
                        "idempotent commit has a different byte count"
                    )
                return entry
            if received_bytes != entry.manifest.expected_bytes:
                self._fail_entry_locked(
                    entry,
                    f"received {received_bytes} KV bytes, expected {entry.manifest.expected_bytes}",
                )
                raise EntryConflictError(entry.error)
            entry.upload_committed_bytes = received_bytes
            if entry.upload_authorization is None:
                # Legacy successful commit is an explicit completion report, not
                # a timeout inference, but it is still not transport-terminal
                # proof. Lifecycle uploads below never take this branch.
                self._publish_stored_locked(entry, received_bytes)
                entry.upload_pending = False
                legacy_unpin = True
            else:
                if entry.upload_close_requested:
                    raise EntryConflictError("upload authorization is closing")
                # A byte-complete report alone does not publish STORED: the
                # sender must also have reached a successful native terminal.
                self._try_publish_stored_locked(entry)
        if legacy_unpin:
            entry.allocation_guard.unpin(_LEGACY_UPLOAD_OWNER)
        return entry

    def _publish_stored_locked(
        self, entry: EntryShardRecord, received_bytes: int
    ) -> None:
        entry.state = transition(entry.state, EntryShardState.STORED)
        entry.received_bytes = received_bytes
        entry.stored_at = time.monotonic()
        entry.expires_at = entry.stored_at + self.entry_ttl_secs
        for delivery in entry.deliveries.values():
            if delivery.state == DeliveryState.WAITING_SOURCE:
                delivery.state = transition(delivery.state, DeliveryState.D_RESERVED)
        if self.prompt_index is not None:
            # Complete and visible: this is the only point an index may be
            # built from. Recording readiness is all that happens under the
            # lock; the build itself is a separate, caller-driven step.
            self.prompt_index.note_kv_readable(entry.key.transfer_id)
        self.metrics.increment("vector_p_to_v_bytes", received_bytes)
        self.metrics.increment("vector_entries_stored")

    def _try_publish_stored_locked(self, entry: EntryShardRecord) -> None:
        """Publish STORED only when business and transport both succeeded.

        A late native success after business cancellation, TTL expiry or close
        permits reclamation only. It must never resurrect the request.
        """
        if entry.release_requested or entry.state == EntryShardState.STORED:
            return
        if entry.upload_committed_bytes is None:
            return
        if entry.upload_terminal != TransportState.TERMINAL_SUCCESS:
            return
        self._publish_stored_locked(entry, entry.upload_committed_bytes)

    def sync_upload(
        self,
        identity: WriteIdentity,
        state: TransportState,
        closed: bool,
    ) -> Dict[str, object]:
        """Exchange upload transport state with the owning P incarnation.

        This is the only path that can release a lifecycle upload's hold on the
        destination allocation, and only for a matching identity whose sender
        reports a closed authorization together with a transport terminal
        state. Timeout, HTTP failure, TTL expiry, a missing report or a P
        restart are never accepted as completion proof.
        """
        if not isinstance(identity, WriteIdentity):
            raise EntryConflictError("upload sync requires a write identity")
        if not isinstance(state, TransportState):
            raise EntryConflictError("upload sync requires a transport state")
        if not isinstance(closed, bool):
            raise EntryConflictError("upload sync requires a boolean closed flag")

        observation = None
        with self._lock:
            entry = self._entry(identity.key)
            authorization = entry.upload_authorization
            if authorization is None or entry.upload_identity is None:
                raise EntryConflictError("entry has no lifecycle upload authorization")
            if identity != entry.upload_identity:
                raise EntryConflictError(
                    "upload write identity does not match this entry"
                )
            # Region id, allocation generation, receiver epoch and rank are
            # re-checked against the live destination, so a retried report can
            # never close a different allocation generation.
            identity.validate_destination(entry.target_region)

            if state == TransportState.UNKNOWN:
                self._isolated_reason = "P upload transport terminal state is unknown"
                entry.upload_close_requested = True
                return self._upload_sync_reply_locked(entry, terminal_ack=False)

            if not closed:
                return self._upload_sync_reply_locked(
                    entry, terminal_ack=entry.upload_terminal is not None
                )

            if state not in _UPLOAD_TERMINAL_STATES:
                raise EntryConflictError(
                    "a closed upload report requires a transport terminal state"
                )
            effective = state
            if state == TransportState.NOT_SUBMITTED and entry.upload_begun:
                # begin consumed the one-shot gate. A proven pre-native
                # rejection is a safe failed gate, not a claim that the
                # authorization never began.
                effective = TransportState.TERMINAL_FAILED
            if entry.upload_terminal is not None and entry.upload_terminal != effective:
                raise EntryConflictError(
                    "upload already reported a different terminal state"
                )
            entry.upload_terminal = effective
            entry.upload_pending = False
            observation = (authorization, identity, effective)
            if effective == TransportState.TERMINAL_SUCCESS:
                self._try_publish_stored_locked(entry)
            elif entry.state not in ENTRY_SHARD_TERMINAL_STATES:
                self._fail_entry_locked(
                    entry, "P upload did not reach a successful transport terminal"
                )

        if observation is not None:
            authorization, identity, effective = observation
            # Closing and unpinning run outside the store lock: the unpin can
            # invoke the allocation release callback.
            authorization.close()
            authorization.observe_terminal(identity, effective)
        self._progress_releases()
        with self._lock:
            return self._upload_sync_reply_locked(entry, terminal_ack=True)

    def _upload_sync_reply_locked(
        self, entry: EntryShardRecord, *, terminal_ack: bool
    ) -> Dict[str, object]:
        return {
            "identity": entry.upload_identity.to_dict(),
            "close_requested": bool(entry.upload_close_requested),
            "terminal_ack": bool(terminal_ack),
            "entry_state": entry.state.value,
            "upload_terminal": (
                entry.upload_terminal.value if entry.upload_terminal else None
            ),
            "resources_released": bool(entry.resources_released),
        }

    def _fanin_plan(self, manifest):
        if self._fanin_max_slices is None:
            raise EntryConflictError(
                "full-KV fan-in is disabled; explicit bounds required"
            )
        return validate_fanin_plan(manifest, max_slices=self._fanin_max_slices)

    def reserve_fanin_delivery(
        self, manifest, *, expected_sender_epoch=None
    ) -> DeliveryShardRecord:
        """Reserve against the actual Entry allocation under its business lock."""
        manifest = copy.deepcopy(manifest)
        plan = self._fanin_plan(manifest)
        delivery_id = f"{plan.delivery_id}:d{plan.destination.rank}:v{self.rank}"
        with self._lock:
            if (
                expected_sender_epoch is not None
                and expected_sender_epoch != self.worker_epoch
            ):
                raise EntryConflictError("fan-in reserve sender epoch mismatch")
            if self._closed or self._isolated_reason:
                raise EntryConflictError("V store is closed or isolated")
            if (plan.key, delivery_id) in self._fenced_deliveries:
                raise EntryConflictError("delivery has been fenced")
            entry = self._entry(plan.key)
            if entry.resources_released or entry.release_requested:
                raise EntryConflictError("entry resources have already been released")
            existing = entry.deliveries.get(delivery_id)
            if existing is not None:
                if (
                    existing.fanin_writer is None
                    or existing.fanin_writer.plan.fingerprint != plan.fingerprint
                ):
                    raise EntryConflictError(
                        "delivery id already exists with a different fan-in plan"
                    )
                return existing
            if self._delivery_records >= self._max_delivery_records:
                raise ResourceExhaustedError("V Delivery record capacity exceeded")
            source = MemorySlice(
                self.registration,
                entry.allocation.start_page * self.page_bytes,
                entry.manifest.expected_bytes,
            )
            writer = FullKVFanInWriter(
                manifest,
                engine=self.transfer_engine,
                source=source,
                source_guard=entry.allocation_guard,
                source_key=entry.key,
                source_layout=entry.layout,
                source_rank=self.rank,
                sender_epoch=self.worker_epoch,
                max_slices=self._fanin_max_slices,
                max_inflight=self._fanin_max_inflight,
                use_native_batch=self._fanin_native_batch,
            )
            now = time.monotonic()
            delivery = DeliveryShardRecord(
                delivery_id=delivery_id,
                entry_key=plan.key,
                destination=copy.deepcopy(plan.destination),
                state=DeliveryState.D_RESERVED
                if entry.state == EntryShardState.STORED
                else DeliveryState.WAITING_SOURCE,
                created_at=now,
                deadline=now + self.delivery_timeout_secs,
                source_guard=entry.allocation_guard,
                authorization=writer._authorization,
                fanin_writer=writer,
            )
            entry.deliveries[delivery_id] = delivery
            self._delivery_records += 1
            self._active_progress[(entry.key, delivery_id)] = (entry, delivery)
            entry.active_delivery_count += 1
            self.metrics.increment("vector_deliveries_created")
            return delivery

    def fence_fanin_delivery(self, manifest, identity):
        """Close a known writer, or tombstone an unreserved writer before proof."""
        plan = self._fanin_plan(manifest)
        if (
            not isinstance(identity, WriteIdentity)
            or identity.key != plan.key
            or self.rank not in plan.writers
            or identity.transfer_id
            != f"{plan.delivery_id}:d{plan.destination.rank}:v{self.rank}"
        ):
            raise EntryConflictError("fan-in fence identity mismatch")
        identity.validate_destination(plan.destination)
        with self._lock:
            if identity.sender_epoch != self.worker_epoch:
                raise EntryConflictError("fan-in fence sender epoch mismatch")
            entry = self._entry(plan.key)
            if entry.layout.fingerprint != plan.storage.fingerprint:
                raise EntryConflictError("fan-in fence storage layout mismatch")
            delivery = entry.deliveries.get(identity.transfer_id)
            if delivery is not None and (
                delivery.fanin_writer is None
                or delivery.fanin_writer.plan.fingerprint != plan.fingerprint
            ):
                raise EntryConflictError("fan-in fence plan mismatch")
            if delivery is None:
                token = (plan.key, identity.transfer_id)
                previous = self._absent_write_fences.get(token)
                if previous is not None and previous != identity:
                    raise EntryConflictError("fan-in absent fence identity mismatch")
                if previous is None:
                    if len(self._absent_write_fences) >= self._max_absent_write_fences:
                        raise ResourceExhaustedError(
                            "absent write fence capacity exceeded"
                        )
                    self._absent_write_fences[token] = identity
                self._fenced_deliveries.add(token)
            else:
                if delivery.authorization.identity != identity:
                    raise EntryConflictError("fan-in stored writer identity mismatch")
                self._cancel_delivery_locked(entry, delivery, "Decode fenced fan-in")
        # Check and close under one lock; native progress must run outside it.
        if delivery is not None:
            self._progress_delivery(entry, delivery)
            self._progress_releases()
            return delivery.fanin_writer.snapshot()
        return {
            "protocol": plan.protocol,
            "plan_fingerprint": plan.fingerprint,
            "source_rank": self.rank,
            "identity": identity.to_dict(),
            "fenced": True,
            "transport_state": "not_submitted",
            "transferred_bytes": 0,
        }

    def reserve_delivery(
        self,
        key: KVEntryKey,
        delivery_id: str,
        destination: RemoteRegionDescriptor,
    ) -> DeliveryShardRecord:
        destination = copy.deepcopy(destination)
        if destination.rail != self.rail:
            raise EntryConflictError(
                f"rank {self.rank} requires rail {self.rail}, destination uses {destination.rail}"
            )
        with self._lock:
            if self._closed or self._isolated_reason:
                raise EntryConflictError("V store is closed or isolated")
            if (key, delivery_id) in self._fenced_deliveries:
                raise EntryConflictError("delivery has been fenced")
            entry = self._entry(key)
            if entry.resources_released or entry.release_requested:
                raise EntryConflictError("entry resources have already been released")
            existing = entry.deliveries.get(delivery_id)
            if existing is not None:
                if existing.fanin_writer is not None:
                    raise EntryConflictError("delivery id belongs to full-KV fan-in")
                if existing.destination != destination:
                    raise EntryConflictError(
                        "delivery id already exists with a different destination"
                    )
                return existing
            if self._delivery_records >= self._max_delivery_records:
                raise ResourceExhaustedError("V Delivery record capacity exceeded")
            sparse = None
            if SPARSE_DELIVERY_KEY in destination.backend_metadata:
                sparse = SparseDeliveryManifest.from_dict(
                    destination.backend_metadata[SPARSE_DELIVERY_KEY]
                )
                self._validate_sparse_destination(entry, destination, sparse)
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
                sparse_manifest=sparse,
            )
            delivery.source_guard = entry.allocation_guard
            metadata = destination.backend_metadata
            if (
                PVD_RECEIVER_EPOCH_METADATA_KEY in metadata
                or PVD_GENERATION_METADATA_KEY in metadata
            ):
                identity = WriteIdentity(
                    protocol=PVD_TRANSFER_LIFECYCLE_PROTOCOL,
                    sender_epoch=self.worker_epoch,
                    receiver_epoch=metadata.get(PVD_RECEIVER_EPOCH_METADATA_KEY),
                    transfer_id=delivery_id,
                    region_id=destination.region_id,
                    generation=metadata.get(PVD_GENERATION_METADATA_KEY),
                    shard_rank=destination.rank,
                    key=key,
                )
                delivery.authorization = WriteAuthorization(
                    identity, entry.allocation_guard
                )
            else:
                # Staged legacy callers receive no lifecycle-v1 fence proof.
                entry.allocation_guard.pin(delivery.owner)
            entry.deliveries[delivery_id] = delivery
            self._delivery_records += 1
            self._active_progress[(entry.key, delivery_id)] = (entry, delivery)
            entry.active_delivery_count += 1
            self.metrics.increment("vector_deliveries_created")
            return delivery

    def start_delivery(self, key: KVEntryKey, delivery_id: str) -> DeliveryShardRecord:
        with self._lock:
            if self._closed or self._isolated_reason:
                raise EntryConflictError("V store is closed or isolated")
            if (key, delivery_id) in self._fenced_deliveries:
                raise EntryConflictError("delivery has been fenced")
            entry = self._entry(key)
            delivery = entry.deliveries[delivery_id]
            if delivery.state in DELIVERY_TERMINAL_STATES or delivery.state in (
                DeliveryState.DELIVERED,
                DeliveryState.V_WRITING,
            ):
                return delivery
            if entry.release_requested:
                raise EntryConflictError("entry release requested")
            if entry.state != EntryShardState.STORED:
                return delivery
            delivery.state = transition(delivery.state, DeliveryState.D_RESERVED)
            delivery.state = transition(delivery.state, DeliveryState.V_WRITING)
            delivery.submitting = True

        if delivery.fanin_writer is not None:
            try:
                delivery.fanin_writer.start()
            finally:
                with self._lock:
                    delivery.submitting = False
                self._progress_delivery(entry, delivery)
                self._progress_releases()
            return delivery

        attempted = False
        packing_safe = True
        try:
            # The reservation already pins Entry pages. Packing and native
            # calls deliberately run without the store's business lock.
            try:
                local = self._prepare_delivery_source(entry, delivery)
            finally:
                # Cancellation can win before the adapter's own CUDA fence.
                # Even then, packing kernels must finish reading Entry pages
                # before a NOT_SUBMITTED path may release their source pin.
                if self.pool.is_cuda:
                    try:
                        torch.cuda.synchronize(self.pool.device)
                    except Exception:
                        packing_safe = False
                        raise
            with self._lock:
                if delivery.state in DELIVERY_TERMINAL_STATES:
                    delivery.local_terminal = TransportState.NOT_SUBMITTED
                    return delivery
                if delivery.authorization:
                    delivery.authorization.begin(delivery.authorization.identity)
                    delivery.authorization_begun = True
                attempted = True
            handle = self.transfer_engine.submit_put(local, delivery.destination)
            with self._lock:
                delivery.transfer_handle = handle
        except Exception as exc:
            logger.exception(
                "PVD V delivery source/submit failed: rank=%s delivery_id=%s",
                self.rank,
                delivery_id,
            )
            with self._lock:
                uncertain = (
                    attempted
                    or not packing_safe
                    or delivery.local_terminal == TransportState.UNKNOWN
                )
                delivery.local_terminal = (
                    TransportState.UNKNOWN
                    if uncertain
                    else TransportState.NOT_SUBMITTED
                )
                if uncertain:
                    self._isolated_reason = (
                        "V submission or GPU packing safety is unknown"
                    )
                self._cancel_delivery_locked(
                    entry, delivery, str(exc), DeliveryState.FAILED
                )
        finally:
            with self._lock:
                delivery.submitting = False
            self._progress_delivery(entry, delivery)
            self._progress_releases()
        return delivery

    def _prepare_delivery_source(self, entry, delivery) -> MemorySlice:
        if delivery.sparse_manifest is not None:
            return self._prepare_sparse_source(entry, delivery)
        allocation_offset = entry.allocation.start_page * self.page_bytes
        local = MemorySlice(
            self.registration, allocation_offset, entry.manifest.expected_bytes
        )
        layout_value = delivery.destination.backend_metadata.get("pvd_layout")
        if layout_value is None:
            if delivery.destination.rank != self.rank:
                raise EntryConflictError(
                    "legacy delivery requires matching V and D ranks"
                )
            return local
        destination_layout = layout_from_destination(delivery.destination)
        source_rank, source_head_offset = source_rank_and_head_offset(
            entry.layout, destination_layout, delivery.destination.rank
        )
        if source_rank != self.rank:
            raise EntryConflictError("destination belongs to another V shard")
        token_count = entry.manifest.page_count * entry.layout.page_size
        destination_bytes = [
            int(v) for v in destination_layout.extra["component_bytes_per_token"]
        ]
        if delivery.destination.length != sum(destination_bytes) * token_count:
            raise EntryConflictError("destination byte count does not match layout")
        if destination_layout.fingerprint == entry.layout.fingerprint:
            return local
        source_bytes = [int(v) for v in entry.layout.extra["component_bytes_per_token"]]
        source_region = self.pool[
            allocation_offset : allocation_offset + entry.manifest.expected_bytes
        ]
        # Preallocate the final buffer and copy component views directly. No
        # per-component materialization or concatenation peak is necessary.
        final_bytes = sum(destination_bytes) * token_count
        budget = budget_of(self.transfer_engine)
        budget_owner = f"v-repack:{delivery.owner}"
        if budget is not None:
            budget.reserve(budget_owner, final_bytes, 0)
        try:
            staging = torch.empty(
                final_bytes, dtype=torch.uint8, device=self.pool.device
            )
        except BaseException:
            if budget is not None:
                budget.release(budget_owner)
            raise
        registered = [None]

        def _release_staging():
            if registered[0] is not None:
                self.transfer_engine.release_memory(registered[0])
            if budget is not None:
                budget.release(budget_owner)

        # Own backing bytes BEFORE copies or native registration. On CUDA the
        # start_delivery finally block synchronizes packing before safe release;
        # failure to synchronize quarantines this guard along with the Entry.
        guard = ResourceGuard(staging, _release_staging)
        guard.pin(delivery.owner)
        with self._lock:
            delivery.staging_guard = guard
        guard.request_release()
        component_base = destination_base = 0
        for source_bpt, destination_bpt in zip(
            source_bytes, destination_bytes, strict=True
        ):
            bytes_per_head = source_bpt // entry.layout.kv_heads_per_rank
            byte_start = source_head_offset * bytes_per_head
            source = source_region[
                component_base : component_base + token_count * source_bpt
            ].reshape(token_count, source_bpt)
            extent = token_count * destination_bpt
            staging[destination_base : destination_base + extent].reshape(
                token_count, destination_bpt
            ).copy_(source[:, byte_start : byte_start + destination_bpt])
            component_base += token_count * source_bpt
            destination_base += extent
        try:
            registration = self.transfer_engine.register_memory(
                staging,
                endpoint="pvd-vector-slice",
                rank=self.rank,
                rail=self.rail,
                metadata={"delivery_id": delivery.delivery_id},
            )
        except BaseException:
            with self._lock:
                delivery.local_terminal = TransportState.UNKNOWN
                self._isolated_reason = "repack staging registration outcome unknown"
            raise
        registered[0] = registration
        return MemorySlice(registration, 0, staging.numel())

    def _validate_sparse_destination(self, entry, destination, manifest):
        # Never silently move GPU KV to host. CUDA uses an explicit synchronous
        # baseline, not a claim of overlap, RDMA visibility or validated kernels.
        if self.pool.device.type != "cpu" and not (
            self.pool.device.type == "cuda" and self.allow_cuda_sparse_packing
        ):
            raise EntryConflictError("sparse GPU packing requires explicit CUDA opt-in")
        metadata = destination.backend_metadata
        if (
            self.prompt_index is None
            or budget_of(self.transfer_engine) is None
            or "pvd_layout" in metadata
            or PVD_RECEIVER_EPOCH_METADATA_KEY not in metadata
            or PVD_GENERATION_METADATA_KEY not in metadata
        ):
            raise EntryConflictError(
                "sparse delivery requires index, budget and lifecycle authorization; no dense layout"
            )
        first = manifest.specs[0]
        shard, layout = entry.manifest, entry.layout
        if (
            first.entry_transfer_id != entry.key.transfer_id
            or first.layout_fingerprint != layout.fingerprint
            or (manifest.dtype, manifest.head_dim) != (layout.kv_dtype, layout.head_dim)
            or destination.length != manifest.nbytes
        ):
            raise EntryConflictError(
                "sparse Entry/layout/destination byte extent mismatch"
            )
        valid = (shard.page_count - 1) * layout.page_size + shard.last_page_valid_tokens
        start_head = self.rank * layout.kv_heads_per_rank
        for spec in manifest.specs:
            if (
                not shard.layer_start <= spec.layer < shard.layer_end
                or not start_head
                <= spec.kv_head
                < start_head + layout.kv_heads_per_rank
                or any(t >= valid for t in spec.token_ids)
            ):
                raise EntryConflictError(
                    "sparse selection is outside this source shard"
                )
        # Revalidated under a pinned index record when copying at start_delivery.
        with self.prompt_index.pin_selection(manifest):
            pass

    def _prepare_sparse_source(self, entry, delivery):
        manifest = delivery.sparse_manifest
        budget = budget_of(self.transfer_engine)
        owner = f"v-sparse:{delivery.owner}"
        budget.reserve(owner, manifest.nbytes, 0)
        try:
            staging = torch.empty(
                manifest.nbytes, dtype=torch.uint8, device=self.pool.device
            )
        except BaseException:
            budget.release(owner)
            raise
        registered = [None]

        def release():
            if registered[0] is not None:
                self.transfer_engine.release_memory(registered[0])
            budget.release(owner)

        guard = ResourceGuard(staging, release)
        guard.pin(delivery.owner)
        with self._lock:
            delivery.staging_guard = guard
        guard.request_release()  # only terminal progress may remove the owner
        offset = entry.allocation.start_page * self.page_bytes
        source = self.pool[offset : offset + entry.manifest.expected_bytes]
        lease = ExitStack()
        descriptor = lease.enter_context(self.prompt_index.pin_selection(manifest))
        delivery.packing_index_lease = lease
        workspace = None
        try:
            if self.fused_cuda_sparse_packing:
                from sglang.srt.disaggregation.pvd.triton_sparse_pack import (
                    SparsePackWorkspace,
                )

                workspace = SparsePackWorkspace(
                    manifest,
                    shard=entry.manifest,
                    layout=entry.layout,
                    device=self.pool.device,
                    budget=budget,
                    owner=f"v-sparse-meta:{delivery.owner}",
                )
                delivery.packing_workspace = workspace
            copy_sparse_kv_into(
                source,
                staging,
                manifest=manifest,
                layout=entry.layout,
                shard=entry.manifest,
                entry_transfer_id=entry.key.transfer_id,
                index_version=descriptor.index_version,
                id_mapping_version=descriptor.id_mapping_version,
                allow_cuda=self.allow_cuda_sparse_packing,
                fused_workspace=workspace,
            )
        except SparsePackCompletionUnknown:
            with self._lock:
                delivery.local_terminal = TransportState.UNKNOWN
                self._isolated_reason = "sparse metadata upload completion unknown"
            raise
        finally:
            # Keep Entry, staging and index leases through both successful and
            # partly failed copies. If completion is unknown, retain everything
            # for worker isolation; a later incidental synchronize is no repair.
            try:
                if self.pool.is_cuda:
                    torch.cuda.synchronize(self.pool.device)
            except BaseException:
                with self._lock:
                    delivery.local_terminal = TransportState.UNKNOWN
                    self._isolated_reason = "sparse CUDA packing completion unknown"
                raise
            if workspace is not None:
                workspace.release_after_fence()
                delivery.packing_workspace = None
            lease.close()
            delivery.packing_index_lease = None
        try:
            registration = self.transfer_engine.register_memory(
                staging,
                endpoint="pvd-vector-sparse",
                rank=self.rank,
                rail=self.rail,
                metadata={
                    "delivery_id": delivery.delivery_id,
                    "sparse_fingerprint": manifest.fingerprint,
                },
            )
        except BaseException:
            # Registration may have reached native before raising. No handle
            # means no safe unregister proof: quarantine, retain tensor/budget.
            with self._lock:
                delivery.local_terminal = TransportState.UNKNOWN
                self._isolated_reason = "sparse staging registration outcome unknown"
            raise
        registered[0] = registration
        return MemorySlice(registration, 0, manifest.nbytes)

    def _progress_delivery(self, entry, delivery) -> None:
        # One poll owner per delivery, including fencers and TTL/close callers.
        # Do not wait for another native poll under the store lock.
        if not delivery.progress_lock.acquire(blocking=False):
            return
        try:
            if delivery.progress_settled:
                return
            if delivery.fanin_writer is not None:
                self._progress_fanin_delivery(entry, delivery)
                if (
                    delivery.state
                    in (DeliveryState.DELIVERED, *DELIVERY_TERMINAL_STATES)
                    and delivery.fanin_writer.cleanup_complete()
                ):
                    self._settle_delivery_progress(entry, delivery)
                return
            with self._lock:
                if delivery.submitting:
                    return
                handle = delivery.transfer_handle
                terminal = delivery.local_terminal
                cancelled = delivery.state in DELIVERY_TERMINAL_STATES
            if handle is not None:
                if handle.transport_state != TransportState.UNKNOWN:
                    try:
                        if (
                            cancelled
                            and not handle.transport_state.is_locally_safe_to_release
                        ):
                            self.transfer_engine.abort(handle)
                        if not (
                            handle.transport_state.is_locally_safe_to_release
                            and self.transfer_engine.cleanup_complete(handle)
                        ):
                            self.transfer_engine.poll(handle)
                    except Exception as exc:
                        # A lost native status/handle is not terminal evidence.
                        # Do not repeatedly touch a handle whose safety is lost.
                        with handle._lock:
                            handle.transport_state = TransportState.UNKNOWN
                            handle.status = TransferStatus.FAILED
                            handle.error = f"V native polling failed: {exc}"
                terminal = handle.transport_state
            with self._lock:
                if terminal == TransportState.UNKNOWN:
                    self._isolated_reason = "V transport terminal state is unknown"
                if delivery.state == DeliveryState.V_WRITING and handle is not None:
                    if (
                        terminal == TransportState.TERMINAL_SUCCESS
                        and handle.status == TransferStatus.SUCCESS
                    ):
                        if (
                            delivery.sparse_manifest is not None
                            and handle.transferred_bytes
                            != delivery.sparse_manifest.nbytes
                        ):
                            self._cancel_delivery_locked(
                                entry,
                                delivery,
                                "sparse transfer byte count mismatch",
                                DeliveryState.FAILED,
                            )
                        else:
                            delivery.state = transition(
                                delivery.state, DeliveryState.DELIVERED
                            )
                            self.metrics.increment(
                                "vector_v_to_d_bytes", handle.transferred_bytes
                            )
                            self.metrics.increment("vector_deliveries_completed")
                    elif handle.status in (
                        TransferStatus.FAILED,
                        TransferStatus.CANCELLED,
                    ):
                        self._cancel_delivery_locked(
                            entry,
                            delivery,
                            handle.error or handle.status.value,
                            DeliveryState.FAILED,
                        )
                safe = terminal is not None and terminal.is_locally_safe_to_release
                if safe and delivery.authorization:
                    delivery.authorization.close()
            source_done = staging_done = False
            if safe:
                if delivery.authorization:
                    # begin consumed the sender gate, but the adapter may have
                    # rejected locally before native submission. That is a safe
                    # failed gate, not a claim that begin never happened.
                    auth_terminal = (
                        TransportState.TERMINAL_FAILED
                        if terminal == TransportState.NOT_SUBMITTED
                        and delivery.authorization_begun
                        else terminal
                    )
                    delivery.authorization.observe_terminal(
                        delivery.authorization.identity, auth_terminal
                    )
                    source_done = delivery.authorization.cleanup_complete
                else:
                    source_done = (
                        delivery.source_guard.unpin(delivery.owner)
                        != GuardUnpinOutcome.RELEASE_IN_PROGRESS
                    )
                if delivery.staging_guard is not None:
                    staging_done = (
                        delivery.staging_guard.unpin(delivery.owner)
                        != GuardUnpinOutcome.RELEASE_IN_PROGRESS
                    )
                else:
                    staging_done = True
            if (
                safe
                and source_done
                and staging_done
                and (handle is None or self.transfer_engine.cleanup_complete(handle))
                and delivery.state
                in (DeliveryState.DELIVERED, *DELIVERY_TERMINAL_STATES)
            ):
                self._settle_delivery_progress(entry, delivery)
        except Exception as exc:
            # Native adapter exceptions are normally mapped to UNKNOWN there.
            # Preserve all ownership here, including failed cleanup callbacks.
            logger.warning(
                "V transfer progress retained resources: %s: %s",
                delivery.delivery_id,
                exc,
            )
        finally:
            delivery.progress_lock.release()

    def _settle_delivery_progress(self, entry, delivery) -> None:
        with self._lock:
            delivery.progress_settled = True
            self._active_progress.pop((entry.key, delivery.delivery_id), None)

    def _progress_fanin_delivery(self, entry, delivery):
        with self._lock:
            cancelled = delivery.state in DELIVERY_TERMINAL_STATES
        writer = delivery.fanin_writer
        proof = writer.cancel() if cancelled else writer.poll()
        with self._lock:
            state = TransportState(proof["transport_state"])
            delivery.local_terminal = state
            if state == TransportState.UNKNOWN:
                self._isolated_reason = "V fan-in transport terminal state is unknown"
                self._cancel_delivery_locked(
                    entry,
                    delivery,
                    writer.error or "fan-in uncertain",
                    DeliveryState.FAILED,
                )
            elif delivery.state == DeliveryState.V_WRITING:
                if proof["fenced"] and state == TransportState.TERMINAL_SUCCESS:
                    delivery.state = transition(delivery.state, DeliveryState.DELIVERED)
                    self.metrics.increment(
                        "vector_v_to_d_bytes", proof["transferred_bytes"]
                    )
                    self.metrics.increment("vector_deliveries_completed")
                elif (
                    state in (TransportState.TERMINAL_FAILED, TransportState.DRAINING)
                    or proof["fenced"]
                ):
                    self._cancel_delivery_locked(
                        entry,
                        delivery,
                        writer.error or "fan-in failed",
                        DeliveryState.FAILED,
                    )

    def progress_transfers(self) -> None:
        with self._lock:
            records = list(self._active_progress.values())
        for entry, delivery in records:
            self._progress_delivery(entry, delivery)
        self._progress_releases()

    def poll_delivery(self, key: KVEntryKey, delivery_id: str) -> DeliveryShardRecord:
        with self._lock:
            entry = self._entry(key)
            delivery = entry.deliveries[delivery_id]
        self._progress_delivery(entry, delivery)
        self._progress_releases()
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

    def _cancel_delivery_locked(
        self, entry, delivery, reason, state=DeliveryState.CANCELLED
    ):
        if delivery.state not in DELIVERY_TERMINAL_STATES:
            delivery.state = transition(delivery.state, state)
            delivery.error = reason
            entry.active_delivery_count -= 1
        self._fenced_deliveries.add((entry.key, delivery.delivery_id))
        if delivery.authorization:
            delivery.authorization.close()
        if (
            delivery.fanin_writer is None
            and not delivery.submitting
            and delivery.transfer_handle is None
            and delivery.local_terminal is None
        ):
            delivery.local_terminal = TransportState.NOT_SUBMITTED

    def cancel_delivery(
        self, key: KVEntryKey, delivery_id: str, reason: str
    ) -> DeliveryShardRecord:
        with self._lock:
            entry = self._entry(key)
            delivery = entry.deliveries[delivery_id]
            self._cancel_delivery_locked(entry, delivery, reason)
        self._progress_delivery(entry, delivery)
        self._progress_releases()
        return delivery

    def fence_write(self, identity: WriteIdentity):
        with self._lock:
            if not isinstance(identity, WriteIdentity):
                raise EntryConflictError("complete write identity required")
            if identity.sender_epoch != self.worker_epoch:
                raise EntryConflictError("write authorization identity mismatch")
            entry = self._entry(identity.key)
            delivery = entry.deliveries.get(identity.transfer_id)
            if delivery is None:
                # Entry/delivery records are retained for this worker's life.
                # No record means no write was submitted under this key/id.
                # Check AND close under the SAME lock as reserve/start. Absence
                # alone is NOT proof: the tombstone prevents a delayed request
                # from publishing a write after the receiver unregisters.
                token = (identity.key, identity.transfer_id)
                previous = self._absent_write_fences.get(token)
                if previous is not None and previous != identity:
                    raise EntryConflictError("write authorization identity mismatch")
                if previous is None:
                    if len(self._absent_write_fences) >= self._max_absent_write_fences:
                        raise ResourceExhaustedError(
                            "absent write fence capacity exceeded"
                        )
                    self._absent_write_fences[token] = identity
                    self._fenced_deliveries.add(token)
                return {**identity.to_dict(), "fenced": True}
            if delivery.authorization is None:
                raise EntryConflictError("unknown write authorization")
            if (
                identity != delivery.authorization.identity
                or identity.sender_epoch != self.worker_epoch
            ):
                raise EntryConflictError("write authorization identity mismatch")
            identity.validate_destination(delivery.destination)
            self._cancel_delivery_locked(entry, delivery, "Decode fenced retrieval")
        self._progress_delivery(entry, delivery)
        self._progress_releases()
        return delivery.authorization.fence(identity)

    def fence_delivery(self, key: KVEntryKey, delivery_id: str):
        # Legacy ID-only callers can close a gate, never establish MR safety.
        with self._lock:
            token = (key, delivery_id)
            entry = self.entries.get(key)
            delivery = entry.deliveries.get(delivery_id) if entry else None
            if delivery is None and token not in self._fenced_deliveries:
                if len(self._legacy_absent_fences) >= self._max_legacy_absent_fences:
                    raise ResourceExhaustedError(
                        "legacy absent fence capacity exceeded"
                    )
                self._legacy_absent_fences.add(token)
            self._fenced_deliveries.add(token)
            if delivery:
                self._cancel_delivery_locked(entry, delivery, "legacy fence")
        if delivery:
            self._progress_delivery(entry, delivery)
            self._progress_releases()
        return {"delivery_id": delivery_id, "fenced": False}

    def release_entry(self, key: KVEntryKey) -> None:
        with self._lock:
            entry = self._entry(key)
            if entry.active_delivery_count:
                raise EntryConflictError(
                    f"entry has {entry.active_delivery_count} active deliveries"
                )
            if entry.state == EntryShardState.STORED:
                entry.state = transition(entry.state, EntryShardState.RELEASING)
            self._release_resources_locked(entry)
        self.progress_transfers()

    def cancel_entry(self, key: KVEntryKey, reason: str) -> None:
        with self._lock:
            entry = self.entries.get(key)
            if entry is None:
                if key not in self._absent_entry_cancellations:
                    if (
                        len(self._absent_entry_cancellations)
                        >= self._max_absent_entry_cancellations
                    ):
                        raise ResourceExhaustedError(
                            "absent Entry cancellation fence capacity exceeded"
                        )
                    self._absent_entry_cancellations.add(key)
                return
            self._cancel_entry_locked(entry, reason)
        self.progress_transfers()

    def _cancel_entry_locked(self, entry, reason):
        for delivery in entry.deliveries.values():
            self._cancel_delivery_locked(entry, delivery, reason)
        if entry.state not in (
            EntryShardState.RELEASED,
            EntryShardState.FAILED,
            EntryShardState.CANCELLED,
            EntryShardState.EXPIRED,
        ):
            entry.state = transition(entry.state, EntryShardState.CANCELLED)
        entry.error = reason
        self._release_resources_locked(entry)

    def _fail_entry_locked(self, entry: EntryShardRecord, reason: str) -> None:
        for delivery in entry.deliveries.values():
            self._cancel_delivery_locked(entry, delivery, reason)
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
        # Never execute unregister callbacks while holding the store lock.
        entry.release_requested = True
        self._release_pending[entry.key] = entry
        # Cancel, failure, TTL and close ask the sender to stop and report.
        # They never release a lifecycle upload's hold on the destination.
        if entry.upload_authorization is not None and entry.upload_terminal is None:
            entry.upload_close_requested = True

    def progress_prompt_indexes(self) -> Dict[str, int]:
        """Build one round of pending Prompt indexes. Bounded, caller-driven.

        Each candidate's allocation is pinned for the copy, so an Entry whose
        release has begun is skipped rather than read from: ResourceGuard.pin
        refuses once release is requested. Extraction happens outside the
        store lock. The pin drops after proven completion, including ordinary
        failure, but remains held if extraction/backend completion is unknown.

        Never raises for a build failure. An Entry that cannot be indexed is
        still deliverable, and delivery does not consult this at all.

        ``deferred`` is reported separately from ``failed``: a build the index
        put off because it had no budget for the copies has not failed and
        has spent none of the Entry's attempts, and reporting it as a failure
        would make ordinary backpressure look like a broken worker.
        """
        if self.prompt_index is None:
            return {"built": 0, "failed": 0, "deferred": 0, "skipped": 0}
        candidates = []
        skipped = 0
        with self._lock:
            for entry in self._live_entries.values():
                transfer_id = entry.key.transfer_id
                if entry.state != EntryShardState.STORED or entry.release_requested:
                    continue
                if not self.prompt_index.wants_build(transfer_id):
                    continue
                owner = f"prompt-index:{transfer_id}:{uuid.uuid4().hex[:8]}"
                try:
                    entry.allocation_guard.pin(owner)
                except Exception:
                    # Release has already begun; its pages are not ours to read.
                    skipped += 1
                    continue
                offset = entry.allocation.start_page * self.page_bytes
                packed = self.pool[offset : offset + entry.manifest.expected_bytes]
                gate = self.prompt_index.gate_for(transfer_id)
                deferrals = 0 if gate is None else gate.deferrals
                candidates.append((entry, owner, packed, deferrals))
        built = failed = deferred = 0
        for entry, owner, packed, deferrals_before in candidates:
            try:
                ok = self.prompt_index.build(
                    entry.key.transfer_id,
                    packed,
                    layout=entry.layout,
                    manifest=entry.manifest,
                )
            except Exception as exc:
                ok = False
                logger.warning(
                    "V prompt index build refused for %s: %s", entry.key, exc
                )
            finally:
                if getattr(self.prompt_index, "quarantined", False):
                    # An unfinished CUDA extraction may still read the Entry.
                    # Keep the real allocation pin, not merely the tensor view.
                    with self._lock:
                        self._quarantined_index_sources.append((entry, owner, packed))
                else:
                    entry.allocation_guard.unpin(owner)
            if ok:
                built += 1
                continue
            gate = self.prompt_index.gate_for(entry.key.transfer_id)
            if gate is not None and gate.deferrals > deferrals_before:
                deferred += 1
            else:
                failed += 1
        return {
            "built": built,
            "failed": failed,
            "deferred": deferred,
            "skipped": skipped,
        }

    def _free_allocation(self, entry):
        if self.prompt_index is not None:
            # Stop serving this Entry's index before its pages go back to the
            # allocator. This drops the index's own copies only; the pages,
            # registration and MR are released by their existing owners.
            self.prompt_index.close(entry.key.transfer_id)
        with self._lock:
            if not entry.resources_released:
                self.allocator.free(entry.allocation)
                entry.resources_released = True
                if entry.state == EntryShardState.RELEASING:
                    entry.state = transition(entry.state, EntryShardState.RELEASED)
                self._refresh_metrics()

    def _progress_releases(self):
        with self._lock:
            entries = list(self._release_pending.values())
        for entry in entries:
            try:
                entry.allocation_guard.request_release()
                # The allocation callback can mark resources_released before
                # ResourceGuard has finished it. Do not drop the pool's MR pin
                # until the callback itself has returned successfully.
                if (
                    not entry.resources_released
                    or entry.allocation_guard.value is not None
                ):
                    continue
                outcome = self._pool_guard.unpin(entry.pool_owner)
                if outcome == GuardUnpinOutcome.RELEASE_IN_PROGRESS:
                    continue
                with self._lock:
                    if self._release_pending.get(entry.key) is entry:
                        self._release_pending.pop(entry.key)
                        self._live_entries.pop(entry.key, None)
            except Exception as exc:
                logger.warning(
                    "V allocation release retained resources: %s: %s", entry.key, exc
                )
        if self._closed:
            try:
                self._pool_guard.request_release()
            except Exception as exc:
                logger.warning("V pool unregister will be retried: %s", exc)

    def reap_expired(
        self, now: Optional[float] = None, *, reap_entries: bool = True
    ) -> Dict[str, int]:
        now = time.monotonic() if now is None else now
        expired_deliveries = expired_entries = 0
        with self._lock:
            for entry in self._live_entries.values():
                for delivery in entry.deliveries.values():
                    if (
                        delivery.state not in DELIVERY_TERMINAL_STATES
                        and delivery.deadline <= now
                    ):
                        self._cancel_delivery_locked(
                            entry, delivery, "delivery timeout", DeliveryState.EXPIRED
                        )
                        expired_deliveries += 1
                if (
                    reap_entries
                    and not entry.release_requested
                    and entry.active_delivery_count == 0
                    and entry.expires_at <= now
                ):
                    # An ALLOCATED target may already be visible to P even
                    # before begin_p_write. Its upload pin remains intact.
                    if entry.state != EntryShardState.ALLOCATED:
                        entry.state = transition(entry.state, EntryShardState.EXPIRED)
                    entry.error = "entry TTL expired"
                    self._release_resources_locked(entry)
                    expired_entries += 1
        self.progress_transfers()
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
            snapshot = {
                "rank": self.rank,
                "capabilities": [PVD_TRANSFER_CAPABILITY],
                "ready": not self._closed and self._isolated_reason is None,
                "draining": self._closed and not self.allocator.allocated_pages == 0,
                "world_size": self.world_size,
                "rail": self.rail,
                "device": self.device,
                "sparse_packing_mode": (
                    "cuda_synchronous_experimental"
                    if self.allow_cuda_sparse_packing
                    else "cpu_reference_only"
                ),
                "sparse_pack_kernel": (
                    "triton" if self.fused_cuda_sparse_packing else "torch"
                ),
                "page_bytes": self.page_bytes,
                "worker_epoch": self.worker_epoch,
                "full_kv_fanin": {
                    "enabled": self._fanin_max_slices is not None,
                    "max_slices": self._fanin_max_slices,
                    "max_inflight": self._fanin_max_inflight,
                    "native_batch": self._fanin_native_batch,
                    "protocols": (
                        [
                            FULL_KV_FANIN_PROTOCOL,
                            RANK_PACKED_FULL_KV_FANIN_PROTOCOL,
                        ]
                        if self._fanin_max_slices is not None
                        else []
                    ),
                },
                "closed": self._closed,
                "isolated_reason": self._isolated_reason,
                "absent_write_fences": len(self._absent_write_fences),
                "max_absent_write_fences": self._max_absent_write_fences,
                "absent_entry_cancellations": len(self._absent_entry_cancellations),
                "max_absent_entry_cancellations": (
                    self._max_absent_entry_cancellations
                ),
                "pending_release_entries": len(self._release_pending),
                "live_entries": len(self._live_entries),
                "max_entry_records": self._max_entry_records,
                "delivery_records": self._delivery_records,
                "max_delivery_records": self._max_delivery_records,
                "legacy_absent_fences": len(self._legacy_absent_fences),
                "max_legacy_absent_fences": self._max_legacy_absent_fences,
                "total_pages": self.allocator.total_pages,
                "available_pages": self.allocator.available_pages,
                "largest_contiguous_free_pages": (
                    self.allocator.largest_contiguous_free_pages
                ),
                "entries": [entry.to_dict() for entry in self.entries.values()],
                "quarantined_index_sources": len(self._quarantined_index_sources),
                "metrics": self.metrics.snapshot(),
            }
        # Engine health may acquire the native submission manager's lock.
        snapshot["transport"] = self.transfer_engine.health()
        return snapshot

    def capacity_snapshot(
        self, manifest: Optional[KVEntryManifest] = None
    ) -> Dict[str, object]:
        """Small, coherent control-plane preflight; no Entry list or MR data."""
        with self._lock:
            report = {
                "rank": self.rank,
                "worker_epoch": self.worker_epoch,
                "ready": not self._closed and self._isolated_reason is None,
                "total_pages": self.allocator.total_pages,
                "available_pages": self.allocator.available_pages,
                "largest_contiguous_free_pages": (
                    self.allocator.largest_contiguous_free_pages
                ),
            }
            if manifest is not None:
                report["index_admission_room"] = None
                report["index_build_pending"] = None
                index = self.prompt_index
                if index is not None and index.budget is not None:
                    wants_build = getattr(index, "wants_build", None)
                    if callable(wants_build):
                        report["index_build_pending"] = wants_build(
                            manifest.key.transfer_id
                        )
                    shard = manifest.shard(self.rank)
                    rows, dim = manifest.prompt_token_count, manifest.layout.head_dim
                    heads = (
                        shard.layer_end - shard.layer_start
                    ) * manifest.layout.kv_heads_per_rank
                    try:
                        retained = heads * (
                            rows * dim * 4
                            + index.backend.build_footprint(
                                rows, dim, metric=index.metric
                            )
                        )
                        scratch = index.backend.build_scratch_footprint(
                            rows, dim, metric=index.metric
                        )
                        budget = index.budget.snapshot()
                        free = budget["staging_bytes"] - budget["used_staging_bytes"]
                    except (KeyError, TypeError, ValueError):
                        # An unsupported backend/shape must not prevent full
                        # KV storage; the index build will report its failure.
                        pass
                    else:
                        report["index_required_bytes"] = retained + scratch
                        report["index_available_bytes"] = free
                        report["index_admission_room"] = free >= retained + scratch
            return report

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for entry in self.entries.values():
                self._cancel_entry_locked(entry, "V store closing")
        self.progress_transfers()
