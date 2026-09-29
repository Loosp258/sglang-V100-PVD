"""Per-Entry Prompt indexes on a V rank: gates, vectors and search.

This is the first thing that actually drives ``IndexGate``. It owns one gate
per stored Entry shard, extracts that shard's Prompt K with
``prompt_vectors.extract_prompt_k``, builds an index per (layer, KV head)
through a swappable ``IndexBackend``, and answers searches under the gate's
identity checks.

Deliberate properties:

* **Full-Prompt delivery never waits for it.** Bootstrap and the existing full
  refresh path stay independent of index state. Explicit sparse deliveries
  lease a ready index/version while packing their selected K/V bytes.
* **Off unless a backend is supplied.** ``VectorKVStore`` constructs no
  manager by default, so the existing store behaves exactly as before.
* **No autonomous driver.** ``build_pending`` is one bounded step a caller
  invokes, matching how uploads and decode closes are progressed. A worker
  that stops calling it keeps its gates rather than half-building anything.
* **It frees nothing that is not its own.** ``close`` refuses further use and
  drops this manager's own vectors; the Entry's pages, registration and MR
  are released only by their existing owners, after their own proofs.
* **Every copy it retains is charged, before it exists.** Extraction's copy
  and whatever the backend retains are reserved against a worker-level budget
  ahead of the allocation, under per-attempt owners so a late build cannot
  refund a newer one's reservation, and a search's bounded scratch is
  reserved for its duration. Nothing is refunded while a search still holds
  it: a closed Entry's charge outlives the close until the last searcher
  leaves.
* **Capacity pressure is backpressure, not failure.** An Entry is allowed
  only a few build attempts, and "there is no room for the copies right now"
  must not spend them: that attempt is abandoned and retried next round.
* **A search states its own identity.** ``search`` takes a
  ``SearchRequestIdentity`` describing the Q the caller holds, and every
  comparison is against that. Nothing missing from it is filled in from the
  index's own configuration, because an identity derived from the index
  would agree with the index by construction.

The default backend is the exact CPU one, so a V rank can build and search
with no cuVS present. A CAGRA backend replaces it without touching this file.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from sglang.srt.disaggregation.pvd.index_lifecycle import (
    IndexDescriptor,
    IndexGate,
    IndexState,
)
from sglang.srt.disaggregation.pvd.index_search import (
    BruteForceIndexBackend,
    BuiltIndex,
    IdMapping,
    IndexBackend,
    IndexCompletionUnknown,
    IndexNotReadyError,
    IndexSearchError,
    Selection,
    select,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import (
    POSITIONAL_ENCODINGS,
    ROPE_APPLIED,
    PromptKVectors,
    PromptVectorError,
    extract_prompt_k,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferCapacityError

logger = logging.getLogger(__name__)


@dataclass
class EntryIndex:
    """One Entry shard's gate and, once built, its per-head indexes."""

    gate: IndexGate
    id_mapping_version: str
    vectors: Dict[Tuple[int, int], PromptKVectors] = field(default_factory=dict)
    indexes: Dict[Tuple[int, int], BuiltIndex] = field(default_factory=dict)
    #: Budget owners for the copies this entry currently retains: one for the
    #: extracted vectors, one for whatever the backend holds. Per attempt, so
    #: a build that finishes after a retry cannot refund the retry's charge.
    budget_owners: Tuple[str, ...] = ()
    #: Live searches reading this record's tensors. The charge is not refunded
    #: while this is non-zero, even after close(): the memory is still held.
    users: int = 0
    provisional_pages: int = 0
    provisional_owners: List[str] = field(default_factory=list)
    provisional_failed: bool = False
    group_means: Dict[Tuple[int, int], torch.Tensor] = field(default_factory=dict)
    group_boundaries: List[int] = field(default_factory=lambda: [0])

    def heads(self) -> List[Tuple[int, int]]:
        return sorted(self.indexes)


@dataclass(frozen=True)
class SearchRequestIdentity:
    """What the caller claims about the Q it is about to search with.

    Every field is the caller's own assertion. None of it is defaulted from
    the index being searched: an identity the manager filled in would match
    the index by construction and prove nothing about the query. The two
    ``expected_*`` pins are optional because a caller that has never seen a
    version cannot supply one -- and a search that omits them is reported as
    not having checked them, rather than as having passed.
    """

    vector_space: str
    positional_encoding: str
    entry_transfer_id: str
    layer: int
    kv_head: int
    expected_index_version: Optional[str] = None
    expected_id_mapping_version: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("vector_space", "positional_encoding", "entry_transfer_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise IndexSearchError(
                    f"a search request must state its {name}; it is the "
                    "caller's own identity and has no default"
                )
        if self.positional_encoding not in POSITIONAL_ENCODINGS:
            raise IndexSearchError(
                f"positional_encoding must be one of {POSITIONAL_ENCODINGS}, "
                f"got {self.positional_encoding!r}"
            )
        for name in ("layer", "kv_head"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise IndexSearchError(f"{name} must be a non-negative integer")
        for name in ("expected_index_version", "expected_id_mapping_version"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise IndexSearchError(f"{name} must be a non-empty string when given")


@dataclass(frozen=True)
class SearchResult:
    """A selection plus what was actually verified to produce it."""

    selection: Selection
    index_version: str
    id_mapping_version: str
    #: The identity comparisons that were made. A caller that did not pin a
    #: version will not find it named here.
    validated: Tuple[str, ...]


class PromptIndexManager:
    """Builds and searches immutable Prompt indexes for one V rank."""

    def __init__(
        self,
        *,
        vector_space: str,
        backend: Optional[IndexBackend] = None,
        metric: str = "ip",
        positional_encoding: str = ROPE_APPLIED,
        max_build_attempts: int = 3,
        budget: Any = None,
        group_heads: int = 1,
    ) -> None:
        if not isinstance(vector_space, str) or not vector_space.strip():
            raise ValueError("vector_space must be a non-empty string")
        self.vector_space = vector_space
        self.backend = backend or BruteForceIndexBackend()
        self.metric = metric
        self.positional_encoding = positional_encoding
        self.max_build_attempts = max_build_attempts
        self.budget = budget
        if group_heads not in (1, 2, 4):
            raise ValueError("group_heads must be 1, 2 or 4")
        if group_heads > 1 and getattr(self.backend, "name", None) != "cagra":
            raise ValueError("grouped heads require native CAGRA")
        self.group_heads = group_heads

        # Where this manager's copies live: the backend's declared device, so
        # vectors are born where they will be searched. The KV pool is never
        # moved to match; extraction copies across instead.
        self.backend_device = getattr(self.backend, "device", None)
        self._entries: Dict[str, EntryIndex] = {}
        # close() can run from a guard-release callback on a different thread
        # than build()/search(), so the registry is locked like every other
        # per-worker registry here. Heavy work happens outside the lock.
        self._lock = threading.RLock()
        self._quarantine_reason = None
        self._retained_operations = []
        self._retained_records = []
        # A backend-wide native parent limiter is paid for exactly once,
        # before any Entry build. It stays charged while this manager/backend
        # exists: no serving-wide unload operation proves all native children
        # have retired, so an Entry close must never refund the parent cap.
        shared = getattr(self.backend, "shared_footprint", 0)
        if type(shared) is not int or shared < 0:
            raise ValueError("backend shared footprint must be a non-negative integer")
        if shared and budget is None:
            raise ValueError("shared native index footprint requires a budget")
        self.shared_native_budget_bytes = shared
        self._shared_owner = (
            f"prompt-index:shared-native:{uuid.uuid4().hex}" if shared else None
        )
        if shared:
            budget.reserve(self._shared_owner, shared, 0)

    def _group_key(self, layer: int) -> Tuple[int, int]:
        return (layer - layer % (self.group_heads // 2), -1)

    @property
    def quarantined(self):
        with self._lock:
            return self._quarantine_reason is not None

    def _quarantine(self, reason):
        with self._lock:
            if self._quarantine_reason is None:
                self._quarantine_reason = str(reason)

    def _fence(self, *sources):
        if self.quarantined:
            raise IndexCompletionUnknown(self._quarantine_reason)
        try:
            sync = getattr(self.backend, "synchronize", None)
            if callable(sync):
                sync()
            elif (
                self.backend_device is not None
                and torch.device(self.backend_device).type == "cuda"
            ):
                raise IndexCompletionUnknown("CUDA backend lacks completion contract")
            # Extraction may read a CUDA pool even when the index is on CPU.
            devices = {
                x.device
                for x in sources
                if isinstance(x, torch.Tensor) and x.device.type == "cuda"
            }
            for device in devices:
                torch.cuda.synchronize(device)
        except BaseException as exc:
            self._quarantine(exc)
            raise IndexCompletionUnknown(str(exc)) from exc

    def _dispose(self, indexes):
        try:
            dispose = getattr(self.backend, "dispose", None)
            if (
                indexes
                and not callable(dispose)
                and self.backend_device is not None
                and torch.device(self.backend_device).type == "cuda"
            ):
                raise IndexCompletionUnknown("CUDA backend lacks disposal contract")
            for index in indexes.values():
                if isinstance(index, BuiltIndex) and callable(dispose):
                    dispose(index)
            self._fence()
        except BaseException as exc:
            self._quarantine(exc)
            raise IndexCompletionUnknown(str(exc)) from exc

    def _retain_operation(self, *owners):
        with self._lock:
            self._retained_operations.append(owners)

    # -- gates --------------------------------------------------------------

    def gate_for(self, transfer_id: str) -> Optional[IndexGate]:
        with self._lock:
            entry = self._entries.get(transfer_id)
            return entry.gate if entry is not None else None

    def open(self, transfer_id: str) -> IndexGate:
        with self._lock:
            existing = self._entries.get(transfer_id)
            if existing is not None:
                return existing.gate
            record = EntryIndex(
                gate=IndexGate(transfer_id, max_build_attempts=self.max_build_attempts),
                id_mapping_version=f"map:{transfer_id}:{uuid.uuid4().hex[:8]}",
            )
            self._entries[transfer_id] = record
            return record.gate

    # -- budget -------------------------------------------------------------

    def _reserve(self, owner: str, byte_count: int) -> None:
        """Charge bytes before the allocation they pay for exists.

        Index copies never occupy transfer slots, so only bytes are reserved.
        A refusal raises ``TransferCapacityError``, which the build path
        treats as backpressure rather than as a failed build.
        """
        if self.budget is not None:
            self.budget.reserve(owner, byte_count, 0)

    def _release_owners(self, owners: Sequence[str]) -> None:
        """Refund specific owners. Idempotent: releasing twice is a no-op."""
        if self.budget is None:
            return
        for owner in owners:
            self.budget.release(owner)

    def _release_budget_locked(self, record: EntryIndex) -> None:
        """Refund what this entry retains. Idempotent; safe when unset."""
        self._release_owners(record.budget_owners)
        self._release_owners(record.provisional_owners)
        record.budget_owners = ()
        record.provisional_owners.clear()

    def _retire_locked(self, record: EntryIndex) -> None:
        """Drop a detached record's copies once nothing is reading them.

        Called with the lock held, from close() and from the last search to
        leave. Refunding earlier would report memory as free while a search
        still holds a reference to it.
        """
        if (
            record.users > 0
            or self._entries.get(record.gate.entry_transfer_id) is record
        ):
            return
        if self.quarantined:
            if all(r is not record for r in self._retained_records):
                self._retained_records.append(record)
            return
        try:
            self._fence()
            self._dispose(record.indexes)
        except IndexCompletionUnknown as exc:
            self._retained_records.append(record)
            self._retain_operation(record, exc)
            return
        record.vectors.clear()
        record.indexes.clear()
        self._release_budget_locked(record)

    def note_kv_readable(self, transfer_id: str) -> None:
        """The complete Prompt KV is stored and safely visible on this rank."""
        self.open(transfer_id).mark_kv_readable()

    def wants_build(self, transfer_id: str) -> bool:
        if self.quarantined:
            return False
        gate = self.gate_for(transfer_id)
        if gate is None or not gate.kv_readable:
            return False
        with self._lock:
            record = self._entries.get(transfer_id)
            if record is not None and record.provisional_pages:
                return False
        return (
            gate.state in (IndexState.ABSENT, IndexState.FAILED) and not gate.exhausted
        )

    def provisional_pages(self, transfer_id: str) -> int:
        with self._lock:
            record = self._entries.get(transfer_id)
            return 0 if record is None else record.provisional_pages

    def progress_chunked(
        self, transfer_id: str, packed: torch.Tensor, *, layout: Any,
        manifest: Any, complete_pages: int, stored: bool,
    ) -> str:
        """Build/extend only terminal-proven pages; publish only after STORED.

        A failed provisional graph is retired and the ordinary full build may
        retry after STORED. Capacity refusal retains the previous prefix.
        """
        started = time.perf_counter()
        with self._lock:
            if self.quarantined:
                raise IndexCompletionUnknown(self._quarantine_reason)
            record = self._entries.get(transfer_id)
            if record is None:
                raise IndexSearchError(f"no index gate for {transfer_id}")
            if record.gate.searchable:
                return "ready"
            if record.provisional_failed:
                return "fallback"
            if complete_pages <= record.provisional_pages:
                if stored and complete_pages == manifest.page_count and record.provisional_pages:
                    return self._publish_provisional_locked(record, manifest)
                return "unchanged"
            if record.gate.state is IndexState.READY:
                return "ready"
            min_rows = getattr(self.backend, "exact_max_rows", None)
            if min_rows is None:
                min_rows = getattr(self.backend, "intermediate_degree", 0)
            valid_rows = (
                (manifest.page_count - 1) * layout.page_size
                + manifest.last_page_valid_tokens
                if complete_pages == manifest.page_count
                else complete_pages * layout.page_size
            )
            if record.provisional_pages == 0 and valid_rows <= min_rows:
                return "fallback" if stored else "deferred"
            first_page = record.provisional_pages
            owner = f"prompt-index:{transfer_id}:chunk:{uuid.uuid4().hex}"
            scratch_owner = owner + ":scratch"
            index_owner = owner + ":index"
            group_owner = owner + ":group"
            vectors = []
            built = {}
            new_means = {}
            grouped_datasets = []
            try:
                vectors = extract_prompt_k(
                    packed, layout=layout, manifest=manifest,
                    entry_transfer_id=transfer_id,
                    id_mapping_version=record.id_mapping_version,
                    positional_encoding=self.positional_encoding,
                    device=self.backend_device, readable_pages=complete_pages,
                    first_page=first_page, budget=self.budget,
                    budget_owner=owner,
                )
                if not vectors:
                    raise PromptVectorError("chunk produced no Prompt K vectors")
                if self.group_heads > 1:
                    if len(vectors) % self.group_heads:
                        raise IndexSearchError("grouped chunk lacks complete KV heads")
                    self._reserve(group_owner, sum(
                        item.vectors.numel() * item.vectors.element_size()
                        + (item.head_dim * 4 if first_page == 0 else 0)
                        for item in vectors
                    ))
                if first_page == 0:
                    if self.budget is not None:
                        self._reserve(index_owner, sum(
                            self.backend.build_footprint(
                                v.token_count * self.group_heads, v.head_dim,
                                metric=self.metric,
                            )
                            for v in vectors[::self.group_heads]
                        ))
                if self.budget is not None:
                    self._reserve(scratch_owner, max(
                        self.backend.build_scratch_footprint(
                            (valid_rows if first_page else v.token_count)
                            * self.group_heads,
                            v.head_dim, metric=self.metric,
                        ) for v in vectors
                    ))
                if self.group_heads > 1:
                    for group_start in range(0, len(vectors), self.group_heads):
                        pair = sorted(
                            vectors[group_start:group_start + self.group_heads],
                            key=lambda item: (item.layer, item.kv_head),
                        )
                        base_layer = pair[0].layer
                        base_head = pair[0].kv_head - pair[0].kv_head % 2
                        expected_keys = [
                            (base_layer + offset // 2, base_head + offset % 2)
                            for offset in range(self.group_heads)
                        ]
                        if (
                            base_layer % (self.group_heads // 2)
                            or [(item.layer, item.kv_head) for item in pair] != expected_keys
                            or len({item.token_count for item in pair}) != 1
                        ):
                            raise IndexSearchError("grouped chunk has mismatched KV heads")
                        count, dim = pair[0].token_count, pair[0].head_dim
                        grouped = torch.empty(
                            (self.group_heads * count, dim),
                            device=pair[0].vectors.device,
                            dtype=torch.float32,
                        )
                        grouped_datasets.append(grouped)
                        for offset, item in enumerate(pair):
                            key = (item.layer, item.kv_head)
                            mean = (
                                record.group_means[key] if first_page
                                else item.vectors.mean(dim=0)
                            )
                            if first_page == 0:
                                new_means[key] = mean
                            segment = grouped[offset * count:(offset + 1) * count]
                            segment.copy_(item.vectors)
                            segment.sub_(mean)
                        key = self._group_key(base_layer)
                        if first_page:
                            previous = record.indexes[key]
                            built[key] = self.backend.extend(previous, grouped)
                            expected = previous.count + self.group_heads * count
                        else:
                            built[key] = self.backend.build(
                                grouped, vector_space=self.vector_space,
                                metric=self.metric,
                            )
                            expected = self.group_heads * count
                        if (
                            built[key].count != expected
                            or built[key].dim != dim
                            or built[key].vector_space != self.vector_space
                            or built[key].metric != self.metric
                        ):
                            raise IndexSearchError("grouped CAGRA metadata mismatch")
                        self._fence(packed)
                else:
                    for item in vectors:
                        key = (item.layer, item.kv_head)
                        if first_page:
                            previous = record.indexes[key]
                            built[key] = self.backend.extend(previous, item.vectors)
                            if built[key].count != previous.count + item.token_count:
                                raise IndexSearchError("extended CAGRA row count mismatch")
                        else:
                            built[key] = self.backend.build(
                                item.vectors, vector_space=self.vector_space,
                                metric=self.metric,
                            )
                            self._validate_built_index(built[key], item)
                        self._fence(packed)
                self._fence(packed)
            except TransferCapacityError as exc:
                self._fence(packed)
                self._release_owners((owner, scratch_owner, index_owner, group_owner))
                traceback.clear_frames(exc.__traceback__)
                return "deferred"
            except BaseException as exc:
                if isinstance(exc, IndexCompletionUnknown):
                    self._quarantine(exc)
                    self._retain_operation(record, owner, scratch_owner, index_owner, group_owner, vectors, grouped_datasets, built, packed, exc)
                    raise
                try:
                    self._fence(packed)
                    self._dispose({**record.indexes, **built})
                except IndexCompletionUnknown:
                    self._retain_operation(record, owner, scratch_owner, index_owner, group_owner, vectors, grouped_datasets, built, packed, exc)
                    raise
                record.indexes.clear()
                record.vectors.clear()
                record.provisional_pages = 0
                record.provisional_failed = True
                record.group_means.clear()
                record.group_boundaries = [0]
                self._release_owners((owner, scratch_owner, index_owner, group_owner))
                self._release_owners(record.provisional_owners)
                record.provisional_owners.clear()
                traceback.clear_frames(exc.__traceback__)
                logger.warning("PVD provisional index failed for %s: %s", transfer_id, exc)
                return "fallback" if stored else "failed"
            self._release_owners((scratch_owner,))
            record.provisional_owners.append(owner)
            if self.group_heads > 1:
                record.provisional_owners.append(group_owner)
                record.group_means.update(new_means)
                record.group_boundaries.append(valid_rows)
            if first_page == 0:
                record.provisional_owners.append(index_owner)
            record.indexes.update(built)
            record.vectors = {(v.layer, v.kv_head): v for v in vectors}
            record.provisional_pages = complete_pages
            if stored and complete_pages == manifest.page_count:
                logger.info(
                    "PVD Prompt provisional final step: transfer_id=%s rank=%d "
                    "first_page=%d pages=%d heads=%d seconds=%.6f",
                    transfer_id, manifest.rank, first_page, complete_pages,
                    len(built), time.perf_counter() - started,
                )
                return self._publish_provisional_locked(record, manifest)
            logger.info(
                "PVD Prompt provisional graph: transfer_id=%s rank=%d pages=%d first_page=%d "
                "heads=%d outcome=%s seconds=%.6f",
                transfer_id, manifest.rank, complete_pages, first_page, len(built),
                "extended" if first_page else "built_prefix",
                time.perf_counter() - started,
            )
            return "extended" if first_page else "built_prefix"

    def _publish_provisional_locked(self, record: EntryIndex, manifest: Any) -> str:
        if not record.gate.kv_readable or record.provisional_pages != manifest.page_count:
            raise IndexSearchError("provisional graph cannot be published before complete KV")
        count = (manifest.page_count - 1) * record.vectors[next(iter(record.vectors))].mapping.page_size + manifest.last_page_valid_tokens
        for key, item in list(record.vectors.items()):
            index = (
                record.indexes[self._group_key(key[0])]
                if self.group_heads > 1 else record.indexes[key]
            )
            if index.count != count * self.group_heads:
                raise IndexSearchError("provisional graph is missing Prompt rows")
            mapping = IdMapping(
                version=record.id_mapping_version,
                token_ids=tuple(range(count)),
                page_size=item.mapping.page_size,
            )
            record.vectors[key] = PromptKVectors(
                item.entry_transfer_id, item.layer, item.kv_head,
                item.vectors, mapping, item.positional_encoding,
                item.source_dtype,
            )
        if self.group_heads > 1 and record.group_boundaries[-1] != count:
            raise IndexSearchError("grouped graph boundaries are incomplete")
        record.gate.begin_build()
        record.budget_owners = tuple(record.provisional_owners)
        record.provisional_owners.clear()
        record.gate.mark_ready(IndexDescriptor(
            index_version=f"idx:{record.gate.entry_transfer_id}:{uuid.uuid4().hex[:12]}",
            entry_transfer_id=record.gate.entry_transfer_id,
            vector_space=self.vector_space,
            id_mapping_version=record.id_mapping_version,
            vector_count=sum(index.count for index in record.indexes.values()),
            metric=self.metric,
        ))
        logger.info(
            "PVD Prompt provisional graph READY: transfer_id=%s rank=%d pages=%d "
            "heads=%d rows_per_head=%d",
            record.gate.entry_transfer_id, manifest.rank, record.provisional_pages,
            len(record.indexes), count,
        )
        return "ready"

    # -- building -----------------------------------------------------------

    def _validate_built_index(self, index: BuiltIndex, item: PromptKVectors) -> None:
        """A backend call returning is not proof it built the requested index.

        Keep this inside build's failure/refund scope, before publishing any
        record. Otherwise malformed metadata can strand the gate in BUILDING
        or authorize an index from another vector space/metric as READY.
        This validates metadata, not an opaque native handle's contents.
        """
        if not isinstance(index, BuiltIndex):
            raise IndexSearchError("backend built index must be a BuiltIndex")
        if (
            type(index.count) is not int
            or type(index.dim) is not int
            or (index.count, index.dim) != tuple(item.vectors.shape)
            or index.vector_space != self.vector_space
            or index.metric != self.metric
        ):
            raise IndexSearchError(
                "backend built index metadata differs from requested vectors/space/metric"
            )

    def build(
        self,
        transfer_id: str,
        packed: torch.Tensor,
        *,
        layout: Any,
        manifest: Any,
    ) -> bool:
        """Extract and index one stored shard. Returns whether it is searchable.

        A failure is recorded on the gate and reported, never raised: an Entry
        that cannot be indexed is still perfectly deliverable. A *refusal for
        lack of budget* is not even recorded as a failure: the attempt is
        abandoned and the Entry stays a candidate for the next round.
        """
        started = time.perf_counter()
        with self._lock:
            if self.quarantined:
                raise IndexCompletionUnknown(self._quarantine_reason)
            record = self._entries.get(transfer_id)
            if record is None:
                raise IndexSearchError(f"no index gate for {transfer_id}")
            record.gate.begin_build()
            # Whatever the previous attempt retained is gone as of now.
            self._release_budget_locked(record)
            # Per-attempt owners. A build that completes after this Entry has
            # been closed and rebuilt refunds its own charge, never the new
            # attempt's, and two owners let the vectors and the backend's own
            # storage be charged and refunded independently.
            # One token per attempt, and it is what names the resulting
            # index. Deriving the version from the gate's attempt counter
            # repeated it across incarnations: close and reopen resets
            # attempts to 1, so a rebuilt index could be handed the version
            # a caller had pinned against the *previous* one and pass.
            attempt_token = uuid.uuid4().hex[:12]
            epoch = f"prompt-index:{transfer_id}:{attempt_token}"
            vectors_owner = f"{epoch}:vectors"
            index_owner = f"{epoch}:index"
            scratch_owner = f"{epoch}:build-scratch"
            group_owner = f"{epoch}:group"
            mapping_version = record.id_mapping_version

        owners = (vectors_owner, index_owner, scratch_owner, group_owner)
        vectors = []
        build_inputs = []
        group_means = {}
        built = {}
        item = None
        extracted_at = None
        head_build_seconds = []
        backend_path = getattr(self.backend, "name", type(self.backend).__name__)
        # Extraction and the backend build run outside the lock: they copy and
        # index the whole shard, and close() must not block behind them.
        try:
            vectors = extract_prompt_k(
                packed,
                layout=layout,
                manifest=manifest,
                entry_transfer_id=transfer_id,
                id_mapping_version=mapping_version,
                positional_encoding=self.positional_encoding,
                device=self.backend_device,
                budget=self.budget,
                budget_owner=vectors_owner,
            )
            if not vectors:
                raise PromptVectorError("the shard produced no Prompt K vectors")
            extracted_at = time.perf_counter()
            if self.group_heads > 1:
                if len(vectors) % self.group_heads:
                    raise IndexSearchError("stored shard lacks complete grouped KV heads")
                self._reserve(group_owner, sum(
                    item.vectors.numel() * item.vectors.element_size()
                    + item.head_dim * 4 for item in vectors
                ))
                for first in range(0, len(vectors), self.group_heads):
                    group = sorted(
                        vectors[first:first + self.group_heads],
                        key=lambda item: (item.layer, item.kv_head),
                    )
                    base_layer = group[0].layer
                    base_head = group[0].kv_head - group[0].kv_head % 2
                    expected_keys = [
                        (base_layer + offset // 2, base_head + offset % 2)
                        for offset in range(self.group_heads)
                    ]
                    if (
                        base_layer % (self.group_heads // 2)
                        or [(item.layer, item.kv_head) for item in group] != expected_keys
                        or len({item.token_count for item in group}) != 1
                    ):
                        raise IndexSearchError("stored shard has mismatched grouped heads")
                    count, dim = group[0].token_count, group[0].head_dim
                    grouped = torch.empty(
                        (self.group_heads * count, dim),
                        device=group[0].vectors.device, dtype=torch.float32,
                    )
                    build_inputs.append((self._group_key(base_layer), grouped))
                    for offset, member in enumerate(group):
                        key = (member.layer, member.kv_head)
                        mean = member.vectors.mean(dim=0)
                        group_means[key] = mean
                        grouped[offset * count:(offset + 1) * count].copy_(
                            member.vectors - mean
                        )
            else:
                build_inputs = [
                    ((item.layer, item.kv_head), item.vectors) for item in vectors
                ]
            build_path = getattr(self.backend, "build_path", None)
            if callable(build_path):
                backend_path = build_path(int(vectors[0].vectors.shape[0]))
            # The backend's own storage is charged before it is allocated,
            # not discovered afterwards, and retained bytes are charged
            # separately from the transient peak a build passes through:
            # they have different lifetimes, so one reservation cannot
            # describe both without lying about one of them.
            if self.budget is not None:
                shapes = [tuple(value.shape) for _, value in build_inputs]
                self._reserve(
                    index_owner,
                    sum(
                        self.backend.build_footprint(rows, dim, metric=self.metric)
                        for rows, dim in shapes
                    ),
                )
                # Builds run one after another, so only the largest single
                # build's scratch is ever live. It is released below.
                self._reserve(
                    scratch_owner,
                    max(
                        self.backend.build_scratch_footprint(
                            rows, dim, metric=self.metric
                        )
                        for rows, dim in shapes
                    ),
                )
            for key, value in build_inputs:
                head_started = time.perf_counter()
                built[key] = self.backend.build(
                    value, vector_space=self.vector_space, metric=self.metric
                )
                if (
                    not isinstance(built[key], BuiltIndex)
                    or built[key].count != value.shape[0]
                    or built[key].dim != value.shape[1]
                    or built[key].vector_space != self.vector_space
                    or built[key].metric != self.metric
                ):
                    raise IndexSearchError("CAGRA build metadata mismatch")
                # The reservation is max(per-head scratch), not their sum.
                # Python return alone does not make native builds sequential.
                self._fence(packed)
                head_build_seconds.append(time.perf_counter() - head_started)
            self._fence(packed)
            # The transient peak is over; give it back before the index is
            # installed, so an idle index is charged only for what it holds.
            self._release_owners((scratch_owner,))
        except TransferCapacityError as exc:
            try:
                self._fence(packed)
                self._dispose(built)
            except IndexCompletionUnknown:
                self._retain_operation(
                    record, owners, vectors, build_inputs, built, item, packed, exc
                )
                raise
            # Completed Python frames can keep allocations alive via traceback
            # locals. Drop our references AND those frames before advertising
            # capacity. This is not a CUDA/native completion fence.
            traceback.clear_frames(exc.__traceback__)
            item = None
            vectors.clear()
            built.clear()
            # Backpressure, not a failed build. Nothing was retained, so the
            # attempt is given back and this Entry is tried again next round.
            with self._lock:
                self._release_owners(owners)
                if self._entries.get(transfer_id) is record:
                    record.gate.abandon_build(str(exc))
            return False
        except BaseException as exc:
            if isinstance(exc, IndexCompletionUnknown):
                self._quarantine(exc)
            try:
                self._fence(packed)
                self._dispose(built)
            except IndexCompletionUnknown:
                self._retain_operation(
                    record, owners, vectors, build_inputs, built, item, packed, exc
                )
                raise
            traceback.clear_frames(exc.__traceback__)
            item = None
            vectors.clear()
            built.clear()
            with self._lock:
                # Refund whatever this attempt reserved before the failure: a
                # permanently failing Entry must not hold the worker's budget
                # until restart. Unknown owners release as no-ops.
                self._release_owners(owners)
                if self._entries.get(transfer_id) is record:
                    record.gate.mark_failed(str(exc))
            if not isinstance(exc, Exception):
                raise
            return False

        with self._lock:
            if self.quarantined:
                # Another operation may have failed after our final fence.
                # Never publish READY after the worker's quarantine boundary.
                self._retain_operation(record, owners, vectors, build_inputs, built, item, packed)
                raise IndexCompletionUnknown(self._quarantine_reason)
            if self._entries.get(transfer_id) is not record:
                # Closed while we were building. Drop what we made and refund.
                try:
                    self._dispose(built)
                except IndexCompletionUnknown:
                    self._retain_operation(record, owners, vectors, build_inputs, built, item, packed)
                    raise
                item = None
                vectors.clear()
                built.clear()
                self._release_owners(owners)
                return False
            record.budget_owners = (
                (vectors_owner, index_owner, group_owner)
                if self.group_heads > 1 else (vectors_owner, index_owner)
            )
            record.vectors = {(v.layer, v.kv_head): v for v in vectors}
            record.indexes = built
            if self.group_heads > 1:
                record.group_means = group_means
                record.group_boundaries = [0, vectors[0].token_count]
            record.gate.mark_ready(
                IndexDescriptor(
                    index_version=f"idx:{transfer_id}:{attempt_token}",
                    entry_transfer_id=transfer_id,
                    vector_space=self.vector_space,
                    id_mapping_version=mapping_version,
                    vector_count=sum(index.count for index in built.values()),
                    metric=self.metric,
                )
            )
        logger.info(
            "PVD Prompt index ready: transfer_id=%s backend=%s path=%s heads=%d "
            "rows_per_head=%d extract_seconds=%.6f "
            "head_build_total_seconds=%.6f max_head_seconds=%.6f "
            "total_seconds=%.6f",
            transfer_id,
            getattr(self.backend, "name", type(self.backend).__name__),
            backend_path,
            len(vectors),
            int(vectors[0].vectors.shape[0]),
            extracted_at - started,
            sum(head_build_seconds),
            max(head_build_seconds),
            time.perf_counter() - started,
        )
        return True

    # -- searching ----------------------------------------------------------

    def search(
        self,
        identity: SearchRequestIdentity,
        *,
        queries: torch.Tensor,
        top_k: int,
        timings: Optional[Dict[str, float]] = None,
    ) -> SearchResult:
        """Search one (layer, KV head) under the caller's stated identity.

        Everything the caller claims is checked against what was actually
        built -- vector space, Entry, layer and KV head, positional-encoding
        semantics, head dimension, and whichever versions the caller pinned --
        and all of it before the backend is invoked. A query that does not
        belong to this index is refused, not answered.
        """
        if not isinstance(identity, SearchRequestIdentity):
            raise IndexSearchError(
                "a search must carry a SearchRequestIdentity describing the "
                "query; identity is not inferable from the index"
            )
        if not isinstance(queries, torch.Tensor) or queries.ndim != 2:
            raise IndexSearchError(
                "queries must be a 2-D [num_queries, head_dim] tensor"
            )
        transfer_id = identity.entry_transfer_id
        key = (identity.layer, identity.kv_head)
        stage_started = time.perf_counter() if timings is not None else 0.0
        with self._lock:
            if self.quarantined:
                raise IndexCompletionUnknown(self._quarantine_reason)
            record = self._entries.get(transfer_id)
            if record is None:
                raise IndexSearchError(f"no index gate for {transfer_id}")
            # The caller's vector space, not this manager's. Refuses unless
            # READY, and compares only the versions the caller actually pinned.
            if not record.gate.searchable:
                message = (
                    f"index for {transfer_id} is {record.gate.state.value}, not ready"
                )
                if record.gate.exhausted or record.gate.state is IndexState.CLOSED:
                    raise IndexSearchError(message)
                raise IndexNotReadyError(message)
            descriptor, validated = record.gate.authorize_search(
                identity.vector_space,
                expected_id_mapping_version=identity.expected_id_mapping_version,
                expected_index_version=identity.expected_index_version,
                entry_transfer_id=transfer_id,
            )
            item = record.vectors.get(key)
            group_index = (
                record.indexes.get(self._group_key(identity.layer))
                if self.group_heads > 1 else None
            )
            grouped = group_index is not None
            index = group_index if grouped else record.indexes.get(key)
            group_mean = record.group_means.get(key) if grouped else None
            group_boundaries = tuple(record.group_boundaries) if grouped else ()
            if item is None or index is None:
                raise IndexSearchError(
                    f"entry {transfer_id} has no index for layer "
                    f"{identity.layer} KV head {identity.kv_head}"
                )
            # Registered as a reader before the lock is dropped, so a close()
            # racing this search leaves the copies -- and their charge -- in
            # place until this search returns them.
            record.users += 1
        if timings is not None:
            timings["identity_lease"] = time.perf_counter() - stage_started
        scratch_owner = f"prompt-index-search:{transfer_id}:{uuid.uuid4().hex[:8]}"
        failure = None
        placed_queries = queries
        try:
            # Encoding semantics and head dimension are the last two identity
            # checks, and like the rest they precede any backend call.
            item.require_compatible_query(
                positional_encoding=identity.positional_encoding,
                head_dim=int(queries.shape[-1]),
            )
            validated = validated + ("positional_encoding", "layer", "kv_head")
            # A search's scratch is bounded and charged for its duration, so
            # concurrent searches cannot together exceed the worker's budget.
            stage_started = time.perf_counter() if timings is not None else 0.0
            if self.budget is not None:
                self._reserve(
                    scratch_owner,
                    self.backend.search_footprint(
                        index.count, index.dim, int(queries.shape[0]), int(top_k)
                    ) + (
                        ((index.count + 31) // 32) * 4
                        + int(queries.shape[0]) * int(top_k) * 24
                        if grouped else 0
                    ),
                )
            if timings is not None:
                timings["reserve"] = time.perf_counter() - stage_started
            # HTTP queries arrive on CPU, whereas native CAGRA (and the
            # exact CUDA fallback) need the V rank's own GPU. Place only
            # after identity checks and the backend's search reservation:
            # the declared footprint includes the query copy, and an
            # unauthorized or over-budget request must not allocate it.
            stage_started = time.perf_counter() if timings is not None else 0.0
            if (
                self.backend_device is not None
                and placed_queries.device != torch.device(self.backend_device)
            ):
                placed_queries = queries.to(device=self.backend_device)
            if timings is not None:
                timings["query_place"] = time.perf_counter() - stage_started
            if grouped:
                selection = self._select_grouped(
                    index, placed_queries, identity=identity,
                    mapping=item.mapping, top_k=top_k,
                    mean=group_mean, boundaries=group_boundaries,
                    timings=timings,
                )
            else:
                selection = select(
                    self.backend,
                    index,
                    placed_queries,
                    layer=identity.layer,
                    kv_head=identity.kv_head,
                    mapping=item.mapping,
                    top_k=top_k,
                    timings=timings,
                )
        except BaseException as exc:
            failure = exc
            if isinstance(exc, IndexCompletionUnknown):
                self._quarantine(exc)
            raise
        finally:
            try:
                stage_started = time.perf_counter() if timings is not None else 0.0
                self._fence(placed_queries)
                if timings is not None:
                    timings["completion_fence"] = time.perf_counter() - stage_started
            except IndexCompletionUnknown:
                self._retain_operation(
                    record, scratch_owner, item, index, queries, placed_queries, failure
                )
                raise
            else:
                if failure is not None:
                    traceback.clear_frames(failure.__traceback__)
                item = index = placed_queries = None
                self._release_owners((scratch_owner,))
                with self._lock:
                    record.users -= 1
                    self._retire_locked(record)
        return SearchResult(
            selection=selection,
            index_version=descriptor.index_version,
            id_mapping_version=descriptor.id_mapping_version,
            validated=validated,
        )

    def _select_grouped(
        self, index, queries, *, identity, mapping, top_k, mean,
        boundaries, timings,
    ) -> Selection:
        if mean is None or len(boundaries) < 2:
            raise IndexSearchError("grouped CAGRA index lacks a head mean or mapping")
        if top_k > len(mapping) or index.count != self.group_heads * len(mapping):
            raise IndexSearchError("grouped CAGRA row count or Top-K is invalid")
        local_head = (
            identity.layer % (self.group_heads // 2) * 2
            + identity.kv_head % 2
        )
        words = [0] * ((index.count + 31) // 32)
        sections = []
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            start = self.group_heads * begin + local_head * (end - begin)
            sections.append((begin, end, start))
            for row in range(start, start + end - begin):
                words[row // 32] |= 1 << (row % 32)
        bitset = torch.tensor(
            words, dtype=torch.uint32, device=queries.device,
        )
        rows, scores = self.backend.search(
            index, queries, top_k=top_k, bitset=bitset,
        )
        global_rows = rows.to(dtype=torch.int64)
        local_rows = torch.full_like(global_rows, -1)
        valid = torch.zeros_like(global_rows, dtype=torch.bool)
        for begin, end, start in sections:
            in_section = (global_rows >= start) & (
                global_rows < start + end - begin
            )
            local_rows = torch.where(
                in_section, begin + global_rows - start, local_rows,
            )
            valid |= in_section
        if not bool(torch.all(valid)):
            raise IndexSearchError("filtered CAGRA returned an ID from another head")
        original_scores = scores + (queries @ mean).unsqueeze(1)
        head_index = BuiltIndex(
            index.vector_space, index.metric, index.dim,
            len(mapping), index.handle,
        )
        return select(
            self.backend, head_index, queries,
            layer=identity.layer, kv_head=identity.kv_head,
            mapping=mapping, top_k=top_k, timings=timings,
            backend_result=(local_rows, original_scores),
        )

    def search_many(
        self,
        requests: Sequence[Tuple[SearchRequestIdentity, torch.Tensor, int]],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SearchResult, ...]:
        """Search one Entry batch under one lease, reservation and fence.

        A uniform small IP batch may use the backend's grouped exact path.
        Everything else keeps the ordinary per-item backend, but still gets
        an atomic admission/lifetime boundary. No device allocation or
        backend call occurs until every caller identity has been checked.
        """
        if not 2 <= len(requests) <= 64:
            raise IndexSearchError("search_many requires 2..64 requests")
        first_identity = requests[0][0]
        if not isinstance(first_identity, SearchRequestIdentity):
            raise IndexSearchError("each search needs a caller identity")
        transfer_id = first_identity.entry_transfer_id
        for identity, queries, top_k in requests:
            if not isinstance(identity, SearchRequestIdentity):
                raise IndexSearchError("each search needs a caller identity")
            if identity.entry_transfer_id != transfer_id:
                raise IndexSearchError("search batch must use one Entry")
            if (
                not isinstance(queries, torch.Tensor)
                or queries.ndim != 2
                or queries.shape[0] == 0
            ):
                raise IndexSearchError("queries must have non-empty 2-D shape")
            if type(top_k) is not int or top_k <= 0:
                raise IndexSearchError("top_k must be a positive integer")
        if self.group_heads > 1:
            if metadata is not None:
                metadata["path"] = "grouped_cagra"
            return tuple(
                self.search(identity, queries=queries, top_k=top_k)
                for identity, queries, top_k in requests
            )

        prepared = []
        with self._lock:
            if self.quarantined:
                raise IndexCompletionUnknown(self._quarantine_reason)
            record = self._entries.get(transfer_id)
            if record is None:
                raise IndexSearchError(f"no index gate for {transfer_id}")
            if not record.gate.searchable:
                message = (
                    f"index for {transfer_id} is {record.gate.state.value}, not ready"
                )
                if record.gate.exhausted or record.gate.state is IndexState.CLOSED:
                    raise IndexSearchError(message)
                raise IndexNotReadyError(message)
            for identity, queries, top_k in requests:
                descriptor, validated = record.gate.authorize_search(
                    identity.vector_space,
                    expected_id_mapping_version=identity.expected_id_mapping_version,
                    expected_index_version=identity.expected_index_version,
                    entry_transfer_id=transfer_id,
                )
                item = record.vectors.get((identity.layer, identity.kv_head))
                index = record.indexes.get((identity.layer, identity.kv_head))
                if item is None or index is None:
                    raise IndexSearchError(
                        f"entry {transfer_id} has no index for layer "
                        f"{identity.layer} KV head {identity.kv_head}"
                    )
                item.require_compatible_query(
                    positional_encoding=identity.positional_encoding,
                    head_dim=int(queries.shape[-1]),
                )
                if top_k > index.count:
                    raise IndexSearchError("top_k exceeds indexed vector count")
                prepared.append(
                    (
                        identity,
                        queries,
                        top_k,
                        descriptor,
                        validated + ("positional_encoding", "layer", "kv_head"),
                        item,
                        index,
                    )
                )
            # One batch-wide reader holds every index and mapping through the
            # final device fence, even if close() removes this Entry meanwhile.
            record.users += 1

        scratch_owner = (
            f"prompt-index-search-batch:{transfer_id}:{uuid.uuid4().hex[:8]}"
        )
        placed_queries = []
        raw_results = None
        failure = None
        try:
            indexes = tuple(row[6] for row in prepared)
            num_queries = int(prepared[0][1].shape[0])
            top_k = prepared[0][2]
            supports_grouped = getattr(self.backend, "supports_grouped_exact", None)
            grouped = (
                callable(supports_grouped)
                and all(
                    row[1].shape[0] == num_queries and row[2] == top_k
                    for row in prepared
                )
                and supports_grouped(indexes, num_queries=num_queries, top_k=top_k)
            )
            if metadata is not None:
                metadata["path"] = "grouped_exact" if grouped else "individual"
            if grouped:
                scratch_bytes = self.backend.grouped_exact_footprint(
                    indexes, num_queries=num_queries, top_k=top_k
                )
            else:
                scratch_bytes = sum(
                    self.backend.search_footprint(
                        row[6].count, row[6].dim, int(row[1].shape[0]), row[2]
                    )
                    for row in prepared
                )
            if self.budget is not None:
                self._reserve(scratch_owner, scratch_bytes)
            for row in prepared:
                query = row[1]
                if self.backend_device is not None and query.device != torch.device(
                    self.backend_device
                ):
                    query = query.to(device=self.backend_device)
                placed_queries.append(query)
            if grouped:
                raw_results = self.backend.search_grouped_exact(
                    indexes, placed_queries, top_k=top_k
                )
            selections = []
            for n, row in enumerate(prepared):
                identity, _, item_top_k, descriptor, validated, item, index = row
                selections.append(
                    SearchResult(
                        selection=select(
                            self.backend,
                            index,
                            placed_queries[n],
                            layer=identity.layer,
                            kv_head=identity.kv_head,
                            mapping=item.mapping,
                            top_k=item_top_k,
                            backend_result=raw_results[n] if grouped else None,
                        ),
                        index_version=descriptor.index_version,
                        id_mapping_version=descriptor.id_mapping_version,
                        validated=validated,
                    )
                )
        except BaseException as exc:
            failure = exc
            if isinstance(exc, IndexCompletionUnknown):
                self._quarantine(exc)
            raise
        finally:
            try:
                self._fence(*placed_queries)
            except IndexCompletionUnknown:
                self._retain_operation(
                    record,
                    scratch_owner,
                    prepared,
                    requests,
                    placed_queries,
                    raw_results,
                    failure,
                )
                raise
            else:
                if failure is not None:
                    traceback.clear_frames(failure.__traceback__)
                raw_results = placed_queries = prepared = indexes = item = index = None
                self._release_owners((scratch_owner,))
                with self._lock:
                    record.users -= 1
                    self._retire_locked(record)
        return tuple(selections)

    @contextmanager
    def pin_selection(self, manifest):
        """Validate and lease an immutable index/mapping during sparse packing.

        The caller separately pins the source Entry bytes. Closing/rebuilding
        the index while this lease exists cannot refund or replace its held
        record. The copied payload needs no index lease after packing ends.
        This is selection freshness validation, not destination authorization.
        """
        from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest

        if not isinstance(manifest, SparseDeliveryManifest):
            raise IndexSearchError("explicit sparse delivery manifest required")
        first = manifest.specs[0]
        with self._lock:
            if self.quarantined:
                raise IndexCompletionUnknown(self._quarantine_reason)
            record = self._entries.get(first.entry_transfer_id)
            if record is None or not record.gate.searchable:
                raise IndexSearchError("sparse source index is not ready")
            descriptor, _ = record.gate.authorize_search(
                self.vector_space,
                expected_index_version=first.index_version,
                expected_id_mapping_version=first.id_mapping_version,
                entry_transfer_id=first.entry_transfer_id,
            )
            for spec in manifest.specs:
                item = record.vectors.get((spec.layer, spec.kv_head))
                if item is None or (item.source_dtype, item.head_dim) != (
                    manifest.dtype,
                    manifest.head_dim,
                ):
                    raise IndexSearchError(
                        "sparse group dtype/head does not match built index"
                    )
                if item.mapping.version != spec.id_mapping_version or any(
                    token not in item.mapping.token_ids for token in spec.token_ids
                ):
                    raise IndexSearchError(
                        "sparse token selection is outside the current mapping"
                    )
            record.users += 1
        try:
            yield descriptor
        finally:
            item = None
            with self._lock:
                record.users -= 1
                self._retire_locked(record)

    # -- teardown -----------------------------------------------------------

    def close(self, transfer_id: str) -> None:
        """Stop serving this Entry's index and drop this manager's vectors.

        The Entry's pages, registration and MR are untouched: they are freed
        by their existing owners, after their own safety checks.
        """
        with self._lock:
            record = self._entries.pop(transfer_id, None)
            if record is None:
                return
            record.gate.close()
            # Refund the vector copies this manager made -- but only once no
            # search still holds them. A search in flight keeps its reference
            # alive, so reporting the bytes as free here would let the worker
            # over-commit by exactly the amount still in use. The last search
            # to leave retires the record instead.
            self._retire_locked(record)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            records = list(self._entries.items())
        return {
            "vector_space": self.vector_space,
            "backend": getattr(self.backend, "name", "unknown"),
            "device": str(self.backend_device) if self.backend_device else None,
            "metric": self.metric,
            "quarantined": self.quarantined,
            "quarantine_reason": self._quarantine_reason,
            "retained_operations": len(self._retained_operations),
            "retained_records": len(self._retained_records),
            "positional_encoding": self.positional_encoding,
            "budget": (None if self.budget is None else self.budget.snapshot()),
            "entries": {
                transfer_id: {
                    **record.gate.snapshot(),
                    "indexed_heads": len(record.indexes),
                    "active_searches": record.users,
                }
                for transfer_id, record in records
            },
        }
