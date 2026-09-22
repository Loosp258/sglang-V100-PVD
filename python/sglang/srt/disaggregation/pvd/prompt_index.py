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

import threading
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
    IndexBackend,
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
    ) -> None:
        if not isinstance(vector_space, str) or not vector_space.strip():
            raise ValueError("vector_space must be a non-empty string")
        self.vector_space = vector_space
        self.backend = backend or BruteForceIndexBackend()
        self.metric = metric
        self.positional_encoding = positional_encoding
        self.max_build_attempts = max_build_attempts
        self.budget = budget
        # Where this manager's copies live: the backend's declared device, so
        # vectors are born where they will be searched. The KV pool is never
        # moved to match; extraction copies across instead.
        self.backend_device = getattr(self.backend, "device", None)
        self._entries: Dict[str, EntryIndex] = {}
        # close() can run from a guard-release callback on a different thread
        # than build()/search(), so the registry is locked like every other
        # per-worker registry here. Heavy work happens outside the lock.
        self._lock = threading.RLock()

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
        record.budget_owners = ()

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
        record.vectors.clear()
        record.indexes.clear()
        self._release_budget_locked(record)

    def note_kv_readable(self, transfer_id: str) -> None:
        """The complete Prompt KV is stored and safely visible on this rank."""
        self.open(transfer_id).mark_kv_readable()

    def wants_build(self, transfer_id: str) -> bool:
        gate = self.gate_for(transfer_id)
        if gate is None or not gate.kv_readable:
            return False
        return (
            gate.state in (IndexState.ABSENT, IndexState.FAILED) and not gate.exhausted
        )

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
        with self._lock:
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
            mapping_version = record.id_mapping_version

        owners = (vectors_owner, index_owner, scratch_owner)
        vectors = []
        built = {}
        item = None
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
            # The backend's own storage is charged before it is allocated,
            # not discovered afterwards, and retained bytes are charged
            # separately from the transient peak a build passes through:
            # they have different lifetimes, so one reservation cannot
            # describe both without lying about one of them.
            if self.budget is not None:
                shapes = [
                    (int(item.vectors.shape[0]), int(item.vectors.shape[1]))
                    for item in vectors
                ]
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
            for item in vectors:
                built[(item.layer, item.kv_head)] = self.backend.build(
                    item.vectors, vector_space=self.vector_space, metric=self.metric
                )
                self._validate_built_index(built[(item.layer, item.kv_head)], item)
            # The transient peak is over; give it back before the index is
            # installed, so an idle index is charged only for what it holds.
            self._release_owners((scratch_owner,))
        except TransferCapacityError as exc:
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
        except Exception as exc:
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
            return False

        with self._lock:
            if self._entries.get(transfer_id) is not record:
                # Closed while we were building. Drop what we made and refund.
                item = None
                vectors.clear()
                built.clear()
                self._release_owners(owners)
                return False
            record.budget_owners = (vectors_owner, index_owner)
            record.vectors = {(v.layer, v.kv_head): v for v in vectors}
            record.indexes = built
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
        return True

    # -- searching ----------------------------------------------------------

    def search(
        self,
        identity: SearchRequestIdentity,
        *,
        queries: torch.Tensor,
        top_k: int,
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
        with self._lock:
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
            index = record.indexes.get(key)
            if item is None or index is None:
                raise IndexSearchError(
                    f"entry {transfer_id} has no index for layer "
                    f"{identity.layer} KV head {identity.kv_head}"
                )
            # Registered as a reader before the lock is dropped, so a close()
            # racing this search leaves the copies -- and their charge -- in
            # place until this search returns them.
            record.users += 1
        scratch_owner = f"prompt-index-search:{transfer_id}:{uuid.uuid4().hex[:8]}"
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
            if self.budget is not None:
                self._reserve(
                    scratch_owner,
                    self.backend.search_footprint(
                        index.count, index.dim, int(queries.shape[0]), int(top_k)
                    ),
                )
            selection = select(
                self.backend,
                index,
                queries,
                layer=identity.layer,
                kv_head=identity.kv_head,
                mapping=item.mapping,
                top_k=top_k,
            )
        except BaseException as exc:
            # Preserve the exception and traceback locations, but not finished
            # backend/select frames' tensor locals while refunding scratch.
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            item = index = None
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
