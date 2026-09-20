"""Per-Entry Prompt indexes on a V rank: gates, vectors and search.

This is the first thing that actually drives ``IndexGate``. It owns one gate
per stored Entry shard, extracts that shard's Prompt K with
``prompt_vectors.extract_prompt_k``, builds an index per (layer, KV head)
through a swappable ``IndexBackend``, and answers searches under the gate's
identity checks.

Deliberate properties:

* **Delivery never waits for it.** Nothing here is consulted on the bootstrap
  or refresh path, and ``IndexGate.deliverable`` stays independent of index
  state. Turning this off changes nothing about how KV is served.
* **Off unless a backend is supplied.** ``VectorKVStore`` constructs no
  manager by default, so the existing store behaves exactly as before.
* **No autonomous driver.** ``build_pending`` is one bounded step a caller
  invokes, matching how uploads and decode closes are progressed. A worker
  that stops calling it keeps its gates rather than half-building anything.
* **It frees nothing.** ``close`` refuses further use and drops this
  manager's own vectors; the Entry's pages, registration and MR are released
  only by their existing owners, after their own proofs.

The default backend is the exact CPU one, so a V rank can build and search
with no cuVS present. A CAGRA backend replaces it without touching this file.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    IndexSearchError,
    Selection,
    select,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import (
    ROPE_APPLIED,
    PromptKVectors,
    PromptVectorError,
    extract_prompt_k,
)


@dataclass
class EntryIndex:
    """One Entry shard's gate and, once built, its per-head indexes."""

    gate: IndexGate
    id_mapping_version: str
    vectors: Dict[Tuple[int, int], PromptKVectors] = field(default_factory=dict)
    indexes: Dict[Tuple[int, int], BuiltIndex] = field(default_factory=dict)

    def heads(self) -> List[Tuple[int, int]]:
        return sorted(self.indexes)


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
        self._entries: Dict[str, EntryIndex] = {}

    # -- gates --------------------------------------------------------------

    def gate_for(self, transfer_id: str) -> Optional[IndexGate]:
        entry = self._entries.get(transfer_id)
        return entry.gate if entry is not None else None

    def open(self, transfer_id: str) -> IndexGate:
        existing = self._entries.get(transfer_id)
        if existing is not None:
            return existing.gate
        record = EntryIndex(
            gate=IndexGate(transfer_id, max_build_attempts=self.max_build_attempts),
            id_mapping_version=f"map:{transfer_id}:{uuid.uuid4().hex[:8]}",
        )
        self._entries[transfer_id] = record
        return record.gate

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
        that cannot be indexed is still perfectly deliverable.
        """
        record = self._entries.get(transfer_id)
        if record is None:
            raise IndexSearchError(f"no index gate for {transfer_id}")
        record.gate.begin_build()
        try:
            vectors = extract_prompt_k(
                packed,
                layout=layout,
                manifest=manifest,
                entry_transfer_id=transfer_id,
                id_mapping_version=record.id_mapping_version,
                positional_encoding=self.positional_encoding,
                budget=self.budget,
                budget_owner=(
                    f"prompt-index:{transfer_id}" if self.budget is not None else None
                ),
            )
            if not vectors:
                raise PromptVectorError("the shard produced no Prompt K vectors")
            built = {}
            for item in vectors:
                built[(item.layer, item.kv_head)] = self.backend.build(
                    item.vectors, vector_space=self.vector_space, metric=self.metric
                )
        except Exception as exc:
            record.gate.mark_failed(str(exc))
            return False
        record.vectors = {(v.layer, v.kv_head): v for v in vectors}
        record.indexes = built
        total = sum(index.count for index in built.values())
        record.gate.mark_ready(
            IndexDescriptor(
                index_version=f"idx:{transfer_id}:{record.gate.attempts}",
                entry_transfer_id=transfer_id,
                vector_space=self.vector_space,
                id_mapping_version=record.id_mapping_version,
                vector_count=total,
                metric=self.metric,
            )
        )
        return True

    # -- searching ----------------------------------------------------------

    def search(
        self,
        transfer_id: str,
        *,
        layer: int,
        kv_head: int,
        queries: torch.Tensor,
        top_k: int,
        positional_encoding: str,
    ) -> Selection:
        """Search one (layer, KV head) and return a logical Selection."""
        record = self._entries.get(transfer_id)
        if record is None:
            raise IndexSearchError(f"no index gate for {transfer_id}")
        # Refuses unless READY, and re-checks the query's vector space and the
        # id-mapping version against what was actually built.
        record.gate.authorize_search(self.vector_space, record.id_mapping_version)
        key = (layer, kv_head)
        item = record.vectors.get(key)
        index = record.indexes.get(key)
        if item is None or index is None:
            raise IndexSearchError(
                f"entry {transfer_id} has no index for layer {layer} KV head {kv_head}"
            )
        item.require_compatible_query(
            positional_encoding=positional_encoding,
            head_dim=int(queries.shape[-1]) if queries.ndim == 2 else -1,
        )
        return select(
            self.backend,
            index,
            queries,
            layer=layer,
            kv_head=kv_head,
            mapping=item.mapping,
            top_k=top_k,
        )

    # -- teardown -----------------------------------------------------------

    def close(self, transfer_id: str) -> None:
        """Stop serving this Entry's index and drop this manager's vectors.

        The Entry's pages, registration and MR are untouched: they are freed
        by their existing owners, after their own safety checks.
        """
        record = self._entries.pop(transfer_id, None)
        if record is None:
            return
        record.gate.close()
        record.vectors.clear()
        record.indexes.clear()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "vector_space": self.vector_space,
            "backend": getattr(self.backend, "name", "unknown"),
            "metric": self.metric,
            "positional_encoding": self.positional_encoding,
            "entries": {
                transfer_id: {
                    **record.gate.snapshot(),
                    "indexed_heads": len(record.indexes),
                }
                for transfer_id, record in self._entries.items()
            },
        }
