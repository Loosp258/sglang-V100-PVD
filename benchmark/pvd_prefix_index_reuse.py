"""Experimental, opt-in prefix index reuse; not wired into V serving.

Run against an IndexBackend (including cagra-auto on a configured V rank) to
measure whether sharing an immutable Prompt-K prefix repays split-search cost.
The caller supplies a candidate prefix identity. Every reused head is checked
against the actual K values before any handle is shared; a token digest alone
is not an authorization to reuse a graph.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.disaggregation.pvd.index_search import BuiltIndex, IndexBackend


Head = tuple[int, int]


@dataclass
class PrefixGroup:
    identity: Hashable
    rows: int
    vectors: dict[Head, torch.Tensor]
    indexes: dict[Head, BuiltIndex]
    users: int = 0


@dataclass
class SplitEntry:
    prefix: PrefixGroup
    tails: dict[Head, BuiltIndex]
    rows: int
    closed: bool = False


@dataclass(frozen=True)
class BuildTiming:
    verify_seconds: float
    prefix_build_seconds: float
    tail_build_seconds: float
    prefix_builds: int
    tail_builds: int


class PrefixIndexReuseExperiment:
    """Own one shared prefix group and per-Entry tails on one backend/device.

    This intentionally does not provide serving budget, native completion,
    protocol identity or close-race guarantees. Those belong in V's manager.
    """

    def __init__(
        self,
        backend: IndexBackend,
        *,
        vector_space: str,
        metric: str,
        retain_idle: bool = False,
    ):
        self.backend = backend
        self.vector_space = vector_space
        self.metric = metric
        self.retain_idle = retain_idle
        self.groups: dict[Hashable, PrefixGroup] = {}

    def build(
        self,
        vectors: Mapping[Head, torch.Tensor],
        *,
        prefix_rows: int,
        prefix_identity: Hashable,
    ) -> tuple[SplitEntry, BuildTiming]:
        if not vectors or prefix_identity is None:
            raise ValueError("heads and a prefix identity are required")
        heads = set(vectors)
        row_counts = {int(value.shape[0]) for value in vectors.values()}
        dims = {int(value.shape[1]) for value in vectors.values()}
        if len(row_counts) != 1 or len(dims) != 1:
            raise ValueError("all heads must have the same shape")
        rows = row_counts.pop()
        if not 0 < prefix_rows <= rows:
            raise ValueError("prefix must be a nonempty part of the Entry")
        existing = self.groups.get(prefix_identity)
        verify_start = perf_counter()
        if existing is not None:
            if existing.rows != prefix_rows or set(existing.vectors) != heads:
                raise ValueError("prefix identity has a different layout")
            for head, value in vectors.items():
                if not torch.equal(existing.vectors[head], value[:prefix_rows]):
                    raise ValueError("prefix identity does not match stored K")
        verify_seconds = perf_counter() - verify_start

        prefix_seconds = tail_seconds = 0.0
        prefix_builds = tail_builds = 0
        group = existing
        tails: dict[Head, BuiltIndex] = {}
        try:
            if group is None:
                group = PrefixGroup(prefix_identity, prefix_rows, {}, {})
                start = perf_counter()
                for head, value in vectors.items():
                    # The snapshot also protects the equality check from a
                    # caller mutating its own Entry's extracted K later.
                    snapshot = value[:prefix_rows].contiguous().clone()
                    group.vectors[head] = snapshot
                    group.indexes[head] = self.backend.build(
                        snapshot, vector_space=self.vector_space, metric=self.metric
                    )
                    prefix_builds += 1
                self.backend.synchronize()
                prefix_seconds = perf_counter() - start
            if prefix_rows < rows:
                start = perf_counter()
                for head, value in vectors.items():
                    tails[head] = self.backend.build(
                        value[prefix_rows:].contiguous(),
                        vector_space=self.vector_space,
                        metric=self.metric,
                    )
                    tail_builds += 1
                self.backend.synchronize()
                tail_seconds = perf_counter() - start
        except BaseException:
            self.backend.synchronize()
            for index in tails.values():
                self.backend.dispose(index)
            if existing is None and group is not None:
                for index in group.indexes.values():
                    self.backend.dispose(index)
            raise
        if existing is None:
            self.groups[prefix_identity] = group
        group.users += 1
        return SplitEntry(group, tails, rows), BuildTiming(
            verify_seconds, prefix_seconds, tail_seconds, prefix_builds, tail_builds
        )

    def search(
        self, entry: SplitEntry, head: Head, queries: torch.Tensor, *, top_k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge two local top-k lists; scores use the backend's convention."""
        if entry.closed:
            raise ValueError("Entry index has been closed")
        if not 0 < top_k <= entry.rows:
            raise ValueError("top_k exceeds this Entry")
        prefix = entry.prefix.indexes[head]
        if not entry.tails:
            return self.backend.search(prefix, queries, top_k=top_k)
        tail = entry.tails[head]
        prefix_rows, prefix_scores = self.backend.search(
            prefix, queries, top_k=min(top_k, prefix.count)
        )
        tail_rows, tail_scores = self.backend.search(
            tail, queries, top_k=min(top_k, tail.count)
        )
        rows = torch.cat((prefix_rows.long(), tail_rows.long() + prefix.count), dim=1)
        scores = torch.cat((prefix_scores, tail_scores), dim=1)
        # Prefix rows come first, so stable sorting keeps the lower logical
        # row when scores tie, matching the exact full-index contract.
        ordered, positions = torch.sort(scores, dim=1, descending=True, stable=True)
        return torch.gather(rows, 1, positions[:, :top_k]), ordered[:, :top_k]

    def close(self, entry: SplitEntry) -> None:
        if entry.closed:
            raise ValueError("Entry index has already been closed")
        self.backend.synchronize()
        for index in entry.tails.values():
            self.backend.dispose(index)
        entry.closed = True
        group = entry.prefix
        group.users -= 1
        if group.users == 0 and not self.retain_idle:
            self.evict(group.identity)

    def evict(self, identity: Hashable) -> None:
        """Explicitly retire an idle cached prefix after backend completion."""
        group = self.groups[identity]
        if group.users:
            raise ValueError("cannot evict a prefix used by an Entry")
        self.backend.synchronize()
        for index in group.indexes.values():
            self.backend.dispose(index)
        del self.groups[identity]
