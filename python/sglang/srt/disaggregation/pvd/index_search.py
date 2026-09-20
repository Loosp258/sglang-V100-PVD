"""Retrieval backends and logical selection for the V-side Prompt index.

The point of this module is that CAGRA becomes a swap rather than a rewrite.
``BruteForceIndexBackend`` is exact, CPU-only and needs no cuVS, so the search
path, the id mapping and the merge policy can all be built and tested now; it
is also the ground truth a CAGRA backend's recall will later be measured
against, which is why it is not a throwaway stub.

Two design rules are enforced here rather than left to the caller:

* Results are **logical**. A search returns token and page ids through an
  explicitly versioned ``IdMapping``, never raw addresses. Whoever gathers KV
  resolves addresses itself, under its own bounds checks.
* Scores from unrelated layers are never merged into one global Top-K by
  default. ``merge_selections`` requires a named policy and refuses anything
  it does not implement, because "just take the best overall" silently
  assumes the layers are comparable.

Nothing here registers memory, touches the transport or frees anything.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

METRICS = ("ip", "l2")

#: Merge policies. There is deliberately no default: the caller states one.
MERGE_POLICIES = ("per_layer", "union", "intersection")


class IndexSearchError(ValueError):
    """A search was refused rather than answered approximately."""


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IndexSearchError(f"{name} must be a non-empty string")
    return value


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise IndexSearchError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class IdMapping:
    """Vector row -> logical Prompt position, with its own version.

    Versioned separately from the index: a mapping can be rebuilt without
    rebuilding vectors, and trusting a stale one would return the wrong
    tokens for perfectly good scores.
    """

    version: str
    token_ids: Tuple[int, ...]
    page_size: int

    def __post_init__(self) -> None:
        _require_text("version", self.version)
        _require_positive_int("page_size", self.page_size)
        if not isinstance(self.token_ids, tuple) or not self.token_ids:
            raise IndexSearchError("an id mapping needs at least one token")
        if any(
            isinstance(t, bool) or not isinstance(t, int) or t < 0
            for t in self.token_ids
        ):
            raise IndexSearchError("token ids must be non-negative integers")

    def __len__(self) -> int:
        return len(self.token_ids)

    def token_of(self, row: int) -> int:
        if isinstance(row, bool) or not isinstance(row, int):
            raise IndexSearchError("a vector row must be an integer")
        if not 0 <= row < len(self.token_ids):
            raise IndexSearchError(f"vector row {row} is outside the id mapping")
        return self.token_ids[row]

    def page_of(self, row: int) -> int:
        return self.token_of(row) // self.page_size


@dataclass(frozen=True)
class Selection:
    """What a search chose, in Prompt terms, for one layer and KV head.

    ``kv_head`` is optional only so callers that index a whole layer keep
    working; when it is set it stays attached through merging, so head
    identity is never silently collapsed into a per-layer result.
    """

    layer: int
    token_ids: Tuple[int, ...]
    page_ids: Tuple[int, ...]
    scores: Tuple[float, ...]
    metric: str
    id_mapping_version: str
    kv_head: Optional[int] = None

    @property
    def key(self):
        """What identifies this selection: the head too, when there is one."""
        return self.layer if self.kv_head is None else (self.layer, self.kv_head)


@dataclass(frozen=True)
class BuiltIndex:
    """An immutable built index. The backend owns whatever is inside."""

    vector_space: str
    metric: str
    dim: int
    count: int
    handle: object


class IndexBackend(abc.ABC):
    """Swappable retrieval implementation. CAGRA plugs in here."""

    name: str = "abstract"

    @abc.abstractmethod
    def build(
        self, vectors: torch.Tensor, *, vector_space: str, metric: str
    ) -> BuiltIndex:
        """Build an immutable index over one layer's Prompt K vectors."""

    @abc.abstractmethod
    def search(
        self, index: BuiltIndex, queries: torch.Tensor, *, top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (rows, scores), both [num_queries, top_k], higher is better."""


def _validate_vectors(vectors: torch.Tensor, name: str) -> None:
    if not isinstance(vectors, torch.Tensor):
        raise IndexSearchError(f"{name} must be a tensor")
    if vectors.ndim != 2:
        raise IndexSearchError(f"{name} must be 2-D [rows, dim]")
    if vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise IndexSearchError(f"{name} must not be empty")
    if not torch.isfinite(vectors).all():
        raise IndexSearchError(f"{name} contains non-finite values")


class BruteForceIndexBackend(IndexBackend):
    """Exact search. The recall ground truth, and the no-cuVS default."""

    name = "brute_force"

    def build(
        self, vectors: torch.Tensor, *, vector_space: str, metric: str
    ) -> BuiltIndex:
        _require_text("vector_space", vector_space)
        if metric not in METRICS:
            raise IndexSearchError(f"metric must be one of {METRICS}, got {metric!r}")
        _validate_vectors(vectors, "index vectors")
        stored = vectors.detach().clone().to(torch.float32)
        return BuiltIndex(
            vector_space=vector_space,
            metric=metric,
            dim=int(stored.shape[1]),
            count=int(stored.shape[0]),
            handle=stored,
        )

    def search(
        self, index: BuiltIndex, queries: torch.Tensor, *, top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(index, BuiltIndex):
            raise IndexSearchError("a built index is required")
        _validate_vectors(queries, "queries")
        _require_positive_int("top_k", top_k)
        if queries.shape[1] != index.dim:
            raise IndexSearchError(
                f"queries have dim {queries.shape[1]}, index has {index.dim}"
            )
        if top_k > index.count:
            raise IndexSearchError(
                f"top_k {top_k} exceeds the {index.count} indexed vectors"
            )
        stored = index.handle
        q = queries.detach().to(torch.float32)
        if index.metric == "ip":
            scores = q @ stored.T
        else:
            scores = -torch.cdist(q, stored)
        # Stable, deterministic ties: prefer the lower row so a test does not
        # depend on kernel ordering.
        scores, rows = torch.sort(scores, dim=-1, descending=True, stable=True)
        return rows[:, :top_k].contiguous(), scores[:, :top_k].contiguous()


def select(
    backend: IndexBackend,
    index: BuiltIndex,
    queries: torch.Tensor,
    *,
    layer: int,
    mapping: IdMapping,
    top_k: int,
    kv_head: Optional[int] = None,
) -> Selection:
    """Search one layer and return the choice in Prompt terms, not addresses."""
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise IndexSearchError("layer must be a non-negative integer")
    if kv_head is not None and (
        isinstance(kv_head, bool) or not isinstance(kv_head, int) or kv_head < 0
    ):
        raise IndexSearchError("kv_head must be a non-negative integer")
    if not isinstance(mapping, IdMapping):
        raise IndexSearchError("an id mapping is required to return token ids")
    if len(mapping) != index.count:
        raise IndexSearchError(
            f"id mapping covers {len(mapping)} rows, index holds {index.count}"
        )
    rows, scores = backend.search(index, queries, top_k=top_k)
    # One ordered, de-duplicated selection per layer: the same Prompt token
    # chosen by several queries is fetched once, keeping its best score.
    best: Dict[int, float] = {}
    for row, score in zip(rows.reshape(-1).tolist(), scores.reshape(-1).tolist()):
        token = mapping.token_of(int(row))
        if token not in best or score > best[token]:
            best[token] = float(score)
    ordered = sorted(best.items(), key=lambda item: (-item[1], item[0]))
    token_ids = tuple(token for token, _ in ordered)
    return Selection(
        layer=layer,
        kv_head=kv_head,
        token_ids=token_ids,
        page_ids=tuple(sorted({token // mapping.page_size for token in token_ids})),
        scores=tuple(score for _, score in ordered),
        metric=index.metric,
        id_mapping_version=mapping.version,
    )


def merge_selections(
    selections: Sequence[Selection], *, policy: str
) -> Mapping[int, Tuple[int, ...]]:
    """Combine per-layer selections under an explicitly named policy.

    Results are keyed by ``Selection.key``: the layer, or ``(layer, kv_head)``
    when the selections carry head identity.

    ``per_layer`` keeps each selection's own choice, which is the only policy
    that assumes nothing. ``union`` and ``intersection`` combine the token sets but
    never the scores: scores from different layers are not comparable, and
    ranking across them would be a silent modelling claim.
    """
    if policy not in MERGE_POLICIES:
        raise IndexSearchError(
            f"merge policy must be one of {MERGE_POLICIES}; scores from "
            "different layers are not comparable, so there is no default"
        )
    if not selections:
        raise IndexSearchError("nothing to merge")
    versions = {s.id_mapping_version for s in selections}
    if len(versions) != 1:
        raise IndexSearchError("selections come from different id mappings")
    keys = [s.key for s in selections]
    if len(set(keys)) != len(keys):
        raise IndexSearchError("duplicate layer or KV head in the selections to merge")

    if policy == "per_layer":
        return {s.key: s.token_ids for s in selections}
    sets = [set(s.token_ids) for s in selections]
    combined = set.union(*sets) if policy == "union" else set.intersection(*sets)
    return {key: tuple(sorted(combined)) for key in keys}
