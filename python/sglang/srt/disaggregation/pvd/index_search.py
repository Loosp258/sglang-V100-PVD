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
from typing import Dict, Mapping, Sequence, Tuple

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
    """What a search chose, in Prompt terms, for one layer."""

    layer: int
    token_ids: Tuple[int, ...]
    page_ids: Tuple[int, ...]
    scores: Tuple[float, ...]
    metric: str
    id_mapping_version: str


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
) -> Selection:
    """Search one layer and return the choice in Prompt terms, not addresses."""
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise IndexSearchError("layer must be a non-negative integer")
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

    ``per_layer`` keeps each layer's own choice, which is the only policy that
    assumes nothing. ``union`` and ``intersection`` combine the token sets but
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
    layers = [s.layer for s in selections]
    if len(set(layers)) != len(layers):
        raise IndexSearchError("duplicate layer in the selections to merge")

    if policy == "per_layer":
        return {s.layer: s.token_ids for s in selections}
    sets = [set(s.token_ids) for s in selections]
    combined = set.union(*sets) if policy == "union" else set.intersection(*sets)
    return {layer: tuple(sorted(combined)) for layer in layers}
