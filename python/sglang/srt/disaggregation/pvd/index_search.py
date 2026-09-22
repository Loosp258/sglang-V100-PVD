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
* **A backend declares where it lives and what it costs.** ``device`` says
  which device its indexes and queries must be on, and the two footprint
  methods report the bytes a build retains and a search temporarily needs, so
  the owner can reserve before allocating instead of discovering the cost
  afterwards. A dtype conversion is not a device move, so neither is ever
  inferred from the other.

Device policy for the exact reference backend
---------------------------------------------
``BruteForceIndexBackend`` is **CPU-only, by declaration**. On a V worker the
GPU holds the authoritative KV pool, and an exact float32 mirror of every
indexed prompt would compete with it for exactly the memory the pool needs;
the queries, meanwhile, arrive as JSON over HTTP and are born on the host.
So this backend states ``device = cpu`` and moves both sides there
explicitly, at build time and at search time. The stored KV pool is never
moved to suit it: extraction already copies, and that copy is where the
device transition happens. A device-resident backend (CAGRA) simply declares
its own device and the same machinery reserves and places against that.

Nothing here registers memory, touches the transport or frees anything.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

METRICS = ("ip", "l2")

#: Merge policies. There is deliberately no default: the caller states one.
MERGE_POLICIES = ("per_layer", "union", "intersection")


class IndexSearchError(ValueError):
    """A search was refused rather than answered approximately."""


class IndexNotReadyError(IndexSearchError):
    """The caller may retry after the index progresses."""


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
    """An immutable built index. The backend owns whatever is inside.

    ``aux`` is whatever else a backend retains alongside the vectors. It is
    part of the index's retained footprint, not a free extra. The exact
    backend does not need auxiliary tensors.
    """

    vector_space: str
    metric: str
    dim: int
    count: int
    handle: object
    aux: object = None

    def retained_tensors(self):
        """Every tensor this index keeps alive, for accounting and tests."""
        return tuple(
            item for item in (self.handle, self.aux) if isinstance(item, torch.Tensor)
        )


class IndexBackend(abc.ABC):
    """Swappable retrieval implementation. CAGRA plugs in here."""

    name: str = "abstract"

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        """Where this backend keeps indexes and expects queries.

        Declared, never inferred. Callers place vectors and queries here
        before building or searching; a backend does not quietly relocate the
        authoritative storage it was handed.
        """

    @abc.abstractmethod
    def build(
        self, vectors: torch.Tensor, *, vector_space: str, metric: str
    ) -> BuiltIndex:
        """Build an immutable index over one layer's Prompt K vectors."""

    @abc.abstractmethod
    def search(
        self, index: BuiltIndex, queries: torch.Tensor, *, top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (rows, scores), both [num_queries, top_k], on ``device``.

        Rows are integer vector IDs, unique within each query. Scores are finite
        floats, higher is better: dot products for ``ip``, negative Euclidean
        distances (NOT negative squared distances) for ``l2``. A native backend
        returning squared distances must convert them, not change this contract.
        """

    @abc.abstractmethod
    def build_footprint(self, rows: int, dim: int, *, metric: str) -> int:
        """Bytes a built index over ``[rows, dim]`` retains on ``device``.

        This is what the index holds for as long as it is searchable, on top
        of the caller's own vectors. An owner reserves this *before* calling
        ``build``, which is the only order in which a refusal is free. The
        metric is an input because it can change what has to be kept.
        """

    @abc.abstractmethod
    def build_scratch_footprint(self, rows: int, dim: int, *, metric: str) -> int:
        """Bounded bytes a build needs transiently, beyond what it retains.

        Declared separately because retained bytes are not peak build bytes:
        validation, placement and any precompute overlap. An owner reserves
        this for the build and gives it back afterwards.
        """

    @abc.abstractmethod
    def search_footprint(
        self, rows: int, dim: int, num_queries: int, top_k: int
    ) -> int:
        """Bounded scratch bytes one search needs, held only for its duration.

        A bound, not a measurement: it must not be exceeded, and a backend
        whose scratch cannot be bounded has no business being searched under
        a budget. "Bounded" here means bounded in the size of the index --
        a term that grows with the indexed prompt is not a bound.
        """


def resolve_device(spec: object) -> torch.device:
    """Turn a device spelling into one explicit, stable execution device.

    ``"cpu"`` and ``"cpu:0"`` name the same device and must compare equal;
    so do ``"cuda"`` and the ``cuda:N`` a tensor reports, but only for the
    N this process would actually use. So ``cuda`` is resolved **now**, at
    configuration time, rather than being matched loosely later: a policy
    that means "whatever device is current when a search happens" is not a
    policy. Two different GPU indices stay different, which is the case
    that must never be softened -- reading an index from the wrong GPU is
    exactly the bug a device check exists to catch.
    """
    device = spec if isinstance(spec, torch.device) else torch.device(spec)
    if device.type == "cpu":
        if device.index not in (None, 0):
            raise IndexSearchError(
                f"there is no CPU {device.index}; use 'cpu' or 'cpu:0'"
            )
        return torch.device("cpu")
    if device.type == "cuda" and device.index is None:
        if not torch.cuda.is_available():
            raise IndexSearchError(
                "device 'cuda' cannot be resolved: no CUDA device is available"
            )
        return torch.device("cuda", torch.cuda.current_device())
    return device


def same_device(left: torch.device, right: torch.device) -> bool:
    """Whether two devices are the same execution device, not the same spelling."""
    try:
        return resolve_device(left) == resolve_device(right)
    except IndexSearchError:
        return left == right


def _validate_vectors(
    vectors: torch.Tensor, name: str, *, chunk_rows: Optional[int] = None
) -> None:
    if not isinstance(vectors, torch.Tensor):
        raise IndexSearchError(f"{name} must be a tensor")
    if vectors.ndim != 2:
        raise IndexSearchError(f"{name} must be 2-D [rows, dim]")
    if vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise IndexSearchError(f"{name} must not be empty")
    # Checked in row blocks: ``isfinite`` over the whole tensor materialises
    # temporaries proportional to the whole index, which is the cost this
    # backend is trying to stop scaling with the indexed prompt.
    rows = int(vectors.shape[0])
    step = rows if not chunk_rows else min(int(chunk_rows), rows)
    for start in range(0, rows, step):
        if not torch.isfinite(vectors[start : start + step]).all():
            raise IndexSearchError(f"{name} contains non-finite values")


class BruteForceIndexBackend(IndexBackend):
    """Exact search on the host, in bounded passes over the index.

    CPU by declaration -- see the module docstring. ``device`` is settable so
    a deployment that really does want the exact backend on an accelerator
    can say so explicitly and have every placement and reservation follow;
    it is never derived from whatever tensor happened to arrive, and it is
    resolved once, at construction, into one concrete execution device.

    Bounded, not merely exact
    -------------------------
    The obvious implementations of this are all unbounded in the size of the
    index, which makes their cost impossible to reserve for:

    * ``torch.cdist`` on the L2 path takes the matrix-multiply route and
      materialises a padded copy of the whole index plus its full square --
      measured at 541,184 peak tensor bytes for 1,024 x 128 with a single
      query, against a nominal 16,908.
    * ``q @ stored.T`` materialises a contiguous transpose of the whole
      index, so even the inner-product path paid that.
    * ``torch.sort`` over the full score matrix allocates values, int64
      indices and its own scratch across every indexed vector.
    * ``torch.isfinite(...).all()`` at build time allocates a float and a
      bool image of the whole input at once.

    So the index is swept in row blocks of ``chunk_rows``. Every temporary
    is proportional to a block, never to the index: L2 subtracts each query
    from a block before squaring (avoiding norm-expansion cancellation),
    and the top-k is carried forward and merged one block at a time.
    The footprint methods below are bounds over that scheme, and
    ``test_pvd_index_search`` measures the real peak against them.

    Tie-breaking is unchanged: on equal scores the lower row wins. Carried
    results always come from earlier blocks and are placed first, and a
    stable sort keeps them ahead of the block being merged.
    """

    name = "brute_force"

    _F32 = 4
    _I64 = 8
    #: Rows per pass. The one knob that turns every unbounded temporary
    #: above into a bounded one; it trades passes for peak memory and
    #: changes no result.
    DEFAULT_CHUNK_ROWS = 512

    def __init__(self, device: object = "cpu", *, chunk_rows: int = 0) -> None:
        self._device = resolve_device(device)
        self._chunk_rows = _require_positive_int(
            "chunk_rows", chunk_rows or self.DEFAULT_CHUNK_ROWS
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def chunk_rows(self) -> int:
        return self._chunk_rows

    def _pass_rows(self, rows: int) -> int:
        return min(self._chunk_rows, rows)

    # -- declared costs -----------------------------------------------------

    def build_footprint(self, rows: int, dim: int, *, metric: str) -> int:
        """Bytes a built index **retains**: its owned float32 copy."""
        _require_positive_int("rows", rows)
        _require_positive_int("dim", dim)
        if metric not in METRICS:
            raise IndexSearchError(f"metric must be one of {METRICS}, got {metric!r}")
        return rows * dim * self._F32

    def build_scratch_footprint(self, rows: int, dim: int, *, metric: str) -> int:
        """Bytes a build needs **transiently**, on top of what it retains.

        Retained bytes are not peak build bytes: validation also allocates.
        It is swept in blocks so its workspace does not grow with the index.
        """
        _require_positive_int("rows", rows)
        _require_positive_int("dim", dim)
        if metric not in METRICS:
            raise IndexSearchError(f"metric must be one of {METRICS}, got {metric!r}")
        block = self._pass_rows(rows)
        # isfinite over a block: a float image (abs) and several bool images
        # (the comparison and the reduction), live together.
        validation = block * dim * (self._F32 + 4)
        # The placed float32 copy is deliberately NOT counted here: it is the
        # retained index, already charged by build_footprint. Counting it
        # again would hide an unbounded validation temporary behind a term
        # that merely looks large.
        return validation

    def search_footprint(
        self, rows: int, dim: int, num_queries: int, top_k: int
    ) -> int:
        """Bounded scratch one search needs, held only for its duration."""
        _require_positive_int("rows", rows)
        _require_positive_int("dim", dim)
        _require_positive_int("num_queries", num_queries)
        _require_positive_int("top_k", top_k)
        block = self._pass_rows(rows)
        keep = min(top_k, rows)
        # Query placement and validation temporaries (conservative bound).
        query = num_queries * dim * self._F32 * 2 + num_queries * self._F32
        # IP may need a contiguous transpose; L2 needs a difference block.
        transpose = block * dim * self._F32
        # The block's score matrix, plus one spare for the distance chain.
        scores = num_queries * block * self._F32 * 2
        ids = block * self._I64
        # Merging carries: the two concatenations, the sort's values and
        # indices, and the sort's own scratch over the same width.
        width = keep + block
        merge = num_queries * width * (self._F32 + self._I64) * 3
        carried = num_queries * keep * (self._F32 + self._I64) * 2
        return query + transpose + scores + ids + merge + carried

    # -- building -----------------------------------------------------------

    def build(
        self, vectors: torch.Tensor, *, vector_space: str, metric: str
    ) -> BuiltIndex:
        _require_text("vector_space", vector_space)
        if metric not in METRICS:
            raise IndexSearchError(f"metric must be one of {METRICS}, got {metric!r}")
        _validate_vectors(vectors, "index vectors", chunk_rows=self._chunk_rows)
        # Placement is explicit in both axes. ``.to(torch.float32)`` converts a
        # dtype and moves nothing, so the device is named separately; this is
        # a copy even when both already match, because the index owns its
        # bytes and must not alias a caller's tensor.
        stored = (
            vectors.detach()
            .to(device=self.device, dtype=torch.float32, copy=True)
            .contiguous()
        )
        return BuiltIndex(
            vector_space=vector_space,
            metric=metric,
            dim=int(stored.shape[1]),
            count=int(stored.shape[0]),
            handle=stored,
        )

    # -- searching ----------------------------------------------------------

    def search(
        self, index: BuiltIndex, queries: torch.Tensor, *, top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(index, BuiltIndex):
            raise IndexSearchError("a built index is required")
        _validate_vectors(queries, "queries", chunk_rows=self._chunk_rows)
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
        if not same_device(stored.device, self.device):
            raise IndexSearchError(
                f"index is on {stored.device} but this backend serves "
                f"{self.device}; rebuild it under the current device policy"
            )
        # The query is placed on the backend's device on purpose. An HTTP
        # caller sends numbers, not a device, so somebody has to decide, and
        # deciding here keeps that decision in one place.
        q = queries.detach().to(device=self.device, dtype=torch.float32)
        best_scores: Optional[torch.Tensor] = None
        best_rows: Optional[torch.Tensor] = None
        for lo in range(0, index.count, self._chunk_rows):
            hi = min(lo + self._chunk_rows, index.count)
            block = stored[lo:hi]
            if index.metric == "l2":
                # Subtract first. Expanding squared norms catastrophically
                # cancels for nearby, large vectors and can change top-k.
                # One query at a time bounds the difference workspace by
                # block_rows * dim (already reserved as the block workspace).
                scores = torch.empty(
                    (q.shape[0], hi - lo), dtype=torch.float32, device=self.device
                )
                for query_row in range(q.shape[0]):
                    delta = torch.sub(block, q[query_row])
                    delta.square_()
                    torch.sum(delta, dim=-1, out=scores[query_row])
                    del delta
                scores.sqrt_().neg_()
            else:
                scores = q @ block.T
            ids = torch.arange(lo, hi, dtype=torch.int64, device=scores.device).expand(
                scores.shape[0], -1
            )
            if best_scores is None:
                candidate_scores, candidate_rows = scores, ids
            else:
                candidate_scores = torch.cat([best_scores, scores], dim=1)
                candidate_rows = torch.cat([best_rows, ids], dim=1)
            keep = min(top_k, candidate_scores.shape[1])
            # Stable and descending: carried results sit first and come from
            # earlier blocks, so an equal score still resolves to the lower
            # row, exactly as the previous full sort did.
            ordered, order = torch.sort(
                candidate_scores, dim=-1, descending=True, stable=True
            )
            best_scores = ordered[:, :keep].contiguous()
            best_rows = torch.gather(candidate_rows, 1, order[:, :keep]).contiguous()
        return best_rows, best_scores


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
    _require_positive_int("top_k", top_k)
    if top_k > index.count:
        raise IndexSearchError("top_k exceeds indexed vector count")
    if (
        not isinstance(queries, torch.Tensor)
        or queries.ndim != 2
        or queries.shape[0] == 0
        or queries.shape[1] != index.dim
    ):
        raise IndexSearchError("queries must have non-empty [num_queries, dim] shape")
    result = backend.search(index, queries, top_k=top_k)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise IndexSearchError("backend must return (rows, scores)")
    rows, scores = result
    if not isinstance(rows, torch.Tensor) or not isinstance(scores, torch.Tensor):
        raise IndexSearchError("backend rows and scores must be tensors")
    expected_shape = (queries.shape[0], top_k)
    if tuple(rows.shape) != expected_shape or tuple(scores.shape) != expected_shape:
        raise IndexSearchError("backend result shape must be [num_queries, top_k]")
    if rows.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ):
        raise IndexSearchError("backend rows must have integer dtype, not bool/float")
    if not scores.dtype.is_floating_point:
        raise IndexSearchError("backend scores must have floating dtype")
    if any(not same_device(t.device, backend.device) for t in (rows, scores)):
        raise IndexSearchError("backend result device differs from declared device")
    # Materialize only the bounded result, as logical selection already does.
    # Validate on the host without another CUDA sort/mask or reshape allocation.
    host_rows, host_scores = rows.tolist(), scores.tolist()
    for query_rows, query_scores in zip(host_rows, host_scores):
        if any(not 0 <= row < index.count for row in query_rows):
            raise IndexSearchError("backend row is outside the indexed vector range")
        if len(set(query_rows)) != top_k:
            raise IndexSearchError("backend returned duplicate rows within a query")
        if any(not math.isfinite(score) for score in query_scores):
            raise IndexSearchError("backend scores must be finite")
    # One ordered, de-duplicated selection per layer: the same Prompt token
    # chosen by several queries is fetched once, keeping its best score.
    best: Dict[int, float] = {}
    for query_rows, query_scores in zip(host_rows, host_scores):
        for row, score in zip(query_rows, query_scores):
            token = mapping.token_of(row)
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
