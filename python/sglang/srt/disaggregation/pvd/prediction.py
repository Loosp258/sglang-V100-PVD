"""Configurable draft prediction and isolated target-model probing.

This module is NOT wired into serving. It defines the interfaces and the
isolation rules for the predictive retrieval path so that generic development
can proceed before any model is chosen and before V100S hardware exists.

The confirmed data path:

    read-only snapshot of the committed prefix
        -> a separate draft model predicts tokens
        -> an isolated target-model probe computes Q at defined positions
        -> queries carrying layer/head/position semantics
        -> V searches the target model's Prompt K

Two rules are structural here rather than advisory:

* Predicted tokens are never committed output. A ``DraftPrediction`` is not a
  token stream the scheduler may append; nothing in this module can advance a
  committed count.
* A query is only usable if it is in the target model's vector space. Draft-
  model hidden states must never be searched against target-model K, so the
  pipeline compares the probe's declared space against the configured target
  and refuses a mismatch instead of coercing it.

RNG isolation is enforced by running every prediction and probe inside
``run_isolated``, which forks the torch RNG state. A prediction branch that
samples must not perturb the committed sampler.

Nothing here loads a model. Concrete providers are supplied at experiment
time; ``FakeDraftProvider`` and ``FakeTargetProbe`` exist so the surrounding
machinery can be developed and tested without weights.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

#: Recorded when weights come from a local path with no resolvable revision.
#: Never invent a revision, and never download or hash weights to produce one.
UNKNOWN_REVISION = "local/unknown"


class PredictionConfigError(ValueError):
    """An unsupported or incoherent prediction configuration."""


class VectorSpaceError(ValueError):
    """A query was produced in the wrong vector space to be searched."""


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PredictionConfigError(f"{name} must be a non-empty string")
    return value


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PredictionConfigError(f"{name} must be a positive integer")
    return value


# --------------------------------------------------------------------------
# Committed state, as a read-only snapshot
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CommittedPrefix:
    """An immutable view of what the target model has actually committed.

    Prediction and probing receive this instead of the live request, so they
    cannot reach committed output ids, positions, KV or sampler state. The
    token ids are a tuple: a provider that tries to append to them fails.

    ``tokens`` is the whole committed prefix: the prompt plus every token the
    target model has committed. ``committed_position`` is the separate refresh
    counter ``n`` -- committed Decode tokens, excluding the first token P
    sampled -- so it is bounded by the prefix but is not an index into it. The
    next sequence position is ``len(tokens)``.
    """

    request_id: str
    tokens: Tuple[int, ...]
    committed_position: int
    version: str

    def __post_init__(self) -> None:
        _require_text("request_id", self.request_id)
        _require_text("version", self.version)
        if not isinstance(self.tokens, tuple):
            raise PredictionConfigError("tokens must be a tuple")
        if any(isinstance(t, bool) or not isinstance(t, int) for t in self.tokens):
            raise PredictionConfigError("tokens must be integers")
        if (
            isinstance(self.committed_position, bool)
            or not isinstance(self.committed_position, int)
            or self.committed_position < 0
        ):
            raise PredictionConfigError(
                "committed_position must be a non-negative integer"
            )
        if self.committed_position > len(self.tokens):
            raise PredictionConfigError(
                "committed_position is past the end of the snapshot"
            )


def snapshot_committed(
    request_id: str,
    tokens: Sequence[int],
    committed_position: int,
    version: str,
) -> CommittedPrefix:
    """Copy live committed state into a snapshot the prediction branch owns.

    The copy is the point: a later mutation of the caller's sequence must not
    be visible to an in-flight prediction, and a batch-membership change does
    not invalidate a request-local snapshot.
    """
    return CommittedPrefix(
        request_id=request_id,
        tokens=tuple(tokens),
        committed_position=committed_position,
        version=version,
    )


# --------------------------------------------------------------------------
# Draft side
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftConfig:
    """Where the draft model comes from and what it is allowed to cost.

    ``model_name_or_path`` is required and never defaulted: no model is
    hard-coded. ``revision`` is optional; it is recorded for reproducibility
    when available and is not a prerequisite for development.
    """

    model_name_or_path: str
    revision: Optional[str] = None
    device: str = "cpu"
    dtype: Optional[str] = None
    predict_tokens: int = 8
    max_prediction_batch: int = 1

    def __post_init__(self) -> None:
        _require_text("model_name_or_path", self.model_name_or_path)
        _require_text("device", self.device)
        _require_positive_int("predict_tokens", self.predict_tokens)
        _require_positive_int("max_prediction_batch", self.max_prediction_batch)
        if self.revision is not None:
            _require_text("revision", self.revision)
        if self.dtype is not None:
            _require_text("dtype", self.dtype)

    def resolved_source(self) -> Dict[str, str]:
        """What to record in a run report; never a fabricated revision."""
        return {
            "model": self.model_name_or_path,
            "revision": self.revision or UNKNOWN_REVISION,
        }


@dataclass(frozen=True)
class DraftPrediction:
    """Speculative tokens. Never committed output, never a KV-quality claim."""

    request_id: str
    prefix_version: str
    tokens: Tuple[int, ...]
    source: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text("request_id", self.request_id)
        _require_text("prefix_version", self.prefix_version)
        if not isinstance(self.tokens, tuple) or not self.tokens:
            raise PredictionConfigError("a prediction must carry at least one token")
        if any(isinstance(t, bool) or not isinstance(t, int) for t in self.tokens):
            raise PredictionConfigError("predicted tokens must be integers")


class DraftProvider(abc.ABC):
    """A separate small model that predicts future tokens.

    Implementations validate tokenizer compatibility, device, dtype and budget
    when they load, and refuse an unsupported configuration explicitly rather
    than silently changing semantics.
    """

    @abc.abstractmethod
    def describe(self) -> Dict[str, str]:
        """Resolved model identity for the run report."""

    @abc.abstractmethod
    def predict(self, prefix: CommittedPrefix, max_tokens: int) -> DraftPrediction:
        """Predict at most ``max_tokens`` continuations of ``prefix``."""


# --------------------------------------------------------------------------
# Probe side
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeConfig:
    """Which target-model attention vectors a probe is asked to capture."""

    target_model_id: str
    layers: Tuple[int, ...]
    head_start: int = 0
    head_count: int = 1

    def __post_init__(self) -> None:
        _require_text("target_model_id", self.target_model_id)
        if not isinstance(self.layers, tuple) or not self.layers:
            raise PredictionConfigError("probe requires at least one layer")
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in self.layers
        ):
            raise PredictionConfigError("probe layers must be non-negative integers")
        if len(set(self.layers)) != len(self.layers):
            raise PredictionConfigError("probe layers must be unique")
        if isinstance(self.head_start, bool) or not isinstance(self.head_start, int):
            raise PredictionConfigError("head_start must be an integer")
        if self.head_start < 0:
            raise PredictionConfigError("head_start must be non-negative")
        _require_positive_int("head_count", self.head_count)


@dataclass(frozen=True)
class QueryVectors:
    """Retrieval queries with explicit space, position and version semantics.

    ``vector_space`` names the model whose K these may be searched against. It
    is compared, never assumed: a query built from the draft model is not
    usable against target-model K, however similar the shapes happen to be.
    ``valid_length`` bounds how many positions are real; padding is never
    searched.
    """

    vector_space: str
    version: str
    layer: int
    head_start: int
    head_count: int
    positions: Tuple[int, ...]
    valid_length: int
    vectors: Any = None

    def __post_init__(self) -> None:
        _require_text("vector_space", self.vector_space)
        _require_text("version", self.version)
        for name in ("layer", "head_start"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PredictionConfigError(f"{name} must be a non-negative integer")
        _require_positive_int("head_count", self.head_count)
        if not isinstance(self.positions, tuple) or not self.positions:
            raise PredictionConfigError("a query must cover at least one position")
        if any(
            isinstance(p, bool) or not isinstance(p, int) or p < 0
            for p in self.positions
        ):
            raise PredictionConfigError("query positions must be non-negative")
        if list(self.positions) != sorted(self.positions):
            raise PredictionConfigError("query positions must be ascending")
        if len(set(self.positions)) != len(self.positions):
            raise PredictionConfigError("query positions must be unique")
        if (
            isinstance(self.valid_length, bool)
            or not isinstance(self.valid_length, int)
            or not 0 < self.valid_length <= len(self.positions)
        ):
            raise PredictionConfigError(
                "valid_length must be within the query positions"
            )


class TargetProbe(abc.ABC):
    """An isolated target-model execution that produces retrieval queries.

    The probe runs the target model, so its queries live in the target's Q/K
    space. It must not touch committed output ids, positions, KV or sampler
    state; temporary state it creates belongs to the prediction branch and is
    released through that branch's lifecycle.
    """

    @abc.abstractmethod
    def capture(
        self, prefix: CommittedPrefix, prediction: DraftPrediction
    ) -> Tuple[QueryVectors, ...]:
        """Compute Q at the predicted positions, one entry per probed layer."""


# --------------------------------------------------------------------------
# Isolation
# --------------------------------------------------------------------------


def run_isolated(fn, *args, **kwargs):
    """Run a prediction-branch callable without disturbing committed RNG.

    torch RNG state is forked for the call, so a draft model that samples
    cannot change what the committed sampler produces next. This covers the
    RNG only: committed ids, positions and KV are protected structurally, by
    handing the branch an immutable snapshot rather than the live request.
    """
    with torch.random.fork_rng(devices=[], enabled=True):
        return fn(*args, **kwargs)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


class PredictionPipeline:
    """draft -> probe -> queries, with the space check that makes it safe."""

    def __init__(
        self,
        provider: DraftProvider,
        probe: TargetProbe,
        draft_config: DraftConfig,
        probe_config: ProbeConfig,
    ) -> None:
        if draft_config.model_name_or_path == probe_config.target_model_id:
            # Then it is not a separate draft model, and the whole point of
            # the second model is lost. Refuse rather than silently degrade.
            raise PredictionConfigError(
                "the draft model must be separate from the target model"
            )
        self.provider = provider
        self.probe = probe
        self.draft_config = draft_config
        self.probe_config = probe_config

    def describe(self) -> Dict[str, Any]:
        return {
            "draft": dict(self.provider.describe()),
            "target_model_id": self.probe_config.target_model_id,
            "layers": list(self.probe_config.layers),
        }

    def run(self, prefix: CommittedPrefix) -> Tuple[QueryVectors, ...]:
        if not isinstance(prefix, CommittedPrefix):
            raise PredictionConfigError("prediction requires a committed snapshot")
        prediction = run_isolated(
            self.provider.predict, prefix, self.draft_config.predict_tokens
        )
        if not isinstance(prediction, DraftPrediction):
            raise PredictionConfigError("draft provider returned no prediction")
        if prediction.request_id != prefix.request_id:
            raise PredictionConfigError("prediction belongs to another request")
        if prediction.prefix_version != prefix.version:
            raise PredictionConfigError("prediction was made against a stale prefix")
        if len(prediction.tokens) > self.draft_config.predict_tokens:
            raise PredictionConfigError("draft provider exceeded its token budget")

        queries = run_isolated(self.probe.capture, prefix, prediction)
        if not isinstance(queries, tuple) or not queries:
            raise PredictionConfigError("probe produced no queries")
        seen = set()
        for query in queries:
            if not isinstance(query, QueryVectors):
                raise PredictionConfigError("probe returned a non-query object")
            if query.vector_space != self.probe_config.target_model_id:
                raise VectorSpaceError(
                    f"query is in {query.vector_space!r} but V searches "
                    f"{self.probe_config.target_model_id!r}"
                )
            if query.layer not in self.probe_config.layers:
                raise PredictionConfigError("probe returned an unrequested layer")
            if query.layer in seen:
                raise PredictionConfigError("probe returned a duplicate layer")
            seen.add(query.layer)
        return queries


# --------------------------------------------------------------------------
# Fakes, for development without weights
# --------------------------------------------------------------------------


class FakeDraftProvider(DraftProvider):
    """Deterministic stand-in. It samples, so it exercises RNG isolation."""

    def __init__(self, config: DraftConfig, tokens: Optional[Sequence[int]] = None):
        self.config = config
        self._tokens = None if tokens is None else tuple(tokens)
        self.calls = []

    def describe(self) -> Dict[str, str]:
        return self.config.resolved_source()

    def predict(self, prefix: CommittedPrefix, max_tokens: int) -> DraftPrediction:
        self.calls.append((prefix.request_id, max_tokens))
        # Deliberately consume RNG: a real draft model samples, and that must
        # not be visible to the committed sampler.
        torch.rand(1)
        tokens = self._tokens
        if tokens is None:
            base = prefix.tokens[-1] if prefix.tokens else 0
            tokens = tuple(base + i + 1 for i in range(max_tokens))
        return DraftPrediction(
            request_id=prefix.request_id,
            prefix_version=prefix.version,
            tokens=tokens[:max_tokens],
            source=self.describe(),
        )


class FakeTargetProbe(TargetProbe):
    """Stand-in probe. Emits one query per configured layer, in target space."""

    def __init__(self, config: ProbeConfig, vector_space: Optional[str] = None):
        self.config = config
        self.vector_space = vector_space or config.target_model_id
        self.calls = []

    def capture(
        self, prefix: CommittedPrefix, prediction: DraftPrediction
    ) -> Tuple[QueryVectors, ...]:
        self.calls.append((prefix.request_id, prediction.tokens))
        torch.rand(1)
        # Predicted tokens occupy the sequence positions after the committed
        # prefix. committed_position is the refresh counter, not an index.
        start = len(prefix.tokens)
        positions = tuple(start + i for i in range(len(prediction.tokens)))
        return tuple(
            QueryVectors(
                vector_space=self.vector_space,
                version=f"{prefix.version}:probe",
                layer=layer,
                head_start=self.config.head_start,
                head_count=self.config.head_count,
                positions=positions,
                valid_length=len(positions),
                vectors=torch.zeros(len(positions), self.config.head_count),
            )
            for layer in self.config.layers
        )
