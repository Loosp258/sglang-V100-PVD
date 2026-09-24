"""A concrete DraftProvider backed by a Hugging Face causal LM.

This is the first real provider behind the interfaces in ``prediction.py``. It
is still opt-in: nothing constructs it, ``transformers`` is imported lazily so
it is not a new hard dependency, and the loading step is injectable so the
surrounding machinery stays testable without weights.

What it refuses, loudly, rather than working around:

* a vocabulary that does not match the target model's. Predicted token ids are
  fed to the target model's probe, so two tokenizers that disagree would make
  the probe compute Q for different text than the draft predicted. A different
  vocabulary needs a translation step that does not exist yet.
* a device or dtype that the loaded model did not actually end up on.
* a prediction budget of zero or a request for more tokens than configured.

It records the revision the loader actually resolved. When weights come from a
local path with no revision, that is reported as ``local/unknown``; no
revision is invented and no weights are downloaded or hashed to produce one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch
from sglang.srt.disaggregation.pvd.prediction import (
    UNKNOWN_REVISION,
    CommittedPrefix,
    DraftConfig,
    DraftPrediction,
    DraftProvider,
    PredictionConfigError,
)

#: Fixed probe text. Two tokenizers that encode this identically, and agree on
#: size and special ids, are treated as interchangeable for id passing. This is
#: a cheap guard against the common mistake, not a proof of equivalence.
_VOCAB_PROBE = "The quick brown fox jumps over the lazy dog 0123456789"


@dataclass(frozen=True)
class VocabularySignature:
    """What must match before one model's token ids may be fed to another."""

    size: int
    bos_token_id: Optional[int]
    eos_token_id: Optional[int]
    fingerprint: str
    allowed_ids: Optional[frozenset[int]] = field(default=None, repr=False)
    mapping_fingerprint: Optional[str] = None

    @property
    def exact_mapping_available(self) -> bool:
        return self.allowed_ids is not None and self.mapping_fingerprint is not None

    def contains(self, token_id: int) -> bool:
        if type(token_id) is not int or token_id < 0:
            return False
        if self.allowed_ids is not None:
            return token_id in self.allowed_ids
        # Hand-built test signatures do not have a tokenizer mapping.  Keep
        # their legacy contiguous base vocabulary, but include declared
        # special ids that legitimately live above tokenizer.vocab_size.
        return token_id < self.size or token_id in (
            self.bos_token_id,
            self.eos_token_id,
        )

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "VocabularySignature":
        size = getattr(tokenizer, "vocab_size", None)
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise PredictionConfigError("tokenizer has no usable vocab_size")
        try:
            probe_ids = tokenizer.encode(_VOCAB_PROBE)
        except Exception as exc:
            raise PredictionConfigError(f"tokenizer could not encode: {exc}") from exc
        if not probe_ids:
            raise PredictionConfigError("tokenizer produced no ids for the probe")
        digest = hashlib.sha256(
            ",".join(str(int(i)) for i in probe_ids).encode("utf-8")
        ).hexdigest()
        try:
            vocabulary = tokenizer.get_vocab()
        except AttributeError:
            vocabulary = None
        except Exception as exc:
            raise PredictionConfigError(
                f"tokenizer could not expose its token IDs: {exc}"
            ) from exc
        if vocabulary is not None:
            if not isinstance(vocabulary, dict) or not vocabulary:
                raise PredictionConfigError("tokenizer has no usable token-ID mapping")
            if any(type(token) is not str for token in vocabulary):
                raise PredictionConfigError(
                    "tokenizer mapping contains a non-string token"
                )
            if any(
                type(token_id) is not int or token_id < 0
                for token_id in vocabulary.values()
            ):
                raise PredictionConfigError("tokenizer contains an invalid token ID")
            valid_ids = frozenset(vocabulary.values())
            if len(valid_ids) != len(vocabulary):
                raise PredictionConfigError("tokenizer maps multiple tokens to one ID")
            canonical = sorted(vocabulary.items(), key=lambda pair: (pair[1], pair[0]))
            mapping_fingerprint = hashlib.sha256(
                json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            special_ids = getattr(tokenizer, "all_special_ids", ()) or ()
            special_ids = (
                *special_ids,
                getattr(tokenizer, "bos_token_id", None),
                getattr(tokenizer, "eos_token_id", None),
            )
            if any(
                type(token_id) is not int or token_id not in valid_ids
                for token_id in special_ids
                if token_id is not None
            ):
                raise PredictionConfigError(
                    "tokenizer declares a special ID absent from its token mapping"
                )
        else:
            # Minimal tokenizer doubles may expose only vocab_size and special
            # IDs.  Real HF tokenizers expose get_vocab(), which is required
            # for exact membership of added tokens and holes.
            specials = (
                getattr(tokenizer, "bos_token_id", None),
                getattr(tokenizer, "eos_token_id", None),
            )
            valid_ids = frozenset(range(size)).union(
                token_id
                for token_id in specials
                if type(token_id) is int and token_id >= 0
            )
            mapping_fingerprint = None
        if any(
            type(token_id) is not int or token_id not in valid_ids
            for token_id in probe_ids
        ):
            raise PredictionConfigError(
                "tokenizer probe produced an undeclared token ID"
            )
        return cls(
            size=size,
            bos_token_id=getattr(tokenizer, "bos_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            fingerprint=digest,
            allowed_ids=valid_ids,
            mapping_fingerprint=mapping_fingerprint,
        )


@dataclass(frozen=True)
class LoadedDraft:
    """What a loader must return: the two objects plus what it resolved."""

    tokenizer: Any
    model: Any
    resolved_revision: Optional[str] = None


def transformers_loader(config: DraftConfig) -> LoadedDraft:
    """Default loader. Imports ``transformers`` only when actually called."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised via a fake
        raise PredictionConfigError(
            "the Hugging Face draft provider requires `transformers`; install "
            "it, or supply a different loader"
        ) from exc
    kwargs: Dict[str, Any] = {}
    if config.revision is not None:
        kwargs["revision"] = config.revision
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, **kwargs)
    if config.dtype is not None:
        kwargs["torch_dtype"] = getattr(torch, config.dtype)
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **kwargs)
    model = model.to(config.device)
    model.eval()
    resolved = getattr(getattr(model, "config", None), "_commit_hash", None)
    return LoadedDraft(tokenizer, model, resolved or config.revision)


class HuggingFaceDraftProvider(DraftProvider):
    """Predict continuations with a separate small causal LM."""

    def __init__(
        self,
        config: DraftConfig,
        target_vocab: VocabularySignature,
        *,
        loader: Optional[Callable[[DraftConfig], LoadedDraft]] = None,
    ) -> None:
        if not isinstance(config, DraftConfig):
            raise PredictionConfigError("a DraftConfig is required")
        if not isinstance(target_vocab, VocabularySignature):
            raise PredictionConfigError(
                "the target model's vocabulary signature is required; token ids "
                "must never be passed between incompatible vocabularies"
            )
        loaded = (loader or transformers_loader)(config)
        if not isinstance(loaded, LoadedDraft):
            raise PredictionConfigError("draft loader returned no model")

        draft_vocab = VocabularySignature.from_tokenizer(loaded.tokenizer)
        if draft_vocab != target_vocab:
            raise PredictionConfigError(
                "draft and target tokenizers are incompatible: the draft's "
                f"{draft_vocab} does not match the target's {target_vocab}. "
                "Predicted ids would mean different text to the probe."
            )
        self._check_placement(config, loaded.model)

        self.config = config
        self.tokenizer = loaded.tokenizer
        self.model = loaded.model
        self.vocab = draft_vocab
        self._resolved_revision = loaded.resolved_revision or config.revision

    @staticmethod
    def _check_placement(config: DraftConfig, model: Any) -> None:
        """The model must be where and what the configuration asked for."""
        device = getattr(model, "device", None)
        if device is not None and not str(device).startswith(config.device):
            raise PredictionConfigError(
                f"draft model loaded on {device}, configuration asked for "
                f"{config.device}"
            )
        if config.dtype is not None:
            dtype = getattr(model, "dtype", None)
            if dtype is not None and str(dtype) != f"torch.{config.dtype}":
                raise PredictionConfigError(
                    f"draft model loaded as {dtype}, configuration asked for "
                    f"torch.{config.dtype}"
                )

    def describe(self) -> Dict[str, str]:
        return {
            "model": self.config.model_name_or_path,
            "revision": self._resolved_revision or UNKNOWN_REVISION,
            "vocab_fingerprint": self.vocab.fingerprint,
        }

    def predict(self, prefix: CommittedPrefix, max_tokens: int) -> DraftPrediction:
        if not isinstance(prefix, CommittedPrefix):
            raise PredictionConfigError("prediction requires a committed snapshot")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise PredictionConfigError("max_tokens must be an integer")
        if not 0 < max_tokens <= self.config.predict_tokens:
            raise PredictionConfigError(
                f"max_tokens must be within the configured budget of "
                f"{self.config.predict_tokens}"
            )
        if not prefix.tokens:
            raise PredictionConfigError("cannot predict from an empty prefix")
        if any(not self.vocab.contains(token) for token in prefix.tokens):
            raise PredictionConfigError(
                "committed prefix contains an id outside the shared vocabulary"
            )

        # The caller runs this inside prediction.run_isolated, so sampling here
        # cannot disturb the committed sampler. No grad: this branch never
        # contributes to committed state.
        input_ids = torch.tensor([list(prefix.tokens)], dtype=torch.long)
        with torch.no_grad():
            generated = self.model.generate(
                input_ids.to(self.config.device), max_new_tokens=max_tokens
            )
        produced = [int(t) for t in generated[0][len(prefix.tokens) :]]
        if not produced:
            raise PredictionConfigError("draft model produced no continuation")
        if any(not self.vocab.contains(token) for token in produced):
            raise PredictionConfigError(
                "draft model produced an id outside the shared vocabulary"
            )
        return DraftPrediction(
            request_id=prefix.request_id,
            prefix_version=prefix.version,
            tokens=tuple(produced[:max_tokens]),
            source=self.describe(),
        )
