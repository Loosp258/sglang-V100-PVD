"""The Hugging Face draft provider, with an injected loader and no weights.

No model is downloaded and `transformers` is never imported. These tests cover
the guards -- vocabulary compatibility, placement, budget -- and the shape of a
prediction. They say nothing about a real draft model's quality or speed.
"""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.draft_hf import (
    HuggingFaceDraftProvider,
    LoadedDraft,
    VocabularySignature,
    transformers_loader,
)
from sglang.srt.disaggregation.pvd.prediction import (
    UNKNOWN_REVISION,
    DraftConfig,
    PredictionConfigError,
    run_isolated,
    snapshot_committed,
)

DRAFT = "org/draft-small"


class FakeTokenizer:
    def __init__(self, vocab_size=32000, offset=0, bos=1, eos=2):
        self.vocab_size = vocab_size
        self.bos_token_id = bos
        self.eos_token_id = eos
        self._offset = offset

    def encode(self, text):
        return [self._offset + (ord(c) % 97) for c in text[:16]]


class FakeModel:
    """Appends deterministic ids, like a greedy decode would."""

    def __init__(self, device="cpu", dtype=None, new_tokens=None):
        self.device = device
        self.dtype = dtype
        self.calls = []
        self._new_tokens = new_tokens

    def generate(self, input_ids, max_new_tokens):
        self.calls.append((input_ids.shape[-1], max_new_tokens))
        torch.rand(1)  # a real draft model samples
        if self._new_tokens is not None:
            extra = list(self._new_tokens)[:max_new_tokens]
        else:
            last = int(input_ids[0, -1])
            extra = [last + i + 1 for i in range(max_new_tokens)]
        return torch.cat([input_ids, torch.tensor([extra], dtype=torch.long)], dim=-1)


def make_loader(tokenizer=None, model=None, revision=None):
    def loader(config):
        return LoadedDraft(
            tokenizer or FakeTokenizer(),
            model or FakeModel(device=config.device),
            revision,
        )

    return loader


def target_signature(tokenizer=None):
    return VocabularySignature.from_tokenizer(tokenizer or FakeTokenizer())


def make_provider(config=None, tokenizer=None, model=None, revision=None, target=None):
    return HuggingFaceDraftProvider(
        config or DraftConfig(DRAFT, predict_tokens=4),
        target or target_signature(),
        loader=make_loader(tokenizer, model, revision),
    )


def make_prefix(tokens=(10, 11, 12)):
    return snapshot_committed("req-1", tokens, len(tokens), "v0")


# --------------------------------------------------------------------------
# Vocabulary compatibility
# --------------------------------------------------------------------------


def test_matching_tokenizers_load():
    provider = make_provider()
    assert provider.vocab == target_signature()


@pytest.mark.parametrize(
    "tokenizer",
    [
        FakeTokenizer(vocab_size=50000),
        FakeTokenizer(offset=1),
        FakeTokenizer(bos=99),
        FakeTokenizer(eos=99),
    ],
)
def test_an_incompatible_vocabulary_is_refused(tokenizer):
    with pytest.raises(PredictionConfigError, match="incompatible"):
        make_provider(tokenizer=tokenizer)


def test_added_special_id_is_valid_but_a_padding_hole_is_not():
    class AddedTokenTokenizer(FakeTokenizer):
        def __init__(self):
            super().__init__(vocab_size=10, eos=12)

        def encode(self, text):
            return [1, 2, 3]

        def get_vocab(self):
            return {str(token_id): token_id for token_id in range(10)} | {"eos": 12}

    tokenizer = AddedTokenTokenizer()
    vocabulary = VocabularySignature.from_tokenizer(tokenizer)
    assert vocabulary.size == 10
    assert vocabulary.exact_mapping_available
    assert vocabulary.contains(12)
    assert not vocabulary.contains(11)
    assert not vocabulary.contains(13)

    accepted = make_provider(
        tokenizer=tokenizer,
        model=FakeModel(new_tokens=[12]),
        target=vocabulary,
    )
    assert accepted.predict(make_prefix(tokens=(1, 12)), 1).tokens == (12,)

    refused = make_provider(
        tokenizer=tokenizer,
        model=FakeModel(new_tokens=[11]),
        target=vocabulary,
    )
    with pytest.raises(PredictionConfigError, match="outside the shared vocabulary"):
        refused.predict(make_prefix(tokens=(1, 12)), 1)


def test_full_mapping_fingerprint_catches_unprobed_token_swap():
    class MappedTokenizer(FakeTokenizer):
        def __init__(self, swapped):
            super().__init__(vocab_size=6)
            self.swapped = swapped

        def encode(self, text):
            return [1, 2, 3]

        def get_vocab(self):
            return {
                "one": 1,
                "two": 2,
                "three": 3,
                "alpha": 5 if self.swapped else 4,
                "beta": 4 if self.swapped else 5,
                "zero": 0,
            }

    first = VocabularySignature.from_tokenizer(MappedTokenizer(False))
    second = VocabularySignature.from_tokenizer(MappedTokenizer(True))
    assert first.allowed_ids == second.allowed_ids
    assert first.fingerprint == second.fingerprint
    assert first.mapping_fingerprint != second.mapping_fingerprint
    with pytest.raises(PredictionConfigError, match="incompatible"):
        make_provider(tokenizer=MappedTokenizer(True), target=first)


def test_tokenizer_without_mapping_is_marked_non_exact():
    signature = VocabularySignature.from_tokenizer(FakeTokenizer())
    assert not signature.exact_mapping_available


def test_exact_mapping_refuses_a_special_id_missing_from_get_vocab():
    class MissingSpecial(FakeTokenizer):
        all_special_ids = [12]

        def __init__(self):
            super().__init__(vocab_size=10, eos=12)

        def encode(self, text):
            return [1]

        def get_vocab(self):
            return {str(token_id): token_id for token_id in range(10)}

    with pytest.raises(PredictionConfigError, match="special ID absent"):
        VocabularySignature.from_tokenizer(MissingSpecial())


def test_a_target_signature_is_mandatory():
    with pytest.raises(PredictionConfigError, match="vocabulary signature"):
        HuggingFaceDraftProvider(DraftConfig(DRAFT), None, loader=make_loader())


@pytest.mark.parametrize(
    "tokenizer",
    [
        SimpleNamespace(vocab_size=0, encode=lambda t: [1]),
        SimpleNamespace(vocab_size=True, encode=lambda t: [1]),
        SimpleNamespace(vocab_size=10, encode=lambda t: []),
    ],
)
def test_an_unusable_tokenizer_is_refused(tokenizer):
    with pytest.raises(PredictionConfigError):
        VocabularySignature.from_tokenizer(tokenizer)


def test_a_tokenizer_that_raises_is_reported_not_swallowed():
    def explode(text):
        raise RuntimeError("no vocab file")

    with pytest.raises(PredictionConfigError, match="could not encode"):
        VocabularySignature.from_tokenizer(
            SimpleNamespace(vocab_size=10, encode=explode)
        )


# --------------------------------------------------------------------------
# Placement and configuration
# --------------------------------------------------------------------------


def test_a_model_on_the_wrong_device_is_refused():
    config = DraftConfig(DRAFT, device="cuda:0")
    with pytest.raises(PredictionConfigError, match="loaded on cpu"):
        make_provider(config=config, model=FakeModel(device="cpu"))


def test_a_model_with_the_wrong_dtype_is_refused():
    config = DraftConfig(DRAFT, dtype="float16")
    with pytest.raises(PredictionConfigError, match="float32"):
        make_provider(config=config, model=FakeModel(dtype=torch.float32))


def test_a_matching_dtype_loads():
    config = DraftConfig(DRAFT, dtype="float16")
    make_provider(config=config, model=FakeModel(dtype=torch.float16))


def test_a_loader_that_returns_nothing_is_refused():
    with pytest.raises(PredictionConfigError, match="no model"):
        HuggingFaceDraftProvider(
            DraftConfig(DRAFT), target_signature(), loader=lambda c: None
        )


def test_a_config_is_required():
    with pytest.raises(PredictionConfigError, match="DraftConfig"):
        HuggingFaceDraftProvider("org/draft", target_signature())


def test_the_missing_dependency_message_names_the_package(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_transformers(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("no module named transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_transformers)
    with pytest.raises(PredictionConfigError, match="transformers"):
        transformers_loader(DraftConfig(DRAFT))


# --------------------------------------------------------------------------
# Revision reporting
# --------------------------------------------------------------------------


def test_an_unresolvable_revision_is_reported_as_unknown():
    assert make_provider().describe()["revision"] == UNKNOWN_REVISION


def test_a_resolved_revision_is_reported():
    provider = make_provider(revision="deadbeef")
    assert provider.describe()["revision"] == "deadbeef"


def test_the_configured_revision_is_used_when_the_loader_resolves_none():
    provider = make_provider(config=DraftConfig(DRAFT, revision="v1.2"))
    assert provider.describe()["revision"] == "v1.2"


def test_the_report_names_the_model_and_its_vocabulary():
    report = make_provider().describe()
    assert report["model"] == DRAFT
    assert report["vocab_fingerprint"] == target_signature().fingerprint


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------


def test_a_prediction_follows_the_committed_prefix():
    provider = make_provider()
    prediction = provider.predict(make_prefix((10, 11, 12)), 3)
    assert prediction.tokens == (13, 14, 15)
    assert prediction.request_id == "req-1"
    assert prediction.prefix_version == "v0"


def test_the_model_is_asked_for_exactly_the_requested_tokens():
    model = FakeModel()
    provider = make_provider(model=model)
    provider.predict(make_prefix(), 2)
    assert model.calls == [(3, 2)]


def test_a_longer_continuation_is_truncated_to_the_request():
    model = FakeModel(new_tokens=[7, 8, 9, 10, 11])
    provider = make_provider(model=model)
    assert len(provider.predict(make_prefix(), 2).tokens) == 2


class OverProducingModel(FakeModel):
    """Ignores max_new_tokens, as a misconfigured generate() can."""

    def generate(self, input_ids, max_new_tokens):
        self.calls.append((input_ids.shape[-1], max_new_tokens))
        extra = [900 + i for i in range(max_new_tokens + 3)]
        return torch.cat([input_ids, torch.tensor([extra], dtype=torch.long)], dim=-1)


def test_a_model_that_ignores_the_token_limit_is_still_truncated():
    """The budget is enforced on what is returned, not only on what is asked."""
    provider = make_provider(model=OverProducingModel())
    prediction = provider.predict(make_prefix(), 2)
    assert prediction.tokens == (900, 901)


@pytest.mark.parametrize("bad", [0, -1, 5, True, 1.0])
def test_a_request_outside_the_budget_is_refused(bad):
    provider = make_provider(config=DraftConfig(DRAFT, predict_tokens=4))
    with pytest.raises(PredictionConfigError):
        provider.predict(make_prefix(), bad)


def test_an_empty_prefix_is_refused():
    with pytest.raises(PredictionConfigError, match="empty prefix"):
        make_provider().predict(snapshot_committed("r", (), 0, "v0"), 2)


def test_a_prefix_outside_the_shared_vocabulary_is_refused():
    provider = make_provider()
    with pytest.raises(PredictionConfigError, match="outside the shared vocabulary"):
        provider.predict(make_prefix((10, 999_999)), 2)


def test_a_model_that_returns_nothing_is_refused():
    provider = make_provider(model=FakeModel(new_tokens=[]))
    with pytest.raises(PredictionConfigError, match="no continuation"):
        provider.predict(make_prefix(), 2)


def test_a_live_request_object_is_refused():
    with pytest.raises(PredictionConfigError, match="committed snapshot"):
        make_provider().predict({"tokens": (1, 2)}, 2)


def test_predicting_does_not_disturb_the_committed_sampler():
    provider = make_provider()
    torch.manual_seed(99)
    expected = torch.rand(4)

    torch.manual_seed(99)
    run_isolated(provider.predict, make_prefix(), 3)
    assert torch.equal(torch.rand(4), expected)


def test_the_fake_model_really_consumes_rng():
    """Otherwise the isolation test above would pass vacuously."""
    provider = make_provider()
    torch.manual_seed(99)
    before = torch.random.get_rng_state().clone()
    provider.predict(make_prefix(), 3)
    assert not torch.equal(torch.random.get_rng_state(), before)


def test_prediction_leaves_the_snapshot_untouched():
    prefix = make_prefix((10, 11, 12))
    make_provider().predict(prefix, 3)
    assert prefix.tokens == (10, 11, 12)
    assert prefix.committed_position == 3
