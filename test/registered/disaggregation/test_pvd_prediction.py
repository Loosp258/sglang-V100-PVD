"""Draft/probe interfaces: configurability, isolation and vector-space safety.

CPU/logic only, with fake providers. No model is loaded, so nothing here says
anything about a real draft model's quality, a real probe's fidelity, or GPU
behaviour. What it does establish is that the surrounding machinery refuses
the unsafe combinations.
"""

import pytest
import torch
from sglang.srt.disaggregation.pvd.prediction import (
    UNKNOWN_REVISION,
    CommittedPrefix,
    DraftConfig,
    DraftPrediction,
    FakeDraftProvider,
    FakeTargetProbe,
    PredictionConfigError,
    PredictionPipeline,
    ProbeConfig,
    QueryVectors,
    TargetProbe,
    VectorSpaceError,
    run_isolated,
    snapshot_committed,
)

TARGET = "target/model-8b"
DRAFT = "draft/model-1b"


def make_prefix(request_id="req-1", tokens=(11, 12, 13), position=3, version="v0"):
    return snapshot_committed(request_id, tokens, position, version)


def make_pipeline(provider=None, probe=None, layers=(0, 1), predict_tokens=4):
    draft_cfg = DraftConfig(DRAFT, predict_tokens=predict_tokens)
    probe_cfg = ProbeConfig(TARGET, layers=layers)
    return PredictionPipeline(
        provider or FakeDraftProvider(draft_cfg),
        probe or FakeTargetProbe(probe_cfg),
        draft_cfg,
        probe_cfg,
    )


# --------------------------------------------------------------------------
# Configurability: no hard-coded model, optional revision
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["hf-org/small", "/mnt/models/local-draft"])
def test_any_name_or_local_path_is_accepted(name):
    assert DraftConfig(name).model_name_or_path == name


@pytest.mark.parametrize("bad", ["", "   ", None, 7])
def test_a_model_is_required_and_never_defaulted(bad):
    with pytest.raises(PredictionConfigError, match="model_name_or_path"):
        DraftConfig(bad)


def test_a_missing_revision_is_recorded_as_unknown_not_invented():
    assert DraftConfig(DRAFT).resolved_source() == {
        "model": DRAFT,
        "revision": UNKNOWN_REVISION,
    }


def test_a_supplied_revision_is_recorded_for_reproducibility():
    cfg = DraftConfig(DRAFT, revision="abc123")
    assert cfg.resolved_source()["revision"] == "abc123"


@pytest.mark.parametrize(
    "field,bad",
    [
        ("predict_tokens", 0),
        ("predict_tokens", -1),
        ("predict_tokens", True),
        ("predict_tokens", 1.0),
        ("max_prediction_batch", 0),
        ("revision", ""),
        ("dtype", "  "),
        ("device", ""),
    ],
)
def test_incoherent_configuration_is_refused_explicitly(field, bad):
    with pytest.raises(PredictionConfigError):
        DraftConfig(DRAFT, **{field: bad})


def test_the_draft_must_be_a_separate_model():
    cfg = DraftConfig(TARGET)
    probe_cfg = ProbeConfig(TARGET, layers=(0,))
    with pytest.raises(PredictionConfigError, match="separate"):
        PredictionPipeline(
            FakeDraftProvider(cfg), FakeTargetProbe(probe_cfg), cfg, probe_cfg
        )


# --------------------------------------------------------------------------
# The snapshot is immutable and detached
# --------------------------------------------------------------------------


def test_the_snapshot_copies_rather_than_aliases_committed_tokens():
    live = [1, 2, 3]
    prefix = snapshot_committed("req-1", live, 3, "v0")
    live.append(4)
    assert prefix.tokens == (1, 2, 3)


def test_a_provider_cannot_mutate_committed_state():
    prefix = make_prefix()
    with pytest.raises((AttributeError, TypeError)):
        prefix.tokens = (9,)
    with pytest.raises((AttributeError, TypeError)):
        prefix.committed_position = 99
    with pytest.raises((AttributeError, TypeError)):
        prefix.tokens.append(9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request_id": ""},
        {"version": ""},
        {"tokens": [1, 2]},
        {"tokens": (1, True)},
        {"committed_position": -1},
        {"committed_position": True},
        {"committed_position": 99},
    ],
)
def test_an_incoherent_snapshot_is_refused(kwargs):
    base = dict(request_id="r", tokens=(1, 2, 3), committed_position=3, version="v0")
    base.update(kwargs)
    with pytest.raises(PredictionConfigError):
        CommittedPrefix(**base)


# --------------------------------------------------------------------------
# RNG isolation
# --------------------------------------------------------------------------


def test_run_isolated_leaves_the_committed_rng_untouched():
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    run_isolated(lambda: torch.rand(16))
    assert torch.equal(torch.random.get_rng_state(), before)


def test_the_committed_sampler_produces_the_same_draw_after_a_prediction():
    torch.manual_seed(7)
    expected = torch.rand(4)

    torch.manual_seed(7)
    make_pipeline().run(make_prefix())
    assert torch.equal(torch.rand(4), expected)


def test_the_fake_provider_really_does_consume_rng():
    """Otherwise the isolation test above would pass vacuously."""
    torch.manual_seed(7)
    before = torch.random.get_rng_state().clone()
    cfg = DraftConfig(DRAFT)
    FakeDraftProvider(cfg).predict(make_prefix(), 4)
    assert not torch.equal(torch.random.get_rng_state(), before)


# --------------------------------------------------------------------------
# Predicted tokens are not committed output
# --------------------------------------------------------------------------


def test_a_prediction_does_not_advance_the_committed_position():
    prefix = make_prefix(position=3)
    make_pipeline().run(prefix)
    assert prefix.committed_position == 3
    assert prefix.tokens == (11, 12, 13)


def test_a_prediction_for_another_request_is_refused():
    class Foreign(FakeDraftProvider):
        def predict(self, prefix, max_tokens):
            return DraftPrediction("someone-else", prefix.version, (1, 2))

    with pytest.raises(PredictionConfigError, match="another request"):
        make_pipeline(provider=Foreign(DraftConfig(DRAFT))).run(make_prefix())


def test_a_prediction_against_a_stale_prefix_is_refused():
    class Stale(FakeDraftProvider):
        def predict(self, prefix, max_tokens):
            return DraftPrediction(prefix.request_id, "an-older-version", (1, 2))

    with pytest.raises(PredictionConfigError, match="stale prefix"):
        make_pipeline(provider=Stale(DraftConfig(DRAFT))).run(make_prefix())


def test_the_prediction_budget_is_enforced():
    class Greedy(FakeDraftProvider):
        def predict(self, prefix, max_tokens):
            return DraftPrediction(
                prefix.request_id, prefix.version, tuple(range(max_tokens + 3))
            )

    with pytest.raises(PredictionConfigError, match="token budget"):
        make_pipeline(provider=Greedy(DraftConfig(DRAFT))).run(make_prefix())


def test_the_provider_is_asked_for_exactly_its_configured_budget():
    provider = FakeDraftProvider(DraftConfig(DRAFT, predict_tokens=6))
    make_pipeline(provider=provider, predict_tokens=6).run(make_prefix())
    assert provider.calls == [("req-1", 6)]


def test_an_empty_prediction_is_refused():
    with pytest.raises(PredictionConfigError, match="at least one token"):
        DraftPrediction("r", "v0", ())


# --------------------------------------------------------------------------
# Vector space: draft Q must never be searched against target K
# --------------------------------------------------------------------------


def test_queries_in_the_draft_space_are_refused():
    probe_cfg = ProbeConfig(TARGET, layers=(0, 1))
    probe = FakeTargetProbe(probe_cfg, vector_space=DRAFT)
    with pytest.raises(VectorSpaceError, match=DRAFT):
        make_pipeline(probe=probe).run(make_prefix())


def test_queries_in_the_target_space_are_accepted():
    queries = make_pipeline().run(make_prefix())
    assert [q.layer for q in queries] == [0, 1]
    assert all(q.vector_space == TARGET for q in queries)
    assert all(q.valid_length == len(q.positions) for q in queries)


def test_an_unrequested_layer_is_refused():
    probe = FakeTargetProbe(ProbeConfig(TARGET, layers=(0, 5)))
    with pytest.raises(PredictionConfigError, match="unrequested layer"):
        make_pipeline(probe=probe, layers=(0, 1)).run(make_prefix())


def test_a_duplicate_layer_is_refused():
    class Duplicating(TargetProbe):
        def capture(self, prefix, prediction):
            q = QueryVectors(TARGET, "v", 0, 0, 1, (0, 1), 2)
            return (q, q)

    with pytest.raises(PredictionConfigError, match="duplicate layer"):
        make_pipeline(probe=Duplicating(), layers=(0, 1)).run(make_prefix())


def test_a_probe_that_returns_nothing_is_refused():
    class Empty(TargetProbe):
        def capture(self, prefix, prediction):
            return ()

    with pytest.raises(PredictionConfigError, match="no queries"):
        make_pipeline(probe=Empty()).run(make_prefix())


def test_a_probe_that_returns_a_foreign_object_is_refused():
    class Junk(TargetProbe):
        def capture(self, prefix, prediction):
            return ({"layer": 0},)

    with pytest.raises(PredictionConfigError, match="non-query"):
        make_pipeline(probe=Junk()).run(make_prefix())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vector_space": ""},
        {"version": ""},
        {"layer": -1},
        {"layer": True},
        {"head_count": 0},
        {"positions": ()},
        {"positions": [0, 1]},
        {"positions": (1, 0)},
        {"positions": (0, 0)},
        {"positions": (-1, 0)},
        {"valid_length": 0},
        {"valid_length": 3},
        {"valid_length": True},
    ],
)
def test_an_incoherent_query_is_refused(kwargs):
    base = dict(
        vector_space=TARGET,
        version="v",
        layer=0,
        head_start=0,
        head_count=1,
        positions=(0, 1),
        valid_length=2,
    )
    base.update(kwargs)
    with pytest.raises(PredictionConfigError):
        QueryVectors(**base)


def test_query_positions_start_after_the_committed_prefix():
    """committed_position is the refresh counter n, not a sequence index."""
    prefix = make_prefix(tokens=tuple(range(20)), position=4)
    queries = make_pipeline(predict_tokens=3).run(prefix)
    assert queries[0].positions == (20, 21, 22)
    assert prefix.committed_position == 4


# --------------------------------------------------------------------------
# Probe configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_model_id": ""},
        {"layers": ()},
        {"layers": [0]},
        {"layers": (0, 0)},
        {"layers": (-1,)},
        {"layers": (True,)},
        {"head_start": -1},
        {"head_count": 0},
    ],
)
def test_an_incoherent_probe_configuration_is_refused(kwargs):
    base = dict(target_model_id=TARGET, layers=(0, 1), head_start=0, head_count=2)
    base.update(kwargs)
    with pytest.raises(PredictionConfigError):
        ProbeConfig(**base)


def test_the_pipeline_reports_what_was_actually_loaded():
    pipeline = make_pipeline()
    assert pipeline.describe() == {
        "draft": {"model": DRAFT, "revision": UNKNOWN_REVISION},
        "target_model_id": TARGET,
        "layers": [0, 1],
    }


def test_the_pipeline_requires_a_snapshot_not_a_live_request():
    with pytest.raises(PredictionConfigError, match="committed snapshot"):
        make_pipeline().run({"request_id": "r", "tokens": (1,)})
