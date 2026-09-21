"""Pure collector contracts; actual model execution has a separate strict smoke."""

import pytest
import torch
from sglang.srt.disaggregation.pvd.prediction import (
    CommittedPrefix,
    PredictionConfigError,
    ProbeConfig,
)
from sglang.srt.disaggregation.pvd.target_probe import PostRopeQueryCapture


def collector(layers=(0, 1)):
    return PostRopeQueryCapture(
        ProbeConfig("target", layers, head_start=1, head_count=2),
        CommittedPrefix("req", (1, 2, 3), 0, "prefix-v1"),
        2,
        query_heads=4,
        head_dim=8,
    )


def test_captured_rows_are_prediction_positions_and_global_query_heads():
    capture = collector()
    q = torch.arange(160.0).reshape(5, 32)
    expected = q.reshape(5, 4, 8)[3:, 1:3].clone()
    capture.capture(0, torch.arange(5), q)
    capture.capture(1, torch.arange(5), q + 1)
    q.zero_()
    result = capture.finish()
    assert len(result) == 2
    torch.testing.assert_close(result[0].vectors, expected)
    torch.testing.assert_close(result[1].vectors, expected + 1)
    assert result[0].positions == (3, 4)
    assert result[0].positional_encoding == "rope_applied"
    assert result[0].request_id == "req"
    assert result[0].prefix_version == "prefix-v1"
    assert result[0].head_start == 1 and result[0].head_count == 2
    assert result[0].version == result[1].version
    capture.close()
    with pytest.raises(PredictionConfigError, match="capture every"):
        capture.finish()


def test_requested_layer_missing_is_not_silently_omitted():
    capture = collector()
    capture.capture(0, torch.arange(5), torch.zeros(5, 32))
    with pytest.raises(PredictionConfigError, match="every requested layer"):
        capture.finish()


def test_real_collector_uses_the_existing_v_search_encoding_contract():
    from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED

    capture = collector((0,))
    capture.capture(0, torch.arange(5), torch.ones(5, 32))
    assert capture.finish()[0].positional_encoding == ROPE_APPLIED


def test_duplicate_capture_is_refused():
    capture = collector((0,))
    capture.capture(0, torch.arange(5), torch.zeros(5, 32))
    with pytest.raises(PredictionConfigError, match="twice"):
        capture.capture(0, torch.arange(5), torch.zeros(5, 32))


@pytest.mark.parametrize("positions", [torch.arange(5) + 1, torch.arange(4)])
def test_wrong_positions_are_refused(positions):
    with pytest.raises(PredictionConfigError, match="positions"):
        collector().capture(0, positions, torch.zeros(5, 32))


@pytest.mark.parametrize("shape", [(4, 32), (5, 16), (5, 4, 8)])
def test_wrong_head_shape_is_refused(shape):
    with pytest.raises(PredictionConfigError, match="shape"):
        collector().capture(0, torch.arange(5), torch.zeros(*shape))


def test_nonfinite_queries_are_refused():
    with pytest.raises(PredictionConfigError, match="non-finite"):
        collector().capture(0, torch.arange(5), torch.full((5, 32), float("nan")))


def test_head_bounds_are_checked():
    with pytest.raises(PredictionConfigError, match="Q heads"):
        PostRopeQueryCapture(
            ProbeConfig("target", (0,), head_count=5),
            CommittedPrefix("r", (1,), 0, "v"),
            1,
            query_heads=4,
            head_dim=8,
        )


def test_close_prevents_late_writes():
    capture = collector()
    capture.close()
    with pytest.raises(PredictionConfigError, match="closed"):
        capture.capture(0, torch.arange(5), torch.zeros(5, 32))


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_prediction_count_is_refused(bad):
    with pytest.raises(PredictionConfigError, match="positive integers"):
        PostRopeQueryCapture(
            ProbeConfig("target", (0,)),
            CommittedPrefix("r", (1,), 0, "v"),
            bad,
            query_heads=4,
            head_dim=8,
        )
