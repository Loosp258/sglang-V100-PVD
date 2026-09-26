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
    capture = collector((0,))
    capture.capture(0, torch.arange(5), torch.full((5, 32), float("nan")))
    with pytest.raises(PredictionConfigError, match="non-finite"):
        capture.finish()


def test_bound_positions_avoid_per_layer_host_transfer(monkeypatch):
    capture = collector()
    positions = torch.arange(5, dtype=torch.int64)
    capture.bind_positions(positions)

    def forbidden(_tensor):
        raise AssertionError("bound positions must not be copied to Python per layer")

    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    q = torch.arange(160.0).reshape(5, 32)
    capture.capture(0, positions, q)
    capture.capture(1, positions, q + 1)
    result = capture.finish()
    torch.testing.assert_close(result[0].vectors, q.reshape(5, 4, 8)[3:, 1:3])


def test_bound_positions_reject_alias_and_in_place_change():
    capture = collector((0,))
    positions = torch.arange(5, dtype=torch.int64)
    capture.bind_positions(positions)
    with pytest.raises(PredictionConfigError, match="changed after binding"):
        capture.capture(0, positions.clone(), torch.zeros(5, 32))
    positions.add_(1)
    with pytest.raises(PredictionConfigError, match="changed after binding"):
        capture.capture(0, positions, torch.zeros(5, 32))


def test_inference_positions_have_no_version_counter_but_are_checked_at_finish():
    capture = collector((0,))
    with torch.inference_mode():
        positions = torch.arange(5, dtype=torch.int64)
        assert positions.is_inference()
        capture.bind_positions(positions)
        capture.capture(0, positions, torch.ones(5, 32))
        assert capture.finish()[0].positions == (3, 4)

    changed = collector((0,))
    with torch.inference_mode():
        positions = torch.arange(5, dtype=torch.int64)
        changed.bind_positions(positions)
        changed.capture(0, positions, torch.ones(5, 32))
        positions.add_(1)
        with pytest.raises(PredictionConfigError, match="changed after binding"):
            changed.finish()


def test_bound_positions_validate_values_before_forward():
    capture = collector((0,))
    with pytest.raises(PredictionConfigError, match="complete prefix"):
        capture.bind_positions(torch.arange(1, 6, dtype=torch.int64))
    capture.bind_positions(torch.arange(5, dtype=torch.int64))
    with pytest.raises(PredictionConfigError, match="rebound"):
        capture.bind_positions(torch.arange(5, dtype=torch.int64))


def test_incremental_capture_keeps_absolute_positions_and_local_q_rows():
    prefix = CommittedPrefix("r", (1, 2, 3), 0, "v")
    capture = PostRopeQueryCapture(
        ProbeConfig("target", (0,), head_start=1, head_count=2),
        prefix,
        2,
        query_heads=4,
        head_dim=8,
        forward_start=3,
    )
    positions = torch.tensor([3, 4], dtype=torch.int64)
    capture.bind_positions(positions)
    q = torch.arange(64.0).reshape(2, 32)
    capture.capture(0, positions, q)
    result = capture.finish()[0]
    assert result.positions == (3, 4)
    torch.testing.assert_close(result.vectors, q.reshape(2, 4, 8)[:, 1:3])


def test_incremental_committed_capture_can_start_before_requested_rows():
    prefix = CommittedPrefix("r", (1, 2, 3, 4, 5), 2, "v2")
    capture = PostRopeQueryCapture(
        ProbeConfig("target", (0,)),
        prefix,
        0,
        query_heads=2,
        head_dim=4,
        committed_positions=(3, 4),
        forward_start=2,
    )
    q = torch.arange(24.0).reshape(3, 8)
    capture.capture(0, torch.tensor([2, 3, 4]), q)
    assert capture.finish()[0].positions == (3, 4)
    torch.testing.assert_close(capture.finish()[0].vectors, q.reshape(3, 2, 4)[1:, :1])


@pytest.mark.parametrize("start", [-1, True, 4, 5])
def test_incremental_capture_refuses_missing_query_rows(start):
    with pytest.raises(PredictionConfigError, match="forward start"):
        PostRopeQueryCapture(
            ProbeConfig("target", (0,)),
            CommittedPrefix("r", (1, 2, 3), 0, "v"),
            1,
            query_heads=2,
            head_dim=4,
            forward_start=start,
        )


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


def test_committed_capture_selects_actual_prefix_rows_without_appending():
    prefix = CommittedPrefix("r", (1, 2, 3), 2, "actual")
    capture = PostRopeQueryCapture(
        ProbeConfig("target", (0,), head_start=1, head_count=2),
        prefix,
        0,
        query_heads=4,
        head_dim=8,
        committed_positions=(1, 2),
    )
    q = torch.arange(96.0).reshape(3, 32)
    capture.capture(0, torch.arange(3), q)
    result = capture.finish()[0]
    assert result.positions == (1, 2)
    assert result.prefix_version == "actual"
    torch.testing.assert_close(result.vectors, q.reshape(3, 4, 8)[[1, 2], 1:3])
    assert result.vectors.data_ptr() != q.data_ptr()
    capture.close()


@pytest.mark.parametrize("positions", [(), (3,), (-1,), (True,), (2, 1), (1, 1)])
def test_committed_capture_refuses_invalid_positions(positions):
    with pytest.raises(PredictionConfigError, match="in-prefix positions"):
        PostRopeQueryCapture(
            ProbeConfig("target", (0,)),
            CommittedPrefix("r", (1, 2, 3), 2, "v"),
            0,
            query_heads=4,
            head_dim=8,
            committed_positions=positions,
        )


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
