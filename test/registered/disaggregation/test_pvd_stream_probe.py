"""CPU-only SSE timing parser tests; not live Gateway evidence."""

import json

import pytest
import run_pvd_stream_probe as stream_probe


class Response:
    def __init__(self, counts, *, done=True, content_type="text/event-stream"):
        self.headers = {"Content-Type": content_type}
        self.lines = [
            (
                "data: "
                + json.dumps({"meta_info": {"completion_tokens": count}})
                + "\n"
            ).encode()
            for count in counts
        ]
        if done:
            self.lines.append(b"data: [DONE]\n")

    def __iter__(self):
        return iter(self.lines)


def test_each_token_event_yields_true_observed_intertoken_gaps(monkeypatch):
    times = iter((1.0, 2.0, 3.0, 4.0))
    monkeypatch.setattr(stream_probe.time, "perf_counter", lambda: next(times))
    result = stream_probe._observe(Response((1, 2, 3)), 0.0, 3)
    assert result["token_observation_times_seconds"] == [1.0, 2.0, 3.0]
    assert result["ttft_seconds"] == 1.0
    assert result["median_observed_intertoken_seconds"] == 1.0
    assert result["true_tpot_observable"]
    assert result["client_elapsed_seconds"] == 4.0


def test_coalesced_sse_cannot_be_claimed_as_true_tpot(monkeypatch):
    times = iter((1.0, 2.0, 3.0))
    monkeypatch.setattr(stream_probe.time, "perf_counter", lambda: next(times))
    result = stream_probe._observe(Response((1, 3)), 0.0, 3)
    assert result["token_observation_times_seconds"] == [1.0, 2.0, 2.0]
    assert result["coalesced_tokens"] == 1
    assert not result["true_tpot_observable"]


@pytest.mark.parametrize(
    "response,reason",
    [
        (Response((1, 2), done=False), "DONE"),
        (Response((1, 1, 0)), "regressed"),
        (Response((3,)), "excessive"),
        (Response((1,), content_type="application/json"), "SSE"),
    ],
)
def test_invalid_streams_are_refused(response, reason):
    with pytest.raises(ValueError, match=reason):
        stream_probe._observe(response, 0.0, 2)
