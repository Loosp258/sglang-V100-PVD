"""CPU-only contract tests for the bounded live load probe."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import run_pvd_live_load as live_load


class Response:
    def __init__(
        self, counts, *, prompt_tokens=109, done=True, finish_reason=None, texts=None
    ):
        self.headers = {"Content-Type": "text/event-stream"}
        self.lines = [
            (
                "data: "
                + json.dumps(
                    {
                        **({"text": text} if texts is not None else {}),
                        "meta_info": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": count,
                            "finish_reason": finish_reason,
                        },
                    }
                )
                + "\n"
            ).encode()
            for count, text in zip(
                counts,
                texts if texts is not None else (None,) * len(counts),
                strict=True,
            )
        ]
        if done:
            self.lines.append(b"data: [DONE]\n")

    def __iter__(self):
        return iter(self.lines)


def test_observe_reports_prompt_and_token_gaps(monkeypatch):
    times = iter((1.0, 2.0, 3.0))
    monkeypatch.setattr(live_load.time, "perf_counter", lambda: next(times))
    result = live_load._observe(Response((1, 2)), 0.0, 2)
    assert result["prompt_tokens"] == 109
    assert result["completion_tokens"] == 2
    assert result["ttft_seconds"] == 1.0
    assert result["median_observed_gap_seconds"] == 1.0
    assert result["elapsed_seconds"] == 3.0
    assert result["true_tpot_observable"]


def test_observe_does_not_call_coalesced_events_true_tpot(monkeypatch):
    times = iter((1.0, 2.0))
    monkeypatch.setattr(live_load.time, "perf_counter", lambda: next(times))
    result = live_load._observe(Response((3,)), 0.0, 3)
    assert result["coalesced_tokens"] == 2
    assert not result["true_tpot_observable"]


@pytest.mark.parametrize(
    "response,reason",
    [
        (Response((1,), done=False), "DONE"),
        (Response((2, 1)), "regressed"),
        (Response((1,), prompt_tokens=0), "prompt token count"),
    ],
)
def test_observe_rejects_incomplete_or_inconsistent_stream(response, reason):
    with pytest.raises(ValueError, match=reason) as exc:
        live_load._observe(response, 0.0, 2)
    if reason == "DONE":
        assert "received=1/2 events=1 finish_type=None" in str(exc.value)


def test_observe_reports_early_finish_type_without_response_content():
    response = Response((1,), finish_reason={"type": "stop", "matched": "secret"})
    with pytest.raises(ValueError, match="finish_type='stop'") as exc:
        live_load._observe(response, 0.0, 2)
    assert "secret" not in str(exc.value)


def test_observe_hashes_only_the_final_cumulative_text(monkeypatch):
    times = iter((1.0, 2.0, 3.0))
    monkeypatch.setattr(live_load.time, "perf_counter", lambda: next(times))
    result = live_load._observe(Response((1, 2), texts=("A", "AB")), 0.0, 2)
    assert result["output_sha256"] == hashlib.sha256(b"AB").hexdigest()


def test_observe_refuses_non_text_sse_output():
    with pytest.raises(ValueError, match="output text"):
        live_load._observe(Response((1,), texts=(123,)), 0.0, 1)


def test_collect_bounds_actual_gateway_prompt_and_summarizes_concurrency(monkeypatch):
    def fake_request(url, text, expected_tokens, timeout, barrier):
        barrier.wait()
        assert url == "http://gateway"
        assert "PVD_LOAD_" in text and expected_tokens == 2 and timeout == 5
        return 10.0, {
            "prompt_tokens": 109,
            "completion_tokens": 2,
            "sse_events": 2,
            "coalesced_tokens": 0,
            "ttft_seconds": 1.0,
            "elapsed_seconds": 2.0,
            "median_observed_gap_seconds": 1.0,
            "max_observed_gap_seconds": 1.0,
            "true_tpot_observable": True,
        }

    monkeypatch.setattr(live_load, "_request", fake_request)
    args = SimpleNamespace(
        gateway_url="http://gateway",
        clients=2,
        rounds=2,
        sentence="Test sentence.",
        repetitions=10,
        max_new_tokens=2,
        timeout_seconds=5,
        min_prompt_tokens=100,
        max_prompt_tokens=120,
    )
    report = live_load.collect(args)
    assert report["schema"] == "pvd.live_load.v1"
    assert report["all_complete"] and report["all_true_tpot_observable"]
    assert len(report["rounds"]) == 2
    assert [len(item["requests"]) for item in report["rounds"]] == [2, 2]
    assert report["latency_p95_nearest_rank_seconds"] == 2.0
    args.max_prompt_tokens = 108
    with pytest.raises(ValueError, match="outside requested range"):
        live_load.collect(args)


def test_collect_fixed_prefix_is_repeatable_and_single_request_only(monkeypatch):
    texts = []

    def fake_request(url, text, expected_tokens, timeout, barrier):
        barrier.wait()
        texts.append(text)
        return 10.0, {
            "prompt_tokens": 109,
            "completion_tokens": 1,
            "sse_events": 1,
            "coalesced_tokens": 0,
            "ttft_seconds": 1.0,
            "elapsed_seconds": 2.0,
            "median_observed_gap_seconds": None,
            "max_observed_gap_seconds": None,
            "true_tpot_observable": True,
        }

    monkeypatch.setattr(live_load, "_request", fake_request)
    args = SimpleNamespace(
        gateway_url="http://gateway",
        clients=1,
        rounds=1,
        sentence="Test sentence.",
        repetitions=10,
        max_new_tokens=1,
        timeout_seconds=5,
        min_prompt_tokens=100,
        max_prompt_tokens=120,
        fixed_prefix=True,
    )
    assert live_load.collect(args)["fixed_prefix"] is True
    assert live_load.collect(args)["fixed_prefix"] is True
    assert texts == ["PVD_LOAD_FIXED: " + ("Test sentence. " * 10)] * 2
    args.clients = 2
    with pytest.raises(ValueError, match="bounded"):
        live_load.collect(args)


def test_collect_replay_seed_reuses_distinct_prompts_for_concurrent_clients(
    monkeypatch,
):
    observed = []

    def fake_request(url, text, expected_tokens, timeout, barrier):
        barrier.wait()
        observed.append(text)
        return 10.0, {
            "prompt_tokens": 109,
            "completion_tokens": 1,
            "sse_events": 1,
            "coalesced_tokens": 0,
            "ttft_seconds": 1.0,
            "elapsed_seconds": 2.0,
            "median_observed_gap_seconds": None,
            "max_observed_gap_seconds": None,
            "true_tpot_observable": True,
        }

    monkeypatch.setattr(live_load, "_request", fake_request)
    args = SimpleNamespace(
        gateway_url="http://gateway",
        clients=4,
        rounds=1,
        sentence="Test sentence.",
        repetitions=10,
        max_new_tokens=1,
        timeout_seconds=5,
        min_prompt_tokens=100,
        max_prompt_tokens=120,
        fixed_prefix=False,
        replay_seed="matched_20260926",
    )
    first = live_load.collect(args)
    first_prompts = sorted(observed)
    observed.clear()
    second = live_load.collect(args)
    assert first["run_id"] == second["run_id"] == args.replay_seed
    assert first["replay_seed"] == args.replay_seed
    assert sorted(observed) == first_prompts
    assert len(set(first_prompts)) == 4
    assert all("PVD_LOAD_matched_20260926_0_" in text for text in first_prompts)

    args.replay_seed = "../unsafe"
    with pytest.raises(ValueError, match="bounded"):
        live_load.collect(args)
    args.replay_seed = "matched_20260926"
    args.fixed_prefix = True
    with pytest.raises(ValueError, match="bounded"):
        live_load.collect(args)
