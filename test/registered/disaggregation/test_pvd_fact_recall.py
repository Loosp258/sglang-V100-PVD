"""Deterministic varied-fact load probe without a live Gateway."""

import hashlib
import json
import re

import pytest
import run_pvd_fact_recall as probe


class SSEResponse:
    def __init__(self, events=(), *, done=True, content_type="text/event-stream"):
        self.status = 200
        self.headers = {"Content-Type": content_type}
        self.lines = []
        for event in events:
            self.lines.extend(
                (
                    b"event: message\r\n",
                    b"data: " + json.dumps(event).encode("utf-8") + b"\r\n",
                    b"\r\n",
                )
            )
        if done:
            self.lines.append(b"data: [DONE]\r\n")
        self.read_limits = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def readline(self, size=-1):
        self.read_limits.append(size)
        if not self.lines:
            return b""
        line = self.lines.pop(0)
        if size >= 0 and len(line) > size:
            self.lines.insert(0, line[size:])
            return line[:size]
        return line


def test_fact_prompts_are_reproducible_unique_and_answered_by_one_record():
    first, expected = probe.make_prompt("quality_20260927", 2, records=48)
    assert (first, expected) == probe.make_prompt("quality_20260927", 2, records=48)
    assert first != probe.make_prompt("quality_20260927", 3, records=48)[0]
    codes = re.findall(r"access code is (\d{5})", first)
    assert len(codes) == len(set(codes)) == 48
    target = re.search(
        r"What is the five-digit access code for record ID (R\d{3})", first
    )
    assert target is not None
    record = re.search(
        rf"Record ID {target.group(1)}: .*access code is (\d{{5}})", first
    )
    assert record is not None and record.group(1) == expected


@pytest.mark.parametrize(
    "seed,case,records", [("../bad", 0, 48), ("ok", -1, 48), ("ok", 0, 641)]
)
def test_fact_prompt_refuses_unbounded_shape(seed, case, records):
    with pytest.raises(ValueError):
        probe.make_prompt(seed, case, records=records)


@pytest.mark.parametrize(
    "target_tokens,records",
    [(4096, 139), (8192, 278), (16384, 557)],
)
def test_long_prompt_profiles_are_bounded_and_reproducible(target_tokens, records):
    assert probe.records_for_target_prompt_tokens(target_tokens) == records
    prompt, expected = probe.make_prompt("long_context_20260927", 0, records=records)
    assert prompt == probe.make_prompt("long_context_20260927", 0, records=records)[0]
    assert len(re.findall(r"^Record ID R\d{3}:", prompt, flags=re.MULTILINE)) == records
    assert re.search(rf"access code is {expected}\.", prompt)


def test_long_prompt_collect_reports_actual_tokens_and_input_hash(monkeypatch):
    seen = []

    def fake_request(url, text, expected, max_tokens, timeout):
        assert url == "http://gateway" and max_tokens == 128 and timeout == 5
        input_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        seen.append(input_hash)
        return {
            "input_sha256": input_hash,
            "output_sha256": "o" * 64,
            "expected_code": expected,
            "first_code": expected,
            "first_code_matches": True,
            "prompt_tokens": 4096,
            "completion_tokens": 128,
            "elapsed_seconds": 0.01,
        }

    monkeypatch.setattr(probe, "_request", fake_request)
    reports = [
        probe.collect(
            "http://gateway",
            "long_context_20260927",
            cases=2,
            target_prompt_tokens=4096,
            max_tokens=128,
            timeout=5,
            concurrency=1,
        )
        for _ in range(2)
    ]
    assert reports[0]["input_set_sha256"] == reports[1]["input_set_sha256"]
    assert [item["input_sha256"] for item in reports[0]["results"]] == [
        item["input_sha256"] for item in reports[1]["results"]
    ]
    assert reports[0]["target_prompt_tokens"] == 4096
    assert reports[0]["prompt_tokens_min"] == reports[0]["prompt_tokens_max"] == 4096
    assert reports[0]["records_per_case"] == 139
    assert len(seen) == 4


def test_stream_parser_hashes_cumulative_text_and_reports_token_timing(monkeypatch):
    response = SSEResponse(
        (
            {
                "text": "The ",
                "meta_info": {"completion_tokens": 1, "prompt_tokens": 4096},
            },
            {
                "text": "The code is 12345",
                "meta_info": {"completion_tokens": 3, "prompt_tokens": 4096},
            },
        )
    )
    times = iter((1.0, 3.0, 4.0))
    monkeypatch.setattr(probe.time, "perf_counter", lambda: next(times))

    result = probe._observe_stream(
        response, 0.0, "long prompt", "12345", 3, "cumulative"
    )

    assert result["input_sha256"] == hashlib.sha256(b"long prompt").hexdigest()
    assert result["output_sha256"] == hashlib.sha256(b"The code is 12345").hexdigest()
    assert result["first_code_matches"] is True
    assert result["prompt_tokens"] == 4096 and result["completion_tokens"] == 3
    assert result["ttft_seconds"] == 1.0
    assert result["intertoken_gap_p50_seconds"] == 0.0
    assert result["intertoken_gap_p95_seconds"] == 2.0
    assert result["intertoken_gap_max_seconds"] == 2.0
    assert result["early_32_gap_p50_seconds"] is None
    assert result["late_32_gap_p50_seconds"] is None
    assert result["completion_seconds"] == 3.0
    assert result["decode_seconds"] == 2.0 and result["wall_seconds"] == 4.0
    assert result["coalesced_tokens"] == 1
    assert result["true_tpot_observable"] is False
    assert max(response.read_limits) == probe.MAX_SSE_LINE_BYTES + 1


def test_decode_windows_exclude_middle_refresh_gap():
    gaps = [0.1] * 32 + [8.0] + [0.15] * 62 + [0.2] * 32
    assert probe._decode_window_p50(gaps) == (0.1, 0.2)
    assert probe._decode_window_p50(gaps[:20]) == (None, None)


def test_stream_delta_text_has_same_output_hash_as_final_text():
    response = SSEResponse(
        (
            {"text": "The ", "meta_info": {"completion_tokens": 1, "prompt_tokens": 9}},
            {
                "text": "code is ",
                "meta_info": {"completion_tokens": 2, "prompt_tokens": 9},
            },
            {
                "text": "12345",
                "meta_info": {"completion_tokens": 3, "prompt_tokens": 9},
            },
        )
    )

    result = probe._observe_stream(
        response, probe.time.perf_counter(), "p", "12345", 3, "delta"
    )

    assert result["output_sha256"] == hashlib.sha256(b"The code is 12345").hexdigest()
    assert result["first_code_matches"] is True


def test_stream_and_nonstream_collect_hash_the_same_prompt_and_output(monkeypatch):
    seed = "stream_compare_20260927"
    prompt, expected = probe.make_prompt(seed, 0, records=139)

    def fake_nonstream(url, text, expected_code, max_tokens, timeout):
        assert text == prompt and expected_code == expected
        assert max_tokens == 128 and timeout == 5
        return {
            "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "output_sha256": hashlib.sha256(expected_code.encode("utf-8")).hexdigest(),
            "expected_code": expected_code,
            "first_code": expected_code,
            "first_code_matches": True,
            "prompt_tokens": 4096,
            "completion_tokens": 128,
            "elapsed_seconds": 0.1,
        }

    monkeypatch.setattr(probe, "_request", fake_nonstream)
    common = {
        "cases": 1,
        "target_prompt_tokens": 4096,
        "max_tokens": 128,
        "timeout": 5,
        "concurrency": 1,
    }
    nonstream = probe.collect("http://gateway", seed, **common)

    def fake_urlopen(request, timeout):
        body = json.loads(request.data)
        assert timeout == 5 and body["stream"] is True
        assert body["text"] == prompt
        assert body["sampling_params"]["max_new_tokens"] == 128
        return SSEResponse(
            (
                {
                    "text": expected,
                    "meta_info": {"completion_tokens": 128, "prompt_tokens": 4096},
                },
            )
        )

    monkeypatch.setattr(probe.urllib.request, "urlopen", fake_urlopen)
    streaming = probe.collect("http://gateway", seed, stream=True, **common)

    assert streaming["input_set_sha256"] == nonstream["input_set_sha256"]
    assert (
        streaming["results"][0]["input_sha256"]
        == nonstream["results"][0]["input_sha256"]
    )
    assert (
        streaming["results"][0]["output_sha256"]
        == nonstream["results"][0]["output_sha256"]
    )
    assert streaming["results"][0]["first_code_matches"] is True
    assert streaming["stream_metrics"]["coalesced_tokens"] == 127
    assert streaming["stream_metrics"]["intertoken_gap_max_seconds"] == 0.0
    assert "_intertoken_gaps_seconds" not in streaming["results"][0]


@pytest.mark.parametrize(
    "response,max_tokens,reason",
    [
        (SSEResponse(content_type="application/json"), 1, "SSE"),
        (
            SSEResponse(
                (
                    {
                        "text": "a",
                        "meta_info": {"completion_tokens": 1, "prompt_tokens": 1},
                    },
                ),
                done=False,
            ),
            1,
            "DONE",
        ),
        (
            SSEResponse(
                (
                    {
                        "text": "a",
                        "meta_info": {"completion_tokens": 1, "prompt_tokens": 1},
                    },
                    {
                        "text": "a",
                        "meta_info": {"completion_tokens": 0, "prompt_tokens": 1},
                    },
                )
            ),
            2,
            "regressed",
        ),
        (
            SSEResponse(
                (
                    {
                        "text": "a",
                        "meta_info": {"completion_tokens": 2, "prompt_tokens": 1},
                    },
                )
            ),
            1,
            "excessive",
        ),
        (SSEResponse(), 1, "SSE stream ended"),
    ],
)
def test_stream_parser_rejects_invalid_or_incomplete_sse(response, max_tokens, reason):
    with pytest.raises(ValueError, match=reason):
        probe._observe_stream(
            response,
            probe.time.perf_counter(),
            "p",
            "12345",
            max_tokens,
            "cumulative",
        )


def test_stream_parser_bounds_sse_line_and_output_size():
    oversized_line = SSEResponse()
    oversized_line.lines = [b"data: " + b"x" * probe.MAX_SSE_LINE_BYTES]
    with pytest.raises(ValueError, match="SSE line exceeds"):
        probe._observe_stream(
            oversized_line, probe.time.perf_counter(), "p", "12345", 1, "cumulative"
        )

    oversized_text = SSEResponse(
        (
            {
                "text": "x" * (probe.MAX_OUTPUT_CHARS + 1),
                "meta_info": {"completion_tokens": 1, "prompt_tokens": 1},
            },
        )
    )
    with pytest.raises(ValueError, match="unbounded text"):
        probe._observe_stream(
            oversized_text, probe.time.perf_counter(), "p", "12345", 1, "cumulative"
        )


def test_fact_collect_reports_correctness_and_hashes_without_raw_output(monkeypatch):
    seen = []

    def fake_request(url, text, expected, max_tokens, timeout):
        assert url == "http://gateway" and max_tokens == 20 and timeout == 5
        seen.append((text, expected))
        return {
            "input_sha256": "i" * 64,
            "output_sha256": "o" * 64,
            "expected_code": expected,
            "first_code": expected if len(seen) % 2 == 0 else "00000",
            "first_code_matches": len(seen) % 2 == 0,
            "prompt_tokens": 900,
            "completion_tokens": 20,
            "elapsed_seconds": float(len(seen)),
        }

    monkeypatch.setattr(probe, "_request", fake_request)
    result = probe.collect(
        "http://gateway",
        "quality_20260927",
        cases=2,
        records=48,
        max_tokens=20,
        timeout=5,
    )
    assert result["schema"] == "pvd.fact_recall.v1"
    assert result["correct"] == 1
    assert len(seen) == 2 and len({text for text, _ in seen}) == 2
    assert all("text" not in item for item in result["results"])
    assert result["waves"][0]["requests"] == 2
    assert result["synchronized_start"] is False


def test_fact_collect_barrier_covers_each_bounded_wave(monkeypatch):
    barriers = []

    class RecordingBarrier:
        def __init__(self, parties):
            self.parties, self.waits = parties, 0
            barriers.append(self)

        def wait(self, *, timeout):
            assert 0 < timeout <= 30
            self.waits += 1

    monkeypatch.setattr(probe.threading, "Barrier", RecordingBarrier)
    monkeypatch.setattr(
        probe,
        "_request",
        lambda url, text, expected, max_tokens, timeout: {
            "input_sha256": "i" * 64,
            "output_sha256": "o" * 64,
            "expected_code": expected,
            "first_code": expected,
            "first_code_matches": True,
            "prompt_tokens": 900,
            "completion_tokens": max_tokens,
            "elapsed_seconds": 0.01,
        },
    )
    result = probe.collect(
        "http://gateway",
        "quality_20260927",
        cases=3,
        records=48,
        max_tokens=20,
        timeout=5,
        concurrency=2,
        synchronized_start=True,
    )
    assert [(barrier.parties, barrier.waits) for barrier in barriers] == [
        (2, 2),
        (1, 1),
    ]
    assert [wave["requests"] for wave in result["waves"]] == [2, 1]
    assert result["concurrency"] == 2 and result["synchronized_start"] is True
    assert result["correct"] == 3 and result["total_elapsed_seconds"] >= 0


@pytest.mark.parametrize("concurrency", [0, 5, True, 2.5])
def test_fact_collect_rejects_invalid_concurrency(concurrency):
    with pytest.raises(ValueError, match="concurrency"):
        probe.collect(
            "http://gateway",
            "quality_20260927",
            cases=2,
            records=48,
            max_tokens=20,
            timeout=5,
            concurrency=concurrency,
        )
