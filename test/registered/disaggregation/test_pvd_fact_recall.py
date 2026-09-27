"""Deterministic varied-fact load probe without a live Gateway."""

import re

import pytest
import run_pvd_fact_recall as probe


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
    "seed,case,records", [("../bad", 0, 48), ("ok", -1, 48), ("ok", 0, 97)]
)
def test_fact_prompt_refuses_unbounded_shape(seed, case, records):
    with pytest.raises(ValueError):
        probe.make_prompt(seed, case, records=records)


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
