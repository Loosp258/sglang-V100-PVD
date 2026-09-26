"""Client-only benchmark schedule checks; no Gateway or GPU required."""

import json

import pytest

import bench_pvd_gateway_pairs as bench


def test_fixed_schedule_preserves_existing_prompt():
    assert bench.prompts_for_round(2, 0, "fixed") == [
        "EEFTRITON EEFTRITON  Case 0.",
        "EEFTRITON EEFTRITON  Case 1.",
    ]
    assert bench.prompts_for_round(2, 9, "fixed") == bench.prompts_for_round(
        2, 0, "fixed"
    )


def test_unique_schedule_is_deterministic_and_pairwise_distinct():
    a = bench.prompts_for_round(2, 0, "unique")
    b = bench.prompts_for_round(2, 1, "unique")
    warmup = bench.prompts_for_round(2, -1, "unique")
    assert a == bench.prompts_for_round(2, 0, "unique")
    assert len({*a, *b, *warmup}) == 6
    assert all(prompt.startswith("EEFTRITON EEFTRITON  Case ") for prompt in a)
    with pytest.raises(ValueError, match="schedule"):
        bench.prompts_for_round(2, 0, "unknown")


def test_warmups_excluded_and_prompt_schedule_is_reported(monkeypatch, capsys):
    observed = []

    def fake_request(_url, prompt, _tokens, _timeout):
        observed.append(prompt)
        return {
            "status": 200,
            "prompt_sha256": bench.hashlib.sha256(prompt.encode()).hexdigest(),
        }

    monkeypatch.setattr(bench, "request", fake_request)
    bench.main(
        [
            "--rounds",
            "2",
            "--warmup-rounds",
            "1",
            "--prompt-schedule",
            "unique",
            "--repetitions",
            "2",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["prompt_schedule"] == "unique"
    assert len(report["warmups"]) == 1
    assert len(report["rounds"]) == 2
    assert len(observed) == 6
    for index, row in zip((-1, 0, 1), report["warmups"] + report["rounds"]):
        expected = bench.prompts_for_round(2, index, "unique")
        assert {response["prompt_sha256"] for response in row["responses"]} == {
            bench.hashlib.sha256(prompt.encode()).hexdigest() for prompt in expected
        }


@pytest.mark.parametrize("flag,value", [("--rounds", "0"), ("--warmup-rounds", "6")])
def test_invalid_bounds_refused_before_network(monkeypatch, flag, value):
    monkeypatch.setattr(
        bench,
        "request",
        lambda *_args: (_ for _ in ()).throw(AssertionError("network was used")),
    )
    with pytest.raises(SystemExit) as exc:
        bench.main([flag, value])
    assert exc.value.code == 2
