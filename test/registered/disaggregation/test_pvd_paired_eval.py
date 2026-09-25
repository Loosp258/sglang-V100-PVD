"""CPU-only checks for reproducible paired PVD observations."""

import json
from types import SimpleNamespace

import pytest
import run_pvd_paired_eval as eval_tool


def test_dataset_rejects_duplicate_ids_and_bounds(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"id":"a","text":"one"}\n{"id":"a","text":"two"}\n')
    with pytest.raises(ValueError, match="unique"):
        eval_tool._dataset(dataset)
    dataset.write_text('{"id":"a","text":"one"}\n{"id":"b","text":"two"}\n')
    items, digest = eval_tool._dataset(dataset)
    assert [item["id"] for item in items] == ["a", "b"]
    assert len(digest) == 64


def test_collect_and_compare_fixed_dataset(monkeypatch, tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"id":"a","text":"one"}\n{"id":"b","text":"two"}\n')
    calls = []

    def fake_request(url, text, max_new_tokens, timeout_seconds):
        calls.append((url, text, max_new_tokens, timeout_seconds))
        return {
            "output_ids": [1, 2, 3] if text == "one" else [4, 5, 6],
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "client_elapsed_seconds": 2.0 if text == "one" else 4.0,
        }

    monkeypatch.setattr(eval_tool, "_request", fake_request)
    full_path = tmp_path / "full.json"
    options = dict(
        gateway_url="http://example.test",
        dataset=str(dataset),
        config_id="fixed-config",
        max_new_tokens=3,
        timeout_seconds=10.0,
    )
    eval_tool.collect(SimpleNamespace(**options, mode="full", output=str(full_path)))
    assert calls == [
        ("http://example.test", "one", 3, 10.0),
        ("http://example.test", "two", 3, 10.0),
    ]
    with pytest.raises(FileExistsError):
        eval_tool.collect(
            SimpleNamespace(**options, mode="full", output=str(full_path))
        )
    assert len(calls) == 2, "an existing output must be refused before GPU requests"
    full = json.loads(full_path.read_text())
    candidate = json.loads(full_path.read_text())
    candidate["mode_label"] = "predictive"
    candidate["results"][1]["output_ids"] = [4, 9, 6]
    candidate["results"][1]["client_elapsed_seconds"] = 5.0
    comparison = eval_tool.compare_reports(full, candidate)
    assert comparison["exact_output_match_fraction"] == 0.5
    assert comparison["mean_common_prefix_tokens"] == 2.0
    assert comparison["latency"]["full"]["median_seconds"] == 3.0
    assert comparison["latency"]["predictive"]["p95_nearest_rank_seconds"] == 5.0


def test_compare_refuses_mismatched_or_unverified_runs():
    def report(mode):
        return {
            "schema": "pvd.paired_eval.v1",
            "mode_label": mode,
            "mode_verified_by_script": False,
            "sequential_requests": True,
            "dataset_sha256": "dataset",
            "max_new_tokens": 2,
            "results": [
                {
                    "id": "a",
                    "output_ids": [1, 2],
                    "client_elapsed_seconds": 1.0,
                }
            ],
        }

    full, predictive = report("full"), report("predictive")
    predictive["dataset_sha256"] = "other"
    with pytest.raises(ValueError, match="dataset"):
        eval_tool.compare_reports(full, predictive)
    predictive["dataset_sha256"] = "dataset"
    predictive["results"][0]["id"] = "other"
    with pytest.raises(ValueError, match="request IDs"):
        eval_tool.compare_reports(full, predictive)
    predictive["results"][0]["id"] = "a"
    predictive["mode_verified_by_script"] = True
    with pytest.raises(ValueError, match="mislabeled"):
        eval_tool.compare_reports(full, predictive)


def test_request_uses_greedy_fixed_length_and_checks_reply(monkeypatch):
    calls = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            assert limit == 2 * 1024 * 1024
            return json.dumps(
                {
                    "output_ids": [11, 12],
                    "meta_info": {"prompt_tokens": 4, "completion_tokens": 2},
                }
            ).encode()

    def fake_open(request, timeout):
        calls.append((request, timeout))
        return Response()

    monkeypatch.setattr(eval_tool.urllib.request, "urlopen", fake_open)
    result = eval_tool._request("http://gateway/", "Prompt", 2, 5.0)
    request, timeout = calls[0]
    assert request.full_url == "http://gateway/generate" and timeout == 5.0
    assert json.loads(request.data) == {
        "text": "Prompt",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 2,
            "ignore_eos": True,
        },
    }
    assert result["output_ids"] == [11, 12]
    assert result["client_elapsed_seconds"] > 0
