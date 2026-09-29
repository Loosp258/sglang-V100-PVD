"""Predictor-selected P chunks keep an exact first-prefix boundary."""

import json
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.pvd.conn import PVDKVSender
from sglang.srt.disaggregation.pvd.split_upload_policy import (
    SplitUploadPolicy,
    plan_split,
)


def test_policy_uses_calibrated_prompt_and_full_fallback(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({
        "schema": "pvd-exact16-split-policy-v1",
        "choices": {"2156": {"prefix": 2048}, "1024": {"prefix": 0}},
    }))
    policy = SplitUploadPolicy(str(path), page_size=1)
    assert policy.prefix_for(2156) == 2048
    assert policy.prefix_for(1024) == 0
    assert policy.prefix_for(1536) == 0


def test_policy_rejects_non_page_aligned_prefix(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({
        "schema": "pvd-exact16-split-policy-v1",
        "choices": {"2156": {"prefix": 2047}},
    }))
    with pytest.raises(ValueError, match="invalid calibrated split"):
        SplitUploadPolicy(str(path), page_size=16)


def _sender(prefix):
    sender = PVDKVSender.__new__(PVDKVSender)
    sender._chunked = True
    sender._final_submitted = False
    sender._next_page = 0
    sender._expected_pages = 2156
    sender._publish_future = None
    sender._predicted_prefix_pages = prefix
    sender._predicted_prefix_rows = prefix
    sender._will_send_final = False
    sender.key = SimpleNamespace(transfer_id="split-test")
    return sender


def test_sender_waits_for_exact_predicted_boundary_then_final():
    sender = _sender(2048)
    assert not sender.should_send_kv_chunk(512, False)
    assert not sender.should_send_kv_chunk(1536, False)
    assert sender.should_send_kv_chunk(2048, False)
    sender._next_page = 2048
    assert not sender.should_send_kv_chunk(64, False)
    assert sender.should_send_kv_chunk(108, True)


def test_sender_falls_back_to_full_when_boundary_is_skipped():
    sender = _sender(2048)
    assert not sender.should_send_kv_chunk(2100, False)
    assert sender._predicted_prefix_pages == 0
    assert sender.should_send_kv_chunk(2156, True)


def test_zero_prefix_never_publishes_early():
    sender = _sender(0)
    assert not sender.should_send_kv_chunk(512, False)
    assert sender.should_send_kv_chunk(2156, True)


def test_parametric_policy_decides_from_length_without_choice_rows(tmp_path):
    profile = {
        "schema": "pvd-exact16-parametric-v1",
        "config": {"graph_degree": 16, "group_heads": 4,
                   "prefill_chunk_tokens": 512, "page_size": 1},
        "domain": {"min_prompt_tokens": 512, "max_prompt_tokens": 2048,
                   "max_prefix_tokens": 1536, "min_tail_tokens": 64},
        "ranks": [
            {"arrival": [0, 0.1], "build": [0, 1, 0],
             "extend": [0, 0, 1.5]}
            for _ in range(2)
        ],
        "uncertainty_seconds": 0.1,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    policy = SplitUploadPolicy(str(path), page_size=1)
    assert policy.prefix_for(1024) == 512
    assert policy.prefix_for(1536) == 1024
    assert policy.prefix_for(3072) == 0
    decision = plan_split(profile, 1536)
    assert [item["prefix"] for item in decision["choices"]] == [1024, 512]


def test_parametric_policy_rejects_wrong_page_size(tmp_path):
    profile = {
        "schema": "pvd-exact16-parametric-v1",
        "config": {"graph_degree": 16, "group_heads": 4,
                   "prefill_chunk_tokens": 512, "page_size": 1},
        "domain": {"min_prompt_tokens": 512, "max_prompt_tokens": 2048,
                   "max_prefix_tokens": 1536, "min_tail_tokens": 64},
        "ranks": [
            {"arrival": [0, 0.1], "build": [0, 1, 0],
             "extend": [0, 0, 1.5]}
            for _ in range(2)
        ],
        "uncertainty_seconds": 0.1,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="invalid exact-16 split timing profile"):
        SplitUploadPolicy(str(path), page_size=2)
