"""Content identity must never be a path/revision/timestamp shortcut."""

import pytest
from sglang.srt.disaggregation.pvd.probe_lane_identity import (
    ProbeLaneIdentityError,
    checkpoint_identity,
)


def _checkpoint(root):
    (root / "config.json").write_bytes(b'{"architecture":"Qwen2"}')
    (root / "model-00001-of-00002.safetensors").write_bytes(b"weights-a")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"weights-b")
    (root / "model.safetensors.index.json").write_bytes(b"index")
    (root / "tokenizer.json").write_bytes(b"tokens")
    (root / "tokenizer_config.json").write_bytes(b"specials")


def test_checkpoint_digest_is_content_derived_and_domains_are_separate(tmp_path):
    _checkpoint(tmp_path)
    first = checkpoint_identity(tmp_path)
    assert len(first.weights_sha256) == len(first.tokenizer_sha256) == 64
    assert first.weights_sha256 != first.tokenizer_sha256
    assert first.weight_bytes > first.tokenizer_bytes > 0
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"weights-c")
    second = checkpoint_identity(tmp_path)
    assert first.weights_sha256 != second.weights_sha256
    assert first.tokenizer_sha256 == second.tokenizer_sha256
    (tmp_path / "tokenizer_config.json").write_bytes(b"different")
    third = checkpoint_identity(tmp_path)
    assert second.weights_sha256 == third.weights_sha256
    assert second.tokenizer_sha256 != third.tokenizer_sha256


def test_checkpoint_refuses_missing_or_symlinked_members(tmp_path):
    with pytest.raises(ProbeLaneIdentityError, match="safetensors"):
        checkpoint_identity(tmp_path)
    _checkpoint(tmp_path)
    (tmp_path / "tokenizer.json").unlink()
    with pytest.raises(ProbeLaneIdentityError, match="tokenizer"):
        checkpoint_identity(tmp_path)
    (tmp_path / "tokenizer.json").symlink_to(tmp_path / "config.json")
    with pytest.raises(ProbeLaneIdentityError, match="cannot open"):
        checkpoint_identity(tmp_path)
