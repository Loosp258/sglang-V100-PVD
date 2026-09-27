"""Content-derived checkpoint identities for a private prediction lane.

The digest covers bytes, not a model path, revision label or file timestamp.
The first implementation accepts local safetensors checkpoints and a fast
tokenizer; other formats must add an explicit audited manifest policy.
"""

from __future__ import annotations

import hashlib
import os
import stat
import struct
from dataclasses import dataclass
from pathlib import Path


class ProbeLaneIdentityError(ValueError):
    """Model files are missing, ambiguous or changed during hashing."""


@dataclass(frozen=True)
class ProbeLaneCheckpointIdentity:
    weights_sha256: str
    tokenizer_sha256: str
    weight_bytes: int
    tokenizer_bytes: int


_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)
_MAX_FILES = 128
_MAX_TOTAL_BYTES = 256 << 30


def _hash_files(
    root: Path, names: tuple[str, ...], *, domain: bytes
) -> tuple[str, int]:
    if not names or len(names) > _MAX_FILES or len(set(names)) != len(names):
        raise ProbeLaneIdentityError("bounded distinct checkpoint file set required")
    digest = hashlib.sha256(domain)
    total = 0
    for name in sorted(names):
        if not name or Path(name).name != name or name in (".", ".."):
            raise ProbeLaneIdentityError("checkpoint file must be one basename")
        path = root / name
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as exc:
            raise ProbeLaneIdentityError(f"cannot open checkpoint file {name}") from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size < 0:
                raise ProbeLaneIdentityError("checkpoint member must be a regular file")
            total += before.st_size
            if total > _MAX_TOTAL_BYTES:
                raise ProbeLaneIdentityError("checkpoint bytes exceed identity bound")
            label = name.encode("utf-8")
            digest.update(struct.pack("<I", len(label)))
            digest.update(label)
            digest.update(struct.pack("<Q", before.st_size))
            observed = 0
            while block := os.read(descriptor, 4 << 20):
                digest.update(block)
                observed += len(block)
            after = os.fstat(descriptor)
            if observed != before.st_size or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise ProbeLaneIdentityError("checkpoint changed during digest")
        finally:
            os.close(descriptor)
    return digest.hexdigest(), total


def checkpoint_identity(model_path: str | Path) -> ProbeLaneCheckpointIdentity:
    """Hash exact local Qwen2-style safetensors and tokenizer artifacts."""
    root = Path(model_path)
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise ProbeLaneIdentityError("checkpoint directory is unavailable") from exc
    if not stat.S_ISDIR(root_info.st_mode) or root.is_symlink():
        raise ProbeLaneIdentityError("checkpoint root must be a real directory")
    weights = tuple(path.name for path in root.glob("*.safetensors"))
    if not weights or not (root / "config.json").is_file():
        raise ProbeLaneIdentityError("local safetensors and config.json required")
    if tuple(sorted(weights)) != weights:
        weights = tuple(sorted(weights))
    weight_names = ("config.json", *weights)
    if (root / "model.safetensors.index.json").exists():
        weight_names += ("model.safetensors.index.json",)
    if not (root / "tokenizer.json").is_file():
        raise ProbeLaneIdentityError("fast tokenizer.json required")
    tokenizer_names = tuple(name for name in _TOKENIZER_FILES if (root / name).exists())
    weight_digest, weight_bytes = _hash_files(
        root, weight_names, domain=b"pvd.probe.weights.v1\0"
    )
    tokenizer_digest, tokenizer_bytes = _hash_files(
        root, tokenizer_names, domain=b"pvd.probe.tokenizer.v1\0"
    )
    return ProbeLaneCheckpointIdentity(
        weight_digest, tokenizer_digest, weight_bytes, tokenizer_bytes
    )
