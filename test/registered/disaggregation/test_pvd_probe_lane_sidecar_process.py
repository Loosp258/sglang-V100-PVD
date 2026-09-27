"""A real child process tests peer identity, lifetime and cleanup without CUDA."""

import socket
import tempfile
import time
from pathlib import Path

import pytest
from sglang.srt.disaggregation.pvd.probe_lane_identity import (
    ProbeLaneCheckpointIdentity,
)
from sglang.srt.disaggregation.pvd.probe_lane_sidecar_process import (
    ProbeSidecarStartupError,
    launch_probe_sidecar,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

FIXTURE = Path(__file__).parent / "fixtures/pvd_fake_probe_sidecar.py"
CHECKPOINT = ProbeLaneCheckpointIdentity("a" * 64, "b" * 64, 100, 10)
pytestmark = pytest.mark.skipif(
    not hasattr(socket, "SO_PEERCRED"), reason="Linux Unix peer credentials required"
)


@pytest.fixture
def short_tmp_root():
    # AF_UNIX has a short pathname limit; pytest's ordinary nested tmp_path
    # can exceed it before the code under test even starts a child.
    with tempfile.TemporaryDirectory(prefix="pvd-owner-", dir="/tmp") as root:
        yield Path(root)


def _launch(short_tmp_root):
    return launch_probe_sidecar(
        FIXTURE,
        [],
        checkpoint=CHECKPOINT,
        target_model_id="target/qwen2",
        reply_budget=TransferBudget(1 << 20, 2),
        startup_timeout=5,
        directory_parent=short_tmp_root,
    )


def test_owner_mints_peer_pid_and_private_directory(short_tmp_root):
    owner = _launch(short_tmp_root)
    try:
        assert owner.process.pid == owner.client.expected_server_pid
        assert owner.checkpoint is CHECKPOINT
        assert owner.socket_dir.stat().st_mode & 0o077 == 0
        assert owner.client.path.is_socket()
        owner.check_alive()
    finally:
        owner.close()
    assert not owner.socket_dir.exists()
    assert owner.process.poll() is not None
    with pytest.raises(ProbeSidecarStartupError, match="no longer alive"):
        owner.check_alive()
    owner.close()


def test_owner_exposes_only_selected_gpu_to_child(short_tmp_root, monkeypatch):
    monkeypatch.setenv("PVD_FAKE_SIDECAR_MODE", "assert_cuda_env")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    owner = launch_probe_sidecar(
        FIXTURE,
        [],
        checkpoint=CHECKPOINT,
        target_model_id="target/qwen2",
        reply_budget=TransferBudget(1 << 20, 2),
        startup_timeout=5,
        directory_parent=short_tmp_root,
        cuda_visible_devices="0",
    )
    try:
        owner.check_alive()
        assert owner.client.expected_server_pid == owner.process.pid
    finally:
        owner.close()


def test_owner_forwards_bounded_child_rejection(short_tmp_root, monkeypatch, caplog):
    monkeypatch.setenv("PVD_FAKE_SIDECAR_MODE", "diagnostic")
    owner = _launch(short_tmp_root)
    try:
        for _ in range(100):
            if "reason=reply_capacity" in caplog.text:
                break
            time.sleep(0.01)
        assert "reason=reply_capacity" in caplog.text
    finally:
        owner.close()


def test_unexpected_child_exit_is_reported_once(short_tmp_root, caplog):
    owner = _launch(short_tmp_root)
    try:
        owner.process.terminate()
        owner.process.wait(timeout=5)
        for _ in range(3):
            with pytest.raises(ProbeSidecarStartupError, match="no longer alive"):
                owner.check_alive()
        reports = [
            record
            for record in caplog.records
            if "PVD probe sidecar exited pid=" in record.message
        ]
        assert len(reports) == 1
    finally:
        owner.close()


@pytest.mark.parametrize("mode", ["wrong_pid", "wrong_hash", "exit"])
def test_bad_ready_or_early_exit_does_not_leave_a_child(
    short_tmp_root, monkeypatch, mode
):
    monkeypatch.setenv("PVD_FAKE_SIDECAR_MODE", mode)
    with pytest.raises(ProbeSidecarStartupError):
        _launch(short_tmp_root)
    assert list(short_tmp_root.iterdir()) == []


@pytest.mark.parametrize(
    "override", ["--socket-dir", "--expected-d-pid", "--target-model-id"]
)
def test_caller_cannot_override_owner_identity(short_tmp_root, override):
    with pytest.raises(ProbeSidecarStartupError, match="may not override"):
        launch_probe_sidecar(
            FIXTURE,
            [override, "value"],
            checkpoint=CHECKPOINT,
            target_model_id="target/qwen2",
            reply_budget=TransferBudget(1 << 20, 2),
            directory_parent=short_tmp_root,
        )
    assert list(short_tmp_root.iterdir()) == []
