"""Fail-closed ownership of the experimental local Q-probe sidecar process.

The Scheduler is the parent and therefore knows both peer PIDs. This module
does not enable the sidecar in serving; a startup caller must explicitly own
the returned process and pass its client/checkpoint to waiting admission.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from sglang.srt.disaggregation.pvd.probe_lane_identity import (
    ProbeLaneCheckpointIdentity,
)
from sglang.srt.disaggregation.pvd.probe_lane_unix import ProbeLaneUnixClient
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

logger = logging.getLogger(__name__)


class ProbeSidecarStartupError(RuntimeError):
    pass


@dataclass
class ProbeSidecarProcess:
    """Own only this child and its private, freshly created socket directory."""

    process: subprocess.Popen
    socket_dir: Path
    client: ProbeLaneUnixClient | None
    checkpoint: ProbeLaneCheckpointIdentity
    _output_thread: threading.Thread
    _closed: bool = False

    def check_alive(self):
        if self._closed or self.process.poll() is not None:
            if not self._closed:
                logger.warning(
                    "PVD probe sidecar exited pid=%d returncode=%s",
                    self.process.pid,
                    self.process.returncode,
                )
            raise ProbeSidecarStartupError("probe sidecar is no longer alive")

    def close(self, *, timeout: float = 10.0):
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "PVD probe sidecar kill after timeout pid=%d", self.process.pid
                )
                self.process.kill()
                self.process.wait(timeout=timeout)
        self._output_thread.join(timeout=timeout)
        socket = self.socket_dir / "probe.sock"
        if socket.is_socket():
            socket.unlink()
        self.socket_dir.rmdir()
        logger.info(
            "PVD probe sidecar closed pid=%d returncode=%s",
            self.process.pid,
            self.process.returncode,
        )


def launch_probe_sidecar(
    script: str | Path,
    args: list[str],
    *,
    checkpoint: ProbeLaneCheckpointIdentity,
    target_model_id: str,
    reply_budget: TransferBudget,
    startup_timeout: float = 180.0,
    directory_parent: str | Path | None = None,
    python_executable: str | Path = sys.executable,
    cuda_visible_devices: str | None = None,
) -> ProbeSidecarProcess:
    """Start a GPU0 sidecar with exact child identity and bounded startup wait.

    ``args`` are the model and budget arguments, never the peer PID or socket
    directory. Those are minted by this owner so callers cannot accidentally
    pair with another D process. The script is an explicit, existing local
    file; no shell or network model download is involved.
    """
    if (
        not isinstance(checkpoint, ProbeLaneCheckpointIdentity)
        or not isinstance(reply_budget, TransferBudget)
        or type(target_model_id) is not str
        or not target_model_id
        or type(startup_timeout) not in (int, float)
        or not 0 < startup_timeout <= 600
        or type(args) is not list
        or not all(type(arg) is str for arg in args)
        or (
            cuda_visible_devices is not None
            and (
                type(cuda_visible_devices) is not str
                or not cuda_visible_devices.isdecimal()
            )
        )
    ):
        raise ProbeSidecarStartupError("explicit bounded sidecar identity required")
    unresolved = Path(script)
    if unresolved.is_symlink() or not unresolved.is_file():
        raise ProbeSidecarStartupError("sidecar script must be a regular local file")
    path = unresolved.resolve(strict=True)
    owner_flags = ("--expected-d-pid", "--socket-dir", "--target-model-id")
    if any(
        arg == flag or arg.startswith(flag + "=")
        for arg in args
        for flag in owner_flags
    ):
        raise ProbeSidecarStartupError("caller may not override sidecar peer identity")
    root = Path(tempfile.mkdtemp(prefix="pvd-probe-", dir=directory_parent))
    os.chmod(root, 0o700)
    if len(os.fsencode(root / "probe.sock")) >= 100:
        root.rmdir()
        raise ProbeSidecarStartupError("sidecar socket path exceeds portable bound")
    ready: queue.Queue[dict | BaseException] = queue.Queue(maxsize=1)
    output_tail: deque[str] = deque(maxlen=8)
    diagnostic_count = 0
    child = None
    reader_thread = None
    try:
        command = [
            str(python_executable),
            str(path),
            *args,
            "--target-model-id",
            target_model_id,
            "--socket-dir",
            str(root),
            "--expected-d-pid",
            str(os.getpid()),
        ]
        child = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=(
                {**os.environ, "CUDA_VISIBLE_DEVICES": cuda_visible_devices}
                if cuda_visible_devices is not None
                else None
            ),
        )

        def drain():
            nonlocal diagnostic_count
            try:
                for line in child.stdout:
                    bounded = line.rstrip()[:512]
                    output_tail.append(bounded)
                    if "PVD probe lane reject reason=" in bounded:
                        diagnostic_count += 1
                        if diagnostic_count & (diagnostic_count - 1) == 0:
                            logger.warning(
                                "PVD probe sidecar child pid=%d %s",
                                child.pid,
                                bounded,
                            )
                    if len(line) > 4096 or not line.startswith("{"):
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        value = exc
                    if (
                        isinstance(value, dict)
                        and value.get("schema") == "pvd.probe.sidecar.ready.v1"
                        and ready.empty()
                    ):
                        ready.put_nowait(value)
            except BaseException as exc:
                if ready.empty():
                    ready.put_nowait(exc)

        reader_thread = threading.Thread(target=drain, daemon=True)
        reader_thread.start()
        deadline = time.monotonic() + startup_timeout
        while True:
            if child.poll() is not None:
                raise ProbeSidecarStartupError(
                    "probe sidecar exited before readiness: " + " | ".join(output_tail)
                )
            try:
                result = ready.get(
                    timeout=min(0.1, max(0.001, deadline - time.monotonic()))
                )
                break
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise ProbeSidecarStartupError("probe sidecar readiness timed out")
        expected_socket = root / "probe.sock"
        if (
            type(result) is not dict
            or result.get("schema") != "pvd.probe.sidecar.ready.v1"
            or result.get("pid") != child.pid
            or result.get("socket") != str(expected_socket)
            or result.get("weights_sha256") != checkpoint.weights_sha256
            or result.get("tokenizer_sha256") != checkpoint.tokenizer_sha256
            or result.get("device") != "cuda:0"
            or not expected_socket.is_socket()
            or child.poll() is not None
        ):
            raise ProbeSidecarStartupError("probe sidecar readiness identity mismatch")
        client = ProbeLaneUnixClient(
            root,
            "probe.sock",
            expected_server_pid=child.pid,
            reply_budget=reply_budget,
        )
        logger.info("PVD probe sidecar ready pid=%d device=cuda:0", child.pid)
        return ProbeSidecarProcess(child, root, client, checkpoint, reader_thread)
    except BaseException:
        if child is not None:
            owner = ProbeSidecarProcess(
                child, root, None, checkpoint, reader_thread or threading.Thread()
            )
            owner.close()
        else:
            root.rmdir()
        raise
