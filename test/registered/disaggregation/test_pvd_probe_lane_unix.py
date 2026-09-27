"""Same-UID exact-PID Unix transport with a fake prediction handler."""

import asyncio
import dataclasses
import os
import socket
import stat
import struct
import tempfile
import time
from pathlib import Path

import pytest
from sglang.srt.disaggregation.pvd.probe_lane_protocol import ProbeLaneProtocolError
from sglang.srt.disaggregation.pvd.probe_lane_unix import (
    ProbeLaneUnixClient,
    ProbeLaneUnixServer,
)
from sglang.srt.disaggregation.pvd.probe_lane_wire import MAX_TICKET_FRAME_BYTES
from test_pvd_probe_lane_protocol import reply_for, ticket

pytestmark = pytest.mark.skipif(
    not hasattr(socket, "SO_PEERCRED"), reason="Linux Unix peer credentials required"
)


@pytest.fixture
def socket_dir():
    # AF_UNIX sun_path is short; the CloudLab pytest workspace is too deep.
    with tempfile.TemporaryDirectory(prefix="pvd-lane-", dir="/tmp") as name:
        root = Path(name)
        root.chmod(0o700)
        yield root


def private_dir(tmp_path):
    tmp_path.chmod(0o700)
    return tmp_path


def server(tmp_path, handler=reply_for, *, client_pid=None):
    return ProbeLaneUnixServer(
        private_dir(tmp_path),
        "probe.sock",
        expected_client_pid=os.getpid() if client_pid is None else client_pid,
        target_model_id="target-checkpoint",
        weights_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        handler=handler,
    )


def client(tmp_path, *, server_pid=None):
    return ProbeLaneUnixClient(
        private_dir(tmp_path),
        "probe.sock",
        expected_server_pid=os.getpid() if server_pid is None else server_pid,
    )


def test_unix_probe_round_trip_permissions_and_retirement(socket_dir):
    async def run():
        service = await server(socket_dir).start()
        try:
            assert stat.S_IMODE(service.path.stat().st_mode) == 0o600
            rows = await client(socket_dir).request(ticket())
            assert tuple(q.layer for q in rows) == (0, 1)
            assert rows[0].vectors[0, 0, 0] == 1
        finally:
            await service.aclose()
        assert not service.path.exists()

    asyncio.run(run())


def test_unix_client_refuses_wrong_server_pid(socket_dir):
    async def run():
        service = await server(socket_dir).start()
        try:
            with pytest.raises(ProbeLaneProtocolError, match="PID/UID"):
                await client(socket_dir, server_pid=os.getpid() + 1).request(ticket())
        finally:
            await service.aclose()

    asyncio.run(run())


def test_unix_server_refuses_wrong_client_pid_before_handler(socket_dir):
    async def run():
        called = []

        def handler(bound):
            called.append(bound)
            return reply_for(bound)

        service = await server(socket_dir, handler, client_pid=os.getpid() + 1).start()
        try:
            with pytest.raises(ProbeLaneProtocolError, match="complete reply"):
                await client(socket_dir).request(ticket())
            assert not called
        finally:
            await service.aclose()

    asyncio.run(run())


def test_unix_server_rejects_oversized_frame_before_handler(socket_dir):
    async def run():
        called = []

        def handler(bound):
            called.append(bound)
            return reply_for(bound)

        service = await server(socket_dir, handler).start()
        try:
            reader, writer = await asyncio.open_unix_connection(str(service.path))
            writer.write(struct.pack(">I", MAX_TICKET_FRAME_BYTES + 1))
            await writer.drain()
            assert await reader.read() == b""
            writer.close()
            await writer.wait_closed()
            assert not called
        finally:
            await service.aclose()

    asyncio.run(run())


def test_unix_probe_deadline_and_serial_handler(socket_dir):
    async def run():
        active = 0
        peak = 0

        async def handler(bound):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.02)
                return reply_for(bound)
            finally:
                active -= 1

        service = await server(socket_dir, handler).start()
        try:
            results = await asyncio.gather(
                client(socket_dir).request(ticket()),
                client(socket_dir).request(ticket()),
            )
            assert all(len(rows) == 2 for rows in results)
            assert peak == 1
            expired = dataclasses.replace(
                ticket(), deadline_monotonic=time.monotonic() + 0.001
            )
            with pytest.raises(ProbeLaneProtocolError, match="timed out|expired"):
                await client(socket_dir).request(expired)
        finally:
            await service.aclose()

    asyncio.run(run())


def test_unix_path_requires_private_owned_directory(socket_dir):
    socket_dir.chmod(0o755)
    with pytest.raises(ProbeLaneProtocolError, match="owner-private"):
        ProbeLaneUnixClient(socket_dir, "probe.sock", expected_server_pid=os.getpid())
    socket_dir.chmod(0o700)
    with pytest.raises(ProbeLaneProtocolError, match="basename"):
        ProbeLaneUnixClient(socket_dir, "../foreign", expected_server_pid=os.getpid())
