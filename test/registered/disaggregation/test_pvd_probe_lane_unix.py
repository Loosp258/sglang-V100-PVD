"""Same-UID exact-PID Unix transport with a fake prediction handler."""

import asyncio
import dataclasses
import multiprocessing
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
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferBudget,
    TransferCapacityError,
)
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


def server(tmp_path, handler=reply_for, *, client_pid=None, max_seen_nonces=1024):
    return ProbeLaneUnixServer(
        private_dir(tmp_path),
        "probe.sock",
        expected_client_pid=os.getpid() if client_pid is None else client_pid,
        target_model_id="target-checkpoint",
        weights_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        handler=handler,
        max_seen_nonces=max_seen_nonces,
    )


def client(tmp_path, *, server_pid=None, budget=None):
    return ProbeLaneUnixClient(
        private_dir(tmp_path),
        "probe.sock",
        expected_server_pid=os.getpid() if server_pid is None else server_pid,
        reply_budget=budget or TransferBudget(1 << 20, 4),
    )


def _child_server(directory, parent_pid, control):
    async def run():
        service = await ProbeLaneUnixServer(
            directory,
            "probe.sock",
            expected_client_pid=parent_pid,
            target_model_id="target-checkpoint",
            weights_sha256="a" * 64,
            tokenizer_sha256="b" * 64,
            handler=reply_for,
        ).start()
        try:
            control.send(os.getpid())
            await asyncio.to_thread(control.recv)
        finally:
            await service.aclose()
            control.close()

    asyncio.run(run())


def _start_child_server(socket_dir):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_child_server, args=(socket_dir, os.getpid(), child)
    )
    process.start()
    child.close()
    if not parent.poll(10):
        process.terminate()
        process.join(timeout=5)
        parent.close()
        raise AssertionError("probe sidecar did not start")
    assert parent.recv() == process.pid
    return process, parent


def _stop_child_server(process, control):
    try:
        if process.is_alive():
            control.send("stop")
            process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        control.close()


def test_real_process_credentials_and_stale_pid_after_restart(socket_dir):
    first, first_control = _start_child_server(socket_dir)
    try:
        old_client = client(socket_dir, server_pid=first.pid)

        async def request_once(selected):
            async with selected.request(ticket()) as rows:
                return tuple(row.layer for row in rows)

        assert asyncio.run(request_once(old_client)) == (0, 1)
    finally:
        _stop_child_server(first, first_control)
    assert not (socket_dir / "probe.sock").exists()

    second, second_control = _start_child_server(socket_dir)
    try:
        with pytest.raises(ProbeLaneProtocolError, match="PID/UID"):
            asyncio.run(request_once(old_client))
        assert asyncio.run(request_once(client(socket_dir, server_pid=second.pid))) == (
            0,
            1,
        )
    finally:
        _stop_child_server(second, second_control)


def test_unix_probe_round_trip_permissions_and_retirement(socket_dir):
    async def run():
        service = await server(socket_dir).start()
        try:
            assert stat.S_IMODE(service.path.stat().st_mode) == 0o600
            budget = TransferBudget(1 << 20, 1)
            async with client(socket_dir, budget=budget).request(ticket()) as rows:
                assert tuple(q.layer for q in rows) == (0, 1)
                assert rows[0].vectors[0, 0, 0] == 1
                assert budget.snapshot()["used_staging_bytes"] > 0
            assert budget.snapshot()["used_staging_bytes"] == 0
        finally:
            await service.aclose()
        assert not service.path.exists()

    asyncio.run(run())


def test_unix_client_refuses_wrong_server_pid(socket_dir):
    async def run():
        service = await server(socket_dir).start()
        try:
            with pytest.raises(ProbeLaneProtocolError, match="PID/UID"):
                async with client(socket_dir, server_pid=os.getpid() + 1).request(
                    ticket()
                ):
                    pass
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
                async with client(socket_dir).request(ticket()):
                    pass
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

            async def request_one():
                async with client(socket_dir).request(ticket()) as rows:
                    return len(rows)

            results = await asyncio.gather(request_one(), request_one())
            assert results == [2, 2]
            assert peak == 1
            expired = dataclasses.replace(
                ticket(), deadline_monotonic=time.monotonic() + 0.001
            )
            with pytest.raises(ProbeLaneProtocolError, match="timed out|expired"):
                async with client(socket_dir).request(expired):
                    pass
        finally:
            await service.aclose()

    asyncio.run(run())


def test_unix_path_requires_private_owned_directory(socket_dir):
    socket_dir.chmod(0o755)
    with pytest.raises(ProbeLaneProtocolError, match="owner-private"):
        ProbeLaneUnixClient(
            socket_dir,
            "probe.sock",
            expected_server_pid=os.getpid(),
            reply_budget=TransferBudget(1 << 20, 1),
        )
    socket_dir.chmod(0o700)
    with pytest.raises(ProbeLaneProtocolError, match="basename"):
        ProbeLaneUnixClient(
            socket_dir,
            "../foreign",
            expected_server_pid=os.getpid(),
            reply_budget=TransferBudget(1 << 20, 1),
        )


def test_unix_client_refuses_unbudgeted_reply_before_io(socket_dir):
    async def run():
        service = await server(socket_dir).start()
        try:
            budget = TransferBudget(1, 1)
            with pytest.raises(TransferCapacityError):
                async with client(socket_dir, budget=budget).request(ticket()):
                    pass
            assert budget.snapshot()["reservations"] == 0
        finally:
            await service.aclose()

    asyncio.run(run())


def test_unix_replay_and_bounded_ledger_refuse_before_handler(socket_dir):
    async def run():
        called = []

        def handler(bound):
            called.append(bound.nonce)
            return reply_for(bound)

        service = await server(socket_dir, handler, max_seen_nonces=1).start()
        try:
            first = ticket()
            async with client(socket_dir).request(first):
                pass
            with pytest.raises(ProbeLaneProtocolError, match="complete reply"):
                async with client(socket_dir).request(first):
                    pass
            with pytest.raises(ProbeLaneProtocolError, match="complete reply"):
                async with client(socket_dir).request(ticket()):
                    pass
            assert called == [first.nonce]
            assert service._seen_nonces == {first.nonce: first.deadline_monotonic}
        finally:
            await service.aclose()

    asyncio.run(run())


def test_unix_client_refunds_budget_when_consumer_fails(socket_dir):
    async def run():
        service = await server(socket_dir).start()
        try:
            budget = TransferBudget(1 << 20, 1)
            with pytest.raises(RuntimeError, match="consumer failed"):
                async with client(socket_dir, budget=budget).request(ticket()):
                    assert budget.snapshot()["reservations"] == 1
                    raise RuntimeError("consumer failed")
            assert budget.snapshot()["reservations"] == 0
        finally:
            await service.aclose()

    asyncio.run(run())
