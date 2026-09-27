"""Same-UID exact-PID Unix transport with a fake prediction handler."""

import asyncio
import dataclasses
import multiprocessing
import os
import socket
import stat
import struct
import tempfile
import threading
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


def server(
    tmp_path,
    handler=reply_for,
    *,
    client_pid=None,
    max_seen_nonces=1024,
    reply_budget=None,
):
    return ProbeLaneUnixServer(
        private_dir(tmp_path),
        "probe.sock",
        expected_client_pid=os.getpid() if client_pid is None else client_pid,
        target_model_id="target-checkpoint",
        weights_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        handler=handler,
        reply_budget=reply_budget or TransferBudget(1 << 20, 4),
        max_seen_nonces=max_seen_nonces,
    )


def client(tmp_path, *, server_pid=None, budget=None, background_loop=None):
    return ProbeLaneUnixClient(
        private_dir(tmp_path),
        "probe.sock",
        expected_server_pid=os.getpid() if server_pid is None else server_pid,
        reply_budget=budget or TransferBudget(1 << 20, 4),
        background_loop=background_loop,
    )


def _child_server(directory, parent_pid, control, delay, gates):
    async def run():
        async def handler(bound):
            if gates is not None:
                entered, release = gates
                entered.set()
                await asyncio.to_thread(release.wait)
            if delay:
                await asyncio.sleep(delay)
            return reply_for(bound)

        service = await ProbeLaneUnixServer(
            directory,
            "probe.sock",
            expected_client_pid=parent_pid,
            target_model_id="target-checkpoint",
            weights_sha256="a" * 64,
            tokenizer_sha256="b" * 64,
            handler=handler,
            reply_budget=TransferBudget(1 << 20, 4),
        ).start()
        try:
            control.send(os.getpid())
            await asyncio.to_thread(control.recv)
        finally:
            await service.aclose()
            control.close()

    asyncio.run(run())


def _start_child_server(socket_dir, *, delay=0, gates=None):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_child_server, args=(socket_dir, os.getpid(), child, delay, gates)
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


def _start_background_loop():
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(5)
    return loop, thread


def _stop_background_loop(loop, thread):
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    assert not thread.is_alive()
    loop.close()


def test_background_unix_exchange_finishes_while_owner_loop_is_paused(socket_dir):
    process, control = _start_child_server(socket_dir, delay=0.05)
    io_loop, io_thread = _start_background_loop()
    try:
        budget = TransferBudget(1 << 20, 1)
        lane = client(
            socket_dir,
            server_pid=process.pid,
            budget=budget,
            background_loop=io_loop,
        )

        async def run():
            async def exchange():
                async with lane.request(ticket()) as rows:
                    return tuple(row.layer for row in rows)

            task = asyncio.create_task(exchange())
            for _ in range(100):
                if lane._background_inflight:
                    break
                await asyncio.sleep(0.001)
            assert len(lane._background_inflight) == 1
            future = next(iter(lane._background_inflight))
            # Block the owner as a synchronous target Decode forward does.
            assert future.result(timeout=5)
            assert future.done()
            assert not task.done()
            assert budget.snapshot()["reservations"] == 1
            assert await task == (0, 1)
            assert budget.snapshot()["reservations"] == 0
            assert not lane._background_inflight

        asyncio.run(run())
    finally:
        _stop_background_loop(io_loop, io_thread)
        _stop_child_server(process, control)


def test_cancelled_background_exchange_drains_before_budget_refund(socket_dir):
    context = multiprocessing.get_context("spawn")
    entered, release = context.Event(), context.Event()
    process, control = _start_child_server(
        socket_dir, gates=(entered, release)
    )
    io_loop, io_thread = _start_background_loop()
    try:
        budget = TransferBudget(1 << 20, 1)
        lane = client(
            socket_dir,
            server_pid=process.pid,
            budget=budget,
            background_loop=io_loop,
        )

        async def run():
            async def exchange():
                async with lane.request(ticket()):
                    pass

            task = asyncio.create_task(exchange())
            for _ in range(100):
                if lane._background_inflight:
                    break
                await asyncio.sleep(0.001)
            assert len(lane._background_inflight) == 1
            assert entered.wait(5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert budget.snapshot()["reservations"] == 1
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert budget.snapshot()["reservations"] == 0
            assert not lane._background_inflight

        asyncio.run(run())
    finally:
        release.set()
        _stop_background_loop(io_loop, io_thread)
        _stop_child_server(process, control)


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


def test_unix_server_admits_and_refunds_reply_budget(socket_dir):
    async def run():
        called = []

        def handler(bound):
            called.append(bound.nonce)
            assert budget.snapshot()["reservations"] == 1
            return reply_for(bound)

        budget = TransferBudget(1 << 20, 1)
        service = await server(socket_dir, handler, reply_budget=budget).start()
        try:
            async with client(socket_dir).request(ticket()):
                pass
            assert budget.snapshot()["reservations"] == 0
            assert len(called) == 1
        finally:
            await service.aclose()

    asyncio.run(run())


def test_two_waiting_sidecar_requests_hold_distinct_reply_reservations(
    socket_dir, caplog
):
    async def run():
        release_first = asyncio.Event()
        entered = asyncio.Event()
        budget = TransferBudget(1 << 20, 2)

        async def handler(bound):
            if not entered.is_set():
                entered.set()
                await release_first.wait()
            return reply_for(bound)

        service = await server(socket_dir, handler, reply_budget=budget).start()

        async def exchange():
            async with client(socket_dir).request(ticket()) as rows:
                assert rows

        try:
            first = asyncio.create_task(exchange())
            await asyncio.wait_for(entered.wait(), timeout=3)
            second = asyncio.create_task(exchange())
            for _ in range(100):
                if budget.snapshot()["reservations"] == 2:
                    break
                await asyncio.sleep(0.01)
            assert budget.snapshot()["reservations"] == 2
            with pytest.raises(ProbeLaneProtocolError, match="complete reply"):
                await exchange()
            assert budget.snapshot()["reservations"] == 2
            assert "reason=reply_capacity" in caplog.text
            release_first.set()
            await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
            assert budget.snapshot()["reservations"] == 0
        finally:
            release_first.set()
            await service.aclose()

    asyncio.run(run())


def test_unix_server_refuses_capacity_before_handler(socket_dir):
    async def run():
        called = []
        budget = TransferBudget(1, 1)
        service = await server(
            socket_dir,
            lambda bound: called.append(bound) or reply_for(bound),
            reply_budget=budget,
        ).start()
        try:
            with pytest.raises(ProbeLaneProtocolError, match="complete reply"):
                async with client(socket_dir).request(ticket()):
                    pass
            assert called == []
            assert budget.snapshot()["reservations"] == 0
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
