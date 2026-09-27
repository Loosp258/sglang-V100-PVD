"""V control RPCs must finish during synchronous D forwards and drain on cancel."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from sglang.srt.disaggregation.pvd.control_server import HttpShardClient


def _start_loop():
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


def _stop_loop(loop, thread):
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    assert not thread.is_alive()
    loop.close()


def _start_server(received, release=None):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            if release is not None:
                release[0].set()
                assert release[1].wait(5)
            payload = json.dumps({"accepted": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_background_control_rpc_finishes_while_owner_loop_is_paused():
    received = []
    server, server_thread = _start_server(received)
    io_loop, io_thread = _start_loop()
    try:

        async def run():
            client = HttpShardClient(
                0, f"http://127.0.0.1:{server.server_port}", background_loop=io_loop
            )
            try:
                task = asyncio.create_task(
                    client._request("POST", "/mutation", {"sequence": 1})
                )
                for _ in range(100):
                    if client._background_inflight:
                        break
                    await asyncio.sleep(0.001)
                assert len(client._background_inflight) == 1
                future = next(iter(client._background_inflight))
                # Blocks the owner loop like a synchronous target forward.
                assert future.result(timeout=5) == {"accepted": True}
                assert not task.done()
                assert await task == {"accepted": True}
                assert not client._background_inflight
            finally:
                await client.close()

        asyncio.run(run())
        assert received == [{"sequence": 1}]
    finally:
        _stop_loop(io_loop, io_thread)
        _stop_server(server, server_thread)


def test_cancelled_mutating_control_rpc_drains_before_client_close():
    received = []
    entered, release = threading.Event(), threading.Event()
    server, server_thread = _start_server(received, (entered, release))
    io_loop, io_thread = _start_loop()
    try:

        async def run():
            client = HttpShardClient(
                0, f"http://127.0.0.1:{server.server_port}", background_loop=io_loop
            )
            task = asyncio.create_task(
                client._request("POST", "/mutation", {"sequence": 2})
            )
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert len(client._background_inflight) == 1
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not client._background_inflight
            await client.close()
            assert client._session is None

        asyncio.run(run())
        assert received == [{"sequence": 2}]
    finally:
        release.set()
        _stop_loop(io_loop, io_thread)
        _stop_server(server, server_thread)
