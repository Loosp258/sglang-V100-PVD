"""Opt-in local Unix transport for an isolated prediction process.

Only a private directory and exact same-UID/expected-PID peers are accepted.
The handler is injected; this module loads no model and registers no serving
hook. CUDA handlers must execute on their owner thread and fence on failure.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import socket
import stat
import struct
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from sglang.srt.disaggregation.pvd.probe_lane_protocol import (
    MAX_LANE_REPLY_BYTES,
    ProbeLaneProtocolError,
    ProbeLaneReply,
    ProbeLaneTicket,
)
from sglang.srt.disaggregation.pvd.probe_lane_wire import (
    MAX_REPLY_FRAME_OVERHEAD,
    MAX_TICKET_FRAME_BYTES,
    decode_reply,
    decode_ticket,
    encode_reply,
    encode_ticket,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferBudget,
    TransferCapacityError,
)

logger = logging.getLogger(__name__)


def _private_socket_path(directory: str | Path, name: str) -> Path:
    if (
        type(name) is not str
        or not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise ProbeLaneProtocolError("socket name must be one basename")
    root = Path(directory)
    info = root.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise ProbeLaneProtocolError("socket directory must be owner-private")
    path = root / name
    if len(os.fsencode(path)) >= 100:
        raise ProbeLaneProtocolError("Unix socket path exceeds portable bound")
    return path


def _peer_credentials(writer: asyncio.StreamWriter) -> tuple[int, int, int]:
    if not hasattr(socket, "SO_PEERCRED"):
        raise ProbeLaneProtocolError("Linux SO_PEERCRED is required")
    sock = writer.get_extra_info("socket")
    if sock is None or sock.family != socket.AF_UNIX:
        raise ProbeLaneProtocolError("an AF_UNIX peer is required")
    raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    return struct.unpack("3i", raw)


async def _read_frame(reader: asyncio.StreamReader, *, limit: int) -> bytes:
    length = struct.unpack(">I", await reader.readexactly(4))[0]
    if not 0 < length <= limit:
        raise ProbeLaneProtocolError("Unix probe frame exceeds byte bound")
    return await reader.readexactly(length)


async def _write_frame(writer: asyncio.StreamWriter, frame: bytes, *, limit: int):
    if not 0 < len(frame) <= limit:
        raise ProbeLaneProtocolError("Unix probe frame exceeds byte bound")
    writer.write(struct.pack(">I", len(frame)))
    writer.write(frame)
    await writer.drain()


class ProbeLaneUnixClient:
    def __init__(
        self,
        directory,
        name,
        *,
        expected_server_pid: int,
        reply_budget: TransferBudget,
        background_loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.path = _private_socket_path(directory, name)
        if type(expected_server_pid) is not int or expected_server_pid <= 0:
            raise ProbeLaneProtocolError("exact sidecar PID required")
        if not isinstance(reply_budget, TransferBudget):
            raise ProbeLaneProtocolError("explicit host Q reply budget required")
        if background_loop is not None and (
            not isinstance(background_loop, asyncio.AbstractEventLoop)
            or not background_loop.is_running()
            or background_loop.is_closed()
        ):
            raise ProbeLaneProtocolError("running probe I/O loop required")
        if background_loop is not None:
            try:
                owner_loop = asyncio.get_running_loop()
            except RuntimeError:
                owner_loop = None
            if owner_loop is background_loop:
                raise ProbeLaneProtocolError(
                    "probe I/O loop must differ from owner loop"
                )
        self.expected_server_pid = expected_server_pid
        self.reply_budget = reply_budget
        self.background_loop = background_loop
        self._background_inflight = set()

    @asynccontextmanager
    async def request(self, ticket: ProbeLaneTicket):
        """Q tensors remain charged until the caller finishes materializing rows.

        The bound includes raw frame, slices, mutable decoding and cloned Q,
        plus fixed metadata overhead. Callers must not retain Q after exit.
        """
        if self.background_loop is not None and (
            asyncio.get_running_loop() is self.background_loop
        ):
            raise ProbeLaneProtocolError("probe I/O loop must differ from owner loop")
        frame = encode_ticket(ticket)
        if ticket.deadline_monotonic <= time.monotonic():
            raise ProbeLaneProtocolError("probe ticket expired before send")
        owner = f"pvd-probe-lane:{uuid.uuid4().hex}"
        self.reply_budget.reserve(
            owner, 8 * ticket.max_reply_bytes + MAX_REPLY_FRAME_OVERHEAD, 1
        )

        async def exchange():
            reader, writer = await asyncio.open_unix_connection(str(self.path))
            try:
                pid, uid, _ = _peer_credentials(writer)
                if pid != self.expected_server_pid or uid != os.getuid():
                    raise ProbeLaneProtocolError("sidecar PID/UID differs")
                await _write_frame(writer, frame, limit=MAX_TICKET_FRAME_BYTES)
                reply = await _read_frame(
                    reader,
                    limit=ticket.max_reply_bytes + MAX_REPLY_FRAME_OVERHEAD,
                )
                return decode_reply(ticket, reply)
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass

        async def timed_exchange():
            remaining = ticket.deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise ProbeLaneProtocolError("probe ticket expired before send")
            try:
                return await asyncio.wait_for(exchange(), timeout=remaining)
            except TimeoutError as exc:
                raise ProbeLaneProtocolError("probe lane request timed out") from exc
            except (OSError, asyncio.IncompleteReadError) as exc:
                raise ProbeLaneProtocolError(
                    "probe lane closed before complete reply"
                ) from exc

        try:
            if self.background_loop is None:
                rows = await timed_exchange()
            else:
                if not self.background_loop.is_running():
                    raise ProbeLaneProtocolError("probe I/O loop stopped")
                coroutine = timed_exchange()
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        coroutine, self.background_loop
                    )
                except BaseException:
                    coroutine.close()
                    raise
                self._background_inflight.add(future)
                wrapped = asyncio.wrap_future(future)
                try:
                    # The owner loop may pause for a synchronous Decode forward.
                    # Keep the exchange and its host reply budget alive until the
                    # actual I/O coroutine exits, including after cancellation.
                    try:
                        rows = await asyncio.shield(wrapped)
                    except asyncio.CancelledError:
                        while not wrapped.done():
                            try:
                                await asyncio.shield(wrapped)
                            except asyncio.CancelledError:
                                continue
                            except Exception:
                                break
                        try:
                            wrapped.result()
                        except BaseException:
                            pass
                        raise
                finally:
                    if wrapped.done():
                        self._background_inflight.discard(future)
            yield rows
        finally:
            self.reply_budget.release(owner)


class ProbeLaneUnixServer:
    def __init__(
        self,
        directory,
        name,
        *,
        expected_client_pid: int,
        target_model_id: str,
        weights_sha256: str,
        tokenizer_sha256: str,
        handler: Callable[[ProbeLaneTicket], ProbeLaneReply],
        reply_budget: TransferBudget,
        max_connections: int = 4,
        max_seen_nonces: int = 1024,
    ):
        self.path = _private_socket_path(directory, name)
        if type(expected_client_pid) is not int or expected_client_pid <= 0:
            raise ProbeLaneProtocolError("exact D client PID required")
        if not callable(handler):
            raise ProbeLaneProtocolError("prediction handler required")
        if not isinstance(reply_budget, TransferBudget):
            raise ProbeLaneProtocolError("explicit sidecar reply budget required")
        if type(max_connections) is not int or not 1 <= max_connections <= 8:
            raise ProbeLaneProtocolError("bounded connection count required")
        if type(max_seen_nonces) is not int or not 1 <= max_seen_nonces <= 4096:
            raise ProbeLaneProtocolError("bounded replay ledger required")
        self.expected_client_pid = expected_client_pid
        self.target_model_id = target_model_id
        self.weights_sha256 = weights_sha256
        self.tokenizer_sha256 = tokenizer_sha256
        self.handler = handler
        self.reply_budget = reply_budget
        self.max_connections = max_connections
        self.max_seen_nonces = max_seen_nonces
        self._seen_nonces: dict[str, float] = {}
        self._server = None
        self._active: set[asyncio.Task] = set()
        self._compute_lock = asyncio.Lock()
        self._inode = None
        self._closing = False
        self._rejection_counts: dict[str, int] = {}

    def _record_rejection(self, reason: str):
        # Fixed categories and logarithmic emission keep diagnostics bounded
        # even when an unauthenticated peer repeatedly opens the socket.
        count = self._rejection_counts.get(reason, 0) + 1
        self._rejection_counts[reason] = count
        if count & (count - 1) == 0:
            try:
                logger.warning(
                    "PVD probe lane reject reason=%s count=%d", reason, count
                )
            except Exception:
                pass  # Logging cannot change rejection or reply ownership.

    async def start(self):
        if self._server is not None or self._closing or os.path.lexists(self.path):
            raise ProbeLaneProtocolError("Unix probe socket already exists or closed")
        self._server = await asyncio.start_unix_server(
            self._accept, path=str(self.path), backlog=self.max_connections
        )
        os.chmod(self.path, 0o600)
        self._inode = self.path.stat().st_ino
        return self

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        task = asyncio.current_task()
        if self._closing or len(self._active) >= self.max_connections:
            self._record_rejection("connection_capacity")
            writer.close()
            await writer.wait_closed()
            return
        self._active.add(task)
        reply_owner = None
        try:
            pid, uid, _ = _peer_credentials(writer)
            if pid != self.expected_client_pid or uid != os.getuid():
                raise ProbeLaneProtocolError("D client PID/UID differs")
            frame = await asyncio.wait_for(
                _read_frame(reader, limit=MAX_TICKET_FRAME_BYTES), timeout=5.0
            )
            ticket = decode_ticket(
                frame,
                target_model_id=self.target_model_id,
                weights_sha256=self.weights_sha256,
                tokenizer_sha256=self.tokenizer_sha256,
            )
            now = time.monotonic()
            self._seen_nonces = {
                nonce: deadline
                for nonce, deadline in self._seen_nonces.items()
                if deadline > now
            }
            if ticket.nonce in self._seen_nonces:
                raise ProbeLaneProtocolError("probe ticket nonce replayed")
            if len(self._seen_nonces) >= self.max_seen_nonces:
                raise ProbeLaneProtocolError("probe replay ledger is full")
            reply_owner = f"pvd-probe-reply:{ticket.nonce}"
            self.reply_budget.reserve(
                reply_owner, 8 * ticket.max_reply_bytes + MAX_REPLY_FRAME_OVERHEAD, 1
            )
            self._seen_nonces[ticket.nonce] = ticket.deadline_monotonic
            async with self._compute_lock:
                if time.monotonic() >= ticket.deadline_monotonic:
                    raise ProbeLaneProtocolError("probe ticket expired in queue")
                response = self.handler(ticket)
                if inspect.isawaitable(response):
                    response = await response
            if time.monotonic() >= ticket.deadline_monotonic:
                raise ProbeLaneProtocolError("probe computation missed deadline")
            payload = encode_reply(ticket, response)
            await _write_frame(
                writer,
                payload,
                limit=min(MAX_LANE_REPLY_BYTES, ticket.max_reply_bytes)
                + MAX_REPLY_FRAME_OVERHEAD,
            )
        # Never send a partial or unauthenticated result. D treats a closed
        # connection as failure under the explicit boundary policy.
        except TransferCapacityError:
            self._record_rejection("reply_capacity")
        except ProbeLaneProtocolError:
            self._record_rejection("protocol")
        except TimeoutError:
            self._record_rejection("ticket_timeout")
        except (OSError, EOFError, asyncio.IncompleteReadError):
            self._record_rejection("peer_closed")
        except Exception:
            self._record_rejection("handler_error")
            raise
        finally:
            if reply_owner is not None:
                self.reply_budget.release(reply_owner)
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
            self._active.discard(task)

    async def aclose(self):
        if self._closing:
            return
        self._closing = True
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        if self._active:
            await asyncio.gather(*tuple(self._active), return_exceptions=True)
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(info.st_mode) and info.st_ino == self._inode:
            self.path.unlink()
