"""Optional direct P->D initial Prompt KV delivery.

The Prefill bootstrap HTTP process is a rendezvous only. It never owns CUDA
memory or claims that a PUT completed; the P transfer engine supplies terminal
proof and D installs the bytes after checking the exact request and generation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Mapping

import aiohttp

from sglang.srt.disaggregation.pvd.protocol import (
    PVD_GENERATION_METADATA_KEY,
    PVD_RECEIVER_EPOCH_METADATA_KEY,
    RemoteRegionDescriptor,
)
from sglang.srt.disaggregation.pvd.transfer_engine import MemorySlice, TransferStatus
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransportState

logger = logging.getLogger(__name__)


class DirectBootstrapError(RuntimeError):
    pass


class DirectBootstrapClient:
    def __init__(self, host: str, port: int, *, poll_interval: float = 0.005):
        if not isinstance(host, str) or not host or type(port) is not int or port <= 0:
            raise ValueError("direct bootstrap requires a P bootstrap address")
        self.base = f"http://{host}:{port}/pvd/direct"
        self.poll_interval = poll_interval

    async def _request(self, method: str, part: str, payload: Mapping):
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method,
                f"{self.base}/{part}",
                json=payload if method == "POST" else None,
                params=payload if method == "GET" else None,
            ) as response:
                body = await response.json()
                if response.status != 200:
                    raise DirectBootstrapError(
                        f"P direct {part} rejected {response.status}: {body}"
                    )
                return body

    async def post(self, part: str, payload: Mapping):
        return await self._request("POST", part, payload)

    async def get(self, part: str, identity: Mapping):
        return await self._request("GET", part, identity)

    async def wait(self, part: str, identity: Mapping, *, timeout: float = 300.0):
        deadline = time.monotonic() + timeout
        while True:
            reply = await self.get(part, identity)
            if reply.get("state") != "waiting":
                return reply
            if time.monotonic() >= deadline:
                raise DirectBootstrapError(f"P direct {part} timed out")
            await asyncio.sleep(self.poll_interval)


async def publish_and_send_direct(
    *, client: DirectBootstrapClient, key, delivery_id: str, manifest,
    first_token, packed, engine, sender_epoch: str, rail: str, initial_shard_routes=(),
):
    """P-owned task: keep the packed source alive through native completion."""
    identity = {"transfer_id": key.transfer_id, "delivery_id": delivery_id}
    nbytes = int(packed.numel())
    await client.post(
        "entry",
        {
            **identity,
            "manifest": manifest.to_dict(),
            "first_token": first_token.to_dict(),
            "expected_bytes": nbytes,
            "sender_epoch": sender_epoch,
            "initial_shard_routes": list(initial_shard_routes),
        },
    )
    destination_reply = await client.wait("destination", identity)
    if (
        destination_reply.get("transfer_id") != key.transfer_id
        or destination_reply.get("delivery_id") != delivery_id
        or destination_reply.get("receiver_epoch") is None
        or destination_reply.get("generation") is None
    ):
        raise DirectBootstrapError("D direct destination identity changed")
    destination = RemoteRegionDescriptor.from_dict(destination_reply["destination"])
    if (
        destination.rank != 0
        or destination.length != nbytes
        or destination.rail != rail
        or destination.backend_metadata.get(PVD_RECEIVER_EPOCH_METADATA_KEY)
        != destination_reply["receiver_epoch"]
        or destination.backend_metadata.get(PVD_GENERATION_METADATA_KEY)
        != destination_reply["generation"]
    ):
        raise DirectBootstrapError("D direct destination layout or rail differs")
    registration = engine.register_memory(
        packed, endpoint="pvd-direct-prefill", rank=0, rail=rail
    )
    handle = None
    try:
        handle = engine.submit_put(MemorySlice(registration, 0, nbytes), destination)
        while True:
            status = engine.poll(handle)
            if status != TransferStatus.PENDING and engine.cleanup_complete(handle):
                break
            await asyncio.sleep(client.poll_interval)
        state = (
            "terminal_success"
            if status == TransferStatus.SUCCESS
            and handle.transport_state == TransportState.TERMINAL_SUCCESS
            and handle.transferred_bytes == nbytes
            else "terminal_failed"
        )
        proof = {
            **identity,
            "state": state,
            "transferred_bytes": handle.transferred_bytes,
            "sender_epoch": sender_epoch,
            "receiver_epoch": destination_reply["receiver_epoch"],
            "generation": destination_reply["generation"],
        }
        # Retry a lost response; the rendezvous accepts exact replay only.
        while True:
            try:
                await client.post("terminal", proof)
                break
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(client.poll_interval)
        if state != "terminal_success":
            raise DirectBootstrapError(handle.error or "P->D native PUT failed")
        logger.info(
            "PVD direct initial KV terminal: transfer_id=%s bytes=%d",
            key.transfer_id, nbytes,
        )
    finally:
        # A nonterminal native handle must retain its registered source. The
        # loop above has no timeout after submission for exactly this reason.
        if handle is None or engine.cleanup_complete(handle):
            engine.release_memory(registration)


def direct_initial_steps(refresher, reqs):
    """Scheduler generator for the TP1 initial D install, before any V search."""
    from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeSession

    manager = refresher.manager
    sessions = []
    try:
        for req in reqs:
            session = manager.decode_sessions[manager.key_for(req)]
            if session.clock.round != 0 or not session.due():
                continue
            rows = manager.scheduler.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.origin_input_ids) : manager.page_size
            ]
            # The fan-in subclass may use rank-packed V shard staging. P sends
            # the canonical TP1 compute layout, so use the base full-KV path.
            descriptor = PVDDecodeSession.prepare(
                session, (rows // manager.page_size).cpu()
            )
            sessions.append((session, descriptor))
            entry = session._direct_entry
            if (
                entry is None
                or entry.get("transfer_id") != session.key.transfer_id
                or entry.get("delivery_id") != req.pvd_delivery_id
                or entry.get("expected_bytes") != session.staging.numel()
            ):
                raise DirectBootstrapError("P direct metadata differs from D staging")
    except Exception as exc:
        for session, _ in sessions:
            if not session._direct_pending and session._refresh_owner is not None:
                session.release_refresh()  # No D descriptor was published.
        return [(req, str(exc)) for req in reqs]
    if not sessions:
        return []

    async def deliver():
        replies = []
        for session, descriptor in sessions:
            identity = {
                "transfer_id": session.key.transfer_id,
                "delivery_id": session.req.pvd_delivery_id,
            }
            session._direct_pending = True  # An uncertain POST may publish the MR.
            await session._direct_client.post(
                "destination",
                {
                    **identity,
                    "destination": descriptor["destination"],
                    "receiver_epoch": session.receiver_epoch,
                    "generation": session.generation,
                },
            )
            proof = await session._direct_client.wait("terminal", identity)
            if (
                proof.get("state") != "terminal_success"
                or proof.get("transferred_bytes") != session.staging.numel()
                or proof.get("sender_epoch") != session._direct_entry.get("sender_epoch")
                or proof.get("receiver_epoch") != session.receiver_epoch
                or proof.get("generation") != session.generation
            ):
                raise DirectBootstrapError(f"P direct KV proof failed: {proof}")
            replies.append(
                {
                    "delivery_id": session.clock.pending[0],
                    "key": session.key.to_dict(),
                    "sequence_id": session.key.req_id,
                    "selection": "full_prompt",
                    "token_ranges": [[0, len(session.req.origin_input_ids)]],
                    "state": "delivered",
                }
            )
        return replies

    future = manager.control.submit(deliver())
    replies, error = yield future
    if error:
        # A failed or uncertain POST may have published one destination. Only
        # the later, never-published descriptors can be released immediately.
        for session, _ in sessions:
            if not session._direct_pending and session._refresh_owner is not None:
                session.release_refresh()
        return [(session.req, error) for session, _ in sessions]
    try:
        if len(replies) != len(sessions):
            raise DirectBootstrapError("P direct receipt count changed")
        for (session, _), reply in zip(sessions, replies, strict=True):
            if session._closed or session.req.finished():
                raise DirectBootstrapError("D request closed during direct KV delivery")
            PVDDecodeSession.unpack(session, reply)
    except Exception as exc:
        return [(session.req, str(exc)) for session, _ in sessions]

    async def ack():
        for session, _ in sessions:
            await session._direct_client.post(
                "ack",
                {
                    "transfer_id": session.key.transfer_id,
                    "delivery_id": session.req.pvd_delivery_id,
                },
            )

    future = manager.control.submit(ack())
    _, error = yield future
    if error:
        return [(session.req, error) for session, _ in sessions]
    for session, _ in sessions:
        session._direct_pending = False
        session._complete_refresh()
        logger.info(
            "PVD direct initial KV installed: transfer_id=%s bytes=%d",
            session.key.transfer_id,
            session.staging.numel(),
        )
    return []
