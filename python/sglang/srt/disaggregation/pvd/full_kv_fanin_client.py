"""Explicit D fan-in session: bounded HTTP and one MR until all writers close.

Not an automatic Decode scheduler factory. The caller owns the session, client,
and any local-import pins until drain; dropping a Python object is not cancellation.
"""

import asyncio
import copy
import json
import math
import threading
from collections.abc import Mapping

import aiohttp
from sglang.srt.disaggregation.pvd.full_kv_fanin import FullKVFanInReceiver
from sglang.srt.disaggregation.pvd.full_kv_fanin_proof import validate_fanin_proof
from sglang.srt.disaggregation.pvd.protocol import (
    ProtocolValidationError,
    WriteIdentity,
)


class FanInHTTPClient:
    """No hidden retry/redirect; an error is never transport closure evidence."""

    def __init__(self, base_url, *, timeout_seconds, max_response_bytes, session=None):
        if not isinstance(base_url, str) or not base_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("explicit HTTP(S) V coordinator required")
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("finite positive fan-in RPC timeout required")
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ValueError("positive fan-in response bound required")
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.max_response_bytes = max_response_bytes
        self._session, self._owns_session = session, session is None
        self._closed = False

    async def request(self, operation, payload):
        if self._closed:
            raise RuntimeError("fan-in HTTP client is closed")
        if operation not in {"reserve", "start", "poll", "ack", "fence", "preflight"}:
            raise ValueError("unknown fan-in operation")
        if self._session is None:
            self._session = aiohttp.ClientSession()
        async with self._session.post(
            f"{self.base_url}/v1/fanin/{operation}",
            json=payload,
            timeout=self.timeout,
            allow_redirects=False,
        ) as response:
            data = bytearray()
            async for chunk in response.content.iter_chunked(
                min(65536, self.max_response_bytes + 1)
            ):
                if len(data) + len(chunk) > self.max_response_bytes:
                    raise ProtocolValidationError("fan-in response exceeds byte bound")
                data.extend(chunk)
            if response.status != 200:
                raise RuntimeError(f"fan-in {operation} HTTP {response.status}")
            value = json.loads(data)
            if not isinstance(value, dict):
                raise ProtocolValidationError("fan-in response must be an object")
            return value

    async def close(self):
        # Closing HTTP does not fence outstanding writes or release any D MR.
        self._closed = True
        if self._owns_session and self._session is not None:
            await self._session.close()
        self._session = None


class FullKVFanInDelivery:
    """Owner-thread session. All V epochs are pinned BEFORE reserve is sent.

    ready is network-only. ack_after_install is an explicit caller obligation:
    import, local fences and all-D-rank installation agreement must precede it.
    This class cannot establish those facts by observing network completion.
    """

    def __init__(self, receiver, *, source_epochs, client):
        if not isinstance(receiver, FullKVFanInReceiver) or not callable(
            getattr(client, "request", None)
        ):
            raise ValueError("real fan-in receiver and async request client required")
        self._epochs = copy.deepcopy(source_epochs)
        self._manifest, self._identities = receiver.publish_for_sources(self._epochs)
        self._receiver, self._client = receiver, client
        self._owner_thread = threading.get_ident()
        self._lock = asyncio.Lock()
        self._state = "prepared"
        self._cancelled = self._closed = False
        self._ack_sent = False
        self._proofs = {}
        self._byte_counts = {
            int(r): sum(p["length"] for p in parts)
            for r, parts in self._manifest["writers"].items()
        }

    def _check(self):
        if self._closed or threading.get_ident() != self._owner_thread:
            raise RuntimeError("fan-in delivery closed or on another thread")

    @property
    def ready(self):
        self._check()
        return (
            not self._cancelled
            and self._state in {"delivered", "released"}
            and self._receiver.ready
        )

    @property
    def state(self):
        return self._state

    def _payload(self, operation):
        if operation in {"reserve", "fence"}:
            return {
                "manifest": copy.deepcopy(self._manifest),
                "source_epochs": dict(self._epochs),
            }
        return {
            "delivery_id": self._manifest["delivery_id"],
            "destination_rank": self._manifest["destination"]["rank"],
        }

    def _consume(self, reply, operation):
        fields = {
            "delivery_id",
            "destination_rank",
            "plan_fingerprint",
            "state",
            "write_identities",
            "writer_proofs",
            "fenced",
            "entry_count_held",
            "error",
        }
        if not isinstance(reply, Mapping) or set(reply) != fields:
            raise ProtocolValidationError("complete fan-in group response required")
        expected = {str(r): i.to_dict() for r, i in self._identities.items()}
        if (
            reply["delivery_id"] != self._manifest["delivery_id"]
            or type(reply["destination_rank"]) is not int
            or reply["destination_rank"] != self._manifest["destination"]["rank"]
            or reply["plan_fingerprint"] != self._manifest["plan_fingerprint"]
            or not isinstance(reply["write_identities"], Mapping)
            or set(reply["write_identities"]) != set(expected)
            or any(
                WriteIdentity.from_dict(reply["write_identities"][r])
                != self._identities[int(r)]
                for r in expected
            )
        ):
            raise ProtocolValidationError("fan-in group identity/plan mismatch")
        proofs = reply["writer_proofs"]
        if not isinstance(proofs, Mapping) or not set(proofs).issubset(expected):
            raise ProtocolValidationError("invalid fan-in writer proof map")
        for rank, proof in proofs.items():
            reported, terminal = validate_fanin_proof(
                proof,
                fingerprint=self._manifest["plan_fingerprint"],
                identities=self._identities,
                byte_counts=self._byte_counts,
                protocol=self._manifest["protocol"],
            )
            if str(reported) != rank or terminal is None:
                raise ProtocolValidationError("fan-in proof key/terminal mismatch")
            if rank in self._proofs and self._proofs[rank] != proof:
                raise ProtocolValidationError("fan-in terminal proof changed")
        if (
            type(reply["fenced"]) is not bool
            or reply["fenced"] != (set(proofs) == set(expected))
            or type(reply["entry_count_held"]) is not bool
            or reply["state"]
            not in {
                "reserved",
                "waiting_source",
                "writing",
                "delivered",
                "cancelling",
                "cancelled",
                "released",
            }
            or (
                reply["state"] in {"released", "cancelled"}
                and (not reply["fenced"] or reply["entry_count_held"])
            )
            or (
                reply["state"] not in {"released", "cancelled"}
                and not reply["entry_count_held"]
            )
            or (reply["state"] == "released" and not self._ack_sent)
            or (
                reply["state"] in {"delivered", "released"}
                and (
                    not reply["fenced"]
                    or any(
                        p["transport_state"] != "terminal_success"
                        for p in proofs.values()
                    )
                )
            )
        ):
            raise ProtocolValidationError("fan-in group state contradicts closure")
        # Validate the entire envelope before installing any proof.
        for rank, proof in proofs.items():
            self._receiver.observe(proof)
            self._proofs[rank] = copy.deepcopy(proof)
        if reply["state"] in {"cancelling", "cancelled"}:
            self.cancel()
        self._state = reply["state"]
        return copy.deepcopy(reply)

    async def _request(self, operation):
        self._check()
        async with self._lock:
            self._check()
            if operation == "ack" and not self.ready:
                raise ProtocolValidationError(
                    "all writers must deliver before install ACK"
                )
            if self._cancelled:
                operation = "fence"
            if operation == "ack":
                self._ack_sent = True
            try:
                reply = await self._client.request(operation, self._payload(operation))
                return self._consume(reply, operation)
            except BaseException:
                # The peer may have processed the request. Keep every MR pin.
                self.cancel()
                raise

    async def reserve(self):
        return await self._request("reserve")

    async def start(self):
        return await self._request("start")

    async def poll(self):
        return await self._request("poll")

    async def ack_after_install(self):
        """Caller has completed import/local fence/all-D-rank install agreement."""
        return await self._request("ack")

    def cancel(self):
        self._check()
        self._cancelled = True
        self._receiver.cancel()
        self._state = "cancelling"

    async def fence(self):
        self.cancel()
        return await self._request("fence")

    def close(self):
        self._check()
        if self._lock.locked():
            raise ProtocolValidationError("fan-in RPC still pending")
        if not self._cancelled and self._state != "released":
            raise ProtocolValidationError(
                "install ACK or cancellation required before close"
            )
        if set(self._proofs) != {str(r) for r in self._identities}:
            raise ProtocolValidationError("all possible V writers must be fenced")
        # A local release callback can fail after removing the network pin. No
        # new RPC may restart this publication; the MR owner retries cleanup.
        self._closed = True
        self._receiver.close()  # Independently requires every exact terminal proof.
