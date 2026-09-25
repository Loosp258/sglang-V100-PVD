"""Full Prompt fan-in in the existing Decode waiting/periodic refresh driver.

All transport ownership lives on the control loop. Preparation, unpack, rank
agreement and completion receipts remain on the Scheduler thread.
"""

import asyncio
import math

import torch
from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeSession
from sglang.srt.disaggregation.pvd.fanin_scatter import scatter_rank_packed_bytes
from sglang.srt.disaggregation.pvd.full_kv_fanin import FullKVFanInReceiver
from sglang.srt.disaggregation.pvd.full_kv_fanin_client import (
    FanInHTTPClient,
    FullKVFanInDelivery,
)
from sglang.srt.disaggregation.pvd.full_kv_fanin_plan import (
    FULL_KV_FANIN_PROTOCOL,
    RANK_PACKED_FULL_KV_FANIN_PROTOCOL,
)
from sglang.srt.disaggregation.pvd.kv_packer import kv_components
from sglang.srt.disaggregation.pvd.protocol import (
    KVEntryManifest,
    RemoteRegionDescriptor,
)
from sglang.srt.disaggregation.pvd.sharding import (
    rank_packed_full_shard_fanin_plan,
    source_shard_intersections,
)


class PVDDecodeFanInSession(PVDDecodeSession):
    def __init__(self, manager, req):
        super().__init__(manager, req)
        self._fanin = self._fanin_client = self._manifest = None
        self._fanin_lock = asyncio.Lock()
        self._network_receipt = None
        self._rank_packed_plan = None
        self._canonical_staging = None
        self._canonical_budget = None
        self._canonical_budget_owner = None

    async def initialize(self, runtime):
        record = await super().initialize(runtime)
        self._manifest = KVEntryManifest.from_dict(record["manifest"])
        return record

    def prepare(self, pages):
        if self._fanin is not None:
            raise RuntimeError("previous fan-in still owns the receive region")
        if self._manifest is None:
            raise RuntimeError("fan-in Entry manifest is not ready")
        if getattr(self.manager, "full_kv_fanin_rank_packed", False):
            tokens = (
                math.ceil(len(self.req.origin_input_ids) / self.manager.page_size)
                * self.manager.page_size
            )
            plan = rank_packed_full_shard_fanin_plan(
                self._manifest.layout,
                self.manager.layout(),
                compute_rank=self.manager.tp_rank,
                token_count=tokens,
            )
            if self._rank_packed_plan is not None and self._rank_packed_plan != plan:
                raise RuntimeError("immutable rank-packed Prompt plan changed")
            if self._canonical_staging is None:
                budget = getattr(self.manager, "transfer_budget", None)
                if budget is None:
                    raise RuntimeError("rank-packed canonical buffer requires a budget")
                owner = (
                    f"decode-rank-packed:{self.key.transfer_id}:"
                    f"{self.consumer_id}:{self.manager.tp_rank}"
                )
                budget.reserve(owner, plan.staging_bytes, 0)
                try:
                    device = kv_components(self.manager.kv_pool)[0].device
                    canonical = torch.empty(
                        plan.staging_bytes, dtype=torch.uint8, device=device
                    )
                except BaseException:
                    budget.release(owner)
                    raise
                self._canonical_staging = canonical
                self._canonical_budget, self._canonical_budget_owner = budget, owner
            self._rank_packed_plan = plan
        result = super().prepare(pages)
        # The old round has drained before prepare is legal. Update this MR's
        # protocol metadata without registering/deregistering the native region.
        self.registration.descriptor = RemoteRegionDescriptor.from_dict(
            result["destination"]
        )
        self._network_receipt = None
        return result

    async def _client(self):
        if self._fanin_client is None:
            self._fanin_client = FanInHTTPClient(
                self.client.base_url,
                timeout_seconds=300,
                max_response_bytes=self.manager.scheduler.server_args.pvd_full_kv_fanin_response_bytes,
            )
        return self._fanin_client

    async def deliver_fanin(self, interval):
        # Bound the whole operation, including preflight/reserve. A fixed 200
        # polls would reject long prompts whose bounded writer needs more PUTs.
        return await asyncio.wait_for(self._deliver_fanin(interval), timeout=300)

    async def _deliver_fanin(self, interval):
        async with self._fanin_lock:
            if self._closed or self.req.finished():
                raise RuntimeError("request closed before fan-in publication")
            client = await self._client()
            health = await client.request("preflight", {})
            if self._closed:
                raise RuntimeError("request closed during fan-in preflight")
            layout = self.manager.layout()
            parts = source_shard_intersections(
                self._manifest.layout, layout, self.manager.tp_rank
            )
            required = {part.storage_rank for part in parts}
            rank_packed = self._rank_packed_plan is not None
            epochs = {}
            seen = set()
            for shard in health["shards"]:
                rank = shard.get("rank")
                if type(rank) is not int or rank in seen:
                    raise ValueError("invalid V rank set in fan-in preflight")
                seen.add(rank)
                if rank in required:
                    if (
                        shard.get("ready") is not True
                        or shard.get("rail") != self.manager.rail
                        or shard.get("full_kv_fanin", {}).get("enabled") is not True
                    ):
                        raise ValueError(
                            "fan-in source is unavailable or has another rail"
                        )
                    if rank_packed and (
                        RANK_PACKED_FULL_KV_FANIN_PROTOCOL
                        not in shard["full_kv_fanin"].get("protocols", ())
                        or shard["full_kv_fanin"].get("native_batch") is not True
                    ):
                        raise ValueError(
                            "V source lacks rank-packed native fan-in capability"
                        )
                    epochs[str(rank)] = shard.get("worker_epoch")
            if set(epochs) != {str(r) for r in required} or any(
                not isinstance(v, str) or not v.strip() for v in epochs.values()
            ):
                raise ValueError("fan-in preflight omitted required source epochs")
            if health.get("full_kv_fanin", {}).get("enabled") is not True:
                raise ValueError("V coordinator fan-in is disabled")
            if rank_packed and (
                RANK_PACKED_FULL_KV_FANIN_PROTOCOL
                not in health["full_kv_fanin"].get("protocols", ())
            ):
                raise ValueError("V coordinator lacks rank-packed fan-in capability")
            receiver = FullKVFanInReceiver(
                key=self.key,
                delivery_id=self.clock.pending[0],
                registration=self.registration,
                guard=self.receive_guard,
                storage=self._manifest.layout,
                compute=layout,
                token_count=math.ceil(
                    len(self.req.origin_input_ids) / self.manager.page_size
                )
                * self.manager.page_size,
                max_slices=self.manager.full_kv_fanin_max_slices,
                protocol=(
                    RANK_PACKED_FULL_KV_FANIN_PROTOCOL
                    if rank_packed
                    else FULL_KV_FANIN_PROTOCOL
                ),
            )
            try:
                self._fanin = FullKVFanInDelivery(
                    receiver, source_epochs=epochs, client=client
                )
            except BaseException:
                receiver.close()  # Constructor validation precedes publication.
                raise
            await self._fanin.reserve()
            await self._fanin.start()
            while True:
                if self._closed or self.req.finished():
                    raise RuntimeError("request closed during fan-in delivery")
                if self._fanin.ready:
                    reply = {
                        "delivery_id": self.clock.pending[0],
                        "key": self.key.to_dict(),
                        "sequence_id": self.key.req_id,
                        "selection": "full_prompt",
                        "token_ranges": [[0, len(self.req.origin_input_ids)]],
                        "state": "delivered",
                    }
                    # Private object identity crosses the local Future only;
                    # neither V nor a caller-supplied JSON field can mint it.
                    self._network_receipt = reply
                    return reply
                if self._fanin.state in {"cancelled", "cancelling"}:
                    raise RuntimeError("V fan-in failed or is draining")
                await self._fanin.poll()
                await asyncio.sleep(interval)

    def unpack(self, reply):
        if reply is not self._network_receipt:
            raise ValueError("fan-in result is not this session's completion receipt")
        if self._rank_packed_plan is None:
            super().unpack(reply)
            return
        if self._canonical_staging is None:
            raise RuntimeError("rank-packed canonical staging is unavailable")
        # _network_receipt exists only after the complete V writer set has
        # terminal proof. Fence device visibility before reading the RDMA target.
        self.synchronize()
        scatter_rank_packed_bytes(
            self.staging, self._canonical_staging, self._rank_packed_plan
        )
        super().unpack(reply, staging=self._canonical_staging)

    async def ack_fanin(self):
        async with self._fanin_lock:
            if self._closed or self._fanin is None or self._network_receipt is None:
                raise RuntimeError("fan-in session closed before installation ACK")
            await self._fanin.ack_after_install()
            self._fanin.close()
            self._fanin = None

    async def progress_close(self):
        async with self._fanin_lock:
            if not self._closed:
                return False
            if self._fanin is not None:
                try:
                    await self._fanin.fence()
                    self._fanin.close()
                except Exception:
                    return False
                self._fanin = None
            # No control coroutine can publish after acquiring this lock while
            # _closed is set. The remaining pin is the original session owner.
            if self._refresh_owner is not None:
                self.release_refresh()
            return True

    async def close(self):
        result = await super().close()
        if result:
            if self._fanin_client is not None:
                await self._fanin_client.close()
                self._fanin_client = None
            if self._canonical_staging is not None:
                self._canonical_staging = None
                self._canonical_budget.release(self._canonical_budget_owner)
                self._canonical_budget = self._canonical_budget_owner = None
        return result


def refresh_fanin_steps(refresher, reqs):
    """Existing bootstrap driver polls each rank's own control-loop Future."""
    manager = refresher.manager
    sessions, due, error = [], [], None
    try:
        sessions = [manager.decode_sessions[manager.key_for(req)] for req in reqs]
        due = [
            s
            for s in sessions
            if s._cuda_refresh_driver is None and (s.due() or s.lease_error)
        ]
    except Exception as exc:
        error = str(exc)
    readiness = refresher._exchange(due=[s.key.transfer_id for s in due], error=error)
    errors = [v["error"] for v in readiness if v["error"]]
    if errors:
        return [(r, "; ".join(errors)) for r in reqs]
    wanted = {key for v in readiness for key in v["due"]}
    due = [s for s in sessions if s.key.transfer_id in wanted]
    if not due:
        return []
    prepared = []
    try:
        for s in due:
            rows = manager.scheduler.req_to_token_pool.req_to_token[
                s.req.req_pool_idx, : len(s.req.origin_input_ids) : manager.page_size
            ]
            s.prepare((rows // manager.page_size).cpu())
            prepared.append(s)
    except Exception as exc:
        error = str(exc)
    error = refresher._agree_error(error)
    if error:
        for s in prepared:
            s.release_refresh()  # No control coroutine was submitted.
        return [(s.req, error) for s in due]
    rounds = refresher._exchange(
        rounds=[(s.key.to_dict(), s.clock.pending) for s in due]
    )
    if any(v["rounds"] != rounds[0]["rounds"] for v in rounds):
        for s in prepared:
            s.release_refresh()
        return [(s.req, "D ranks disagree on fan-in refresh rounds") for s in due]

    async def deliver():
        return await asyncio.gather(
            *(s.deliver_fanin(refresher.POLL_INTERVAL_SECONDS) for s in due),
            return_exceptions=True,
        )

    future = None
    try:
        future = manager.control.submit(deliver())
    except Exception as exc:
        error = str(exc)
    results, future_error = yield future
    error = error or future_error
    if any(s._closed or s.req.finished() for s in due):
        error = error or "request closed during fan-in retrieval"
    if not error:
        try:
            if len(results) != len(due):
                raise ValueError("fan-in result count mismatch")
            for reply in results:
                if isinstance(reply, BaseException):
                    raise RuntimeError(str(reply))
        except Exception as exc:
            error = str(exc)
    error = refresher._agree_error(error)
    if error:
        return [(s.req, error) for s in due]
    try:
        for s, reply in zip(due, results):
            s.unpack(reply)
    except Exception as exc:
        error = str(exc)
    error = refresher._agree_error(error)
    if error:
        return [(s.req, error) for s in due]

    async def ack():
        replies = await asyncio.gather(
            *(s.ack_fanin() for s in due), return_exceptions=True
        )
        for reply in replies:
            if isinstance(reply, BaseException):
                raise RuntimeError(str(reply))

    future = None
    try:
        future = manager.control.submit(ack())
    except Exception as exc:
        error = str(exc)
    _, future_error = yield future
    error = error or future_error
    if any(s._closed or s.req.finished() for s in due):
        error = error or "request closed during fan-in ACK"
    error = refresher._agree_error(error)
    if error:
        return [(s.req, error) for s in due]
    for s in due:
        s._complete_refresh()
    return []
