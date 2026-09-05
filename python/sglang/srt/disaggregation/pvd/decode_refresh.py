"""Request-owned receive buffers and a synchronous continuous-batch KV barrier.

Control traffic is issued by D rank 0. All TP ranks agree on preparation,
delivery, local unpack and ACK before any rank launches the next forward.
"""

from __future__ import annotations

import asyncio
import logging
import math

import torch
from sglang.srt.disaggregation.pvd.kv_packer import kv_components, unpack_full_prompt_kv
from sglang.srt.disaggregation.pvd.retrieval import RefreshClock

logger = logging.getLogger(__name__)


class PVDDecodeSession:
    def __init__(self, manager, req):
        self.manager = manager
        self.req = req
        self.key = manager.key_for(req)
        self.client = manager.client_for(req)
        self.clock = RefreshClock(
            req.pvd_delivery_id, manager.scheduler.server_args.pvd_kv_refresh_interval
        )
        self.consumer_id = req.pvd_delivery_id
        self.staging = None
        self.registration = None
        self.pages = None
        self.lease_error = None
        self._lease_task = None
        self._closed = False
        self._close_future = None

    async def initialize(self, runtime):
        record = await self.manager.wait_for_stored_entry(self.key, runtime)
        if self._closed:
            raise RuntimeError("Decode session closed before KV_READY")
        if self.manager.tp_rank == 0:
            lease = await self.client.renew_consumer(self.key, self.consumer_id)
            if self._closed:
                await self.client.release_consumer(self.key, self.consumer_id)
                raise RuntimeError("Decode session closed during lease acquisition")
            self._lease_task = asyncio.create_task(self._keepalive(lease))
        return record

    async def _keepalive(self, lease):
        try:
            while True:
                await asyncio.sleep(lease["renew_after_seconds"])
                lease = await self.client.renew_consumer(self.key, self.consumer_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.lease_error = str(exc)

    @property
    def decode_tokens(self):
        # The first output token is sampled on P, before the first D forward.
        return max(0, len(self.req.output_ids) - 1)

    def due(self):
        return self.clock.due(self.decode_tokens)

    def prepare(self, pages):
        if self._closed or self.lease_error:
            raise RuntimeError(self.lease_error or "Decode session closed")
        self.pages = pages
        components = kv_components(self.manager.kv_pool)
        tokens = (
            math.ceil(len(self.req.origin_input_ids) / self.manager.page_size)
            * self.manager.page_size
        )
        size = sum(tokens * c[0].numel() * c.element_size() for c in components)
        if self.staging is None:
            self.staging = torch.empty(
                size, dtype=torch.uint8, device=components[0].device
            )
            self.registration = self.manager.transfer_engine.register_memory(
                self.staging,
                endpoint="pvd-decode",
                rank=self.manager.tp_rank,
                rail=self.manager.rail,
                metadata={"pvd_layout": self.manager.layout().to_dict()},
            )
        elif self.staging.numel() != size:
            raise ValueError("immutable Prompt KV changed size during Decode")
        delivery_id = self.clock.begin(self.decode_tokens)
        return {
            "key": self.key.to_dict(),
            "sequence_id": self.key.req_id,
            "delivery_id": delivery_id,
            "selection": "full_prompt",
            "destination": self.registration.descriptor.to_dict(),
        }

    def unpack(self, reply):
        expected_id = self.clock.pending[0] if self.clock.pending else None
        if (
            reply.get("delivery_id") != expected_id
            or reply.get("key") != self.key.to_dict()
            or reply.get("sequence_id") != self.key.req_id
        ):
            raise ValueError("retrieval reply identity mismatch")
        if reply.get("state") != "delivered":
            raise RuntimeError(f"V retrieval failed: {reply.get('error', reply)}")
        if reply.get("selection") != "full_prompt" or reply.get("token_ranges") != [
            [0, len(self.req.origin_input_ids)]
        ]:
            raise ValueError("unsupported retrieval selection or token ranges")
        self.synchronize()
        unpack_full_prompt_kv(
            self.staging,
            self.manager.kv_pool,
            self.pages,
            page_size=self.manager.page_size,
            prompt_token_count=len(self.req.origin_input_ids),
        )
        self.synchronize()

    def synchronize(self):
        if self.staging is not None and self.staging.is_cuda:
            torch.cuda.synchronize(self.staging.device)

    def schedule_close(self):
        if self._close_future is None:
            self._closed = True
            self._close_future = self.manager.control.submit(self.close())

    async def close(self):
        self._closed = True
        if self._lease_task is not None:
            self._lease_task.cancel()
            await asyncio.gather(self._lease_task, return_exceptions=True)
        # Each rank fences before releasing its own registered memory. If V is
        # unreachable, retain the destination until fencing succeeds. A timed-
        # out HTTP call alone does not prove that remote RDMA writes stopped.
        if self.clock.pending is not None:
            while True:
                try:
                    reply = await self.client.fence_retrieval(self.clock.pending[0])
                    if (
                        reply.get("fenced") is not True
                        or reply.get("delivery_id") != self.clock.pending[0]
                    ):
                        raise RuntimeError("invalid retrieval fence response")
                    break
                except Exception:
                    logger.warning(
                        "Retaining PVD receive buffer until V fences %s",
                        self.clock.pending[0],
                    )
                    await asyncio.sleep(1)
        self.synchronize()
        if self.registration is not None:
            self.manager.transfer_engine.release_memory(self.registration)
            self.registration = None
            self.staging = None
        if self.manager.tp_rank == 0:
            try:
                await self.client.release_consumer(self.key, self.consumer_id)
            except Exception:
                logger.warning("PVD consumer lease will expire for %s", self.key)


class PVDDecodeRefresher:
    def __init__(self, manager):
        self.manager = manager

    def _exchange(self, **payload):
        return self.manager.gather_rank_objects(
            {"rank": self.manager.tp_rank, **payload}
        )

    def _agree_error(self, error):
        errors = [v["error"] for v in self._exchange(error=error) if v["error"]]
        return "; ".join(errors) if errors else None

    def cleanup_finished(self):
        for key, session in list(self.manager.decode_sessions.items()):
            if session.req.finished():
                self.release_request(session.req)

    def release_request(self, req):
        """Also used when a queued request is removed without finished_reason."""
        session = self.manager.decode_sessions.pop(self.manager.key_for(req), None)
        if session is not None:
            session.schedule_close()

    def refresh(self, reqs):
        """Return (request, error) pairs to abort before prepare_for_decode."""
        sessions, due, error = [], [], None
        try:
            sessions = [
                self.manager.decode_sessions[self.manager.key_for(req)] for req in reqs
            ]
            # Include lease failures even when a periodic update is not due.
            due = [s for s in sessions if s.due() or s.lease_error]
        except Exception as exc:
            error = str(exc)
        # The rank-0 heartbeat is asynchronous; agree on the union before any
        # per-sequence collective so ranks cannot disagree on collective count.
        readiness = self._exchange(due=[s.key.transfer_id for s in due], error=error)
        errors = [item["error"] for item in readiness if item["error"]]
        if errors:
            return [(req, "; ".join(errors)) for req in reqs]
        wanted = {key for item in readiness for key in item["due"]}
        due = [s for s in sessions if s.key.transfer_id in wanted]
        if not due:
            return []
        local, error = [], None
        try:
            for session in due:
                req = session.req
                locations = self.manager.scheduler.req_to_token_pool.req_to_token[
                    req.req_pool_idx,
                    : len(req.origin_input_ids) : self.manager.page_size,
                ]
                pages = (locations // self.manager.page_size).cpu()
                local.append(session.prepare(pages))
        except Exception as exc:
            error = str(exc)
        error = self._agree_error(error)
        if error:
            return [(s.req, error) for s in due]
        ranks = self._exchange(sequences=local)
        identities = [
            {k: v for k, v in item.items() if k != "destination"} for item in local
        ]
        if any(
            [
                {k: v for k, v in item.items() if k != "destination"}
                for item in rank["sequences"]
            ]
            != identities
            for rank in ranks
        ):
            return [
                (s.req, "PVD TP ranks disagree on retrieval identities") for s in due
            ]
        payloads = []
        for i, item in enumerate(local):
            payload = {k: v for k, v in item.items() if k != "destination"}
            payload["destinations"] = {
                str(rank["rank"]): rank["sequences"][i]["destination"] for rank in ranks
            }
            payloads.append(payload)

        async def retrieve_groups():
            groups = {}
            for session, payload in zip(due, payloads):
                group = self.manager.vector_group_for(session.req)
                groups.setdefault(group, []).append(payload)

            async def get(group, sequences):
                response = await self.manager.clients[group].retrieve(sequences)
                return response["results"]

            batches = await asyncio.gather(
                *(get(group, seqs) for group, seqs in groups.items()),
                return_exceptions=True,
            )
            results = []
            for batch in batches:
                if isinstance(batch, BaseException):
                    raise batch
                results.extend(batch)
            return results

        results = None
        if self.manager.tp_rank == 0:
            try:
                results = self.manager.control.submit(retrieve_groups()).result()
            except Exception as exc:
                error = str(exc)
        status = self._exchange(results=results, error=error)[0]
        error = status["error"]
        if not error:
            try:
                results = status["results"]
                by_id = {r["delivery_id"]: r for r in results}
                if len(by_id) != len(due) or len(results) != len(due):
                    raise ValueError("invalid retrieval result count")
                for session in due:
                    session.unpack(by_id[session.clock.pending[0]])
            except Exception as exc:
                error = str(exc)
        error = self._agree_error(error)
        if error:
            return [(s.req, error) for s in due]

        async def ack_all():
            responses = await asyncio.gather(
                *(s.client.ack_delivery(s.clock.pending[0]) for s in due),
                return_exceptions=True,
            )
            for session, response in zip(due, responses):
                if isinstance(response, BaseException):
                    raise response
                if (
                    response.get("delivery_id") != session.clock.pending[0]
                    or response.get("state") != "released"
                ):
                    raise ValueError("invalid retrieval ACK response")

        if self.manager.tp_rank == 0:
            try:
                self.manager.control.submit(ack_all()).result()
            except Exception as exc:
                error = str(exc)
        error = self._agree_error(error)
        if error:
            return [(s.req, error) for s in due]
        for session in due:
            session.clock.complete(session.clock.pending[0])
            logger.debug(
                "PVD refresh complete: sequence=%s round=%s decode_tokens=%s",
                session.key.req_id,
                session.clock.round - 1,
                session.decode_tokens,
            )
        return []
