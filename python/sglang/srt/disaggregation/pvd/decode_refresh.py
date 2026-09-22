"""Request-owned buffers, async bootstrap, and a synchronous periodic barrier.

Control traffic is issued by D rank 0. All TP ranks agree on preparation,
delivery, local unpack and ACK before admitting a newcomer, or before the next
forward when an already-running request has reached its periodic boundary.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import threading
import uuid

import torch
from sglang.srt.disaggregation.pvd.kv_packer import kv_components, unpack_full_prompt_kv
from sglang.srt.disaggregation.pvd.protocol import (
    PVD_GENERATION_METADATA_KEY,
    PVD_RECEIVER_EPOCH_METADATA_KEY,
    PVD_TRANSFER_LIFECYCLE_PROTOCOL,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.retrieval import RefreshClock
from sglang.srt.disaggregation.pvd.transfer_lifecycle import ResourceGuard, budget_of
from sglang.srt.disaggregation.pvd.worker_epoch import worker_epoch

logger = logging.getLogger(__name__)

# Bounded fence attempts made while a close is being driven inline. Exhausting
# them RETAINS the receive buffer and hands the session to the manager's
# pending-close list; it never releases an unfenced destination.
_INLINE_CLOSE_ATTEMPTS = 3


@dataclasses.dataclass(frozen=True)
class InitialPromptReceipt:
    """Local evidence minted only after delivery/unpack/ACK rank agreement.

    Not a wire authorization or a substitute for a native completion fence.
    Its object identity remains bound to the receiving session.
    """

    req: object
    request_id: str
    key: object
    receiver_epoch: str
    slot: object
    prompt: tuple
    outputs: tuple
    pages: tuple
    generation: str
    layout: str
    owner_thread: int


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
        # This D worker's process incarnation. V stamps it into every write
        # authorization, so a restarted D can never be handed a stale one.
        self.receiver_epoch = worker_epoch()
        # The registered staging buffer is reused across refreshes, but each
        # refresh publishes its own generation. A late write from an earlier
        # refresh therefore cannot be mistaken for the current one.
        self.receive_guard = None
        self._budget = None
        self._budget_owner = None
        self.generation = None
        self.identities = {}
        self._refresh_owner = None
        self._fenced = False
        self._initial_receipt = None
        self._cuda_refresh_driver = None

    def _complete_refresh(self):
        """Called ONLY at the final successful ACK/TP-agreement site below."""
        initial = self.clock.round == 0 and self.clock.pending[1] == 0
        self.release_refresh()
        self.clock.complete(self.clock.pending[0])
        if initial:
            self._initial_receipt = InitialPromptReceipt(
                self.req,
                getattr(self.req, "rid", None),
                self.key,
                self.receiver_epoch,
                self.req.req_pool_idx,
                tuple(self.req.origin_input_ids),
                tuple(self.req.output_ids),
                tuple(self.pages.tolist()),
                self.generation,
                self.manager.layout().fingerprint,
                threading.get_ident(),
            )

    def require_initial_prompt(self):
        """Return this session's receipt only while the exact newcomer is live."""
        receipt, req, manager = self._initial_receipt, self.req, self.manager
        if (
            receipt is None
            or receipt.owner_thread != threading.get_ident()
            or receipt.req is not req
            or not receipt.request_id
            or receipt.request_id != req.rid
            or receipt.key != self.key
            or manager.key_for(req) != self.key
            or manager.decode_sessions.get(self.key) is not self
            or self._closed
            or self.lease_error
            or not self._fenced
            or self._refresh_owner is not None
            or self.clock.pending is not None
            or self.clock.round != 1
            or self.clock.last_tokens != 0
            or receipt.receiver_epoch != self.receiver_epoch
            or receipt.generation != self.generation
            or receipt.slot != req.req_pool_idx
            or type(req.req_pool_idx) is not int
            or type(receipt.slot) is not int
            or receipt.slot <= 0
            or receipt.prompt != tuple(req.origin_input_ids)
            or receipt.outputs != tuple(req.output_ids)
            or len(receipt.outputs) != 1
            or receipt.pages != tuple(self.pages.tolist())
            or receipt.layout != manager.layout().fingerprint
            or req.finished()
            or req.is_retracted
        ):
            raise RuntimeError("exact live initial Prompt completion required")
        gate = manager.bootstrap_gate_for(req)
        if (
            gate is None
            or not gate.is_runnable
            or not gate.handed_off
            or gate.receiver_epoch != receipt.receiver_epoch
            or gate.delivery_id != f"{self.key.transfer_id}:bootstrap"
            or gate.prompt_tokens != len(receipt.prompt)
            or not any(req is queued for queued in manager.scheduler.waiting_queue)
        ):
            raise RuntimeError("installed final-waiting-queue bootstrap required")
        return receipt

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
        if self._cuda_refresh_driver is not None:
            return False
        return self.clock.due(self.decode_tokens)

    def prepare(self, pages):
        if self._cuda_refresh_driver is not None:
            raise RuntimeError("full Prompt refresh ownership transferred to CUDA driver")
        if self._closed or self.lease_error:
            raise RuntimeError(self.lease_error or "Decode session closed")
        self.pages = pages
        components = kv_components(self.manager.kv_pool)
        tokens = (
            math.ceil(len(self.req.origin_input_ids) / self.manager.page_size)
            * self.manager.page_size
        )
        size = sum(tokens * c[0].numel() * c.element_size() for c in components)
        if self._refresh_owner is not None:
            raise RuntimeError(
                "previous PVD refresh destination is still unfenced; "
                "the staging buffer cannot be reused yet"
            )
        if self.staging is None:
            # Reserve before allocating. An over-budget refresh must be refused
            # while torch.empty has not been called, not after the allocation
            # has already pushed the worker past its limit.
            budget = getattr(self.manager, "transfer_budget", None) or budget_of(
                self.manager.transfer_engine
            )
            budget_owner = (
                f"decode-staging:{self.key.transfer_id}:{self.manager.tp_rank}"
            )
            if budget is not None:
                budget.reserve(budget_owner, size, 0)
            self._budget = budget
            self._budget_owner = budget_owner
            self.staging = torch.empty(
                size, dtype=torch.uint8, device=components[0].device
            )
            registration = self.manager.transfer_engine.register_memory(
                self.staging,
                endpoint="pvd-decode",
                rank=self.manager.tp_rank,
                rail=self.manager.rail,
                metadata={"pvd_layout": self.manager.layout().to_dict()},
            )
            self.registration = registration

            def _release_receive():
                self.manager.transfer_engine.release_memory(registration)
                if budget is not None:
                    # Refund only after the MR is really gone.
                    budget.release(budget_owner)

            # Deregistration is a release request, never an immediate action.
            self.receive_guard = ResourceGuard(registration, _release_receive)
        elif self.staging.numel() != size:
            raise ValueError("immutable Prompt KV changed size during Decode")
        delivery_id = self.clock.begin(self.decode_tokens)
        # Pin before the descriptor is published. From this point a remote
        # writer may exist, so nothing may deregister this MR or reuse the
        # buffer until a matching identity fence proves the write stopped.
        owner = f"pvd-refresh:{delivery_id}"
        self.receive_guard.pin(owner)
        self._refresh_owner = owner
        self.generation = uuid.uuid4().hex
        self.identities = {}
        self._fenced = False
        descriptor = dataclasses.replace(
            self.registration.descriptor,
            backend_metadata={
                **self.registration.descriptor.backend_metadata,
                PVD_RECEIVER_EPOCH_METADATA_KEY: self.receiver_epoch,
                PVD_GENERATION_METADATA_KEY: self.generation,
            },
        )
        return {
            "key": self.key.to_dict(),
            "sequence_id": self.key.req_id,
            "delivery_id": delivery_id,
            "selection": "full_prompt",
            "destination": descriptor.to_dict(),
        }

    def expected_identity(self, shard_rank: int, sender_epoch: str) -> WriteIdentity:
        """Build the identity this D rank will accept for the current refresh.

        Every field except the V worker's incarnation is owned by D and is
        reconstructed here rather than copied from a reply.
        """
        if self.clock.pending is None or self._refresh_owner is None:
            raise RuntimeError("no PVD refresh is in flight")
        return WriteIdentity(
            protocol=PVD_TRANSFER_LIFECYCLE_PROTOCOL,
            sender_epoch=sender_epoch,
            receiver_epoch=self.receiver_epoch,
            transfer_id=f"{self.clock.pending[0]}:d{shard_rank}",
            region_id=self.registration.descriptor.region_id,
            generation=self.generation,
            shard_rank=shard_rank,
            key=self.key,
        )

    def adopt_identities(self, reply) -> None:
        """Save the write authorizations for this refresh.

        Only the V incarnation is new information. Everything else is checked
        against what this session published, so a forged or stale reply cannot
        install an authorization that a later fence would accept.
        """
        values = reply.get("write_identities")
        if not isinstance(values, dict) or not values:
            raise ValueError("retrieval reply carries no write identities")
        if self.clock.pending is None or self._refresh_owner is None:
            raise RuntimeError("no PVD refresh is in flight")
        delivery_id = self.clock.pending[0]
        adopted = {}
        for rank_value, value in values.items():
            identity = WriteIdentity.from_dict(value)
            rank = int(rank_value)
            if identity.shard_rank != rank:
                raise ValueError("write identity rank does not match its key")
            if identity.key != self.key:
                raise ValueError("write identity belongs to a different entry")
            if identity.transfer_id != f"{delivery_id}:d{rank}":
                raise ValueError("write identity does not belong to this refresh")
            adopted[rank] = identity
        own = adopted.get(self.manager.tp_rank)
        if own is None:
            raise ValueError("retrieval reply omits this D rank's write identity")
        # Only this rank owns its own destination, so only its own identity can
        # be verified in full: region, allocation generation and receiver epoch
        # belong to whichever rank published them. The other ranks' identities
        # are carried so the fence can close the whole delivery; they are never
        # this rank's evidence that its own buffer is safe.
        if own != self.expected_identity(self.manager.tp_rank, own.sender_epoch):
            raise ValueError(
                "retrieval reply write identity does not match this refresh"
            )
        # Sender epochs legitimately differ per V shard: each storage rank is
        # its own worker process with its own incarnation. Only this rank's
        # own sender epoch is pinned, by the full comparison above.
        self.identities = adopted

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

    def release_refresh(self) -> None:
        """Drop this refresh's destination pin after a proven transport terminal.

        Requires V's terminal proof, a matching fence, or the caller's proof
        that the prepared descriptor has never been published. A timeout is
        never sufficient. The MR stays registered until session close.
        """
        owner, self._refresh_owner = self._refresh_owner, None
        if owner is not None:
            self.receive_guard.unpin(owner)
        self._fenced = True

    async def progress_close(self) -> bool:
        """Drive one bounded fence step for the outstanding refresh.

        Returns True only when no remote writer can still touch the receive
        buffer. Returning False always retains every resource; it never
        degrades into releasing an unfenced destination.
        """
        if self._refresh_owner is None:
            return True
        delivery_id = self.clock.pending[0] if self.clock.pending else None
        if delivery_id is None:
            return False
        if not self.identities:
            # The descriptor was published but no authorization was adopted,
            # so recover V's saved identities before fencing. Being unable to
            # recover them is not proof that no write exists.
            try:
                record = await self.client.poll_delivery(delivery_id)
                self.adopt_identities(record)
            except Exception as exc:
                logger.warning(
                    "PVD receive buffer retained; cannot recover identities for %s: %s",
                    delivery_id,
                    exc,
                )
                return False
        try:
            reply = await self.client.fence_retrieval(
                delivery_id,
                [identity.to_dict() for identity in self.identities.values()],
            )
        except Exception as exc:
            logger.warning(
                "Retaining PVD receive buffer until V fences %s: %s",
                delivery_id,
                exc,
            )
            return False
        if reply.get("delivery_id") != delivery_id:
            # The transport client validates replies too. This second check
            # keeps a mismatched reply from ever reaching release_refresh.
            logger.warning(
                "PVD fence reply for %s named a different delivery; retained",
                delivery_id,
            )
            return False
        if reply.get("fenced") is not True:
            logger.info(
                "PVD fence for %s is still pending; receive buffer retained",
                delivery_id,
            )
            return False
        self.release_refresh()
        return True

    async def drive_close(self, attempts: int = _INLINE_CLOSE_ATTEMPTS) -> bool:
        for _ in range(max(1, attempts)):
            if await self.progress_close():
                return True
        return False

    def schedule_close(self):
        if self._close_future is None:
            self._closed = True
            self._close_future = self.manager.control.submit(self.close())

    async def close(self) -> bool:
        self._closed = True
        if self._lease_task is not None:
            self._lease_task.cancel()
            await asyncio.gather(self._lease_task, return_exceptions=True)
        # Each rank fences before releasing its own registered memory. A timed
        # out HTTP call, a cancelled request or an expired consumer lease do
        # not prove that remote RDMA writes stopped, so an undrained session is
        # handed to the manager rather than released here.
        if not await self.drive_close():
            logger.warning(
                "PVD receive buffer for %s is retained pending a valid fence",
                self.key,
            )
            self.manager.retain_pending_close(self)
            return False
        self.synchronize()
        if self.receive_guard is not None:
            # Request only: the guard defers deregistration until no refresh
            # owner remains, and keeps ownership if the callback fails.
            self.receive_guard.request_release()
            self.receive_guard = None
            self.registration = None
            self.staging = None
        if self.manager.tp_rank == 0:
            try:
                await self.client.release_consumer(self.key, self.consumer_id)
            except Exception:
                logger.warning("PVD consumer lease will expire for %s", self.key)
        return True


class PVDDecodeRefresher:
    # A retrieval may still be writing when V answers. Poll it to a terminal
    # state instead of treating "in flight" as a failure; the bound keeps one
    # scheduler step finite, and an unfinished delivery stays pinned.
    POLL_ATTEMPTS = 200
    POLL_INTERVAL_SECONDS = 0.005

    def __init__(self, manager):
        self.manager = manager
        self._bootstrap = None

    @property
    def bootstrap_pending(self):
        return self._bootstrap is not None

    @staticmethod
    def _future_result(future):
        if future is None:
            return None, None
        try:
            return future.result(), None
        except Exception as exc:
            return None, str(exc)

    def start_bootstrap(self, reqs):
        """Prepare on the scheduler thread; yield before waiting for V or ACK."""
        if self.bootstrap_pending:
            raise RuntimeError("a bootstrap wave is already pending")
        steps = self._refresh_steps(reqs)
        try:
            future = next(steps)
        except StopIteration as done:
            return list(reqs), done.value
        self._bootstrap = (list(reqs), steps, future)
        return [], []

    def poll_bootstrap(self):
        """Never wait on a network future. TP/GPU work stays on this thread."""
        if self._bootstrap is None:
            return [], []
        reqs, steps, future = self._bootstrap
        ready = future is None or future.done()
        # Only rank 0 owns the HTTP future. Agree before advancing generators,
        # otherwise ranks could enter different preparation/install collectives.
        if not all(item["ready"] for item in self._exchange(ready=ready)):
            return [], []
        try:
            future = steps.send(self._future_result(future))
        except StopIteration as done:
            self._bootstrap = None
            return reqs, done.value
        self._bootstrap = (reqs, steps, future)
        return [], []

    async def _drive_delivery(self, client, result):
        """Advance one retrieval result until V reports a terminal state."""
        delivery_id = result.get("delivery_id")
        for _ in range(self.POLL_ATTEMPTS):
            if result.get("state") != "v_writing":
                return result
            try:
                record = await client.poll_delivery(delivery_id)
            except Exception as exc:
                result["state"] = "failed"
                result["error"] = f"delivery poll failed: {exc}"
                return result
            result["state"] = record.get("state")
            result["error"] = record.get("error")
            if record.get("write_identities"):
                result["write_identities"] = record["write_identities"]
            if result.get("state") != "v_writing":
                return result
            await asyncio.sleep(self.POLL_INTERVAL_SECONDS)
        result["state"] = "failed"
        result["error"] = "V delivery did not reach a terminal state in time"
        return result

    async def progress_pending_closes(self) -> int:
        """Drive retained close sessions one bounded step each."""
        pending = self.manager.pending_decode_closes
        if not pending:
            return 0
        remaining = []
        for session in list(pending):
            try:
                drained = await session.progress_close()
            except Exception as exc:
                logger.warning("PVD retained close failed to progress: %s", exc)
                drained = False
            if drained:
                try:
                    await session.close()
                except Exception as exc:
                    logger.warning("PVD retained close cleanup failed: %s", exc)
                    remaining.append(session)
            else:
                remaining.append(session)
        self.manager.pending_decode_closes = remaining
        return len(remaining)

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
        # Retained sessions are owned by the manager, not by any request, so
        # they are driven here as well as from an explicit progress call.
        if self.manager.pending_decode_closes:
            self.manager.control.submit(self.progress_pending_closes())

    def release_request(self, req):
        """Also used when a queued request is removed without finished_reason."""
        close_gate = getattr(self.manager, "close_bootstrap_gate", None)
        if close_gate is not None:
            close_gate(req)
        key = self.manager.key_for(req)
        session = self.manager.decode_sessions.get(key)
        if session is not None and getattr(session, "_cuda_refresh_driver", None) is not None:
            # The sparse controller still needs this Entry's consumer lease.
            # Its driver closes the full session only AFTER native sparse drain.
            session._cuda_refresh_driver.cancel(req)
            return
        session = self.manager.decode_sessions.pop(key, None)
        if session is not None:
            session.schedule_close()

    def refresh(self, reqs):
        """Return (request, error) pairs to abort before prepare_for_decode."""
        steps = self._refresh_steps(reqs)
        try:
            future = next(steps)
            while True:
                future = steps.send(self._future_result(future))
        except StopIteration as done:
            return done.value

    def _refresh_steps(self, reqs):
        """Shared safe delivery protocol; drivers choose blocking or polling.

        Yield only for control-plane futures. Never run this generator (which
        performs TP collectives and GPU copies) on the control-loop thread.
        """
        sessions, due, error = [], [], None
        try:
            sessions = [
                self.manager.decode_sessions[self.manager.key_for(req)] for req in reqs
            ]
            # Include lease failures even when a periodic update is not due.
            due = [
                s for s in sessions
                if getattr(s, "_cuda_refresh_driver", None) is None
                and (s.due() or s.lease_error)
            ]
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
            # No descriptor has left D yet. Prepared pins can be dropped
            # without asking V to fence a delivery that was never published.
            for session, _ in zip(due, local):
                if session._refresh_owner is not None:
                    session.release_refresh()
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
            for session in due:
                session.release_refresh()
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
                client = self.manager.clients[group]
                response = await client.retrieve(sequences)
                results = response["results"]
                # V may answer while its writes are still on the wire. That is
                # not a failure: drive each one to a terminal state.
                return await asyncio.gather(
                    *(self._drive_delivery(client, result) for result in results)
                )

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

        future = None
        if self.manager.tp_rank == 0:
            try:
                future = self.manager.control.submit(retrieve_groups())
            except Exception as exc:
                error = str(exc)
        results, future_error = yield future
        error = error or future_error
        status = self._exchange(results=results, error=error)[0]
        error = status["error"]
        # Cancellation may have freed the request's final KV pages while the
        # WRITE still targeted its pinned staging buffer. Never unpack into
        # recycled pages; close/fence owns disposal of that staging buffer.
        if any(s._closed or s.req.finished() for s in due):
            error = error or "PVD request closed during retrieval"
        error = self._agree_error(error)
        if not error:
            try:
                results = status["results"]
                by_id = {r["delivery_id"]: r for r in results}
                if len(by_id) != len(due) or len(results) != len(due):
                    raise ValueError("invalid retrieval result count")
                for session in due:
                    reply = by_id[session.clock.pending[0]]
                    # Save the authorizations before touching the buffer, so a
                    # later fence compares against what this rank published
                    # rather than against whatever a fence reply claims.
                    session.adopt_identities(reply)
                    session.unpack(reply)
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

        future = None
        if self.manager.tp_rank == 0:
            try:
                future = self.manager.control.submit(ack_all())
            except Exception as exc:
                error = str(exc)
        _, future_error = yield future
        error = error or future_error
        if any(s._closed or s.req.finished() for s in due):
            error = error or "PVD request closed during ACK"
        error = self._agree_error(error)
        if error:
            return [(s.req, error) for s in due]
        for session in due:
            # Every shard reported DELIVERED, which V sets only after seeing a
            # native TERMINAL_SUCCESS. That is the transport-terminal proof
            # this refresh's destination pin was waiting for.
            session._complete_refresh()
            logger.debug(
                "PVD refresh complete: sequence=%s round=%s decode_tokens=%s",
                session.key.req_id,
                session.clock.round - 1,
                session.decode_tokens,
            )
        return []
