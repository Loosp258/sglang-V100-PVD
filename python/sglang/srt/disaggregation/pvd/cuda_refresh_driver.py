"""Owner-thread CUDA refresh polling over authoritative Req output_ids.

This explicit driver is not a serving-mode switch. The caller must share its
target arbiter/lock with CUDA batch execution and poll BETWEEN synchronous
forwards. The normal result processor remains the only token writer. Retiring
a registration drains its controller, not the caller's Req/KV allocator rows.
"""

import asyncio
import logging
import math
import os
import time
from array import array
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType

from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import (
    LifecycleError,
    TargetExecutionArbiter,
)
from sglang.srt.disaggregation.pvd.cuda_prefetch_request import CUDAPrefetchRequest
from sglang.srt.disaggregation.pvd.prediction import CommittedPrefix

logger = logging.getLogger(__name__)


def _timeline(message, *args):
    if os.environ.get("PVD_PROFILE_REFRESH_TIMELINE") == "1":
        try:
            logger.info(message, *args)
        except Exception:
            pass  # Diagnostics must never change refresh ownership.


def _task_site(task):
    """Bounded await chain of code locations, never Prompt or Q contents."""
    try:
        awaited = task.get_coro()
        locations = []
        for _ in range(8):
            frame = getattr(awaited, "cr_frame", None) or getattr(
                awaited, "gi_frame", None
            )
            if frame is not None:
                locations.append(
                    f"{os.path.basename(frame.f_code.co_filename)}:"
                    f"{frame.f_lineno}:{frame.f_code.co_name}"
                )
            next_awaited = getattr(awaited, "cr_await", None) or getattr(
                awaited, "gi_yieldfrom", None
            )
            if next_awaited is None:
                break
            awaited = next_awaited
        return " > ".join(locations) if locations else "no-python-frame"
    except Exception:
        return "unavailable"


@dataclass
class _Request:
    req: object
    controller: CUDAPrefetchRequest
    clients: object
    prompt: tuple
    outputs: tuple
    slot: int
    timeout: float
    refresh: object = None
    close_task: object = None
    capture_lease: object = None
    deadline: float | None = None
    ready: bool = False
    stopping: bool = False
    quarantined: bool = False
    error: object = None
    retirement: object = None
    full_session: object = None
    receiver_lease: object = None
    provisional: bool = False
    provisional_source: object = None
    provisional_pool_owner: object = None
    boundary_observed_at: float | None = None
    last_progress_log_at: float | None = None


class CUDARefreshDriver:
    """Bounded registrations; one capture launch per owner poll, no I/O wait.

    With a synchronous caller, a private loop is advanced by one nonblocking
    event-loop iteration per poll. An async caller uses its existing loop.
    Probe/copy/stage/install all run on this same thread. A poll can execute a
    synchronous model probe: this is NOT a compute-overlap or latency claim.
    """

    def __init__(
        self, arbiter, *, max_requests, max_prefix_tokens, clock=time.monotonic
    ):
        if not isinstance(arbiter, TargetExecutionArbiter):
            raise LifecycleError("explicit shared target arbiter required")
        arbiter.owner()
        if any(type(v) is not int or v <= 0 for v in (max_requests, max_prefix_tokens)):
            raise LifecycleError("positive request and prefix bounds required")
        self.arbiter = arbiter
        self.max_requests, self.max_prefix_tokens = max_requests, max_prefix_tokens
        self._clock = clock
        self._records = {}
        self._execution_lock = None
        self._closing = self._pumping = False
        self._source_quarantine = None
        poll_turns = os.environ.get("PVD_REFRESH_POLL_TURNS", "1")
        if (
            not poll_turns.isascii()
            or not poll_turns.isdecimal()
            or not 1 <= int(poll_turns) <= 8
        ):
            raise LifecycleError("PVD_REFRESH_POLL_TURNS must be an integer in [1, 8]")
        self._poll_turns = int(poll_turns)
        try:
            self._loop = asyncio.get_running_loop()
            self._owns_loop = False
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            self._owns_loop = True

    def _owner(self):
        self.arbiter.owner()
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if (self._owns_loop and current not in (None, self._loop)) or (
            not self._owns_loop and current is not self._loop
        ):
            raise LifecycleError("CUDA refresh must use its original owner loop")

    def _tokens(self, values):
        if (
            not isinstance(values, (list, tuple, array))
            or len(values) > self.max_prefix_tokens
        ):
            raise LifecycleError("bounded authoritative token sequence required")
        if not values or any(type(v) is not int or v < 0 for v in values):
            raise LifecycleError("nonempty nonnegative token sequence required")
        return tuple(values)

    def register(
        self,
        req,
        controller,
        *,
        clients,
        timeout_seconds,
        initial_import_pending=False,
        initial_session=None,
        pool_owner=None,
    ):
        self._owner()
        if type(initial_import_pending) is not bool:
            raise LifecycleError("initial import state must be explicit")
        if initial_import_pending:
            from sglang.srt.disaggregation.pvd.cuda_model_attention import (
                CUDAModelPools,
            )
            from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeSession
            from sglang.srt.disaggregation.pvd.transfer_lifecycle import ResourceGuard

            if (
                not isinstance(initial_session, PVDDecodeSession)
                or initial_session.req is not req
                or initial_session._cuda_refresh_driver is not None
                or getattr(initial_session, "_cuda_prompt_importer", None) is not None
                or not isinstance(pool_owner, ResourceGuard)
                or not isinstance(pool_owner.value, CUDAModelPools)
            ):
                raise LifecycleError(
                    "unclaimed full Prompt source and model pools required"
                )
            receipt = initial_session.require_initial_prompt()
            cache = initial_session.manager.scheduler.tree_cache
            pools = pool_owner.value
            if (
                receipt.request_id != req.rid
                or receipt.slot != req.req_pool_idx
                or pools.req_pool is not cache.req_to_token_pool
                or pools.kv_pool is not cache.token_to_kv_pool_allocator.get_kvcache()
            ):
                raise LifecycleError("provisional source and model pools differ")
        elif initial_session is not None or pool_owner is not None:
            raise LifecycleError(
                "initial source/owner require provisional registration"
            )
        if (
            self._closing
            or self._source_quarantine is not None
            or self._loop.is_closed()
            or len(self._records) >= self.max_requests
        ):
            raise LifecycleError("CUDA refresh admission is closed or full")
        if not isinstance(controller, CUDAPrefetchRequest):
            raise LifecycleError("explicit CUDA request controller required")
        controller._require_query_drained()
        identity = controller.group.coordinator.identity
        if (
            req.rid != identity[0]
            or req.rid in self._records
            or getattr(controller, "_refresh_driver_claimed", False)
            or getattr(controller, "_initial_import_pending", False)
            is not initial_import_pending
            or type(req.req_pool_idx) is not int
            or req.req_pool_idx <= 0
            or req.is_retracted
            or req.finished()
            or any(r.slot == req.req_pool_idx for r in self._records.values())
        ):
            raise LifecycleError("unique live Req/controller/slot required")
        prompt, outputs = (
            self._tokens(req.origin_input_ids),
            self._tokens(req.output_ids),
        )
        state = controller.group.coordinator.snapshot()
        if (
            len(outputs) != 1
            or len(prompt) + 1 > self.max_prefix_tokens
            or state["state"] != "idle"
            or state["installed_tokens"] != (None if initial_import_pending else 0)
            or (not initial_import_pending and not controller.can_decode(0))
            or (initial_import_pending and state["completed"] is not None)
            or controller._active is not None
            or controller._tasks
            or any(
                m["prompt_tokens"] != len(prompt) for m in controller._metadata.values()
            )
        ):
            raise LifecycleError(
                "complete Prompt plus P's first token required before admission"
            )
        if (
            set(clients) != set(controller._routes)
            or any(type(rank) is not int for rank in clients)
            or type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or controller.pipeline.draft_config.predict_tokens < state["lead_tokens"]
        ):
            raise LifecycleError(
                "exact routes, sufficient draft horizon and finite timeout required"
            )
        lock = controller.pipeline._lock
        if self._execution_lock is not None and lock is not self._execution_lock:
            raise LifecycleError("all requests must share the target execution lock")
        self._execution_lock = lock
        self._records[req.rid] = _Request(
            req,
            controller,
            MappingProxyType(dict(clients)),
            prompt,
            outputs,
            req.req_pool_idx,
            float(timeout_seconds),
            provisional=initial_import_pending,
            provisional_source=initial_session,
            provisional_pool_owner=pool_owner,
        )
        if initial_import_pending:
            # Make release_request() delegate to this driver before any bank
            # copy can begin; no await or fallible call separates the claims.
            initial_session._cuda_refresh_driver = self
        controller._refresh_driver_claimed = True

    def quarantine_provisional(self, req, reason):
        """Retain an unclaimed receiver after any uncertain admission mutation.

        This is deliberately not rollback. A provisional record cannot be
        polled, decoded or released here; both model pools are poisoned before
        another allocator user can reuse their rows. The source session and
        pool owner remain strongly referenced until worker termination or an
        explicit future recovery protocol proves every native operation safe.
        """
        self._owner()
        record = self._records.get(req.rid)
        if (
            record is None
            or record.req is not req
            or not record.provisional
            or record.full_session is not None
            or record.provisional_source is None
            or record.provisional_pool_owner is None
            or type(reason) is not str
            or not reason.strip()
        ):
            raise LifecycleError(
                "exact provisional Req, source and pool owner required"
            )
        if record.quarantined:
            return
        session = record.provisional_source
        pool_owner = record.provisional_pool_owner
        cache = session.manager.scheduler.tree_cache
        pools = pool_owner.value
        if (
            pools.req_pool is not cache.req_to_token_pool
            or pools.kv_pool is not cache.token_to_kv_pool_allocator.get_kvcache()
        ):
            raise LifecycleError("provisional source and model pools differ")
        message = "CUDA provisional admission uncertain: " + reason[:256]
        cache.req_to_token_pool.pvd_cuda_retirement_error = message
        cache.token_to_kv_pool_allocator.pvd_cuda_retirement_error = message
        record.error = LifecycleError(message)
        record.quarantined = True
        self._source_quarantine = message
        if not self.arbiter.busy:
            record.receiver_lease = self.arbiter.acquire()

    def claim_received_session(self, session):
        """One-way switch after receiver import, registration and release binding.

        The original session keeps its consumer lease and staging registration
        until sparse controller close proves no Delivery writer remains.
        """
        from sglang.srt.disaggregation.pvd.cuda_prompt_bootstrap import (
            CUDAPromptBootstrap,
        )
        from sglang.srt.disaggregation.pvd.cuda_request_release import (
            CUDARequestRelease,
        )
        from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeSession

        self._owner()
        if not isinstance(session, PVDDecodeSession):
            raise LifecycleError("real full Prompt receiver session required")
        receipt = session.require_initial_prompt()
        record = self._records.get(session.req.rid)
        importer = getattr(session, "_cuda_prompt_importer", None)
        if (
            self.arbiter.busy
            or record is None
            or record.req is not session.req
            or record.stopping
            or record.full_session is not None
            or session._cuda_refresh_driver
            is not (self if record.provisional else None)
            or (record.provisional and record.provisional_source is not session)
            or not isinstance(record.retirement, CUDARequestRelease)
            or record.retirement.state != "attached"
            or record.retirement.driver is not self
            or not isinstance(importer, CUDAPromptBootstrap)
            or importer.group is not record.controller.group
            or not importer._used
            or importer._quarantined
            or importer._receive_lease is not None
            or importer._received_session is not session
            or importer._received_receipt is not receipt
            or importer._received_pool_owner is not record.retirement.pool_owner
            or importer._lock is not self._execution_lock
            or record.prompt != receipt.prompt
            or record.outputs != receipt.outputs
            or record.slot != receipt.slot
            or record.refresh is not None
            or not record.controller.can_decode(0)
        ):
            raise LifecycleError(
                "completed import, registration and release owner required"
            )
        record.retirement._binding()
        # No await or fallible work between the two ownership publications.
        record.full_session = session
        session._cuda_refresh_driver = self
        record.provisional = False
        record.provisional_source = None
        record.provisional_pool_owner = None

    def _session_binding(self, record):
        session = record.full_session or record.provisional_source
        if session is not None and (
            session.req is not record.req
            or session._cuda_refresh_driver is not self
            or session.manager.key_for(record.req) != session.key
            or session.manager.decode_sessions.get(session.key) is not session
        ):
            raise LifecycleError("CUDA source session ownership changed")
        return session

    def _session_live(self, record):
        session = self._session_binding(record)
        if session is not None and (
            session._closed
            or session.lease_error
            or not session._fenced
            or session._refresh_owner is not None
            or session.clock.pending is not None
            or session.clock.round != 1
            or session.clock.last_tokens != 0
        ):
            raise LifecycleError(
                "CUDA source session closed, changed or lost its lease"
            )

    async def _close_controller(self, record):
        await record.controller.aclose()
        session = self._session_binding(record)
        if session is not None:
            # Keep keepalive on its original control loop. Never await its
            # asyncio Task from the driver's separate owner loop.
            session.schedule_close()
            closed = await asyncio.wrap_future(session._close_future)
            if (
                closed is not True
                or session.receive_guard is not None
                or session._refresh_owner is not None
            ):
                raise LifecycleError("full Prompt source close remains undrained")
            session.manager.close_bootstrap_gate(record.req)

    def _quarantine_receiver(self, record, exc):
        """A failed bound-source close is not permission to reuse shared pools."""
        self._source_quarantine = str(exc)[:512]
        if not self.arbiter.busy:
            record.receiver_lease = self.arbiter.acquire()
        source = record.full_session or record.provisional_source
        cache = record.retirement.cache if record.retirement is not None else None
        if cache is None:
            cache = source.manager.scheduler.tree_cache
        reason = "CUDA receiver close uncertain; worker pools quarantined"
        cache.req_to_token_pool.pvd_cuda_retirement_error = reason
        cache.token_to_kv_pool_allocator.pvd_cuda_retirement_error = reason

    def _observe(self, record):
        self._session_live(record)
        req, controller = record.req, record.controller
        outputs = self._tokens(req.output_ids)
        if (
            req.rid != controller.group.coordinator.identity[0]
            or self._tokens(req.origin_input_ids) != record.prompt
            or req.req_pool_idx != record.slot
            or outputs[: len(record.outputs)] != record.outputs
            or len(record.prompt) + len(outputs) > self.max_prefix_tokens
        ):
            raise LifecycleError("authoritative Req identity/prefix/slot changed")
        record.outputs = outputs
        return len(outputs) - 1  # P's first token never advances the D clock.

    def _release_capture(self, record):
        if record.capture_lease is not None:
            pipeline = record.controller.pipeline
            if (
                self._source_quarantine is not None
                or pipeline._quarantined
                or pipeline.probe._quarantined
                or pipeline.provider.degraded
                or record.controller._session._copy_unknown
            ):
                # A retained RLock is reentrant on this same owner thread.
                # Keep the shared arbiter lease too, so a later model batch
                # cannot enter merely because it can reacquire that RLock.
                return
            self.arbiter.release(record.capture_lease)
            record.capture_lease = None

    @contextmanager
    def _capture(self, record):
        try:
            if (
                self._source_quarantine is not None
                or record.stopping
                or record.capture_lease is None
            ):
                raise LifecycleError("stale CUDA capture dispatch")
            expected = record.outputs
            self._observe(record)
            if (
                record.outputs != expected
                or record.req.finished()
                or record.req.is_retracted
            ):
                raise LifecycleError("queued CUDA prefix changed before capture")
            yield
        finally:
            self._release_capture(record)

    def _stop(self, record, reason):
        if record.stopping or record.quarantined:
            return
        if record.provisional:
            source = record.provisional_source
            cache = source.manager.scheduler.tree_cache
            importer = getattr(source, "_cuda_prompt_importer", None)
            if (
                record.retirement is None
                or getattr(importer, "_quarantined", False)
                or getattr(cache.req_to_token_pool, "pvd_cuda_retirement_error", None)
                is not None
                or getattr(
                    cache.token_to_kv_pool_allocator,
                    "pvd_cuda_retirement_error",
                    None,
                )
                is not None
            ):
                self.quarantine_provisional(record.req, reason)
                return
        record.stopping = True
        if record.refresh is not None:
            record.refresh.cancel()
        try:
            record.controller.cancel(reason)
        except BaseException as exc:
            record.error, record.quarantined = exc, True
            if record.full_session is not None or record.provisional_source is not None:
                self._quarantine_receiver(record, exc)
            raise

    def cancel(self, req, reason="request finished, cancelled or retracted"):
        self._owner()
        record = self._records.get(req.rid)
        if record is None or record.req is not req:
            raise LifecycleError("unregistered Req incarnation")
        self._stop(record, reason)

    def _advance(self):
        due = None
        for key, record in tuple(self._records.items()):
            if record.quarantined:
                continue
            if record.provisional and not record.stopping:
                continue
            if (
                os.environ.get("PVD_PROFILE_REFRESH_TIMELINE") == "1"
                and record.refresh is not None
                and not record.refresh.done()
            ):
                now = self._clock()
                if (
                    record.last_progress_log_at is None
                    or now - record.last_progress_log_at >= 10.0
                ):
                    record.last_progress_log_at = now
                    children = getattr(record.controller, "_tasks", ())
                    _timeline(
                        "PVD timeline event=refresh_pending request_id=%s "
                        "parent_site=%s child_sites=%s t=%.6f",
                        record.req.rid,
                        _task_site(record.refresh),
                        tuple(_task_site(task) for task in children),
                        now,
                    )
            if record.refresh is not None and record.refresh.done():
                self._release_capture(record)  # Also cancel-before-first-dispatch.
                try:
                    record.refresh.result()
                except (Exception, asyncio.CancelledError) as exc:
                    record.error = exc
                    self._stop(record, "CUDA refresh failed")
                else:
                    if not record.ready:
                        _timeline(
                            "PVD timeline event=refresh_ready request_id=%s "
                            "boundary=%s t=%.6f",
                            record.req.rid,
                            record.controller.pending_install_boundary,
                            self._clock(),
                        )
                    record.ready = not record.stopping
            if record.close_task is not None:
                if record.close_task.done():
                    try:
                        record.close_task.result()
                        session = self._session_binding(record)
                        if record.retirement is not None:
                            record.retirement.release_after_controller_close()
                    except (Exception, asyncio.CancelledError) as exc:
                        # No retry or capacity refund after uncertain cleanup.
                        record.error, record.quarantined = exc, True
                        if (
                            record.full_session is not None
                            or record.provisional_source is not None
                        ):
                            self._quarantine_receiver(record, exc)
                            return
                    else:
                        if session is not None:
                            session.manager.decode_sessions.pop(session.key)
                            session._cuda_refresh_driver = None
                        del self._records[key]
                continue
            if not record.stopping:
                try:
                    if record.req.finished() or record.req.is_retracted:
                        self._stop(record, "Req finished or retracted")
                    else:
                        n = self._observe(record)
                        controller = record.controller
                        controller._live()
                        state = controller.group.coordinator.snapshot()
                        boundary = (
                            controller.pending_install_boundary
                            or state["next_boundary"]
                        )
                        if n > boundary:
                            raise LifecycleError(
                                "Decode crossed an uninstalled refresh boundary"
                            )
                        if n == boundary and record.boundary_observed_at is None:
                            # Scheduler polling may observe the boundary after
                            # the token was committed. This is an observed
                            # lower bound, not the exact token timestamp or
                            # network-only wait.
                            record.boundary_observed_at = self._clock()
                        if (
                            record.deadline is not None
                            and self._clock() >= record.deadline
                        ):
                            raise LifecycleError("CUDA refresh timeout")
                        if record.ready and n == boundary and not self.arbiter.busy:
                            if controller.try_install(
                                {rank: n for rank in controller._routes}
                            ):
                                observed_at = record.boundary_observed_at
                                record.boundary_observed_at = None
                                record.refresh = record.deadline = None
                                record.ready = False
                                # Optional diagnostics cannot turn an already
                                # committed bank switch into a failed refresh.
                                try:
                                    logger.info(
                                        "PVD boundary installed: request_id=%s "
                                        "boundary=%d observed_to_install_seconds=%.6f",
                                        record.req.rid,
                                        boundary,
                                        max(0.0, self._clock() - observed_at),
                                    )
                                except Exception:
                                    pass
                                _timeline(
                                    "PVD timeline event=installed request_id=%s "
                                    "boundary=%d t=%.6f",
                                    record.req.rid,
                                    boundary,
                                    self._clock(),
                                )
                        elif (
                            record.refresh is None
                            and n >= boundary - state["lead_tokens"]
                        ):
                            if due is None:
                                due = (record, n, boundary)
                except Exception as exc:
                    record.error = exc
                    self._stop(record, "Req observation or CUDA refresh failed")
            if record.stopping and (record.refresh is None or record.refresh.done()):
                coroutine = self._close_controller(record)
                try:
                    record.close_task = self._loop.create_task(coroutine)
                except BaseException:
                    coroutine.close()
                    raise
        if due is not None and not self._closing and not self.arbiter.busy:
            record, n, boundary = due
            prefix = CommittedPrefix(
                record.req.rid,
                record.prompt + record.outputs,
                n,
                f"{record.controller.group.coordinator.identity[1]}:{n}",
            )
            record.capture_lease = self.arbiter.acquire()
            record.deadline = self._clock() + record.timeout
            coroutine = record.controller.refresh(
                prefix,
                query_positions=(len(prefix.tokens) + boundary - n - 1,),
                clients=record.clients,
                execution_scope=lambda: self._capture(record),
                index_ready_wait_seconds=record.timeout,
            )
            try:
                record.refresh = self._loop.create_task(coroutine)
                _timeline(
                    "PVD timeline event=refresh_scheduled request_id=%s "
                    "boundary=%d committed_tokens=%d t=%.6f",
                    record.req.rid,
                    boundary,
                    n,
                    self._clock(),
                )
            except BaseException:
                coroutine.close()
                self._release_capture(record)
                record.deadline = None
                raise

    def poll(self):
        self._owner()
        if self._source_quarantine is not None:
            raise LifecycleError(
                "CUDA receiver close uncertain; driver quarantined: "
                + self._source_quarantine
            )
        if self._pumping or self._loop.is_closed():
            raise LifecycleError("CUDA refresh loop is closed or reentered")
        if self.arbiter.busy and not any(
            r.capture_lease is not None for r in self._records.values()
        ):
            raise LifecycleError("poll only between target forwards/result processing")
        if not self._owns_loop:
            self._pumping = True
            try:
                self._advance()
                return self.snapshot()
            finally:
                self._pumping = False
        failures = []

        def advance():
            try:
                self._advance()
            except BaseException as exc:
                failures.append(exc)

        self._pumping = True
        try:
            # Each turn is nonblocking: stop is already scheduled before
            # run_forever. Extra bounded turns let callbacks completed on the
            # control I/O loop be consumed before another target forward.
            for _ in range(self._poll_turns):
                self._loop.call_soon(advance)
                self._loop.call_soon(self._loop.stop)
                self._loop.run_forever()
                if failures:
                    break
        finally:
            self._pumping = False
        if failures:
            raise failures[0]
        return self.snapshot()

    def begin_shutdown(self):
        self._owner()
        self._closing = True
        failures = []
        for record in self._records.values():
            try:
                self._stop(record, "CUDA refresh driver shutting down")
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise failures[0]

    def close_loop(self):
        self._owner()
        if not self._closing or self._records:
            raise LifecycleError(
                "all CUDA controller owners must drain before loop close"
            )
        if self._owns_loop and not self._loop.is_closed():
            if asyncio.all_tasks(self._loop):
                raise LifecycleError("pending owner-loop tasks prevent close")
            self._loop.close()

    def snapshot(self):
        self._owner()
        return {
            "closing": self._closing,
            "source_quarantine": self._source_quarantine,
            "drained": self._closing and not self._records,
            "requests": {
                key: {
                    "committed_tokens": len(r.outputs) - 1,
                    "refresh_pending": r.refresh is not None,
                    "ready": r.ready,
                    "stopping": r.stopping,
                    "quarantined": r.quarantined,
                    "provisional": r.provisional,
                    "owns_provisional_source": r.provisional_source is not None,
                    "owns_full_receiver": r.full_session is not None,
                    "retirement_state": None
                    if r.retirement is None
                    else r.retirement.state,
                    "error": None if r.error is None else str(r.error)[:512],
                }
                for key, r in self._records.items()
            },
        }
