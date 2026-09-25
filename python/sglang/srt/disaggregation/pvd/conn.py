"""Scheduler-facing PVD sender/receiver adapters.

The existing PD queues remain responsible for request admission and KV page
allocation.  These adapters replace only their point-to-point handshake and
data movement when ``disaggregation_topology == 'pvd'``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import math
import os
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import torch.distributed
from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferMetric
from sglang.srt.disaggregation.pvd.bootstrap import BootstrapGate, BootstrapState
from sglang.srt.disaggregation.pvd.client import PVDCoordinatorClient
from sglang.srt.disaggregation.pvd.decode_refresh import (
    PVDDecodeRefresher,
    PVDDecodeSession,
)
from sglang.srt.disaggregation.pvd.kv_packer import (
    PVD_TENSOR_LAYOUT,
    describe_kv_layout,
    pack_full_prompt_kv,
    pack_full_prompt_kv_head_shard,
)
from sglang.srt.disaggregation.pvd.protocol import (
    FirstTokenMetadata,
    KVEntryKey,
    KVEntryManifest,
    KVLayoutSignature,
    KVShardManifest,
)
from sglang.srt.disaggregation.pvd.runtime import (
    PVDDecodeRuntime,
    PVDEntryLease,
    PVDPrefillRuntime,
)
from sglang.srt.disaggregation.pvd.sharding import validate_compute_layout
from sglang.srt.disaggregation.pvd.transfer_progress import (
    PVD_TRANSFER_CAPABILITY,
    TransferProgress,
)
from sglang.srt.disaggregation.pvd.upload_manager import PVDUploadManager
from sglang.srt.disaggregation.pvd.worker_epoch import worker_epoch


class PVDConnectionError(RuntimeError):
    pass


@dataclass(frozen=True, eq=False)
class PVDSelectedRouteBinding:
    """A selected V reply bound to the exact D request and Gateway choice.

    This is local control-plane provenance, not a transport or storage lease.
    It must not be reconstructed from a V JSON reply by a caller.
    """

    manager: object
    req: object
    rid: str
    key: KVEntryKey
    group_id: str
    delivery_id: str
    selected: object


def _build_sparse_receive_engine(server_args, shared_engine, existing_adapter, budget):
    """Opt-in D TP1 HCA sessions; existing full-Prompt runtime is unchanged."""
    rails = getattr(server_args, "pvd_d_receive_rails", None)
    if not rails:
        return existing_adapter
    from sglang.srt.disaggregation.pvd.multi_rail_receive import (
        create_native_receive_group,
    )

    return create_native_receive_group(
        hostname=shared_engine.hostname,
        gpu_id=shared_engine.gpu_id,
        rails=tuple(rails.split(",")),
        transfer_budget=budget,
        existing_adapter=existing_adapter,
    )


def _build_sparse_receive_registry(server_args, engine, budget, *, epoch, device):
    """Create the owner-thread CUDA destination registry only when opted in."""
    if not getattr(server_args, "pvd_d_receive_rails", None):
        return None
    from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import (
        CUDASparseReceiveRegistry,
    )

    return CUDASparseReceiveRegistry(
        engine,
        budget,
        receiver_epoch=epoch,
        device=device,
    )


class _AsyncControlLoop:
    """One daemon asyncio loop per scheduler process for aiohttp operations."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run, name="pvd-control-plane", daemon=True
        )
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coroutine) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)


class _RouteLookupCompletion(concurrent.futures.Future):
    """Expose coroutine termination, not cancellation of its proxy Future."""

    def cancel(self) -> bool:
        # Cancelling run_coroutine_threadsafe's proxy marks it done before the
        # underlying coroutine has necessarily unwound. Route capacity and rid
        # reuse must instead wait for the coroutine's own final publication.
        return False


class PVDKVManager:
    """Minimal shared context used by both scheduler-facing PVD adapters."""

    def __init__(
        self,
        *,
        scheduler,
        kv_pool,
        metadata_buffers,
        tp_rank: int,
        tp_size: int,
        gloo_group,
    ) -> None:
        from sglang.srt.disaggregation.pvd.mooncake_engine import (
            MooncakePVDTransferEngine,
        )
        from sglang.srt.distributed.parallel_state import get_mooncake_transfer_engine

        if tp_size not in (1, 2, 4):
            raise PVDConnectionError("PVD 2.0 currently supports compute TP 1, 2 or 4")
        if scheduler.ps.pp_size != 1:
            raise PVDConnectionError("PVD does not support pipeline parallelism")
        if scheduler.tp_worker.is_hybrid_swa:
            raise PVDConnectionError("PVD does not support hybrid/SWA KV pools")
        if hasattr(kv_pool, "get_state_buf_infos"):
            state_ptrs, _, _ = kv_pool.get_state_buf_infos()
            if state_ptrs:
                raise PVDConnectionError(
                    "PVD does not support KV pools with SWA/DSA/Mamba state buffers"
                )
        req_to_token_pool = getattr(scheduler, "req_to_token_pool", None)
        if req_to_token_pool is not None and hasattr(
            req_to_token_pool, "get_state_buf_infos"
        ):
            state_ptrs, _, _ = req_to_token_pool.get_state_buf_infos()
            if state_ptrs:
                raise PVDConnectionError(
                    "PVD does not support request-scoped Mamba state buffers"
                )

        shared_engine = get_mooncake_transfer_engine()
        if shared_engine is None:
            raise PVDConnectionError("PVD requires the shared Mooncake transfer engine")

        self.scheduler = scheduler
        self.kv_pool = kv_pool
        self.metadata_buffers = metadata_buffers
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.gloo_group = gloo_group
        self.page_size = kv_pool.page_size
        self.rails = [
            item.strip() for item in scheduler.server_args.pvd_rank_rails.split(",")
        ]
        if len(self.rails) != self.tp_size:
            raise PVDConnectionError(
                f"PVD requires one rail per compute rank; got {len(self.rails)} "
                f"rails for TP={self.tp_size}"
            )
        self.rail = self.rails[tp_rank]
        self.model_instance_id = scheduler.server_args.pvd_model_instance_id
        self.control = _AsyncControlLoop()
        coordinator_map = getattr(
            scheduler.server_args, "pvd_vector_coordinator_map", None
        )
        if not coordinator_map and scheduler.server_args.pvd_vector_coordinator_url:
            coordinator_map = {
                "default": scheduler.server_args.pvd_vector_coordinator_url.rstrip("/")
            }
        if not coordinator_map:
            raise PVDConnectionError("PVD has no trusted vector coordinator groups")
        self.clients = {
            group_id: PVDCoordinatorClient(url, timeout_seconds=300.0)
            for group_id, url in coordinator_map.items()
        }
        # Explicit budget: admission is decided before any staging tensor is
        # allocated, and the same limits are shared by every adapter hanging
        # off this worker's Mooncake engine.
        budget_bytes = scheduler.server_args.pvd_transfer_staging_budget_bytes
        max_inflight = scheduler.server_args.pvd_transfer_max_inflight
        if budget_bytes is None or max_inflight is None:
            raise PVDConnectionError(
                "PVD requires --pvd-transfer-staging-budget-bytes and "
                "--pvd-transfer-max-inflight"
            )
        # ModelRunner's preflight already installed the worker-level manager.
        # Reuse its accounting object, not another budget with equal limits.
        self.transfer_engine = MooncakePVDTransferEngine.from_existing(
            shared_engine, rail=self.rail
        )
        self.transfer_budget = self.transfer_engine.lifecycle_manager.budget
        limits = self.transfer_budget.snapshot()
        if (limits["staging_bytes"], limits["max_inflight"]) != (
            int(budget_bytes),
            int(max_inflight),
        ):
            raise PVDConnectionError(
                "PVD worker budget does not match this rank's configuration: "
                f"engine={limits['staging_bytes']}/{limits['max_inflight']}, "
                f"configured={budget_bytes}/{max_inflight}"
            )
        self.sparse_receive_engine = _build_sparse_receive_engine(
            scheduler.server_args,
            shared_engine,
            self.transfer_engine,
            self.transfer_budget,
        )
        # One incarnation per worker process, shared by every group's runtime.
        # A P compute rank with TP1 uploads to both V storage shards under this
        # epoch; the two writes stay distinct through their write identities.
        self.worker_epoch = worker_epoch()
        self.sparse_receive_registry = _build_sparse_receive_registry(
            scheduler.server_args,
            self.sparse_receive_engine,
            self.transfer_budget,
            epoch=self.worker_epoch,
            device=f"cuda:{shared_engine.gpu_id}",
        )
        self.capabilities = [PVD_TRANSFER_CAPABILITY]
        # Upload records outlive their senders, so the manager is owned here.
        self.upload_manager = PVDUploadManager()
        self._upload_progress_future: Optional[concurrent.futures.Future] = None
        self.prefill_runtimes = {
            group_id: PVDPrefillRuntime(
                model_instance_id=self.model_instance_id,
                coordinator=client,
                transfer_engine=self.transfer_engine,
                upload_manager=self.upload_manager,
                worker_epoch=self.worker_epoch,
            )
            for group_id, client in self.clients.items()
        }
        self.decode_runtimes = {
            group_id: PVDDecodeRuntime(
                coordinator=client,
                transfer_engine=self.transfer_engine,
            )
            for group_id, client in self.clients.items()
        }
        self.kv_args = SimpleNamespace(state_types=[])
        self.is_dummy_cp_rank = False
        self._layout_description = describe_kv_layout(kv_pool)
        self.decode_sessions = {}
        # Sessions whose receive buffer could not be fenced yet. They are owned
        # by this manager, not by any request, so an aborted or finished
        # request can never take its unfenced destination away with it.
        self.pending_decode_closes = []
        self.decode_refresher = PVDDecodeRefresher(self)
        # Opt-in: pull a newcomer's initial Prompt KV when the scheduler puts
        # it into the final waiting queue, rather than on its first in-batch
        # refresh. Gates live here, not on the request, so an aborted request
        # cannot take its pending identity away with it.
        self.waiting_queue_bootstrap = bool(
            getattr(scheduler.server_args, "pvd_waiting_queue_bootstrap", False)
        )
        self.full_kv_fanin_max_slices = getattr(
            scheduler.server_args, "pvd_full_kv_fanin_max_slices", None
        )
        self.full_kv_fanin_rank_packed = bool(
            getattr(scheduler.server_args, "pvd_full_kv_fanin_rank_packed", False)
        )
        self.bootstrap_gates: Dict[Any, BootstrapGate] = {}
        # One bounded driver per worker. It never creates a task per failed
        # request; it advances the owners that already hold the resources.
        self.transfer_progress = TransferProgress(
            upload_manager=self.upload_manager,
            decode_owner=self,
            decode_refresher=self.decode_refresher,
        )
        if (
            scheduler.server_args.disaggregation_mode == "decode"
            and scheduler.enable_overlap
        ):
            raise PVDConnectionError(
                "PVD 3.0 KV refresh requires overlap scheduling disabled"
            )

    # -- waiting-queue bootstrap -------------------------------------------

    def open_bootstrap_gate(self, req) -> Optional[BootstrapGate]:
        """Create this request's single initial-pull gate, if enabled."""
        if not self.waiting_queue_bootstrap:
            return None
        key = self.key_for(req)
        if key in self.bootstrap_gates:
            raise PVDConnectionError("duplicate PVD bootstrap gate for one request")
        gate = BootstrapGate(
            delivery_id=f"{key.transfer_id}:bootstrap",
            receiver_epoch=self.worker_epoch,
            prompt_tokens=len(req.origin_input_ids),
        )
        self.bootstrap_gates[key] = gate
        return gate

    def bootstrap_gate_for(self, req) -> Optional[BootstrapGate]:
        if not self.waiting_queue_bootstrap:
            return None
        return self.bootstrap_gates.get(self.key_for(req))

    def bootstrap_runnable(self, req) -> bool:
        """False means: leave this request in the waiting queue, unadmitted.

        A request with no gate is unaffected, so the legacy path and ordinary
        PD are never gated by this.
        """
        gate = self.bootstrap_gate_for(req)
        return True if gate is None else gate.is_runnable

    def close_bootstrap_gate(self, req) -> None:
        """Refuse further bootstrap progress. Releases and fences nothing."""
        if not self.waiting_queue_bootstrap:
            return
        gate = self.bootstrap_gates.pop(self.key_for(req), None)
        if gate is not None:
            gate.close()

    def bootstrap_staging_bytes(self, req) -> int:
        """Staging bytes this rank's initial pull will reserve for one request."""
        needed = self.local_shard_manifest(len(req.origin_input_ids)).expected_bytes
        # Rank-packed fan-in retains both the registered wire destination and
        # an unregistered canonical reorder buffer until request close.
        return needed * (2 if getattr(self, "full_kv_fanin_rank_packed", False) else 1)

    def _bootstrap_staging_headroom(self) -> Optional[int]:
        """Unreserved staging bytes, or None when no budget is enforced.

        Advisory only. ``PVDDecodeSession.prepare`` makes the real reservation,
        and it can still refuse; this check exists so the common case backs off
        quietly instead of taking the whole pull batch down with it.
        """
        budget = getattr(self, "transfer_budget", None)
        if budget is None:
            return None
        limits = budget.snapshot()
        return limits["staging_bytes"] - limits["used_staging_bytes"]

    def enter_waiting_queue(self, reqs) -> List[tuple]:
        """Progress initial pulls and retry the entire final waiting queue.

        HTTP delivery/poll/ACK run asynchronously. Preparation, TP agreement
        and installation run on the scheduler thread. One bounded-by-budget
        wave is active at a time; it never joins the running-batch barrier.
        """
        if not self.waiting_queue_bootstrap:
            return []
        failures = []

        def finish(completed, errors):
            failed = {id(req): error for req, error in errors}
            for req in completed:
                # An abort may already have removed the request and released
                # its final pages. Never resurrect it or release them twice.
                gate = self.bootstrap_gate_for(req)
                if gate is None or not any(req is queued for queued in reqs):
                    continue
                if id(req) in failed:
                    failures.append((req, failed[id(req)]))
                    continue
                try:
                    gate.mark_received(gate.ticket)
                    gate.mark_installed(gate.ticket)
                    gate.handoff()
                except Exception as exc:
                    failures.append((req, f"PVD bootstrap install was refused: {exc}"))

        finish(*self.decode_refresher.poll_bootstrap())
        if self.decode_refresher.bootstrap_pending:
            return failures

        headroom = self._bootstrap_staging_headroom()
        budget = getattr(self, "transfer_budget", None)
        capacity = budget.snapshot()["staging_bytes"] if budget is not None else None
        candidates, local_errors = {}, {}
        by_key = {}
        for req in reqs:
            gate = self.bootstrap_gate_for(req)
            if gate is None:
                continue
            key = self.key_for(req).transfer_id
            by_key[key] = req
            if gate.state is BootstrapState.CLOSED:
                local_errors[key] = "PVD bootstrap gate is closed"
                continue
            if gate.state is not BootstrapState.QUEUED:
                continue
            try:
                gate.enter_waiting_queue()
                needed = self.bootstrap_staging_bytes(req)
                if capacity is not None and needed > capacity:
                    local_errors[key] = (
                        "PVD initial Prompt KV exceeds Decode staging capacity: "
                        f"required={needed} configured={capacity} bytes"
                    )
                    continue
                if gate.can_request():
                    candidates[key] = needed
            except Exception as exc:
                local_errors[key] = str(exc)

        # Budget and source readiness can differ between TP ranks. Agree on
        # one ordered set BEFORE entering any per-wave collective; a local
        # early return here would deadlock another rank preparing a pull.
        ranks = self.gather_rank_objects(
            {
                "rank": self.tp_rank,
                "candidates": candidates,
                "headroom": headroom,
                "errors": local_errors,
            }
        )
        errors = {key: error for rank in ranks for key, error in rank["errors"].items()}
        failures.extend(
            (by_key[key], error) for key, error in errors.items() if key in by_key
        )
        remaining = [rank["headroom"] for rank in ranks]
        pulling = []
        for key in ranks[0]["candidates"]:
            if key in errors or any(key not in rank["candidates"] for rank in ranks):
                continue
            needs = [rank["candidates"][key] for rank in ranks]
            if any(
                free is not None and need > free for need, free in zip(needs, remaining)
            ):
                continue
            for i, need in enumerate(needs):
                if remaining[i] is not None:
                    remaining[i] -= need
            req = by_key[key]
            self.bootstrap_gate_for(req).begin()
            pulling.append(req)
        if pulling:
            finish(*self.decode_refresher.start_bootstrap(pulling))
        return failures

    def retain_pending_close(self, session) -> None:
        """Keep an unfenced Decode receive buffer alive under manager ownership.

        Called when a close could not prove that remote writes stopped. The
        session keeps its registration, its pin and its saved identities; only
        a later successful fence releases them.
        """
        if session not in self.pending_decode_closes:
            self.pending_decode_closes.append(session)

    def progress_decode_closes(self) -> Optional[concurrent.futures.Future]:
        """Trigger one bounded step over retained Decode closes."""
        if not self.pending_decode_closes:
            return None
        return self.control.submit(self.transfer_progress.tick_control())

    def transfer_health(self) -> Dict[str, Any]:
        """Worker-level readiness, capability and transfer accounting."""
        snapshot = self.transfer_progress.snapshot()
        snapshot["worker_epoch"] = self.worker_epoch
        snapshot["capabilities"] = list(self.capabilities)
        snapshot["budget"] = self.transfer_budget.snapshot()
        snapshot["transport"] = self.transfer_engine.health()
        if self.sparse_receive_engine is not self.transfer_engine:
            snapshot["sparse_receive_transport"] = self.sparse_receive_engine.health()
        if self.sparse_receive_registry is not None:
            snapshot["sparse_receive_destinations"] = len(
                self.sparse_receive_registry.snapshot()
            )
        return snapshot

    def progress_uploads(self) -> Optional[concurrent.futures.Future]:
        """Trigger one bounded upload progress step.

        This is a convenience trigger, not the owner of the work: records live
        in ``self.upload_manager`` and keep their pins whether or not anyone
        calls this. Task 7 adds the bounded autonomous driver; until then a
        worker that stops triggering retains resources instead of releasing
        them, and a test can call ``upload_manager.progress()`` directly.
        """
        if self.transfer_progress.outstanding() == 0:
            return None
        # Native observation is cheap and synchronous, so do it on the caller's
        # thread: a handle reaches its terminal state even while the control
        # plane is backing off.
        self.transfer_progress.tick()
        future = self._upload_progress_future
        if future is not None and not future.done():
            return future
        self._upload_progress_future = self.control.submit(
            self.transfer_progress.tick_control()
        )
        return self._upload_progress_future

    def abandon_uploads(self, key: KVEntryKey, reason: str) -> None:
        """Stop future submissions for an entry without releasing anything."""
        self.upload_manager.abandon_key(key, reason)
        self.progress_uploads()

    def key_for(self, req) -> KVEntryKey:
        transfer_id = getattr(req, "pvd_transfer_id", None)
        if not transfer_id:
            raise PVDConnectionError(
                "PVD request is missing pvd_transfer_id; send it through the PVD Gateway"
            )
        # P and D may independently assign SGLang rid values. The Gateway's
        # transfer id is therefore also the cross-role request identity.
        return KVEntryKey(
            model_instance_id=self.model_instance_id,
            req_id=transfer_id,
            transfer_id=transfer_id,
        )

    def vector_group_for(self, req) -> str:
        group_id = getattr(req, "pvd_vector_group_id", None)
        if not group_id:
            raise PVDConnectionError(
                "PVD request is missing pvd_vector_group_id; send it through "
                "the PVD Gateway"
            )
        if group_id not in self.clients:
            raise PVDConnectionError(
                f"PVD request selected unknown vector group {group_id!r}; "
                f"trusted groups are {sorted(self.clients)}"
            )
        return group_id

    def start_selected_cuda_routes(self, req) -> concurrent.futures.Future:
        """Resolve Gateway-selected V routes without blocking the Scheduler.

        The selected Entry and group are captured before crossing threads.
        This does not allocate a D destination or admit a prediction request.
        """
        from sglang.srt.disaggregation.pvd.client import PVDSelectedShardRoutes
        from sglang.srt.disaggregation.pvd.multi_rail_receive import (
            RailMappedReceiveEngine,
        )

        if self.sparse_receive_registry is None:
            raise PVDConnectionError("D sparse receive was not initialized")
        key = self.key_for(req)
        group_id = self.vector_group_for(req)
        client = self.clients[group_id]
        rid = req.rid
        delivery_id = req.pvd_delivery_id
        engine = self.sparse_receive_engine

        completion = _RouteLookupCompletion()

        async def discover():
            try:
                selected = await client.selected_shard_routes(key)
                if (
                    not isinstance(selected, PVDSelectedShardRoutes)
                    or selected.manifest.key != key
                    or len(selected.shards) != 2
                    or tuple(route.rank for route in selected.shards) != (0, 1)
                ):
                    raise PVDConnectionError("selected V route identity changed")
                if isinstance(engine, RailMappedReceiveEngine):
                    missing = [
                        route.rail
                        for route in selected.shards
                        if not engine.supports_rail(route.rail)
                    ]
                else:
                    missing = [
                        route.rail
                        for route in selected.shards
                        if route.rail != engine.rail
                    ]
                if missing:
                    raise PVDConnectionError(
                        f"D has no preflighted receive adapter for V rails {missing}"
                    )
                if engine.health().get("healthy") is not True:
                    raise PVDConnectionError("D sparse receive transport is unhealthy")
                result = PVDSelectedRouteBinding(
                    self, req, rid, key, group_id, delivery_id, selected
                )
            except BaseException as exc:
                completion.set_exception(exc)
                raise
            else:
                completion.set_result(result)
                return result

        coroutine = discover()
        try:
            self.control.submit(coroutine)
        except BaseException:
            coroutine.close()
            raise
        return completion

    def assemble_selected_cuda_request(
        self,
        req,
        binding,
        *,
        group,
        pipeline,
        head_mapping,
        vector_space,
        metric,
        top_k,
        max_union_tokens,
        max_head_dim,
        copy_budget,
        aggregate_budget,
        poll_interval_seconds,
        initial_import_pending=False,
    ):
        """Bind a request factory to this worker's owned D receive resources."""
        from sglang.srt.disaggregation.pvd.client import PVDSelectedShardRoutes
        from sglang.srt.disaggregation.pvd.cuda_routed_request import (
            assemble_routed_cuda_request,
        )

        registry = self.sparse_receive_registry
        if (
            registry is None
            or registry.engine is not self.sparse_receive_engine
            or self.tp_size != 1
            or self.tp_rank != 0
        ):
            raise PVDConnectionError("D TP1 sparse receive is not initialized")
        registry._owner()
        if self.sparse_receive_engine.health().get("healthy") is not True:
            raise PVDConnectionError("D sparse receive transport is unhealthy")
        if (
            not isinstance(binding, PVDSelectedRouteBinding)
            or binding.manager is not self
            or binding.req is not req
            or binding.rid != req.rid
            or binding.key != self.key_for(req)
            or binding.group_id != self.vector_group_for(req)
            or binding.delivery_id != req.pvd_delivery_id
            or not isinstance(binding.selected, PVDSelectedShardRoutes)
            or binding.selected.manifest.key != binding.key
        ):
            raise PVDConnectionError(
                "selected V routes differ from this D request or Gateway group"
            )
        selected = binding.selected
        compute_health = self.transfer_engine.health()
        if compute_health.get("healthy") is not True:
            raise PVDConnectionError("D compute transport is unhealthy")
        endpoint = compute_health.get("session_id")
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise PVDConnectionError("D compute Mooncake session is unavailable")
        search_io_loop = (
            None
            if os.environ.get("PVD_SEARCH_BACKGROUND_IO") == "0"
            else self.control.loop
        )
        return assemble_routed_cuda_request(
            selected,
            request_id=req.rid,
            compute_layout=self.layout(),
            compute_rank=self.tp_rank,
            group=group,
            registry=registry,
            pipeline=pipeline,
            head_mapping=head_mapping,
            vector_space=vector_space,
            metric=metric,
            top_k=top_k,
            max_union_tokens=max_union_tokens,
            max_head_dim=max_head_dim,
            copy_budget=copy_budget,
            aggregate_budget=aggregate_budget,
            d_endpoint=endpoint,
            d_rail=self.rail,
            d_rails={route.rank: route.rail for route in selected.shards},
            poll_interval_seconds=poll_interval_seconds,
            initial_import_pending=initial_import_pending,
            search_io_loop=search_io_loop,
        )

    def client_for(self, req) -> PVDCoordinatorClient:
        return self.clients[self.vector_group_for(req)]

    def prefill_runtime_for(self, req) -> PVDPrefillRuntime:
        return self.prefill_runtimes[self.vector_group_for(req)]

    def decode_runtime_for(self, req) -> PVDDecodeRuntime:
        return self.decode_runtimes[self.vector_group_for(req)]

    def _total_kv_heads(self) -> int:
        model_config = self.scheduler.model_config
        getter = getattr(model_config, "get_total_num_kv_heads", None)
        if getter is not None:
            return int(getter())
        return (
            int(self._layout_description["component_token_shapes"][0][0]) * self.tp_size
        )

    def layout(self) -> KVLayoutSignature:
        """Return this P/D compute rank group's layout."""
        model_config = self.scheduler.model_config
        components = self._layout_description
        kv_heads = getattr(self.kv_pool, "head_num", None)
        if kv_heads is None:
            token_shape = components["component_token_shapes"][0]
            kv_heads = token_shape[-2] if len(token_shape) >= 2 else 1
        return KVLayoutSignature(
            model_id=str(model_config.model_path),
            model_revision=str(self.scheduler.server_args.revision or ""),
            kv_dtype=components["component_dtypes"][0],
            page_size=self.page_size,
            num_layers=int(model_config.num_hidden_layers),
            total_kv_heads=self._total_kv_heads(),
            kv_heads_per_rank=int(kv_heads),
            head_dim=int(model_config.head_dim),
            tp_size=self.tp_size,
            pp_size=1,
            tensor_layout=PVD_TENSOR_LAYOUT,
            extra=components,
        )

    def storage_layout(self) -> KVLayoutSignature:
        """Return the stable two-shard V layout, independent of P/D compute TP."""
        compute = self.layout()
        if compute.total_kv_heads % 2:
            raise PVDConnectionError(
                "PVD 2.0 V TP2 storage requires an even number of total KV heads"
            )
        heads = compute.total_kv_heads // 2
        extra = copy.deepcopy(dict(compute.extra))
        source_heads = compute.kv_heads_per_rank
        token_shapes = extra.get("component_token_shapes", [])
        bytes_per_token = extra.get("component_bytes_per_token", [])
        if source_heads <= 0 or any(
            int(value) % source_heads for value in bytes_per_token
        ):
            raise PVDConnectionError("KV components cannot be split by head")
        adjusted_shapes = []
        for shape in token_shapes:
            shape = list(shape)
            if not shape or int(shape[0]) != source_heads:
                raise PVDConnectionError(
                    "PVD heterogeneous TP requires KV head to be tensor dimension 1"
                )
            shape[0] = heads
            adjusted_shapes.append(shape)
        extra["component_token_shapes"] = adjusted_shapes
        extra["component_bytes_per_token"] = [
            int(value) // source_heads * heads for value in bytes_per_token
        ]
        return KVLayoutSignature(
            model_id=compute.model_id,
            model_revision=compute.model_revision,
            kv_dtype=compute.kv_dtype,
            page_size=compute.page_size,
            num_layers=compute.num_layers,
            total_kv_heads=compute.total_kv_heads,
            kv_heads_per_rank=heads,
            head_dim=compute.head_dim,
            tp_size=2,
            pp_size=compute.pp_size,
            tensor_layout=compute.tensor_layout,
            extra=extra,
        )

    def local_shard_manifest(self, prompt_tokens: int) -> KVShardManifest:
        page_count = math.ceil(prompt_tokens / self.page_size)
        bytes_per_token = sum(self._layout_description["component_bytes_per_token"])
        start_layer = int(getattr(self.kv_pool, "start_layer", 0))
        end_layer_value = getattr(self.kv_pool, "end_layer", None)
        end_layer = int(
            end_layer_value
            if end_layer_value is not None
            else start_layer + self.scheduler.model_config.num_hidden_layers
        )
        return KVShardManifest(
            rank=self.tp_rank,
            rail=self.rail,
            expected_bytes=page_count * self.page_size * bytes_per_token,
            page_count=page_count,
            last_page_valid_tokens=prompt_tokens % self.page_size or self.page_size,
            layer_start=start_layer,
            layer_end=end_layer,
        )

    def storage_shard_manifest(
        self, prompt_tokens: int, rank: int, layout: KVLayoutSignature
    ) -> KVShardManifest:
        page_count = math.ceil(prompt_tokens / self.page_size)
        rail = self.rails[rank] if len(self.rails) > rank else self.rails[0]
        bytes_per_token = sum(layout.extra["component_bytes_per_token"])
        start_layer = int(getattr(self.kv_pool, "start_layer", 0))
        end_layer_value = getattr(self.kv_pool, "end_layer", None)
        end_layer = int(
            end_layer_value
            if end_layer_value is not None
            else start_layer + self.scheduler.model_config.num_hidden_layers
        )
        return KVShardManifest(
            rank=rank,
            rail=rail,
            expected_bytes=page_count * self.page_size * bytes_per_token,
            page_count=page_count,
            last_page_valid_tokens=prompt_tokens % self.page_size or self.page_size,
            layer_start=start_layer,
            layer_end=end_layer,
        )

    def gather_rank_objects(self, local: Dict[str, Any]) -> List[Dict[str, Any]]:
        gathered: List[Optional[Dict[str, Any]]] = [None] * self.tp_size
        torch.distributed.all_gather_object(gathered, local, group=self.gloo_group)
        result = [item for item in gathered if item is not None]
        expected = list(range(self.tp_size))
        if sorted(int(item["rank"]) for item in result) != expected:
            raise PVDConnectionError(
                f"PVD rank object exchange did not produce ranks {expected}"
            )
        return result

    async def wait_for_stored_entry(
        self,
        key: KVEntryKey,
        runtime: PVDDecodeRuntime,
        *,
        timeout_seconds: float = 300.0,
    ) -> Dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            match = await runtime.select_entry(key)
            state = match.get("state")
            if match.get("found") and state == "stored":
                return match["entry"]
            if state in ("failed", "cancelled", "released"):
                raise PVDConnectionError(
                    f"PVD Entry became terminal before delivery: {match}"
                )
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"timed out waiting for PVD Entry {key}")
            await asyncio.sleep(0.01)


class PVDKVSender:
    def __init__(self, *, mgr: PVDKVManager, req) -> None:
        self.kv_mgr = mgr
        self.req = req
        self.key = mgr.key_for(req)
        self.client = mgr.client_for(req)
        self.prefill_runtime = mgr.prefill_runtime_for(req)
        self.conclude_state = None
        self._error: Optional[BaseException] = None
        self._lease: Optional[PVDEntryLease] = None
        self._publish_future: Optional[concurrent.futures.Future] = None
        self._started_at: Optional[float] = None
        self._metric = KVTransferMetric()

        if mgr.tp_size not in (1, 2):
            raise PVDConnectionError("PVD Prefill currently supports TP1 or TP2")
        storage_layout = mgr.storage_layout()
        if mgr.tp_size == 1:
            shards = {
                rank: mgr.storage_shard_manifest(
                    len(req.origin_input_ids), rank, storage_layout
                )
                for rank in range(2)
            }
            uploader_epochs = {rank: mgr.worker_epoch for rank in shards}
            owned_shard_ranks = set(shards)
        else:
            local_shard = mgr.local_shard_manifest(len(req.origin_input_ids))
            gathered = mgr.gather_rank_objects(
                {
                    "rank": mgr.tp_rank,
                    "shard": local_shard.to_dict(),
                    "epoch": mgr.worker_epoch,
                }
            )
            shards = {
                int(item["rank"]): KVShardManifest.from_dict(item["shard"])
                for item in gathered
            }
            uploader_epochs = {int(item["rank"]): item["epoch"] for item in gathered}
            owned_shard_ranks = {mgr.tp_rank}
        self._create_future = mgr.control.submit(
            self.prefill_runtime.create_entry(
                req_id=self.key.req_id,
                transfer_id=self.key.transfer_id,
                layout=storage_layout,
                prompt_token_count=len(req.origin_input_ids),
                shards=shards,
                uploader_epochs=uploader_epochs,
                owned_shard_ranks=owned_shard_ranks,
            )
        )
        self._expected_pages = next(iter(shards.values())).page_count

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        if num_kv_indices != self._expected_pages:
            raise PVDConnectionError(
                f"PVD Prefill must publish every prompt page: "
                f"expected {self._expected_pages}, got {num_kv_indices}"
            )

    def pop_decode_prefix_len(self) -> int:
        return 0

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        # Entry is immutable and always contains the complete prompt KV.
        return last_chunk

    def _first_token(self) -> Optional[FirstTokenMetadata]:
        if self.kv_mgr.tp_rank != 0:
            return None
        req = self.req
        logprob = req.logprob.output_token_logprobs_val
        logprob_idx = req.logprob.output_token_logprobs_idx
        top_values = req.logprob.output_top_logprobs_val
        top_indices = req.logprob.output_top_logprobs_idx
        return FirstTokenMetadata(
            output_token_id=int(req.output_ids[0]),
            cached_tokens=int(req.cached_tokens),
            cached_tokens_device=int(req.cached_tokens_device),
            cached_tokens_host=int(req.cached_tokens_host),
            cached_tokens_storage=int(req.cached_tokens_storage),
            output_token_logprob=float(logprob[0]) if logprob else None,
            output_token_logprob_index=int(logprob_idx[0]) if logprob_idx else None,
            output_top_logprobs_values=list(top_values[0]) if top_values else None,
            output_top_logprobs_indices=list(top_indices[0]) if top_indices else None,
        )

    def send(self, kv_indices, state_indices: Optional[List] = None):
        try:
            self._send(kv_indices, state_indices)
        except BaseException as exc:
            self._error = exc
            self.conclude_state = KVPoll.Failed
            self.kv_mgr.control.submit(self.client.cancel_entry(self.key, str(exc)))

    def _send(self, kv_indices, state_indices: Optional[List] = None):
        if state_indices and any(item is not None for item in state_indices):
            raise PVDConnectionError("PVD does not support auxiliary KV state")
        if self._lease is None:
            self._lease = self._create_future.result()
        self._started_at = time.monotonic()
        if self.kv_mgr.tp_size == 1:
            storage_heads = self._lease.manifest.layout.kv_heads_per_rank
            packed_shards = [
                pack_full_prompt_kv_head_shard(
                    self.kv_mgr.kv_pool,
                    kv_indices,
                    page_size=self.kv_mgr.page_size,
                    head_start=rank * storage_heads,
                    head_count=storage_heads,
                )
                for rank in range(2)
            ]
            self._metric.transfer_total_bytes = sum(
                packed.expected_bytes for packed in packed_shards
            )

            async def publish_all():
                return await asyncio.gather(
                    *(
                        self.prefill_runtime.publish_tensor_shard(
                            lease=self._lease,
                            rank=rank,
                            tensor=packed.tensor,
                            endpoint="pvd-prefill",
                            rail=self.kv_mgr.rail,
                            first_token=self._first_token() if rank == 0 else None,
                        )
                        for rank, packed in enumerate(packed_shards)
                    )
                )

            self._publish_future = self.kv_mgr.control.submit(publish_all())
        else:
            packed = pack_full_prompt_kv(
                self.kv_mgr.kv_pool, kv_indices, page_size=self.kv_mgr.page_size
            )
            if packed.page_count != self._expected_pages:
                raise PVDConnectionError(
                    f"PVD packed {packed.page_count} pages, expected {self._expected_pages}"
                )
            self._metric.transfer_total_bytes = packed.expected_bytes
            self._publish_future = self.kv_mgr.control.submit(
                self.prefill_runtime.publish_tensor_shard(
                    lease=self._lease,
                    rank=self.kv_mgr.tp_rank,
                    tensor=packed.tensor,
                    endpoint="pvd-prefill",
                    rail=self.kv_mgr.rail,
                    first_token=self._first_token(),
                )
            )

    def poll(self) -> int:
        # Progress is triggered here for live requests, but the manager owns
        # the records: a sender that is never polled again must not strand an
        # upload. See PVDKVManager.progress_uploads.
        self.kv_mgr.progress_uploads()
        if self.conclude_state is not None:
            return self.conclude_state
        try:
            if self._lease is None:
                if not self._create_future.done():
                    return KVPoll.Bootstrapping
                self._lease = self._create_future.result()
                return KVPoll.WaitingForInput
            if self._publish_future is None:
                return KVPoll.WaitingForInput
            if not self._publish_future.done():
                return KVPoll.Transferring
            self._publish_future.result()
            if self._started_at is not None:
                self._metric.transfer_latency_s = time.monotonic() - self._started_at
            self.conclude_state = KVPoll.Success
        except BaseException as exc:
            self._error = exc
            self.conclude_state = KVPoll.Failed
        return self.conclude_state

    def get_transfer_metric(self) -> KVTransferMetric:
        return self._metric

    def failure_exception(self):
        raise PVDConnectionError(str(self._error or "unknown PVD Prefill failure"))

    def abort(self):
        self.conclude_state = KVPoll.Failed
        # Forbid any further submission under these identities first, then ask
        # V to cancel. V records a close request; it does not release the
        # destination pages, and neither does this call.
        self.kv_mgr.abandon_uploads(self.key, "Prefill request aborted")
        self.kv_mgr.control.submit(
            self.client.cancel_entry(self.key, "Prefill request aborted")
        )

    def clear(self):
        # The scheduler is done with this request object. Any upload that never
        # reached native submission is now provably never going to; one that
        # did keeps its handle, its source pin and its pending terminal report
        # inside the manager until V acknowledges a terminal state.
        self.kv_mgr.abandon_uploads(self.key, "Prefill sender cleared")


class PVDKVReceiver:
    def __init__(self, *, mgr: PVDKVManager, req) -> None:
        self.kv_mgr = mgr
        self.req = req
        self.key = mgr.key_for(req)
        self.client = mgr.client_for(req)
        self.decode_runtime = mgr.decode_runtime_for(req)
        self.delivery_id = getattr(req, "pvd_delivery_id", None)
        if not self.delivery_id:
            raise PVDConnectionError(
                "PVD request is missing pvd_delivery_id; send it through the PVD Gateway"
            )
        self.conclude_state = None
        self.require_staging = False
        self._error: Optional[BaseException] = None
        self._entry_record: Optional[Dict[str, Any]] = None
        self._entry_future: Optional[concurrent.futures.Future] = None
        self._admitted = False
        if getattr(mgr, "full_kv_fanin_max_slices", None) is not None:
            from sglang.srt.disaggregation.pvd.decode_fanin import PVDDecodeFanInSession

            self.session = PVDDecodeFanInSession(mgr, req)
        else:
            self.session = PVDDecodeSession(mgr, req)
        if self.key in mgr.decode_sessions:
            raise PVDConnectionError("duplicate active PVD Decode transfer identity")
        mgr.decode_sessions[self.key] = self.session
        # Opened here so an abort between enqueue and the waiting queue still
        # finds a gate to close. None unless the opt-in flag is set.
        self.bootstrap_gate = (
            mgr.open_bootstrap_gate(req)
            if getattr(mgr, "waiting_queue_bootstrap", False)
            else None
        )

    def init(self, prefill_dp_rank: int):
        self._entry_future = self.kv_mgr.control.submit(
            self.session.initialize(self.decode_runtime)
        )

    def _validate_entry(self, record: Dict[str, Any]) -> KVEntryManifest:
        manifest = KVEntryManifest.from_dict(record["manifest"])
        try:
            validate_compute_layout(manifest.layout, self.kv_mgr.layout())
        except ValueError as exc:
            raise PVDConnectionError(
                f"PVD Entry layout does not match Decode KV layout: {exc}"
            ) from exc
        if manifest.prompt_token_count != len(self.req.origin_input_ids):
            raise PVDConnectionError(
                "PVD Entry prompt length does not match Decode request"
            )
        return manifest

    def _write_first_token_metadata(self, aux_index: int) -> None:
        token = self._entry_record.get("first_token")
        if token is None:
            raise PVDConnectionError("stored PVD Entry has no first-token metadata")
        metadata = FirstTokenMetadata.from_dict(token)
        buffers = self.kv_mgr.metadata_buffers
        buffers.output_ids[aux_index, 0] = metadata.output_token_id
        buffers.cached_tokens[aux_index, 0] = metadata.cached_tokens
        buffers.cached_tokens[aux_index, 1] = metadata.cached_tokens_device
        buffers.cached_tokens[aux_index, 2] = metadata.cached_tokens_host
        buffers.cached_tokens[aux_index, 3] = metadata.cached_tokens_storage
        if metadata.output_token_logprob is not None:
            buffers.output_token_logprobs_val[aux_index, 0] = (
                metadata.output_token_logprob
            )
        if metadata.output_token_logprob_index is not None:
            buffers.output_token_logprobs_idx[aux_index, 0] = (
                metadata.output_token_logprob_index
            )
        if metadata.output_top_logprobs_values:
            values = torch.tensor(
                metadata.output_top_logprobs_values,
                dtype=buffers.output_top_logprobs_val.dtype,
                device=buffers.output_top_logprobs_val.device,
            )
            buffers.output_top_logprobs_val[aux_index, : values.numel()] = values
        if metadata.output_top_logprobs_indices:
            indices = torch.tensor(
                metadata.output_top_logprobs_indices,
                dtype=buffers.output_top_logprobs_idx.dtype,
                device=buffers.output_top_logprobs_idx.device,
            )
            buffers.output_top_logprobs_idx[aux_index, : indices.numel()] = indices
        buffers.bootstrap_room[aux_index, 0] = self.req.bootstrap_room or 0

    def send_metadata(
        self,
        kv_indices,
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
    ):
        try:
            self._send_metadata(
                kv_indices,
                aux_index=aux_index,
                state_indices=state_indices,
                decode_prefix_len=decode_prefix_len,
            )
        except BaseException as exc:
            self._error = exc
            self.session.schedule_close()
            self.conclude_state = KVPoll.Failed

    def _send_metadata(
        self,
        kv_indices,
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
    ):
        if aux_index is None:
            raise PVDConnectionError("PVD Decode requires a metadata buffer index")
        if decode_prefix_len not in (None, 0):
            raise PVDConnectionError("PVD requires Decode radix cache to be disabled")
        if state_indices and any(item is not None for item in state_indices):
            raise PVDConnectionError("PVD does not support auxiliary KV state")
        if self._entry_record is None:
            self._entry_record = self._entry_future.result()
        manifest = self._validate_entry(self._entry_record)
        local_shard = self.kv_mgr.local_shard_manifest(manifest.prompt_token_count)
        if len(kv_indices) != local_shard.page_count:
            raise PVDConnectionError(
                f"Decode allocated {len(kv_indices)} pages, Entry requires "
                f"{local_shard.page_count}"
            )
        self._write_first_token_metadata(aux_index)
        if self.bootstrap_gate is not None:
            # V holds this Entry. The pull still waits for the final waiting
            # queue; observing KV_STORED here only removes one precondition.
            self.bootstrap_gate.mark_source_ready()
        # KV_READY admission into the waiting queue. With the waiting-queue
        # bootstrap enabled the request is still not runnable: no RDMA
        # destination is exposed until it reaches that queue.
        self._admitted = True

    def poll(self) -> int:
        if self.conclude_state is not None:
            return self.conclude_state
        try:
            if self._entry_record is None:
                if self._entry_future is None or not self._entry_future.done():
                    return KVPoll.Bootstrapping
                self._entry_record = self._entry_future.result()
                self._validate_entry(self._entry_record)
                return KVPoll.WaitingForInput
            if not self._admitted:
                return KVPoll.WaitingForInput
            self.conclude_state = KVPoll.Success
        except BaseException as exc:
            self._error = exc
            self.session.schedule_close()
            self.conclude_state = KVPoll.Failed
        return self.conclude_state

    def failure_exception(self):
        raise PVDConnectionError(str(self._error or "unknown PVD Decode failure"))

    def abort(self):
        self.conclude_state = KVPoll.Failed
        self.session.schedule_close()
        if self.bootstrap_gate is not None:
            self.kv_mgr.close_bootstrap_gate(self.req)

    def clear(self):
        if self.conclude_state != KVPoll.Success:
            self.session.schedule_close()
