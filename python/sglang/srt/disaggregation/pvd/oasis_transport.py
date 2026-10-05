"""Per-layer CAGRA selection and native V -> D CPU-cache misses.

The CPU receive registry proves exact native terminal success before reading.
Each worker owns its asyncio loop, CUDA stream and receive records. Opt-in
request clients use the manager's background I/O loop across layer jobs.
Unknown remote/native completion keeps those owners in the request quarantine.
"""

import asyncio
import os
import threading
import time

import torch

from sglang.srt.disaggregation.pvd.control_server import HttpShardClient
from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import CUDASparseReceiveRecord, CUDASparseReceiveRegistry
from sglang.srt.disaggregation.pvd.oasis_pipeline import LayerReply, select_resident
from sglang.srt.disaggregation.pvd.oasis_qwen import PromptBank
from sglang.srt.disaggregation.pvd.oasis_receive_slots import OasisReceiveSlotPool
from sglang.srt.disaggregation.pvd.oasis_ready_cleanup import OwnedReadyCleanup
from sglang.srt.disaggregation.pvd.oasis_pinned_scratch import PinnedScratchPool, scratch_bytes
from sglang.srt.disaggregation.pvd.oasis_gpu_backup import OasisGPUBackupPool
from sglang.srt.disaggregation.pvd.oasis_bank_install import install_batched_bank, install_tensor_bound
from sglang.srt.disaggregation.pvd.oasis_cache_install import (
    cache_install_tensor_bound, install_cpu_payloads, row_cache_profile,
)
from sglang.srt.disaggregation.pvd.prompt_index import SearchRequestIdentity
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED
from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient, SearchScope
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
from sglang.srt.disaggregation.pvd.sparse_receiver import SparseReceiveRecord, SparseReceiveRegistry
from sglang.srt.disaggregation.pvd.sparse_token_runs import consecutive_token_runs


class _HeadCPUCache:
    """Views into one charged contiguous Prompt allocation, no per-row owners."""
    def __init__(self, rows, valid):
        self.rows, self.valid = rows, valid

    def __contains__(self, token):
        return type(token) is int and 0 <= token < len(self.valid) and bool(self.valid[token])

    def __getitem__(self, token):
        if token not in self:
            raise KeyError("uncached Prompt token")
        return self.rows[token]

    def __setitem__(self, token, value):
        if (type(token) is not int or not 0 <= token < len(self.valid)
                or token in self or value.shape != self.rows.shape[1:]
                or value.device.type != "cpu" or value.dtype != self.rows.dtype):
            raise ValueError("unique bounded CPU Prompt KV row required")
        self.rows[token].copy_(value)
        self.valid[token] = True


class OasisCPUReceiveRecord(SparseReceiveRecord):
    def copy_to_cache(self, cache):
        self._live()
        if (self._lock.locked() or not self._ready or not self._safe
                or self._installed or self._buffer.device.type != "cpu"):
            raise RuntimeError("one exact terminal-success CPU receive required")
        views = self.manifest.payload_views(self._buffer)
        try:
            if self._registry.batched_cache_install:
                profile = install_cpu_payloads(views, cache,
                    capacity=self._registry.cache_capacity, retained=[])
            else:
                for payload in views:
                    head = payload.spec.kv_head
                    for offset, token in enumerate(payload.spec.token_ids):
                        if token in cache[head]:
                            raise RuntimeError("duplicate remote cache row")
                        # The synchronous clone owns its bytes before V is ACKed.
                        cache[head][token] = payload.tensor[:, offset].clone()
                profile = row_cache_profile(views)
            self.profile.update(profile, cache_install_complete=True, cache_d2h_fenced=False)
        finally:
            for payload in views:
                payload.close()
        self._installed = True


class OasisCPUReceiveRegistry(SparseReceiveRegistry):
    def __init__(self, *args, batched_cache_install=False, cache_capacity=2048, **kwargs):
        if type(batched_cache_install) is not bool:
            raise ValueError('explicit boolean batched cache install required')
        cache_install_tensor_bound(cache_capacity)
        super().__init__(*args, **kwargs)
        self.batched_cache_install, self.cache_capacity = batched_cache_install, cache_capacity

    def _new_record(self, manifest, identity, client):
        return OasisCPUReceiveRecord(self, manifest, identity, client)


class OasisCUDAReceiveRecord(CUDASparseReceiveRecord):
    def __init__(self, *args):
        super().__init__(*args)
        self._slot_lease = None

    def copy_to_gpu_rows(self, pool, layer):
        self._live()
        if self._lock.locked() or not self._ready or not self._safe or self._installed:
            raise RuntimeError('one exact terminal-success CUDA receive required')
        try:
            self._registry.ordering.after_remote_write(self._registration)
            self._ordered = True
            reader = pool.copy_and_enqueue(self.manifest, self._buffer, layer)
            # The independent GPU clone has completed before the MR is ACKed.
            # Its bank reader and background backup keep separate storage pins.
            self._installed = True
            return reader
        except BaseException:
            self._local_unknown = 'GPU receive clone/ordering completion unknown'
            raise

    def _release_destination(self):
        if self._slot_lease is not None and self._local_unknown is not None:
            self._slot_lease.quarantine(self._local_unknown)
        super()._release_destination()

    def _unregister_destination(self):
        if self._slot_lease is None:
            return super()._unregister_destination()
        self._registry._owner()
        # close() has fenced every published unacknowledged write. A normal
        # ACK additionally proves successful delivery and local installation.
        if (self._local_unknown is not None or not self._closing
                or (self._published and not self._safe)
                or getattr(self, "_cache_copy_owners", ())
                or (self._acknowledged and not (
                    self._ready and self._installed and self._ordered))):
            raise RuntimeError("receive slot lacks remote/local retirement proof")
        self._slot_lease.release_after_proof()
        self._slot_lease = None
        self._registration = self._buffer = None

    def copy_to_cache(self, cache):
        self._live()
        if self._lock.locked() or not self._ready or not self._safe or self._installed:
            raise RuntimeError("one exact terminal-success CUDA receive required")
        self._cache_copy_owners = []
        try:
            # Retain the existing conservative GPUDirect ordering policy.
            # This device fence is included in delivery latency, not hidden.
            try:
                self._registry.ordering.after_remote_write(self._registration)
            except BaseException:
                self._local_unknown = "CUDA receive ordering unknown"
                raise
            self._ordered = True
            views = self.manifest.payload_views(self._buffer)
            self._cache_copy_owners.extend(views)
            scratch = getattr(self._registry, 'pinned_scratch', None)
            host = (scratch.receive[:self.manifest.nbytes] if scratch is not None
                    else torch.empty_like(self._buffer, device="cpu", pin_memory=True))
            self._cache_copy_owners.append(host)
            host.copy_(self._buffer, non_blocking=True)
            torch.cuda.current_stream(self._registry.device).synchronize()
            cpu_views = self.manifest.payload_views(host)
            self._cache_copy_owners.extend(cpu_views)
            if self._registry.batched_cache_install:
                profile = install_cpu_payloads(cpu_views, cache,
                    capacity=self._registry.cache_capacity, retained=self._cache_copy_owners)
            else:
                for payload in cpu_views:
                    head = payload.spec.kv_head
                    for offset, token in enumerate(payload.spec.token_ids):
                        if token in cache[head]:
                            raise RuntimeError("duplicate remote cache row")
                        cache[head][token] = payload.tensor[:, offset].clone()
                profile = row_cache_profile(cpu_views)
            self._installed = True
            self.profile.update(profile, cache_install_complete=True, cache_d2h_fenced=True)
        except BaseException:
            try:
                torch.cuda.current_stream(self._registry.device).synchronize()
            except BaseException:
                self._local_unknown = "Oasis cache D2H completion unknown"
            raise
        finally:
            if self._local_unknown is None:
                self._cache_copy_owners.clear()


class OasisCUDAReceiveRegistry(CUDASparseReceiveRegistry):
    def __init__(self, *args, receive_pool=None, batched_cache_install=False,
                 cache_capacity=2048, **kwargs):
        if type(batched_cache_install) is not bool:
            raise ValueError('explicit boolean batched cache install required')
        cache_install_tensor_bound(cache_capacity)
        super().__init__(*args, **kwargs)
        self.batched_cache_install, self.cache_capacity = batched_cache_install, cache_capacity
        if receive_pool is not None and (
                receive_pool.engine is not self.engine
                or receive_pool.budget is not self.budget
                or receive_pool.device != self.device
                or receive_pool.receiver_epoch != self.receiver_epoch):
            raise ValueError("request receive pool differs from worker registry")
        self.receive_pool = receive_pool

    def _destination_charge(self, manifest):
        # Physical maximum capacity is persistently charged by the pool.
        return 0 if self.receive_pool is not None else manifest.nbytes

    def _prepare_registration(self, record, **route):
        if self.receive_pool is None:
            return super()._prepare_registration(record, **route)
        record._registration_unknown = True
        try:
            lease = self.receive_pool.acquire(record.manifest, record.identity,
                endpoint=route["endpoint"], rail=route["rail"], device=self.device,
                ordering=self.ordering)
        except BaseException:
            if self.receive_pool.snapshot()['quarantine'] is None:
                # Known admission failure: no destination was published or MR
                # registration left unknown. The pool still owns idle MRs.
                record._registration_unknown = False
                self.budget.release(record.owner)
                self._records.pop(record.identity.transfer_id, None)
            raise
        record._slot_lease = lease
        record._buffer, record._registration = lease.buffer, lease.registration
        record.profile.update(reuse_receive_slots=True,
            allocate_seconds=lease.allocate_seconds,
            register_seconds=lease.register_seconds,
            physical_register_calls=lease.physical_register_calls)

    def _new_record(self, manifest, identity, client):
        return OasisCUDAReceiveRecord(self, manifest, identity, client)


class OasisLayerTransport:
    def __init__(self, manager, selected, *, request_id, incarnation, device,
                 vector_space, capacity, max_new, top_k, timeout=60, reuse_io=False,
                 combine_reserve_start=False, reuse_receive_slots=False, workers=2,
                 gpu_receive_to_bank=False, backup_budget_bytes=33554432,
                 staged_transport=False, sort_missing_tokens=False,
                 batched_bank_install=False, install_scratch_bytes=33554432,
                 batched_cache_install=False, ready_before_cleanup=False, binary_queries=False,
                 fused_search_delivery=False, compact_cache_snapshots=False,
                 reuse_pinned_scratch=False, event_bank_ready=False, binary_control_channel=False):
        manifest = selected.manifest
        if (manifest.layout.num_layers != 28 or manifest.layout.total_kv_heads != 4
                or manifest.layout.kv_heads_per_rank != 2
                or manifest.layout.head_dim != 128
                or manifest.layout.kv_dtype != "torch.float16"
                or tuple(route.rank for route in selected.shards) != (0, 1)
                or not 1 <= capacity <= 2048 or not 0 <= max_new <= capacity
                or not 1 <= top_k <= 512 or timeout <= 0
                or type(reuse_io) is not bool or type(combine_reserve_start) is not bool
                or type(reuse_receive_slots) is not bool
                or type(gpu_receive_to_bank) is not bool
                or type(staged_transport) is not bool
                or type(sort_missing_tokens) is not bool
                or type(batched_bank_install) is not bool
                or type(batched_cache_install) is not bool
                or type(ready_before_cleanup) is not bool
                or type(binary_queries) is not bool
                or type(fused_search_delivery) is not bool
                or type(compact_cache_snapshots) is not bool
                or type(reuse_pinned_scratch) is not bool or type(event_bank_ready) is not bool
                or type(binary_control_channel) is not bool
                or type(workers) is not int or not 1 <= workers <= 4):
            raise ValueError("bounded selected two-rank Qwen2.5-7B routes required")
        if ready_before_cleanup and not fused_search_delivery and (staged_transport or gpu_receive_to_bank or reuse_io
                or reuse_receive_slots or combine_reserve_start or batched_bank_install
                or batched_cache_install or workers != 2):
            raise ValueError('READY cleanup experiment requires isolated two-worker baseline')
        if staged_transport and (reuse_io or combine_reserve_start or reuse_receive_slots
                                 or gpu_receive_to_bank or workers != 2):
            raise ValueError('staged experiment requires unchanged two-worker packed/cache baseline')
        if batched_bank_install and (gpu_receive_to_bank or staged_transport
                or type(install_scratch_bytes) is not int
                or install_scratch_bytes < workers * install_tensor_bound(capacity)):
            raise ValueError('batched install requires isolated mode and admitted request scratch')
        if batched_cache_install and (gpu_receive_to_bank or staged_transport or batched_bank_install
                or type(install_scratch_bytes) is not int
                or install_scratch_bytes < workers * cache_install_tensor_bound(capacity)):
            raise ValueError('batched cache install requires isolated mode and admitted request scratch')
        self.manager, self.selected = manager, selected
        if binary_queries and ((ready_before_cleanup and not fused_search_delivery) or staged_transport or gpu_receive_to_bank or reuse_io
                or (reuse_receive_slots and not fused_search_delivery) or combine_reserve_start or batched_bank_install
                or batched_cache_install or workers != 2):
            raise ValueError('binary Q experiment requires isolated two-worker baseline')
        if fused_search_delivery and (staged_transport
                or gpu_receive_to_bank or reuse_io or combine_reserve_start
                or batched_bank_install or batched_cache_install or sort_missing_tokens or workers != 2
                or capacity > 32):
            raise ValueError('fused selection requires isolated bounded two-worker baseline')
        self.request_id, self.incarnation = request_id, incarnation
        self.device = torch.device(device)
        self.capacity, self.max_new, self.top_k = capacity, max_new, top_k
        self.timeout, self.vector_space = timeout, vector_space
        self.reuse_io = reuse_io
        self.combine_reserve_start = combine_reserve_start
        self.reuse_receive_slots = reuse_receive_slots
        self.gpu_receive_to_bank = gpu_receive_to_bank
        self.staged_transport = staged_transport
        self.sort_missing_tokens = sort_missing_tokens
        self.batched_bank_install = batched_bank_install
        self.batched_cache_install = batched_cache_install
        self.ready_before_cleanup = ready_before_cleanup
        self.binary_queries = binary_queries
        self.fused_search_delivery = fused_search_delivery
        if compact_cache_snapshots and not fused_search_delivery:
            raise ValueError('compact cache snapshots require fused delivery')
        self.compact_cache_snapshots = compact_cache_snapshots
        if (reuse_pinned_scratch or event_bank_ready) and (
                workers != 2 or capacity > 32 or staged_transport or gpu_receive_to_bank
                or batched_bank_install or batched_cache_install or (ready_before_cleanup and not fused_search_delivery)
                or type(install_scratch_bytes) is not int
                or workers*scratch_bytes(capacity) > install_scratch_bytes):
            raise ValueError('pinned/event experiment requires bounded ordinary two-worker installation')
        if event_bank_ready and not reuse_pinned_scratch:
            raise ValueError('event bank publication requires owned pinned scratch')
        self.reuse_pinned_scratch, self.event_bank_ready = reuse_pinned_scratch, event_bank_ready
        if binary_control_channel and (not binary_queries or not fused_search_delivery or reuse_io):
            raise ValueError('binary control channel requires fused binary Q and its own search clients')
        self.binary_control_channel=binary_control_channel
        self._channel_clients=None
        self.pinned_pool = (PinnedScratchPool(manager.transfer_budget,capacity=capacity,slots=workers)
                            if reuse_pinned_scratch else None)
        self.cleanup = OwnedReadyCleanup(workers=workers, capacity=56) if (ready_before_cleanup or event_bank_ready) else None
        self.stages = None
        self.receive_pool = (OasisReceiveSlotPool(manager.sparse_receive_engine,
            manager.transfer_budget, device=self.device,
            receiver_epoch=manager.worker_epoch, slots_per_rank=workers,
            capacity_bytes=capacity * 2 * 512) if reuse_receive_slots else None)
        self._shared_clients = None
        self._shared_close_future = None
        self._closing = False
        self._io_loop = None
        if reuse_io or binary_control_channel:
            self._io_loop = getattr(getattr(manager, "control", None), "loop", None)
            if (not isinstance(self._io_loop, asyncio.AbstractEventLoop)
                    or not self._io_loop.is_running() or self._io_loop.is_closed()):
                raise ValueError("reuse_io requires the manager's running I/O loop")
            self._outside_io_loop()
        shard = manifest.shards[0]
        self.prompt_tokens = ((shard.page_count - 1) * manifest.layout.page_size
                              + shard.last_page_valid_tokens)
        self.scope = SearchScope(self.prompt_tokens, manifest.layout.page_size, 128, "ip")
        self._cpu_cache = torch.empty((28, 4, self.prompt_tokens, 2, 128), dtype=torch.float16)
        self._cache_valid = torch.zeros((28, 4, self.prompt_tokens), dtype=torch.bool)
        self.cache = [[_HeadCPUCache(self._cpu_cache[layer, head], self._cache_valid[layer, head])
                       for head in range(4)] for layer in range(28)]
        self.gpu_backups = (OasisGPUBackupPool(self.cache, manager.transfer_budget,
            device=self.device, request_id=request_id, incarnation=incarnation,
            max_bytes=backup_budget_bytes, workers=workers) if gpu_receive_to_bank else None)
        self.local, self.lock = threading.local(), threading.Lock()
        self.workers, self.versions, self.trace = [], {}, []
        self._io_counts = dict(job_count=0, worker_loops_created=0,
            search_clients_created=0, control_clients_created=0,
            search_sessions_created=0, control_sessions_created=0)
        self._seen_shared_sessions = set()
        self.quarantined = False
        self.closed = False
        # Validate every rail before any asynchronous native registration.
        health = manager.sparse_receive_engine.health()
        self.endpoints = {}
        for route in selected.shards:
            state = health.get("rails", {}).get(route.rail, health)
            if (state.get("healthy") is not True
                    or not isinstance(state.get("session_id"), str)
                    or not state["session_id"]):
                raise ValueError("selected V rail has no healthy D receive session")
            self.endpoints[route.rank] = state["session_id"]
        if staged_transport:
            from sglang.srt.disaggregation.pvd.oasis_stages import BoundedLayerStages
            self.stages = BoundedLayerStages(
                (self._stage_search, self._stage_delivery, self._stage_install),
                initialize=self._stage_worker, retire=self._retire_stage_worker)

    def _outside_io_loop(self):
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._io_loop and running is not None:
            raise RuntimeError("cannot block or create Oasis clients on their I/O loop")

    def _new_clients(self, *, background_loop=None, kinds=('search', 'control')):
        clients = dict(
            search={r.rank: PVDShardSearchClient(r.url, timeout_seconds=self.timeout,
                        binary_queries=self.binary_queries,
                        binary_control_channel=self.binary_control_channel,
                        background_loop=background_loop) for r in self.selected.shards
                    if 'search' in kinds},
            control={r.rank: HttpShardClient(r.rank, r.url, timeout_seconds=self.timeout,
                         background_loop=background_loop) for r in self.selected.shards
                     if 'control' in kinds})
        self._io_counts["search_clients_created"] += len(clients["search"])
        self._io_counts["control_clients_created"] += len(clients["control"])
        return clients

    def _worker(self):
        if not hasattr(self.local, "state"):
            # Admission and client creation share the retirement lock. A close
            # cannot see an empty worker list while native owners are created.
            with self.lock:
                if self.closed or self._closing or self.quarantined:
                    raise RuntimeError("layer transport is closing or quarantined")
                if self.binary_control_channel:
                    if self._channel_clients is None:
                        self._channel_clients=self._new_clients(background_loop=self._io_loop,kinds=('search',))
                    clients=self._new_clients(kinds=('control',))
                    clients['search']=self._channel_clients['search']
                elif self.reuse_io:
                    if self._shared_clients is None:
                        self._shared_clients = self._new_clients(background_loop=self._io_loop)
                    clients = self._shared_clients
                else:
                    clients = self._new_clients()
                state = dict(loop=asyncio.new_event_loop(),
                    stream=torch.cuda.Stream(device=self.device), **clients,
                    owner_thread=threading.get_ident(),
                    registry=OasisCUDAReceiveRegistry(self.manager.sparse_receive_engine,
                        self.manager.transfer_budget, receiver_epoch=self.manager.worker_epoch,
                        device=self.device, combine_reserve_start=self.combine_reserve_start,
                        batched_cache_install=self.batched_cache_install, cache_capacity=self.capacity,
                        receive_pool=self.receive_pool))
                self.local.state = state
                self.workers.append(state)
                self._io_counts["job_count"] += 1
                self._io_counts["worker_loops_created"] += 1
        return self.local.state

    def _retire_worker(self, state):
        if state["owner_thread"] != threading.get_ident():
            raise RuntimeError("layer receive owners must retire on their worker thread")
        with self.lock:
            for kind in ("search", "control"):
                for rank, client in state[kind].items():
                    if client._session is None:
                        continue
                    identity = kind, rank
                    if self.reuse_io and identity in self._seen_shared_sessions:
                        continue
                    self._io_counts[f"{kind}_sessions_created"] += 1
                    if self.reuse_io:
                        self._seen_shared_sessions.add(identity)
        if not self.reuse_io:
            closing=({**state,'search':{}} if self.binary_control_channel else state)
            state["loop"].run_until_complete(self._close_clients(closing))
        del self.local.state
        if not state["registry"]._records and not state.get("quarantine"):
            state["loop"].close()
            with self.lock:
                self.workers.remove(state)

    @staticmethod
    async def _close_clients(clients):
        # Finish every close, including an exceptional one, before reporting
        # failure. An unfinished close never proves safe request retirement.
        outcomes = await asyncio.gather(
            *(c.close() for c in clients["search"].values()),
            *(c.close() for c in clients["control"].values()),
            return_exceptions=True)
        errors = [item for item in outcomes if isinstance(item, BaseException)]
        if errors:
            raise RuntimeError("Oasis HTTP client close failed; retain request owners") from errors[0]

    def io_snapshot(self):
        with self.lock:
            deliveries = [d for row in self.trace for d in row.get("deliveries", ())]
            owners = [d for row in self.trace for d in row.get('fused_profiles',())] if self.fused_search_delivery else deliveries
            sums = {name: sum(d[field] for d in owners) for name, field in (
                ("registration_count", "physical_register_calls"),
                ("unregistration_count", "physical_unregister_calls"),
                ("reserve_rpc_count", "reserve_calls"), ("start_rpc_count", "start_calls"),
                ("combined_rpc_count", "combined_calls"), ("poll_rpc_count", "poll_calls"),
                ("ack_rpc_count", "ack_calls"))}
            pool = self.receive_pool.snapshot() if self.receive_pool is not None else None
            if pool is not None:
                sums["registration_count"] = pool["physical_register_calls"]
                sums["unregistration_count"] = pool["physical_release_calls"]
            return dict(reuse_io=self.reuse_io, **self._io_counts, **sums,
                ready_before_cleanup=self.ready_before_cleanup,
                binary_queries=self.binary_queries,
                fused_search_delivery=self.fused_search_delivery,
                compact_cache_snapshots=self.compact_cache_snapshots,
                reuse_pinned_scratch=self.reuse_pinned_scratch,event_bank_ready=self.event_bank_ready,
                pinned_scratch=self.pinned_pool.snapshot() if self.pinned_pool else None,
                binary_control_channel=self.binary_control_channel,
                binary_channels={r:(c._channel.snapshot() if c._channel else None)
                    for r,c in self._channel_clients['search'].items()} if self._channel_clients else {},
                fused_rpc_count=sum(p['fused_calls'] for row in self.trace for p in row.get('fused_profiles',())),
                owned_cleanup=self.cleanup.snapshot() if self.cleanup else None,
                staged_transport=self.staged_transport,
                stages=self.stages.snapshot() if self.stages else None,
                stage_trace=list(self.stages.trace) if self.stages else None,
                gpu_receive_to_bank=self.gpu_receive_to_bank,
                sort_missing_tokens=self.sort_missing_tokens,
                batched_bank_install=self.batched_bank_install,
                batched_cache_install=self.batched_cache_install,
                gpu_backup=self.gpu_backups.snapshot() if self.gpu_backups else None,
                combine_reserve_start=self.combine_reserve_start,
                reuse_receive_slots=self.reuse_receive_slots, receive_pool=pool,
                delivery_count=len(deliveries),
                manager_io_loop_reused=self.reuse_io,
                shared_close_submitted=self._shared_close_future is not None,
                closing=self._closing, closed=self.closed)

    async def _select_and_fetch(self, state, ticket, query, bank, bootstrap,
                                *, select_only=False, plan=None, deadline=None):
        state["delivery_profiles"] = []
        state['fused_profiles'] = []
        state['gpu_readers'] = []
        state['fresh_gpu_rows'] = {}
        layer_cache = self.cache[ticket.layer]
        selected_ids = [None] * 4 if plan is None else list(plan['chosen'])
        plans = {}
        started = time.perf_counter()
        remote_rows = 0

        async def shard_job(route):
            nonlocal remote_rows
            if plan is not None:
                await deliver(route, plan['wires'][route.rank])
                return
            with self.lock:
                pin = self.versions.get(route.rank, (None, None))
            heads = tuple(range(route.rank * 2, route.rank * 2 + 2))
            requests = [(SearchRequestIdentity(self.vector_space, ROPE_APPLIED,
                self.selected.manifest.key.transfer_id, ticket.layer, head, *pin), query[head * 7:(head + 1) * 7],
                min(self.top_k, self.prompt_tokens), self.scope) for head in heads]
            if self.fused_search_delivery:
                chosen, rows = await self._fused_fetch(state,ticket,route,requests,bank,bootstrap)
                for head, ids in zip(heads,chosen,strict=True): selected_ids[head]=ids
                remote_rows += rows
                return
            results = await state["search"][route.rank].search_many(requests)
            resident = bank() if callable(bank) else bank
            pair = (results[0].index_version, results[0].id_mapping_version)
            if any((item.index_version, item.id_mapping_version) != pair for item in results):
                raise RuntimeError("layer selection returned mixed index versions")
            with self.lock:
                if self.versions.setdefault(route.rank, pair) != pair:
                    raise RuntimeError("immutable Prompt index version changed")
            specs = []
            for head, result in zip(heads, results, strict=True):
                # Native reply scores rank the GQA query union on this head.
                ranked = tuple(t for _, t in sorted(zip(result.scores, result.token_ids), reverse=True))
                old = () if resident is None else resident.ids[head]
                chosen = select_resident(ranked, old, capacity=self.capacity,
                    max_new=self.capacity if bootstrap else self.max_new)
                if not chosen:
                    raise RuntimeError("empty layer working set")
                selected_ids[head] = chosen
                missing = tuple(t for t in chosen if t not in layer_cache[head]
                    and not (self.gpu_backups and (t in old
                        or self.gpu_backups.contains(ticket.layer, head, t))))
                if missing:
                    # Only wire order changes. Chosen bank order and ranking
                    # remain exact; cache installation keys each row by token.
                    if self.sort_missing_tokens:
                        missing = tuple(sorted(missing))
                    specs.append(SparseKVSpec(ticket.request_id, ticket.incarnation,
                        f"oasis:{'bootstrap' if bootstrap else 'lookahead'}:{ticket.step}:{ticket.layer}",
                        0 if bootstrap else ticket.step + 1,
                        self.selected.manifest.key.transfer_id, *pair,
                        self.selected.manifest.layout.fingerprint, ticket.layer, head, missing))
            if select_only:
                plans[route.rank] = tuple(specs)
                return
            await deliver(route, specs)

        async def deliver(route, specs):
            nonlocal remote_rows
            if not specs:
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('layer expired before native destination submission')
            wire = SparseDeliveryManifest(tuple(specs), "torch.float16", 128)
            registry = state["registry"]
            record = registry.prepare(wire, key=self.selected.manifest.key, rank=route.rank,
                rail=route.rail, endpoint=self.endpoints[route.rank], sender_epoch=route.sender_epoch,
                client=state["control"][route.rank], owner_scope=self.incarnation)
            try:
                ready = await record.start()
                native_deadline = min(deadline, time.monotonic() + self.timeout) if deadline is not None else time.monotonic() + self.timeout
                while not ready:
                    if time.monotonic() >= native_deadline:
                        raise TimeoutError("Oasis native layer delivery expired")
                    await asyncio.sleep(0.001)
                    ready = await record.poll()
                tick = time.perf_counter()
                if self.gpu_backups:
                    reader = record.copy_to_gpu_rows(self.gpu_backups, ticket.layer)
                    state['gpu_readers'].append(reader)
                    state['fresh_gpu_rows'].update(reader.rows)
                else:
                    record.copy_to_cache(layer_cache)
                cache_copy_seconds = time.perf_counter() - tick
                if self.ready_before_cleanup:
                    state.setdefault('pending_cleanup', []).append(record)
                else:
                    await record.ack()
                    if not await record.close():
                        raise RuntimeError("terminal layer destination did not retire")
                remote_rows += sum(len(s.token_ids) for s in specs)
                state["delivery_profiles"].append(dict(record.profile, rank=route.rank,
                    nbytes=wire.nbytes, remote_rows=sum(len(s.token_ids) for s in specs),
                    cache_copy_seconds=cache_copy_seconds,
                    gpu_receive_to_bank=self.gpu_receive_to_bank))
                state["delivery_profiles"][-1].update(
                    sort_missing_tokens=self.sort_missing_tokens,
                    wire_runs=sum(sum(1 for _ in consecutive_token_runs(s.token_ids)) for s in specs),
                    wire_ids_sorted=all(tuple(sorted(s.token_ids)) == s.token_ids for s in specs),
                    wire_group_rows=[len(s.token_ids) for s in specs],
                )
            except BaseException:
                # Drain or retain all native destinations. Never infer success
                # from an HTTP cancellation, timeout, or receiver destruction.
                if not await record.close():
                    self.quarantined = True
                raise

        outcomes = await asyncio.gather(*(shard_job(r) for r in self.selected.shards),
                                        return_exceptions=True)
        errors = [o for o in outcomes if isinstance(o, BaseException)]
        if errors:
            raise errors[0]
        if select_only:
            return dict(chosen=tuple(selected_ids), wires=plans,
                        search_seconds=time.perf_counter() - started)
        return tuple(selected_ids), remote_rows, time.perf_counter() - started

    async def _fused_fetch(self,state,ticket,route,requests,bank,bootstrap):
        from sglang.srt.disaggregation.pvd.fused_search_delivery import prepare_fused, start_fused
        from sglang.srt.disaggregation.pvd.cache_snapshot import pack_cache_snapshot
        resident=bank() if callable(bank) else bank
        heads=[route.rank*2,route.rank*2+1]
        scope=dict(request_id=ticket.request_id,incarnation=ticket.incarnation,
            operation_id=f"oasis:{'bootstrap' if bootstrap else 'lookahead'}:{ticket.step}:{ticket.layer}",
            target_tokens=0 if bootstrap else ticket.step+1,
            entry_transfer_id=self.selected.manifest.key.transfer_id,
            layout_fingerprint=self.selected.manifest.layout.fingerprint,layer=ticket.layer,heads=heads,
            capacity=self.capacity,max_new=self.capacity if bootstrap else self.max_new,
            prompt_tokens=self.prompt_tokens,dtype=self.selected.manifest.layout.kv_dtype,
            head_dim=self.selected.manifest.layout.head_dim,
            resident=[list(resident.ids[h]) if resident is not None else [] for h in heads],
            cached=[(pack_cache_snapshot(self._cache_valid[ticket.layer,h].numpy())
                     if self.compact_cache_snapshots else self._cache_valid[ticket.layer,h].nonzero().flatten().tolist())
                    for h in heads])
        search_client=state['search'][route.rank]
        items=[search_client._prepare_search(identity,queries=q,top_k=k,scope=s)[0] for identity,q,k,s in requests]
        if not self.binary_queries and os.environ.get('PVD_PACKED_QUERY_BATCH') == '1':
            from sglang.srt.disaggregation.pvd.search_wire import pack_query_rows
            for item in items: item.update(pack_query_rows(item.pop('queries')))
        import uuid
        search=dict(batch_protocol='pvd.search.batch.v1',batch_id=uuid.uuid4().hex,items=items)
        record=prepare_fused(state['registry'],scope,search,key=self.selected.manifest.key,rank=route.rank,
            rail=route.rail,endpoint=self.endpoints[route.rank],sender_epoch=route.sender_epoch,
            client=state['control'][route.rank],owner_scope=self.incarnation,
            binary_queries=self.binary_queries)
        try:
            chosen,ready=await start_fused(record,search_client,requests)
            pair=(record.fused_results[0]['index_version'],record.fused_results[0]['id_mapping_version'])
            with self.lock:
                if self.versions.setdefault(route.rank,pair) != pair: raise RuntimeError('immutable fused index changed')
            if getattr(record,'fused_no_miss',False):
                if not await record.close(): raise RuntimeError('zero-miss fused authorization did not fence')
                state['fused_profiles'].append(dict(record.profile,nbytes=0,rank=route.rank))
                return chosen,0
            deadline=time.monotonic()+self.timeout
            while not ready:
                if time.monotonic() >= deadline: raise TimeoutError('fused native delivery expired')
                await asyncio.sleep(0.001);ready=await record.poll()
            tick=time.perf_counter();record.copy_to_cache(self.cache[ticket.layer])
            copy_seconds=time.perf_counter()-tick
            if self.ready_before_cleanup:
                # Native completion and local copy already succeeded. Keep the
                # registration on this owner until bank publication and cleanup.
                state.setdefault('pending_cleanup', []).append(record)
            else:
                await record.ack()
                if not await record.close(): raise RuntimeError('fused destination did not retire')
            rows=sum(len(s.token_ids) for s in record.manifest.specs)
            profile=dict(record.profile,nbytes=record.manifest.nbytes,remote_rows=rows,rank=route.rank,
                cache_copy_seconds=copy_seconds,gpu_receive_to_bank=False,sort_missing_tokens=False,
                wire_runs=sum(sum(1 for _ in consecutive_token_runs(s.token_ids)) for s in record.manifest.specs),
                wire_ids_sorted=all(tuple(sorted(s.token_ids))==s.token_ids for s in record.manifest.specs),
                wire_group_rows=[len(s.token_ids) for s in record.manifest.specs])
            state['delivery_profiles'].append(profile);state['fused_profiles'].append(profile)
            return chosen,rows
        except BaseException:
            if not await record.close(): self.quarantined=True
            raise

    async def _finish_owned_cleanup(self, state):
        errors = []
        for record in state.get('pending_cleanup', ()):
            try:
                await record.ack()
            except BaseException as exc:
                errors.append(exc)
            try:
                if not await record.close():
                    self.quarantined = True
                    errors.append(RuntimeError('owned destination did not retire'))
            except BaseException as exc:
                self.quarantined = True
                errors.append(exc)
            for profile in state.get('delivery_profiles', ()):
                if profile.get('rank') == record.identity.shard_rank:
                    profile.update(record.profile)
        if errors:
            raise RuntimeError('owned ACK/receive cleanup failed') from errors[0]

    def _stage_worker(self, stage, index):
        # Each persistent loop/client/registry is created and retired on this
        # actual stage thread. No registry or mutable job state crosses threads.
        with self.lock:
            if self.closed or self.quarantined:
                raise RuntimeError('persistent stage owner admission closed')
            kinds = ('search',) if stage == 'search' else ('control',) if stage == 'delivery' else ()
            state = dict(owner_thread=threading.get_ident(), stage=stage,
                         search={}, control={})
            self.workers.append(state)
            try:
                state.update(self._new_clients(kinds=kinds))
                state['loop'] = asyncio.new_event_loop()
                state['stream'] = torch.cuda.Stream(device=self.device)
                if stage == 'delivery':
                    state['registry'] = OasisCUDAReceiveRegistry(self.manager.sparse_receive_engine,
                        self.manager.transfer_budget, receiver_epoch=self.manager.worker_epoch,
                        device=self.device)
                self._io_counts['worker_loops_created'] += 1
            except BaseException:
                self.quarantined = True
                raise
            return state

    def _retire_stage_worker(self, stage, state):
        if state['owner_thread'] != threading.get_ident() or state['stage'] != stage:
            raise RuntimeError('persistent stage retirement on foreign thread')
        registry = state.get('registry')
        if state.get('quarantine') or registry is not None and registry._records:
            self.quarantined = True
            raise RuntimeError('retain undrained native stage owners')
        with self.lock:
            for kind in ('search', 'control'):
                self._io_counts[kind + '_sessions_created'] += sum(
                    client._session is not None for client in state[kind].values())
        state['loop'].run_until_complete(self._close_clients(state))
        state['loop'].close()
        with self.lock:
            self.workers.remove(state)

    def _drain_stage_payload(self, context):
        try:
            context['event'].synchronize()
            if context.get('stream') is not None:
                context['stream'].synchronize()
        except BaseException:
            self.quarantined = True
            with self.lock:
                self.workers.append(dict(quarantine=[context]))
            raise

    @torch.inference_mode()
    def _stage_search(self, context, state):
        context['stream'] = state['stream']
        with torch.cuda.device(self.device), torch.cuda.stream(state['stream']):
            state['stream'].wait_event(context['event'])
            host = torch.empty_like(context['query'], device='cpu', pin_memory=True)
            context['retained'].append(host)
            host.copy_(context['query'], non_blocking=True)
            state['stream'].synchronize()
            context['plan'] = state['loop'].run_until_complete(self._select_and_fetch(
                state, context['ticket'], host.tolist(), context['bank'],
                context['bootstrap'], select_only=True, deadline=context['deadline']))
        return context

    @torch.inference_mode()
    def _stage_delivery(self, context, state):
        context['stream'] = state['stream']
        with torch.cuda.device(self.device), torch.cuda.stream(state['stream']):
            chosen, rows, seconds = state['loop'].run_until_complete(self._select_and_fetch(
                state, context['ticket'], None, context['bank'], context['bootstrap'],
                plan=context['plan'], deadline=context['deadline']))
            context.update(chosen=chosen, remote_rows=rows,
                rpc_seconds=context['plan']['search_seconds'] + seconds,
                deliveries=state['delivery_profiles'])
        return context

    @torch.inference_mode()
    def _stage_install(self, context, state):
        context['stream'] = state['stream']
        ticket, chosen = context['ticket'], context['chosen']
        resident = context['bank']() if callable(context['bank']) else context['bank']
        with torch.cuda.device(self.device), torch.cuda.stream(state['stream']):
            width = max(map(len, chosen))
            keys = torch.zeros((4, width, 128), device=self.device, dtype=torch.float16)
            values = torch.zeros_like(keys)
            valid = torch.zeros((4, width), device=self.device, dtype=torch.bool)
            context['retained'].extend((resident, keys, values, valid))
            if resident is not None and resident.completion is not None:
                state['stream'].wait_event(resident.completion)
            for head, ids in enumerate(chosen):
                old = {} if resident is None else {token: i for i, token in enumerate(resident.ids[head])}
                hits = [(i, old[token]) for i, token in enumerate(ids) if token in old]
                misses = [(i, token) for i, token in enumerate(ids) if token not in old]
                if hits:
                    dst, src = zip(*hits)
                    keys[head, list(dst)] = resident.keys[head, list(src)]
                    values[head, list(dst)] = resident.values[head, list(src)]
                if misses:
                    host = torch.stack([self.cache[ticket.layer][head][token]
                                        for _, token in misses]).pin_memory()
                    gpu = host.to(self.device, non_blocking=True)
                    context['retained'].extend((host, gpu))
                    dst = [i for i, _ in misses]
                    keys[head, dst], values[head, dst] = gpu[:, 0], gpu[:, 1]
                valid[head, :len(ids)] = True
            complete = torch.cuda.Event()
            complete.record()
            complete.synchronize()
        with self.lock:
            self.trace.append(dict(step=ticket.step, layer=ticket.layer,
                remote_rows=context['remote_rows'], rpc_seconds=context['rpc_seconds'],
                deliveries=context['deliveries']))
        return LayerReply(ticket, PromptBank(chosen, keys, values, valid, complete))

    def job(self, query, bank, *, bootstrap=False):
        if (self.closed or self._closing or self.quarantined
                or self.gpu_backups and self.gpu_backups.quarantined):
            raise RuntimeError("layer transport is closed or quarantined")
        query = query.detach().clone()
        if query.shape != (28, 128) or query.device != self.device:
            raise ValueError("target post-RoPE Q required")
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        if self.staged_transport:
            transport = self
            class StagedCallback:
                def __call__(self, ticket):
                    raise RuntimeError('staged callback must be submitted with publication deadline')

                def submit_layer(self, ticket, *, published, timeout):
                    if (ticket.request_id, ticket.incarnation) != (transport.request_id, transport.incarnation):
                        raise RuntimeError('foreign staged layer transport ticket')
                    context = dict(query=query, event=event, bank=bank, bootstrap=bootstrap,
                        ticket=ticket, deadline=published + timeout, retained=[query, event, bank])
                    try:
                        future = transport.stages.submit((bootstrap, ticket), ticket, context,
                            published=published, timeout=timeout, cleanup=transport._drain_stage_payload)
                    except BaseException:
                        transport._drain_stage_payload(context)
                        raise
                    with transport.lock:
                        transport._io_counts['job_count'] += 1
                    return future
            return StagedCallback()

        @torch.inference_mode()
        def run(ticket, publish_ready=None):
            if (ticket.request_id, ticket.incarnation) != (self.request_id, self.incarnation):
                raise RuntimeError("foreign layer transport ticket")
            state = self._worker()
            retained = [query, event, bank]
            scratch = None
            try:
                with torch.cuda.device(self.device), torch.cuda.stream(state["stream"]):
                    if self.pinned_pool is not None:
                        scratch = self.pinned_pool.acquire()
                        retained.append(scratch)
                        state['registry'].pinned_scratch = scratch
                        scratch.begin()
                    state["stream"].wait_event(event)
                    host_q = (scratch.query if scratch is not None
                              else torch.empty_like(query, device="cpu", pin_memory=True))
                    retained.append(host_q)
                    host_q.copy_(query, non_blocking=True)
                    state["stream"].synchronize()
                    chosen, remote_rows, rpc_seconds = state["loop"].run_until_complete(
                        self._select_and_fetch(state, ticket,
                            host_q.float().numpy() if self.binary_queries else host_q.tolist(), bank, bootstrap))
                    resident = bank() if callable(bank) else bank
                    if resident is not None and resident.completion is not None:
                        state['stream'].wait_event(resident.completion)
                    install_started = time.perf_counter()
                    install_profile = None
                    if self.batched_bank_install:
                        keys, values, valid, install_profile = install_batched_bank(
                            chosen, resident, self.cache[ticket.layer], device=self.device,
                            capacity=self.capacity, prompt_tokens=self.prompt_tokens, retained=retained)
                    else:
                        if not self.gpu_receive_to_bank:
                            install_profile = dict(mode='per_head', selected_rows=sum(map(len, chosen)),
                                resident_rows=0, cpu_rows=0, kv_h2d_bytes=0, kv_h2d_calls=0,
                                resident_gather_calls=0, resident_scatter_calls=0,
                                cpu_scatter_calls=0, cuda=True)
                        width = max(map(len, chosen))
                        keys = torch.zeros((4, width, 128), device=self.device, dtype=torch.float16)
                        values, valid = torch.zeros_like(keys), torch.zeros((4, width),
                            device=self.device, dtype=torch.bool)
                        retained.extend((keys, values, valid))
                        for head, ids in enumerate(chosen):
                            old = {} if resident is None else {t: i for i, t in enumerate(resident.ids[head])}
                            hits = [(i, old[t]) for i, t in enumerate(ids) if t in old]
                            misses = [(i, t) for i, t in enumerate(ids) if t not in old]
                            if install_profile is not None:
                                install_profile['resident_rows'] += len(hits)
                                install_profile['cpu_rows'] += len(misses)
                                install_profile['kv_h2d_bytes'] += len(misses) * 512
                                install_profile['kv_h2d_calls'] += int(bool(misses))
                                install_profile['resident_gather_calls'] += 2 * int(bool(hits))
                                install_profile['resident_scatter_calls'] += 2 * int(bool(hits))
                                install_profile['cpu_scatter_calls'] += 2 * int(bool(misses))
                            if hits:
                                dst, src = zip(*hits)
                                keys[head, list(dst)] = resident.keys[head, list(src)]
                                values[head, list(dst)] = resident.values[head, list(src)]
                            if misses:
                                cpu_misses = []
                                for dst, token in misses:
                                    row = state['fresh_gpu_rows'].get((head, token))
                                    if row is None and self.gpu_backups:
                                        reader = self.gpu_backups.acquire(ticket.layer, head, token)
                                        if reader is not None:
                                            state['gpu_readers'].append(reader)
                                            row = reader.rows[head, token]
                                    if row is None:
                                        cpu_misses.append((dst, token))
                                    else:
                                        keys[head, dst], values[head, dst] = row[0], row[1]
                                if cpu_misses:
                                    rows = [self.cache[ticket.layer][head][t] for _, t in cpu_misses]
                                    if scratch is None:
                                        host = torch.stack(rows).pin_memory()
                                    else:
                                        host = scratch.bank[head,:len(rows)]
                                        torch.stack(rows,out=host)
                                    gpu = host.to(self.device, non_blocking=True)
                                    retained.extend((host, gpu))
                                    dst = [i for i, _ in cpu_misses]
                                    keys[head, dst], values[head, dst] = gpu[:, 0], gpu[:, 1]
                            valid[head, :len(ids)] = True
                    complete = torch.cuda.Event()
                    retained.append(complete)
                    complete.record()
                    if self.event_bank_ready:
                        if publish_ready is None:
                            raise RuntimeError('event bank requires owned publication callback')
                        # READY permits a consumer stream to enqueue wait_event;
                        # it is not a claim that local GPU copies have finished.
                        publish_ready(LayerReply(ticket,PromptBank(chosen,keys,values,valid,complete)))
                    # Source owners can retire only after the H2D/copy event.
                    if scratch is None:
                        complete.synchronize()
                    else:
                        scratch.finish(complete)
                    if install_profile is not None:
                        install_profile.update(completion_proven=True,
                            install_seconds=time.perf_counter() - install_started)
                    # Borrowed row aliases must disappear before a reader's
                    # last unpin may drop storage and refund its byte charge.
                    state['fresh_gpu_rows'].clear()
                    row = None
                    for reader in state['gpu_readers']:
                        reader.release_after_copy(complete)
                    state['gpu_readers'].clear()
                with self.lock:
                    self.trace.append(dict(step=ticket.step, layer=ticket.layer,
                        remote_rows=remote_rows, rpc_seconds=rpc_seconds,
                        bank_install=install_profile,
                        fused_profiles=state.get('fused_profiles',[]),
                        deliveries=state.get("delivery_profiles", [])))
                reply = LayerReply(ticket, PromptBank(chosen, keys, values, valid, complete))
                if publish_ready is not None and not self.event_bank_ready:
                    publish_ready(reply)
                return reply
            except BaseException:
                try:
                    state["stream"].synchronize()
                    if scratch is not None and scratch.active:
                        scratch.abort(state['stream'])
                except BaseException:
                    self.quarantined = True
                    state.setdefault("quarantine", []).append(retained)
                if state.get('gpu_readers'):
                    self.quarantined = True
                    state.setdefault('quarantine', []).append(state['gpu_readers'])
                raise
            finally:
                state['registry'].pinned_scratch = None
                # Native registries remain job/thread-local. Request clients
                # alone survive on the already-running background I/O loop.
                try:
                    if self.ready_before_cleanup:
                        state['loop'].run_until_complete(self._finish_owned_cleanup(state))
                finally:
                    self._retire_worker(state)

        if self.cleanup is not None:
            transport = self
            class OwnedCallback:
                def __call__(self, ticket):
                    raise RuntimeError('owned callback requires bounded submission')

                def submit_layer(self, ticket, *, published, timeout):
                    return transport.cleanup.submit(lambda publish: run(ticket, publish),
                        published=published, timeout=timeout)
            return OwnedCallback()

        return run

    def close(self):
        if self.cleanup is not None:
            self.cleanup.close()
        if self.stages is not None:
            self.stages.close()
        if self.reuse_io or self.binary_control_channel:
            self._outside_io_loop()
        with self.lock:
            if self.closed:
                return
            if self.workers or self.quarantined:
                raise RuntimeError("retain undrained layer transport owners")
            # Caller joins LayerLookahead before this transition. Reject late
            # jobs while waiting for the real HTTP-close future to complete.
            self._closing = True
            clients=self._channel_clients if self.binary_control_channel else self._shared_clients
            if clients is not None:
                if self._shared_close_future is None:
                    if not self._io_loop.is_running() or self._io_loop.is_closed():
                        raise RuntimeError("Oasis I/O loop stopped; retain request owners")
                    coroutine = self._close_clients(clients)
                    try:
                        self._shared_close_future = self.manager.control.submit(coroutine)
                    except BaseException:
                        coroutine.close()
                        raise
                future = self._shared_close_future
            else:
                future = None
        if future is not None:
            # Timeout/cancellation/error leaves the cached future, clients and
            # cache intact. Retrying close never submits a duplicate close.
            future.result(timeout=self.timeout)
        if self.receive_pool is not None:
            self.receive_pool.close()
        if self.pinned_pool is not None:
            self.pinned_pool.close()
        if self.gpu_backups is not None:
            self.gpu_backups.close()
        self.cache.clear()
        self._cpu_cache = self._cache_valid = None
        self._shared_clients = None
        self.closed = True
