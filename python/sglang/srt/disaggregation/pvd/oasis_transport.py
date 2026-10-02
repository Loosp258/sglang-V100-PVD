"""Per-layer CAGRA selection and native V -> D CPU-cache misses.

The CPU receive registry proves exact native terminal success before reading.
Each worker owns its asyncio loop, clients, CUDA stream and receive records.
Unknown remote/native completion keeps those owners in the request quarantine.
"""

import asyncio
import threading
import time

import torch

from sglang.srt.disaggregation.pvd.control_server import HttpShardClient
from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import CUDASparseReceiveRecord, CUDASparseReceiveRegistry
from sglang.srt.disaggregation.pvd.oasis_pipeline import LayerReply, select_resident
from sglang.srt.disaggregation.pvd.oasis_qwen import PromptBank
from sglang.srt.disaggregation.pvd.prompt_index import SearchRequestIdentity
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED
from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient, SearchScope
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
from sglang.srt.disaggregation.pvd.sparse_receiver import SparseReceiveRecord, SparseReceiveRegistry


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
            for payload in views:
                head = payload.spec.kv_head
                for offset, token in enumerate(payload.spec.token_ids):
                    if token in cache[head]:
                        raise RuntimeError("duplicate remote cache row")
                    # The synchronous clone owns its bytes before V is ACKed.
                    cache[head][token] = payload.tensor[:, offset].clone()
        finally:
            for payload in views:
                payload.close()
        self._installed = True


class OasisCPUReceiveRegistry(SparseReceiveRegistry):
    def _new_record(self, manifest, identity, client):
        return OasisCPUReceiveRecord(self, manifest, identity, client)


class OasisCUDAReceiveRecord(CUDASparseReceiveRecord):
    def copy_to_cache(self, cache):
        self._live()
        if self._lock.locked() or not self._ready or not self._safe or self._installed:
            raise RuntimeError("one exact terminal-success CUDA receive required")
        self._cache_copy_owners = []
        try:
            # Retain the existing conservative GPUDirect ordering policy.
            # This device fence is included in delivery latency, not hidden.
            self._registry.ordering.after_remote_write(self._registration)
            self._ordered = True
            views = self.manifest.payload_views(self._buffer)
            self._cache_copy_owners.extend(views)
            host = torch.empty_like(self._buffer, device="cpu", pin_memory=True)
            self._cache_copy_owners.append(host)
            host.copy_(self._buffer, non_blocking=True)
            torch.cuda.current_stream(self._registry.device).synchronize()
            cpu_views = self.manifest.payload_views(host)
            self._cache_copy_owners.extend(cpu_views)
            for payload in cpu_views:
                head = payload.spec.kv_head
                for offset, token in enumerate(payload.spec.token_ids):
                    if token in cache[head]:
                        raise RuntimeError("duplicate remote cache row")
                    cache[head][token] = payload.tensor[:, offset].clone()
            self._installed = True
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
    def _new_record(self, manifest, identity, client):
        return OasisCUDAReceiveRecord(self, manifest, identity, client)


class OasisLayerTransport:
    def __init__(self, manager, selected, *, request_id, incarnation, device,
                 vector_space, capacity, max_new, top_k, timeout=60):
        manifest = selected.manifest
        if (manifest.layout.num_layers != 28 or manifest.layout.total_kv_heads != 4
                or manifest.layout.kv_heads_per_rank != 2
                or manifest.layout.head_dim != 128
                or manifest.layout.kv_dtype != "torch.float16"
                or tuple(route.rank for route in selected.shards) != (0, 1)
                or not 1 <= capacity <= 2048 or not 0 <= max_new <= capacity
                or not 1 <= top_k <= 512 or timeout <= 0):
            raise ValueError("bounded selected two-rank Qwen2.5-7B routes required")
        self.manager, self.selected = manager, selected
        self.request_id, self.incarnation = request_id, incarnation
        self.device = torch.device(device)
        self.capacity, self.max_new, self.top_k = capacity, max_new, top_k
        self.timeout, self.vector_space = timeout, vector_space
        shard = manifest.shards[0]
        self.prompt_tokens = ((shard.page_count - 1) * manifest.layout.page_size
                              + shard.last_page_valid_tokens)
        self.scope = SearchScope(self.prompt_tokens, manifest.layout.page_size, 128, "ip")
        self._cpu_cache = torch.empty((28, 4, self.prompt_tokens, 2, 128), dtype=torch.float16)
        self._cache_valid = torch.zeros((28, 4, self.prompt_tokens), dtype=torch.bool)
        self.cache = [[_HeadCPUCache(self._cpu_cache[layer, head], self._cache_valid[layer, head])
                       for head in range(4)] for layer in range(28)]
        self.local, self.lock = threading.local(), threading.Lock()
        self.workers, self.versions, self.trace = [], {}, []
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

    def _worker(self):
        if not hasattr(self.local, "state"):
            loop = asyncio.new_event_loop()
            state = dict(loop=loop, stream=torch.cuda.Stream(device=self.device),
                search={r.rank: PVDShardSearchClient(r.url, timeout_seconds=self.timeout)
                        for r in self.selected.shards},
                control={r.rank: HttpShardClient(r.rank, r.url, timeout_seconds=self.timeout)
                         for r in self.selected.shards},
                registry=OasisCUDAReceiveRegistry(self.manager.sparse_receive_engine,
                    self.manager.transfer_budget, receiver_epoch=self.manager.worker_epoch,
                    device=self.device))
            self.local.state = state
            with self.lock:
                self.workers.append(state)
        return self.local.state

    async def _select_and_fetch(self, state, ticket, query, bank, bootstrap):
        layer_cache = self.cache[ticket.layer]
        selected_ids = [None] * 4
        started = time.perf_counter()
        remote_rows = 0

        async def shard_job(route):
            nonlocal remote_rows
            with self.lock:
                pin = self.versions.get(route.rank, (None, None))
            heads = tuple(range(route.rank * 2, route.rank * 2 + 2))
            requests = [(SearchRequestIdentity(self.vector_space, ROPE_APPLIED,
                self.selected.manifest.key.transfer_id, ticket.layer, head, *pin), query[head * 7:(head + 1) * 7],
                min(self.top_k, self.prompt_tokens), self.scope) for head in heads]
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
                missing = tuple(t for t in chosen if t not in layer_cache[head])
                if missing:
                    specs.append(SparseKVSpec(ticket.request_id, ticket.incarnation,
                        f"oasis:{'bootstrap' if bootstrap else 'lookahead'}:{ticket.step}:{ticket.layer}",
                        0 if bootstrap else ticket.step + 1,
                        self.selected.manifest.key.transfer_id, *pair,
                        self.selected.manifest.layout.fingerprint, ticket.layer, head, missing))
            if not specs:
                return
            wire = SparseDeliveryManifest(tuple(specs), "torch.float16", 128)
            registry = state["registry"]
            record = registry.prepare(wire, key=self.selected.manifest.key, rank=route.rank,
                rail=route.rail, endpoint=self.endpoints[route.rank], sender_epoch=route.sender_epoch,
                client=state["control"][route.rank], owner_scope=self.incarnation)
            try:
                ready = await record.start()
                deadline = time.monotonic() + self.timeout
                while not ready:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Oasis native layer delivery expired")
                    await asyncio.sleep(0.001)
                    ready = await record.poll()
                record.copy_to_cache(layer_cache)
                await record.ack()
                if not await record.close():
                    raise RuntimeError("terminal layer destination did not retire")
                remote_rows += sum(len(s.token_ids) for s in specs)
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
        return tuple(selected_ids), remote_rows, time.perf_counter() - started

    def job(self, query, bank, *, bootstrap=False):
        if self.closed or self.quarantined:
            raise RuntimeError("layer transport is closed or quarantined")
        query = query.detach().clone()
        if query.shape != (28, 128) or query.device != self.device:
            raise ValueError("target post-RoPE Q required")
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))

        @torch.inference_mode()
        def run(ticket):
            if (ticket.request_id, ticket.incarnation) != (self.request_id, self.incarnation):
                raise RuntimeError("foreign layer transport ticket")
            state = self._worker()
            retained = [query, event, bank]
            try:
                with torch.cuda.device(self.device), torch.cuda.stream(state["stream"]):
                    state["stream"].wait_event(event)
                    host_q = torch.empty_like(query, device="cpu", pin_memory=True)
                    retained.append(host_q)
                    host_q.copy_(query, non_blocking=True)
                    state["stream"].synchronize()
                    chosen, remote_rows, rpc_seconds = state["loop"].run_until_complete(
                        self._select_and_fetch(state, ticket, host_q.tolist(), bank, bootstrap))
                    resident = bank() if callable(bank) else bank
                    width = max(map(len, chosen))
                    keys = torch.zeros((4, width, 128), device=self.device, dtype=torch.float16)
                    values, valid = torch.zeros_like(keys), torch.zeros((4, width),
                        device=self.device, dtype=torch.bool)
                    retained.extend((keys, values, valid))
                    for head, ids in enumerate(chosen):
                        old = {} if resident is None else {t: i for i, t in enumerate(resident.ids[head])}
                        hits = [(i, old[t]) for i, t in enumerate(ids) if t in old]
                        misses = [(i, t) for i, t in enumerate(ids) if t not in old]
                        if hits:
                            dst, src = zip(*hits)
                            keys[head, list(dst)] = resident.keys[head, list(src)]
                            values[head, list(dst)] = resident.values[head, list(src)]
                        if misses:
                            host = torch.stack([self.cache[ticket.layer][head][t] for _, t in misses]).pin_memory()
                            gpu = host.to(self.device, non_blocking=True)
                            retained.extend((host, gpu))
                            dst = [i for i, _ in misses]
                            keys[head, dst], values[head, dst] = gpu[:, 0], gpu[:, 1]
                        valid[head, :len(ids)] = True
                    complete = torch.cuda.Event()
                    complete.record()
                    # Source owners can retire only after the H2D/copy event.
                    complete.synchronize()
                with self.lock:
                    self.trace.append(dict(step=ticket.step, layer=ticket.layer,
                        remote_rows=remote_rows, rpc_seconds=rpc_seconds))
                return LayerReply(ticket, PromptBank(chosen, keys, values, valid, complete))
            except BaseException:
                try:
                    state["stream"].synchronize()
                except BaseException:
                    self.quarantined = True
                    state.setdefault("quarantine", []).append(retained)
                raise
            finally:
                # Clients/registries are loop-affine and drained on their owner
                # worker. Successful operations leave no receive registrations.
                async def close_clients():
                    await asyncio.gather(*(c.close() for c in state["search"].values()),
                                         *(c.close() for c in state["control"].values()))
                state["loop"].run_until_complete(close_clients())
                del self.local.state
                if not state["registry"]._records and not state.get("quarantine"):
                    state["loop"].close()
                    with self.lock:
                        self.workers.remove(state)

        return run

    def close(self):
        if self.workers or self.quarantined:
            raise RuntimeError("retain undrained layer transport owners")
        self.cache.clear()
        self._cpu_cache = self._cache_valid = None
        self.closed = True
