"""Shared CPU request-loop fixture, using real local V shard HTTP services.

Not production serving: P->V copies bytes locally, no model Decode scheduling,
source Entry ownership is held by the fixture, and no native RDMA is submitted.
"""

import asyncio
import uuid
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import torch
from aiohttp.test_utils import TestServer
from sglang.srt.disaggregation.pvd.control_server import (
    HttpShardClient,
    create_shard_app,
)
from sglang.srt.disaggregation.pvd.cpu_prefetch_request import CPUPrefetchRequest
from sglang.srt.disaggregation.pvd.cpu_sparse_delivery import (
    CPUReceiveRoute,
    CPUSparseDelivery,
)
from sglang.srt.disaggregation.pvd.kv_packer import (
    describe_kv_layout,
    pack_full_prompt_kv_head_shard,
)
from sglang.srt.disaggregation.pvd.probe_search import ProbeSearchRoute
from sglang.srt.disaggregation.pvd.prompt_index import (
    PromptIndexManager,
    SearchRequestIdentity,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED, QueryHeadMapping
from sglang.srt.disaggregation.pvd.protocol import (
    KVEntryKey,
    KVEntryManifest,
    KVLayoutSignature,
    KVShardManifest,
)
from sglang.srt.disaggregation.pvd.search_client import (
    PVDShardSearchClient,
    SearchScope,
)
from sglang.srt.disaggregation.pvd.sparse_install import CPUInstallGroup
from sglang.srt.disaggregation.pvd.sparse_payload import (
    SparseKVPayload,
    SparseKVSpec,
    pack_sparse_kv,
)
from sglang.srt.disaggregation.pvd.sparse_receiver import SparseReceiveRegistry
from sglang.srt.disaggregation.pvd.sparse_working_set import CPUSparseWorkingSet
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from sglang.srt.disaggregation.pvd.vector_store import VectorKVStore


class ControlledFixture:
    def __init__(self, pool, prefix, pipeline):
        self.pool, self.prefix = pool, prefix
        self.prompt_tokens = len(prefix.tokens)
        self.incarnation = uuid.uuid4().hex
        space = pipeline.probe_config.target_model_id
        heads = pool.k_buffer[0].shape[1]
        assert heads % 2 == 0
        self.mapping = QueryHeadMapping(pipeline.probe_config.head_count, heads)
        extra = deepcopy(describe_kv_layout(pool))
        extra["component_token_shapes"] = [
            [heads // 2, s[1]] for s in extra["component_token_shapes"]
        ]
        extra["component_bytes_per_token"] = [
            n // 2 for n in extra["component_bytes_per_token"]
        ]
        self.layout = KVLayoutSignature(
            model_id=space,
            model_revision="fixture",
            kv_dtype="torch.float32",
            page_size=1,
            num_layers=len(pool.k_buffer),
            total_kv_heads=heads,
            kv_heads_per_rank=heads // 2,
            head_dim=pool.k_buffer[0].shape[-1],
            tp_size=2,
            pp_size=1,
            tensor_layout=extra["tensor_layout"],
            extra=extra,
        )
        self.packed = [
            pack_full_prompt_kv_head_shard(
                pool,
                range(self.prompt_tokens),
                page_size=1,
                head_start=r * (heads // 2),
                head_count=heads // 2,
            )
            for r in (0, 1)
        ]
        self.shards = [
            KVShardManifest(
                rank=r,
                rail="cpu-fixture",
                expected_bytes=p.expected_bytes,
                page_count=self.prompt_tokens,
                last_page_valid_tokens=1,
                layer_start=0,
                layer_end=len(pool.k_buffer),
            )
            for r, p in enumerate(self.packed)
        ]
        self.manifest = KVEntryManifest(
            key=KVEntryKey.new(space, prefix.request_id),
            layout=self.layout,
            prompt_token_count=self.prompt_tokens,
            shards=self.shards,
        )
        self.stores, self.offsets, self.budgets, banks, routes = {}, {}, [], {}, {}
        scope = SearchScope(self.prompt_tokens, 1, self.layout.head_dim, "ip")
        for rank in (0, 1):
            budget = TransferBudget(8 << 20, 32)
            self.budgets.append(budget)
            manager = PromptIndexManager(vector_space=space, metric="ip", budget=budget)
            store = VectorKVStore(
                rank=rank,
                world_size=2,
                rail="cpu-fixture",
                device="cpu",
                total_pages=self.prompt_tokens * 2,
                page_bytes=self.packed[rank].expected_bytes // self.prompt_tokens,
                endpoint=f"controlled-{rank}",
                transfer_engine=FakeTransferEngine(),
                allow_cpu_for_tests=True,
                prompt_index=manager,
            )
            self.stores[rank] = store
            entry = store.create_entry(self.manifest)
            store.begin_p_write(self.manifest.key)
            offset = entry.allocation.start_page * store.page_bytes
            self.offsets[rank] = offset
            store.pool[offset : offset + self.packed[rank].expected_bytes] = (
                self.packed[rank].tensor
            )
            store.commit_p_write(self.manifest.key, self.packed[rank].expected_bytes)
            groups = tuple(
                (layer, head)
                for layer in range(len(pool.k_buffer))
                for head in range(rank * (heads // 2), (rank + 1) * (heads // 2))
            )
            bank_budget = TransferBudget(8 << 20, 2)
            self.budgets.append(bank_budget)
            banks[rank] = CPUSparseWorkingSet(
                request_id=prefix.request_id,
                incarnation=self.incarnation,
                entry_transfer_id=self.manifest.key.transfer_id,
                layout_fingerprint=self.layout.fingerprint,
                expected_groups=groups,
                prompt_tokens=self.prompt_tokens,
                head_dim=self.layout.head_dim,
                max_union_tokens=self.prompt_tokens,
                budget=bank_budget,
            )
            routes[rank] = tuple(
                ProbeSearchRoute(
                    qhead,
                    SearchRequestIdentity(
                        space,
                        ROPE_APPLIED,
                        self.manifest.key.transfer_id,
                        layer,
                        self.mapping.kv_head_for(qhead),
                    ),
                    scope,
                    2,
                )
                for layer, head in groups
                for qhead in range(self.mapping.num_query_heads)
                if self.mapping.kv_head_for(qhead) == head
            )
        self.group = CPUInstallGroup(banks, interval=4, lead_tokens=1)
        initial = self.group.begin(0)
        # Bootstrap before any retrieval index build: no fake search prerequisite.
        for rank in (0, 1):
            rows = []
            try:
                for layer, head in banks[rank].expected_groups:
                    spec = SparseKVSpec(
                        prefix.request_id,
                        self.incarnation,
                        initial.operation_id,
                        0,
                        self.manifest.key.transfer_id,
                        "bootstrap-no-index",
                        "bootstrap-logical-positions",
                        self.layout.fingerprint,
                        layer,
                        head,
                        tuple(range(self.prompt_tokens)),
                    )
                    rows.append(
                        SparseKVPayload(
                            spec,
                            torch.stack(
                                (
                                    pool.k_buffer[layer][: self.prompt_tokens, head],
                                    pool.v_buffer[layer][: self.prompt_tokens, head],
                                )
                            ),
                        )
                    )
                self.group.stage(initial, rank, rows)
            finally:
                for row in rows:
                    row.close()
        assert self.group.try_install(initial, {0: 0, 1: 0})
        for store in self.stores.values():
            assert store.progress_prompt_indexes()["built"] == 1
        self.request = CPUPrefetchRequest(
            self.group,
            pipeline,
            head_mapping=self.mapping,
            rank_routes=routes,
            max_union_tokens=self.prompt_tokens,
        )
        self.packed_specs = []

    def refresh_prefix(self, count=3):
        # P's first token plus `count` D tokens, fixed fixture ids, not sampled.
        return replace(
            self.prefix,
            tokens=self.prefix.tokens + tuple(range(9, 10 + count)),
            committed_position=count,
            version=f"committed-{count}",
        )

    @contextmanager
    def pack_source(self, rank, specs):
        store = self.stores[rank]
        current = store.prompt_index.gate_for(self.manifest.key.transfer_id).descriptor
        assert current is not None
        budget = TransferBudget(8 << 20, len(specs))
        self.budgets.append(budget)
        with ExitStack() as stack:
            rows = [
                stack.enter_context(
                    pack_sparse_kv(
                        store.pool[
                            self.offsets[rank] : self.offsets[rank]
                            + self.packed[rank].expected_bytes
                        ],
                        layout=self.layout,
                        shard=self.shards[rank],
                        spec=spec,
                        entry_transfer_id=self.manifest.key.transfer_id,
                        index_version=current.index_version,
                        id_mapping_version=current.id_mapping_version,
                        budget=budget,
                    )
                )
                for spec in specs
            ]
            for row in rows:
                spec = row.spec
                expected = torch.stack(
                    (
                        self.pool.k_buffer[spec.layer][
                            list(spec.token_ids), spec.kv_head
                        ],
                        self.pool.v_buffer[spec.layer][
                            list(spec.token_ids), spec.kv_head
                        ],
                    )
                )
                torch.testing.assert_close(row.tensor, expected, atol=0, rtol=0)
                self.packed_specs.append(spec)
            yield rows
        assert budget.snapshot()["used_staging_bytes"] == 0

    @asynccontextmanager
    async def clients(self, *, wire_delivery=False):
        async with AsyncExitStack() as stack:
            clients, delivery_routes = {}, {}
            for rank, store in self.stores.items():
                server = await stack.enter_async_context(
                    TestServer(create_shard_app(store))
                )
                clients[rank] = PVDShardSearchClient(str(server.make_url("")))
                stack.push_async_callback(clients[rank].close)
                if wire_delivery:
                    control = HttpShardClient(rank, str(server.make_url("")))
                    stack.push_async_callback(control.close)
                    delivery_routes[rank] = CPUReceiveRoute(
                        control, store.worker_epoch, f"cpu-D-{rank}", store.rail
                    )
                    budget = TransferBudget(8 << 20, 32)
                    self.budgets.append(budget)
                    store.transfer_engine.lifecycle_manager = SimpleNamespace(
                        budget=budget
                    )
            delivery = None
            if wire_delivery:
                if getattr(self.request, "_lifecycle_claimed", False):
                    raise ValueError("configure wire Delivery before fixture admission")
                budget = TransferBudget(8 << 20, 32)
                self.budgets.append(budget)
                registry = SparseReceiveRegistry(
                    FakeTransferEngine(), budget, receiver_epoch=self.incarnation
                )
                delivery = CPUSparseDelivery(
                    self.group,
                    registry,
                    key=self.manifest.key,
                    routes=delivery_routes,
                    poll_interval_seconds=0.001,
                )
                prior = self.request
                self.request = CPUPrefetchRequest(
                    self.group,
                    prior.pipeline,
                    head_mapping=self.mapping,
                    rank_routes=prior._routes,
                    max_union_tokens=self.prompt_tokens,
                    delivery=delivery,
                )
            try:
                yield clients
            finally:
                if delivery is not None:

                    async def drain():
                        # Cancelling HTTP does not cancel its server-side thread.
                        # Wait for actual fence proof; timeout never frees a MR.
                        while await delivery.close():
                            await asyncio.sleep(0.001)

                    await asyncio.wait_for(drain(), 5)

    def close(self):
        try:
            self.request.close()
        finally:
            for store in self.stores.values():
                store.close()
        assert all(b.snapshot()["used_staging_bytes"] == 0 for b in self.budgets)


async def verify_controlled_roundtrip(pool, prefix, pipeline, *, boundary_start=False):
    fixture = ControlledFixture(pool, prefix, pipeline)
    try:
        async with fixture.clients() as clients:
            snapshot = fixture.refresh_prefix(4 if boundary_start else 3)
            if boundary_start:
                assert not fixture.request.can_decode(4)
            epoch = await fixture.request.refresh(
                snapshot,
                query_positions=(
                    len(snapshot.tokens) - 1
                    if boundary_start
                    else len(snapshot.tokens),
                ),
                clients=clients,
                pack_source=fixture.pack_source,
            )
            assert fixture.group.coordinator.snapshot()["installed_tokens"] == 0
            assert not fixture.request.can_decode(4)
            assert fixture.request.try_install({0: 4, 1: 4})
            assert fixture.request.can_decode(4)
            for rank in (0, 1):
                with fixture.group.read(rank, 4) as groups:
                    assert all(
                        s.operation_id == epoch.operation_id
                        and s.incarnation == epoch.incarnation
                        for s, _ in groups.values()
                    )
            return {
                "shared_epoch": True,
                "query_source": "committed" if boundary_start else "predicted",
                "v_shards": 2,
                "installed_boundary": 4,
                "kv_groups": len(fixture.packed_specs),
                "online_gpu_rdma_validated": False,
            }
    finally:
        fixture.close()
