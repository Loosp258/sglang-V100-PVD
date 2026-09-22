"""CUDA request assembly: CPU policy substitution + real HTTP, fake payloads."""

import asyncio

import pytest
from sglang.srt.disaggregation.pvd.cpu_prefetch_request import CPUPrefetchRequest
from sglang.srt.disaggregation.pvd.cuda_prefetch_request import CUDAPrefetchRequest
from sglang.srt.disaggregation.pvd.prediction import (
    ProbeConfig,
    QueryVectors,
    snapshot_committed,
)
from sglang.srt.disaggregation.pvd.probe_search import ProbeSearchRoute
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.search_client import (
    PVDShardSearchClient,
    SearchScope,
)
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError
from test_pvd_cpu_sparse_delivery import wait_acks
from test_pvd_cuda_probe_search import bridge
from test_pvd_cuda_sparse_delivery import case, complete
from test_pvd_prompt_index import ident


def controller(c, monkeypatch):
    _, _, _, pipeline, _, budget, _ = bridge(monkeypatch)
    pipeline.probe_config = ProbeConfig("target/model-8b", (0, 1), head_count=1)
    pipeline.probe.config = pipeline.probe_config
    captures = []

    def capture(prefix, positions):
        captures.append(tuple(positions))
        return tuple(
            QueryVectors(
                vector_space="target/model-8b",
                version="captured-Q",
                layer=layer,
                head_start=0,
                head_count=1,
                positions=tuple(positions),
                valid_length=len(positions),
                vectors=c.pool.k_buffer[layer][3, 0].repeat(len(positions), 1, 1),
                prefix_version=prefix.version,
                positional_encoding="rope_applied",
                request_id=prefix.request_id,
            )
            for layer in (0, 1)
        )

    pipeline.probe.capture = lambda p, d: capture(
        p, range(len(p.tokens), len(p.tokens) + len(d.tokens))
    )
    pipeline.probe.capture_committed = capture
    describe = c.group.describe_banks
    monkeypatch.setattr(
        c.group,
        "describe_banks",
        lambda: {
            rank: {**meta, "device": "cuda:0"} for rank, meta in describe().items()
        },
    )
    scope = SearchScope(8, c.entry.layout.page_size, c.entry.layout.head_dim, "l2")
    routes = tuple(
        ProbeSearchRoute(0, ident(c.entry.key.transfer_id, layer), scope, 1)
        for layer in (0, 1)
    )
    kwargs = dict(
        copy_budget=budget,
        max_head_dim=8,
        head_mapping=QueryHeadMapping(1, 1),
        rank_routes={0: routes},
        max_union_tokens=2,
        delivery=c.sink,
    )
    request = CUDAPrefetchRequest(c.group, pipeline, **kwargs)
    monkeypatch.setattr(
        request._session, "_query_device", lambda t: t.device.type == "cpu"
    )
    return request, captures, kwargs


async def run_refresh(c, request, client, count):
    prefix = snapshot_committed("consumer", [1] * (8 + count), count, f"prefix-{count}")
    position = len(prefix.tokens) - int(count == 4)
    task = asyncio.create_task(
        request.refresh(prefix, query_positions=(position,), clients={0: client})
    )

    async def progress():
        while not task.done():
            for delivery in tuple(c.store.entries[c.entry.key].deliveries.values()):
                handle = delivery.transfer_handle
                if handle and not handle.transport_state.is_locally_safe_to_release:
                    c.engine.finish(handle)
            c.store.progress_transfers()
            await asyncio.sleep(0.001)
        return await task

    return await asyncio.wait_for(progress(), 5)


def test_two_periodic_rounds_capture_search_deliver_and_install(monkeypatch):
    async def run():
        async with case(monkeypatch) as c:
            # Fixture bootstrap only: production initial Prompt stays index-independent.
            await complete(c, 0, tuple(range(8)))
            request, captures, _ = controller(c, monkeypatch)
            client = PVDShardSearchClient(c.client.base_url)
            try:
                for count in (3, 7):
                    epoch = await run_refresh(c, request, client, count)
                    assert epoch.target_tokens == count + 1
                    assert not request.can_decode(count + 1)
                    assert request.pending_install_boundary == count + 1
                    assert request.try_install({0: count + 1})
                    await wait_acks(c.sink)
                    assert request.can_decode(count + 1)
                    permit = c.group.runtime.begin_forward(count + 1)
                    with c.group.read(0, count + 1) as groups:
                        assert all(
                            spec.token_ids == (3,) for spec, tensor in groups.values()
                        )
                    assert c.group.runtime.finish_forward(
                        permit, readers_drained=True, succeeded=True
                    )
                    assert c.registry.budget.snapshot()["used_staging_bytes"] == 0
                assert len(captures) == 2
            finally:
                await client.close()
                await request.aclose()

    asyncio.run(run())


def test_first_start_at_boundary_uses_committed_q_not_draft(monkeypatch):
    async def run():
        async with case(monkeypatch) as c:
            await complete(c, 0, tuple(range(8)))
            request, captures, _ = controller(c, monkeypatch)

            def forbidden(*args):
                raise AssertionError("boundary fallback ran draft")

            monkeypatch.setattr(request.pipeline.provider, "predict", forbidden)
            client = PVDShardSearchClient(c.client.base_url)
            try:
                await run_refresh(c, request, client, 4)
                assert captures == [(11,)]
                assert request.try_install({0: 4})
                await wait_acks(c.sink)
            finally:
                await client.close()
                await request.aclose()

    asyncio.run(run())


def test_controller_refuses_missing_initial_prompt(monkeypatch):
    async def run():
        async with case(monkeypatch) as c:
            with pytest.raises(InstallProtocolError, match="initial Prompt"):
                controller(c, monkeypatch)

    asyncio.run(run())


def test_cuda_controller_rejects_local_payload_override_and_cpu_controller(monkeypatch):
    async def run():
        async with case(monkeypatch) as c:
            await complete(c, 0, tuple(range(8)))
            request, _, kwargs = controller(c, monkeypatch)
            try:
                with pytest.raises(TypeError, match="CPUInstallGroup"):
                    CPUPrefetchRequest(
                        c.group,
                        request.pipeline,
                        head_mapping=kwargs["head_mapping"],
                        rank_routes=kwargs["rank_routes"],
                        max_union_tokens=2,
                    )
                with pytest.raises(ValueError, match="exactly one"):
                    await request.refresh(
                        None, query_positions=(), clients={}, pack_source=lambda: None
                    )
                kwargs["delivery"] = None
                with pytest.raises(ValueError, match="owned CUDA Delivery"):
                    CUDAPrefetchRequest(c.group, request.pipeline, **kwargs)
            finally:
                await request.aclose()

    asyncio.run(run())


def test_close_cannot_claim_success_with_quarantined_query_owner(monkeypatch):
    async def run():
        async with case(monkeypatch) as c:
            await complete(c, 0, tuple(range(8)))
            request, _, _ = controller(c, monkeypatch)
            request._session._copy_unknown = True
            with pytest.raises(InstallProtocolError, match="query ownership"):
                await request.aclose()
            assert not request.can_decode(0)

    asyncio.run(run())


def test_search_refusal_cancels_request_without_advancing_boundary(monkeypatch):
    async def run():
        async with case(monkeypatch) as c:
            await complete(c, 0, tuple(range(8)))
            request, _, _ = controller(c, monkeypatch)

            class RefusingClient:
                async def search(self, *args, **kwargs):
                    raise ValueError("stale index")

            try:
                with pytest.raises(ValueError, match="stale index"):
                    await run_refresh(c, request, RefusingClient(), 3)
                assert not request.can_decode(3)
                assert c.group.coordinator.snapshot()["installed_tokens"] == 0
                assert request._session.copy_budget.snapshot()["reservations"] == 0
            finally:
                await request.aclose()

    asyncio.run(run())
