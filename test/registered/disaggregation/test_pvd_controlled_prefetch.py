"""Controlled request epochs over real local HTTP, no Scheduler/RDMA claim."""

import asyncio
from dataclasses import replace

import pytest
import torch
from pvd_controlled_prefetch import ControlledFixture, verify_controlled_roundtrip
from sglang.srt.disaggregation.pvd.cpu_prefetch_request import LatePrefetchStart
from sglang.srt.disaggregation.pvd.prediction import (
    CommittedPrefix,
    DraftConfig,
    FakeDraftProvider,
    FakeTargetProbe,
    PredictionPipeline,
    ProbeConfig,
)
from sglang.srt.disaggregation.pvd.probe_search import (
    ProbeSearchSession,
    StaleProbeSearch,
)
from sglang.srt.disaggregation.pvd.sparse_install import (
    InstallEpoch,
    InstallProtocolError,
)
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from test_pvd_prompt_vectors import FakePool


def components(request="r"):
    pool = FakePool(layers=2, heads=2, dtype=torch.float32)
    prefix = CommittedPrefix(request, (1, 2, 3, 4, 5), 0, "initial")
    draft_config, probe_config = (
        DraftConfig("fixture", predict_tokens=2),
        ProbeConfig("target", (0, 1), head_count=4),
    )

    class NonzeroProbe(FakeTargetProbe):
        def _queries(self, prefix, positions):
            generator = torch.Generator().manual_seed(312)
            return tuple(
                replace(q, vectors=torch.randn(q.vectors.shape, generator=generator))
                for q in super()._queries(prefix, positions)
            )

    draft, probe = (
        FakeDraftProvider(draft_config, tokens=(7, 8)),
        NonzeroProbe(probe_config, head_dim=8),
    )
    return pool, prefix, PredictionPipeline(draft, probe, draft_config, probe_config)


def test_http_loop_uses_one_capture_and_one_epoch_for_both_shards():
    pool, prefix, pipeline = components()
    result = asyncio.run(verify_controlled_roundtrip(pool, prefix, pipeline))
    assert result["shared_epoch"] and result["installed_boundary"] == 4
    assert result["kv_groups"] == 4
    assert len(pipeline.provider.calls) == len(pipeline.probe.calls) == 1


class DelayedClient:
    def __init__(self, client):
        self.client = client
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    async def search(self, *args, **kwargs):
        self.entered.set()
        await self.release.wait()
        return await self.client.search(*args, **kwargs)


@pytest.mark.parametrize("boundary_start", [False, True])
def test_late_reply_waits_at_boundary_then_installs_without_recapture(boundary_start):
    async def run():
        fixture = ControlledFixture(*components())
        try:
            async with fixture.clients() as clients:
                delayed = DelayedClient(clients[1])
                prefix = fixture.refresh_prefix(4 if boundary_start else 3)
                task = asyncio.create_task(
                    fixture.request.refresh(
                        prefix,
                        query_positions=(len(prefix.tokens) - int(boundary_start),),
                        clients={0: clients[0], 1: delayed},
                        pack_source=fixture.pack_source,
                    )
                )
                try:
                    await asyncio.wait_for(delayed.entered.wait(), 5)
                    if not boundary_start:
                        assert fixture.request.can_decode(3)
                    assert not fixture.request.can_decode(4)
                    assert not fixture.request.try_install({0: 4, 1: 4})
                    assert fixture.group.coordinator.snapshot()["installed_tokens"] == 0
                    delayed.release.set()
                    epoch = await asyncio.wait_for(task, 5)
                    assert all(
                        s.operation_id == epoch.operation_id
                        for s in fixture.packed_specs
                    )
                    assert fixture.request.try_install({0: 4, 1: 4})
                    assert len(fixture.request.pipeline.provider.calls) == int(
                        not boundary_start
                    )
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        finally:
            fixture.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "reason", ["user cancelled", "prefix replaced", "Entry replaced"]
)
def test_cancel_or_replacement_discards_inflight_results(reason):
    async def run():
        fixture = ControlledFixture(*components())
        try:
            async with fixture.clients() as clients:
                delayed = DelayedClient(clients[1])
                prefix = fixture.refresh_prefix()
                task = asyncio.create_task(
                    fixture.request.refresh(
                        prefix,
                        query_positions=(len(prefix.tokens),),
                        clients={0: clients[0], 1: delayed},
                        pack_source=fixture.pack_source,
                    )
                )
                try:
                    await asyncio.wait_for(delayed.entered.wait(), 5)
                    fixture.request.cancel(reason)
                    delayed.release.set()
                    with pytest.raises(
                        (asyncio.CancelledError, StaleProbeSearch, InstallProtocolError)
                    ):
                        await asyncio.wait_for(task, 5)
                    assert fixture.packed_specs == []
                    assert fixture.group.coordinator.snapshot()["installed_tokens"] == 0
                    assert not fixture.request.can_decode(4)
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        finally:
            fixture.close()

    asyncio.run(run())


def test_new_request_does_not_change_old_inflight_window():
    async def run():
        old = ControlledFixture(*components("old"))
        new = None
        try:
            async with old.clients() as clients:
                delayed = DelayedClient(clients[1])
                prefix = old.refresh_prefix()
                task = asyncio.create_task(
                    old.request.refresh(
                        prefix,
                        query_positions=(len(prefix.tokens),),
                        clients={0: clients[0], 1: delayed},
                        pack_source=old.pack_source,
                    )
                )
                try:
                    await asyncio.wait_for(delayed.entered.wait(), 5)
                    before = old.group.coordinator.snapshot()
                    new = ControlledFixture(*components("new"))
                    new.close()
                    assert old.group.coordinator.snapshot() == before
                    delayed.release.set()
                    await asyncio.wait_for(task, 5)
                    assert old.request.try_install({0: 4, 1: 4})
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        finally:
            old.close()

    asyncio.run(run())


def test_source_index_rebuild_after_search_refuses_payload_before_install():
    async def run():
        fixture = ControlledFixture(*components())
        try:
            async with fixture.clients() as clients:

                def changed_source(rank, specs):
                    if rank == 1:
                        index = fixture.stores[1].prompt_index
                        index.close(fixture.manifest.key.transfer_id)
                        index.note_kv_readable(fixture.manifest.key.transfer_id)
                        fixture.stores[1].progress_prompt_indexes()
                    return fixture.pack_source(rank, specs)

                prefix = fixture.refresh_prefix()
                with pytest.raises(SparsePayloadError, match="index"):
                    await fixture.request.refresh(
                        prefix,
                        query_positions=(len(prefix.tokens),),
                        clients=clients,
                        pack_source=changed_source,
                    )
                assert fixture.group.coordinator.snapshot()["installed_tokens"] == 0
                assert not fixture.request.can_decode(4)
        finally:
            fixture.close()

    asyncio.run(run())


def test_next_round_uses_new_epoch_without_relabeling_previous_specs():
    async def run():
        fixture = ControlledFixture(*components())
        try:
            async with fixture.clients() as clients:
                epochs = []
                for n in (3, 7):
                    prefix = fixture.refresh_prefix(n)
                    epochs.append(
                        await fixture.request.refresh(
                            prefix,
                            query_positions=(len(prefix.tokens),),
                            clients=clients,
                            pack_source=fixture.pack_source,
                        )
                    )
                    assert fixture.request.try_install({0: n + 1, 1: n + 1})
                assert epochs[0].operation_id != epochs[1].operation_id
                assert [s.target_tokens for s in fixture.packed_specs] == [4] * 4 + [
                    8
                ] * 4
                assert {s.operation_id for s in fixture.packed_specs[:4]} == {
                    epochs[0].operation_id
                }
                assert len(fixture.request.pipeline.provider.calls) == 2
        finally:
            fixture.close()

    asyncio.run(run())


def test_first_start_past_boundary_is_refused_without_changing_clock():
    async def run():
        fixture = ControlledFixture(*components())
        try:
            before = fixture.group.coordinator.snapshot()
            prefix = fixture.refresh_prefix(5)
            with pytest.raises(LatePrefetchStart, match="past an uninstalled"):
                await fixture.request.refresh(
                    prefix,
                    query_positions=(len(prefix.tokens),),
                    clients={0: object(), 1: object()},
                    pack_source=fixture.pack_source,
                )
            assert fixture.group.coordinator.snapshot() == before
            assert not fixture.request.can_decode(4)
            assert fixture.request.pipeline.provider.calls == []
        finally:
            fixture.close()

    asyncio.run(run())


def test_boundary_first_start_uses_committed_q_without_draft(monkeypatch):
    pool, prefix, pipeline = components()

    def forbidden(*args, **kwargs):
        raise AssertionError("boundary fallback entered the predictive path")

    monkeypatch.setattr(pipeline.provider, "branch", forbidden)
    monkeypatch.setattr(pipeline.provider, "predict", forbidden)
    monkeypatch.setattr(pipeline.probe, "capture", forbidden)
    result = asyncio.run(
        verify_controlled_roundtrip(pool, prefix, pipeline, boundary_start=True)
    )
    assert result["query_source"] == "committed"
    assert result["installed_boundary"] == 4
    assert pipeline.provider.calls == []
    assert pipeline.probe.calls == [(prefix.request_id, ())]


def test_boundary_first_start_rejects_future_query_positions():
    async def run():
        fixture = ControlledFixture(*components())
        try:
            prefix = fixture.refresh_prefix(4)
            with pytest.raises(ValueError, match="declared query source"):
                await fixture.request.refresh(
                    prefix,
                    query_positions=(len(prefix.tokens),),
                    clients={0: object(), 1: object()},
                    pack_source=fixture.pack_source,
                )
            assert fixture.request.pipeline.provider.calls == []
            assert fixture.request.pipeline.probe.calls == []
            assert not fixture.request.can_decode(4)
            assert fixture.group.coordinator.snapshot()["installed_tokens"] == 0
        finally:
            fixture.close()

    asyncio.run(run())


def controlled_capture():
    fixture = ControlledFixture(*components())
    prefix = fixture.refresh_prefix()
    epoch = fixture.group.begin(3)
    session = ProbeSearchSession(
        prefix.request_id, epoch.entry_transfer_id, incarnation=epoch.incarnation
    )
    window = session.begin(
        prefix,
        target_tokens=4,
        query_positions=(len(prefix.tokens),),
        install_epoch=epoch,
    )
    routes = fixture.request._routes[0] + fixture.request._routes[1]
    prepared = session.prepare(
        window, fixture.request.pipeline, routes=routes, head_mapping=fixture.mapping
    )
    return fixture, session, window, prepared, epoch


@pytest.mark.parametrize(
    "partitions",
    [{0: (0,)}, {0: (0, 0), 1: tuple(range(1, 8))}, {0: tuple(range(8)), 1: ()}],
)
def test_partition_must_cover_queries_exactly_once(partitions):
    fixture, session, _, prepared, _ = controlled_capture()
    try:
        with pytest.raises(ValueError, match="every query exactly once"):
            session.fork_prepared(prepared, partitions)
    finally:
        session.close()
        fixture.close()


def test_fork_uses_same_window_and_parent_invalidation_rejects_all_children():
    fixture, session, window, prepared, epoch = controlled_capture()
    try:
        with pytest.raises(ValueError, match="unused capture"):
            session.fork_prepared(
                replace(prepared), {0: tuple(range(4)), 1: tuple(range(4, 8))}
            )
        children = session.fork_prepared(
            prepared, {0: tuple(range(4)), 1: tuple(range(4, 8))}
        )
        for child, part in children.values():
            assert part.window is window
            assert child.incarnation == epoch.incarnation
        session.invalidate()
        for child, _ in children.values():
            with pytest.raises(StaleProbeSearch):
                child.take_selection(window)
        with pytest.raises(StaleProbeSearch, match="replayed"):
            session.begin(
                window.prefix,
                target_tokens=4,
                query_positions=window.query_positions,
                install_epoch=epoch,
            )
    finally:
        session.close()
        fixture.close()


@pytest.mark.parametrize(
    "field", ["request_id", "incarnation", "entry_transfer_id", "target_tokens"]
)
def test_controlled_session_refuses_foreign_epoch(field):
    prefix = CommittedPrefix("r", (1, 2, 3), 1, "v")
    session = ProbeSearchSession("r", "entry", incarnation="inc")
    epoch = InstallEpoch("r", "inc", "entry", "operation", 1, 4)
    epoch = replace(epoch, **{field: 8 if field == "target_tokens" else "other"})
    with pytest.raises(StaleProbeSearch, match="exact installation epoch"):
        session.begin(
            prefix, target_tokens=4, query_positions=(3,), install_epoch=epoch
        )


def test_one_failed_search_cancels_sibling_and_never_stages_payload():
    async def run():
        fixture = ControlledFixture(*components())
        entered, drained = asyncio.Event(), asyncio.Event()

        class Blocked:
            async def search(self, *args, **kwargs):
                entered.set()
                try:
                    await asyncio.Future()
                finally:
                    drained.set()

        class Fails:
            async def search(self, *args, **kwargs):
                await entered.wait()
                raise RuntimeError("search rank failed")

        try:
            prefix = fixture.refresh_prefix()
            with pytest.raises(RuntimeError, match="search rank failed"):
                await asyncio.wait_for(
                    fixture.request.refresh(
                        prefix,
                        query_positions=(len(prefix.tokens),),
                        clients={0: Fails(), 1: Blocked()},
                        pack_source=fixture.pack_source,
                    ),
                    5,
                )
            assert drained.is_set()
            assert fixture.packed_specs == []
            assert not fixture.request.can_decode(4)
        finally:
            fixture.close()

    asyncio.run(run())


def test_second_start_during_inflight_round_is_refused_without_cancelling_it():
    async def run():
        fixture = ControlledFixture(*components())
        try:
            async with fixture.clients() as clients:
                delayed = DelayedClient(clients[1])
                prefix = fixture.refresh_prefix()
                kwargs = {
                    "query_positions": (len(prefix.tokens),),
                    "clients": {0: clients[0], 1: delayed},
                    "pack_source": fixture.pack_source,
                }
                task = asyncio.create_task(fixture.request.refresh(prefix, **kwargs))
                try:
                    await asyncio.wait_for(delayed.entered.wait(), 5)
                    before = fixture.group.coordinator.snapshot()
                    with pytest.raises(ValueError, match="one outstanding"):
                        await fixture.request.refresh(prefix, **kwargs)
                    assert fixture.group.coordinator.snapshot() == before
                    delayed.release.set()
                    await asyncio.wait_for(task, 5)
                    assert fixture.request.try_install({0: 4, 1: 4})
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        finally:
            fixture.close()

    asyncio.run(run())


def test_controlled_mode_cannot_mint_an_unbound_window_or_use_standalone_mode():
    prefix = CommittedPrefix("r", (1, 2, 3), 1, "v")
    epoch = InstallEpoch("r", "inc", "entry", "operation", 1, 4)
    controlled = ProbeSearchSession("r", "entry", incarnation="inc")
    with pytest.raises(ValueError, match="new request controller"):
        controlled.replace_entry("new-entry")
    with pytest.raises(StaleProbeSearch, match="exact installation epoch"):
        controlled.begin(prefix, target_tokens=4, query_positions=(3,))
    standalone = ProbeSearchSession("r", "entry")
    with pytest.raises(ValueError, match="standalone"):
        standalone.begin(
            prefix, target_tokens=4, query_positions=(3,), install_epoch=epoch
        )


def test_zero_lead_is_refused_at_controller_construction(monkeypatch):
    from sglang.srt.disaggregation.pvd.cpu_prefetch_request import CPUPrefetchRequest

    fixture = ControlledFixture(*components())
    try:
        original = fixture.group.coordinator.snapshot
        monkeypatch.setattr(
            fixture.group.coordinator,
            "snapshot",
            lambda: {**original(), "lead_tokens": 0},
        )
        with pytest.raises(ValueError, match="positive lead window"):
            CPUPrefetchRequest(
                fixture.group,
                fixture.request.pipeline,
                head_mapping=fixture.mapping,
                rank_routes=fixture.request._routes,
                max_union_tokens=5,
            )
    finally:
        fixture.close()
