"""CPU fake probe -> actual V HTTP/exact index. No model/GPU acceptance."""

import asyncio
from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch
from sglang.srt.disaggregation.pvd.prediction import (
    DraftConfig,
    FakeDraftProvider,
    PredictionPipeline,
    ProbeConfig,
    QueryVectors,
    TargetProbe,
    snapshot_committed,
)
from sglang.srt.disaggregation.pvd.probe_search import (
    MAX_CONCURRENT_SHARD_SEARCHES,
    ProbeSearchRoute,
    ProbeSearchSession,
    StaleProbeSearch,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.search_client import (
    PVDShardSearchClient,
    SearchRefused,
    ShardSearchResult,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_prompt_index import shard_client
from test_pvd_search_client import fixture


class ScratchProbe(TargetProbe):
    """Owns budgeted CPU scratch, poisons it on exit to expose borrowed Q."""

    def __init__(self, vector, *, changes=None, fail=False, head_count=1):
        self.vector = vector
        self.changes = changes or {}
        self.fail = fail
        self.head_count = head_count
        self.budget = TransferBudget(4096, 1)
        self.tensor = None
        self.closed = 0

    @contextmanager
    def branch(self):
        self.budget.reserve("probe", 2 * self.head_count * len(self.vector) * 4, 1)
        try:
            self.tensor = torch.tensor(self.vector).repeat(2, self.head_count, 1)
            torch.rand(1)  # scope entry/exit must be isolated too
            yield
        finally:
            if self.tensor is not None:
                self.tensor.fill_(float("nan"))
            self.tensor = None
            self.budget.release("probe")
            self.closed += 1
            torch.rand(1)

    def capture(self, prefix, prediction):
        assert not torch.is_grad_enabled()
        assert self.budget.snapshot()["used_inflight"] == 1
        torch.rand(1)
        if self.fail:
            raise RuntimeError("capture failed")
        start = len(prefix.tokens)
        query = QueryVectors(
            vector_space="target/model-8b",
            version="probe-output-v1",
            layer=0,
            head_start=0,
            head_count=self.head_count,
            positions=(start, start + 1),
            valid_length=2,
            vectors=self.tensor,
            prefix_version=prefix.version,
            positional_encoding="rope_applied",
            request_id=prefix.request_id,
        )
        return (replace(query, **self.changes),)


def setup(*, changes=None, fail=False, heads=1):
    index, store, identity, rows, scope = fixture()
    draft_config = DraftConfig("configurable/draft", predict_tokens=2)
    probe = ScratchProbe(rows[0], changes=changes, fail=fail, head_count=heads)
    pipeline = PredictionPipeline(
        FakeDraftProvider(draft_config, tokens=(31, 32)),
        probe,
        draft_config,
        ProbeConfig(identity.vector_space, (0,), head_count=heads),
    )
    session = ProbeSearchSession("request", identity.entry_transfer_id)
    prefix = snapshot_committed("request", [10, 11, 12, 13], 2, "prefix-v1")
    window = session.begin(prefix, target_tokens=4, query_positions=(5,))
    route = ProbeSearchRoute(0, identity, scope, 1)
    return index, store, session, window, pipeline, probe, route


def prepare(session, window, pipeline, route, heads=1):
    return session.prepare(
        window, pipeline, routes=(route,), head_mapping=QueryHeadMapping(heads, 1)
    )


def test_full_model_route_count_is_bounded_by_rows_not_sixty_four():
    _, _, session, window, pipeline, probe, route = setup(heads=784)
    probe.budget = TransferBudget(131072, 1)
    routes = tuple(replace(route, query_head=head) for head in range(784))
    prepared = session.prepare(
        window, pipeline, routes=routes, head_mapping=QueryHeadMapping(784, 1)
    )
    assert len(prepared.queries) == 784
    assert probe.closed == 1


def test_route_position_product_is_refused_before_probe_allocation():
    _, _, session, window, pipeline, probe, route = setup(heads=2049)
    session.invalidate()
    window = session.begin(window.prefix, target_tokens=4, query_positions=(4, 5))
    routes = tuple(replace(route, query_head=head) for head in range(2049))
    with pytest.raises(ValueError, match="prepared-query row bound"):
        session.prepare(
            window, pipeline, routes=routes, head_mapping=QueryHeadMapping(2049, 1)
        )
    assert probe.closed == 0


def test_probe_http_roundtrip_isolated_owned_and_consumed_once():
    async def run():
        _, store, session, window, pipeline, probe, route = setup()
        rng = torch.random.get_rng_state().clone()
        prepared = prepare(session, window, pipeline, route)
        assert torch.equal(rng, torch.random.get_rng_state())
        assert probe.tensor is None and probe.closed == 1
        assert probe.budget.snapshot()["used_staging_bytes"] == 0
        assert window.prefix.tokens == (10, 11, 12, 13)
        assert window.prefix.committed_position == 2
        assert len(prepared.queries[0].rows) == 1  # requested position only
        async with shard_client(store) as http:
            client = PVDShardSearchClient(str(http.make_url("")))
            try:
                await session.search(prepared, client)
                session.observe(4)  # normal committed progress is not invalidation
                result = session.take_selection(window)
                assert result.selections[0].token_ids == (3,)
                assert result.queries[0].query_version == "probe-output-v1"
                assert result.window.target_tokens == 4
                with pytest.raises(StaleProbeSearch):
                    session.take_selection(window)
            finally:
                await client.close()

    asyncio.run(run())


def test_explicit_index_ready_wait_reuses_one_immutable_probe():
    class InitiallyUnready(PVDShardSearchClient):
        def __init__(self, url):
            super().__init__(url)
            self.calls = []

        async def search(self, identity, *, queries, top_k, scope):
            self.calls.append((identity, queries))
            if len(self.calls) < 3:
                raise SearchRefused(400, "index_not_ready", "building")
            return await super().search(
                identity, queries=queries, top_k=top_k, scope=scope
            )

    async def run():
        _, store, session, window, pipeline, probe, route = setup()
        prepared = prepare(session, window, pipeline, route)
        async with shard_client(store) as http:
            client = InitiallyUnready(str(http.make_url("")))
            try:
                await session.search(prepared, client, index_ready_wait_seconds=1.0)
                assert len(client.calls) == 3
                assert client.calls[0] == client.calls[1] == client.calls[2]
                assert probe.closed == 1  # No draft/probe recapture on retry.
                assert session.take_selection(window).selections[0].token_ids == (3,)
            finally:
                await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("code", ["request_refused", "index_not_ready"])
def test_index_ready_wait_never_retries_fatal_or_expired_refusal(code):
    class Refusing(PVDShardSearchClient):
        calls = 0

        async def search(self, *args, **kwargs):
            self.calls += 1
            raise SearchRefused(400, code, "refused")

    async def run():
        _, _, session, window, pipeline, _, route = setup()
        prepared = prepare(session, window, pipeline, route)
        client = Refusing("http://127.0.0.1:1")
        with pytest.raises(SearchRefused):
            await session.search(
                prepared,
                client,
                index_ready_wait_seconds=1.0 if code == "request_refused" else 0.0,
            )
        assert client.calls == 1
        with pytest.raises(StaleProbeSearch):
            session.take_selection(window)
        await client.close()

    asyncio.run(run())


def test_index_ready_wait_bounds_a_slow_successful_http_attempt():
    class SlowSuccess(PVDShardSearchClient):
        calls = 0

        async def search(self, identity, *, queries, top_k, scope):
            self.calls += 1
            await asyncio.sleep(0.1)
            return await super().search(
                identity, queries=queries, top_k=top_k, scope=scope
            )

    async def run():
        _, store, session, window, pipeline, _, route = setup()
        prepared = prepare(session, window, pipeline, route)
        async with shard_client(store) as http:
            client = SlowSuccess(str(http.make_url("")))
            try:
                with pytest.raises(TimeoutError, match="index readiness deadline"):
                    await session.search(
                        prepared, client, index_ready_wait_seconds=0.01
                    )
                assert client.calls == 1
                with pytest.raises(StaleProbeSearch):
                    session.take_selection(window)
            finally:
                await client.close()

    asyncio.run(run())


def test_index_ready_wait_expires_after_retryable_capacity_refusal():
    class AtCapacity(PVDShardSearchClient):
        calls = 0

        async def search(self, *args, **kwargs):
            self.calls += 1
            raise SearchRefused(507, "index_capacity", "index budget full")

    async def run():
        _, _, session, window, pipeline, _, route = setup()
        prepared = prepare(session, window, pipeline, route)
        client = AtCapacity("http://127.0.0.1:1")
        try:
            with pytest.raises(TimeoutError, match="index readiness deadline"):
                await session.search(prepared, client, index_ready_wait_seconds=0.01)
            assert client.calls == 1
            with pytest.raises(StaleProbeSearch):
                session.take_selection(window)
        finally:
            await client.close()

    asyncio.run(run())


def test_index_ready_wait_cancel_does_not_publish_selection():
    class Waiting(PVDShardSearchClient):
        def __init__(self):
            super().__init__("http://127.0.0.1:1")
            self.entered = asyncio.Event()

        async def search(self, *args, **kwargs):
            self.entered.set()
            await asyncio.Event().wait()

    async def run():
        _, _, session, window, pipeline, _, route = setup()
        prepared = prepare(session, window, pipeline, route)
        client = Waiting()
        task = asyncio.create_task(
            session.search(prepared, client, index_ready_wait_seconds=1.0)
        )
        await client.entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(StaleProbeSearch):
            session.take_selection(window)
        await client.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"prefix_version": None}, "prefix identity"),
        ({"request_id": None}, "request identity"),
        ({"request_id": "another-request"}, "another request"),
        ({"prefix_version": "old"}, "stale prefix"),
        ({"positional_encoding": None}, "encoding mismatch"),
        ({"positional_encoding": "none"}, "encoding mismatch"),
        ({"positions": (3, 4)}, "outside the predicted"),
        ({"valid_length": 1}, "padding"),
        ({"vectors": torch.zeros(2, 8)}, "CPU float Q"),
        ({"vectors": torch.zeros(2, 1, 7)}, "CPU float Q"),
        ({"vectors": torch.zeros(2, 1, 8, dtype=torch.int64)}, "CPU float Q"),
        ({"vectors": torch.full((2, 1, 8), float("nan"))}, "finite"),
        ({"vectors": torch.full((2, 1, 8), 1e100, dtype=torch.float64)}, "finite"),
        ({"head_start": 1}, "unrequested query heads"),
        ({"layer": 1}, "unrequested layer"),
        ({"vector_space": "draft"}, "but V searches"),
    ],
)
def test_bad_probe_is_refused_and_releases_scratch(changes, error):
    _, _, session, window, pipeline, probe, route = setup(changes=changes)
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match=error):
        prepare(session, window, pipeline, route)
    assert probe.closed == 1
    assert probe.budget.snapshot()["used_staging_bytes"] == 0
    assert torch.equal(rng, torch.random.get_rng_state())
    with pytest.raises(StaleProbeSearch):
        session.take_selection(window)


def test_exception_in_capture_releases_scope_and_restores_rng():
    _, _, session, window, pipeline, probe, route = setup(fail=True)
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="capture failed"):
        prepare(session, window, pipeline, route)
    assert probe.closed == 1
    assert probe.budget.snapshot()["reservations"] == 0
    assert torch.equal(rng, torch.random.get_rng_state())


class PausedClient(PVDShardSearchClient):
    def __init__(self, url):
        super().__init__(url)
        self.arrived = asyncio.Event()
        self.resume = asyncio.Event()

    async def search(self, *args, **kwargs):
        result = await super().search(*args, **kwargs)
        self.arrived.set()
        await self.resume.wait()
        return result


@pytest.mark.parametrize(
    "action", ["close", "replace", "same_entry", "invalidate", "expire", "cancel"]
)
def test_late_response_cannot_become_a_selection(action):
    async def run():
        _, store, session, window, pipeline, probe, route = setup()
        prepared = prepare(session, window, pipeline, route)
        async with shard_client(store) as http:
            client = PausedClient(str(http.make_url("")))
            task = asyncio.create_task(session.search(prepared, client))
            try:
                await asyncio.wait_for(client.arrived.wait(), timeout=2)
                assert probe.budget.snapshot()["used_staging_bytes"] == 0
                if action == "close":
                    session.close()
                elif action == "replace":
                    session.replace_entry("new-entry")
                elif action == "same_entry":
                    session.replace_entry(window.entry_transfer_id)
                elif action == "invalidate":
                    session.invalidate()
                elif action == "expire":
                    with pytest.raises(StaleProbeSearch, match="expired"):
                        session.observe(5)
                else:
                    task.cancel()
                client.resume.set()
                with pytest.raises(
                    asyncio.CancelledError if action == "cancel" else StaleProbeSearch
                ):
                    await task
                with pytest.raises(StaleProbeSearch):
                    session.take_selection(window)
            finally:
                client.resume.set()
                await client.close()

    asyncio.run(run())


def test_new_request_does_not_invalidate_another_requests_window():
    async def run():
        _, store, session, window, pipeline, _, route = setup()
        prepared = prepare(session, window, pipeline, route)
        async with shard_client(store) as http:
            client = PausedClient(str(http.make_url("")))
            task = asyncio.create_task(session.search(prepared, client))
            try:
                await asyncio.wait_for(client.arrived.wait(), timeout=2)
                other = ProbeSearchSession("newcomer", "entry-2")
                other.begin(
                    snapshot_committed("newcomer", (1, 2), 0, "p"),
                    target_tokens=8,
                    query_positions=(2,),
                )
                other.close()
                session.observe(3)
                client.resume.set()
                await task
                assert session.take_selection(window).selections[0].token_ids == (3,)
            finally:
                client.resume.set()
                await client.close()

    asyncio.run(run())


def test_request_id_reuse_and_forged_prepared_operation_are_refused():
    _, _, session, window, pipeline, _, route = setup()
    prepared = prepare(session, window, pipeline, route)
    other = ProbeSearchSession(session.request_id, session.entry_transfer_id)
    assert other.incarnation != session.incarnation
    with pytest.raises(StaleProbeSearch):
        other.take_selection(window)
    with pytest.raises(StaleProbeSearch):
        session.take_selection(replace(window))

    async def run():
        with pytest.raises(ValueError, match="unused prepared"):
            await session.search(replace(prepared), None)

    asyncio.run(run())


def test_multiple_q_heads_coalesce_with_explicit_provenance():
    async def run():
        _, store, session, window, pipeline, _, route = setup(heads=2)
        routes = (route, replace(route, query_head=1))
        prepared = session.prepare(
            window, pipeline, routes=routes, head_mapping=QueryHeadMapping(2, 1)
        )
        async with shard_client(store) as http:
            client = PVDShardSearchClient(str(http.make_url("")))
            try:
                await session.search(prepared, client)
                result = session.take_selection(window)
                assert len(result.selections) == 1
                assert result.query_groups == ((0, 1),)
                assert [q.route.query_head for q in result.queries] == [0, 1]
                assert "index_version" not in result.selections[0].validated
                assert len({s.index_version for s in result.selections}) == 1
            finally:
                await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("fail_head", [None, 1])
def test_group_searches_are_version_pinned_bounded_and_cancelled(fail_head):
    async def run():
        _, _, session, window, pipeline, _, route = setup(heads=12)
        routes = tuple(
            replace(
                route,
                query_head=head,
                identity=replace(route.identity, kv_head=head),
            )
            for head in range(12)
        )
        prepared = session.prepare(
            window, pipeline, routes=routes, head_mapping=QueryHeadMapping(12, 12)
        )

        class RecordingClient:
            def __init__(self):
                self.calls = []
                self.active = 0
                self.peak = 0

            async def search(self, identity, *, queries, top_k, scope):
                self.calls.append(identity)
                self.active += 1
                self.peak = max(self.peak, self.active)
                try:
                    await asyncio.sleep(0.01 if identity.kv_head != 2 else 0.05)
                    if identity.kv_head == fail_head:
                        raise RuntimeError("search failed")
                    return ShardSearchResult(
                        identity=identity,
                        index_version="index-v1",
                        id_mapping_version="mapping-v1",
                        token_ids=(0,),
                        page_ids=(0,),
                        scores=(1.0,),
                        metric=scope.metric,
                        validated=(),
                    )
                finally:
                    self.active -= 1

        client = RecordingClient()
        if fail_head is not None:
            with pytest.raises(RuntimeError, match="search failed"):
                await session.search(prepared, client)
            with pytest.raises(ValueError):
                session.take_selection(window)
        else:
            await session.search(prepared, client)
            result = session.take_selection(window)
            assert len(result.selections) == 12
            assert tuple(s.identity.kv_head for s in result.selections) == tuple(
                range(12)
            )
        assert client.active == 0
        assert 1 < client.peak <= MAX_CONCURRENT_SHARD_SEARCHES
        assert client.calls[0].expected_index_version is None
        assert all(
            call.expected_index_version == "index-v1"
            and call.expected_id_mapping_version == "mapping-v1"
            for call in client.calls[1:]
        )

    asyncio.run(run())


def test_rebuild_between_groups_discards_the_entire_selection():
    async def run():
        index, store, session, window, pipeline, _, route = setup(heads=2)
        prepared = session.prepare(
            window,
            pipeline,
            routes=(route, replace(route, query_head=1, top_k=2)),
            head_mapping=QueryHeadMapping(2, 1),
        )

        class RebuildingClient(PVDShardSearchClient):
            async def search(self, *args, **kwargs):
                result = await super().search(*args, **kwargs)
                index.close(window.entry_transfer_id)
                index.note_kv_readable(window.entry_transfer_id)
                store.progress_prompt_indexes()
                return result

        async with shard_client(store) as http:
            client = RebuildingClient(str(http.make_url("")))
            try:
                with pytest.raises(SearchRefused):
                    await session.search(prepared, client)
                with pytest.raises(StaleProbeSearch):
                    session.take_selection(window)
            finally:
                await client.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "change,error",
    [
        ({"entry_transfer_id": "wrong"}, "another Entry"),
        ({"vector_space": "draft"}, "vector space"),
        ({"positional_encoding": "none"}, "post-RoPE"),
        ({"kv_head": 1}, "mapping mismatch"),
    ],
)
def test_bad_route_is_rejected_before_probe_allocation(change, error):
    _, _, session, window, pipeline, probe, route = setup()
    route = replace(route, identity=replace(route.identity, **change))
    with pytest.raises(ValueError, match=error):
        prepare(session, window, pipeline, route)
    assert probe.closed == 0 and probe.tensor is None


@pytest.mark.parametrize("positions", [(), (4, 4), (5, 4), (3,), (True,), [4]])
def test_window_positions_are_explicit_and_not_guessed(positions):
    session = ProbeSearchSession("r", "entry")
    prefix = snapshot_committed("r", (1, 2, 3, 4), 2, "prefix")
    with pytest.raises(ValueError, match="query positions"):
        session.begin(prefix, target_tokens=4, query_positions=positions)


def test_cpu_bridge_refuses_a_gpu_configuration_before_running_provider():
    _, _, session, window, pipeline, probe, route = setup()
    pipeline.draft_config = replace(pipeline.draft_config, device="cuda:0")
    with pytest.raises(ValueError, match="CPU execution"):
        prepare(session, window, pipeline, route)
    assert probe.closed == 0


def test_adapter_scopes_close_when_draft_raises_before_probe_capture():
    _, _, session, window, pipeline, probe, route = setup()

    class FailingDraft(FakeDraftProvider):
        closed = False

        @contextmanager
        def branch(self):
            torch.rand(1)
            try:
                yield
            finally:
                self.closed = True
                torch.rand(1)

        def predict(self, prefix, max_tokens):
            raise RuntimeError("draft failed")

    provider = FailingDraft(pipeline.draft_config)
    pipeline.provider = provider
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="draft failed"):
        prepare(session, window, pipeline, route)
    assert provider.closed and probe.closed == 1
    assert probe.budget.snapshot()["reservations"] == 0
    assert torch.equal(rng, torch.random.get_rng_state())


def test_missing_requested_layer_is_not_a_partial_success():
    _, _, session, window, pipeline, probe, route = setup()
    pipeline.probe_config = replace(pipeline.probe_config, layers=(0, 1))
    with pytest.raises(ValueError, match="omitted a requested layer"):
        prepare(session, window, pipeline, route)
    assert probe.closed == 1


def test_failed_search_clears_the_window_and_can_be_restarted_explicitly():
    async def run():
        index, store, session, window, pipeline, _, route = setup()
        prepared = prepare(session, window, pipeline, route)
        index.close(window.entry_transfer_id)
        index.note_kv_readable(window.entry_transfer_id)
        async with shard_client(store) as http:
            client = PVDShardSearchClient(str(http.make_url("")))
            try:
                with pytest.raises(SearchRefused) as caught:
                    await session.search(prepared, client)
                assert caught.value.retryable
                replacement = session.begin(
                    window.prefix, target_tokens=4, query_positions=(5,)
                )
                assert replacement.operation_id != window.operation_id
                with pytest.raises(StaleProbeSearch):
                    await session.search(prepared, client)
                store.progress_prompt_indexes()
                await session.search(
                    prepare(session, replacement, pipeline, route), client
                )
                assert session.take_selection(replacement).selections[0].token_ids == (
                    3,
                )
            finally:
                await client.close()

    asyncio.run(run())


def test_ready_result_is_still_discarded_if_entry_changes_before_consumption():
    async def run():
        _, store, session, window, pipeline, _, route = setup()
        prepared = prepare(session, window, pipeline, route)
        async with shard_client(store) as http:
            client = PVDShardSearchClient(str(http.make_url("")))
            try:
                await session.search(prepared, client)
                session.replace_entry("replacement")
                with pytest.raises(StaleProbeSearch):
                    session.take_selection(window)
            finally:
                await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("target", [0, 1, 2, True, 2.5, -1])
def test_bootstrap_or_nonfuture_boundary_is_not_a_periodic_probe(target):
    session = ProbeSearchSession("r", "entry")
    with pytest.raises(ValueError):
        session.begin(
            snapshot_committed("r", (1, 2, 3), 2, "p"),
            target_tokens=target,
            query_positions=(3,),
        )


def test_no_parallel_window_and_no_implicit_committed_counter_regression():
    _, _, session, window, _, _, _ = setup()
    with pytest.raises(ValueError, match="outstanding"):
        session.begin(window.prefix, target_tokens=4, query_positions=(5,))
    with pytest.raises(ValueError, match="regressed"):
        session.observe(1)
    session.observe(3)
    session.invalidate()
    with pytest.raises(StaleProbeSearch, match="regressed"):
        session.begin(window.prefix, target_tokens=4, query_positions=(5,))
