"""Exact model-bank binding and Req bridge; real model execution is a smoke gate."""

import asyncio
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
from pvd_controlled_prefetch import ControlledFixture
from sglang.srt.disaggregation.pvd.cpu_batch_dispatch import CPUBatchDispatcher
from sglang.srt.disaggregation.pvd.cpu_batch_forward import CPUBatchForwardExecutor
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import (
    CPUDecodeLifecycle,
    LifecycleError,
    TargetExecutionArbiter,
)
from sglang.srt.disaggregation.pvd.cpu_rank_batch import CPURankBatchDispatcher
from sglang.srt.disaggregation.pvd.cpu_runtime_group import CPURuntimeInstallGroup
from sglang.srt.disaggregation.pvd.cpu_schedule_bridge import CPUScheduleBridge
from sglang.srt.disaggregation.pvd.rank_install_wire import RankInstallMessage
from sglang.srt.disaggregation.pvd.sparse_install import (
    CPUInstalledPromptView,
    InstallProtocolError,
)
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVPayload
from test_pvd_controlled_prefetch import components
from test_pvd_cpu_batch_dispatch import results
from test_pvd_cpu_schedule_bridge import append, req
from test_pvd_sparse_install import cpu_group, payload


@contextmanager
def setup():
    arbiter = TargetExecutionArbiter()
    d = CPURankBatchDispatcher(arbiter, max_requests=2)
    fixtures, lives = [], []
    try:
        for name in ("a", "b"):
            f = ControlledFixture(*components(name), rank_runtime=True)
            fixtures.append(f)
            life = CPUDecodeLifecycle(name, f.prefix.tokens, 9, arbiter=arbiter)
            life.admit(f.request)
            lives.append(life)
        yield d, lives, fixtures
    finally:
        if d._ticket is not None:
            d.fail(d._ticket, "drained test fixture cleanup")
        for f in fixtures:
            f.close()


def bridge_for(d, lives, ticket):
    # Explicit double for executor completion; no claim of model execution here.
    e = CPUBatchForwardExecutor.__new__(CPUBatchForwardExecutor)
    e.dispatcher, e._storage = d, dict(zip(lives, (1, 2), strict=True))
    e._completed_operation = None
    batch = NS(
        reqs=[req(life, slot) for life, slot in e._storage.items()],
        device="cpu",
        enable_overlap=False,
        forward_mode=NS(is_decode=lambda: True),
        spec_algorithm=NS(is_none=lambda: True),
        is_spec_v2=False,
    )
    processor = NS(enable_overlap=False, enable_overlap_mlx=False)
    result = NS(next_token_ids=[31, 41], copy_done=None)
    return e, batch, CPUScheduleBridge(e, batch, ticket), processor, result


def test_exact_permits_are_required_by_the_banks_actually_read_by_attention():
    with setup() as (d, lives, fixtures):
        view = CPUInstalledPromptView(fixtures[0].group, 0)
        with pytest.raises(InstallProtocolError, match="exact runtime"), view.read():
            pass
        ticket = d.begin(lives)
        for life, f, permit in zip(
            lives, fixtures, d._rank_ticket.members, strict=True
        ):
            assert permit.identity == f.group.coordinator.identity
            assert permit.committed_tokens == life.committed_tokens
            assert f.group.runtime.exchange.coordinator is f.group.coordinator
            with CPUInstalledPromptView(f.group, 0).read() as groups:
                assert all(
                    spec.operation_id == permit.installed_epoch.operation_id
                    for spec, _ in groups.values()
                )
            with pytest.raises(InstallProtocolError, match="drain before close"):
                f.group.close()  # marks cancelled; must NOT free owned bank
        assert d.complete(ticket, results(ticket)) == ()
        assert not d.arbiter.busy


def test_successful_req_writer_is_observed_once_and_holds_runtime_through_commit():
    with setup() as (d, lives, fixtures):
        ticket = d.begin(lives)
        e, batch, bridge, processor, result = bridge_for(d, lives, ticket)
        e._completed_operation = ticket.operation_id
        with bridge.processing(processor, batch, result):
            append(bridge, batch, result)
            assert [life.outputs for life in lives] == [(9,), (9,)]
            assert d.arbiter.busy
            assert all(f.group.runtime._forward is not None for f in fixtures)
        assert [life.outputs for life in lives] == [(9, 31), (9, 41)]
        assert not d.arbiter.busy
        assert all(f.group.runtime._forward is None for f in fixtures)
        with (
            pytest.raises(LifecycleError, match="replayed"),
            bridge.processing(processor, batch, result),
        ):
            pass


def test_slot_unbinding_waits_for_rank_result_scope_not_only_cpu_ticket():
    with setup() as (d, lives, fixtures):
        ticket = d.begin(lives)
        executor, _, _, _, _ = bridge_for(d, lives, ticket)
        lives[0].terminate("client cancelled")
        with d.result_scope(ticket):
            CPUBatchDispatcher.complete(d, ticket, results(ticket))
            assert lives[0]._permit is None
            assert fixtures[0].group.runtime._forward is not None
            assert d.arbiter.busy
            with pytest.raises(LifecycleError, match="result scope"):
                executor.unregister_storage(lives[0])
            assert executor._storage[lives[0]] == 1
        executor.unregister_storage(lives[0])
        executor.unregister_storage(lives[0])  # successful retirement is idempotent
        assert lives[0] not in executor._storage
        assert executor._storage[lives[1]] == 2


@pytest.mark.parametrize(
    "failure", ["cancel", "loss", "retract", "bad_event", "timeout"]
)
def test_failed_runtime_filters_its_req_before_authoritative_write(failure):
    with setup() as (d, lives, fixtures):
        ticket = d.begin(lives)
        e, batch, bridge, processor, result = bridge_for(d, lives, ticket)
        e._completed_operation = ticket.operation_id
        runtime = fixtures[0].group.runtime
        if failure == "cancel":
            runtime.cancel()
        elif failure == "loss":
            runtime.peer_lost(peer_rank=0, peer_epoch="cpu-model-worker-0")
        elif failure == "bad_event":
            runtime.post(b"{}", peer_rank=0, peer_epoch="cpu-model-worker-0")
        elif failure == "timeout":
            runtime._deadline = 0  # deterministic already-expired control deadline
        else:
            batch.reqs[0].is_retracted = True
        assert d.arbiter.busy and runtime._forward is not None
        with bridge.processing(processor, batch, result):
            assert not bridge.accepts(batch.reqs[0])
            assert bridge.accepts(batch.reqs[1])
            append(bridge, batch, result)
        assert tuple(batch.reqs[0].output_ids) == lives[0].outputs == (9,)
        assert tuple(batch.reqs[1].output_ids) == lives[1].outputs == (9, 41)
        assert lives[0].state == "aborted" and not d.arbiter.busy


def test_early_result_does_not_release_either_cpu_or_rank_execution():
    with setup() as (d, lives, fixtures):
        ticket = d.begin(lives)
        _, batch, bridge, processor, result = bridge_for(d, lives, ticket)
        with (
            pytest.raises(LifecycleError, match="has not completed"),
            bridge.processing(processor, batch, result),
        ):
            pass
        assert d.arbiter.busy and all(life._permit for life in lives)
        assert all(f.group.runtime._forward for f in fixtures)
        bridge.fail_after_execution("actual synchronous execution unwound")
        assert all(life.state == "aborted" for life in lives)
        assert all(f.group.runtime._forward is None for f in fixtures)


@pytest.mark.parametrize("failure", ["writer", "missing", "changed_prefix"])
def test_bad_result_aborts_both_ownership_layers_without_rollback(failure):
    with setup() as (d, lives, fixtures):
        ticket = d.begin(lives)
        e, batch, bridge, processor, result = bridge_for(d, lives, ticket)
        e._completed_operation = ticket.operation_id
        if failure == "changed_prefix":
            batch.reqs[0].origin_input_ids[0] += 1
        with (
            pytest.raises((LifecycleError, RuntimeError)),
            bridge.processing(processor, batch, result),
        ):
            if failure == "writer":
                batch.reqs[0].output_ids.append(31)
                raise RuntimeError("partial stream failed")
        assert all(life.state == "aborted" for life in lives)
        assert all(life.outputs == (9,) for life in lives)
        assert not d.arbiter.busy
        assert all(f.group.runtime._forward is None for f in fixtures)
        if failure == "writer":
            assert tuple(batch.reqs[0].output_ids) == (9, 31)


def test_changed_group_binding_cannot_read_or_complete_another_request():
    with setup() as (d, lives, fixtures):
        ticket = d.begin(lives)
        original = lives[0].controller.group
        try:
            lives[0].controller.group = fixtures[1].group
            with pytest.raises(LifecycleError, match="binding changed"):
                d._match(ticket)
            assert d.arbiter.busy
        finally:
            lives[0].controller.group = original
        d.complete(ticket, results(ticket))


def test_same_round_number_with_foreign_operation_is_not_an_installed_bank():
    with setup() as (d, lives, fixtures):
        ticket = d.begin(lives)
        runtime = fixtures[0].group.runtime
        original = runtime._forward
        runtime._forward = replace(
            original,
            installed_epoch=replace(original.installed_epoch, operation_id="other"),
        )
        try:
            with (
                pytest.raises(InstallProtocolError, match="exact runtime"),
                CPUInstalledPromptView(fixtures[0].group, 0).read(),
            ):
                pass
        finally:
            runtime._forward = original
        d.complete(ticket, results(ticket))


def test_refresh_cannot_install_until_readers_and_result_scope_drain():
    with setup() as (d, lives, fixtures):
        for _ in range(3):
            ticket = d.begin(lives)
            d.complete(ticket, results(ticket))
        group = fixtures[0].group
        epoch = group.begin(3)
        ticket = d.begin(lives)
        for rank in group._banks:
            with group.read(rank, 3) as groups:
                payloads = [
                    SparseKVPayload(
                        replace(
                            spec,
                            operation_id=epoch.operation_id,
                            target_tokens=4,
                            token_ids=(0,),
                        ),
                        data[:, :1].clone(),
                    )
                    for spec, data in groups.values()
                ]
                try:
                    group.stage(epoch, rank, payloads)
                    assert not group.try_install(epoch, {0: 4, 1: 4})
                finally:
                    for payload in payloads:
                        payload.close()
        e, batch, bridge, processor, result = bridge_for(d, lives, ticket)
        e._completed_operation = ticket.operation_id
        with bridge.processing(processor, batch, result):
            assert not group.try_install(epoch, {0: 4, 1: 4})
            assert group.coordinator.snapshot()["installed_tokens"] == 0
            append(bridge, batch, result)
        assert group.try_install(epoch, {0: 4, 1: 4})
        assert group.runtime.can_decode(4)
        assert fixtures[1].group.coordinator.snapshot()["installed_tokens"] == 0


def test_missing_resumed_blocks_group_admission_and_delivery_install_ack(monkeypatch):
    _, banks, budgets = cpu_group()
    group = CPURuntimeInstallGroup(
        banks,
        interval=4,
        lead_tokens=1,
        peer_epochs={r: f"worker-{r}" for r in banks},
        timeout_seconds=30,
        max_pending_events=32,
        max_pending_bytes=65536,
    )
    post = group._post

    def drop_resumed(rank, raw):
        if rank == 1 and RankInstallMessage.decode(raw).kind == "resumed":
            return
        post(rank, raw)

    monkeypatch.setattr(group, "_post", drop_resumed)
    try:
        epoch = group.begin(0)
        for rank in banks:
            packed = payload(epoch, rank)
            try:
                group.stage(epoch, rank, [packed])
            finally:
                packed.close()
        assert not group.try_install(epoch, {0: 0, 1: 0})
        assert group.coordinator.snapshot()["installed_tokens"] == 0
        assert group.runtime.snapshot()["phase"] == "resuming"
        assert not group.can_decode(0)
        assert all(not group.installation_complete(r) for r in group._receipts.values())
        # Simulate retry delivery of the exact previously-issued RESUME command.
        commands = group.runtime.exchange.resume_commands(epoch)
        post(1, group._peers[1].command(commands[1]))
        group.progress()
        assert group.can_decode(0)
        assert all(group.installation_complete(r) for r in group._receipts.values())
    finally:
        group.close()
        assert all(b.snapshot()["used_staging_bytes"] == 0 for b in budgets.values())


@pytest.mark.parametrize("use_driver", [False, True])
def test_lifecycle_can_finalize_refresh_after_delayed_resume_without_resetting_clock(
    use_driver,
):
    async def run():
        with setup() as (d, lives, fixtures):
            old, group = lives[0], fixtures[0].group
            for _ in range(3):
                ticket = d.begin(lives)
                d.complete(ticket, results(ticket))
            async with fixtures[0].clients() as clients:
                if use_driver:
                    from sglang.srt.disaggregation.pvd.cpu_refresh_driver import (
                        CPURefreshDriver,
                    )

                    driver = CPURefreshDriver(d.arbiter)
                    driver.register(
                        old,
                        clients=clients,
                        pack_source=fixtures[0].pack_source,
                        timeout_seconds=30,
                    )
                    task = driver.progress().launched[0].task
                else:
                    task = old.launch_refresh(
                        query_positions=(len(old.snapshot().tokens),),
                        clients=clients,
                        pack_source=fixtures[0].pack_source,
                        timeout_seconds=30,
                    )
                epoch = await task
                ticket = d.begin(lives)
                d.complete(ticket, results(ticket))
                assert old.committed_tokens == 4
                post, dropped = group._post, []

                def drop(rank, raw):
                    if rank == 1 and RankInstallMessage.decode(raw).kind == "resumed":
                        dropped.append(raw)
                    else:
                        post(rank, raw)

                group._post = drop
                try:
                    assert not old.try_install({0: 4, 1: 4})
                finally:
                    group._post = post
                assert group.coordinator.snapshot()["next_boundary"] == 8
                assert fixtures[0].request._ready is epoch
                assert not old.can_decode()
                post(1, dropped[0])
                if use_driver:
                    assert driver.progress().installed == ("a",)
                else:
                    assert old.try_install({0: 4, 1: 4})
                assert old.committed_tokens == 4 and old.can_decode()
                assert fixtures[0].request._ready is None and old._deadline is None
                if use_driver:
                    await driver.close()

    asyncio.run(run())
