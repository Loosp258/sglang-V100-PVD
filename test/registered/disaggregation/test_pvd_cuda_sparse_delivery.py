"""Actual localhost HTTP and delayed fake writes; CUDA placement is substituted."""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.cpu_sparse_delivery import CPUSparseDelivery
from sglang.srt.disaggregation.pvd.cuda_runtime_group import CUDARuntimeInstallGroup
from sglang.srt.disaggregation.pvd.cuda_sparse_delivery import (
    CUDAReceiveRoute,
    CUDASparseDelivery,
)
from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import CUDASparseReceiveRegistry
from sglang.srt.disaggregation.pvd.cuda_working_set import CUDASparseWorkingSet
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError
from sglang.srt.disaggregation.pvd.sparse_receiver import SparseReceiveError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_cpu_sparse_delivery import wait_acks
from test_pvd_sparse_receiver import receiving


@asynccontextmanager
async def case(monkeypatch):
    async with receiving() as c:
        events = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        registry = CUDASparseReceiveRegistry(
            c.engine, TransferBudget(65536, 8), receiver_epoch="D-cuda", device="cuda:0"
        )
        registry.device = torch.device("cpu")
        registry.ordering = SimpleNamespace(
            prepare=lambda _: events.append("register_order"),
            after_remote_write=lambda _: events.append("read_order"),
        )
        bank = CUDASparseWorkingSet(
            device="cuda:0",
            dtype=torch.float32,
            budget=TransferBudget(65536, 8),
            request_id="consumer",
            incarnation="inc",
            entry_transfer_id=c.entry.key.transfer_id,
            layout_fingerprint=c.entry.layout.fingerprint,
            expected_groups=((0, 0), (1, 0)),
            prompt_tokens=8,
            head_dim=c.entry.layout.head_dim,
            max_union_tokens=8,
        )
        bank.device = torch.device("cpu")
        monkeypatch.setattr(bank, "_synchronize", lambda: events.append("bank_drain"))
        group = CUDARuntimeInstallGroup(
            {0: bank},
            interval=4,
            lead_tokens=1,
            peer_epochs={0: "rank-D"},
            timeout_seconds=30,
            max_pending_events=8,
            max_pending_bytes=65536,
        )
        route = CUDAReceiveRoute(
            c.client, c.store.worker_epoch, "D-cuda-endpoint", c.store.rail
        )
        sink = CUDASparseDelivery(
            group,
            registry,
            key=c.entry.key,
            routes={0: route},
            poll_interval_seconds=0.001,
        )
        result = SimpleNamespace(
            **{
                **vars(c),
                "group": group,
                "registry": registry,
                "bank": bank,
                "sink": sink,
                "events": events,
                "route": route,
            }
        )
        try:
            yield result
        finally:
            for delivery in c.store.entries[c.entry.key].deliveries.values():
                handle = delivery.transfer_handle
                if handle and not handle.transport_state.is_locally_safe_to_release:
                    c.engine.finish(handle)
            c.store.progress_transfers()
            await sink.close()
            if bank.snapshot()["quarantine"] is None:
                group.close()


async def start(c, count, tokens):
    epoch = c.group.begin(count)
    specs = tuple(
        replace(
            s,
            operation_id=epoch.operation_id,
            target_tokens=epoch.target_tokens,
            token_ids=tokens,
        )
        for s in c.manifest.specs
    )
    task = asyncio.create_task(c.sink.stage(epoch, 0, specs))

    async def wait():
        while True:
            record = c.sink._rounds.get(epoch, {}).get(0)
            if record is not None:
                delivery = c.store.entries[c.entry.key].deliveries.get(
                    record.identity.transfer_id
                )
                if delivery is not None and delivery.transfer_handle is not None:
                    return record, delivery
            if task.done():
                await task
            await asyncio.sleep(0.001)

    record, delivery = await asyncio.wait_for(wait(), 5)
    return epoch, task, record, delivery


async def complete(c, count, tokens):
    epoch, task, record, delivery = await start(c, count, tokens)
    c.engine.finish(delivery.transfer_handle)
    c.store.progress_transfers()
    receipt = await asyncio.wait_for(task, 5)
    assert not c.group.installation_complete(receipt)
    with pytest.raises(SparseReceiveError, match="all ranks"):
        record.confirm_install()
    c.sink.require_installable(epoch)
    assert c.group.try_install(epoch, {0: epoch.target_tokens})
    c.sink.installed(epoch)
    await wait_acks(c.sink)
    return epoch, record


def test_http_delivery_two_rounds_install_ack_and_return_receive_budget(monkeypatch):
    async def run():
        async with case(monkeypatch) as c:
            for count, tokens in ((0, tuple(range(8))), (3, (1, 6, 2))):
                epoch, record = await complete(c, count, tokens)
                assert record.snapshot()["acknowledged"] and record.snapshot()["closed"]
                assert c.sink.snapshot()["pending_rounds"] == 0
                assert c.registry.snapshot() == {}
                permit = c.group.runtime.begin_forward(epoch.target_tokens)
                with c.group.read(0, epoch.target_tokens) as groups:
                    for (layer, head), (spec, tensor) in groups.items():
                        expected = torch.stack(
                            (
                                c.pool.k_buffer[layer][list(tokens), head],
                                c.pool.v_buffer[layer][list(tokens), head],
                            )
                        )
                        torch.testing.assert_close(tensor, expected, atol=0, rtol=0)
                        assert spec.operation_id == epoch.operation_id
                assert c.group.runtime.finish_forward(
                    permit, readers_drained=True, succeeded=True
                )
            assert c.events.count("read_order") == 2
            assert c.registry.budget.snapshot()["used_staging_bytes"] == 0

    asyncio.run(run())


def test_lost_ack_retries_without_reinstall_or_new_destination(monkeypatch):
    async def run():
        async with case(monkeypatch) as c:
            original = c.client.ack_delivery

            async def lost(*args, **kwargs):
                await original(*args, **kwargs)
                raise TimeoutError("lost ACK")

            monkeypatch.setattr(c.client, "ack_delivery", lost)
            epoch, record = await complete(c, 0, tuple(range(8)))
            assert c.sink.snapshot()["errors"]
            assert not record.snapshot()["closed"]
            before = c.group.coordinator.snapshot()
            monkeypatch.setattr(c.client, "ack_delivery", original)
            await c.sink.retry_acks(epoch)
            assert c.group.coordinator.snapshot() == before
            assert not c.sink.snapshot()["errors"] and record.snapshot()["closed"]

    asyncio.run(run())


def test_cancelled_delayed_write_retains_destination_until_real_fake_completion(
    monkeypatch,
):
    async def run():
        async with case(monkeypatch) as c:
            _, task, record, delivery = await start(c, 0, tuple(range(8)))
            c.sink.cancel()
            c.group.cancel()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            assert await c.sink.close()
            assert record.identity.region_id not in c.engine.released
            assert c.registry.budget.snapshot()["used_staging_bytes"] > 0
            assert "read_order" not in c.events
            c.engine.finish(delivery.transfer_handle)
            c.store.progress_transfers()
            assert not await c.sink.close()
            assert record.snapshot()["closed"]
            assert "read_order" not in c.events

    asyncio.run(run())


def test_local_visibility_failure_retains_mr_and_refuses_install(monkeypatch):
    async def run():
        async with case(monkeypatch) as c:

            def fail(_):
                raise RuntimeError("visibility unknown")

            c.registry.ordering.after_remote_write = fail
            epoch, task, record, delivery = await start(c, 0, tuple(range(8)))
            c.engine.finish(delivery.transfer_handle)
            c.store.progress_transfers()
            with pytest.raises(RuntimeError, match="visibility unknown"):
                await task
            assert not c.group.try_install(epoch, {0: 0})
            assert await c.sink.close()
            assert not record.snapshot()["closed"]
            assert record.identity.region_id not in c.engine.released

    asyncio.run(run())


def test_runtime_timeout_aborts_polling_but_does_not_fence_remote_write(monkeypatch):
    async def run():
        async with case(monkeypatch) as c:
            now = [time.monotonic()]
            c.group.runtime._clock = lambda: now[0]
            c.group._timeout = 1
            _, task, record, delivery = await start(c, 0, tuple(range(8)))
            now[0] += 2
            with pytest.raises(InstallProtocolError, match="terminal"):
                await asyncio.wait_for(task, 5)
            assert await c.sink.close()
            assert record.identity.region_id not in c.engine.released
            assert "read_order" not in c.events
            c.engine.finish(delivery.transfer_handle)
            c.store.progress_transfers()
            assert not await c.sink.close()
            assert record.snapshot()["closed"]

    asyncio.run(run())


def test_dtype_comes_from_destination_bank_before_publication(monkeypatch):
    async def run():
        async with case(monkeypatch) as c:
            c.sink._metadata[0]["dtype"] = "torch.float16"
            epoch = c.group.begin(0)
            specs = tuple(
                replace(
                    s,
                    operation_id=epoch.operation_id,
                    target_tokens=0,
                    token_ids=tuple(range(8)),
                )
                for s in c.manifest.specs
            )

            def inspect(manifest, **kwargs):
                assert manifest.dtype == "torch.float16"
                assert kwargs["rail"] == c.route.rail
                assert kwargs["endpoint"] == c.route.endpoint
                raise RuntimeError("inspection before publication")

            monkeypatch.setattr(c.registry, "prepare", inspect)
            with pytest.raises(RuntimeError, match="inspection"):
                await c.sink.stage(epoch, 0, specs)

    asyncio.run(run())


def test_cpu_and_cuda_sink_owners_cannot_be_mixed(monkeypatch):
    async def run():
        async with case(monkeypatch) as c:
            with pytest.raises(SparseReceiveError, match="CPU"):
                CPUSparseDelivery(
                    c.group,
                    c.registry,
                    key=c.entry.key,
                    routes={0: c.route},
                    poll_interval_seconds=0.001,
                )
            with pytest.raises(SparseReceiveError, match="CUDA"):
                CUDASparseDelivery(
                    object(),
                    c.registry,
                    key=c.entry.key,
                    routes={0: c.route},
                    poll_interval_seconds=0.001,
                )
            c.registry.device = torch.device("cuda:1")
            try:
                with pytest.raises(SparseReceiveError, match="device differ"):
                    CUDASparseDelivery(
                        c.group,
                        c.registry,
                        key=c.entry.key,
                        routes={0: c.route},
                        poll_interval_seconds=0.001,
                    )
            finally:
                c.registry.device = torch.device("cpu")

    asyncio.run(run())
