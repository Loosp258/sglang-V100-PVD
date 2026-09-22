"""Real HTTP protocol with CPU tensors/delayed fake writes, NOT GPUDirect proof."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.cuda_rank_install import CUDARankInstallParticipant
from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import CUDASparseReceiveRegistry
from sglang.srt.disaggregation.pvd.cuda_working_set import CUDASparseWorkingSet
from sglang.srt.disaggregation.pvd.rank_install_wire import (
    RankInstallExchange,
    RankInstallMessage,
)
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_install import RankInstallCoordinator
from sglang.srt.disaggregation.pvd.sparse_receiver import SparseReceiveError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferBudget,
)
from test_pvd_sparse_receiver import finish, receiving


@asynccontextmanager
async def cuda_policy_case(monkeypatch):
    async with receiving() as c:
        events = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        registry = CUDASparseReceiveRegistry(
            c.engine,
            TransferBudget(65536, 8),
            receiver_epoch="cuda-policy-D",
            device="cuda:0",
        )
        # Explicit test policy: real CPU tensors, not a hidden production fallback.
        registry.device = torch.device("cpu")
        registry.ordering = SimpleNamespace(
            prepare=lambda _: events.append("prepare_sync_memops"),
            after_remote_write=lambda _: events.append("order_after_remote_write"),
        )
        bank_budget = TransferBudget(65536, 8)
        bank = CUDASparseWorkingSet(
            device="cuda:0",
            dtype=torch.float32,
            budget=bank_budget,
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
        peer = CUDARankInstallParticipant(bank, rank=0, peer_epoch="rank-D", interval=4)
        exchange = RankInstallExchange(
            RankInstallCoordinator(
                "consumer",
                "inc",
                c.entry.key.transfer_id,
                rank_layouts={0: c.entry.layout.fingerprint},
                interval=4,
                lead_tokens=1,
            ),
            peer_epochs={0: "rank-D"},
        )
        initial = exchange.begin(0)
        full = SparseDeliveryManifest(
            tuple(
                replace(
                    s,
                    operation_id=initial.operation_id,
                    target_tokens=0,
                    token_ids=tuple(range(8)),
                )
                for s in c.manifest.specs
            ),
            "torch.float32",
            c.entry.layout.head_dim,
        )
        backing = torch.cat(
            [
                torch.stack(
                    (c.pool.k_buffer[s.layer][:8, 0], c.pool.v_buffer[s.layer][:8, 0])
                )
                .reshape(-1)
                .view(torch.uint8)
                for s in full.specs
            ]
        )
        guard = ResourceGuard(backing, lambda: None)
        exchange.receive(
            peer.stage(initial, full.payload_views(backing), source_guard=guard),
            peer_rank=0,
        )
        exchange.receive(peer.park(0), peer_rank=0)
        exchange.receive(
            peer.command(exchange.install_commands(initial)[0]), peer_rank=0
        )
        exchange.receive(
            peer.command(exchange.resume_commands(initial)[0]), peer_rank=0
        )
        guard.request_release()
        epoch = exchange.begin(3)
        manifest = replace(
            c.manifest,
            specs=tuple(
                replace(s, operation_id=epoch.operation_id) for s in c.manifest.specs
            ),
        )
        record = registry.prepare(
            manifest,
            key=c.entry.key,
            rank=0,
            rail=c.store.rail,
            endpoint="cuda-policy",
            sender_epoch=c.store.worker_epoch,
            client=c.client,
        )
        events.clear()
        case = SimpleNamespace(
            **{
                **vars(c),
                "registry": registry,
                "record": record,
                "epoch": epoch,
                "bank": bank,
                "peer": peer,
                "exchange": exchange,
                "events": events,
                "bank_budget": bank_budget,
            }
        )
        try:
            yield case
        finally:
            for delivery in c.store.entries[c.entry.key].deliveries.values():
                handle = delivery.transfer_handle
                if handle and not handle.transport_state.is_locally_safe_to_release:
                    c.engine.finish(handle)
            c.store.progress_transfers()
            await registry.close()
            if bank.snapshot()["quarantine"] is None:
                peer.close()


def test_delivery_orders_copies_then_requires_all_resumed_before_ack(monkeypatch):
    async def run():
        async with cuda_policy_case(monkeypatch) as c:
            r = c.record
            assert not await r.start()
            with pytest.raises(SparseReceiveError, match="successful delivery"):
                r.stage(c.peer, c.epoch, exchange=c.exchange)
            assert not c.events
            finish(c)
            assert await r.poll()
            prepared = r.stage(c.peer, c.epoch, exchange=c.exchange)
            assert c.events == ["order_after_remote_write", "bank_drain"]
            c.exchange.receive(prepared, peer_rank=0)
            c.exchange.receive(c.peer.park(4), peer_rank=0)
            c.exchange.receive(
                c.peer.command(c.exchange.install_commands(c.epoch)[0]), peer_rank=0
            )
            with pytest.raises(SparseReceiveError, match="all ranks"):
                await r.ack()
            resume = c.exchange.resume_commands(c.epoch)[0]
            resumed = c.peer.command(resume)
            with pytest.raises(SparseReceiveError, match="all ranks"):
                await r.ack()
            c.exchange.receive(resumed, peer_rank=0)
            assert not c.exchange.installation_complete(
                replace(
                    RankInstallMessage.decode(prepared).receipt, staging_id="foreign"
                )
            )
            await r.ack()
            assert await r.close()
            assert not c.registry.snapshot()
            with c.peer.read(4) as groups:
                for (layer, head), (spec, data) in groups.items():
                    expected = torch.stack(
                        (
                            c.pool.k_buffer[layer][list(spec.token_ids), head],
                            c.pool.v_buffer[layer][list(spec.token_ids), head],
                        )
                    )
                    torch.testing.assert_close(data, expected, rtol=0, atol=0)
            assert c.registry.budget.snapshot()["used_staging_bytes"] == 0

    asyncio.run(run())


def test_cancel_with_late_write_keeps_mr_and_never_orders_gpu_read(monkeypatch):
    async def run():
        async with cuda_policy_case(monkeypatch) as c:
            await c.record.start()
            assert not await c.record.close()
            assert not c.events
            assert c.record.identity.region_id not in c.engine.released
            finish(c)
            assert await c.record.close()
            assert not c.events
            assert c.registry.budget.snapshot()["used_staging_bytes"] == 0

    asyncio.run(run())


@pytest.mark.parametrize("fault", ["visibility", "bank"])
def test_unknown_local_completion_retains_destination_and_budget(monkeypatch, fault):
    async def run():
        async with cuda_policy_case(monkeypatch) as c:
            await c.record.start()
            finish(c)
            assert await c.record.poll()

            def fail(*args):
                raise RuntimeError("CUDA completion unknown")

            if fault == "visibility":
                c.registry.ordering.after_remote_write = fail
            else:
                monkeypatch.setattr(c.bank, "_synchronize", fail)
            with pytest.raises(RuntimeError, match="completion unknown"):
                c.record.stage(c.peer, c.epoch, exchange=c.exchange)
            assert not await c.record.close()
            assert c.record.identity.region_id not in c.engine.released
            assert (
                c.registry.budget.snapshot()["used_staging_bytes"]
                == c.record.manifest.nbytes
            )
            assert c.record.snapshot()["local_completion_unknown"]
            assert not c.record.snapshot()["staged"]
            if fault == "bank":
                assert c.bank.snapshot()["source_guard_held"]

    asyncio.run(run())


def test_native_release_failure_keeps_receive_charge_and_can_retry(monkeypatch):
    async def run():
        async with cuda_policy_case(monkeypatch) as c:
            original = c.engine.release_memory

            def fail(_):
                raise RuntimeError("unregister failed")

            monkeypatch.setattr(c.engine, "release_memory", fail)
            with pytest.raises(RuntimeError, match="unregister failed"):
                await c.record.close()
            assert c.registry.budget.snapshot()["used_staging_bytes"] > 0
            assert not c.record.snapshot()["closed"]
            monkeypatch.setattr(c.engine, "release_memory", original)
            assert await c.record.close()
            assert c.registry.budget.snapshot()["used_staging_bytes"] == 0

    asyncio.run(run())


def test_source_guard_is_empty_before_receive_budget_is_refunded(monkeypatch):
    async def run():
        async with cuda_policy_case(monkeypatch) as c:
            original = c.registry.budget.release

            def release(owner):
                assert c.record._source_guard.value is None
                assert c.record._buffer is None
                assert c.record._registration is None
                original(owner)

            monkeypatch.setattr(c.registry.budget, "release", release)
            c.record._source_guard.pin("consumer")
            assert not await c.record.close()
            assert c.registry.budget.snapshot()["used_staging_bytes"] > 0
            assert c.record.identity.region_id not in c.engine.released
            c.record._source_guard.unpin("consumer")
            # Native release can finish before the owner's next close poll;
            # capacity stays charged until guard storage has actually gone.
            assert c.record._source_guard.value is None
            assert c.registry.budget.snapshot()["used_staging_bytes"] > 0
            assert await c.record.close()
            assert c.registry.budget.snapshot()["used_staging_bytes"] == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    "field",
    ["generation", "region_id", "sender_epoch", "transferred_bytes", "transport_state"],
)
def test_foreign_terminal_reply_cannot_reach_cuda_ordering(monkeypatch, field):
    async def run():
        async with cuda_policy_case(monkeypatch) as c:
            await c.record.start()
            finish(c)
            reply = await c.client.poll_delivery(
                c.entry.key, c.record.identity.transfer_id
            )
            if field in ("generation", "region_id", "sender_epoch"):
                reply["write_identity"][field] = "foreign"
            elif field == "transferred_bytes":
                reply[field] -= 1
            else:
                reply[field] = "unknown"
            with pytest.raises(ValueError):
                c.record._observe(reply)
            with pytest.raises(SparseReceiveError, match="successful delivery"):
                c.record.stage(c.peer, c.epoch, exchange=c.exchange)
            assert not c.events

    asyncio.run(run())
