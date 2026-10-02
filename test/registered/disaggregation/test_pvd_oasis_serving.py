"""Formal actual-token commit/lease drainage and native CPU cache protocol."""

import asyncio
from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.pvd.oasis_scheduler import OasisSchedulerBinding
from sglang.srt.disaggregation.pvd.oasis_transport import OasisCPUReceiveRegistry
from sglang.srt.disaggregation.pvd.oasis_pipeline import LayerLookahead, LayerReply
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_sparse_receiver import finish, receiving


def binding_case():
    req = SimpleNamespace(rid="r", output_ids=[7], is_retracted=False,
                          finished=lambda: False)
    owner = SimpleNamespace(committed=[], state="awaiting_actual_commit",
                            actual_committed=lambda t: owner.committed.append(t), close=lambda: ())
    processor = object()
    binding = object.__new__(OasisSchedulerBinding)
    binding._thread = threading.get_ident()
    binding.quarantined, binding._processing, binding._deferred_release = False, False, None
    binding.scheduler = SimpleNamespace(batch_result_processor=processor)
    releases = []
    binding.manager = SimpleNamespace(decode_refresher=SimpleNamespace(
        release_request=lambda req: releases.append(req)))
    batch = SimpleNamespace(reqs=[req])
    result = SimpleNamespace(next_token_ids=torch.tensor([19]))
    binding.records = {req.rid: (req, owner, None, ())}
    binding._dispatch = (batch, req, owner, (7,), None, result)
    return binding, req, owner, processor, batch, result, releases


def test_ordinary_sampler_commits_exactly_one_and_defers_consumer_release():
    b, req, owner, processor, batch, result, releases = binding_case()
    with b.processing(processor, batch, result):
        req.output_ids.append(19)
        assert b.release(req) is True
        assert b.owns(req) and not releases and not owner.committed
    assert owner.committed == [19] and releases == [req]
    assert not b.owns(req) and b._dispatch is None


@pytest.mark.parametrize("changed", ["future", "none", "foreign"])
def test_bad_actual_commit_quarantines_and_retains_request(changed):
    b, req, owner, processor, batch, result, releases = binding_case()
    with pytest.raises(RuntimeError):
        with b.processing(processor, batch, result):
            if changed == "future":
                req.output_ids.append(99)
            elif changed == "foreign":
                batch.reqs = [SimpleNamespace(rid="r")]
    assert b.quarantined and b.owns(req) and not owner.committed and not releases


def test_foreign_req_and_result_cannot_commit():
    b, req, owner, processor, batch, result, releases = binding_case()
    assert not b.owns(SimpleNamespace(rid=req.rid))
    with pytest.raises(RuntimeError):
        with b.processing(processor, batch, SimpleNamespace(next_token_ids=[19])):
            pytest.fail("foreign result reached sampler commit")


def test_two_futures_allow_query_publication_before_prior_bank_consumption():
    pipe = LayerLookahead("r", "e", layers=1, max_pending_per_layer=2)
    try:
        pipe.publish(0, 0, lambda ticket: LayerReply(ticket, "prior"))
        pipe.publish(1, 0, lambda ticket: LayerReply(ticket, "next"))
        with pytest.raises(RuntimeError):
            pipe.publish(2, 0, lambda ticket: LayerReply(ticket, "unbounded"))
        assert pipe.consume(0, 0) == "prior"
        assert pipe.consume(1, 0) == "next"
    finally:
        assert pipe.close() == ()


def test_native_cpu_cache_reads_only_after_terminal_and_ack_owns_clone():
    async def run():
        async with receiving() as c:
            assert await c.record.close()
            manifest = replace(c.manifest, specs=(c.manifest.specs[0],))
            registry = OasisCPUReceiveRegistry(c.engine, TransferBudget(65536, 32), receiver_epoch="D-inc")
            r = registry.prepare(manifest, key=c.entry.key, rank=0, rail=c.store.rail,
                endpoint="D", sender_epoch=c.store.worker_epoch, client=c.client)
            c.registry, c.record = registry, r
            cache = [{} for _ in range(c.entry.layout.total_kv_heads)]
            assert not await r.start()
            with pytest.raises(RuntimeError, match="terminal-success"):
                r.copy_to_cache(cache)
            finish(c)
            assert await r.poll()
            r.copy_to_cache(cache)
            spec = manifest.specs[0]
            expected = torch.stack((c.pool.k_buffer[spec.layer][list(spec.token_ids), spec.kv_head],
                c.pool.v_buffer[spec.layer][list(spec.token_ids), spec.kv_head]), dim=1)
            actual = torch.stack([cache[spec.kv_head][t] for t in spec.token_ids])
            torch.testing.assert_close(actual, expected)
            with pytest.raises(RuntimeError):
                r.copy_to_cache(cache)
            await r.ack()
            assert await r.close()
            assert registry.snapshot() == {} and r._buffer is None
            torch.testing.assert_close(torch.stack([cache[spec.kv_head][t] for t in spec.token_ids]), expected)
    asyncio.run(run())
