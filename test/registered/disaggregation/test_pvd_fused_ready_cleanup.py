"""Real fused HTTP/FP16 path; CUDA ordering is explicitly a CPU policy double."""
import asyncio
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.pvd.oasis_transport import OasisLayerTransport
from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient, SearchScope
from test_pvd_oasis_receive_slot_records import slot_case
from test_pvd_prompt_index import ident


@pytest.mark.parametrize('binary', [False, True])
@pytest.mark.parametrize('failure', [None, 'ack'])
def test_fused_copy_returns_before_ack_and_retains_slot_until_owned_cleanup(monkeypatch, binary, failure):
    async def run():
        async with slot_case(monkeypatch) as c:
            search = PVDShardSearchClient(c.client.base_url, binary_queries=binary)
            owner = object.__new__(OasisLayerTransport)
            owner.selected = SimpleNamespace(manifest=c.entry)
            owner.capacity = owner.max_new = 4
            owner.prompt_tokens, owner.timeout = 8, 5
            owner.compact_cache_snapshots = False
            owner.binary_queries, owner.ready_before_cleanup = binary, True
            owner._cache_valid = torch.zeros((1, 2, 8), dtype=torch.bool)
            owner.cache, owner.versions = [[{}, {}]], {}
            import threading
            owner.lock, owner.quarantined = threading.Lock(), False
            owner.endpoints, owner.incarnation = {0: 'D'}, 'inc'
            ticket = SimpleNamespace(request_id='req', incarnation='inc', step=0, layer=0)
            route = SimpleNamespace(rank=0, rail=c.store.rail, sender_epoch=c.store.worker_epoch)
            requests = [(ident(c.entry.key.transfer_id, kv_head=h), [[1.0] * 128], 2,
                         SearchScope(8, c.entry.layout.page_size, 128, 'l2')) for h in (0, 1)]
            state = dict(registry=c.registry, search={0: search}, control={0: c.client},
                         fused_profiles=[], delivery_profiles=[])
            try:
                task = asyncio.create_task(owner._fused_fetch(state, ticket, route, requests, None, False))
                deadline = asyncio.get_running_loop().time() + 5
                while not c.engine.pending:
                    assert asyncio.get_running_loop().time() < deadline
                    await asyncio.sleep(0.001)
                delivery = next(iter(c.store.entries[c.entry.key].deliveries.values()))
                c.engine.finish(delivery.transfer_handle)
                chosen, rows = await task
                assert rows == 4 and len(state['pending_cleanup']) == 1
                record = state['pending_cleanup'][0]
                assert record.profile['ack_calls'] == 0 and c.pool.snapshot()['leased_slots'] == 1
                for h, ids in enumerate(chosen):
                    for token in ids:
                        oracle = torch.stack((c.source.k_buffer[0][token, h], c.source.v_buffer[0][token, h]))
                        assert torch.equal(owner.cache[0][h][token], oracle)
                entered, release = asyncio.Event(), asyncio.Event()
                original = record.ack
                async def slow_ack():
                    entered.set()
                    await release.wait()
                    if failure:
                        raise RuntimeError('lost ACK')
                    await original()
                record.ack = slow_ack
                cleaning = asyncio.create_task(owner._finish_owned_cleanup(state))
                await entered.wait()
                assert not cleaning.done() and c.pool.snapshot()['leased_slots'] == 1
                release.set()
                if failure:
                    with pytest.raises(RuntimeError, match='cleanup failed'):
                        await cleaning
                else:
                    await cleaning
                assert c.pool.snapshot()['leased_slots'] == 0
            finally:
                await search.close()
    asyncio.run(run())
