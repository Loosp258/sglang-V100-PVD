"""Real CPU server: fused replies encode once and keep exact selection bytes."""
import asyncio
import json
import pytest
from sglang.srt.disaggregation.pvd import control_server as server
from sglang.srt.disaggregation.pvd.fused_search_delivery import start_fused
from test_pvd_fused_search_delivery import setup, prepared
from test_pvd_prompt_index import shard_client


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('channel', [False, True])
def test_fused_typed_results_single_final_encoding(monkeypatch, rank, channel):
    monkeypatch.setenv('PVD_TYPED_BATCH_RESULTS', '1')
    encoded = []
    original = server._bounded_search_response
    def count(reply):
        encoded.append(reply)
        return original(reply)
    monkeypatch.setattr(server, '_bounded_search_response', count)
    async def run():
        _, store, manifest, pool, _, requests, selection = setup(rank)
        async with shard_client(store) as http:
            search, control, budget, registry, record = await prepared(
                store, manifest, requests, selection, http, binary=True, channel=channel)
            try:
                chosen, ready = await start_fused(record, search, requests)
                assert not ready and len(encoded) == 1
                assert encoded[0]['chosen'] == [list(ids) for ids in chosen]
                delivery = store.entries[manifest.key].deliveries[record.identity.transfer_id]
                store.transfer_engine.finish(delivery.transfer_handle)
                assert await record.poll()
                cache = [{} for _ in range(4)]
                record.copy_to_cache(cache)
                for head in selection['heads']:
                    for token, row in cache[head].items():
                        import torch
                        assert torch.equal(row, torch.stack((pool.k_buffer[0][token,head], pool.v_buffer[0][token,head])))
                await record.ack()
                assert await record.close() and not registry._records
                assert budget.snapshot()['used_staging_bytes'] == 0
            finally:
                await search.close(); await control.close()
        store.close()
    asyncio.run(run())


def test_response_limit_checks_the_exact_sent_bytes():
    reply = {'results': [], 'batch_id': '中文'}
    response = server._bounded_search_response(reply)
    assert json.loads(response.body) == reply
    with pytest.raises(ValueError, match='2 MiB'):
        server._bounded_search_response({'value': 'x' * (2 * 1024 * 1024)})
    with pytest.raises(ValueError):
        server._bounded_search_response({'value': float('nan')})
