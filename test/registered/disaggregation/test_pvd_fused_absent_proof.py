"""Real HTTP zero-miss proofs, slot generation and original fence recovery."""
import asyncio
from dataclasses import replace

import pytest
from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient
from sglang.srt.disaggregation.pvd.fused_search_delivery import start_fused
from test_pvd_fused_receive_slots import prepared
from test_pvd_oasis_receive_slot_records import slot_case, writer


@pytest.mark.parametrize('binary,channel', [(False,False),(True,False),(True,True)])
def test_zero_miss_response_closes_exact_authorization_without_extra_fence(monkeypatch, binary, channel):
    async def run():
        async with slot_case(monkeypatch) as c:
            search = PVDShardSearchClient(c.client.base_url, binary_queries=binary, binary_control_channel=channel)
            try:
                previous = None
                async def no_extra_fence(*args):
                    pytest.fail('valid proof must eliminate the HTTP fence')
                monkeypatch.setattr(c.client, 'fence_delivery', no_extra_fence)
                for n in range(2):
                    record, requests = prepared(c, search, f'hit{n}', cached=True, zero_miss_proof=True)
                    if previous:
                        assert previous.region_id == record.identity.region_id
                        assert previous.generation != record.identity.generation
                    _, ready = await start_fused(record, search, requests)
                    assert ready and record._safe and record._absent_write_closed
                    assert not record._acknowledged
                    assert c.store._absent_write_fences[(c.entry.key, record.identity.transfer_id)] == record.identity
                    with pytest.raises(Exception, match='fenced'):
                        c.store.reserve_delivery(c.entry.key, record.identity.transfer_id, record._registration.descriptor)
                    previous = record.identity
                    assert await record.close()
                    assert c.pool.snapshot()['leased_slots'] == 0
                    assert not c.registry._records and not c.engine.pending
                assert c.pool.snapshot()['physical_register_calls'] == 1
            finally:
                await search.close()
    asyncio.run(run())


@pytest.mark.parametrize('damage', ['missing','generation','false','lost'])
def test_bad_or_lost_proof_requires_original_identity_fence(monkeypatch, damage):
    async def run():
        async with slot_case(monkeypatch) as c:
            search = PVDShardSearchClient(c.client.base_url, binary_queries=True)
            record, requests = prepared(c, search, 'bad-proof', cached=True, zero_miss_proof=True)
            original = search._post_json
            async def corrupt(*args, **kwargs):
                reply = await original(*args, **kwargs)
                if damage == 'lost': raise TimeoutError('lost response')
                if damage == 'missing': reply.pop('absent_write_fence')
                if damage == 'generation': reply['absent_write_fence']['generation'] = 'stale'
                if damage == 'false': reply['absent_write_fence']['fenced'] = False
                return reply
            monkeypatch.setattr(search, '_post_json', corrupt)
            fence, calls = c.client.fence_delivery, []
            async def checked(identity):
                calls.append(identity)
                return await fence(identity)
            monkeypatch.setattr(c.client, 'fence_delivery', checked)
            try:
                with pytest.raises((ValueError, TimeoutError)):
                    await start_fused(record, search, requests)
                assert not record._safe and c.pool.snapshot()['leased_slots'] == 1
                assert await record.close()
                assert calls == [record.identity] and c.pool.snapshot()['leased_slots'] == 0
            finally:
                await search.close()
    asyncio.run(run())


def test_absent_proof_rejects_existing_writer_without_cancelling_it(monkeypatch):
    async def run():
        async with slot_case(monkeypatch) as c:
            search = PVDShardSearchClient(c.client.base_url, binary_queries=True)
            record, requests = prepared(c, search, 'miss', zero_miss_proof=True)
            try:
                await start_fused(record, search, requests)
                delivery = writer(c, record)
                with pytest.raises(Exception, match='already has a delivery'):
                    c.store.fence_absent_write(record.identity)
                assert delivery.authorization.fence(record.identity)['fenced'] is False
                c.engine.finish(delivery.transfer_handle)
                assert await record.close()
            finally:
                await search.close()
    asyncio.run(run())
