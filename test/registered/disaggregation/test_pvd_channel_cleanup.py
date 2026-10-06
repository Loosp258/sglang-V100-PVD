"""Real CPU websocket cleanup and identity-bound HTTP recovery."""
import asyncio
from dataclasses import replace
import pytest
from sglang.srt.disaggregation.pvd.fused_binary_channel import (
    ChannelRecordControl, pack_cleanup, unpack_cleanup, BinaryChannelError)
from sglang.srt.disaggregation.pvd.fused_search_delivery import start_fused
from test_pvd_fused_search_delivery import setup, prepared
from test_pvd_prompt_index import shard_client


@pytest.mark.parametrize('rank',[0,1])
@pytest.mark.parametrize('mode',['ack','fence','disconnect','hit','proof_hit'])
def test_repeated_records_share_channel_and_retire_exact_owners(rank,mode):
    async def run():
        _,store,manifest,_,_,requests,selection=setup(rank)
        if mode in ('hit','proof_hit'):
            selection['cached']=[list(range(manifest.prompt_token_count))]*2
        async with shard_client(store) as http:
            search=None;controls=[]
            try:
                for _ in range(6):
                    new,control,budget,registry,record=await prepared(store,manifest,requests,selection,http,binary=True,channel=True)
                    if search is None: search=new
                    else: await new.close()
                    controls.append(control)
                    record._client=ChannelRecordControl(search,control,record.identity,record.fused_scope)
                    record.fused_channel_cleanup=True
                    record.fused_zero_miss_proof=mode=='proof_hit'
                    chosen,ready=await start_fused(record,search,requests)
                    if mode not in ('hit','proof_hit'):
                        assert not ready
                        delivery=store.entries[manifest.key].deliveries[record.identity.transfer_id]
                        store.transfer_engine.finish(delivery.transfer_handle)
                        assert await record.poll()
                        record.copy_to_cache([{} for _ in range(4)])
                    if mode=='ack': await record.ack()
                    if mode=='disconnect':
                        await search._channel.close()
                    assert await record.close()
                    assert not registry._records and budget.snapshot()['used_staging_bytes']==0
                    if mode=='disconnect': break
                snap=search._channel.snapshot()
                assert snap['connections']==1 and snap['owned']==0
                expected=6 if mode=='proof_hit' else 1 if mode=='disconnect' else 12
                assert snap['requests']==expected
                if mode=='disconnect':assert controls[0]._session is not None
                if mode not in ('disconnect','proof_hit'):
                    assert all(c._session is None for c in controls) if mode=='hit' else True
            finally:
                if search:await search.close()
                for control in controls:await control.close()
        store.close()
    asyncio.run(run())


def test_cleanup_codec_and_bad_generation_never_ack_other_writer():
    async def run():
        _,store,manifest,_,_,requests,selection=setup(0)
        async with shard_client(store) as http:
            search,control,budget,registry,record=await prepared(store,manifest,requests,selection,http,binary=True,channel=True)
            record.fused_channel_cleanup=True
            try:
                await start_fused(record,search,requests)
                binding=['request','incarnation',manifest.key.transfer_id,[0,1]]
                raw=pack_cleanup('ack',record.identity,binding)
                assert unpack_cleanup(raw)['identity']==record.identity
                bad=replace(record.identity,generation='wrong')
                from sglang.srt.disaggregation.pvd.search_client import SearchTransportError
                with pytest.raises(SearchTransportError):
                    await search._post_fused_channel(pack_cleanup('ack',bad,binding),cleanup=True)
                assert not record._acknowledged and not await record.close()
                delivery=store.entries[manifest.key].deliveries[record.identity.transfer_id]
                store.transfer_engine.finish(delivery.transfer_handle)
                assert await record.close() and not registry._records
            finally:
                await search.close(); await control.close()
        store.close()
    asyncio.run(run())


def test_cleanup_channel_requires_configured_request_channel(monkeypatch,tmp_path):
    from test_pvd_delivery_followup_config import load,config
    cfg=config(4);cfg['channel_cleanup']=True
    with pytest.raises(ValueError,match='binary channel'):load(monkeypatch,tmp_path,cfg)
    cfg['binary_control_channel']=True
    assert load(monkeypatch,tmp_path,cfg)['channel_cleanup'] is True


def test_query_and_cleanup_have_separate_bounded_channel_slots():
    from aiohttp import web
    from aiohttp.test_utils import TestServer
    from sglang.srt.disaggregation.pvd.fused_binary_channel import FusedBinaryChannel,PATH,unpack_request,pack_response
    async def run():
        entered=[];release=asyncio.Event();tasks=[]
        async def handler(request):
            ws=web.WebSocketResponse();await ws.prepare(request)
            async def reply(seq):
                await release.wait()
                await ws.send_bytes(pack_response(seq,200,b'{"ok":true}'))
            async for message in ws:
                seq,_=unpack_request(message.data);entered.append(seq)
                tasks.append(asyncio.create_task(reply(seq)))
            await asyncio.gather(*tasks)
            return ws
        app=web.Application();app.router.add_get(PATH,handler)
        async with TestServer(app) as server:
            channel=FusedBinaryChannel(str(server.make_url('')).rstrip('/'),timeout=2,max_response_bytes=1024)
            jobs=[asyncio.create_task(channel.exchange(b'opaque',cleanup=cleanup)) for cleanup in (False,False,True,True)]
            while len(entered)<4:await asyncio.sleep(.001)
            with pytest.raises(BinaryChannelError,match='capacity'):await channel.exchange(b'extra')
            with pytest.raises(BinaryChannelError,match='capacity'):await channel.exchange(b'extra',cleanup=True)
            release.set()
            assert await asyncio.gather(*jobs)==[{'ok':True}]*4
            assert channel.snapshot()['pending_peak']==4
            await channel.close()
    asyncio.run(run())
