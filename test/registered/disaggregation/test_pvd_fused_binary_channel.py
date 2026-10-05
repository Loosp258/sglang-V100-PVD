"""Actual local WebSocket framing, concurrency and original WRITE fencing."""
import asyncio
import threading
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from sglang.srt.disaggregation.pvd.fused_binary_channel import (
    FusedBinaryChannel,BinaryChannelError,PATH,unpack_request,pack_response,MAX_RESPONSE_BYTES)
from sglang.srt.disaggregation.pvd.fused_search_delivery import start_fused
from sglang.srt.disaggregation.pvd.search_client import SearchTransportError
from test_pvd_fused_search_delivery import setup,prepared
from test_pvd_prompt_index import shard_client
from test_pvd_oasis_transport_io import transport, background_io


def test_two_out_of_order_responses_use_one_connection_and_keep_sequence_identity():
    async def run():
        connections=[]
        async def handler(request):
            ws=web.WebSocketResponse();await ws.prepare(request);connections.append(ws)
            first=unpack_request((await ws.receive()).data)
            second=unpack_request((await ws.receive()).data)
            await ws.send_bytes(pack_response(second[0],200,b'{"value":2}'))
            await ws.send_bytes(pack_response(first[0],200,b'{"value":1}'))
            async for _ in ws:pass
            return ws
        app=web.Application();app.router.add_get(PATH,handler)
        async with TestServer(app) as http:
            channel=FusedBinaryChannel(str(http.make_url('')).rstrip('/'),timeout=2,max_response_bytes=1024)
            one,two=await asyncio.gather(channel.exchange(b'one'),channel.exchange(b'two'))
            assert one=={'value':1} and two=={'value':2}
            assert len(connections)==channel.snapshot()['connections']==1
            assert channel.snapshot()['pending_peak']==2
            await channel.close();assert channel.snapshot()['owned']==0
    asyncio.run(run())


@pytest.mark.parametrize('damage',['wrong_sequence','short','text','oversize','disconnect'])
def test_bad_or_lost_response_fails_closed_without_reconnect(damage):
    async def run():
        async def handler(request):
            ws=web.WebSocketResponse();await ws.prepare(request)
            sequence,_=unpack_request((await ws.receive()).data)
            if damage=='wrong_sequence':await ws.send_bytes(pack_response(sequence+1,200,b'{}'))
            if damage=='short':await ws.send_bytes(b'bad')
            if damage=='text':await ws.send_str('{}')
            if damage=='oversize':await ws.send_bytes(b'x'*2048)
            if damage=='disconnect':await ws.close()
            async for _ in ws:pass
            return ws
        app=web.Application();app.router.add_get(PATH,handler)
        async with TestServer(app) as http:
            channel=FusedBinaryChannel(str(http.make_url('')).rstrip('/'),timeout=2,max_response_bytes=1024)
            try:
                with pytest.raises(BinaryChannelError):await channel.exchange(b'request')
                with pytest.raises(BinaryChannelError):await channel.exchange(b'retry')
                assert channel.snapshot()['connections']==1
            finally:
                await channel.close()
    asyncio.run(run())


def test_cancelled_fused_await_fences_late_writer_and_close_joins_channel_work(monkeypatch):
    async def run():
        index,store,manifest,_,_,requests,selection=setup(0)
        entered,release=threading.Event(),threading.Event()
        original=index.backend.search
        def slow(*args,**kwargs):
            entered.set();assert release.wait(5)
            return original(*args,**kwargs)
        monkeypatch.setattr(index.backend,'search',slow)
        async with shard_client(store) as http:
            search,control,budget,registry,record=await prepared(store,manifest,requests,selection,http,binary=True,channel=True)
            pending=asyncio.create_task(start_fused(record,search,requests))
            closing=None
            try:
                assert await asyncio.to_thread(entered.wait,5)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):await pending
                assert record._published and search._channel.snapshot()['owned']==1
                assert await record.close()  # exact absent-write fence, not cancellation
                closing=asyncio.create_task(search.close())
                await asyncio.sleep(0.01)
                assert not closing.done()
                release.set();await closing
                assert search._channel.snapshot()['owned']==0
                assert not store.transfer_engine.pending and store.transfer_engine.total_put_bytes==0
                assert not registry._records and budget.snapshot()['used_staging_bytes']==0
            finally:
                release.set();await asyncio.gather(pending,return_exceptions=True)
                if closing is not None:await closing
                await search.close();await control.close()
        store.close()
    asyncio.run(run())


def test_disconnect_after_native_submit_does_not_release_destination(monkeypatch):
    async def drop_response(socket,data,**kwargs):
        await socket.close(code=1011)
    monkeypatch.setattr(web.WebSocketResponse,'send_bytes',drop_response)
    async def run():
        _,store,manifest,_,_,requests,selection=setup(0)
        async with shard_client(store) as http:
            search,control,budget,registry,record=await prepared(store,manifest,requests,selection,http,binary=True,channel=True)
            try:
                with pytest.raises(SearchTransportError):await start_fused(record,search,requests)
                assert not await record.close() and record._registration is not None
                delivery=store.entries[manifest.key].deliveries[record.identity.transfer_id]
                assert delivery.transfer_handle.transfer_id in store.transfer_engine.pending
                store.transfer_engine.finish(delivery.transfer_handle)
                assert await record.close() and budget.snapshot()['used_staging_bytes']==0
            finally:
                await search.close();await control.close()
        store.close()
    asyncio.run(run())


def test_request_owned_channels_survive_worker_retirement_and_close_on_io_loop(monkeypatch,background_io):
    from concurrent.futures import ThreadPoolExecutor
    async def begin():
        async def handler(request):
            ws=web.WebSocketResponse();await ws.prepare(request)
            async for message in ws:
                sequence,_=unpack_request(message.data)
                await ws.send_bytes(pack_response(sequence,200,b'{"ok":true}'))
            return ws
        app=web.Application();app.router.add_get(PATH,handler)
        server=TestServer(app);await server.start_server()
        return server
    server=background_io.submit(begin()).result(timeout=5)
    owner=transport(monkeypatch,background_io,reuse_io=False,
        binary_control_channel=True,url=str(server.make_url('')).rstrip('/'))
    observed=[];barrier=threading.Barrier(2)
    def worker():
        for _ in range(2):
            state=owner._worker();barrier.wait(timeout=5)
            for client in state['search'].values():
                observed.append(id(client))
                assert state['loop'].run_until_complete(client._post_fused_channel(b'opaque-test-frame'))=={'ok':True}
            owner._retire_worker(state)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(worker) for _ in range(2)]
            for future in futures:future.result(timeout=5)
        assert len(set(observed))==2 and not owner.workers
        clients=owner._channel_clients['search']
        assert all(not c._closed and c._channel.snapshot()['connections']==1 for c in clients.values())
        assert all(c._channel.snapshot()['requests']==4 for c in clients.values())
        owner.close()
        assert all(c._closed and c._channel.snapshot()['closed'] for c in clients.values())
    finally:
        if not owner.closed:owner.close()
        background_io.submit(server.close()).result(timeout=5)
