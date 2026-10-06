"""Real async owner thread, bounded phases and full CPU-policy transport jobs."""
import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.pvd.oasis_async_jobs import BoundedAsyncLayerJobs
from sglang.srt.disaggregation.pvd.oasis_pipeline import LayerLookahead
from test_pvd_oasis_transport_io import transport, background_io


def stop_test_owner(pool):
    # Only for tests with no native resources, after failed close has proved
    # that the production owner loop remains retained.
    if pool.thread.is_alive():
        pool.loop.call_soon_threadsafe(pool.loop.stop)
        pool.thread.join(5)


def test_slow_cleanup_yields_query_capacity_and_keeps_one_owner_thread():
    pool=BoundedAsyncLayerJobs(capacity=4)
    release=threading.Event()
    cleaned=[]
    async def work(publish, cleanup):
        publish(threading.get_ident())
        await cleanup()
        while not release.is_set(): await asyncio.sleep(0.001)
        cleaned.append(threading.get_ident())
    try:
        first=pool.submit(work,published=time.monotonic(),timeout=5)
        second=pool.submit(work,published=time.monotonic(),timeout=5)
        assert first.result(5)[0]==second.result(5)[0]==pool.thread.ident
        third=pool.submit(work,published=time.monotonic(),timeout=5)
        fourth=pool.submit(work,published=time.monotonic(),timeout=5)
        assert third.result(5)[0]==fourth.result(5)[0]==pool.thread.ident
        with pytest.raises(RuntimeError,match='capacity'):
            pool.submit(work,published=time.monotonic(),timeout=5)
        status=pool.snapshot()
        assert status['active_peak']==2 and status['cleanup_peak']==2 and status['live']==4
        with ThreadPoolExecutor(max_workers=1) as join:
            future=join.submit(pool.close,timeout=5)
            assert not future.done()
            release.set();future.result(5)
        assert cleaned==[pool.thread.ident]*4 and pool.snapshot()['stopped']
    finally:
        release.set();stop_test_owner(pool)


@pytest.mark.parametrize('after_ready',[False,True])
def test_failure_latches_even_after_publication_and_retains_loop(after_ready):
    pool=BoundedAsyncLayerJobs()
    async def work(publish, cleanup):
        if after_ready:
            publish('bank');await cleanup()
        raise RuntimeError('ACK lost')
    try:
        future=pool.submit(work,published=time.monotonic(),timeout=5)
        if after_ready: assert future.result(5)[0]=='bank'
        else:
            with pytest.raises(RuntimeError,match='ACK lost'): future.result(5)
        with pytest.raises(RuntimeError,match='retain owner'): pool.close(timeout=5)
        assert pool.thread.is_alive() and not pool.loop.is_closed()
        with pytest.raises(RuntimeError):pool.submit(work,published=time.monotonic(),timeout=5)
    finally: stop_test_owner(pool)


def test_public_cancellation_does_not_abandon_actual_work():
    pool=BoundedAsyncLayerJobs()
    entered,release=threading.Event(),threading.Event()
    retired=[]
    async def work(publish,cleanup):
        entered.set()
        while not release.is_set():await asyncio.sleep(0.001)
        publish('bank');await cleanup();retired.append(True)
    try:
        future=pool.submit(work,published=time.monotonic(),timeout=5)
        assert entered.wait(5) and future.cancel()
        release.set();pool.close(timeout=5)
        assert retired==[True] and future.cancelled()
    finally:release.set();stop_test_owner(pool)


def test_expired_admission_never_creates_native_owner():
    pool=BoundedAsyncLayerJobs()
    calls=[]
    async def work(*args):calls.append(True)
    try:
        future=pool.submit(work,published=time.monotonic()-2,timeout=1)
        with pytest.raises(TimeoutError):future.result(5)
        with pytest.raises(RuntimeError):pool.close(timeout=5)
        assert not calls and pool.snapshot()['live']==0
    finally:stop_test_owner(pool)


def test_close_timeout_preserves_original_drain_future_and_actual_owner():
    pool=BoundedAsyncLayerJobs()
    entered,release=threading.Event(),threading.Event()
    retired=[]
    async def work(publish,cleanup):
        publish('bank');await cleanup();entered.set()
        while not release.is_set():await asyncio.sleep(0.001)
        retired.append(True)
    try:
        assert pool.submit(work,published=time.monotonic(),timeout=5).result(5)[0]=='bank'
        assert entered.wait(5)
        with pytest.raises(TimeoutError):pool.close(timeout=0.01)
        original=pool._drain_future
        assert original and pool.thread.is_alive() and pool.snapshot()['live']==1
        release.set();pool.close(timeout=5)
        assert pool._drain_future is original and retired==[True] and pool.snapshot()['stopped']
    finally:release.set();stop_test_owner(pool)


def cuda_policy(monkeypatch):
    # Other legacy tests reload the pipeline module. Bind this test assembly
    # to the current exact reply class instead of accepting a foreign class.
    from sglang.srt.disaggregation.pvd import oasis_transport
    monkeypatch.setattr(oasis_transport,'LayerReply',LayerLookahead.consume.__globals__['LayerReply'])
    class Event:
        def record(self,*args):pass
        def synchronize(self):pass
        def query(self):return True
    class Stream:
        def wait_event(self,event):pass
        def synchronize(self):pass
    monkeypatch.setattr(torch.cuda,'Stream',lambda **kwargs:Stream())
    monkeypatch.setattr(torch.cuda,'Event',Event)
    monkeypatch.setattr(torch.cuda,'current_stream',lambda *args:Stream())
    monkeypatch.setattr(torch.cuda,'device',lambda *args:contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda,'stream',lambda *args:contextlib.nullcontext())
    empty_like=torch.empty_like
    monkeypatch.setattr(torch,'empty_like',lambda *args,**kwargs:
        empty_like(*args,**{k:v for k,v in kwargs.items() if k!='pin_memory'}))
    monkeypatch.setattr(torch.Tensor,'pin_memory',lambda tensor:tensor)


def test_full_transport_jobs_progress_while_two_original_acks_are_blocked(monkeypatch,background_io):
    owner=transport(monkeypatch,background_io,reuse_io=False,ready_before_cleanup=True,
        binary_control_channel=True,async_layer_jobs=True)
    cuda_policy(monkeypatch)
    release=threading.Event()
    calls=[]
    for layer in range(4):
        for head in range(4):
            owner.cache[layer][head][1]=torch.full((2,128),layer*4+head+1,dtype=torch.float16)
    class Record:
        profile={}
        def __init__(self,layer):self.identity=SimpleNamespace(shard_rank=0);self.layer=layer
        async def ack(self):
            calls.append(('ack',self.layer,threading.get_ident()))
            while not release.is_set():await asyncio.sleep(0.001)
        async def close(self):calls.append(('close',self.layer,threading.get_ident()));return True
    async def fetch(state,ticket,*args):
        assert state['owner_thread']==threading.get_ident()==owner.async_jobs.thread.ident
        calls.append(('fetch',ticket.layer,threading.get_ident()))
        state.update(pending_cleanup=[Record(ticket.layer)],gpu_readers=[],fresh_gpu_rows={},delivery_profiles=[])
        return ((1,),(1,),(1,),(1,)),4,0
    monkeypatch.setattr(owner,'_select_and_fetch',fetch)
    lookahead=LayerLookahead('request','incarnation',layers=28,timeout=5)
    try:
        for layer in range(4):
            lookahead.publish(0,layer,owner.job(torch.zeros((28,128)),None))
            bank=lookahead.consume(0,layer)
            assert torch.equal(bank.keys[:,0,0],torch.arange(layer*4+1,layer*4+5,dtype=torch.float16))
        assert sum(c[0]=='fetch' for c in calls)==4 and not any(c[0]=='close' for c in calls)
        assert owner.async_jobs.snapshot()['cleanup_peak']==2
        assert len(owner.workers)==4
        release.set();lookahead.close();owner.close()
        assert not owner.workers and not owner.async_jobs.thread.is_alive()
        assert all(c[2]==owner.async_jobs.thread.ident for c in calls)
        assert owner._io_counts['worker_loops_created']==1
    finally:
        release.set();lookahead.close()
        if not owner.closed:owner.close()
