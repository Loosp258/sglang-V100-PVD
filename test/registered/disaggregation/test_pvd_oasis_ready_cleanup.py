"""Actual bounded threads and same-thread cleanup; no CUDA/native timing claim."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import contextlib
from types import SimpleNamespace

import pytest
from sglang.srt.disaggregation.pvd.oasis_ready_cleanup import OwnedReadyCleanup
from sglang.srt.disaggregation.pvd.oasis_transport import OasisLayerTransport
from sglang.srt.disaggregation.pvd.oasis_pipeline import LayerLookahead
from test_pvd_oasis_transport_io import transport, background_io
import torch


def test_ready_precedes_slow_cleanup_and_close_joins_original_owner():
    pool = OwnedReadyCleanup(workers=1, capacity=2)
    entered, release = threading.Event(), threading.Event()
    threads, retained = [], [object()]
    def work(publish):
        threads.append(threading.get_ident())
        publish(retained[0])
        entered.set()
        assert release.wait(5)
        threads.append(threading.get_ident())
        retained.clear()
    future = pool.submit(work, published=time.monotonic(), timeout=5)
    reply, started, ready = future.result(timeout=5)
    assert reply is retained[0] and started <= ready and entered.wait(5)
    assert pool.snapshot()['live'] == 1
    with ThreadPoolExecutor(max_workers=1) as closing:
        joined = closing.submit(pool.close)
        assert not joined.done() and retained
        release.set()
        joined.result(timeout=5)
    assert not retained and threads[0] == threads[1]


def test_public_cancellation_cannot_cancel_owned_cleanup_and_capacity_is_bounded():
    pool = OwnedReadyCleanup(workers=1, capacity=2)
    entered, release = threading.Event(), threading.Event()
    done = []
    def work(publish):
        entered.set()
        assert release.wait(5)
        publish('bank')
        done.append(threading.get_ident())
    first = pool.submit(work, published=time.monotonic(), timeout=5)
    assert entered.wait(5)
    second = pool.submit(work, published=time.monotonic(), timeout=5)
    assert first.cancel() and second.cancel()
    with pytest.raises(RuntimeError, match='capacity'):
        pool.submit(work, published=time.monotonic(), timeout=5)
    release.set()
    pool.close()
    assert len(done) == 2 and done[0] == done[1]


@pytest.mark.parametrize('after_ready', [False, True])
def test_cleanup_failure_latches_and_close_reports_even_after_ready(after_ready):
    pool = OwnedReadyCleanup(workers=1, capacity=1)
    def work(publish):
        if after_ready:
            publish('valid-independent-bank')
        raise RuntimeError('ACK failure')
    future = pool.submit(work, published=time.monotonic(), timeout=5)
    if after_ready:
        assert future.result(timeout=5)[0] == 'valid-independent-bank'
    else:
        with pytest.raises(RuntimeError, match='ACK failure'):
            future.result(timeout=5)
    with pytest.raises(RuntimeError, match='failed'):
        pool.close()
    with pytest.raises(RuntimeError):
        pool.submit(work, published=time.monotonic(), timeout=5)


def test_expired_queued_work_does_not_create_native_owner():
    pool = OwnedReadyCleanup(workers=1, capacity=1)
    calls = []
    future = pool.submit(lambda publish: calls.append(True), published=time.monotonic()-2, timeout=1)
    with pytest.raises(TimeoutError):
        future.result(timeout=5)
    with pytest.raises(RuntimeError):
        pool.close()
    assert not calls


@pytest.mark.parametrize('failure', ['ack', 'unknown', 'close', None])
def test_transport_owned_cleanup_attempts_all_records_preserves_unknown(failure):
    owner = object.__new__(OasisLayerTransport)
    owner.quarantined = False
    calls = []
    thread = threading.get_ident()
    class Record:
        def __init__(self, rank):
            self.identity = SimpleNamespace(shard_rank=rank)
            self.profile = {'ack_calls': 0}
        async def ack(self):
            assert threading.get_ident() == thread
            calls.append(('ack', self.identity.shard_rank))
            self.profile['ack_calls'] += 1
            if self.identity.shard_rank == 0 and failure == 'ack':
                raise RuntimeError('ACK lost')
        async def close(self):
            assert threading.get_ident() == thread
            calls.append(('close', self.identity.shard_rank))
            if self.identity.shard_rank == 0 and failure == 'close':
                raise RuntimeError('unregister unknown')
            return not (self.identity.shard_rank == 0 and failure == 'unknown')
    state = dict(pending_cleanup=[Record(0), Record(1)],
                 delivery_profiles=[dict(rank=0), dict(rank=1)])
    if failure:
        with pytest.raises(RuntimeError, match='cleanup failed'):
            asyncio.run(owner._finish_owned_cleanup(state))
    else:
        asyncio.run(owner._finish_owned_cleanup(state))
    assert calls == [('ack',0),('close',0),('ack',1),('close',1)]
    assert owner.quarantined == (failure in ('unknown', 'close'))
    assert all(p['ack_calls'] == 1 for p in state['delivery_profiles'])


@pytest.mark.parametrize('fused_channel', [False, True])
def test_full_job_ready_before_ack_same_worker_using_explicit_cpu_cuda_policy(
    monkeypatch, background_io, fused_channel
):
    owner = transport(monkeypatch, background_io, reuse_io=False, ready_before_cleanup=True,
                      binary_control_channel=fused_channel)
    entered, release = threading.Event(), threading.Event()
    calls = []
    class Event:
        def record(self, *args): pass
        def synchronize(self): calls.append(('gpu-complete-policy', threading.get_ident()))
    class Stream:
        def wait_event(self, event): pass
        def synchronize(self): pass
    monkeypatch.setattr(torch.cuda, 'Stream', lambda **kwargs: Stream())
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *args: Stream())
    monkeypatch.setattr(torch.cuda, 'device', lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, 'stream', lambda *args: contextlib.nullcontext())
    original_empty = torch.empty_like
    monkeypatch.setattr(torch, 'empty_like', lambda *args, **kwargs:
        original_empty(*args, **{k:v for k,v in kwargs.items() if k != 'pin_memory'}))
    monkeypatch.setattr(torch.Tensor, 'pin_memory', lambda tensor: tensor)
    for head in range(4):
        owner.cache[0][head][1] = torch.full((2,128), head+1, dtype=torch.float16)
    class Record:
        identity = SimpleNamespace(shard_rank=0)
        profile = dict(ack_calls=0)
        async def ack(self):
            calls.append(('ack', threading.get_ident()));entered.set()
            assert await asyncio.to_thread(release.wait, 5)
            self.profile['ack_calls'] += 1
        async def close(self):
            calls.append(('close', threading.get_ident()));return True
    async def fetch(state, *args):
        state.update(pending_cleanup=[Record()], gpu_readers=[], fresh_gpu_rows={}, delivery_profiles=[])
        return ((1,),(1,),(1,),(1,)), 4, 0
    monkeypatch.setattr(owner, '_select_and_fetch', fetch)
    lookahead = LayerLookahead('request', 'incarnation', layers=28, timeout=5)
    try:
        lookahead.publish(0,0,owner.job(torch.zeros((28,128)),None))
        bank = lookahead.consume(0,0)
        assert entered.wait(5) and owner.workers and owner.cleanup.snapshot()['live'] == 1
        assert bank.ids == ((1,),(1,),(1,),(1,))
        assert torch.equal(bank.keys[:,0,0],torch.tensor([1,2,3,4],dtype=torch.float16))
        assert calls[0][0] == 'gpu-complete-policy' and calls[1][0] == 'ack'
        release.set()
        lookahead.close();owner.close()
        assert calls[0][1] == calls[1][1] == calls[2][1] and not owner.workers
    finally:
        release.set();lookahead.close()
        if not owner.closed: owner.close()
