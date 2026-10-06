"""Cleanup overlap, failure and cancellation preserve both rank owners."""
import asyncio
from types import SimpleNamespace
import pytest
from sglang.srt.disaggregation.pvd.oasis_transport import OasisLayerTransport


@pytest.mark.parametrize('failure', [False, True])
@pytest.mark.parametrize('cancel', [False, True])
def test_both_rank_ack_enter_before_release_and_close_joined(failure, cancel):
    async def run():
        owner = object.__new__(OasisLayerTransport)
        owner.parallel_owned_cleanup, owner.quarantined = True, False
        entered = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()
        closed = []
        def record(rank):
            async def ack():
                entered[rank].set()
                await release.wait()
                if failure and rank == 0:
                    raise RuntimeError('lost rank0 ACK')
            async def close():
                closed.append(rank)
                return True
            return SimpleNamespace(identity=SimpleNamespace(shard_rank=rank), ack=ack, close=close, profile={'ack_calls': 1})
        state = {'pending_cleanup': [record(0), record(1)], 'delivery_profiles': [{'rank': 0}, {'rank': 1}]}
        task = asyncio.create_task(owner._finish_owned_cleanup(state))
        await asyncio.wait_for(asyncio.gather(*(e.wait() for e in entered)), 1)
        assert not task.done() and not closed
        if cancel:
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError): await task
        elif failure:
            with pytest.raises(RuntimeError, match='cleanup failed'): await task
        else:
            await task
        assert sorted(closed) == [0, 1]
        assert all(p['ack_calls'] == 1 for p in state['delivery_profiles'])
    asyncio.run(run())


def test_parallel_cleanup_config_requires_owned_ready(monkeypatch, tmp_path):
    from test_pvd_delivery_followup_config import load, config
    cfg = config(5)
    assert load(monkeypatch, tmp_path, cfg)['parallel_owned_cleanup'] is False
    cfg['parallel_owned_cleanup'] = True
    with pytest.raises(ValueError, match='READY cleanup'): load(monkeypatch, tmp_path, cfg)
    cfg['ready_before_cleanup'] = True
    assert load(monkeypatch, tmp_path, cfg)['parallel_owned_cleanup'] is True
    cfg['parallel_owned_cleanup'] = 1
    with pytest.raises(ValueError): load(monkeypatch, tmp_path, cfg)
