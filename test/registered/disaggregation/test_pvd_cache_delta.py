"""Exact delta sets, bounded owners, replay and loss/resync semantics."""
import copy
import json
import numpy as np
import pytest
from sglang.srt.disaggregation.pvd.cache_delta import (
    CacheDeltaSender,CacheDeltaReceiver,CacheDeltaResync,full_resync_delta,delta_scratch_bytes)
from sglang.srt.disaggregation.pvd.cache_snapshot import cache_ids,pack_cache_snapshot
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


def scope(n=8192,layer=0,heads=None):
    return dict(prompt_tokens=n,layer=layer,heads=heads or [0,1],cached=[[],[]])


@pytest.mark.parametrize('n',[1,8,2159,8192,32768])
def test_snapshots_equal_old_oracle_and_confirm_independent_of_ack(n):
    sender=CacheDeltaSender(n);budget=TransferBudget(32<<20,4)
    receiver=CacheDeltaReceiver(n,[0,1],budget)
    valid=np.zeros(n,dtype=bool)
    try:
        for start,end in ((0,min(8,n)),(min(8,n),min(32,n)),(min(32,n),n)):
            for token in range(start,end):
                sender.publish(0,0,token);sender.publish(0,1,n-1-token);valid[token]=True
            cached,delta=sender.snapshot(0,[0,1])
            assert cached[0]==pack_cache_snapshot(valid)
            expanded,proof,commit=receiver.prepare(scope(n),delta);commit()
            assert expanded['cached']==cached
            sender.confirm(delta,proof)
            cached,empty=sender.snapshot(0,[0,1])
            assert all(s['added']==[] and s['full'] is None for s in empty['states'])
            again,_,commit=receiver.prepare(scope(n),empty);commit()
            assert again['cached']==cached
        assert delta_scratch_bytes(n)==112*(2*n+5*((n+7)//8)+1024)
    finally:receiver.close()
    assert budget.snapshot()['used_staging_bytes']==0


def test_lost_confirmation_replays_union_but_reordered_old_delta_needs_full():
    sender=CacheDeltaSender(64);receiver=CacheDeltaReceiver(64,[0,1],TransferBudget(1<<20,4))
    try:
        _,first=sender.snapshot(0,[0,1]);_,proof,commit=receiver.prepare(scope(64),first);commit();sender.confirm(first,proof)
        sender.publish(0,0,3)
        old_cached,old=sender.snapshot(0,[0,1]);_,_,commit=receiver.prepare(scope(64),old);commit()
        sender.publish(0,0,9)
        cached,new=sender.snapshot(0,[0,1]);expanded,proof,commit=receiver.prepare(scope(64),new);commit()
        assert expanded['cached']==cached
        with pytest.raises(CacheDeltaResync):receiver.prepare(scope(64),old)
        expanded,_,commit=receiver.prepare(scope(64),full_resync_delta(old,old_cached));commit()
        assert expanded['cached']==old_cached and receiver.states[(0,0)][0]==2
        sender.confirm(new,proof)
        receiver.states.clear()
        _,lost=sender.snapshot(0,[0,1])
        with pytest.raises(CacheDeltaResync):receiver.prepare(scope(64),lost)
        restored,_,commit=receiver.prepare(scope(64),full_resync_delta(lost,cached));commit()
        assert restored['cached']==cached
    finally:receiver.close()


@pytest.mark.parametrize('damage',['digest','version','bool','ids','head','layer'])
def test_bad_delta_never_changes_committed_set(damage):
    sender=CacheDeltaSender(64);receiver=CacheDeltaReceiver(64,[0,1],TransferBudget(1<<20,4))
    try:
        _,first=sender.snapshot(0,[0,1]);_,proof,commit=receiver.prepare(scope(64),first);commit();sender.confirm(first,proof)
        sender.publish(0,0,3)
        _,delta=sender.snapshot(0,[0,1]);bad=copy.deepcopy(delta)
        if damage=='digest':bad['states'][0]['digest']='0'*64
        elif damage=='version':bad['states'][0]['version']=2
        elif damage=='bool':bad['states'][0]['version']=True
        elif damage=='ids':bad['states'][0]['added']=[3,3]
        elif damage=='head':bad['heads']=[2,3]
        else:bad['layer']=1
        with pytest.raises(ValueError):receiver.prepare(scope(64),bad)
        assert receiver.states[(0,0)][0]==0
    finally:receiver.close()


def test_dense_unchanged_wire_is_smaller_without_full_prompt_scans(monkeypatch):
    sender=CacheDeltaSender(32768);receiver=CacheDeltaReceiver(32768,[0,1],TransferBudget(32<<20,4))
    try:
        for head in (0,1):
            for token in range(32768):sender.publish(0,head,token)
        cached,first=sender.snapshot(0,[0,1]);_,proof,commit=receiver.prepare(scope(32768),first);commit();sender.confirm(first,proof)
        def fail(*args,**kwargs):raise AssertionError('D should not scan Prompt validity')
        monkeypatch.setattr(np,'flatnonzero',fail)
        _,delta=sender.snapshot(0,[0,1])
        assert len(json.dumps(delta,separators=(',',':')).encode())<len(json.dumps(cached).encode())/10
    finally:receiver.close()
