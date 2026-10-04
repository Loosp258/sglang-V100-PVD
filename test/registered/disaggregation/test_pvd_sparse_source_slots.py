"""Actual CPU packing/PUT bytes, explicit delayed transport, actual CUDA skips."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd import server, vector_store
from sglang.srt.disaggregation.pvd.request_state import DeliveryState
from sglang.srt.disaggregation.pvd.sparse_copy import copy_sparse_kv_into
from sglang.srt.disaggregation.pvd.sparse_source_slots import SparseSourceSlotPool, SparseSourceSlotUnknown
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget, TransferCapacityError, TransportState
from sglang.srt.disaggregation.pvd.v_source_profile import VSourceProfile, copy_source_profile
from test_pvd_sparse_store_delivery import ready, destination
from test_pvd_selected_component_views import component_case
from test_pvd_cuda_sparse_packing import args, cuda_policy


def pooled_store(slots=2):
    store, entry, raw, manifest, budget = ready()
    store.reuse_sparse_source_slots = True
    store.sparse_source_slots = slots
    store.sparse_source_slot_bytes = manifest.nbytes
    return store, entry, raw, manifest, budget


def start(store, entry, manifest, operation):
    manifest = replace(manifest, specs=tuple(replace(s, operation_id=operation) for s in manifest.specs))
    target = destination(store, manifest)
    delivery = store.reserve_delivery(entry.key, operation, target.descriptor)
    store.start_delivery(entry.key, operation)
    return target, delivery


def test_repeated_real_cpu_deliveries_reuse_physical_registration_and_exact_wire():
    store, entry, raw, manifest, budget = pooled_store()
    engine, source_mrs, targets = store.transfer_engine, [], []
    try:
        for i in range(12):
            target, delivery = start(store, entry, manifest, str(i)); targets.append(target)
            local = engine.pending[delivery.transfer_handle.transfer_id][0]
            source_mrs.append(local.registration)
            assert local.registration is delivery.source_slot_lease._slot.registration
            assert local.length == manifest.nbytes
            calls = store._sparse_source_pool.snapshot()['acquired_leases']
            store.start_delivery(entry.key, str(i))
            assert store._sparse_source_pool.snapshot()['acquired_leases'] == calls
            engine.finish(delivery.transfer_handle); store.poll_delivery(entry.key, str(i))
            for p in delivery.sparse_manifest.payload_views(target.buffer):
                s=p.spec
                expected=torch.stack([x[s.layer][list(s.token_ids),s.kv_head] for x in (raw.k_buffer,raw.v_buffer)])
                assert torch.equal(p.tensor.view(torch.uint8), expected.contiguous().view(torch.uint8))
            prof=copy_source_profile(delivery.source_profile.snapshot(),nbytes=manifest.nbytes)
            assert prof['reuse_source_slots'] is True and prof['source_slot_reused'] is (i > 0)
            assert prof['physical_allocate_calls']==prof['physical_register_calls']==int(i==0)
            assert delivery.source_slot_lease is None and delivery.staging_guard.value is None
            assert budget.snapshot()['used_staging_bytes']==manifest.nbytes
            store.ack_delivery(entry.key,str(i))
        assert all(r is source_mrs[0] for r in source_mrs)
        pool=store._sparse_source_pool.snapshot()
        assert pool['physical_register_calls']==1 and pool['returned_leases']==12
        assert source_mrs[0].descriptor.region_id not in engine.released
        store.close()
        assert engine.released.count(source_mrs[0].descriptor.region_id)==1
        assert budget.snapshot()['used_staging_bytes']==0
    finally:
        store.close()
        for target in targets: engine.release_memory(target)


def test_two_pending_leases_capacity_rejection_and_safe_reuse():
    store,entry,_,manifest,budget=pooled_store()
    a,da=start(store,entry,manifest,'a'); b,db=start(store,entry,manifest,'b')
    c,dc=start(store,entry,manifest,'full')
    assert dc.state==DeliveryState.FAILED and dc.local_terminal==TransportState.NOT_SUBMITTED
    assert da.source_slot_lease._slot is not db.source_slot_lease._slot
    assert store._sparse_source_pool.snapshot()['leased_slots']==2
    assert budget.snapshot()['used_staging_bytes']==2*manifest.nbytes
    first=da.source_slot_lease._slot.registration
    store.transfer_engine.finish(da.transfer_handle);store.progress_transfers()
    d,dd=start(store,entry,manifest,'reuse')
    assert dd.source_slot_lease._slot.registration is first
    for delivery in (db,dd): store.transfer_engine.finish(delivery.transfer_handle)
    store.progress_transfers();store.close()
    assert budget.snapshot()['used_staging_bytes']==0
    for target in (a,b,c,d):store.transfer_engine.release_memory(target)


def test_terminal_success_still_retains_slot_until_adapter_cleanup(monkeypatch):
    store,entry,_,manifest,_=pooled_store(1)
    target,d=start(store,entry,manifest,'delay-cleanup')
    engine=store.transfer_engine;cleanup=engine.cleanup_complete
    engine.finish(d.transfer_handle)
    monkeypatch.setattr(engine,'cleanup_complete',lambda handle:False)
    store.progress_transfers()
    assert d.state==DeliveryState.DELIVERED
    assert d.source_slot_lease is not None and not d.progress_settled
    assert store._sparse_source_pool.snapshot()['leased_slots']==1
    monkeypatch.setattr(engine,'cleanup_complete',cleanup)
    store.progress_transfers()
    assert d.source_slot_lease is None and d.progress_settled
    store.close();engine.release_memory(target)


@pytest.mark.parametrize('result',['cancel','failed','short','unknown'])
def test_cancel_native_failure_and_unknown_slot_ownership(result):
    store,entry,_,manifest,budget=pooled_store(1)
    target,d=start(store,entry,manifest,result)
    slot=d.source_slot_lease._slot
    if result=='cancel':store.cancel_entry(entry.key,'cancel pending')
    if result=='unknown':
        d.transfer_handle.transport_state=TransportState.UNKNOWN
    else:
        store.transfer_engine.finish(d.transfer_handle,result!='failed')
        if result=='short':d.transfer_handle.transferred_bytes-=1
    store.progress_transfers()
    if result=='unknown':
        assert d.source_slot_lease is not None and slot.unknown
        assert store._sparse_source_pool.snapshot()['unknown_slots']==1
        store.close()
        assert budget.snapshot()['used_staging_bytes']==manifest.nbytes
        assert not store.entries[entry.key].resources_released
    else:
        assert d.source_slot_lease is None
        assert slot.registration.descriptor.region_id not in store.transfer_engine.released
        store.close();assert budget.snapshot()['used_staging_bytes']==0
    store.transfer_engine.release_memory(target)


@pytest.mark.parametrize('failure',['allocate','register','copy','unregister'])
def test_preparation_failures_and_sticky_unregister_unknown(monkeypatch,failure):
    store,entry,_,manifest,budget=pooled_store(1)
    engine=store.transfer_engine
    # Destination belongs to D and is prepared before source failures are injected.
    target=destination(store,manifest)
    d=store.reserve_delivery(entry.key,'failure',target.descriptor)
    def fail(*a,**k):raise RuntimeError(failure)
    if failure=='allocate':monkeypatch.setattr(vector_store.torch,'empty',fail)
    if failure=='register':monkeypatch.setattr(engine,'register_memory',fail)
    if failure=='copy':monkeypatch.setattr(vector_store,'copy_sparse_kv_into',fail)
    store.start_delivery(entry.key,'failure')
    if failure=='register':
        assert d.local_terminal==TransportState.UNKNOWN
        assert store._sparse_source_pool.snapshot()['unknown_slots']==1
        assert budget.snapshot()['used_staging_bytes']==manifest.nbytes
    elif failure in ('allocate','copy'):
        assert d.local_terminal==TransportState.NOT_SUBMITTED
        assert store._sparse_source_pool.snapshot()['leased_slots']==0
    else:
        engine.finish(d.transfer_handle);store.progress_transfers()
        physical=store._sparse_source_pool._slots[0].registration
        release=engine.release_memory
        monkeypatch.setattr(engine,'release_memory',lambda reg:fail() if reg is physical else release(reg))
        store.close()
        assert budget.snapshot()['used_staging_bytes']==manifest.nbytes
        assert store._sparse_source_pool.snapshot()['physical_release_calls']==1
        monkeypatch.setattr(engine,'release_memory',release)
        store.close()
        assert store._sparse_source_pool.snapshot()['physical_release_calls']==1
    store.close()
    if failure in ('allocate','copy'):assert budget.snapshot()['used_staging_bytes']==0
    engine.release_memory(target)


@pytest.mark.parametrize('rank',[0,1])
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32])
def test_source_pool_uses_physical_mr_and_exact_length_cpu_bytes(rank,dtype):
    source,kw,oracle=component_case(rank,dtype,'multi')
    engine=FakeTransferEngine();budget=TransferBudget(1<<20,8)
    pool=SparseSourceSlotPool(engine,budget,device='cpu',rank=rank,rail='r',capacity_bytes=kw['manifest'].nbytes+256)
    target=engine.register_memory(torch.zeros(kw['manifest'].nbytes,dtype=torch.uint8),endpoint='D',rank=rank,rail='r')
    lease=pool.acquire('job',kw['manifest'].nbytes)
    copy_sparse_kv_into(source,lease.buffer,**kw)
    handle=engine.submit_put(lease.local,target.descriptor)
    assert engine.cleanup_complete(handle)
    for p,expected in zip(kw['manifest'].payload_views(target.buffer),oracle,strict=True):
        assert torch.equal(p.tensor.view(torch.uint8),expected.contiguous().view(torch.uint8))
    lease.release_after_proof()
    with pytest.raises(ValueError):lease.release_after_proof()
    pool.close();engine.release_memory(target)
    assert budget.snapshot()['used_staging_bytes']==0


def test_atomic_capacity_budget_and_live_close():
    engine=FakeTransferEngine();budget=TransferBudget(64,2)
    pool=SparseSourceSlotPool(engine,budget,device='cpu',rank=0,rail='r',slots=2,capacity_bytes=64)
    a=pool.acquire('a',32)
    with pytest.raises(TransferCapacityError):pool.acquire('b',32)
    assert pool.snapshot()['physical_slots']==1
    with pytest.raises(SparseSourceSlotUnknown):pool.close()
    assert pool.snapshot()['physical_slots']==1 and a._active
    a.release_after_proof();pool.close()
    assert budget.snapshot()['used_staging_bytes']==0


def test_default_and_invalid_server_modes():
    assert server.build_parser().get_default('experimental_reuse_sparse_source_slots') is False
    for name in ('allow_cpu_for_tests','experimental_triton_sparse_packing',
                 'experimental_contiguous_sparse_packing','experimental_direct_sparse_batch_put',
                 'experimental_reuse_sparse_pack_fence','experimental_indexed_sparse_packing'):
        configured=args('--experimental-cuda-sparse-packing','--experimental-reuse-sparse-source-slots',
            '--prompt-index-vector-space','test','--prompt-index-budget-bytes','1024')
        setattr(configured,name,True)
        with pytest.raises(ValueError):server._validate_args(configured)
    server._validate_args(args('--experimental-cuda-sparse-packing','--experimental-reuse-sparse-source-slots',
        '--prompt-index-vector-space','test','--prompt-index-budget-bytes','1024',
        '--transfer-staging-budget-bytes','65536'))


def test_source_slot_option_wires_both_ranks_without_gpu_substitution(monkeypatch):
    configured=args('--experimental-cuda-sparse-packing','--experimental-reuse-sparse-source-slots')
    configured.transfer_backend='fake'  # construction wiring only, no serving qualification
    monkeypatch.setattr(server,'_build_prompt_index',lambda _:object())
    monkeypatch.setattr(server,'VectorKVStore',lambda **kw:SimpleNamespace(**kw))
    for rank in (0,1):
        store,_=server._create_store(configured,rank=rank,local_rank=rank,rails=['r0','r1'])
        assert store.reuse_sparse_source_slots and store.sparse_source_slots==2
        assert store.sparse_source_slot_bytes==32768


@pytest.mark.parametrize('unknown',[False,True])
def test_explicit_cpu_cuda_policy_keeps_original_fences_and_unknown_slot(monkeypatch,unknown):
    store,entry,manifest,budget,target,d,_=cuda_policy(monkeypatch)
    store.reuse_sparse_source_slots=True;store.sparse_source_slot_bytes=manifest.nbytes
    calls=[]
    def sync(device):
        calls.append(device)
        if unknown and len(calls)==1:raise RuntimeError('policy-only pack completion unknown')
    monkeypatch.setattr(torch.cuda,'synchronize',sync)
    store.start_delivery(entry.key,d.delivery_id)
    assert len(calls)==2  # original store fences; fake engine does not qualify native readiness
    if unknown:
        assert d.local_terminal==TransportState.UNKNOWN and d.packing_index_lease is not None
        assert store._sparse_source_pool.snapshot()['unknown_slots']==1
        store.close();assert budget.snapshot()['used_staging_bytes']==manifest.nbytes
    else:
        store.transfer_engine.finish(d.transfer_handle);store.progress_transfers();store.close()
        assert budget.snapshot()['used_staging_bytes']==0
    store.transfer_engine.release_memory(target)


def test_concurrent_acquire_never_shares_live_storage_or_overallocates():
    engine=FakeTransferEngine();budget=TransferBudget(256,8)
    pool=SparseSourceSlotPool(engine,budget,device='cpu',rank=0,rail='r',slots=2,capacity_bytes=128)
    barrier=threading.Barrier(4)
    def acquire(i):
        barrier.wait()
        try:return pool.acquire(str(i),64)
        except TransferCapacityError:return None
    with ThreadPoolExecutor(max_workers=4) as workers:
        results=list(workers.map(acquire,range(4)))
    leases=[r for r in results if r is not None]
    assert len(leases)==2 and len({r.buffer.data_ptr() for r in leases})==2
    assert pool.snapshot()['physical_register_calls']==2
    for lease in leases:lease.release_after_proof()
    pool.close();assert budget.snapshot()['used_staging_bytes']==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA device unavailable; no native transport substitution')
@pytest.mark.parametrize('rank',[0,1])
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32])
def test_actual_cuda_nondefault_stream_pool_pack_with_fake_copy_transport(rank,dtype):
    source,kw,oracle=component_case(rank,dtype,'multi')
    device=torch.device('cuda:0');engine=FakeTransferEngine();budget=TransferBudget(1<<20,8)
    pool=SparseSourceSlotPool(engine,budget,device=device,rank=rank,rail='r',capacity_bytes=kw['manifest'].nbytes)
    stream=torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        source=source.to(device);lease=pool.acquire('cuda-job',kw['manifest'].nbytes)
        copy_sparse_kv_into(source,lease.buffer,**kw,allow_cuda=True)
        complete=torch.cuda.Event();complete.record(stream);complete.synchronize()
        actual=lease.buffer.cpu()
    for p,expected in zip(kw['manifest'].payload_views(actual),oracle,strict=True):
        assert torch.equal(p.tensor.view(torch.uint8),expected.contiguous().view(torch.uint8))
    lease.release_after_proof();pool.close()
