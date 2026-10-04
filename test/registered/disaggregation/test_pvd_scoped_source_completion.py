"""Explicit CUDA/native policy doubles, plus a separate actual CUDA event gate."""
from dataclasses import replace
import pytest
import torch
from sglang.srt.disaggregation.pvd import vector_store
from sglang.srt.disaggregation.pvd.transfer_engine import CudaSourceReady, MemorySlice, RegisteredMemory, TransferStatus
from sglang.srt.disaggregation.pvd.protocol import RemoteRegionDescriptor
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransportState
from test_pvd_cuda_sparse_packing import cuda_policy
from test_pvd_mooncake_lifecycle import transport, NativeStub, make_adapter_with_source


class PolicyStream:
    device = 'cuda:0'


class PolicyEvent:
    device = 'cuda:0'
    def __init__(self): self.calls = 0
    def synchronize(self): self.calls += 1


@pytest.mark.parametrize('bad', [None, 'registration', 'slice', 'buffer', 'event', 'device'])
def test_adapter_requires_exact_source_capability_and_waits_before_native(transport, monkeypatch, bad):
    native = NativeStub(statuses=[1])
    adapter, local, remote = make_adapter_with_source(transport, native)
    event, stream = PolicyEvent(), PolicyStream()
    monkeypatch.setattr(torch.cuda, 'Event', PolicyEvent)
    monkeypatch.setattr(torch.cuda, 'Stream', PolicyStream)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args:
        pytest.fail('valid capability must not use a device fence'))
    proof = CudaSourceReady(local.registration, local.registration.buffer, 0,16,event,stream)
    if bad == 'registration': proof=replace(proof, registration=RegisteredMemory(local.registration.descriptor,local.registration.buffer))
    if bad == 'slice': proof=replace(proof,length=8)
    if bad == 'buffer': proof=replace(proof,buffer=object())
    if bad == 'event': proof=replace(proof,event=object())
    if bad == 'device': stream.device='cuda:1'
    handle = adapter.submit_put(replace(local,source_ready=proof),remote)
    if bad:
        assert handle.status is TransferStatus.FAILED and not native.submit_calls
    else:
        assert event.calls == 1 and len(native.submit_calls) == 1
        adapter.poll(handle)
    adapter.release_memory(local.registration)


@pytest.mark.parametrize('failure', [None, 'copy', 'cancel', 'event'])
def test_store_scoped_pack_keeps_leases_through_event_and_unknown(monkeypatch, failure):
    store, entry, manifest, budget, target, delivery, _ = cuda_policy(monkeypatch)
    store.scoped_sparse_source_completion = True
    index = store.prompt_index._entries[entry.key.transfer_id]
    events = []
    class Event:
        def record(self, stream):
            assert index.users == 1; events.append('record')
        def synchronize(self):
            events.append('event-wait')
            assert index.users == 1
            if failure == 'event': raise RuntimeError('unknown CUDA completion')
    monkeypatch.setattr(torch.cuda,'Event',Event)
    producer = object()
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device: producer)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *args:events.append('global-fallback'))
    original = vector_store.copy_sparse_kv_into
    def copy(*args, **kwargs):
        original(*args,**kwargs);events.append('copy')
        if failure == 'copy': raise RuntimeError('partial copy')
        if failure == 'cancel': store.cancel_entry(entry.key,'cancel while packing')
    monkeypatch.setattr(vector_store,'copy_sparse_kv_into',copy)
    store.start_delivery(entry.key,delivery.delivery_id)
    assert events[:3] == ['copy','record','event-wait']
    assert delivery.packing_stream is producer
    if failure == 'event':
        assert delivery.local_terminal is TransportState.UNKNOWN and index.users == 1
        assert delivery.staging_guard.value is not None
    else:
        assert index.users == 0 and delivery.packing_completion_proven
        assert 'global-fallback' not in events
        if failure in ('copy','cancel'):
            assert delivery.local_terminal is TransportState.NOT_SUBMITTED
        else:
            store.transfer_engine.finish(delivery.transfer_handle);store.progress_transfers()
            store.ack_delivery(entry.key,delivery.delivery_id)
        assert budget.snapshot()['used_staging_bytes'] == 0
    store.close();store.transfer_engine.release_memory(target)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA unavailable')
def test_actual_nondefault_producer_event_and_exact_slice():
    device=torch.device('cuda:0');producer=torch.cuda.Stream(device=device)
    with torch.cuda.stream(producer):
        buffer=torch.arange(512,device=device,dtype=torch.float32)
        buffer.mul_(3)
        event=torch.cuda.Event();event.record(producer)
    registration=RegisteredMemory(RemoteRegionDescriptor('v','source',buffer.data_ptr(),
        2048,str(device),0,'rail'),buffer)
    proof=CudaSourceReady(registration,buffer,0,2048,event,producer)
    local=MemorySlice(registration,0,2048,source_ready=proof)
    proof.wait(local)
    assert torch.equal(buffer.cpu(),torch.arange(512,dtype=torch.float32)*3)
    with pytest.raises(ValueError): proof.wait(replace(local,length=1024))
