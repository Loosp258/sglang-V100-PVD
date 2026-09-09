"""Native CUDA/RDMA boundaries are doubles; lifecycle/registration code is real."""

import dataclasses
import gc
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch

from test_pvd_mooncake_metadata import transport  # noqa: F401

from sglang.srt.disaggregation.pvd.protocol import RemoteRegionDescriptor
from sglang.srt.disaggregation.pvd.transfer_engine import (
    FakeTransferEngine,
    MemorySlice,
    RegisteredMemory,
    TransferStatus,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferBudget,
    TransportState,
)


class CudaBuffer:
    is_cuda = True
    device = "cuda:0"

    def is_contiguous(self):
        return True

    def numel(self):
        return 16

    def element_size(self):
        return 1

    def data_ptr(self):
        return 4096


class NativeStub:
    def __init__(self, submit_result=7, statuses=()):
        self.submit_result = submit_result
        self.statuses = list(statuses)
        self.submit_calls = []
        self.check_calls = []
        self.unregister_calls = []
        self.unregister_result = 0

    def register_memory(self, address, length):
        return 0

    def unregister_memory(self, address):
        self.unregister_calls.append(address)
        return self.unregister_result

    def transfer_submit_write(self, *args):
        self.submit_calls.append(args)
        if isinstance(self.submit_result, Exception):
            raise self.submit_result
        return self.submit_result

    def transfer_check_status(self, native_id):
        self.check_calls.append(native_id)
        result = self.statuses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def make_adapter_with_source(transport, native, budget=None):
    wrapper = SimpleNamespace(
        engine=native,
        require_pvd_metadata_policy=lambda: None,
        get_session_id=lambda: "v:1",
        pvd_metadata_version="0.3.13.post1",
    )
    adapter = transport.adapter.MooncakePVDTransferEngine.from_existing(
        wrapper, rail="mlx5_2", budget=budget or TransferBudget(64, 4)
    )
    registration = adapter.register_memory(
        CudaBuffer(), endpoint="v:1", rank=0, rail="mlx5_2"
    )
    remote = RemoteRegionDescriptor("d:2", "target", 8192, 16, "cuda:0", 0, "mlx5_2")
    return adapter, MemorySlice(registration, 0, 16), remote


def test_native_terminal_is_polled_only_once(transport):
    native = NativeStub(statuses=[0, -2, 1])
    adapter, local, remote = make_adapter_with_source(transport, native)
    handle = adapter.submit_put(local, remote)
    assert handle.status == TransferStatus.PENDING
    assert handle.transport_state == TransportState.IN_FLIGHT
    adapter.abort(handle)
    adapter.release_memory(local.registration)
    assert native.unregister_calls == []
    for state in (TransportState.IN_FLIGHT, TransportState.DRAINING,
                  TransportState.TERMINAL_SUCCESS, TransportState.TERMINAL_SUCCESS):
        assert adapter.poll(handle) == TransferStatus.CANCELLED
        assert handle.transport_state == state
    assert native.check_calls == [7, 7, 7]
    assert native.submit_calls == [("d:2", 4096, 8192, 16)]
    assert native.unregister_calls == [4096]
    assert adapter.lifecycle_manager.snapshot()["used_inflight"] == 0
    assert adapter.lifecycle_manager.snapshot()["used_staging_bytes"] == 0


@pytest.mark.parametrize("result", [0, RuntimeError("partial submit")])
def test_untrackable_submit_quarantines_source_and_capacity(transport, result):
    native = NativeStub(submit_result=result)
    adapter, local, remote = make_adapter_with_source(transport, native)
    handle = adapter.submit_put(local, remote)
    adapter.release_memory(local.registration)
    assert handle.transport_state == TransportState.UNKNOWN
    assert handle.status == TransferStatus.FAILED
    adapter.abort(handle)
    adapter.poll(handle)
    assert native.check_calls == []
    assert native.unregister_calls == []
    assert adapter.lifecycle_manager.snapshot()["used_inflight"] == 1


def test_unknown_submit_quarantines_source_from_later_admission(transport):
    native = NativeStub(submit_result=0)
    adapter, local, remote = make_adapter_with_source(transport, native)
    first = adapter.submit_put(local, remote)
    second = adapter.submit_put(local, remote)
    assert first.transport_state == TransportState.UNKNOWN
    assert second.status == TransferStatus.FAILED
    assert second.transport_state == TransportState.NOT_SUBMITTED
    assert len(native.submit_calls) == 1
    assert adapter.lifecycle_manager.snapshot()["used_inflight"] == 1


def test_unknown_submit_gate_blocks_concurrent_admission_before_native(transport):
    class BlockingNative(NativeStub):
        def __init__(self):
            super().__init__(submit_result=0)
            self.first_submit_started = threading.Event()
            self.allow_first_submit = threading.Event()

        def transfer_submit_write(self, *args):
            self.submit_calls.append(args)
            self.first_submit_started.set()
            assert self.allow_first_submit.wait(timeout=5)
            return 0

    native = BlockingNative()
    adapter, local, remote = make_adapter_with_source(transport, native)
    results = []
    first = threading.Thread(target=lambda: results.append(adapter.submit_put(local, remote)))
    first.start()
    assert native.first_submit_started.wait(timeout=5)
    second = threading.Thread(target=lambda: results.append(adapter.submit_put(local, remote)))
    second.start()
    assert second.is_alive()
    native.allow_first_submit.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive()
    assert not second.is_alive()
    assert len(native.submit_calls) == 1
    assert sorted((handle.transport_state for handle in results), key=str) == [
        TransportState.NOT_SUBMITTED,
        TransportState.UNKNOWN,
    ]
    assert adapter.lifecycle_manager.snapshot()["used_inflight"] == 1


@pytest.mark.parametrize("result", [99, RuntimeError("poll failed")])
def test_untrackable_poll_quarantines_without_repoll(transport, result):
    native = NativeStub(statuses=[result])
    adapter, local, remote = make_adapter_with_source(transport, native)
    handle = adapter.submit_put(local, remote)
    adapter.release_memory(local.registration)
    adapter.poll(handle)
    adapter.poll(handle)
    assert handle.transport_state == TransportState.UNKNOWN
    assert native.check_calls == [7]
    assert native.unregister_calls == []


@pytest.mark.parametrize("result,state,status", [
    (1, TransportState.TERMINAL_SUCCESS, TransferStatus.SUCCESS),
    (-1, TransportState.TERMINAL_FAILED, TransferStatus.FAILED),
])
def test_native_terminal_concurrent_polls_free_once(transport, result, state, status):
    native = NativeStub(statuses=[result])
    adapter, local, remote = make_adapter_with_source(transport, native)
    handle = adapter.submit_put(local, remote)
    errors = []

    def poll():
        try:
            adapter.poll(handle)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=poll) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert errors == []
    assert native.check_calls == [7]
    assert handle.status == status
    assert handle.transport_state == state
    assert handle.transferred_bytes == (16 if result == 1 else 0)


def test_shared_wrapper_keeps_one_manager_and_owns_abandoned_source(transport):
    native = NativeStub(statuses=[1])
    adapter, local, remote = make_adapter_with_source(transport, native)
    other = transport.adapter.MooncakePVDTransferEngine.from_existing(
        adapter._engine, rail="mlx5_2"
    )
    assert other.lifecycle_manager is adapter.lifecycle_manager
    handle = adapter.submit_put(local, remote)
    source_ref = weakref.ref(local.registration.buffer)
    adapter.release_memory(local.registration)
    del local, adapter
    gc.collect()
    assert source_ref() is not None
    other.poll(handle)
    gc.collect()
    assert source_ref() is None


def test_shared_wrapper_rejects_a_different_injected_budget(transport):
    native = NativeStub()
    adapter, _, _ = make_adapter_with_source(
        transport, native, budget=TransferBudget(64, 4)
    )
    with pytest.raises(ValueError, match="different budget"):
        transport.adapter.MooncakePVDTransferEngine.from_existing(
            adapter._engine, rail="mlx5_2", budget=TransferBudget(64, 4)
        )


def test_unregister_failure_retains_tensor_and_can_retry_without_native_poll(transport):
    native = NativeStub(statuses=[1])
    adapter, local, remote = make_adapter_with_source(transport, native)
    handle = adapter.submit_put(local, remote)
    source_collected = []
    _source_ref = weakref.ref(
        local.registration.buffer, lambda _: source_collected.append(True)
    )
    native.unregister_result = -1
    adapter.release_memory(local.registration)
    del local
    adapter.poll(handle)
    gc.collect()
    assert source_collected == []
    assert handle.transport_state == TransportState.TERMINAL_SUCCESS
    assert adapter.health()["registered_regions"] == 1
    native.unregister_result = 0
    adapter.poll(handle)
    gc.collect()
    assert source_collected == [True]
    assert native.check_calls == [7]
    assert adapter.health()["registered_regions"] == 0


def test_terminal_poll_retains_capacity_while_release_retry_is_running(transport):
    class RetryNative(NativeStub):
        def __init__(self):
            super().__init__(statuses=[1])
            self.retry_started = threading.Event()
            self.allow_retry_to_fail = threading.Event()

        def unregister_memory(self, address):
            self.unregister_calls.append(address)
            if len(self.unregister_calls) == 2:
                self.retry_started.set()
                assert self.allow_retry_to_fail.wait(timeout=5)
            return -1 if len(self.unregister_calls) < 3 else 0

    native = RetryNative()
    adapter, local, remote = make_adapter_with_source(transport, native)
    handle = adapter.submit_put(local, remote)
    adapter.release_memory(local.registration)
    adapter.poll(handle)
    assert native.unregister_calls == [4096]

    retry_errors = []

    def retry_release():
        try:
            adapter.release_memory(local.registration)
        except RuntimeError as exc:
            retry_errors.append(exc)

    retry_thread = threading.Thread(target=retry_release)
    retry_thread.start()
    assert native.retry_started.wait(timeout=5)

    adapter.poll(handle)
    snapshot = adapter.lifecycle_manager.snapshot()
    assert snapshot["used_inflight"] == 1
    assert snapshot["tracked_transfers"] == 1

    native.allow_retry_to_fail.set()
    retry_thread.join(timeout=5)
    assert not retry_thread.is_alive()
    assert len(retry_errors) == 1

    adapter.poll(handle)
    assert native.check_calls == [7]
    assert native.unregister_calls == [4096, 4096, 4096]
    assert adapter.lifecycle_manager.snapshot()["used_inflight"] == 0


@pytest.mark.parametrize("fault", ["rail", "bounds", "identity", "released", "cuda", "budget"])
def test_pre_submit_failures_never_enter_native_or_pin_source(transport, monkeypatch, fault):
    native = NativeStub()
    budget = TransferBudget(16, 1)
    adapter, local, remote = make_adapter_with_source(transport, native, budget)
    if fault == "rail":
        remote = dataclasses.replace(remote, rail="other")
    elif fault == "bounds":
        remote = dataclasses.replace(remote, length=8)
    elif fault == "identity":
        local = MemorySlice(RegisteredMemory(local.registration.descriptor, CudaBuffer()), 0, 16)
    elif fault == "released":
        adapter.release_memory(local.registration)
    elif fault == "cuda":
        def fail(*args):
            raise RuntimeError("cuda sync")
        monkeypatch.setattr(torch.cuda, "synchronize", fail)
    else:
        budget.reserve("other", 16, 1)
    handle = adapter.submit_put(local, remote)
    assert handle.status == TransferStatus.FAILED
    assert handle.transport_state == TransportState.NOT_SUBMITTED
    assert native.submit_calls == []
    assert budget.snapshot()["used_inflight"] == (1 if fault == "budget" else 0)


def test_missing_async_api_fails_startup(transport):
    wrapper = SimpleNamespace(engine=object(), require_pvd_metadata_policy=lambda: None)
    with pytest.raises(RuntimeError, match="async"):
        transport.adapter.MooncakePVDTransferEngine.from_existing(
            wrapper, rail="mlx5_2", budget=TransferBudget(16, 1)
        )


def test_fake_copy_marks_terminal_success():
    engine = FakeTransferEngine()
    source = engine.register_memory(torch.ones(16, dtype=torch.uint8), endpoint="a", rank=0, rail="r")
    target = engine.register_memory(torch.zeros(16, dtype=torch.uint8), endpoint="b", rank=0, rail="r")
    try:
        handle = engine.submit_put(MemorySlice(source, 0, 16), target.descriptor)
        assert handle.transport_state == TransportState.TERMINAL_SUCCESS
        assert torch.equal(source.buffer, target.buffer)
    finally:
        engine.release_memory(source)
        engine.release_memory(target)


def test_fake_copy_failure_marks_terminal_failure():
    engine = FakeTransferEngine()
    source = engine.register_memory(
        torch.ones(16, dtype=torch.uint8), endpoint="a", rank=0, rail="r"
    )
    try:
        missing_remote = dataclasses.replace(source.descriptor, region_id="missing")
        handle = engine.submit_put(MemorySlice(source, 0, 16), missing_remote)
        assert handle.status == TransferStatus.FAILED
        assert handle.transport_state == TransportState.TERMINAL_FAILED
    finally:
        engine.release_memory(source)
