"""Driver ABI and call-order contracts, no driver or hardware execution."""

import ctypes
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd import cuda_receive_ordering as ordering
from sglang.srt.disaggregation.pvd.sparse_receiver import SparseReceiveError


@pytest.mark.parametrize("failure", [None, "set", "get", "flag"])
def test_pointer_attribute_abi_preserves_64_bit_address_and_checks_readback(
    monkeypatch, failure
):
    events = []
    pointer = 0xABCDEF012345
    signature = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
    )

    def set_value(value, attr, address):
        events.append(
            (
                "set",
                address,
                attr,
                ctypes.cast(value, ctypes.POINTER(ctypes.c_uint)).contents.value,
            )
        )
        return 1 if failure == "set" else 0

    def get_value(value, attr, address):
        events.append(("get", address, attr))
        ctypes.cast(value, ctypes.POINTER(ctypes.c_uint)).contents.value = (
            0 if failure == "flag" else 1
        )
        return 2 if failure == "get" else 0

    library = SimpleNamespace(
        cuPointerSetAttribute=signature(set_value),
        cuPointerGetAttribute=signature(get_value),
    )
    monkeypatch.setattr(ordering.sys, "platform", "linux")
    monkeypatch.setattr(ordering.ctypes, "CDLL", lambda name: library)
    driver = ordering._PointerAttributes()
    if failure is None:
        driver.enable_sync_memops(pointer)
    else:
        with pytest.raises(SparseReceiveError):
            driver.enable_sync_memops(pointer)
    assert events[0] == ("set", pointer, 6, 1)
    assert len(events) == (1 if failure == "set" else 2)
    if len(events) == 2:
        assert events[1] == ("get", pointer, 6)


@pytest.mark.parametrize("pointer", [0, -1, True, 2**64, 1.5])
def test_invalid_pointer_refused_before_driver(pointer):
    driver = ordering._PointerAttributes.__new__(ordering._PointerAttributes)
    with pytest.raises(SparseReceiveError, match="pointer"):
        driver.enable_sync_memops(pointer)


@pytest.mark.parametrize("device", ["cpu", "cuda", "meta"])
def test_explicit_device_is_required(device):
    with pytest.raises(SparseReceiveError, match="indexed CUDA"):
        ordering.CUDAReceiveOrdering(device)


@pytest.mark.skipif(
    not torch.cuda.is_available() or ordering.sys.platform != "linux",
    reason="native SYNC_MEMOPS requires Linux CUDA; no RDMA is emulated",
)
def test_real_cuda_receive_sync_memops_roundtrip():
    buffer = torch.empty(64, dtype=torch.uint8, device="cuda:0")
    policy = ordering.CUDAReceiveOrdering("cuda:0")
    policy.prepare(buffer)  # real driver set + readback, not a double
    buffer.fill_(37)  # local CUDA producer, explicitly NOT an RDMA write
    policy.after_remote_write(SimpleNamespace(buffer=buffer))
    assert buffer.cpu().tolist() == [37] * 64
