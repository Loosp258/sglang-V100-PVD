"""Native parent-limiter contract; full allocation proof needs real cuVS."""

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.cagra_backend import CagraNativeRuntime


def test_child_limit_uses_one_shared_parent_and_rejects_oversized_child(monkeypatch):
    class Limit:
        def __init__(self, upstream, cap):
            self.upstream, self.cap, self.used = upstream, cap, 7

        def get_allocated_bytes(self):
            return self.used

    root = Limit(object(), 256)
    runtime = object.__new__(CagraNativeRuntime)
    runtime.device = torch.device("cpu")
    runtime.lock = threading.RLock()
    runtime.global_limit = root
    runtime.global_native_cap_bytes = 256
    runtime.mr = SimpleNamespace(
        CudaMemoryResource=lambda: object(),
        LimitingResourceAdaptor=Limit,
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())

    first, second = runtime.create(128), runtime.create(128)
    assert first.limit is not second.limit
    assert first.limit.upstream is second.limit.upstream is root
    assert runtime.global_allocated_bytes() == 7
    with pytest.raises(ValueError, match="exceeds shared"):
        runtime.create(257)
