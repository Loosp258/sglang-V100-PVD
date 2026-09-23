"""Native parent-limiter contract; full allocation proof needs real cuVS."""

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.cagra_backend import (
    CagraAutoIndexBackend,
    CagraIndexBackend,
    CagraNativeRuntime,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferBudget,
    TransferCapacityError,
)
from test_pvd_cagra_backend import Runtime, args, backend
from test_pvd_prompt_index import budgeted, stored_entry


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


def _shared_backend(*, intermediate_degree=2, cap=4096):
    runtime = Runtime()
    runtime.global_native_cap_bytes = cap
    runtime.global_limit = object()
    return backend(
        runtime, intermediate_degree=intermediate_degree,
        global_native_cap_bytes=cap,
    )


def test_shared_cap_refuses_runtime_without_native_parent_limiter():
    runtime = Runtime()
    runtime.global_native_cap_bytes = 4096
    with pytest.raises(ValueError, match="lacks a native root limiter"):
        CagraIndexBackend(
            device="cpu", native_bytes_per_index=4096,
            graph_degree=1, intermediate_degree=2, itopk_size=4,
            global_native_cap_bytes=4096, _runtime=runtime,
        )


def test_parent_cap_charged_once_before_entry_build_and_held_after_close():
    native = _shared_backend()
    assert native.shared_footprint == 4096
    assert native.build_footprint(8, 3, metric="ip") == 0
    manager, budget = budgeted(native)
    assert manager.shared_native_budget_bytes == 4096
    assert budget.snapshot()["used_staging_bytes"] == 4096
    store, manifest, _, _ = stored_entry(manager)
    assert store.progress_prompt_indexes()["built"] == 1
    record = manager._entries[manifest.key.transfer_id]
    vectors = sum(item.vectors.numel() * item.vectors.element_size()
                  for item in record.vectors.values())
    assert budget.snapshot()["used_staging_bytes"] == 4096 + vectors
    store.release_entry(manifest.key)
    assert budget.snapshot()["used_staging_bytes"] == 4096
    assert not native._owners


def test_auto_short_exact_copies_are_additional_to_shared_native_cap():
    auto = CagraAutoIndexBackend(_shared_backend(intermediate_degree=16))
    manager, budget = budgeted(auto)
    store, manifest, _, _ = stored_entry(manager, prompt_tokens=2)
    assert store.progress_prompt_indexes()["built"] == 1
    assert budget.snapshot()["used_staging_bytes"] > 4096
    store.release_entry(manifest.key)
    assert budget.snapshot()["used_staging_bytes"] == 4096


def test_shared_budget_uses_distinct_owner_for_each_manager():
    shared = TransferBudget(8192, 1)
    from sglang.srt.disaggregation.pvd.prompt_index import PromptIndexManager

    first = PromptIndexManager(vector_space="model", backend=_shared_backend(),
                               budget=shared)
    second = PromptIndexManager(vector_space="model", backend=_shared_backend(),
                                budget=shared)
    assert first._shared_owner != second._shared_owner
    assert shared.snapshot()["used_staging_bytes"] == 8192
    with pytest.raises(TransferCapacityError):
        PromptIndexManager(vector_space="model", backend=_shared_backend(),
                           budget=shared)


def test_shared_native_cap_cli_requires_one_index_cap_and_total_budget():
    from sglang.srt.disaggregation.pvd.server import _validate_args

    argv = [
        "--prompt-index-backend", "cagra-auto",
        "--prompt-index-vector-space", "model",
        "--prompt-index-budget-bytes", "8192",
        "--prompt-index-cagra-native-bytes", "4096",
        "--prompt-index-cagra-global-native-bytes", "4096",
    ]
    _validate_args(args(*argv))
    for value in ("2048", "16384"):
        changed = list(argv)
        changed[-1] = value
        with pytest.raises(ValueError, match="shared CAGRA"):
            _validate_args(args(*changed))
    exact = list(argv)
    exact[1] = "exact"
    with pytest.raises(ValueError, match="requires a CAGRA backend"):
        _validate_args(args(*exact))
