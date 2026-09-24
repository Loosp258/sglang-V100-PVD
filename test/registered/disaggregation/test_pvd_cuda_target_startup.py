"""Startup is tested with injected constructors; no GPU is claimed here."""

import threading
from types import SimpleNamespace as NS

import pytest
from sglang.srt.disaggregation.pvd import cuda_target_startup as startup
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import (
    LifecycleError,
    TargetExecutionArbiter,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


def setup(monkeypatch, *, fail=None):
    events = []
    native = object()
    req_pool, kv_pool = object(), object()
    allocator = NS(get_kvcache=lambda: kv_pool)
    runner = NS(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        attn_backend=native,
    )
    scheduler = NS(
        tp_worker=NS(model_runner=runner),
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        tree_cache=NS(req_to_token_pool=req_pool, token_to_kv_pool_allocator=allocator),
        pvd_cuda_binding=None,
    )
    manager = NS(kv_pool=kv_pool, scheduler=scheduler)
    scheduler.disagg_decode_prealloc_queue = NS(kv_manager=manager)

    class Workspace:
        def __init__(self, **kw):
            events.append("workspace")
            self.closed = False

        def close(self):
            events.append("workspace.close")
            if fail == "close":
                raise RuntimeError("uncertain CUDA cleanup")
            self.closed = True

        def snapshot(self):
            return {"quarantine": None, "active": False}

    class Driver:
        def __init__(self, arbiter, **kw):
            events.append("driver")
            self.arbiter = arbiter
            self._records = {}

        def _owner(self):
            self.arbiter.owner()

        def begin_shutdown(self):
            events.append("driver.begin_shutdown")

        def close_loop(self):
            events.append("driver.close_loop")

    class Executor:
        def __init__(self, consumer, arbiter, **kw):
            events.append("executor")
            self.consumer = consumer
            self._active = self._quarantined = False

    class Binding:
        def __init__(self, scheduler, driver, executor, **kw):
            events.append("binding")
            self.driver = driver
            assert runner.attn_backend.consumer is executor.consumer
            if fail == "binding-before":
                raise RuntimeError("binding failed")
            if fail == "binding-pinned":
                kw["pool_owner"].pin("binding-construction")
                raise RuntimeError("binding pinned pools before failure")
            scheduler.pvd_cuda_binding = self
            if fail == "binding-after":
                raise RuntimeError("binding published then failed")

        def close(self):
            events.append("binding.close")
            self.driver.close_loop()
            self.closed = True

    def build_backend(runner, **kw):
        events.append("backend")
        if fail in ("backend", "close"):
            raise RuntimeError("backend failed")
        return NS(
            consumer=NS(
                req_pool=req_pool,
                kv_pool=kv_pool,
                _lock=kw["execution_lock"],
            )
        )

    monkeypatch.setattr(startup, "CUDASparseAttentionWorkspace", Workspace)
    monkeypatch.setattr(startup, "make_cuda_sparse_backend", build_backend)
    monkeypatch.setattr(startup, "CUDARefreshDriver", Driver)
    monkeypatch.setattr(startup, "CUDARankBatchExecutor", Executor)
    monkeypatch.setattr(startup, "CUDADecodeSchedulerBinding", Binding)
    kwargs = {
        "device": "cuda:0",
        "dtype": "half",
        "head_dim": 128,
        "chunk_tokens": 32,
        "attention_budget": TransferBudget(65536, 1),
        "output_budget": TransferBudget(65536, 1),
        "max_batch_size": 4,
        "max_requests": 4,
        "max_prefix_tokens": 4096,
        "retire_pools": lambda: events.append("pools.retire"),
    }
    return scheduler, runner, native, kwargs, events


def test_success_assembles_exact_pools_and_retirement(monkeypatch):
    scheduler, runner, native, kwargs, events = setup(monkeypatch)
    shared_lock, shared_arbiter = threading.RLock(), TargetExecutionArbiter()
    kwargs.update(execution_lock=shared_lock, arbiter=shared_arbiter)
    owner = startup.install_cuda_target_components(scheduler, **kwargs)
    assert scheduler.pvd_cuda_binding is owner.binding
    assert runner.attn_backend is owner.backend
    assert owner.pool_owner.value.req_pool is scheduler.req_to_token_pool
    assert (
        owner.pool_owner.value.kv_pool
        is runner.token_to_kv_pool_allocator.get_kvcache()
    )
    assert owner.execution_lock is owner.backend.consumer._lock
    assert owner.driver.arbiter is owner.arbiter
    assert owner.execution_lock is shared_lock
    assert owner.arbiter is shared_arbiter
    assert "pools.retire" not in events
    owner.retire_drained()
    assert events[-5:] == [
        "driver.begin_shutdown",
        "binding.close",
        "driver.close_loop",
        "workspace.close",
        "pools.retire",
    ]
    assert runner.attn_backend is owner.backend
    assert runner.attn_backend is not native


@pytest.mark.parametrize("fail", ["backend", "binding-before"])
def test_failure_rolls_back_only_unpublished_backend(monkeypatch, fail):
    scheduler, runner, native, kwargs, events = setup(monkeypatch, fail=fail)
    with pytest.raises(RuntimeError):
        startup.install_cuda_target_components(scheduler, **kwargs)
    assert runner.attn_backend is native
    assert scheduler.pvd_cuda_binding is None
    assert "workspace.close" in events
    assert "pools.retire" not in events
    if fail == "binding-before":
        assert events.index("driver.begin_shutdown") < events.index("workspace.close")


def test_published_or_uncertain_cleanup_retains_owners(monkeypatch):
    for failure in ("binding-after", "binding-pinned", "close"):
        scheduler, runner, native, kwargs, events = setup(monkeypatch, fail=failure)
        before = len(startup._STARTUP_QUARANTINE)
        with pytest.raises(RuntimeError):
            startup.install_cuda_target_components(scheduler, **kwargs)
        assert len(startup._STARTUP_QUARANTINE) == before + 1
        assert "pools.retire" not in events
        if failure in ("binding-after", "binding-pinned"):
            assert runner.attn_backend is not native
            assert (scheduler.pvd_cuda_binding is not None) == (
                failure == "binding-after"
            )


def test_missing_retirement_or_foreign_pool_is_rejected_before_allocation(monkeypatch):
    scheduler, runner, native, kwargs, events = setup(monkeypatch)
    kwargs["retire_pools"] = None
    with pytest.raises(LifecycleError, match="retirement"):
        startup.install_cuda_target_components(scheduler, **kwargs)
    kwargs["retire_pools"] = lambda: None
    runner.token_to_kv_pool_allocator = object()
    with pytest.raises(LifecycleError, match="exact pools"):
        startup.install_cuda_target_components(scheduler, **kwargs)
    assert not events
    assert runner.attn_backend is native
