"""Release ownership contracts; real Scheduler/cache execution is a WSL gate."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from sglang.srt.disaggregation.pvd.cpu_batch_dispatch import CPUBatchDispatcher
from sglang.srt.disaggregation.pvd.cpu_batch_forward import CPUBatchForwardExecutor
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError
from sglang.srt.disaggregation.pvd.cpu_request_release import CPURequestRelease
from test_pvd_cpu_decode_lifecycle import env


def binding(life):
    # Explicit executor/pool doubles, not a serving model or allocator test.
    executor = CPUBatchForwardExecutor.__new__(CPUBatchForwardExecutor)
    executor.dispatcher = CPUBatchDispatcher(life.arbiter)
    executor._storage = {life: 1}
    executor.runner = NS(
        req_to_token_pool=NS(req_to_token=torch.zeros((3, 16))),
        token_to_kv_pool_allocator=object(),
    )
    req = NS(rid=life.request_id, req_pool_idx=1, finished=lambda: False)
    cache = NS(
        is_chunk_cache=lambda: True,
        req_to_token_pool=executor.runner.req_to_token_pool,
        token_to_kv_pool_allocator=executor.runner.token_to_kv_pool_allocator,
    )
    calls = []

    def release(r, c, *, is_insert):
        assert not life.arbiter.busy and life._permit is None
        assert life not in executor._storage
        calls.append((r, c, is_insert))
        r.req_pool_idx = None

    return CPURequestRelease(req, life, executor), req, cache, release, calls


def test_actual_common_release_entry_dispatches_only_explicit_owner():
    # Execute the actual wrapper source, avoiding common.py's GPU import chain.
    source = Path("python/sglang/srt/mem_cache/common.py").read_text(encoding="utf-8")
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "release_kv_cache"
    )
    namespace = {"Req": object, "BasePrefixCache": object}
    exec(  # noqa: S102 -- compile only this checkout's trusted wrapper function
        compile(ast.Module(body=[fn], type_ignores=[]), "common.py", "exec"), namespace
    )
    with env() as (life, f, _):
        life.admit(f.request)
        guard, req, cache, release, calls = binding(life)
        namespace["_release_kv_cache_now"] = release
        namespace["release_kv_cache"](req, cache, False)
        assert guard.state == "pending" and not calls
        asyncio.run(guard.progress())
        assert len(calls) == 1
        namespace["release_kv_cache"](req, cache, False)
        assert len(calls) == 1
    plain = NS()
    seen = []
    namespace["_release_kv_cache_now"] = lambda *a, **kw: seen.append((a, kw))
    namespace["release_kv_cache"](plain, cache)
    assert seen == [((plain, cache), {"is_insert": True})]
    plain.pvd_cpu_kv_release = object()
    with pytest.raises(TypeError, match="release owner"):
        namespace["release_kv_cache"](plain, cache)
    assert len(seen) == 1


def test_forward_and_result_lease_hold_pool_until_drain_and_release_once():
    async def run():
        with env() as (life, f, _):
            life.admit(f.request)
            guard, req, cache, release, calls = binding(life)
            permit = life.begin_decode()
            guard.defer(req, cache, True, release)
            guard.defer(req, cache, False, release)
            assert life.state == "aborted"
            assert not await guard.progress() and not calls
            assert req.req_pool_idx == 1
            life.complete_decode(permit, 21)  # cancelled output discarded
            assert await guard.progress()
            assert await guard.progress()
            guard.defer(req, cache, True, release)
            assert len(calls) == 1 and calls[0][2] is False
            assert life.committed_tokens == 0 and req.req_pool_idx is None

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["error", "cancel"])
def test_drain_failure_retains_storage_and_allows_retry(monkeypatch, failure):
    async def run():
        with env() as (life, f, _):
            life.admit(f.request)
            guard, req, cache, release, calls = binding(life)
            guard.defer(req, cache, False, release)
            original = life.close
            entered, resume = asyncio.Event(), asyncio.Event()

            async def delayed():
                entered.set()
                await resume.wait()
                raise RuntimeError("remote ownership unknown")

            monkeypatch.setattr(life, "close", delayed)
            task = asyncio.create_task(guard.progress())
            await entered.wait()
            with pytest.raises(LifecycleError, match="already in flight"):
                await guard.progress()
            if failure == "cancel":
                task.cancel()
                error = asyncio.CancelledError
            else:
                resume.set()
                error = RuntimeError
            with pytest.raises(error):
                await task
            assert guard.state == "pending" and not calls
            assert guard.executor._storage[life] == req.req_pool_idx == 1
            monkeypatch.setattr(life, "close", original)
            assert await guard.progress()
            assert len(calls) == 1

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["raise", "slot_not_released"])
def test_partial_allocator_failure_is_quarantined_and_never_replayed(failure):
    async def run():
        with env() as (life, f, _):
            life.admit(f.request)
            guard, req, cache, _, calls = binding(life)

            def failed_release(*args, **kwargs):
                calls.append("partially freed")
                if failure == "raise":
                    raise RuntimeError("allocator failed after side effects")

            guard.defer(req, cache, False, failed_release)
            with pytest.raises((RuntimeError, LifecycleError)):
                await guard.progress()
            assert guard.state == "quarantined"
            with pytest.raises(LifecycleError, match="quarantined"):
                await guard.progress()
            with pytest.raises(LifecycleError, match="quarantined"):
                guard.defer(req, cache, False, failed_release)
            assert calls == ["partially freed"]

    asyncio.run(run())


@pytest.mark.parametrize("change", ["req", "slot", "incarnation", "cache", "radix"])
def test_wrong_binding_cannot_release_anything(change):
    with env() as (life, f, _):
        life.admit(f.request)
        guard, req, cache, release, calls = binding(life)
        if change == "req":
            req = NS(**req.__dict__)
        elif change == "slot":
            req.req_pool_idx = 2
        elif change == "incarnation":
            life.incarnation = "different"
        elif change == "cache":
            cache = NS(
                req_to_token_pool=object(),
                token_to_kv_pool_allocator=object(),
                is_chunk_cache=lambda: True,
            )
        else:
            cache.is_chunk_cache = lambda: False
        with pytest.raises(LifecycleError):
            guard.defer(req, cache, False, release)
        assert not calls


@pytest.mark.parametrize("change", ["slot", "incarnation", "target_busy"])
def test_binding_and_target_lease_rechecked_after_async_drain(monkeypatch, change):
    async def run():
        with env() as (life, f, _):
            life.admit(f.request)
            guard, req, cache, release, calls = binding(life)
            guard.defer(req, cache, False, release)
            original = life.close
            leases = []

            async def drain_then_change():
                await original()
                if change == "slot":
                    req.req_pool_idx = 2
                elif change == "incarnation":
                    life.incarnation = "new"
                else:
                    leases.append(life.arbiter.acquire())

            monkeypatch.setattr(life, "close", drain_then_change)
            try:
                if change == "target_busy":
                    assert not await guard.progress()
                else:
                    with pytest.raises(LifecycleError, match="changed"):
                        await guard.progress()
                assert not calls and guard.state == "pending"
                assert life in guard.executor._storage
            finally:
                for lease in leases:
                    life.arbiter.release(lease)

    asyncio.run(run())
