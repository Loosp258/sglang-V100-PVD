"""CUDA retirement policy on CPU tensors; real source allocator/cache methods."""

import __future__

import ast
import asyncio
import threading
from array import array
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from sglang.srt.disaggregation.pvd import cuda_request_release as release_module
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import (
    LifecycleError,
    TargetExecutionArbiter,
)
from sglang.srt.disaggregation.pvd.cuda_model_attention import CUDAModelPools
from sglang.srt.disaggregation.pvd.cuda_prefetch_request import CUDAPrefetchRequest
from sglang.srt.disaggregation.pvd.cuda_refresh_driver import CUDARefreshDriver
from sglang.srt.disaggregation.pvd.cuda_request_release import CUDARequestRelease
from sglang.srt.disaggregation.pvd.transfer_lifecycle import ResourceGuard


def source(path, name, cls=None, namespace=None):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    nodes = (
        tree.body
        if cls is None
        else next(
            n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
        )
    )
    fn = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    env = {"torch": torch, **(namespace or {})}
    exec(
        compile(
            ast.Module(body=[fn], type_ignores=[]),
            path,
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        env,
    )
    return env[name]


class RequestPool:
    alloc = source(
        "python/sglang/srt/mem_cache/memory_pool.py", "alloc", "ReqToTokenPool"
    )
    free = source(
        "python/sglang/srt/mem_cache/memory_pool.py", "free", "ReqToTokenPool"
    )


class Allocator:
    alloc = source(
        "python/sglang/srt/mem_cache/allocator/token.py",
        "alloc",
        "TokenToKVPoolAllocator",
    )
    free = source(
        "python/sglang/srt/mem_cache/allocator/token.py",
        "free",
        "TokenToKVPoolAllocator",
    )


class Cache:
    cache_finished_req = source(
        "python/sglang/srt/mem_cache/chunk_cache.py", "cache_finished_req", "ChunkCache"
    )


@contextmanager
def case(monkeypatch, *, real=False):
    pool = RequestPool()
    pool.req_to_token = torch.zeros((3, 16), dtype=torch.int32)
    pool.req_to_token[1, :8] = torch.arange(1, 9)
    pool.free_slots = [2]
    kv = object()
    allocator = Allocator()
    allocator.device, allocator.need_sort = "cpu", False
    allocator.get_kvcache = lambda: kv
    allocator.free_pages = torch.arange(9, 17, dtype=torch.int64)
    allocator.is_not_in_free_group, allocator.free_group = True, []
    cache = Cache()
    cache.req_to_token_pool, cache.token_to_kv_pool_allocator = pool, allocator
    cache.page_size = 1
    req = NS(
        rid="r",
        origin_input_ids=array("q", [1] * 8),
        output_ids=array("q", [2]),
        req_pool_idx=1,
        is_retracted=False,
        finished=lambda: False,
        kv_allocated_len=8,
        kv_committed_len=8,
    )
    req.pop_committed_kv_cache = lambda: 8
    req.pop_overallocated_kv_cache = lambda: (8, 8)
    if real:
        try:
            from sglang.srt.managers import schedule_batch as schedule_module
            from sglang.srt.managers.schedule_batch import Req
            from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
            from sglang.srt.mem_cache.cache_init_params import CacheInitParams
            from sglang.srt.mem_cache.chunk_cache import ChunkCache
            from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
            from sglang.srt.sampling.sampling_params import SamplingParams
        except Exception as exc:
            pytest.skip(f"real pool imports unavailable: {type(exc).__name__}: {exc}")
        monkeypatch.setattr(
            schedule_module,
            "get_global_server_args",
            lambda: NS(strip_thinking_cache=False),
        )
        pool = ReqToTokenPool(2, 16, "cpu", False)
        allocator = TokenToKVPoolAllocator(16, torch.float32, "cpu", kv, False)
        cache = ChunkCache(CacheInitParams(True, pool, allocator, 1))
        params = SamplingParams(max_new_tokens=16, ignore_eos=True)
        params.normalize(None)
        req = Req("r", "", [1] * 8, params, vocab_size=32)
        req.output_ids.append(2)
        pool.alloc([req])
        pool.req_to_token[req.req_pool_idx, :8] = allocator.alloc(8).to(torch.int32)
        req.kv_allocated_len = req.kv_committed_len = 8
    driver = CUDARefreshDriver(
        TargetExecutionArbiter(), max_requests=1, max_prefix_tokens=32
    )
    control = object.__new__(CUDAPrefetchRequest)
    control.group = NS(
        coordinator=NS(
            identity=("r", "inc", "entry"),
            snapshot=lambda: {
                "state": "idle",
                "installed_tokens": 0,
                "next_boundary": 4,
                "lead_tokens": 1,
            },
        )
    )
    control._metadata, control._routes = {0: {"prompt_tokens": 8}}, {0: None}
    control._active, control._tasks = None, ()
    control.pipeline = NS(
        _lock=threading.RLock(),
        _quarantined=False,
        probe=NS(_quarantined=False),
        provider=NS(degraded=False),
        draft_config=NS(predict_tokens=1),
    )
    control._session = NS(_copy_unknown=False)
    control.can_decode, control.cancel = lambda n: True, lambda reason: None

    async def close():
        return None

    control.aclose = close
    driver.register(req, control, clients={0: object()}, timeout_seconds=5)
    events = []
    guard = ResourceGuard(
        CUDAModelPools(pool, kv), lambda: events.append("pool-released")
    )
    release = source(
        "python/sglang/srt/mem_cache/common.py",
        "_release_kv_cache_now",
        namespace={
            "get_global_server_args": lambda: NS(
                page_size=1, speculative_algorithm=None, strip_thinking_cache=False
            ),
            "HybridReqToTokenPool": type("Hybrid", (), {}),
        },
    )
    if not real:
        monkeypatch.setattr(release_module, "_require_supported_pools", lambda c: None)
    monkeypatch.setattr(torch.cuda, "device", lambda d: nullcontext())
    owner = CUDARequestRelease(req, driver, cache, pool_owner=guard, release=release)
    monkeypatch.setattr(owner, "_synchronize", lambda: events.append("fence"))
    wrapper = source(
        "python/sglang/srt/mem_cache/common.py",
        "release_kv_cache",
        namespace={"_release_kv_cache_now": release},
    )
    try:
        yield NS(
            pool=pool,
            allocator=allocator,
            cache=cache,
            req=req,
            driver=driver,
            control=control,
            owner=owner,
            guard=guard,
            events=events,
            release=release,
            wrapper=wrapper,
        )
    finally:
        if not driver.arbiter.busy and not any(
            r.quarantined for r in driver._records.values()
        ):
            driver.begin_shutdown()
            for _ in range(12):
                driver.poll()
                if driver.snapshot()["drained"]:
                    break
        # CPU fault fixture only. Production close_loop refuses retained owners.
        assert not asyncio.all_tasks(driver._loop)
        driver._loop.close()


def drain(c):
    for _ in range(12):
        c.driver.poll()
        if not c.driver._records or c.driver._records["r"].quarantined:
            return
    pytest.fail("retirement failed to make progress")


def test_original_cache_entry_defers_then_fences_and_releases_exactly_once(monkeypatch):
    with case(monkeypatch) as c:
        c.wrapper(c.req, c.cache, False)
        assert c.owner.state == "pending" and c.req.req_pool_idx == 1
        assert len(c.allocator.free_pages) == 8 and not c.events
        c.guard.request_release()
        drain(c)
        assert c.owner.state == "released" and c.req.req_pool_idx is None
        assert c.pool.free_slots == [2, 1]
        assert sorted(c.allocator.free_pages.tolist()) == list(range(1, 17))
        assert torch.count_nonzero(c.pool.req_to_token[1]) == 0
        assert c.events == ["fence", "fence", "fence", "pool-released"]
        c.pool.req_to_token[1].fill_(42)
        c.wrapper(c.req, c.cache)
        assert torch.all(c.pool.req_to_token[1] == 42)
        assert c.owner.driver is c.owner.pool_owner is None


def test_mapping_is_read_before_clear_and_slot_stays_owned_until_final_fence(
    monkeypatch,
):
    with case(monkeypatch) as c:
        count = 0

        def fence():
            nonlocal count
            count += 1
            assert c.req.req_pool_idx == 1 and c.pool.free_slots == [2]
            if count == 2:
                assert c.pool.req_to_token[1, :8].tolist() == list(range(1, 9))
                assert len(c.allocator.free_pages) == 16
            if count == 3:
                assert torch.count_nonzero(c.pool.req_to_token[1]) == 0

        monkeypatch.setattr(c.owner, "_synchronize", fence)
        c.wrapper(c.req, c.cache)
        drain(c)
        assert count == 3 and c.owner.state == "released"


@pytest.mark.parametrize("failure_at", [1, 2, 3])
def test_unknown_fence_retains_slot_owner_and_poisoned_pools_refuse_reuse(
    monkeypatch, failure_at
):
    with case(monkeypatch) as c:
        count = 0

        def fence():
            nonlocal count
            count += 1
            if count == failure_at:
                raise RuntimeError("CUDA completion unknown")

        monkeypatch.setattr(c.owner, "_synchronize", fence)
        c.wrapper(c.req, c.cache)
        drain(c)
        assert c.owner.state == "quarantined" and c.driver.arbiter.busy
        assert c.req.req_pool_idx == 1 and c.pool.free_slots == [2]
        c.guard.request_release()
        assert c.guard.value is not None
        for allocate in (
            lambda: c.pool.alloc([]),
            lambda: c.allocator.alloc(1),
            lambda: c.pool.free(c.req),
            lambda: c.allocator.free(torch.tensor([1])),
        ):
            with pytest.raises(RuntimeError, match="retirement unknown"):
                allocate()
        with pytest.raises(LifecycleError, match="quarantined"):
            c.wrapper(c.req, c.cache)
        assert count == failure_at


def test_failed_controller_close_never_calls_original_cache_release(monkeypatch):
    with case(monkeypatch) as c:

        async def fail():
            raise RuntimeError("remote WRITE still owns destination")

        monkeypatch.setattr(c.control, "aclose", fail)
        c.wrapper(c.req, c.cache)
        drain(c)
        assert c.driver._records["r"].quarantined
        assert c.owner.state == "pending" and c.req.req_pool_idx == 1
        assert len(c.allocator.free_pages) == 8 and not c.events


def test_late_pool_poison_blocks_retirement_after_controller_close(monkeypatch):
    with case(monkeypatch) as c:
        c.pool.pvd_cuda_retirement_error = "initial Prompt import completion unknown"
        c.allocator.pvd_cuda_retirement_error = (
            "initial Prompt import completion unknown"
        )
        c.wrapper(c.req, c.cache)
        drain(c)
        assert c.driver._records["r"].quarantined
        assert c.owner.state == "pending" and c.req.req_pool_idx == 1
        assert len(c.allocator.free_pages) == 8 and not c.events
        c.guard.request_release()
        assert c.guard.value is not None


def test_release_inside_free_group_refuses_without_clearing_views(monkeypatch):
    with case(monkeypatch) as c:
        c.allocator.is_not_in_free_group = False
        c.wrapper(c.req, c.cache)
        drain(c)
        assert c.driver._records["r"].quarantined
        assert c.owner.state == "pending" and c.req.req_pool_idx == 1
        assert not c.allocator.free_group and not c.events


def test_foreign_release_policy_rejected_before_stopping_request(monkeypatch):
    with case(monkeypatch) as c:
        with pytest.raises(LifecycleError, match="exact cache"):
            c.owner.defer(c.req, c.cache, True, lambda: None)
        assert c.owner.state == "attached" and not c.driver._records["r"].stopping


def test_target_permit_defers_retirement_until_whole_result_scope_finishes(monkeypatch):
    with case(monkeypatch) as c:
        lease = c.driver.arbiter.acquire()
        c.wrapper(c.req, c.cache)
        assert c.owner.state == "pending" and c.req.req_pool_idx == 1
        with pytest.raises(LifecycleError, match="between target"):
            c.driver.poll()
        assert not c.events
        c.driver.arbiter.release(lease)
        drain(c)
        assert c.owner.state == "released"


def test_real_pools_req_and_chunk_cache_return_capacity_and_reuse_clean_slot(
    monkeypatch,
):
    with case(monkeypatch, real=True) as c:
        c.wrapper(c.req, c.cache, False)
        drain(c)
        assert c.owner.state == "released", repr(c.owner.error)
        assert c.pool.available_size() == 2, repr(c.owner.error)
        assert c.allocator.available_size() == 16
        assert torch.count_nonzero(c.pool.req_to_token[1]) == 0
        assert sorted(c.allocator.alloc(16).tolist()) == list(range(1, 17))


def test_real_allocator_is_poisoned_when_post_free_fence_is_unknown(monkeypatch):
    with case(monkeypatch, real=True) as c:
        calls = []

        def fence():
            calls.append(True)
            if len(calls) == 2:
                raise RuntimeError("native free completion unknown")

        monkeypatch.setattr(c.owner, "_synchronize", fence)
        c.wrapper(c.req, c.cache)
        drain(c)
        assert c.owner.state == "quarantined" and c.req.req_pool_idx == 1
        assert len(calls) == 2 and "native free completion unknown" in str(
            c.owner.error
        )
        with pytest.raises(RuntimeError, match="retirement unknown"):
            c.allocator.alloc(1)
        with pytest.raises(RuntimeError, match="retirement unknown"):
            c.pool.alloc([])


def test_partial_host_slot_free_poison_prevents_reissuing_the_slot(monkeypatch):
    with case(monkeypatch) as c:
        original = c.pool.free

        def partial(req):
            original(req)
            raise RuntimeError("host slot free partially failed")

        monkeypatch.setattr(c.pool, "free", partial)
        c.wrapper(c.req, c.cache)
        drain(c)
        assert c.owner.state == "quarantined" and c.driver.arbiter.busy
        assert 1 in c.pool.free_slots and c.req.req_pool_idx is None
        with pytest.raises(RuntimeError, match="retirement unknown"):
            c.pool.alloc([])
        assert c.owner.pool_owner is c.guard and c.owner._plan is not None


def test_invalid_free_plan_never_publishes_indices_to_real_allocator(monkeypatch):
    with case(monkeypatch) as c:
        c.pool.req_to_token[1, 1] = c.pool.req_to_token[1, 0]
        c.wrapper(c.req, c.cache)
        drain(c)
        assert c.owner.state == "quarantined" and c.req.req_pool_idx == 1
        assert c.allocator.free_pages.tolist() == list(range(9, 17))
