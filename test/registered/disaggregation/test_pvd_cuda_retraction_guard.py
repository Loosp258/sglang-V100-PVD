"""Native retraction must not reset a still-owned CUDA request's KV ledger."""

from types import MethodType
from types import SimpleNamespace as NS

import pytest
from test_pvd_cuda_request_release import case, drain, source


def reset():
    return source(
        "python/sglang/srt/managers/schedule_batch.py", "reset_for_retract", "Req"
    )


@pytest.mark.parametrize("real", [False, True])
def test_reset_is_refused_before_destroying_deferred_release_bookkeeping(
    monkeypatch, real
):
    with case(monkeypatch, real=real) as c:
        c.req.retraction_count, c.req.input_embeds = 0, None
        c.wrapper(c.req, c.cache, False)
        with pytest.raises(RuntimeError, match="CUDA.*retraction"):
            (type(c.req).reset_for_retract if real else reset())(c.req)
        assert c.req.kv_allocated_len == c.req.kv_committed_len == 8
        assert not c.req.is_retracted and c.req.retraction_count == 0
        assert c.req.req_pool_idx == 1
        drain(c)
        assert sorted(c.allocator.free_pages.tolist()) == list(range(1, 17))


def test_release_req_refuses_before_cpu_offload_or_cache_mutation(monkeypatch):
    with case(monkeypatch) as c:
        calls = []
        c.req.retraction_count, c.req.input_embeds = 0, None
        c.req.offload_kv_cache = lambda *args: calls.append("offload")
        c.req.reset_for_retract = MethodType(reset(), c.req)
        release_req = source(
            "python/sglang/srt/managers/schedule_batch.py",
            "release_req",
            "ScheduleBatch",
            namespace={
                "release_kv_cache": lambda *a, **kw: calls.append("release"),
                "evict_from_tree_cache": lambda *a: calls.append("evict"),
                "envs": NS(SGLANG_RETRACT_DECODE_STEPS=NS(get=lambda: 1)),
            },
        )
        batch = NS(
            reqs=[c.req],
            tree_cache=c.cache,
            req_to_token_pool=c.pool,
            token_to_kv_pool_allocator=c.allocator,
            hisparse_coordinator=None,
        )
        with pytest.raises(RuntimeError, match="CUDA.*retraction"):
            release_req(batch, 0, 0, NS(disaggregation_mode="decode"))
        assert calls == []
        assert c.owner.state == "attached" and c.req.req_pool_idx == 1


def test_unbound_native_reset_still_works():
    req = NS(
        retraction_count=0, input_embeds=None, kv_allocated_len=8, kv_committed_len=8
    )
    reset()(req)
    assert req.retraction_count == 1 and req.is_retracted
    assert req.kv_allocated_len == req.kv_committed_len == 0


def test_released_tombstone_cannot_be_reused_as_a_new_incarnation(monkeypatch):
    with case(monkeypatch) as c:
        c.wrapper(c.req, c.cache, False)
        drain(c)
        assert c.owner.state == "released"
        c.req.retraction_count, c.req.input_embeds = 0, None
        with pytest.raises(RuntimeError, match="incarnation"):
            reset()(c.req)
        assert c.req.retraction_count == 0
