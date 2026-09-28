"""Private prefix cache identity and worker ownership tests."""

import threading
from types import SimpleNamespace

import pytest
from sglang.srt.disaggregation.pvd.concurrent_prefix_cache import (
    ConcurrentPrefixCacheOwner,
)
from sglang.srt.disaggregation.pvd.concurrent_prediction_worker import PredictionJob


def job(request_id="a", incarnation="first", length=32):
    return PredictionJob(
        request_id=request_id,
        incarnation=incarnation,
        prefix_version=f"{incarnation}:{length}",
        committed_position=length - 1,
        prefix_tokens=(1,) * length,
    )


def test_cache_reuses_same_incarnation_and_retires_before_request_id_reuse():
    calls = []
    probe = SimpleNamespace(
        prefix_budget=object(),
        register_cached_request=lambda req: calls.append(("register", req)),
        retire_cached_request=lambda req: calls.append(("retire-probe", req)),
    )
    provider = SimpleNamespace(
        factory=SimpleNamespace(prefix_cache_enabled=True),
        set_sidecar_cache_identity=lambda *identity: calls.append(
            ("register-draft", identity)
        ),
        retire_sidecar_cache=lambda **kw: calls.append(("retire-draft", kw)),
    )
    owner = ConcurrentPrefixCacheOwner(probe, provider, max_cached_requests=2)
    owner.prepare(job())
    owner.prepare(job(length=64))
    assert [kind for kind, *_ in calls] == ["register", "register-draft"]
    first_marker = calls[0][1]
    assert first_marker.rid == "a"
    assert not hasattr(first_marker, "req_pool_idx")

    owner.prepare(job(request_id="b"))
    assert [kind for kind, *_ in calls] == [
        "register",
        "register-draft",
        "register",
        "retire-draft",
        "register-draft",
    ]
    assert calls[2][1].rid == "b"
    assert calls[3][1] == {"if_identity": ("a", "first")}
    owner.prepare(job(incarnation="second"))
    assert calls[5][0] == "retire-probe"
    assert calls[5][1] is first_marker
    assert calls[6][0] == "register"
    assert calls[6][1] is not first_marker
    assert calls[7][0] == "retire-draft"
    assert calls[8][0] == "register-draft"


def test_target_cache_evicts_oldest_marker_at_request_capacity():
    calls = []
    probe = SimpleNamespace(
        prefix_budget=object(),
        register_cached_request=lambda req: calls.append(("register", req)),
        retire_cached_request=lambda req: calls.append(("retire", req)),
    )
    provider = SimpleNamespace(factory=SimpleNamespace(prefix_cache_enabled=False))
    owner = ConcurrentPrefixCacheOwner(probe, provider, max_cached_requests=2)
    owner.prepare(job("a"))
    owner.prepare(job("b"))
    owner.prepare(job("c"))
    assert calls[2] == ("retire", calls[0][1])
    assert calls[3][0] == "register"


def test_cache_refuses_cross_thread_retirement_and_quarantines_failed_release():
    release_error = RuntimeError("CUDA fence failed")
    probe = SimpleNamespace(
        prefix_budget=object(),
        register_cached_request=lambda req: None,
        retire_cached_request=lambda req: (_ for _ in ()).throw(release_error),
    )
    provider = SimpleNamespace(factory=SimpleNamespace(prefix_cache_enabled=False))
    owner = ConcurrentPrefixCacheOwner(probe, provider)
    owner.prepare(job())

    errors = []

    def other_thread():
        try:
            owner.retire()
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join()
    assert errors == ["private prefix caches belong to one worker thread"]
    with pytest.raises(RuntimeError, match="CUDA fence failed"):
        owner.retire()
    with pytest.raises(RuntimeError, match="quarantined"):
        owner.prepare(job(request_id="b"))
