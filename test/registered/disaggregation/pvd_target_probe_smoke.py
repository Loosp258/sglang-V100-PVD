"""Real CPU probe checks, called by run_pvd_draft_cpu_smoke.py --probe.

Hooks here are an independent TEST oracle only. Production capture is explicit
and batch-local, not hook-based. No production serving path calls this module.
"""

from types import SimpleNamespace

import torch


def validate_target_probe(runner, full_prefix):
    from sglang.srt.disaggregation.pvd.prediction import (
        CommittedPrefix,
        DraftPrediction,
        PredictionConfigError,
        ProbeConfig,
    )
    from sglang.srt.disaggregation.pvd.target_probe import OfflineLlamaTargetProbe
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
        TransferBudget,
        TransferCapacityError,
    )
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
        get_forward_context,
    )

    budget = TransferBudget(2 << 20, 1)
    kwargs = {
        "target_model_id": "cpu-test-target",
        "max_tokens": 16,
        "max_predict_tokens": 3,
        "transient_bytes_bound": 1 << 20,
        "budget": budget,
    }
    probe = OfflineLlamaTargetProbe(
        runner,
        ProbeConfig("cpu-test-target", (0, 1), head_start=1, head_count=2),
        **kwargs,
    )
    assert probe.model is runner.model
    prefix = CommittedPrefix("probe-r", (1, 4, 13, 7), 0, "v1")
    prediction = DraftPrediction("probe-r", "v1", (19, 27))
    target_pool = runner.token_to_kv_pool
    # Defined nonzero data in every target row, including unused rows, makes
    # comparison exact (no uninitialized NaNs). Fixture memory only.
    for tensor in target_pool.k_buffer + target_pool.v_buffer:
        tensor.fill_(0.25)
    original = [t.clone() for t in target_pool.k_buffer + target_pool.v_buffer]
    mapping = runner.req_to_token_pool.req_to_token.clone()
    free = runner.token_to_kv_pool_allocator.free_pages.clone()
    free_slots = list(runner.req_to_token_pool.free_slots)
    weights = [p.clone() for p in runner.model.parameters()]
    rng = torch.random.get_rng_state().clone()
    context = ForwardContext(attn_backend=runner.attn_backend)
    with forward_context(context):
        with probe.branch():
            result = probe.capture(prefix, prediction)
            assert budget.snapshot()["used_staging_bytes"] == probe.reservation_bytes
            values = [query.vectors.clone() for query in result]
        committed = CommittedPrefix(
            "probe-r", prefix.tokens + prediction.tokens, 2, "actual-v2"
        )
        with probe.branch():
            actual_queries = probe.capture_committed(committed, (5,))
            actual_values = [query.vectors.clone() for query in actual_queries]
        assert get_forward_context() is context
    assert budget.snapshot()["used_staging_bytes"] == 0
    for before, after in zip(
        original, target_pool.k_buffer + target_pool.v_buffer, strict=True
    ):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(
        mapping, runner.req_to_token_pool.req_to_token, rtol=0, atol=0
    )
    torch.testing.assert_close(
        free, runner.token_to_kv_pool_allocator.free_pages, rtol=0, atol=0
    )
    assert runner.req_to_token_pool.free_slots == free_slots
    for before, after in zip(weights, runner.model.parameters(), strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())

    # Independent oracle: observe the actual RoPE output of an ordinary target
    # forward (not the PVD collector) and compare all selected Q rows.
    oracle, before_rope, hooks, seen = [], [], [], set()
    try:
        for layer in runner.model.model.layers:
            # get_rope caches modules: multiple layers can share one instance.
            # Register once per instance and use actual invocation order, not
            # one hook per layer that would overwrite every key on every call.
            rope = layer.self_attn.rotary_emb
            if id(rope) in seen:
                continue
            seen.add(id(rope))

            def pre_hook(module, args):
                before_rope.append(args[1].detach().clone())

            def post_hook(module, args, output):
                oracle.append(output[0].detach().clone())

            hooks.append(rope.register_forward_pre_hook(pre_hook))
            hooks.append(rope.register_forward_hook(post_hook))
        full_prefix(prefix.tokens + prediction.tokens)
    finally:
        for hook in hooks:
            hook.remove()
    errors = []
    assert len(oracle) == len(before_rope) == len(runner.model.model.layers)
    for query, value in zip(result, values, strict=True):
        expected = oracle[query.layer].reshape(6, 4, 8)[4:, 1:3]
        raw = before_rope[query.layer].reshape(6, 4, 8)[4:, 1:3]
        torch.testing.assert_close(value, expected, rtol=2e-4, atol=2e-5)
        assert not torch.allclose(value, raw), "oracle cannot distinguish pre/post RoPE"
        assert query.positions == (4, 5) and query.positional_encoding == "rope_applied"
        errors.append(float((value - expected).abs().max()))

    committed_errors = []
    for query, value in zip(actual_queries, actual_values, strict=True):
        expected = oracle[query.layer].reshape(6, 4, 8)[5:6, 1:3]
        torch.testing.assert_close(value, expected, rtol=2e-4, atol=2e-5)
        assert query.positions == (5,) and query.prefix_version == "actual-v2"
        committed_errors.append(float((value - expected).abs().max()))

    # Failed real forward still restores the context and branch reservation.
    def raise_after_capture(module, args, output):
        raise RuntimeError("injected after first real layer")

    failure_hook = runner.model.model.layers[0].register_forward_hook(
        raise_after_capture
    )
    try:
        with forward_context(context):
            try:
                with probe.branch():
                    probe.capture(prefix, prediction)
            except RuntimeError as exc:
                assert "injected after first real layer" in str(exc)
            else:
                raise AssertionError("failure injection was not reached")
            assert get_forward_context() is context
    finally:
        failure_hook.remove()
    assert budget.snapshot()["used_staging_bytes"] == 0
    with probe.branch():
        again = probe.capture(prefix, prediction)
        for old, new in zip(values, again, strict=True):
            torch.testing.assert_close(old, new.vectors, rtol=0, atol=0)

    try:
        probe.capture(prefix, prediction)
    except PredictionConfigError:
        pass
    else:
        raise AssertionError("capture outside branch was admitted")
    small = OfflineLlamaTargetProbe(
        runner,
        probe.config,
        **{**kwargs, "budget": TransferBudget(1, 1)},
    )
    try:
        with small.branch():
            raise AssertionError("insufficient budget was admitted")
    except TransferCapacityError:
        pass
    # Stale identity, token-range and shape refusals must leave no reservation.
    for invalid in (
        DraftPrediction("another-request", "v1", (19,)),
        DraftPrediction("probe-r", "old-version", (19,)),
        DraftPrediction("probe-r", "v1", (runner.model.config.vocab_size,)),
        DraftPrediction("probe-r", "v1", (1, 2, 3, 4)),
    ):
        try:
            with probe.branch():
                probe.capture(prefix, invalid)
        except PredictionConfigError:
            pass
        else:
            raise AssertionError("invalid probe input was admitted")
        assert budget.snapshot()["used_staging_bytes"] == 0

    from unittest.mock import patch

    from sglang.srt.disaggregation.pvd.draft_forward_adapter import PrivatePoolAllocator

    # Real private pools, but an injected release failure: retain budget and
    # owned state and refuse reuse instead of pretending cleanup succeeded.
    failed_budget = TransferBudget(2 << 20, 1)
    failed_probe = OfflineLlamaTargetProbe(
        runner, probe.config, **{**kwargs, "budget": failed_budget}
    )
    with patch.object(
        PrivatePoolAllocator, "free_request", side_effect=RuntimeError("release failed")
    ):
        try:
            with failed_probe.branch():
                failed_probe.capture(prefix, prediction)
        except RuntimeError as exc:
            assert str(exc) == "release failed"
        else:
            raise AssertionError("cleanup failure was not reached")
    assert failed_probe._quarantined and failed_probe._private_state is not None
    assert (
        failed_budget.snapshot()["used_staging_bytes"] == failed_probe.reservation_bytes
    )
    try:
        with failed_probe.branch():
            raise AssertionError("quarantined probe reused")
    except PredictionConfigError:
        pass

    # A real tiny target exercises request-owned retained KV across two
    # captures. The second capture extends only the newly committed token;
    # speculative suffix rows must not survive either branch.
    prefix_budget = TransferBudget(2 << 20, 1)
    cached = OfflineLlamaTargetProbe(
        runner,
        probe.config,
        **{
            **kwargs,
            "budget": TransferBudget(2 << 20, 1),
            "prefix_budget": prefix_budget,
        },
    )
    req = SimpleNamespace(rid="probe-r")
    cached.register_cached_request(req)
    with cached.branch():
        first = cached.capture(prefix, prediction)
    for expected, actual in zip(values, first, strict=True):
        torch.testing.assert_close(actual.vectors, expected, rtol=2e-4, atol=2e-5)
    assert prefix_budget.snapshot()["used_staging_bytes"] == cached.prefix_cache_bytes
    record = cached._prefix_caches[req.rid]
    assert record.tokens == prefix.tokens and len(record.rows) == len(prefix.tokens)
    next_prefix = CommittedPrefix("probe-r", prefix.tokens + (19,), 1, "v2")
    with cached.branch():
        second = cached.capture(next_prefix, DraftPrediction("probe-r", "v2", (27,)))
    for expected, actual in zip(actual_values, second, strict=True):
        torch.testing.assert_close(actual.vectors, expected, rtol=2e-4, atol=2e-5)
    assert record.tokens == next_prefix.tokens
    assert len(record.rows) == len(next_prefix.tokens)
    try:
        cached.retire_cached_request(SimpleNamespace(rid=req.rid))
    except PredictionConfigError:
        pass
    else:
        raise AssertionError("a different Req incarnation retired the cache")
    with cached.branch():
        cached.capture_committed(committed, (5,))
    assert prefix_budget.snapshot()["used_staging_bytes"] == 0
    cached.retire_cached_request(req)
    assert cached._prefix_caches == {}

    # A full cache budget is a cache miss, never a request-level refusal.
    pressure_budget = TransferBudget(1, 1)
    pressure = OfflineLlamaTargetProbe(
        runner,
        probe.config,
        **{
            **kwargs,
            "budget": TransferBudget(2 << 20, 1),
            "prefix_budget": pressure_budget,
        },
    )
    pressure.register_cached_request(req)
    with pressure.branch():
        fallback = pressure.capture(prefix, prediction)
    for expected, actual in zip(values, fallback, strict=True):
        torch.testing.assert_close(actual.vectors, expected, rtol=0, atol=0)
    assert pressure_budget.snapshot()["used_staging_bytes"] == 0
    assert pressure._prefix_caches[req.rid].resources is None
    pressure.retire_cached_request(req)
    return {
        "target_probe": "passed",
        "layers": len(values),
        "max_q_abs_error": max(errors),
        "committed_prefix_q_max_abs_error": max(committed_errors),
        "matches_real_post_rope_not_pre_rope": True,
        "target_weights_pools_mapping_rng_unchanged": True,
        "failure_restores_context_budget_and_reuse": True,
        "cleanup_failure_quarantines_and_keeps_budget": True,
        "incremental_private_prefix_matches_full_cpu_q": True,
        "incremental_prefix_budget_retired": True,
        "serving_or_gpu_validated": False,
    }
