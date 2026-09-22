"""Actual CPU math and explicit CPU ownership policies; CUDA tests skip separately."""

import math
from dataclasses import replace

import pytest
import torch
from sglang.srt.disaggregation.pvd import cuda_sparse_attention as attention
from sglang.srt.disaggregation.pvd.cuda_rank_install import CUDARankInstallParticipant
from sglang.srt.disaggregation.pvd.cuda_working_set import CUDASparseWorkingSet
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.rank_install_wire import RankInstallExchange
from sglang.srt.disaggregation.pvd.sparse_install import RankInstallCoordinator
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from sglang.srt.disaggregation.pvd.sparse_working_set import reference_attention
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferBudget,
    TransferCapacityError,
)
from test_pvd_cuda_rank_install import complete, setup
from test_pvd_cuda_working_set import options, packed_payloads


def buffers(decode_tokens=0, q_heads=4, *, dtype=torch.float32, device="cpu"):
    generator = torch.Generator().manual_seed(731)
    q = torch.randn(q_heads, 3, generator=generator).to(device=device, dtype=dtype)
    k = torch.randn(decode_tokens + 1, 2, 3, generator=generator).to(
        device=device, dtype=dtype
    )
    v = torch.randn(decode_tokens + 1, 2, 3, generator=generator).to(
        device=device, dtype=dtype
    )
    return attention.AttentionBuffers(q, k, v, torch.empty_like(q))


def groups(tokens=(0, 1, 2, 3), *, dtype=torch.float32):
    rows, _, _ = packed_payloads(0, tokens, dtype=dtype)
    return {(p.spec.layer, p.spec.kv_head): (p.spec, p.tensor) for p in rows}


def oracle(prompt, data, mapping, decode_tokens):
    return reference_attention(
        {key: (spec, tensor.float()) for key, (spec, tensor) in prompt.items()},
        data.q.float(),
        layer=0,
        query_position=4 + decode_tokens,
        mapping=mapping,
        prompt_tokens=4,
        generated_positions=tuple(range(4, 5 + decode_tokens)),
        generated_k=data.generated_k.float(),
        generated_v=data.generated_v.float(),
    )


@pytest.mark.parametrize("chunk", [1, 2, 7])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("tokens", [(0, 1, 2, 3), (3, 1)])
def test_tiled_math_matches_cpu_oracle_without_full_kv_concat(
    monkeypatch, chunk, dtype, tokens
):
    prompt, data, mapping = (
        groups(tokens, dtype=dtype),
        buffers(10, dtype=dtype),
        QueryHeadMapping(4, 2),
    )
    expected = oracle(prompt, data, mapping, 10)
    scratch = torch.empty(attention.scratch_elements(chunk, 3))
    q_before, k_before, v_before = (
        t.clone() for t in (data.q, data.generated_k, data.generated_v)
    )

    def refuse(*args, **kwargs):
        raise AssertionError("no full-context concatenation")

    monkeypatch.setattr(torch, "cat", refuse)
    attention._stream_attention(
        prompt, data, mapping, 0, 1 / math.sqrt(3), scratch, chunk, 3
    )
    torch.testing.assert_close(
        data.output.float(),
        expected,
        atol=0.025 if dtype == torch.float16 else 2e-5,
        rtol=1e-3 if dtype == torch.float16 else 2e-5,
    )
    for actual, before in zip(
        (data.q, data.generated_k, data.generated_v),
        (q_before, k_before, v_before),
        strict=True,
    ):
        torch.testing.assert_close(actual, before, rtol=0, atol=0)


def test_online_softmax_rescales_across_tiles_and_is_stable():
    prompt, data, mapping = groups(), buffers(3), QueryHeadMapping(4, 2)
    data.q.fill_(1000)
    data.generated_k.fill_(1000)
    expected = oracle(prompt, data, mapping, 3)
    scratch = torch.empty(attention.scratch_elements(1, 3))
    attention._stream_attention(
        prompt, data, mapping, 0, 1 / math.sqrt(3), scratch, 1, 3
    )
    assert torch.isfinite(data.output).all()
    torch.testing.assert_close(data.output, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("q_heads,kv_heads", [(2, 2), (4, 1)])
def test_mha_mqa_and_explicit_attention_scale(q_heads, kv_heads):
    prompt, original = groups(), buffers(5, q_heads=q_heads)
    data = attention.AttentionBuffers(
        original.q,
        original.generated_k[:, :kv_heads].contiguous(),
        original.generated_v[:, :kv_heads].contiguous(),
        original.output,
    )
    mapping = QueryHeadMapping(q_heads, kv_heads)
    scaled = attention.AttentionBuffers(
        data.q * (0.2 * math.sqrt(3)), data.generated_k, data.generated_v, data.output
    )
    expected = oracle(prompt, scaled, mapping, 5)
    scratch = torch.empty(attention.scratch_elements(2, 3))
    attention._stream_attention(prompt, data, mapping, 0, 0.2, scratch, 2, 3)
    torch.testing.assert_close(data.output, expected, atol=2e-5, rtol=2e-5)


def policy_workspace(monkeypatch, *, capacity=4096):
    # Only allocation placement and synchronization are replaced, not the math.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    original = torch.empty
    with monkeypatch.context() as local:
        local.setattr(
            torch, "empty", lambda *a, **kw: original(*a, **{**kw, "device": "cpu"})
        )
        budget = TransferBudget(capacity, 2)
        workspace = attention.CUDASparseAttentionWorkspace(
            device="cuda:0",
            dtype=torch.float32,
            head_dim=3,
            chunk_tokens=2,
            budget=budget,
        )
    workspace.device = torch.device("cpu")
    syncs = []
    monkeypatch.setattr(workspace, "_synchronize", lambda: syncs.append(True))
    return workspace, budget, syncs


def ready(monkeypatch):
    peers, exchange, _, _ = setup(monkeypatch)
    complete(peers, exchange, 0)
    workspace, budget, syncs = policy_workspace(monkeypatch)
    data, released = buffers(), []
    resources = ResourceGuard(data, lambda: released.append(True))
    return peers, workspace, budget, syncs, resources, released


def execute(workspace, peer, resources, **overrides):
    options = {
        "decode_tokens": 0,
        "layer": 0,
        "mapping": QueryHeadMapping(4, 2),
        "resources": resources,
        "scale": 1 / math.sqrt(3),
    }
    workspace.execute(peer, **{**options, **overrides})


def test_workspace_is_charged_before_execution_and_input_guard_survives_fence(
    monkeypatch,
):
    peers, workspace, budget, syncs, resources, released = ready(monkeypatch)
    expected = oracle(groups(), resources.value, QueryHeadMapping(4, 2), 0)
    data = resources.value

    def synchronize():
        resources.request_release()
        assert not released
        assert (
            budget.snapshot()["used_staging_bytes"]
            == attention.scratch_elements(2, 3) * 4
        )
        with pytest.raises(SparsePayloadError, match="still owns"):
            workspace.close()
        syncs.append(True)

    monkeypatch.setattr(workspace, "_synchronize", synchronize)
    execute(workspace, peers[0], resources)
    assert syncs == [True] and released == [True]
    torch.testing.assert_close(data.output, expected, atol=2e-5, rtol=2e-5)
    workspace.close()
    assert budget.snapshot()["used_staging_bytes"] == 0
    for peer in peers.values():
        peer.close()


@pytest.mark.parametrize("drain_fails", [False, True])
def test_partial_compute_drains_or_quarantines_guard_and_scratch(
    monkeypatch, drain_fails
):
    peers, workspace, budget, syncs, resources, released = ready(monkeypatch)

    def fail(*args):
        resources.request_release()
        raise RuntimeError("partial compute")

    def synchronize():
        assert not released
        syncs.append(True)
        if drain_fails:
            raise RuntimeError("completion unknown")

    monkeypatch.setattr(attention, "_stream_attention", fail)
    monkeypatch.setattr(workspace, "_synchronize", synchronize)
    with pytest.raises(
        RuntimeError, match="completion unknown" if drain_fails else "partial compute"
    ):
        execute(workspace, peers[0], resources)
    assert syncs == [True]
    if drain_fails:
        assert not released and workspace._held is not None
        with pytest.raises(SparsePayloadError, match="quarantined"):
            workspace.close()
        assert budget.snapshot()["used_staging_bytes"] > 0
    else:
        assert released == [True]
        workspace.close()
        assert budget.snapshot()["used_staging_bytes"] == 0
    for peer in peers.values():
        peer.close()


@pytest.mark.parametrize(
    "bad", ["dtype", "shape", "alias", "layer", "scale", "position"]
)
def test_invalid_execution_never_changes_output_or_enters_compute(monkeypatch, bad):
    peers, workspace, _, syncs, resources, _ = ready(monkeypatch)
    data = resources.value
    overrides = {}
    if bad in ("dtype", "shape", "alias"):
        data = attention.AttentionBuffers(
            data.q.half()
            if bad == "dtype"
            else data.q[:1]
            if bad == "shape"
            else data.q,
            data.generated_k,
            data.generated_v,
            data.q if bad == "alias" else data.output,
        )
        resources = ResourceGuard(data, lambda: None)
    else:
        overrides[
            {"layer": "layer", "scale": "scale", "position": "decode_tokens"}[bad]
        ] = {"layer": 8, "scale": float("nan"), "position": 1}[bad]
    data.output.fill_(-123)
    with pytest.raises(SparsePayloadError):
        execute(workspace, peers[0], resources, **overrides)
    assert not syncs and torch.all(data.output == -123)
    workspace.close()
    resources.request_release()
    for peer in peers.values():
        peer.close()


def test_capacity_is_refused_before_scratch_allocation(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("allocation before reservation")

    monkeypatch.setattr(torch, "empty", refuse)
    with pytest.raises(TransferCapacityError):
        policy_workspace(monkeypatch, capacity=1)


def test_resource_release_error_quarantines_workspace(monkeypatch):
    peers, workspace, budget, _, resources, _ = ready(monkeypatch)

    def fail():
        raise RuntimeError("release failed")

    resources = ResourceGuard(resources.value, fail)
    monkeypatch.setattr(workspace, "_synchronize", resources.request_release)
    with pytest.raises(RuntimeError, match="release failed"):
        execute(workspace, peers[0], resources)
    with pytest.raises(SparsePayloadError, match="quarantined"):
        workspace.close()
    assert workspace._held is not None
    assert budget.snapshot()["used_staging_bytes"] > 0
    for peer in peers.values():
        peer.close()


def test_release_callback_cannot_reenter_workspace(monkeypatch):
    peers, workspace, _, _, resources, _ = ready(monkeypatch)
    other = ResourceGuard(buffers(), lambda: None)

    def callback():
        assert workspace.snapshot()["active"]
        with pytest.raises(SparsePayloadError, match="concurrently"):
            execute(workspace, peers[0], other)
        with pytest.raises(SparsePayloadError, match="still owns"):
            workspace.close()

    resources = ResourceGuard(resources.value, callback)
    monkeypatch.setattr(workspace, "_synchronize", resources.request_release)
    execute(workspace, peers[0], resources)
    assert not workspace.snapshot()["active"]
    workspace.close()
    other.request_release()
    for peer in peers.values():
        peer.close()


def test_reader_exit_failure_keeps_execution_resources_and_budget(monkeypatch):
    peers, workspace, budget, _, resources, released = ready(monkeypatch)

    def fail_reader():
        peers[0]._bank._quarantine = "reader completion unknown"
        raise RuntimeError("reader completion unknown")

    monkeypatch.setattr(workspace, "_synchronize", resources.request_release)
    monkeypatch.setattr(peers[0]._bank, "_drain_reader", fail_reader)
    with pytest.raises(RuntimeError, match="reader completion unknown"):
        execute(workspace, peers[0], resources)
    assert not released
    assert workspace.snapshot()["resources_held"]
    assert budget.snapshot()["used_staging_bytes"] > 0
    with pytest.raises(SparsePayloadError, match="quarantined"):
        workspace.close()


def mapped_buffers(data):
    """Put compact generated KV in discontiguous rows; poison all others."""
    count = data.generated_k.shape[0]
    rows = tuple(range(2 * count, 0, -2))
    shape = (2 * count + 3, *data.generated_k.shape[1:])
    keys = torch.full(shape, float("nan"), dtype=data.q.dtype, device=data.q.device)
    values = torch.full_like(keys, float("nan"))
    for index, row in enumerate(rows):
        keys[row].copy_(data.generated_k[index])
        values[row].copy_(data.generated_v[index])
    return replace(data, generated_k=keys, generated_v=values, generated_rows=rows)


@pytest.mark.parametrize("chunk", [1, 2, 7])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("tokens", [(0, 1, 2, 3), (3, 1)])
def test_mapped_pool_math_never_reads_unselected_rows(
    monkeypatch, chunk, dtype, tokens
):
    prompt, data, mapping = (
        groups(tokens, dtype=dtype),
        buffers(10, dtype=dtype),
        QueryHeadMapping(4, 2),
    )
    expected = oracle(prompt, data, mapping, 10)
    mapped = mapped_buffers(data)
    scratch = torch.empty(attention.scratch_elements(chunk, 3))

    def refuse(*args, **kwargs):
        raise AssertionError("no context-sized concatenation or gather")

    monkeypatch.setattr(torch, "cat", refuse)
    monkeypatch.setattr(torch, "index_select", refuse)
    attention._stream_attention(
        prompt, mapped, mapping, 0, 1 / math.sqrt(3), scratch, chunk, 3
    )
    torch.testing.assert_close(
        mapped.output.float(),
        expected,
        atol=0.025 if dtype == torch.float16 else 2e-5,
        rtol=1e-3 if dtype == torch.float16 else 2e-5,
    )


@pytest.mark.parametrize("bad_rows", [(), (0,), (-1,), (999,), (True,), (1.0,), [1]])
def test_invalid_pool_rows_refused_before_compute(monkeypatch, bad_rows):
    peers, workspace, _, syncs, resources, _ = ready(monkeypatch)
    data = replace(mapped_buffers(resources.value), generated_rows=bad_rows)
    data.output.fill_(-123)
    resources = ResourceGuard(data, lambda: None)
    with pytest.raises(SparsePayloadError, match="pool row mapping"):
        execute(workspace, peers[0], resources)
    assert not syncs and torch.all(data.output == -123)
    workspace.close()
    resources.request_release()
    for peer in peers.values():
        peer.close()


def test_mapped_pool_workspace_matches_compact_result(monkeypatch):
    peers, workspace, _, _, resources, _ = ready(monkeypatch)
    expected = oracle(groups(), resources.value, QueryHeadMapping(4, 2), 0)
    data = mapped_buffers(resources.value)
    resources = ResourceGuard(data, lambda: None)
    execute(workspace, peers[0], resources)
    torch.testing.assert_close(data.output, expected, atol=2e-5, rtol=2e-5)
    workspace.close()
    resources.request_release()
    for peer in peers.values():
        peer.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="real CUDA attention is unverified without CUDA",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_real_cuda_tiled_math_matches_cpu_oracle(dtype):
    prompt, data, mapping = (
        groups(dtype=dtype),
        buffers(10, dtype=dtype),
        QueryHeadMapping(4, 2),
    )
    expected = oracle(prompt, data, mapping, 10)
    cuda_groups = {
        key: (spec, tensor.to("cuda:0")) for key, (spec, tensor) in prompt.items()
    }
    cuda_data = attention.AttentionBuffers(
        *(
            t.to("cuda:0")
            for t in (data.q, data.generated_k, data.generated_v, data.output)
        )
    )
    scratch = torch.empty(attention.scratch_elements(3, 3), device="cuda:0")
    attention._stream_attention(
        cuda_groups, cuda_data, mapping, 0, 1 / math.sqrt(3), scratch, 3, 3
    )
    torch.cuda.synchronize("cuda:0")
    torch.testing.assert_close(
        cuda_data.output.float().cpu(),
        expected,
        atol=0.025 if dtype == torch.float16 else 2e-4,
        rtol=1e-3 if dtype == torch.float16 else 2e-4,
    )
    # Same real CUDA case also exercises a model-pool-shaped mapped source.
    mapped = mapped_buffers(cuda_data)
    attention._stream_attention(
        cuda_groups, mapped, mapping, 0, 1 / math.sqrt(3), scratch, 3, 3
    )
    torch.cuda.synchronize("cuda:0")
    torch.testing.assert_close(
        mapped.output.float().cpu(),
        expected,
        atol=0.025 if dtype == torch.float16 else 2e-4,
        rtol=1e-3 if dtype == torch.float16 else 2e-4,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="real CUDA attention/bank integration requires CUDA",
)
def test_real_cuda_attention_consumes_installed_banks_across_refresh():
    bank_budget, workspace_budget = TransferBudget(4096, 3), TransferBudget(4096, 2)
    bank = CUDASparseWorkingSet(
        device="cuda:0", dtype=torch.float32, budget=bank_budget, **options()
    )
    peer = CUDARankInstallParticipant(bank, rank=0, peer_epoch="worker", interval=4)
    exchange = RankInstallExchange(
        RankInstallCoordinator(
            "r", "inc", "entry", rank_layouts={0: "layout"}, interval=4, lead_tokens=1
        ),
        peer_epochs={0: "worker"},
    )
    workspace = attention.CUDASparseAttentionWorkspace(
        device="cuda:0",
        dtype=torch.float32,
        head_dim=3,
        chunk_tokens=2,
        budget=workspace_budget,
    )
    mapping = QueryHeadMapping(4, 2)
    for count in (0, 3):
        epoch = exchange.begin(count)
        tokens = (0, 1, 2, 3) if count == 0 else (3, 1)
        rows, guard, _ = packed_payloads(epoch.target_tokens, tokens, device="cuda:0")
        for row in rows:
            row.spec = replace(row.spec, operation_id=epoch.operation_id)
        exchange.receive(peer.stage(epoch, rows, source_guard=guard), peer_rank=0)
        guard.request_release()
        exchange.receive(peer.park(epoch.target_tokens), peer_rank=0)
        exchange.receive(peer.command(exchange.install_commands(epoch)[0]), peer_rank=0)
        exchange.receive(peer.command(exchange.resume_commands(epoch)[0]), peer_rank=0)
        assert exchange.can_decode(epoch.target_tokens)
        data = buffers(epoch.target_tokens, device="cuda:0")
        resources = ResourceGuard(data, lambda: None)
        expected = oracle(
            groups(tokens),
            attention.AttentionBuffers(
                *(
                    t.cpu()
                    for t in (data.q, data.generated_k, data.generated_v, data.output)
                )
            ),
            mapping,
            epoch.target_tokens,
        )
        workspace.execute(
            peer,
            decode_tokens=epoch.target_tokens,
            layer=0,
            mapping=mapping,
            resources=resources,
            scale=1 / math.sqrt(3),
        )
        torch.testing.assert_close(data.output.cpu(), expected, atol=2e-4, rtol=2e-4)
        resources.request_release()
    workspace.close()
    peer.close()
    assert workspace_budget.snapshot()["used_staging_bytes"] == 0
    assert bank_budget.snapshot()["used_staging_bytes"] == 0
