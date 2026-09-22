"""Actual CPU tensor math with CUDA placement/fences substituted."""

from dataclasses import replace

import pytest
import torch
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import TargetExecutionArbiter
from sglang.srt.disaggregation.pvd.cuda_rank_batch import (
    CUDARankBatchExecutor,
    CUDARuntimeBatchMember,
)
from sglang.srt.disaggregation.pvd.cuda_runtime_group import CUDARuntimeInstallGroup
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError
from test_pvd_cuda_model_attention import fixture as model_fixture
from test_pvd_cuda_model_attention import run
from test_pvd_cuda_runtime_group import initialize, make_group
from test_pvd_cuda_working_set import packed_payloads


def setup(monkeypatch):
    first, _ = make_group(monkeypatch)
    initialize(first)
    second, _ = make_group(monkeypatch)
    # New request identity before constructing its runtime/participant.
    bank = second._banks[0]
    bank.identity = ("other", *bank.identity[1:])
    second = CUDARuntimeInstallGroup(
        {0: bank},
        interval=4,
        lead_tokens=1,
        peer_epochs={0: "worker"},
        timeout_seconds=10,
        max_pending_events=8,
        max_pending_bytes=65536,
    )
    epoch = second.begin(0)
    payloads, source, _ = packed_payloads()
    for payload in payloads:
        payload.spec = replace(
            payload.spec, request_id="other", operation_id=epoch.operation_id
        )
    second.stage(epoch, 0, payloads, source_guard=source)
    source.request_release()
    assert second.try_install(epoch, {0: 0})
    c = model_fixture(monkeypatch)
    c.req.req_to_token[2, 4:6] = torch.tensor([12, 13])
    c.batch.req_pool_indices = torch.tensor([1, 2])
    c.batch.positions = torch.tensor([5, 5])
    c.batch.seq_lens = torch.tensor([6, 6])
    c.batch.out_cache_loc = torch.tensor([11, 13])
    c.q, c.k, c.v = (t.repeat(2, *([1] * (t.ndim - 1))) for t in (c.q, c.k, c.v))
    arbiter = TargetExecutionArbiter()
    executor = CUDARankBatchExecutor(c.consumer, arbiter, max_requests=2)
    members = (
        CUDARuntimeBatchMember(first, 1, 1),
        CUDARuntimeBatchMember(second, 2, 1),
    )
    return c, executor, members, arbiter


def test_two_request_forward_and_result_keep_all_permits_and_target_lease(monkeypatch):
    c, executor, members, arbiter = setup(monkeypatch)

    def commit(result):
        assert result.shape == (2, 12)
        assert arbiter.busy
        assert all(m.group.runtime._forward is not None for m in members)
        c.owner.request_release()
        assert not c.released
        return "committed"

    assert (
        executor.run(
            members, pool_owner=c.owner, forward=lambda: run(c), process_results=commit
        )
        == "committed"
    )
    assert not arbiter.busy
    assert c.released == [True]
    assert all(m.group.runtime._forward is None for m in members)
    for m in members:
        m.group.close()


def test_one_waiting_member_prevents_entire_batch_without_acquiring_permits(
    monkeypatch,
):
    c, executor, members, arbiter = setup(monkeypatch)
    members = (members[0], replace(members[1], committed_tokens=4))
    with pytest.raises(InstallProtocolError, match="wait or abort"):
        executor.run(
            members,
            pool_owner=c.owner,
            forward=lambda: pytest.fail("dispatched"),
            process_results=lambda r: pytest.fail("committed"),
        )
    assert not arbiter.busy and not executor._quarantined
    assert all(m.group.runtime._forward is None for m in members)
    for m in members:
        m.group.close()


def test_late_peer_loss_discards_all_outputs_and_retires_every_permit(monkeypatch):
    c, executor, members, arbiter = setup(monkeypatch)

    def forward():
        output = run(c)
        assert members[1].group.runtime.peer_lost(peer_rank=0, peer_epoch="worker")
        return output

    with pytest.raises(InstallProtocolError, match="commit nothing"):
        executor.run(
            members,
            pool_owner=c.owner,
            forward=forward,
            process_results=lambda r: pytest.fail("committed"),
        )
    assert not arbiter.busy and not executor._quarantined
    assert all(m.group.runtime._forward is None for m in members)
    for m in members:
        m.group.close()


def test_unknown_completion_retains_all_runtime_and_pool_owners(monkeypatch):
    c, executor, members, arbiter = setup(monkeypatch)

    def fail():
        raise RuntimeError("GPU completion unknown")

    monkeypatch.setattr(c.consumer, "_synchronize", fail)
    with pytest.raises(RuntimeError, match="completion unknown"):
        executor.run(
            members,
            pool_owner=c.owner,
            forward=lambda: run(c),
            process_results=lambda r: pytest.fail("committed"),
        )
    c.owner.request_release()
    assert executor._quarantined and arbiter.busy and not c.released
    assert all(m.group.runtime._forward is not None for m in members)
    for m in members:
        with pytest.raises(InstallProtocolError, match="must drain"):
            m.group.close()
    # CPU fixture teardown before monkeypatch restores CUDA calls, NOT recovery.
    c.consumer._held[0].close()


def test_result_processor_failure_aborts_members_without_replaying(monkeypatch):
    c, executor, members, arbiter = setup(monkeypatch)

    def failed(result):
        raise RuntimeError("existing result processor failed")

    with pytest.raises(RuntimeError, match="result processor"):
        executor.run(
            members, pool_owner=c.owner, forward=lambda: run(c), process_results=failed
        )
    assert not arbiter.busy
    assert all(not m.group.can_decode(1) for m in members)
    for m in members:
        m.group.close()


@pytest.mark.parametrize("fault", ["slot", "request", "count", "async"])
def test_invalid_members_or_async_callbacks_never_dispatch(monkeypatch, fault):
    c, executor, members, arbiter = setup(monkeypatch)
    original = members
    forward = lambda: pytest.fail("forward called")
    if fault == "slot":
        members = (members[0], replace(members[1], slot=1))
    elif fault == "request":
        members = (members[0], replace(members[0], slot=2))
    elif fault == "count":
        members = (replace(members[0], committed_tokens=True), members[1])
    else:

        async def forward():
            pytest.fail("async forward called")

    with pytest.raises(InstallProtocolError):
        executor.run(
            members, pool_owner=c.owner, forward=forward, process_results=lambda r: None
        )
    assert not arbiter.busy and not executor._quarantined
    assert all(m.group.runtime._forward is None for m in original)
    for m in original:
        m.group.close()


def test_failed_forward_drains_before_failing_all_members(monkeypatch):
    c, executor, members, arbiter = setup(monkeypatch)

    def fail():
        run(c)
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        executor.run(
            members,
            pool_owner=c.owner,
            forward=fail,
            process_results=lambda r: pytest.fail("committed"),
        )
    assert c.drains and not arbiter.busy and not executor._quarantined
    assert all(m.group.runtime._forward is None for m in members)
    for m in members:
        m.group.close()
