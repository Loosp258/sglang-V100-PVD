"""CPU policy composition of the CUDA TP1 runtime; not GPU/TP model evidence."""

import time
from dataclasses import replace

import pytest
from sglang.srt.disaggregation.pvd.cuda_runtime_group import CUDARuntimeInstallGroup
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from test_pvd_cuda_model_attention import fixture as model_fixture
from test_pvd_cuda_model_attention import run
from test_pvd_cuda_working_set import packed_payloads, policy_bank


def make_group(monkeypatch, *, clock=time.monotonic):
    bank, budget, _ = policy_bank(monkeypatch)
    return CUDARuntimeInstallGroup(
        {0: bank},
        interval=4,
        lead_tokens=1,
        peer_epochs={0: "worker"},
        timeout_seconds=10,
        max_pending_events=8,
        max_pending_bytes=65536,
        clock=clock,
    ), budget


def stage(group, count):
    epoch = group.begin(count)
    payloads, owner, _ = packed_payloads(
        epoch.target_tokens, (0, 1, 2, 3) if count == 0 else (1, 3)
    )
    for payload in payloads:
        payload.spec = replace(payload.spec, operation_id=epoch.operation_id)
    receipt = group.stage(epoch, 0, payloads, source_guard=owner)
    owner.request_release()
    return epoch, receipt


def initialize(group):
    epoch, receipt = stage(group, 0)
    assert not group.can_decode(0)
    assert group.try_install(epoch, {0: 0})
    assert group.installation_complete(receipt)
    return epoch, receipt


def test_two_rounds_require_runtime_permit_and_exact_resume(monkeypatch):
    group, budget = make_group(monkeypatch)
    _, old = initialize(group)
    epoch, receipt = stage(group, 3)
    assert group.can_decode(3)
    assert not group.installation_complete(old)
    with pytest.raises(InstallProtocolError, match="forward permit"), group.read(0, 3):
        pass
    permit = group.runtime.begin_forward(3)
    with group.read(0, 3) as groups:
        assert groups[(0, 0)][0].token_ids == (0, 1, 2, 3)
        assert not group.try_install(epoch, {0: 4})
    assert group.runtime.finish_forward(permit, readers_drained=True, succeeded=True)
    assert not group.can_decode(4)
    assert group.try_install(epoch, {0: 4})
    assert group.installation_complete(receipt) and group.can_decode(4)
    group.close()
    assert budget.snapshot()["used_staging_bytes"] == 0


@pytest.mark.parametrize(
    "bad", [{}, {0: object()}, {0: object(), 1: object()}, {True: object()}]
)
def test_no_cpu_or_multi_rank_fallback(bad):
    with pytest.raises(InstallProtocolError, match="TP1"):
        CUDARuntimeInstallGroup(
            bad,
            interval=4,
            lead_tokens=1,
            peer_epochs={0: "p"},
            timeout_seconds=1,
            max_pending_events=8,
            max_pending_bytes=65536,
        )


def test_model_forward_owns_permit_until_model_and_readers_complete(monkeypatch):
    group, _ = make_group(monkeypatch)
    initialize(group)
    c = model_fixture(monkeypatch)
    with group.model_forward(c.consumer, slot=1, decode_tokens=1, pool_owner=c.owner):
        assert group.runtime._forward is not None
        output = run(c)
        assert output.shape == (1, 12)
        with pytest.raises(SparsePayloadError, match="owned by a forward"):
            group._banks[0].close()
    assert group.runtime._forward is None
    assert group.can_decode(1)
    group.close()


def test_cancelled_model_result_is_refused_after_safe_drain(monkeypatch):
    group, _ = make_group(monkeypatch)
    initialize(group)
    c = model_fixture(monkeypatch)
    with (
        pytest.raises(SparsePayloadError, match="cancelled"),
        group.model_forward(c.consumer, slot=1, decode_tokens=1, pool_owner=c.owner),
    ):
        run(c)
        group.runtime.cancel()
    assert group.runtime._forward is None
    assert not group.can_decode(1)
    group.close()


def test_unknown_cuda_completion_retains_runtime_permit_and_pool_owner(monkeypatch):
    group, budget = make_group(monkeypatch)
    initialize(group)
    c = model_fixture(monkeypatch)

    def fail():
        raise RuntimeError("unknown completion")

    monkeypatch.setattr(c.consumer, "_synchronize", fail)
    try:
        with (
            pytest.raises(RuntimeError),
            group.model_forward(
                c.consumer, slot=1, decode_tokens=1, pool_owner=c.owner
            ),
        ):
            run(c)
            c.owner.request_release()
        assert group.runtime._forward is not None and not c.released
        assert budget.snapshot()["used_staging_bytes"] > 0
        with pytest.raises(InstallProtocolError, match="must drain"):
            group.close()
    finally:
        # CPU fixture teardown, not a production recovery operation.
        for peer in group._peers.values():
            monkeypatch.setattr(peer._bank, "_drain_reader", lambda: None)
        if c.consumer._held is not None:
            c.consumer._held[0].close()


def test_timeout_stops_local_reads_without_releasing_bank(monkeypatch):
    now = [0.0]
    group, budget = make_group(monkeypatch, clock=lambda: now[0])
    initialize(group)
    stage(group, 3)
    now[0] = 11.0
    group.progress()
    assert not group.can_decode(3)
    assert budget.snapshot()["used_staging_bytes"] > 0
    with pytest.raises(InstallProtocolError):
        group.runtime.begin_forward(3)
    group.close()


def test_latched_peer_loss_discards_completed_model_output(monkeypatch):
    group, _ = make_group(monkeypatch)
    initialize(group)
    c = model_fixture(monkeypatch)
    committed = []
    with pytest.raises(InstallProtocolError, match="refused completed model result"):
        with group.model_forward(
            c.consumer, slot=1, decode_tokens=1, pool_owner=c.owner
        ):
            output = run(c)
            assert group.runtime.peer_lost(peer_rank=0, peer_epoch="worker")
        committed.append(output)
    assert not committed
    assert group.runtime._forward is None
    assert not group.can_decode(1)
    group.close()


def test_received_path_refuses_a_cpu_or_unbound_record(monkeypatch):
    group, _ = make_group(monkeypatch)
    epoch = group.begin(0)
    with pytest.raises(InstallProtocolError, match="CUDA receive record"):
        group.stage_received(object(), epoch)
    group.close()
