"""CUDA participant policies on CPU storage; real-CUDA cases are separate.

The two logical ranks here are not a GPU TP execution or network collective.
"""

from dataclasses import replace

import pytest
import torch
from sglang.srt.disaggregation.pvd.cpu_rank_install import CPURankInstallParticipant
from sglang.srt.disaggregation.pvd.cuda_rank_install import CUDARankInstallParticipant
from sglang.srt.disaggregation.pvd.cuda_working_set import CUDASparseWorkingSet
from sglang.srt.disaggregation.pvd.rank_install_wire import (
    RankInstallExchange,
    RankInstallMessage,
)
from sglang.srt.disaggregation.pvd.sparse_install import (
    InstallProtocolError,
    RankInstallCoordinator,
)
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_cuda_working_set import options, packed_payloads, policy_bank
from test_pvd_sparse_working_set import fixture


def setup(monkeypatch):
    banks, budgets, peers = {}, {}, {}
    for rank in range(2):
        banks[rank], budgets[rank], _ = policy_bank(monkeypatch)
        peers[rank] = CUDARankInstallParticipant(
            banks[rank], rank=rank, peer_epoch=f"worker-{rank}", interval=4
        )
    coordinator = RankInstallCoordinator(
        "r",
        "inc",
        "entry",
        rank_layouts={0: "layout", 1: "layout"},
        interval=4,
        lead_tokens=1,
    )
    exchange = RankInstallExchange(
        coordinator, peer_epochs={r: f"worker-{r}" for r in peers}
    )
    return peers, exchange, banks, budgets


def stage(peer, epoch, *, device="cpu"):
    tokens = (0, 1, 2, 3) if epoch.target_tokens == 0 else (1, 3)
    rows, guard, released = packed_payloads(epoch.target_tokens, tokens, device=device)
    for row in rows:
        row.spec = replace(row.spec, operation_id=epoch.operation_id)
    prepared = peer.stage(epoch, rows, source_guard=guard)
    guard.request_release()
    assert released == [True]
    return prepared


def complete(peers, exchange, count):
    epoch = exchange.begin(count)
    for rank, peer in peers.items():
        exchange.receive(stage(peer, epoch), peer_rank=rank)
        exchange.receive(peer.park(epoch.target_tokens), peer_rank=rank)
    commands = exchange.install_commands(epoch)
    for rank, peer in peers.items():
        reply = peer.command(commands[rank])
        assert peer.command(commands[rank]) == reply
        exchange.receive(reply, peer_rank=rank)
        with (
            pytest.raises(InstallProtocolError, match="global resume"),
            peer.read(epoch.target_tokens),
        ):
            pass
    resumes = exchange.resume_commands(epoch)
    assert not exchange.can_decode(epoch.target_tokens)
    for rank, peer in peers.items():
        reply = peer.command(resumes[rank])
        assert peer.command(resumes[rank]) == reply
        exchange.receive(reply, peer_rank=rank)
        if rank == 0:
            assert not exchange.can_decode(epoch.target_tokens)
    assert exchange.can_decode(epoch.target_tokens)
    return epoch, commands, resumes


def test_two_rounds_require_all_rank_apply_and_resume(monkeypatch):
    peers, exchange, _, budgets = setup(monkeypatch)
    complete(peers, exchange, 0)
    complete(peers, exchange, 3)
    for rank, peer in peers.items():
        with peer.read(4) as groups:
            assert groups[(0, 0)][0].token_ids == (1, 3)
        assert peer.snapshot()["round"] == 2
        peer.close()
        assert budgets[rank].snapshot()["used_staging_bytes"] == 0


def test_reader_drain_is_required_for_park(monkeypatch):
    peers, exchange, banks, _ = setup(monkeypatch)
    complete(peers, exchange, 0)
    epoch = exchange.begin(3)
    with peers[0].read(3):
        exchange.receive(stage(peers[0], epoch), peer_rank=0)
        assert peers[0].park(4) is None
    assert banks[0].snapshot()["readers"] == 0
    assert peers[0].park(4)
    with pytest.raises(InstallProtocolError, match="global resume"), peers[0].read(3):
        pass
    for peer in peers.values():
        peer.close()


def test_missing_applied_and_resume_ack_never_authorize_group_decode(monkeypatch):
    peers, exchange, _, _ = setup(monkeypatch)
    epoch = exchange.begin(0)
    for rank, peer in peers.items():
        exchange.receive(stage(peer, epoch), peer_rank=rank)
        exchange.receive(peer.park(0), peer_rank=rank)
    commands = exchange.install_commands(epoch)
    exchange.receive(peers[0].command(commands[0]), peer_rank=0)
    with pytest.raises(InstallProtocolError, match="all ranks must apply"):
        exchange.resume_commands(epoch)
    assert not exchange.can_decode(0)
    exchange.receive(peers[1].command(commands[1]), peer_rank=1)
    resumes = exchange.resume_commands(epoch)
    exchange.receive(peers[0].command(resumes[0]), peer_rank=0)
    assert not exchange.can_decode(0)
    with pytest.raises(InstallProtocolError, match="global resume"), peers[1].read(0):
        pass
    exchange.receive(peers[1].command(resumes[1]), peer_rank=1)
    assert exchange.can_decode(0)
    for peer in peers.values():
        peer.close()


def test_cancel_during_read_cannot_release_until_completion(monkeypatch):
    peers, exchange, banks, budgets = setup(monkeypatch)
    complete(peers, exchange, 0)
    completions = []
    monkeypatch.setattr(banks[0], "_synchronize", lambda: completions.append(True))
    with peers[0].read(0):
        peers[0].command(exchange.cancel_commands("cancel")[0])
        with pytest.raises(SparsePayloadError, match="owned by a forward"):
            peers[0].close()
        assert not completions
        assert budgets[0].snapshot()["used_staging_bytes"] == 192
    assert completions == [True]
    peers[0].close()
    peers[1].close()
    assert budgets[0].snapshot()["used_staging_bytes"] == 0


@pytest.mark.parametrize("fault", ["copy", "reader"])
def test_unknown_completion_never_emits_usable_rank_receipt(monkeypatch, fault):
    peers, exchange, banks, budgets = setup(monkeypatch)
    old_resume = None
    if fault == "reader":
        _, _, old_resume = complete(peers, exchange, 0)

    def fail():
        raise RuntimeError("CUDA completion unknown")

    monkeypatch.setattr(banks[0], "_synchronize", fail)
    with pytest.raises(RuntimeError, match="completion unknown"):
        if fault == "copy":
            stage(peers[0], exchange.begin(0))
        else:
            with peers[0].read(0):
                pass
    assert budgets[0].snapshot()["used_staging_bytes"] == 192
    if fault == "copy":
        assert peers[0].snapshot()["phase"] is None
    else:
        with pytest.raises(SparsePayloadError, match="quarantined"):
            peers[0].command(old_resume[0])
    with pytest.raises(SparsePayloadError, match="quarantined"):
        peers[0].close()
    peers[1].close()


@pytest.mark.parametrize(
    "field", ["peer", "rank", "staging", "operation", "early_resume"]
)
def test_foreign_or_early_command_does_not_mutate_bank(monkeypatch, field):
    peers, exchange, _, _ = setup(monkeypatch)
    peer = peers[0]
    epoch = exchange.begin(0)
    prepared = RankInstallMessage.decode(stage(peer, epoch))
    peer.park(0)
    command = replace(prepared, kind="install")
    if field == "peer":
        command = replace(command, peer_epoch="stale")
    elif field == "rank":
        command = replace(command, receipt=replace(command.receipt, rank=1))
    elif field == "staging":
        command = replace(command, receipt=replace(command.receipt, staging_id="other"))
    elif field == "operation":
        command = replace(
            command,
            receipt=replace(
                command.receipt, epoch=replace(epoch, operation_id="other")
            ),
        )
    else:
        command = replace(command, kind="resume")
    before = peer.snapshot()
    with pytest.raises(InstallProtocolError):
        peer.command(command.encode())
    assert peer.snapshot() == before
    for p in peers.values():
        p.close()


def test_partial_install_failure_stays_terminal(monkeypatch):
    peers, exchange, banks, budgets = setup(monkeypatch)
    epoch = exchange.begin(0)
    for rank, peer in peers.items():
        exchange.receive(stage(peer, epoch), peer_rank=rank)
        exchange.receive(peer.park(0), peer_rank=rank)
    commands = exchange.install_commands(epoch)
    original = banks[0].install

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("after swap")

    monkeypatch.setattr(banks[0], "install", fail)
    with pytest.raises(RuntimeError, match="after swap"):
        peers[0].command(commands[0])
    assert peers[0].snapshot()["terminal"]
    with pytest.raises(InstallProtocolError, match="terminal"), peers[0].read(0):
        pass
    assert budgets[0].snapshot()["used_staging_bytes"] == 192
    for peer in peers.values():
        peer.close()


def test_old_resume_does_not_replace_pending_round(monkeypatch):
    peers, exchange, _, _ = setup(monkeypatch)
    _, _, old = complete(peers, exchange, 0)
    epoch = exchange.begin(3)
    stage(peers[0], epoch)
    before = peers[0].snapshot()
    with pytest.raises(InstallProtocolError, match="stale or foreign"):
        peers[0].command(old[0])
    assert peers[0].snapshot() == before
    for peer in peers.values():
        peer.close()


def test_cpu_and_cuda_participants_do_not_cross_accept_bank_types(monkeypatch):
    bank, _, _ = policy_bank(monkeypatch)
    with pytest.raises(InstallProtocolError, match="CPU"):
        CPURankInstallParticipant(bank, rank=0, peer_epoch="e", interval=4)
    cpu, _, _ = fixture()
    with pytest.raises(InstallProtocolError, match="CUDA"):
        CUDARankInstallParticipant(cpu, rank=0, peer_epoch="e", interval=4)
    cpu.close()
    bank.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="real CUDA rank bank needs CUDA"
)
def test_real_cuda_local_participant_completes_handshake():
    budget = TransferBudget(4096, 3)
    bank = CUDASparseWorkingSet(
        device="cuda:0", dtype=torch.float32, budget=budget, **options()
    )
    peer = CUDARankInstallParticipant(bank, rank=0, peer_epoch="e", interval=4)
    coordinator = RankInstallCoordinator(
        "r", "inc", "entry", rank_layouts={0: "layout"}, interval=4, lead_tokens=1
    )
    exchange = RankInstallExchange(coordinator, peer_epochs={0: "e"})
    for count in (0, 3):
        epoch = exchange.begin(count)
        exchange.receive(stage(peer, epoch, device="cuda:0"), peer_rank=0)
        exchange.receive(peer.park(epoch.target_tokens), peer_rank=0)
        exchange.receive(peer.command(exchange.install_commands(epoch)[0]), peer_rank=0)
        with (
            pytest.raises(InstallProtocolError, match="global resume"),
            peer.read(epoch.target_tokens),
        ):
            pass
        exchange.receive(peer.command(exchange.resume_commands(epoch)[0]), peer_rank=0)
        with peer.read(epoch.target_tokens) as groups:
            observed = groups[(0, 0)][1].clone()
        assert observed.is_cuda
    peer.close()
    assert budget.snapshot()["used_staging_bytes"] == 0
