"""Per-rank banks never reopen just because their local copy was installed."""

from dataclasses import replace

import pytest
from sglang.srt.disaggregation.pvd.cpu_rank_install import CPURankInstallParticipant
from sglang.srt.disaggregation.pvd.rank_install_wire import (
    RankInstallExchange,
    RankInstallMessage,
)
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from test_pvd_sparse_install import cpu_group, payload


def setup():
    group, banks, budgets = cpu_group()
    peers = {
        r: CPURankInstallParticipant(b, rank=r, peer_epoch=f"worker-{r}", interval=4)
        for r, b in banks.items()
    }
    exchange = RankInstallExchange(
        group.coordinator, peer_epochs={r: f"worker-{r}" for r in peers}
    )
    return peers, exchange, banks, budgets


def complete(peers, exchange, count):
    epoch = exchange.begin(count)
    for rank, peer in peers.items():
        exchange.receive(peer.stage(epoch, [payload(epoch, rank)]), peer_rank=rank)
        exchange.receive(peer.park(epoch.target_tokens), peer_rank=rank)
    commands = exchange.install_commands(epoch)
    for rank, peer in peers.items():
        reply = peer.command(commands[rank])
        assert reply == peer.command(commands[rank])
        exchange.receive(reply, peer_rank=rank)
        with (
            pytest.raises(InstallProtocolError, match="global resume"),
            peer.read(epoch.target_tokens),
        ):
            pass
    resume = exchange.resume_commands(epoch)
    for rank, peer in peers.items():
        peer.command(resume[rank])
        peer.command(resume[rank])
    return epoch, commands, resume


def test_two_rounds_gate_reads_and_refund_banks():
    peers, exchange, _, budgets = setup()
    complete(peers, exchange, 0)
    for rank, peer in peers.items():
        with peer.read(0) as groups:
            assert groups[(0, rank)][0].token_ids == (0, 1, 2, 3)
    complete(peers, exchange, 3)
    for rank, peer in peers.items():
        with peer.read(4) as groups:
            assert groups[(0, rank)][0].token_ids == (1, 3)
        assert peer.snapshot()["round"] == 2
        peer.close()
        assert budgets[rank].snapshot()["used_staging_bytes"] == 0


def test_reader_must_drain_before_park_and_no_new_read_after_park():
    peers, exchange, _, _ = setup()
    complete(peers, exchange, 0)
    epoch = exchange.begin(3)
    with peers[0].read(3):
        exchange.receive(peers[0].stage(epoch, [payload(epoch, 0)]), peer_rank=0)
        assert peers[0].park(4) is None
    assert peers[0].park(4)
    with (
        pytest.raises(InstallProtocolError, match="global resume"),
        peers[0].read(3),
    ):
        pass
    for peer in peers.values():
        peer.close()


@pytest.mark.parametrize(
    "change",
    ["peer", "rank", "staging", "operation", "resume_early", "install_early", "event"],
)
def test_invalid_commands_cannot_change_bank_or_state(change):
    peers, exchange, _, budgets = setup()
    peer = peers[0]
    epoch = exchange.begin(0)
    prepared = RankInstallMessage.decode(peer.stage(epoch, [payload(epoch, 0)]))
    command = replace(prepared, kind="install")
    if change == "peer":
        command = replace(command, peer_epoch="old-process")
    elif change == "rank":
        command = replace(command, receipt=replace(command.receipt, rank=1))
    elif change == "staging":
        command = replace(command, receipt=replace(command.receipt, staging_id="other"))
    elif change == "operation":
        command = replace(
            command,
            receipt=replace(
                command.receipt, epoch=replace(epoch, operation_id="other")
            ),
        )
    elif change == "resume_early":
        command = replace(command, kind="resume")
    elif change == "event":
        command = prepared
    before = peer.snapshot()
    with pytest.raises(InstallProtocolError):
        peer.command(command.encode())
    assert peer.snapshot() == before
    for r, p in peers.items():
        p.close()
        assert budgets[r].snapshot()["used_staging_bytes"] == 0


def test_old_resume_cannot_clear_new_pending_bank():
    peers, exchange, _, _ = setup()
    _, _, old = complete(peers, exchange, 0)
    epoch = exchange.begin(3)
    peers[0].stage(epoch, [payload(epoch, 0)])
    before = peers[0].snapshot()
    with pytest.raises(InstallProtocolError, match="stale or foreign"):
        peers[0].command(old[0])
    assert peers[0].snapshot() == before
    for peer in peers.values():
        peer.close()


def test_cancel_during_read_keeps_charge_until_reader_drains():
    peers, exchange, _, budgets = setup()
    complete(peers, exchange, 0)
    with peers[0].read(0):
        peers[0].command(exchange.cancel_commands("cancel request")[0])
        with pytest.raises(SparsePayloadError, match="owned by a forward"):
            peers[0].close()
        assert budgets[0].snapshot()["used_staging_bytes"] > 0
    peers[0].close()
    peers[1].close()
    assert budgets[0].snapshot()["used_staging_bytes"] == 0


def test_partial_install_exception_never_resumes_local_rank(monkeypatch):
    peers, exchange, banks, _ = setup()
    epoch = exchange.begin(0)
    for rank, peer in peers.items():
        exchange.receive(peer.stage(epoch, [payload(epoch, rank)]), peer_rank=rank)
        exchange.receive(peer.park(0), peer_rank=rank)
    commands = exchange.install_commands(epoch)
    install = banks[0].install

    def fail(*args, **kwargs):
        install(*args, **kwargs)
        raise RuntimeError("failed after local swap")

    monkeypatch.setattr(banks[0], "install", fail)
    with pytest.raises(RuntimeError):
        peers[0].command(commands[0])
    assert peers[0].snapshot()["terminal"]
    with pytest.raises(InstallProtocolError, match="terminal"), peers[0].read(0):
        pass
    for peer in peers.values():
        peer.close()
