"""Strict framing and trusted-channel binding; no transport/GPU completion claim."""

import json
from dataclasses import replace

import pytest
from sglang.srt.disaggregation.pvd.rank_install_wire import (
    MAX_FRAME_BYTES,
    RankInstallExchange,
    RankInstallMessage,
)
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError
from test_pvd_sparse_install import protocol, receipt


def exchange(ranks=(0, 1)):
    return RankInstallExchange(
        protocol(ranks=ranks), peer_epochs={r: f"worker-{r}" for r in ranks}
    )


def event(kind, epoch, rank=0):
    return RankInstallMessage(
        kind,
        f"worker-{rank}",
        receipt(epoch, rank),
        decode_tokens=epoch.target_tokens if kind == "parked" else None,
        reason="rank failed" if kind == "failed" else None,
    )


@pytest.mark.parametrize(
    "kind", ["prepared", "parked", "applied", "install", "resume", "failed"]
)
def test_all_message_kinds_roundtrip_exact_identity(kind):
    message = event(kind, protocol().begin(0))
    assert RankInstallMessage.decode(message.encode()) == message


@pytest.mark.parametrize(
    "change",
    [
        "extra",
        "missing",
        "protocol",
        "kind",
        "peer",
        "receipt_extra",
        "epoch_extra",
        "bool_rank",
        "bool_round",
        "huge_count",
        "negative_count",
        "long_text",
        "surrogate",
        "park_count",
        "reason",
        "nested_type",
        "duplicate",
        "nan",
        "utf8",
        "oversize",
        "depth",
    ],
)
def test_bad_wire_messages_are_refused_without_coordinator_changes(change):
    x = exchange()
    epoch = x.begin(0)
    data = json.loads(event("prepared", epoch).encode())
    if change == "extra":
        data["extra"] = 1
    elif change == "missing":
        del data["reason"]
    elif change == "protocol":
        data["protocol"] = "v-next"
    elif change == "kind":
        data["kind"] = "maybe-applied"
    elif change == "peer":
        data["peer_epoch"] = ""
    elif change == "receipt_extra":
        data["receipt"]["rkey"] = 9
    elif change == "epoch_extra":
        data["receipt"]["epoch"]["extra"] = 0
    elif change == "bool_rank":
        data["receipt"]["rank"] = False
    elif change == "bool_round":
        data["receipt"]["epoch"]["round"] = True
    elif change == "huge_count":
        data["receipt"]["epoch"]["target_tokens"] = 1 << 63
    elif change == "negative_count":
        data["receipt"]["epoch"]["target_tokens"] = -1
    elif change == "long_text":
        data["receipt"]["staging_id"] = "x" * 1025
    elif change == "surrogate":
        data["receipt"]["staging_id"] = "\ud800"
    elif change == "park_count":
        data["decode_tokens"] = 0
    elif change == "reason":
        data["reason"] = "not a failure"
    elif change == "nested_type":
        data["receipt"]["epoch"] = []
    raw = json.dumps(data).encode()
    if change == "duplicate":
        raw = raw.replace(b'"reason": null', b'"reason": null, "reason": null')
    elif change == "nan":
        raw = raw.replace(b'"reason": null', b'"reason": NaN')
    elif change == "utf8":
        raw = b"\xff"
    elif change == "oversize":
        raw = b" " * (MAX_FRAME_BYTES + 1)
    elif change == "depth":
        raw = b"[" * 2000 + b"]" * 2000
    before = x.coordinator.snapshot()
    with pytest.raises(InstallProtocolError):
        x.receive(raw, peer_rank=0)
    assert x.coordinator.snapshot() == before


@pytest.mark.parametrize(
    "field,value",
    [("decode_tokens", None), ("decode_tokens", True), ("decode_tokens", 1)],
)
def test_parked_requires_exact_integer_count(field, value):
    with pytest.raises(InstallProtocolError):
        replace(event("parked", protocol().begin(0)), **{field: value})


@pytest.mark.parametrize("reason", [None, "", "x" * 1025])
def test_failure_requires_bounded_reason(reason):
    with pytest.raises(InstallProtocolError):
        replace(event("failed", protocol().begin(0)), reason=reason)


@pytest.mark.parametrize(
    "change", ["channel", "incarnation", "command", "staging", "old_round"]
)
def test_bound_channel_and_prepared_bank_are_not_inferred_from_peer(change):
    x = exchange()
    epoch = x.begin(0)
    x.receive(event("prepared", epoch).encode(), peer_rank=0)
    bad = event("parked", epoch)
    peer_rank = 0
    if change == "channel":
        peer_rank = 1
    elif change == "incarnation":
        bad = replace(bad, peer_epoch="restarted-worker")
    elif change == "command":
        bad = event("install", epoch)
    elif change == "staging":
        bad = replace(bad, receipt=replace(bad.receipt, staging_id="other"))
    elif change == "old_round":
        bad = replace(bad, receipt=receipt(replace(epoch, round=2), 0))
    before = x.coordinator.snapshot()
    with pytest.raises(InstallProtocolError):
        x.receive(bad.encode(), peer_rank=peer_rank)
    assert x.coordinator.snapshot() == before


@pytest.mark.parametrize("ranks", [(0,), (0, 1), (2, 4, 6, 8)])
def test_install_requires_all_parked_resume_requires_all_applied(ranks):
    x = exchange(ranks)
    epoch = x.begin(0)
    for rank in ranks:
        x.receive(event("prepared", epoch, rank).encode(), peer_rank=rank)
    assert x.install_commands(epoch) == {}
    for rank in ranks:
        x.receive(event("parked", epoch, rank).encode(), peer_rank=rank)
    install = x.install_commands(epoch)
    assert set(install) == set(ranks)
    assert install == x.install_commands(epoch)
    for rank in ranks:
        assert RankInstallMessage.decode(install[rank]) == event("install", epoch, rank)
        with pytest.raises(InstallProtocolError, match="all ranks"):
            x.resume_commands(epoch)
        x.receive(event("applied", epoch, rank).encode(), peer_rank=rank)
        x.receive(event("applied", epoch, rank).encode(), peer_rank=rank)
    resume = x.resume_commands(epoch)
    for rank in ranks:
        assert RankInstallMessage.decode(resume[rank]) == event("resume", epoch, rank)
    assert x.coordinator.snapshot()["round"] == 1
    x.begin(3)
    with pytest.raises(InstallProtocolError, match="all ranks"):
        x.resume_commands(epoch)


def test_partial_application_and_failure_cannot_produce_resume():
    x = exchange()
    epoch = x.begin(0)
    for rank in (0, 1):
        x.receive(event("prepared", epoch, rank).encode(), peer_rank=rank)
        x.receive(event("parked", epoch, rank).encode(), peer_rank=rank)
    x.install_commands(epoch)
    x.receive(event("applied", epoch, 0).encode(), peer_rank=0)
    x.receive(event("failed", epoch, 1).encode(), peer_rank=1)
    with pytest.raises(InstallProtocolError, match="all ranks"):
        x.resume_commands(epoch)
    assert not x.coordinator.can_decode(0)
    commands = x.cancel_commands("rank failed; stop all peers")
    assert all(
        RankInstallMessage.decode(raw).kind == "failed" for raw in commands.values()
    )


def test_peer_binding_is_copied_and_cannot_be_rebound():
    epochs = {0: "worker-0", 1: "worker-1"}
    x = RankInstallExchange(protocol(), peer_epochs=epochs)
    epochs[0] = "new-process"
    assert x.peer_epochs[0] == "worker-0"
    with pytest.raises(TypeError):
        x.peer_epochs[0] = "new-process"


def test_failure_with_wrong_prepared_bank_cannot_cancel_request():
    x = exchange()
    epoch = x.begin(0)
    x.receive(event("prepared", epoch).encode(), peer_rank=0)
    failed = event("failed", epoch)
    failed = replace(failed, receipt=replace(failed.receipt, staging_id="other"))
    with pytest.raises(InstallProtocolError, match="prepared bank"):
        x.receive(failed.encode(), peer_rank=0)
    assert x.coordinator.snapshot()["state"] == "preparing"
