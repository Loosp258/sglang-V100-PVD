"""Real receiver completion policy + CPU CUDA-bank math; no native RDMA/GPU."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest
import torch
from sglang.srt.disaggregation.pvd.bootstrap import BootstrapGate
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import TargetExecutionArbiter
from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeSession
from sglang.srt.disaggregation.pvd.protocol import KVEntryKey
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferCapacityError
from test_pvd_cuda_prompt_bootstrap import setup


def received(monkeypatch, *, limit=65536, complete=True):
    c, group, importer, source, budget, _ = setup(monkeypatch, limit=limit)
    request = NS(
        rid="r",
        origin_input_ids=[1, 2, 3, 4],
        output_ids=[5],
        pvd_delivery_id="delivery",
        req_pool_idx=1,
        is_retracted=False,
        finished=lambda: False,
    )
    key = KVEntryKey("model", "entry", "entry")
    gate = BootstrapGate("entry:bootstrap", "inc", 4)
    gate.enter_waiting_queue()
    gate.mark_source_ready()
    ticket = gate.begin()
    gate.mark_received(ticket)
    gate.mark_installed(ticket)
    gate.handoff()
    allocator = NS(get_kvcache=lambda: c.pool)
    cache = NS(req_to_token_pool=c.req, token_to_kv_pool_allocator=allocator)
    manager = NS(
        scheduler=NS(
            req_to_token_pool=c.req,
            tree_cache=cache,
            waiting_queue=[request],
            server_args=NS(pvd_kv_refresh_interval=4),
        ),
        tp_size=1,
        tp_rank=0,
        page_size=1,
        kv_pool=c.pool,
        key_for=lambda req: key,
        client_for=lambda req: object(),
        layout=lambda: NS(fingerprint="layout"),
        bootstrap_gate_for=lambda req: gate,
    )
    session = PVDDecodeSession(manager, request)
    manager.decode_sessions = {key: session}
    session.receiver_epoch, session.generation = "inc", "generation"
    session.pages = torch.tensor([8, 3, 6, 1])
    session.clock.begin(0)
    if complete:
        # A controlled ACK completion in this fixture, not transport evidence.
        # test_pvd3 covers the real delivery/unpack/ACK generator minting site.
        session._complete_refresh()
    import sglang.srt.disaggregation.pvd.cuda_request_release as release_module

    monkeypatch.setattr(release_module, "_require_supported_pools", lambda cache: None)
    arbiter = TargetExecutionArbiter()
    return NS(**locals())


def install(c):
    return c.importer.install_received(
        c.session,
        arbiter=c.arbiter,
        pool_owner=c.c.owner,
        cache=c.cache,
    )


def test_receiver_backed_import_installs_full_prompt_without_search(monkeypatch):
    c = received(monkeypatch)
    receipt = c.session.require_initial_prompt()
    epoch = install(c)
    assert epoch.target_tokens == 0 and c.group.can_decode(0)
    assert c.importer._received_receipt is receipt
    assert c.session._cuda_prompt_importer is c.importer
    assert not c.arbiter.busy and not c.session._closed
    assert c.budget.snapshot()["used_staging_bytes"] == 0
    with pytest.raises(InstallProtocolError, match="unclaimed"):
        install(c)
    c.group.close()


@pytest.mark.parametrize(
    "fault",
    [
        "no-ack",
        "closed",
        "lease",
        "pending",
        "round",
        "slot",
        "prompt",
        "output",
        "generation",
        "epoch",
        "pages",
        "layout",
        "replaced",
        "gate",
        "waiting",
        "finished",
        "retracted",
    ],
)
def test_stale_or_incomplete_receiver_cannot_import(monkeypatch, fault):
    c = received(monkeypatch, complete=fault != "no-ack")
    if fault == "closed":
        c.session._closed = True
    elif fault == "lease":
        c.session.lease_error = "expired"
    elif fault == "pending":
        c.session.clock.pending = ("other", 0)
    elif fault == "round":
        c.session.clock.round = 2
    elif fault == "slot":
        c.request.req_pool_idx = 2
    elif fault == "prompt":
        c.request.origin_input_ids[0] = 9
    elif fault == "output":
        c.request.output_ids.append(6)
    elif fault == "generation":
        c.session.generation = "new"
    elif fault == "epoch":
        c.session.receiver_epoch = "new"
    elif fault == "pages":
        c.session.pages[0] = 2
    elif fault == "layout":
        c.manager.layout = lambda: NS(fingerprint="new")
    elif fault == "replaced":
        c.manager.decode_sessions[c.key] = object()
    elif fault == "gate":
        c.gate.close()
    elif fault == "waiting":
        c.manager.scheduler.waiting_queue.clear()
    elif fault == "finished":
        c.request.finished = lambda: True
    elif fault == "retracted":
        c.request.is_retracted = True
    with pytest.raises(RuntimeError, match="initial Prompt|waiting-queue"):
        install(c)
    assert not c.group.can_decode(0) and not c.arbiter.busy
    assert c.budget.snapshot()["used_staging_bytes"] == 0
    assert c.importer._received_session is None
    c.group.close()


def test_arbitrary_clock_or_gate_success_does_not_mint_a_receipt(monkeypatch):
    c = received(monkeypatch, complete=False)
    c.session.release_refresh()
    c.session.clock.complete(c.session.clock.pending[0])
    assert c.gate.is_runnable
    with pytest.raises(RuntimeError, match="initial Prompt completion"):
        install(c)
    c.group.close()


def test_post_receive_row_reuse_is_refused_before_copy(monkeypatch):
    c = received(monkeypatch)
    c.c.req.req_to_token[1, 0] = 2
    with pytest.raises(InstallProtocolError, match="mapping changed"):
        install(c)
    assert not c.arbiter.busy and not c.importer._held
    assert c.session._cuda_prompt_importer is c.importer
    assert c.importer._used  # Source pin consumed this one-shot attempt.
    c.group.close()


def test_capacity_refusal_is_retryable_without_new_delivery(monkeypatch):
    c = received(monkeypatch)
    c.budget.reserve("busy", 65536, 0)
    with pytest.raises(TransferCapacityError):
        install(c)
    assert c.session._cuda_prompt_importer is None and not c.arbiter.busy
    c.budget.release("busy")
    assert install(c).target_tokens == 0
    assert c.session.clock.round == 1
    c.group.close()


@pytest.mark.parametrize("fail_at", [1, 2])
def test_unknown_receive_import_retains_real_owners_and_poisoned_pools(
    monkeypatch, fail_at
):
    c = received(monkeypatch)
    calls = []

    def fence():
        calls.append(True)
        if len(calls) == fail_at:
            raise RuntimeError("completion unknown")

    monkeypatch.setattr(c.importer, "_synchronize", fence)
    with pytest.raises(RuntimeError, match="completion unknown"):
        install(c)
    assert c.arbiter.busy and c.importer._receive_lease is not None
    assert c.session._cuda_prompt_importer is c.importer
    assert c.importer._received_session is c.session
    c.c.owner.request_release()
    assert not c.c.released
    assert c.budget.snapshot()["used_staging_bytes"] > 0
    assert "quarantined" in c.allocator.pvd_cuda_retirement_error
    assert c.c.req.pvd_cuda_retirement_error == c.allocator.pvd_cuda_retirement_error
    with pytest.raises(InstallProtocolError, match="unclaimed"):
        install(c)
    assert len(calls) == fail_at


def test_target_busy_or_foreign_thread_never_claims_session(monkeypatch):
    c = received(monkeypatch)
    lease = c.arbiter.acquire()
    with pytest.raises(ValueError, match="busy"):
        install(c)
    assert c.importer._received_session is None
    c.arbiter.release(lease)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(ValueError, match="owner thread"):
            executor.submit(install, c).result()
    assert c.importer._received_session is None
    c.group.close()
