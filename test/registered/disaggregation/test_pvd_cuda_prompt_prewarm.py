"""Req-arrival prompt-only sidecar ownership and reconciliation tests."""

import asyncio
import os
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import TargetExecutionArbiter
from sglang.srt.disaggregation.pvd.cuda_prompt_prewarm import CUDAPromptPrewarmer
from sglang.srt.disaggregation.pvd.cuda_refresh_driver import CUDARefreshDriver
from sglang.srt.disaggregation.pvd.cuda_target_startup import (
    CUDATargetServingComponents,
)
from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeSession
from sglang.srt.disaggregation.pvd.probe_lane_identity import (
    ProbeLaneCheckpointIdentity,
)
from sglang.srt.disaggregation.pvd.probe_lane_unix import ProbeLaneUnixClient
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


def _case(tmp_path, *, prompt_length=8):
    tmp_path.chmod(0o700)
    driver = CUDARefreshDriver(
        TargetExecutionArbiter(), max_requests=2, max_prefix_tokens=16_384
    )
    req = SimpleNamespace(
        rid="request-1",
        origin_input_ids=[7] * prompt_length,
        output_ids=[11],
        pvd_transfer_id="entry-1",
        pvd_delivery_id="delivery-1",
        is_retracted=False,
        finished=lambda: False,
    )
    key = SimpleNamespace(transfer_id="entry-1")
    manager = SimpleNamespace(worker_epoch="epoch-1")
    manager.key_for = lambda actual_req: key if actual_req is req else None
    session = object.__new__(PVDDecodeSession)
    session.manager = manager
    session.req = req
    session.key = key
    session.receiver_epoch = "epoch-1"
    session.consumer_id = "delivery-1"
    session.clock = SimpleNamespace(round=0, pending=None)
    session._closed = False
    checkpoint = ProbeLaneCheckpointIdentity("a" * 64, "b" * 64, 100, 10)
    # The behavior under test never opens a socket. Bypass the Linux-only
    # private-path check while preserving the exact client type requirement.
    client = object.__new__(ProbeLaneUnixClient)
    client.expected_server_pid = os.getpid()
    client.reply_budget = TransferBudget(1 << 20, 2)
    client.background_loop = None
    client._background_inflight = set()
    prewarmer = CUDAPromptPrewarmer(
        driver,
        lane_client=client,
        checkpoint=checkpoint,
        target_model_id="target-model",
        probe_config=SimpleNamespace(
            target_model_id="target-model",
            layers=(0, 2),
            head_start=0,
            head_count=2,
        ),
        head_dim=4,
        timeout_seconds=10,
        max_prefix_tokens=16_383,
    )
    driver.cuda_prompt_prewarm = prewarmer
    return driver, prewarmer, req, session, key


def _receipt(req, key, **changes):
    values = {
        "request_id": req.rid,
        "key": key,
        "receiver_epoch": "epoch-1",
        "prompt": tuple(req.origin_input_ids),
        "outputs": tuple(req.output_ids),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _finish_driver(driver):
    if driver._owns_loop and not driver._loop.is_closed():
        driver.begin_shutdown()
        for _ in range(4):
            if not driver.cuda_prompt_prewarm.pending:
                break
            driver.poll()
        assert not driver.cuda_prompt_prewarm.pending
        driver.close_loop()


def test_ticket_uses_exact_receiver_epoch_entry_and_prompt_only_bounds(tmp_path):
    driver, prewarmer, req, session, key = _case(tmp_path)
    try:
        owner = prewarmer.start(req, session)
        assert owner is not None
        ticket = owner.ticket
        assert ticket.window.incarnation == session.receiver_epoch
        assert ticket.window.entry_transfer_id == key.transfer_id
        assert ticket.window.target_tokens == 0
        assert ticket.window.query_positions == (len(req.origin_input_ids),)
        assert ticket.window.prefix.request_id == req.rid
        assert ticket.window.prefix.tokens == tuple(req.origin_input_ids)
        assert ticket.window.prefix.version == (
            f"{session.receiver_epoch}:prompt-only:{req.pvd_delivery_id}"
        )
        assert len(ticket.layers) == 2
        assert ticket.max_reply_bytes == 2 * 2 * 4 * 4

        receipt = _receipt(req, key)
        session.require_initial_prompt = lambda: receipt
        assert prewarmer.reconcile(session) is owner
        assert prewarmer.owns(owner, req, session)

        # A second owner cannot replace this cache identity before the first
        # request has transferred ownership to formal admission/refresh.
        other_req = SimpleNamespace(rid="request-2")
        assert prewarmer.start(other_req, session) is None
        assert prewarmer.snapshot()["started"] == 1
        assert prewarmer.snapshot()["skipped_busy"] == 1
    finally:
        _finish_driver(driver)


@pytest.mark.parametrize(
    "changes",
    [
        {"request_id": "other-request"},
        {"receiver_epoch": "other-epoch"},
        {"key": SimpleNamespace(transfer_id="other-entry")},
        {"prompt": (99,)},
        {"outputs": (12,)},
    ],
)
def test_receipt_mismatch_cancels_owner_and_falls_back(tmp_path, changes):
    driver, prewarmer, req, session, key = _case(tmp_path)
    try:
        owner = prewarmer.start(req, session)
        assert owner is not None
        session.require_initial_prompt = lambda: _receipt(req, key, **changes)

        assert prewarmer.reconcile(session) is None
        assert not prewarmer.owns(owner, req, session)
        assert prewarmer.snapshot()["cancelled"] == 1
        # The canceled I/O task remains tracked until the owner loop drains it.
        assert prewarmer.pending
    finally:
        _finish_driver(driver)


def test_prompt_over_lane_cap_is_skipped_before_ticket_issue(tmp_path):
    driver, prewarmer, req, session, _ = _case(tmp_path, prompt_length=16_384)
    try:
        assert prewarmer.start(req, session) is None
        assert prewarmer.snapshot()["started"] == 0
        assert prewarmer.snapshot()["skipped_bounds"] == 1
    finally:
        _finish_driver(driver)


@pytest.mark.parametrize("mismatch", ["request", "entry", "epoch"])
def test_req_arrival_requires_exact_session_entry_and_epoch(tmp_path, mismatch):
    driver, prewarmer, req, session, _ = _case(tmp_path)
    if mismatch == "request":
        session.req = SimpleNamespace(rid=req.rid)
    elif mismatch == "entry":
        req.pvd_transfer_id = "other-entry"
    else:
        session.receiver_epoch = "other-epoch"
    try:
        assert prewarmer.start(req, session) is None
        assert prewarmer.snapshot()["started"] == 0
        assert prewarmer.snapshot()["skipped_invalid"] == 1
    finally:
        _finish_driver(driver)


def test_early_ticket_timeout_keeps_normal_fallback_available(tmp_path):
    driver, prewarmer, req, session, key = _case(tmp_path)

    @asynccontextmanager
    async def timeout_request(ticket):
        assert ticket is owner.ticket
        raise TimeoutError("sidecar timed out")
        yield

    try:
        owner = prewarmer.start(req, session)
        assert owner is not None
        prewarmer.lane_client.request = timeout_request
        driver.poll()
        assert owner.task.done()
        assert owner.task.result() is False
        assert owner.state == "failed"
        assert prewarmer.snapshot()["failed"] == 1

        # Reconciliation still succeeds; admission may use its ordinary n=0
        # full-prefix ticket when the early sidecar request timed out.
        session.require_initial_prompt = lambda: _receipt(req, key)
        assert prewarmer.reconcile(session) is owner
        assert prewarmer.owns(owner, req, session)
    finally:
        _finish_driver(driver)


def test_target_close_is_nonblocking_until_cancelled_owner_drains(tmp_path):
    driver, prewarmer, req, session, _ = _case(tmp_path)
    owner = prewarmer.start(req, session)
    assert owner is not None
    calls = []
    backend = object()

    class Binding:
        def close(self):
            calls.append("binding")
            driver.close_loop()

    binding = Binding()
    target = CUDATargetServingComponents(
        scheduler=SimpleNamespace(
            pvd_cuda_binding=binding,
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(attn_backend=backend)
            ),
        ),
        backend=backend,
        native_backend=object(),
        workspace=SimpleNamespace(
            snapshot=lambda: {"quarantine": None, "active": False},
            close=lambda: calls.append("workspace"),
        ),
        arbiter=driver.arbiter,
        execution_lock=object(),
        driver=driver,
        executor=SimpleNamespace(_active=False, _quarantined=False),
        pool_owner=object(),
        binding=binding,
    )

    assert target.close_drained() is False
    assert driver._closing
    assert calls == []
    assert prewarmer.pending
    driver.poll()
    assert not prewarmer.pending
    assert target.close_drained() is True
    assert calls == ["binding", "workspace"]


def test_owner_is_held_until_refresh_install_and_early_task_drain():
    driver = CUDARefreshDriver.__new__(CUDARefreshDriver)
    driver.arbiter = TargetExecutionArbiter()
    driver._owns_loop = True
    driver._loop = None
    calls = []
    loop = asyncio.new_event_loop()
    owner = SimpleNamespace(task=loop.create_future())
    append_task = loop.create_future()
    record = SimpleNamespace(
        early_prewarm_owner=owner,
        early_refresh_installed=False,
        prewarm_task=append_task,
    )

    class Prewarmer:
        def complete(self, actual_owner):
            assert actual_owner is owner
            calls.append(actual_owner)

    driver.cuda_prompt_prewarm = Prewarmer()
    try:
        driver._complete_early_prewarm_if_ready(record)
        assert calls == []
        record.early_refresh_installed = True
        driver._complete_early_prewarm_if_ready(record)
        assert calls == []
        owner.task.set_result(True)
        driver._complete_early_prewarm_if_ready(record)
        assert calls == []
        append_task.set_result(True)
        driver._complete_early_prewarm_if_ready(record)
        assert calls == [owner]
        assert record.early_prewarm_owner is None
    finally:
        loop.close()
