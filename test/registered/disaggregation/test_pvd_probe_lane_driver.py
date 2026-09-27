"""Off-owner Q must not retain D's formal target-forward arbiter."""

import asyncio
import os
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest
import torch
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import TargetExecutionArbiter
from sglang.srt.disaggregation.pvd.cuda_refresh_driver import CUDARefreshDriver
from sglang.srt.disaggregation.pvd.prediction import QueryVectors
from sglang.srt.disaggregation.pvd.probe_lane_identity import (
    ProbeLaneCheckpointIdentity,
)
from sglang.srt.disaggregation.pvd.probe_lane_protocol import ProbeLaneReply
from sglang.srt.disaggregation.pvd.probe_lane_unix import (
    ProbeLaneUnixClient,
    ProbeLaneUnixServer,
)
from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_cuda_prefetch_request import controller
from test_pvd_cuda_refresh_driver import pump, req
from test_pvd_cuda_sparse_delivery import case, complete
from test_pvd_probe_lane_unix import _start_background_loop, _stop_background_loop


def _prewarm_reply(bound):
    queries = tuple(
        QueryVectors(
            bound.target_model_id,
            f"{bound.window.prefix.version}:probe:{bound.nonce}",
            layer,
            bound.head_start,
            bound.head_count,
            bound.window.query_positions,
            len(bound.window.query_positions),
            torch.ones(
                (
                    len(bound.window.query_positions),
                    bound.head_count,
                    bound.head_dim,
                ),
                dtype=torch.float32,
            ),
            prefix_version=bound.window.prefix.version,
            positional_encoding="rope_applied",
            request_id=bound.window.prefix.request_id,
        )
        for layer in bound.layers
    )
    return ProbeLaneReply(
        bound.nonce,
        bound.window.operation_id,
        bound.prefix_digest,
        bound.weights_sha256,
        bound.tokenizer_sha256,
        queries,
    )


def _prewarm_lane(directory, io_loop=None):
    return ProbeLaneUnixClient(
        directory,
        "probe.sock",
        expected_server_pid=os.getpid(),
        reply_budget=TransferBudget(1 << 20, 4),
        background_loop=io_loop,
    )


@pytest.mark.parametrize(
    "ending,background",
    [
        ("install", False),
        ("install", True),
        ("cancel", False),
        ("retract", False),
        ("cancel_delivery", False),
    ],
)
def test_driver_releases_target_arbiter_while_private_q_is_pending(
    monkeypatch, ending, background
):
    monkeypatch.setenv("PVD_REFRESH_POLL_TURNS", "1" if ending == "install" else "4")
    driver = CUDARefreshDriver(
        TargetExecutionArbiter(), max_requests=2, max_prefix_tokens=64
    )
    ctx = case(monkeypatch)
    c = driver._loop.run_until_complete(ctx.__aenter__())
    driver._loop.run_until_complete(complete(c, 0, tuple(range(8))))
    control, captures, _ = controller(c, monkeypatch)
    search = PVDShardSearchClient(c.client.base_url)
    request = req()
    checkpoint = ProbeLaneCheckpointIdentity("a" * 64, "b" * 64, 100, 10)
    with tempfile.TemporaryDirectory(prefix="pvd-lane-", dir="/tmp") as name:
        directory = Path(name)
        directory.chmod(0o700)
        entered, release = (
            (threading.Event(), threading.Event())
            if background
            else (asyncio.Event(), asyncio.Event())
        )
        io_loop, io_thread = _start_background_loop() if background else (None, None)

        async def handler(bound):
            entered.set()
            if background:
                await asyncio.to_thread(release.wait)
            else:
                await release.wait()
            queries = tuple(
                QueryVectors(
                    bound.target_model_id,
                    f"{bound.window.prefix.version}:probe:{bound.nonce}",
                    layer,
                    0,
                    1,
                    bound.window.query_positions,
                    1,
                    c.pool.k_buffer[layer][3, 0].repeat(1, 1, 1).to(torch.float32),
                    prefix_version=bound.window.prefix.version,
                    positional_encoding="rope_applied",
                    request_id=bound.window.prefix.request_id,
                )
                for layer in bound.layers
            )
            return ProbeLaneReply(
                bound.nonce,
                bound.window.operation_id,
                bound.prefix_digest,
                bound.weights_sha256,
                bound.tokenizer_sha256,
                queries,
            )

        server_budget = TransferBudget(1 << 20, 1)
        service = ProbeLaneUnixServer(
            directory,
            "probe.sock",
            expected_client_pid=os.getpid(),
            target_model_id="target/model-8b",
            weights_sha256=checkpoint.weights_sha256,
            tokenizer_sha256=checkpoint.tokenizer_sha256,
            handler=handler,
            reply_budget=server_budget,
        )
        if background:
            asyncio.run_coroutine_threadsafe(service.start(), io_loop).result(5)
        else:
            driver._loop.run_until_complete(service.start())
        lane = ProbeLaneUnixClient(
            directory,
            "probe.sock",
            expected_server_pid=os.getpid(),
            reply_budget=TransferBudget(1 << 20, 1),
            background_loop=io_loop,
        )
        driver.register(
            request,
            control,
            clients={0: search},
            timeout_seconds=10,
            lane_client=lane,
            lane_checkpoint=checkpoint,
        )
        try:
            request.output_ids.extend((3, 4, 5))
            driver.poll()
            # A newly scheduled private ticket begins in this same Scheduler
            # poll, leaving its full lead window for sidecar computation.
            assert lane.reply_budget.snapshot()["reservations"] == 1
            if background:
                assert entered.wait(5)
                formal_forward = driver.arbiter.acquire()
                try:
                    release.set()
                    future = next(iter(lane._background_inflight))
                    assert future.result(timeout=5)
                    assert not driver._records[request.rid].refresh.done()
                finally:
                    driver.arbiter.release(formal_forward)
            pump(driver, c, entered.is_set)
            assert not driver.arbiter.busy
            assert driver._records[request.rid].capture_lease is None
            assert captures == []
            formal_forward = driver.arbiter.acquire()
            driver.arbiter.release(formal_forward)
            delivery_region = None
            if ending == "cancel_delivery":
                release.set()

                def pending_write():
                    for delivery in c.store.entries[c.entry.key].deliveries.values():
                        handle = delivery.transfer_handle
                        if (
                            handle is not None
                            and not handle.transport_state.is_locally_safe_to_release
                        ):
                            return True
                    return False

                pump(driver, c, pending_write, finish=False)
                active_round = c.sink._rounds[control._active]
                delivery_region = next(iter(active_round.values())).identity.region_id
                assert delivery_region not in c.engine.released
                driver.cancel(request, "cancel during RDMA delivery")
                driver.poll()
                assert delivery_region not in c.engine.released
            elif ending == "cancel":
                driver.cancel(request, "test cancellation")
            elif ending == "retract":
                request.is_retracted = True
                driver.poll()
            release.set()
            if ending == "install":
                pump(driver, c, lambda: driver._records[request.rid].ready)
                request.output_ids.append(6)
                pump(
                    driver,
                    c,
                    lambda: (
                        control.group.coordinator.snapshot()["installed_tokens"] == 4
                    ),
                )
            else:
                pump(driver, c, lambda: not driver._records)
                assert control.group.coordinator.snapshot()["installed_tokens"] == 0
                assert c.registry.snapshot() == {}
                assert c.registry.budget.snapshot()["reservations"] == 0
                assert c.sink.snapshot()["pending_rounds"] == 0
                if delivery_region is not None:
                    assert delivery_region in c.engine.released
            assert captures == []
            assert not driver.arbiter.busy
            assert lane.reply_budget.snapshot()["reservations"] == 0
            assert server_budget.snapshot()["reservations"] == 0
        finally:
            release.set()
            driver.begin_shutdown()
            pump(driver, c, lambda: not driver._records)
            if background:
                asyncio.run_coroutine_threadsafe(service.aclose(), io_loop).result(5)
                _stop_background_loop(io_loop, io_thread)
            else:
                driver._loop.run_until_complete(service.aclose())
            driver._loop.run_until_complete(search.close())
            driver._loop.run_until_complete(ctx.__aexit__(None, None, None))
            driver.close_loop()


@pytest.mark.skipif(
    not hasattr(socket, "SO_PEERCRED"),
    reason="Linux Unix peer credentials required",
)
def test_opt_in_prewarm_uses_prompt_and_p_token_without_advancing_refresh(
    monkeypatch,
):
    driver = CUDARefreshDriver(
        TargetExecutionArbiter(), max_requests=2, max_prefix_tokens=64
    )
    ctx = case(monkeypatch)
    c = driver._loop.run_until_complete(ctx.__aenter__())
    driver._loop.run_until_complete(complete(c, 0, tuple(range(8))))
    control, captures, _ = controller(c, monkeypatch)
    search = PVDShardSearchClient(c.client.base_url)
    request = req()
    observed = []
    checkpoint = ProbeLaneCheckpointIdentity("a" * 64, "b" * 64, 100, 10)
    server_budget = TransferBudget(1 << 20, 1)
    with tempfile.TemporaryDirectory(prefix="pvd-prewarm-", dir="/tmp") as name:
        directory = Path(name)
        directory.chmod(0o700)
        service = ProbeLaneUnixServer(
            directory,
            "probe.sock",
            expected_client_pid=os.getpid(),
            target_model_id="target/model-8b",
            weights_sha256=checkpoint.weights_sha256,
            tokenizer_sha256=checkpoint.tokenizer_sha256,
            handler=lambda bound: observed.append(bound) or _prewarm_reply(bound),
            reply_budget=server_budget,
        )
        driver._loop.run_until_complete(service.start())
        lane = _prewarm_lane(directory)
        driver.register(
            request,
            control,
            clients={0: search},
            timeout_seconds=10,
            lane_client=lane,
            lane_checkpoint=checkpoint,
            prewarm_sidecar=True,
        )
        before = control.group.coordinator.snapshot()
        try:
            driver.poll()
            pump(
                driver,
                c,
                lambda: (
                    driver.snapshot()["requests"][request.rid]["sidecar_prewarm"]
                    in ("ready", "failed")
                ),
            )
            after = control.group.coordinator.snapshot()
            assert len(observed) == 1
            ticket = observed[0]
            assert ticket.window.incarnation == control.group.coordinator.identity[1]
            assert (
                ticket.window.entry_transfer_id == control.group.coordinator.identity[2]
            )
            assert ticket.window.prefix.tokens == tuple(
                request.origin_input_ids
            ) + tuple(request.output_ids)
            assert ticket.window.target_tokens == 0
            assert ticket.window.query_positions == (len(ticket.window.prefix.tokens),)
            assert after["installed_tokens"] == before["installed_tokens"] == 0
            assert after["next_boundary"] == before["next_boundary"] == 4
            assert (
                driver.snapshot()["requests"][request.rid]["sidecar_prewarm"] == "ready"
            )
            assert driver.snapshot()["requests"][request.rid]["committed_tokens"] == 0
            assert not driver.snapshot()["requests"][request.rid]["refresh_pending"]
            assert captures == []
            assert control.pending_install_boundary is None
            assert lane.reply_budget.snapshot()["reservations"] == 0
            assert server_budget.snapshot()["reservations"] == 0
        finally:
            driver.begin_shutdown()
            pump(driver, c, lambda: not driver._records)
            driver._loop.run_until_complete(service.aclose())
            driver._loop.run_until_complete(search.close())
            driver._loop.run_until_complete(ctx.__aexit__(None, None, None))
            driver.close_loop()


@pytest.mark.skipif(
    not hasattr(socket, "SO_PEERCRED"),
    reason="Linux Unix peer credentials required",
)
def test_cancelled_prewarm_drains_lane_budget_before_controller_close(monkeypatch):
    driver = CUDARefreshDriver(
        TargetExecutionArbiter(), max_requests=2, max_prefix_tokens=64
    )
    ctx = case(monkeypatch)
    c = driver._loop.run_until_complete(ctx.__aenter__())
    driver._loop.run_until_complete(complete(c, 0, tuple(range(8))))
    control, captures, _ = controller(c, monkeypatch)
    search = PVDShardSearchClient(c.client.base_url)
    request = req()
    checkpoint = ProbeLaneCheckpointIdentity("a" * 64, "b" * 64, 100, 10)
    server_budget = TransferBudget(1 << 20, 2)
    entered, release = threading.Event(), threading.Event()
    io_loop, io_thread = _start_background_loop()

    async def handler(bound):
        entered.set()
        await asyncio.to_thread(release.wait)
        return _prewarm_reply(bound)

    with tempfile.TemporaryDirectory(prefix="pvd-prewarm-", dir="/tmp") as name:
        directory = Path(name)
        directory.chmod(0o700)
        service = ProbeLaneUnixServer(
            directory,
            "probe.sock",
            expected_client_pid=os.getpid(),
            target_model_id="target/model-8b",
            weights_sha256=checkpoint.weights_sha256,
            tokenizer_sha256=checkpoint.tokenizer_sha256,
            handler=handler,
            reply_budget=server_budget,
        )
        asyncio.run_coroutine_threadsafe(service.start(), io_loop).result(5)
        lane = _prewarm_lane(directory, io_loop)
        driver.register(
            request,
            control,
            clients={0: search},
            timeout_seconds=10,
            lane_client=lane,
            lane_checkpoint=checkpoint,
            prewarm_sidecar=True,
        )
        record = driver._records[request.rid]
        try:
            driver.poll()
            assert entered.wait(5)
            assert lane.reply_budget.snapshot()["reservations"] == 1
            assert server_budget.snapshot()["reservations"] == 1
            request.output_ids.extend((3, 4, 5))
            driver.poll()
            assert record.prewarm_overlapped_refresh
            assert driver.snapshot()["requests"][request.rid][
                "sidecar_prewarm_overlapped_refresh"
            ]
            assert record.refresh is not None
            deadline = time.monotonic() + 5
            while lane.reply_budget.snapshot()["reservations"] < 2:
                assert time.monotonic() < deadline
                driver.poll()
                time.sleep(0.001)
            assert server_budget.snapshot()["reservations"] == 2
            driver.cancel(request, "cancel during sidecar prewarm")
            driver.poll()
            assert record.stopping
            assert record.close_task is None
            assert record.prewarm_task is not None
            assert not record.prewarm_task.done()
            assert lane.reply_budget.snapshot()["reservations"] == 1
            release.set()
            pump(driver, c, lambda: not driver._records)
            assert record.prewarm_state == "cancelled"
            assert lane.reply_budget.snapshot()["reservations"] == 0
            assert server_budget.snapshot()["reservations"] == 0
            assert control.group.coordinator.snapshot()["installed_tokens"] == 0
            assert captures == []
        finally:
            release.set()
            driver.begin_shutdown()
            pump(driver, c, lambda: not driver._records)
            asyncio.run_coroutine_threadsafe(service.aclose(), io_loop).result(5)
            _stop_background_loop(io_loop, io_thread)
            driver._loop.run_until_complete(search.close())
            driver._loop.run_until_complete(ctx.__aexit__(None, None, None))
            driver.close_loop()
