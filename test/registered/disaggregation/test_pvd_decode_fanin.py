"""Existing async bootstrap and full Prompt unpack with real V HTTP/fake KV."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import MethodType
from types import SimpleNamespace as NS

import pytest
import torch
from aiohttp import web
from sglang.srt.disaggregation.pvd.client import PVDCoordinatorClient
from sglang.srt.disaggregation.pvd.conn import PVDKVManager, _AsyncControlLoop
from sglang.srt.disaggregation.pvd.control_server import create_coordinator_app
from sglang.srt.disaggregation.pvd.decode_fanin import PVDDecodeFanInSession
from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeRefresher
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_fanin_coordinator import group
from test_pvd_fanin_store import setup
from test_pvd_fanin_writer import BatchEngine
from test_pvd_transfer_admission import base_args


def test_rank_packed_bootstrap_admission_counts_both_gpu_buffers():
    manager = NS(
        full_kv_fanin_rank_packed=True,
        local_shard_manifest=lambda n: NS(expected_bytes=128),
    )
    req = NS(origin_input_ids=list(range(5)))
    assert PVDKVManager.bootstrap_staging_bytes(manager, req) == 256
    manager.full_kv_fanin_rank_packed = False
    assert PVDKVManager.bootstrap_staging_bytes(manager, req) == 128


def test_uncertain_local_scatter_completion_retains_both_buffers(monkeypatch):
    from sglang.srt.disaggregation.pvd import decode_fanin

    session = PVDDecodeFanInSession.__new__(PVDDecodeFanInSession)
    receipt = {"delivered": True}
    session._network_receipt = receipt
    session._rank_packed_plan = object()
    session._canonical_staging = object()
    session.staging = object()
    session.manager = NS(full_kv_fanin_triton_scatter=True)
    session._scatter_completion_unknown = False
    calls = []

    def synchronize():
        calls.append("sync")
        if calls.count("sync") == 2:
            raise RuntimeError("CUDA completion unknown")

    def launch(*args, **kwargs):
        calls.append("launch")
        raise RuntimeError("launch failed after earlier work was queued")

    session.synchronize = synchronize
    monkeypatch.setattr(decode_fanin, "scatter_rank_packed_bytes", launch)
    with pytest.raises(RuntimeError, match="completion unknown"):
        session.unpack(receipt)
    assert calls == ["sync", "launch", "sync"]
    assert session._scatter_completion_unknown

    session._fanin_lock = asyncio.Lock()
    session._fanin = None
    session._closed = True
    session._refresh_owner = "still pinned"
    assert asyncio.run(session.progress_close()) is False
    assert session._refresh_owner == "still pinned"


def test_triton_scatter_flag_does_not_silently_activate_on_pd():
    from sglang.srt.arg_groups.pvd_disaggregation_hook import handle_pvd_disaggregation

    args = base_args(tp_size=1, pvd_rank_rails="mlx5_7")
    args.disaggregation_topology = "pd"
    args.pvd_full_kv_fanin_triton_scatter = True
    with pytest.raises(ValueError, match="requires PVD Decode"):
        handle_pvd_disaggregation(args)


@pytest.mark.parametrize(
    "tp_size,fault,rank_packed",
    [
        (1, None, False),
        (2, None, False),
        (4, None, False),
        (2, "unpack", False),
        (1, "preflight", False),
        (1, "cancel", False),
        (1, "waiting", False),
        (2, "waiting", False),
        (1, None, True),
        (2, None, True),
        (2, "unpack", True),
        (1, "cancel", True),
        (1, "waiting", True),
        (1, "old_v", True),
    ],
)
def test_real_decode_bootstrap_fanin_rank_agreement_and_reuse(
    tp_size, fault, rank_packed
):
    use_waiting = fault == "waiting"
    if use_waiting:
        fault = None
    with setup(
        publish=False,
        engine=BatchEngine() if rank_packed else None,
        native_batch=rank_packed,
    ) as c:
        control = _AsyncControlLoop()
        parent, _, _ = group(c, max_records=32)
        if fault == "old_v":
            current_health = parent.health

            async def old_health():
                health = await current_health()
                health["full_kv_fanin"].pop("protocols", None)
                return health

            parent.health = old_health

        async def serve():
            runner = web.AppRunner(create_coordinator_app(parent))
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            return (
                runner,
                f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}",
            )

        runner, url = control.submit(serve()).result(10)
        barrier, slots = threading.Barrier(tp_size, timeout=10), [None] * tp_size
        sessions, refreshers, clients = [], [], []
        storage = c.manifest.layout
        heads = 4 // tp_size
        layout = replace(
            storage,
            tp_size=tp_size,
            kv_heads_per_rank=heads,
            extra={
                **storage.extra,
                "component_bytes_per_token": [heads * 2],
                "component_token_shapes": [[heads, 1]],
            },
        )
        for rank in range(tp_size):

            def gather(value, rank=rank):
                slots[rank] = value
                barrier.wait()
                result = list(slots)
                barrier.wait()
                return result

            client = PVDCoordinatorClient(url)
            clients.append(client)
            req = NS(
                rid="fanin-decode",
                origin_input_ids=list(range(5)),
                output_ids=[7],
                pvd_delivery_id="decode",
                req_pool_idx=1,
                is_retracted=False,
                done=False,
            )
            req.finished = lambda req=req: req.done

            async def stored(*args):
                return parent.entries[c.key].to_dict()

            manager = NS(
                full_kv_fanin_max_slices=64,
                full_kv_fanin_rank_packed=rank_packed,
                tp_rank=rank,
                tp_size=tp_size,
                key_for=lambda r: c.key,
                client_for=lambda r, client=client: client,
                control=control,
                layout=lambda: layout,
                page_size=4,
                rail="mlx5_7",
                transfer_engine=c.engine,
                transfer_budget=TransferBudget(4096, 32),
                kv_pool=NS(
                    k_buffer=[torch.full((16, heads, 1), 99, dtype=torch.float16)],
                    v_buffer=[],
                ),
                gather_rank_objects=gather,
                wait_for_stored_entry=stored,
                pending_decode_closes=[],
                retain_pending_close=lambda s: None,
                scheduler=NS(
                    server_args=NS(
                        pvd_kv_refresh_interval=4,
                        pvd_full_kv_fanin_response_bytes=65536,
                    ),
                    req_to_token_pool=NS(req_to_token=torch.arange(16).repeat(2, 1)),
                ),
            )
            session = PVDDecodeFanInSession(manager, req)
            control.submit(session.initialize(None)).result(10)
            manager.decode_sessions = {c.key: session}
            sessions.append(session)
            refreshers.append(PVDDecodeRefresher(manager))
            if use_waiting:
                manager.waiting_queue_bootstrap = True
                manager.bootstrap_gates = {}
                manager.worker_epoch = session.receiver_epoch
                manager.decode_refresher = refreshers[-1]
                manager.scheduler.waiting_queue = [req]
                manager.bootstrap_staging_bytes = lambda r: 8 * heads * 2
                for name in (
                    "open_bootstrap_gate",
                    "bootstrap_gate_for",
                    "bootstrap_runnable",
                    "_bootstrap_staging_headroom",
                    "enter_waiting_queue",
                ):
                    setattr(
                        manager, name, MethodType(getattr(PVDKVManager, name), manager)
                    )
                manager.open_bootstrap_gate(req).mark_source_ready()
        if fault == "unpack":

            def failed_unpack(reply):
                raise RuntimeError("rank import failed")

            sessions[-1].unpack = failed_unpack
        if fault == "preflight":
            sessions[0].manager.rail = "mlx5_other"
        try:
            with ThreadPoolExecutor(max_workers=tp_size) as executor:

                def step(method):
                    def invoke(r, s):
                        if not use_waiting:
                            return getattr(r, method)(
                                *([s.req],) if method == "start_bootstrap" else ()
                            )
                        failures = s.manager.enter_waiting_queue([s.req])
                        if s.manager.bootstrap_runnable(s.req):
                            assert s.require_initial_prompt() is s._initial_receipt
                            return [s.req], failures
                        return [], failures

                    futures = [
                        executor.submit(
                            invoke,
                            r,
                            s,
                        )
                        for r, s in zip(refreshers, sessions)
                    ]
                    return [f.result(15) for f in futures]

                assert step("start_bootstrap") == [([], [])] * tp_size
                # Drain only the control futures: no scheduler-side unpack yet.
                for r in refreshers:
                    r._bootstrap[2].result(10)
                if fault == "cancel":
                    sessions[0].req.done = True
                result = step("poll_bootstrap")
                if fault:
                    assert all(errors for _, errors in result)
                    assert all(s.clock.round == 0 for s in sessions)
                    assert not any(
                        record.state == "released"
                        for record in parent.fanin.records.values()
                    )
                    if fault != "unpack":
                        assert all(
                            torch.all(s.manager.kv_pool.k_buffer[0] == 99)
                            for s in sessions
                        )
                else:
                    assert result == [([], [])] * tp_size  # ACK still awaits a poll.
                    for r in refreshers:
                        r._bootstrap[2].result(10)
                    result = step("poll_bootstrap")
                    assert all(
                        done == [s.req] and not errors
                        for (done, errors), s in zip(result, sessions)
                    )
                    assert all(
                        s.clock.round == 1 and s._initial_receipt is not None
                        for s in sessions
                    )
                    full = torch.cat(
                        [raw.view(torch.float16).reshape(8, 2, 1) for raw in c.raw],
                        dim=1,
                    )
                    for rank, s in enumerate(sessions):
                        target = s.manager.kv_pool.k_buffer[0]
                        assert torch.equal(
                            target[:5], full[:5, rank * heads : (rank + 1) * heads]
                        )
                        assert torch.all(target[5:] == 99)
                    region_ids = [s.registration.descriptor.region_id for s in sessions]
                    for s in sessions:
                        s.req.output_ids.extend([1, 2, 3, 4])
                    futures = [
                        executor.submit(r.refresh, [s.req])
                        for r, s in zip(refreshers, sessions)
                    ]
                    assert [f.result(15) for f in futures] == [[]] * tp_size
                    assert [
                        s.registration.descriptor.region_id for s in sessions
                    ] == region_ids
                    assert all(s.clock.round == 2 for s in sessions)
                    assert all(
                        torch.all(s.manager.kv_pool.k_buffer[0][5:] == 99)
                        for s in sessions
                    )
                    if rank_packed:
                        assert c.engine.batch_calls
                        assert all(
                            len(slices) == len(offsets) == 1
                            for slices, offsets in c.engine.batch_calls
                        )
            for s in sessions:
                assert control.submit(s.close()).result(10)
                assert s.manager.transfer_budget.snapshot()["used_staging_bytes"] == 0
            assert parent.entries[c.key].active_delivery_count == 0
        finally:
            for s in sessions:
                control.submit(s.close()).result(10)
            for client in clients:
                control.submit(client.close()).result(10)
            control.submit(runner.cleanup()).result(10)
            control.loop.call_soon_threadsafe(control.loop.stop)
            control.thread.join(10)
            control.loop.close()


@pytest.mark.parametrize(
    "case",
    [
        "enabled",
        "rank_packed_enabled",
        "triton_scatter_enabled",
        "triton_scatter_missing",
        "rank_packed_missing",
        "legacy_tp1",
        "missing_bound",
        "no_waiting",
        "prefill",
    ],
)
def test_fanin_serving_config_requires_complete_opt_in(case):
    from sglang.srt.arg_groups.pvd_disaggregation_hook import handle_pvd_disaggregation

    args = base_args(
        tp_size=1,
        pvd_rank_rails="mlx5_7",
        pvd_waiting_queue_bootstrap=True,
        pvd_full_kv_fanin_max_slices=64,
        pvd_full_kv_fanin_response_bytes=65536,
    )
    if case in ("enabled", "rank_packed_enabled", "triton_scatter_enabled"):
        args.pvd_full_kv_fanin_rank_packed = case != "enabled"
        args.pvd_full_kv_fanin_triton_scatter = case == "triton_scatter_enabled"
        handle_pvd_disaggregation(args)
        assert args.disable_overlap_schedule and args.disable_radix_cache
        return
    if case == "triton_scatter_missing":
        args.pvd_full_kv_fanin_triton_scatter = True
        expected = "requires rank-packed"
    elif case == "rank_packed_missing":
        args.pvd_full_kv_fanin_rank_packed = True
        args.pvd_full_kv_fanin_max_slices = None
        expected = "requires bounded"
    elif case == "legacy_tp1":
        args.pvd_full_kv_fanin_max_slices = args.pvd_full_kv_fanin_response_bytes = None
        expected = "supports"
    elif case == "missing_bound":
        args.pvd_full_kv_fanin_response_bytes = None
        expected = "response-bytes"
    elif case == "no_waiting":
        args.pvd_waiting_queue_bootstrap = False
        expected = "waiting-queue"
    else:
        args.disaggregation_mode = "prefill"
        expected = "requires Decode"
    with pytest.raises(ValueError, match=expected):
        handle_pvd_disaggregation(args)
