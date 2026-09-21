"""Spawn independent CPU rank banks; exchange bounded bytes, never tensors.

This is a local control-protocol acceptance, not torch.distributed/NCCL, model
TP execution, CUDA visibility, native RDMA, or a production Scheduler launch.
"""

import argparse
import json
import multiprocessing
import os
import sys
import types
import uuid
from dataclasses import asdict
from pathlib import Path

FRAME_LIMIT = 16384


def _bootstrap():
    root = Path(__file__).resolve().parents[3]
    for name in ("sglang", "sglang.srt", "sglang.srt.disaggregation"):
        module = types.ModuleType(name)
        module.__path__ = [str(root / "python" / Path(*name.split(".")))]
        sys.modules[name] = module


def _json(value):
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > FRAME_LIMIT:
        raise ValueError("fixture frame too large")
    return raw


def _worker(connection, rank, peer_epoch):
    _bootstrap()
    import torch
    from sglang.srt.disaggregation.pvd.cpu_rank_install import CPURankInstallParticipant
    from sglang.srt.disaggregation.pvd.sparse_install import InstallEpoch
    from sglang.srt.disaggregation.pvd.sparse_payload import (
        SparseKVPayload,
        SparseKVSpec,
    )
    from sglang.srt.disaggregation.pvd.sparse_working_set import CPUSparseWorkingSet
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

    torch.set_num_threads(1)
    budget = TransferBudget(4096, 2)
    bank = CPUSparseWorkingSet(
        request_id="request",
        incarnation="request-inc",
        entry_transfer_id="entry",
        layout_fingerprint=f"layout-{rank}",
        expected_groups=((0, rank),),
        prompt_tokens=4,
        head_dim=2,
        max_union_tokens=2,
        budget=budget,
    )
    peer = CPURankInstallParticipant(bank, rank=rank, peer_epoch=peer_epoch, interval=4)
    held = None
    fail_install = False
    install = bank.install

    def maybe_fail(*args, **kwargs):
        install(*args, **kwargs)
        if fail_install:
            raise RuntimeError("injected failure after local bank swap")

    bank.install = maybe_fail
    connection.send_bytes(
        _json({"fixture": "ready", "pid": os.getpid(), "peer_epoch": peer_epoch})
    )
    try:
        while True:
            raw = connection.recv_bytes(FRAME_LIMIT)
            data = json.loads(raw)
            try:
                if "protocol" in data:
                    reply = peer.command(raw)
                    connection.send_bytes(
                        reply
                        if reply is not None
                        else _json(
                            {"fixture": "command_done", "state": peer.snapshot()}
                        )
                    )
                    continue
                action = data["fixture"]
                if action == "stage":
                    epoch = InstallEpoch(**data["epoch"])
                    tokens = (0, 1, 2, 3) if epoch.target_tokens == 0 else (1, 3)
                    spec = SparseKVSpec(
                        epoch.request_id,
                        epoch.incarnation,
                        epoch.operation_id,
                        epoch.target_tokens,
                        epoch.entry_transfer_id,
                        "index",
                        "mapping",
                        f"layout-{rank}",
                        0,
                        rank,
                        tokens,
                    )
                    keys = torch.tensor(
                        [[rank * 100 + t * 10 + dim for dim in (0, 1)] for t in tokens],
                        dtype=torch.float32,
                    )
                    payload = SparseKVPayload(spec, torch.stack((keys, keys + 1000)))
                    try:
                        reply = peer.stage(epoch, [payload])
                    finally:
                        payload.close()
                    connection.send_bytes(reply)
                elif action == "park":
                    reply = peer.park(data["count"])
                    connection.send_bytes(
                        reply
                        if reply is not None
                        else _json({"fixture": "wait_reader"})
                    )
                elif action == "hold":
                    if held is not None:
                        raise RuntimeError("fixture reader already held")
                    held = peer.read(data["count"])
                    held.__enter__()
                    connection.send_bytes(_json({"fixture": "held"}))
                elif action == "release":
                    held.__exit__(None, None, None)
                    held = None
                    connection.send_bytes(_json({"fixture": "reader_released"}))
                elif action == "read":
                    with peer.read(data["count"]) as groups:
                        spec, tensor = groups[(0, rank)]
                        reply = {
                            "fixture": "read",
                            "tokens": spec.token_ids,
                            "kv": tensor.tolist(),
                        }
                    connection.send_bytes(_json(reply))
                elif action == "fail_install":
                    fail_install = True
                    connection.send_bytes(_json({"fixture": "armed"}))
                elif action == "stop":
                    peer.close()
                    connection.send_bytes(
                        _json({"fixture": "closed", "budget": budget.snapshot()})
                    )
                    return
                else:
                    raise ValueError("unknown fixture control")
            except (ValueError, RuntimeError) as exc:
                # Fixture reports failure; never hides it as APPLIED.
                connection.send_bytes(
                    _json(
                        {
                            "fixture": "error",
                            "error": str(exc),
                            "state": peer.snapshot(),
                        }
                    )
                )
    finally:
        if held is not None:
            held.__exit__(None, None, None)
        peer.close()
        connection.close()


def run(ranks=2, fault="none"):
    if (
        type(ranks) is not int
        or not 1 <= ranks <= 8
        or fault not in ("none", "install", "exit")
    ):
        raise ValueError("explicit rank count 1..8 and known fault required")
    if fault != "none" and ranks < 2:
        raise ValueError("partial install faults require at least two ranks")
    _bootstrap()
    from sglang.srt.disaggregation.pvd.rank_install_wire import (
        RankInstallExchange,
        RankInstallMessage,
    )
    from sglang.srt.disaggregation.pvd.sparse_install import (
        InstallProtocolError,
        RankInstallCoordinator,
    )

    context = multiprocessing.get_context("spawn")
    processes, connections, pids, closed = {}, {}, {}, {}
    epochs = {rank: uuid.uuid4().hex for rank in range(ranks)}
    exchange = RankInstallExchange(
        RankInstallCoordinator(
            "request",
            "request-inc",
            "entry",
            rank_layouts={r: f"layout-{r}" for r in epochs},
            interval=4,
            lead_tokens=1,
        ),
        peer_epochs=epochs,
    )

    def receive(rank):
        if not connections[rank].poll(30):
            raise TimeoutError(f"rank {rank} fixture timed out; not completion proof")
        return connections[rank].recv_bytes(FRAME_LIMIT)

    def rpc(rank, data):
        connections[rank].send_bytes(data if isinstance(data, bytes) else _json(data))
        return receive(rank)

    def fixture(rank, action, **fields):
        return json.loads(rpc(rank, {"fixture": action, **fields}))

    def accept(rank, raw, kind):
        message = RankInstallMessage.decode(raw)
        assert message.kind == kind
        exchange.receive(raw, peer_rank=rank)

    def cannot_resume(epoch):
        try:
            exchange.resume_commands(epoch)
        except InstallProtocolError:
            return
        raise AssertionError("coordinator resumed a partially installed bank")

    def blocked(rank, count):
        reply = fixture(rank, "read", count=count)
        assert reply["fixture"] == "error" and "global resume" in reply["error"], reply

    def check_values(rank, count, tokens):
        reply = fixture(rank, "read", count=count)
        assert reply["tokens"] == list(tokens)
        expected = [
            [
                [rank * 100 + t * 10 + dim + kind * 1000 for dim in (0, 1)]
                for t in tokens
            ]
            for kind in (0, 1)
        ]
        assert reply["kv"] == expected

    try:
        for rank in range(ranks):
            parent, child = context.Pipe(duplex=True)
            process = context.Process(target=_worker, args=(child, rank, epochs[rank]))
            process.start()
            child.close()
            processes[rank], connections[rank] = process, parent
        for rank in range(ranks):
            ready = json.loads(receive(rank))
            assert ready["fixture"] == "ready" and ready["peer_epoch"] == epochs[rank]
            pids[rank] = ready["pid"]
        assert len(set(pids.values())) == ranks and os.getpid() not in pids.values()

        initial = exchange.begin(0)
        for rank in range(ranks):
            accept(
                rank,
                rpc(rank, {"fixture": "stage", "epoch": asdict(initial)}),
                "prepared",
            )
            accept(rank, rpc(rank, {"fixture": "park", "count": 0}), "parked")
        commands = exchange.install_commands(initial)
        for rank in reversed(range(ranks)):
            cannot_resume(initial)
            reply = rpc(rank, commands[rank])
            accept(rank, reply, "applied")
            assert rpc(rank, commands[rank]) == reply
            blocked(rank, 0)
        old_resume = exchange.resume_commands(initial)
        for rank in range(ranks):
            assert not exchange.can_decode(0)
            reply = rpc(rank, old_resume[rank])
            assert not exchange.can_decode(0)  # local reopen is not a global ACK
            accept(rank, reply, "resumed")
            accept(rank, rpc(rank, old_resume[rank]), "resumed")
            check_values(rank, 0, (0, 1, 2, 3))
        assert exchange.can_decode(0)

        refresh = exchange.begin(3)
        assert fixture(0, "hold", count=3)["fixture"] == "held"
        for rank in range(ranks):
            accept(
                rank,
                rpc(rank, {"fixture": "stage", "epoch": asdict(refresh)}),
                "prepared",
            )
        stale = json.loads(rpc(0, old_resume[0]))
        assert stale["fixture"] == "error" and "stale or foreign" in stale["error"]
        assert fixture(0, "park", count=4)["fixture"] == "wait_reader"
        for rank in range(1, ranks):
            accept(rank, rpc(rank, {"fixture": "park", "count": 4}), "parked")
        assert exchange.install_commands(refresh) == {}
        assert fixture(0, "release")["fixture"] == "reader_released"
        accept(0, rpc(0, {"fixture": "park", "count": 4}), "parked")
        commands = exchange.install_commands(refresh)
        for rank in range(ranks):
            cannot_resume(refresh)
            if rank == ranks - 1 and fault != "none":
                if fault == "install":
                    assert fixture(rank, "fail_install")["fixture"] == "armed"
                    reply = json.loads(rpc(rank, commands[rank]))
                    assert reply["fixture"] == "error" and reply["state"]["terminal"]
                else:
                    processes[
                        rank
                    ].terminate()  # CPU-only fixture process, not a GPU fence
                    processes[rank].join(5)
                    assert not processes[rank].is_alive()
                    try:
                        receive(rank)
                    except (EOFError, BrokenPipeError, OSError):
                        pass
                    else:
                        raise AssertionError(
                            "dead rank did not close its control channel"
                        )
                exchange.cancel_commands("peer failed; no partial resume")
                cannot_resume(refresh)
                assert not exchange.can_decode(4)
                blocked(0, 4)
                assert exchange.coordinator.snapshot()["installed_tokens"] == 0
                break
            reply = rpc(rank, commands[rank])
            accept(rank, reply, "applied")
            blocked(rank, 4)
        if fault == "none":
            for rank, command in exchange.resume_commands(refresh).items():
                assert not exchange.can_decode(4)
                accept(rank, rpc(rank, command), "resumed")
                check_values(rank, 4, (1, 3))
            assert exchange.can_decode(4)
            assert exchange.coordinator.snapshot()["installed_tokens"] == 4
        for rank, process in processes.items():
            if process.is_alive():
                reply = fixture(rank, "stop")
                assert reply["fixture"] == "closed", reply
                assert reply["budget"]["used_staging_bytes"] == 0
                assert reply["budget"]["used_inflight"] == 0
                closed[rank] = True
        return {
            "status": "passed",
            "ranks": ranks,
            "fault": fault,
            "independent_processes": len(set(pids.values())),
            "transport": "local multiprocessing pipes, bounded JSON control bytes only",
            "initial_full_prompt_and_refresh_subset": fault == "none",
            "reader_drain_and_global_resume_gate": True,
            "partial_failure_blocks_resume": fault != "none",
            "live_rank_budgets_restored": len(closed),
            "terminated_rank_cleanup_proven": False if fault == "exit" else None,
            "real_model_tp_gpu_rdma_scheduler_validated": False,
        }
    finally:
        for connection in connections.values():
            connection.close()
        for process in processes.values():
            process.join(2)
            if process.is_alive():
                process.terminate()
                process.join(5)


def run_runtime(ranks=2, fault="none"):
    """Drive the real participants through the owner-polled runtime adapter.

    Deadlines use an injected monotonic clock, never wall-clock timing claims.
    Runtime callbacks enqueue locally; this fixture pumps actual pipe I/O later.
    """
    if (
        type(ranks) is not int
        or not 2 <= ranks <= 8
        or fault
        not in (
            "none",
            "install",
            "exit",
            "lost-resume",
            "lost-prepared",
            "forward-cancel",
        )
    ):
        raise ValueError("runtime fixture requires 2..8 ranks and a known fault")
    _bootstrap()
    from sglang.srt.disaggregation.pvd.rank_install_runtime import RankInstallRuntime
    from sglang.srt.disaggregation.pvd.rank_install_wire import (
        RankInstallExchange,
        RankInstallMessage,
    )
    from sglang.srt.disaggregation.pvd.sparse_install import RankInstallCoordinator

    context = multiprocessing.get_context("spawn")
    processes, connections, pids = {}, {}, set()
    epochs = {rank: uuid.uuid4().hex for rank in range(ranks)}
    outgoing, stops, now = [], [], [0.0]
    exchange = RankInstallExchange(
        RankInstallCoordinator(
            "request",
            "request-inc",
            "entry",
            rank_layouts={r: f"layout-{r}" for r in epochs},
            interval=4,
            lead_tokens=1,
        ),
        peer_epochs=epochs,
    )
    runtime = RankInstallRuntime(
        exchange,
        send=lambda rank, raw: outgoing.append((rank, raw)),
        stop_peer=lambda rank, identity, reason: stops.append((rank, identity, reason)),
        max_pending_events=64,
        max_pending_bytes=65536,
        clock=lambda: now[0],
    )

    def receive(rank):
        if not connections[rank].poll(30):
            raise TimeoutError("rank fixture transport did not reply")
        return connections[rank].recv_bytes(FRAME_LIMIT)

    def rpc(rank, value):
        connections[rank].send_bytes(value if type(value) is bytes else _json(value))
        return receive(rank)

    def post(rank, raw):
        assert runtime.post(raw, peer_rank=rank, peer_epoch=epochs[rank])

    def pump():
        for _ in range(8):
            runtime.progress()
            if not outgoing:
                return
            batch, outgoing[:] = tuple(outgoing), []
            for rank, raw in batch:
                message = RankInstallMessage.decode(raw)
                failing = rank == ranks - 1 and message.receipt.epoch.target_tokens == 4
                if failing and fault == "exit" and message.kind == "install":
                    processes[rank].terminate()
                    processes[rank].join(5)
                    assert not processes[rank].is_alive()
                    assert runtime.peer_lost(peer_rank=rank, peer_epoch=epochs[rank])
                    continue
                if failing and fault == "install" and message.kind == "install":
                    assert (
                        json.loads(rpc(rank, {"fixture": "fail_install"}))["fixture"]
                        == "armed"
                    )
                reply = rpc(rank, raw)
                if failing and fault == "lost-resume" and message.kind == "resume":
                    assert RankInstallMessage.decode(reply).kind == "resumed"
                    continue  # peer reopened, but the ACK is deliberately lost
                if failing and fault == "install" and message.kind == "install":
                    error = json.loads(reply)
                    assert error["fixture"] == "error" and error["state"]["terminal"]
                    reply = RankInstallMessage(
                        "failed",
                        epochs[rank],
                        message.receipt,
                        reason="peer install failed",
                    ).encode()
                post(rank, reply)
        raise AssertionError("fixture progress failed to quiesce")

    try:
        for rank in range(ranks):
            parent, child = context.Pipe(duplex=True)
            process = context.Process(target=_worker, args=(child, rank, epochs[rank]))
            process.start()
            child.close()
            processes[rank], connections[rank] = process, parent
        for rank in range(ranks):
            ready = json.loads(receive(rank))
            assert ready["peer_epoch"] == epochs[rank] and ready["pid"] != os.getpid()
            pids.add(ready["pid"])
        assert len(pids) == ranks
        for count in (0, 3):
            epoch = runtime.begin(count, timeout_seconds=5)
            permit = runtime.begin_forward(3) if count == 3 else None
            if permit is not None:
                for rank in range(ranks):
                    assert (
                        json.loads(rpc(rank, {"fixture": "hold", "count": 3}))[
                            "fixture"
                        ]
                        == "held"
                    )
            for rank in range(ranks):
                reply = rpc(rank, {"fixture": "stage", "epoch": asdict(epoch)})
                dropped = count == 3 and rank == ranks - 1 and fault == "lost-prepared"
                if not dropped:
                    post(rank, reply)
                    if permit is not None:
                        assert (
                            json.loads(rpc(rank, {"fixture": "park", "count": 4}))[
                                "fixture"
                            ]
                            == "wait_reader"
                        )
                        assert (
                            json.loads(rpc(rank, {"fixture": "release"}))["fixture"]
                            == "reader_released"
                        )
                    post(
                        rank,
                        rpc(rank, {"fixture": "park", "count": epoch.target_tokens}),
                    )
                elif permit is not None:
                    assert (
                        json.loads(rpc(rank, {"fixture": "release"}))["fixture"]
                        == "reader_released"
                    )
            if permit is not None:
                pump()
                assert runtime.snapshot()["phase"] == "preparing"
                assert runtime.snapshot()["forward_operation_id"] == permit.operation_id
                assert (
                    not outgoing
                )  # even drained rank readers don't retire host execution
                if fault == "forward-cancel":
                    runtime.cancel()
                    assert (
                        runtime.snapshot()["forward_operation_id"]
                        == permit.operation_id
                    )
                accepted = runtime.finish_forward(
                    permit, readers_drained=True, succeeded=True
                )
                assert accepted == (fault != "forward-cancel")
                assert runtime.snapshot()["forward_operation_id"] is None
            pump()
            if count == 3 and fault != "none":
                assert not runtime.can_decode(4)
                if fault in ("lost-resume", "lost-prepared"):
                    assert runtime.snapshot()["phase"] != "failed"
                    now[0] = 5.0  # explicit deadline injection, no timing measurement
                    runtime.progress()
                assert runtime.snapshot()["phase"] == "failed"
                assert {rank for rank, _, _ in stops} == set(epochs)
                assert all(
                    identity == ("request", "request-inc", "entry")
                    for _, identity, _ in stops
                )
                break
            assert runtime.can_decode(epoch.target_tokens)
            for rank in range(ranks):
                result = json.loads(
                    rpc(rank, {"fixture": "read", "count": epoch.target_tokens})
                )
                assert result["tokens"] == ([0, 1, 2, 3] if count == 0 else [1, 3])
        closed = 0
        for rank, process in processes.items():
            if process.is_alive():
                result = json.loads(rpc(rank, {"fixture": "stop"}))
                assert result["fixture"] == "closed", result
                assert result["budget"]["used_staging_bytes"] == 0
                closed += 1
        return {
            "status": "passed",
            "runtime": True,
            "owned_forward_before_refresh": True,
            "cancelled_forward_discarded": fault == "forward-cancel",
            "ranks": ranks,
            "fault": fault,
            "independent_processes": len(pids),
            "live_rank_budgets_restored": closed,
            "all_bound_peers_notified_on_failure": fault != "none",
            "deadline_clock": "injected monotonic, no latency claim",
            "resource_cleanup_proven_by_control": runtime.snapshot()[
                "resource_cleanup_proven"
            ],
            "real_model_tp_gpu_rdma_scheduler_validated": False,
        }
    finally:
        for connection in connections.values():
            connection.close()
        for process in processes.values():
            process.join(2)
            if process.is_alive():
                process.terminate()
                process.join(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument(
        "--fault",
        choices=(
            "none",
            "install",
            "exit",
            "lost-resume",
            "lost-prepared",
            "forward-cancel",
        ),
        default="none",
    )
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    execute = run_runtime if args.runtime else run
    print(json.dumps(execute(args.ranks, args.fault), indent=2))
