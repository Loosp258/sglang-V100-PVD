"""One-shot P -> V -> D Mooncake GPU-buffer relay acceptance, not serving.

Start D, then V, then P in separate processes on their respective nodes. The
V process receives into one registered GPU allocation and uses that *same*
registration as the source of the second native WRITE. TCP carries only
descriptors and terminal acknowledgements; payload bytes never pass over TCP.

Unknown native completion retains its GPU memory registration until process
exit. This small sample does not prove production PVD, GPUDirect zero-copy
performance, concurrent requests, or a complete KV lifecycle.
"""

import argparse
import ipaddress
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path


def _send(stream, value):
    stream.write(json.dumps(value, separators=(",", ":")) + "\n")
    stream.flush()


def _read(stream):
    line = stream.readline()
    if not line:
        raise ConnectionError("control peer closed before terminal acknowledgement")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise TypeError("control message must be an object")
    return value


def _require_peer(connection, expected):
    observed = connection.getpeername()[0]
    if observed != expected:
        raise ConnectionError(f"control peer {observed} is not {expected}")


def _require_descriptor(payload, descriptor_type, *, endpoint, rail, length):
    remote = descriptor_type.from_dict(payload)
    if (
        remote.endpoint.split(":", 1)[0] != endpoint
        or remote.rail != rail
        or remote.length != length
    ):
        raise ValueError("remote descriptor endpoint, rail or length mismatch")
    return remote


def _terminal(engine, handle, statuses, timeout):
    deadline = time.monotonic() + timeout
    while True:
        status = engine.poll(handle)
        if status in (statuses.SUCCESS, statuses.FAILED):
            if not handle.transport_state.is_locally_safe_to_release:
                raise RuntimeError("native status lacks safe-to-release proof")
            return status
        if status != statuses.PENDING or time.monotonic() >= deadline:
            raise RuntimeError("native WRITE completion is unknown; retain GPU MR")
        time.sleep(0.01)


def _require_health(engine):
    health = engine.health()
    if (
        health.get("healthy") is not True
        or health.get("registered_regions") != 0
        or health["lifecycle"]["tracked_transfers"] != 0
    ):
        raise RuntimeError("native relay left an unhealthy or live owner")
    return health


def _pattern(torch, device, length):
    return torch.arange(length, dtype=torch.int32, device=device).remainder(251).to(
        torch.uint8
    )


def _listener(host, port, timeout):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(1)
    listener.settimeout(timeout)
    return listener


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("P", "V", "D"))
    parser.add_argument("--p-ip", required=True)
    parser.add_argument("--v-ip", required=True)
    parser.add_argument("--d-ip", required=True)
    parser.add_argument("--p-to-v-port", type=int, default=28175)
    parser.add_argument("--v-to-d-port", type=int, default=28176)
    parser.add_argument("--rail", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    addresses = (args.p_ip, args.v_ip, args.d_ip)
    try:
        parsed = tuple(ipaddress.ip_address(value) for value in addresses)
    except ValueError as exc:
        parser.error(str(exc))
    if any(value.version != 4 or not value.is_private for value in parsed):
        parser.error("all P/V/D addresses must be private IPv4 addresses")
    if len(set(parsed)) != 3:
        parser.error("P, V and D must have distinct private addresses")
    if (
        not 1 <= args.p_to_v_port <= 65535
        or not 1 <= args.v_to_d_port <= 65535
        or args.p_to_v_port == args.v_to_d_port
    ):
        parser.error("control ports must be distinct and within 1..65535")
    if not 1 <= args.length <= (1 << 20):
        parser.error("length must be within 1..1048576 bytes")
    if not 0 < args.timeout_seconds <= 120:
        parser.error("timeout must be within (0, 120] seconds")
    if args.gpu_id < 0:
        parser.error("gpu-id must be non-negative")
    return args


def _run_p(args, torch, engine, descriptor_type, memory_slice, statuses):
    device = f"cuda:{args.gpu_id}"
    source = _pattern(torch, device, args.length)
    registration = engine.register_memory(
        source, endpoint="relay-p-source", rank=args.gpu_id, rail=args.rail
    )
    terminal = False
    try:
        with socket.create_connection(
            (args.v_ip, args.p_to_v_port), timeout=args.timeout_seconds
        ) as connection:
            _require_peer(connection, args.v_ip)
            connection.settimeout(args.timeout_seconds)
            with connection.makefile("rw", encoding="utf-8") as stream:
                remote = _require_descriptor(
                    _read(stream), descriptor_type,
                    endpoint=args.v_ip, rail=args.rail, length=args.length,
                )
                handle = engine.submit_put(
                    memory_slice(registration, 0, args.length), remote
                )
                status = _terminal(engine, handle, statuses, args.timeout_seconds)
                terminal = True
                _send(stream, {
                    "terminal": True,
                    "success": status == statuses.SUCCESS,
                    "reason": handle.error,
                })
                if status != statuses.SUCCESS:
                    raise RuntimeError(f"P -> V WRITE failed: {handle.error}")
                if _read(stream).get("bytes_equal") is not True:
                    raise RuntimeError("V did not verify the received GPU bytes")
    finally:
        if terminal:
            engine.release_memory(registration)


def _run_d(args, torch, engine):
    device = f"cuda:{args.gpu_id}"
    destination = torch.zeros(args.length, dtype=torch.uint8, device=device)
    registration = engine.register_memory(
        destination, endpoint="relay-d-destination", rank=args.gpu_id,
        rail=args.rail,
    )
    terminal = False
    try:
        with _listener(args.d_ip, args.v_to_d_port, args.timeout_seconds) as listener:
            print(json.dumps({"event": "ready", "mode": "D"}), flush=True)
            connection, _ = listener.accept()
            with connection:
                _require_peer(connection, args.v_ip)
                connection.settimeout(args.timeout_seconds)
                with connection.makefile("rw", encoding="utf-8") as stream:
                    _send(stream, registration.descriptor.to_dict())
                    outcome = _read(stream)
                    terminal = outcome.get("terminal") is True
                    if not terminal or outcome.get("success") is not True:
                        raise RuntimeError(
                            "V has not proved a successful terminal WRITE"
                        )
                    torch.cuda.synchronize(device)
                    equal = bool(torch.equal(
                        destination, _pattern(torch, device, args.length)
                    ))
                    _send(stream, {"bytes_equal": equal})
                    if not equal:
                        raise RuntimeError("D GPU bytes differ from P source")
    finally:
        if terminal:
            engine.release_memory(registration)


def _run_v(args, torch, engine, descriptor_type, memory_slice, statuses):
    device = f"cuda:{args.gpu_id}"
    intermediate = torch.zeros(args.length, dtype=torch.uint8, device=device)
    registration = engine.register_memory(
        intermediate, endpoint="relay-v-intermediate", rank=args.gpu_id,
        rail=args.rail,
    )
    first_terminal = False
    second_started = False
    second_terminal = False
    try:
        with socket.create_connection(
            (args.d_ip, args.v_to_d_port), timeout=args.timeout_seconds
        ) as downstream:
            _require_peer(downstream, args.d_ip)
            downstream.settimeout(args.timeout_seconds)
            with downstream.makefile("rw", encoding="utf-8") as d_stream:
                remote_d = _require_descriptor(
                    _read(d_stream), descriptor_type,
                    endpoint=args.d_ip, rail=args.rail, length=args.length,
                )
                with _listener(
                    args.v_ip, args.p_to_v_port, args.timeout_seconds
                ) as listener:
                    print(json.dumps({"event": "ready", "mode": "V"}), flush=True)
                    upstream, _ = listener.accept()
                    with upstream:
                        _require_peer(upstream, args.p_ip)
                        upstream.settimeout(args.timeout_seconds)
                        with upstream.makefile("rw", encoding="utf-8") as p_stream:
                            _send(p_stream, registration.descriptor.to_dict())
                            outcome = _read(p_stream)
                            first_terminal = outcome.get("terminal") is True
                            if not first_terminal or outcome.get("success") is not True:
                                raise RuntimeError(
                                    "P has not proved a successful terminal WRITE"
                                )
                            torch.cuda.synchronize(device)
                            equal = bool(torch.equal(
                                intermediate, _pattern(torch, device, args.length)
                            ))
                            _send(p_stream, {"bytes_equal": equal})
                            if not equal:
                                raise RuntimeError("V GPU bytes differ from P source")
                # The registered V destination is now the V source. No host
                # payload copy or second GPU allocation is used for the relay.
                second_started = True
                handle = engine.submit_put(
                    memory_slice(registration, 0, args.length), remote_d
                )
                status = _terminal(engine, handle, statuses, args.timeout_seconds)
                second_terminal = True
                _send(d_stream, {
                    "terminal": True,
                    "success": status == statuses.SUCCESS,
                    "reason": handle.error,
                })
                if status != statuses.SUCCESS:
                    raise RuntimeError(f"V -> D WRITE failed: {handle.error}")
                if _read(d_stream).get("bytes_equal") is not True:
                    raise RuntimeError("D did not verify the relayed GPU bytes")
    finally:
        # A submit that raises may still have left native work in flight. In
        # that case keep the V MR and backing GPU tensor until process exit.
        if first_terminal and (not second_started or second_terminal):
            engine.release_memory(registration)


def main(argv=None):
    args = _arguments(argv)
    hostname = {"P": args.p_ip, "V": args.v_ip, "D": args.d_ip}[args.mode]
    report = {
        "schema": "pvd-native-relay-v1",
        "mode": args.mode,
        "status": "failed",
        "hostname": hostname,
        "rail": args.rail,
        "gpu_id": args.gpu_id,
        "length": args.length,
        "control_transport": "tcp_json",
        "data_transport": "mooncake_native_write",
        "same_request_gpu_buffer_relay": False,
        "production_pvd_validated": False,
        "zero_copy_performance_validated": False,
    }
    try:
        os.environ["MC_DISABLE_METACACHE"] = "1"
        root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(root / "python"))
        import torch
        from sglang.srt.disaggregation.pvd.mooncake_engine import (
            MooncakePVDTransferEngine,
        )
        from sglang.srt.disaggregation.pvd.protocol import RemoteRegionDescriptor
        from sglang.srt.disaggregation.pvd.transfer_engine import (
            MemorySlice, TransferStatus,
        )
        from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

        if args.gpu_id >= torch.cuda.device_count():
            raise ValueError("requested CUDA GPU is unavailable")
        engine = MooncakePVDTransferEngine(
            hostname=hostname, gpu_id=args.gpu_id, rail=args.rail,
            budget=TransferBudget(staging_bytes=1 << 20, max_inflight=2),
        )
        if args.mode == "P":
            _run_p(args, torch, engine, RemoteRegionDescriptor, MemorySlice,
                   TransferStatus)
        elif args.mode == "V":
            _run_v(args, torch, engine, RemoteRegionDescriptor, MemorySlice,
                   TransferStatus)
        else:
            _run_d(args, torch, engine)
        health = _require_health(engine)
        report.update(
            status="passed", same_request_gpu_buffer_relay=True,
            gpu=torch.cuda.get_device_name(args.gpu_id),
            mooncake_version=health["mooncake_version"],
            metadata_policy=health["metadata_policy"],
        )
    except Exception as exc:  # noqa: BLE001 -- failed acceptance is never a skip
        report.update(
            reason=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=12),
        )
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
