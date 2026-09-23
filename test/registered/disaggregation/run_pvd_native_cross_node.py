"""One-shot cross-node PVD Mooncake GPU WRITE acceptance, not serving.

Run ``receive`` first on the destination and ``send`` on the source. TCP is
only the descriptor/completion control plane; KV-like bytes use the native
Mooncake WRITE. Use isolated processes and a private network address.
"""

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path


def _send_json(stream, value):
    stream.write(json.dumps(value, separators=(",", ":")) + "\n")
    stream.flush()


def _read_json(stream):
    line = stream.readline()
    if not line:
        raise ConnectionError("control peer closed before terminal acknowledgement")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise TypeError("control message must be an object")
    return value


def _pattern(torch, device, length):
    return (
        torch.arange(length, dtype=torch.int32, device=device)
        .remainder(251)
        .to(torch.uint8)
    )


def _wait_terminal(engine, handle, status_type, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while True:
        status = engine.poll(handle)
        if status in (status_type.SUCCESS, status_type.FAILED):
            # A FAILED result can also represent an untrackable native submit
            # or poll. It is not, by itself, safe-to-unregister proof.
            if not handle.transport_state.is_locally_safe_to_release:
                raise RuntimeError(
                    "native WRITE status lacks terminal or not-submitted proof; "
                    "retain source MR"
                )
            return status
        if status != status_type.PENDING or time.monotonic() >= deadline:
            raise RuntimeError("native WRITE completion is unknown; retain source MR")
        time.sleep(0.01)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("receive", "send"))
    parser.add_argument("--hostname", required=True, help="this node's private RDMA IP")
    parser.add_argument("--peer", required=True, help="other node's private IP")
    parser.add_argument("--port", type=int, default=28173)
    parser.add_argument("--rail", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    args = parser.parse_args(argv)
    if not (1 <= args.port <= 65535 and 1 <= args.length <= 1 << 20):
        parser.error("port or length is outside the bounded validation range")
    if not (0 < args.timeout_seconds <= 120):
        parser.error("timeout must be in (0, 120] seconds")

    report = {
        "schema": "pvd-native-cross-node-v1",
        "mode": args.mode,
        "status": "failed",
        "hostname": args.hostname,
        "peer": args.peer,
        "rail": args.rail,
        "gpu_id": args.gpu_id,
        "control_transport": "tcp_json",
        "data_transport": "mooncake_native_write",
        "cross_node_validated": False,
        "production_pvd_validated": False,
    }
    try:
        if args.hostname == args.peer:
            raise ValueError("cross-node validation requires distinct host addresses")
        os.environ["MC_DISABLE_METACACHE"] = "1"
        root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(root / "python"))
        import torch
        from sglang.srt.disaggregation.pvd.mooncake_engine import (
            MooncakePVDTransferEngine,
        )
        from sglang.srt.disaggregation.pvd.protocol import RemoteRegionDescriptor
        from sglang.srt.disaggregation.pvd.transfer_engine import (
            MemorySlice,
            TransferStatus,
        )
        from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

        if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
            raise ValueError("requested CUDA GPU is unavailable")
        device = f"cuda:{args.gpu_id}"
        engine = MooncakePVDTransferEngine(
            hostname=args.hostname,
            gpu_id=args.gpu_id,
            rail=args.rail,
            budget=TransferBudget(staging_bytes=1 << 20, max_inflight=2),
        )
        if args.mode == "receive":
            destination = torch.zeros(args.length, dtype=torch.uint8, device=device)
            registration = engine.register_memory(
                destination, endpoint="cross-node-destination", rank=0, rail=args.rail
            )
            terminal = False
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    listener.bind((args.hostname, args.port))
                    listener.listen(1)
                    listener.settimeout(args.timeout_seconds)
                    print(json.dumps({"event": "ready", "port": args.port}), flush=True)
                    connection, address = listener.accept()
                    with connection:
                        if address[0] != args.peer:
                            raise ConnectionError(
                                "control connection came from another node"
                            )
                        connection.settimeout(args.timeout_seconds)
                        with connection.makefile("rw", encoding="utf-8") as stream:
                            _send_json(stream, registration.descriptor.to_dict())
                            outcome = _read_json(stream)
                            terminal = outcome.get("terminal") is True
                            if not terminal:
                                raise RuntimeError(
                                    "sender has not proved a terminal native status"
                                )
                            if outcome.get("success") is not True:
                                raise RuntimeError(
                                    f"sender's native WRITE failed: {outcome.get('reason')}"
                                )
                            torch.cuda.synchronize(device)
                            equal = bool(
                                torch.equal(
                                    destination, _pattern(torch, device, args.length)
                                )
                            )
                            _send_json(stream, {"bytes_equal": equal})
                            if not equal:
                                raise RuntimeError(
                                    "remote GPU bytes differ from source"
                                )
                report["bytes_equal"] = True
            finally:
                # On unknown status, the destination MR must not be released
                # while a remote native WRITE could still be in flight.
                if terminal:
                    engine.release_memory(registration)
        else:
            source = _pattern(torch, device, args.length)
            registration = engine.register_memory(
                source, endpoint="cross-node-source", rank=0, rail=args.rail
            )
            terminal = False
            try:
                with socket.create_connection(
                    (args.peer, args.port), timeout=args.timeout_seconds
                ) as connection:
                    connection.settimeout(args.timeout_seconds)
                    with connection.makefile("rw", encoding="utf-8") as stream:
                        remote = RemoteRegionDescriptor.from_dict(_read_json(stream))
                        if remote.length != args.length or remote.rail != args.rail:
                            raise ValueError("remote descriptor length/rail mismatch")
                        if remote.endpoint.split(":", 1)[0] != args.peer:
                            raise ValueError("remote Mooncake endpoint is not the peer")
                        handle = engine.submit_put(
                            MemorySlice(registration, 0, args.length), remote
                        )
                        status = _wait_terminal(
                            engine, handle, TransferStatus, args.timeout_seconds
                        )
                        terminal = True
                        _send_json(
                            stream,
                            {
                                "terminal": True,
                                "success": status == TransferStatus.SUCCESS,
                                "reason": handle.error,
                            },
                        )
                        if status != TransferStatus.SUCCESS:
                            raise RuntimeError(f"native WRITE failed: {handle.error}")
                        confirmation = _read_json(stream)
                        if confirmation.get("bytes_equal") is not True:
                            raise RuntimeError(
                                "receiver did not verify remote GPU bytes"
                            )
                report["bytes_equal"] = True
                report["remote_endpoint"] = remote.endpoint
            finally:
                if terminal:
                    engine.release_memory(registration)
        health = engine.health()
        if (
            health.get("healthy") is not True
            or health.get("registered_regions") != 0
            or health["lifecycle"]["tracked_transfers"] != 0
        ):
            raise RuntimeError("cross-node validation left an unhealthy or live owner")
        report.update(
            status="passed",
            cross_node_validated=True,
            mooncake_version=health["mooncake_version"],
            metadata_policy=health["metadata_policy"],
            gpu=torch.cuda.get_device_name(args.gpu_id),
        )
    except Exception as exc:  # noqa: BLE001 -- failed native validation is never a skip
        report.update(
            reason=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=12),
        )
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
