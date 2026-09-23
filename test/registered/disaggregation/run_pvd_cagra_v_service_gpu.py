"""Bounded native V-service gate: cuVS-first launcher, Mooncake, two GPU ranks.

No Entry is created. This proves startup/preflight and the shared index budget,
not retrieval quality, sparse transfer, or production D admission.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


def _get_json(url):
    with urllib.request.urlopen(url, timeout=2) as response:
        if response.status != 200:
            raise RuntimeError(f"health endpoint returned HTTP {response.status}")
        return json.load(response)


def _port_is_free(port):
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--advertise-host", required=True)
    parser.add_argument("--rail", required=True)
    parser.add_argument("--coordinator-port", type=int, required=True)
    parser.add_argument("--shard-port-base", type=int, required=True)
    parser.add_argument("--expected-mooncake-version", default="0.3.13.post1")
    parser.add_argument("--timeout-seconds", type=int, default=45)
    args = parser.parse_args(argv)
    ports = (
        args.coordinator_port,
        args.shard_port_base,
        args.shard_port_base + 1,
    )
    if (
        not 0 < args.timeout_seconds <= 120
        or len(set(ports)) != 3
        or any(not 0 < port < 65536 or not _port_is_free(port) for port in ports)
    ):
        raise ValueError("bounded timeout and three unused, distinct ports required")

    root_cap, graph_cap, index_budget = 640 << 20, 512 << 20, 768 << 20
    command = [
        sys.executable, "-m", "pvd_cagra_server",
        "--world-size", "2", "--host", "127.0.0.1",
        "--advertise-host", args.advertise_host,
        "--coordinator-port", str(args.coordinator_port),
        "--shard-port-base", str(args.shard_port_base),
        "--pvd-rank-devices", "0,1",
        "--pvd-rank-rails", f"{args.rail},{args.rail}",
        "--transfer-backend", "mooncake", "--strict-rdma-preflight",
        "--total-pages", "64", "--page-bytes", "458752",
        "--transfer-staging-budget-bytes", str(64 << 20),
        "--transfer-max-inflight", "4",
        "--prompt-index-vector-space", "synthetic-target",
        "--prompt-index-budget-bytes", str(index_budget),
        "--prompt-index-backend", "cagra-auto",
        "--prompt-index-cagra-native-bytes", str(graph_cap),
        "--prompt-index-cagra-global-native-bytes", str(root_cap),
        "--prompt-index-cagra-graph-degree", "32",
        "--prompt-index-cagra-intermediate-degree", "64",
        "--prompt-index-cagra-itopk-size", "64",
    ]
    env = dict(os.environ)
    env.update(
        MOONCAKE_PROTOCOL="rdma",
        MC_DISABLE_METACACHE="1",
        SGLANG_HOST_IP=args.advertise_host,
        CUDA_VISIBLE_DEVICES="0,1",
    )
    report = {
        "schema": "pvd-cagra-v-service-gpu-v1",
        "status": "failed",
        "launcher": "pvd_cagra_server",
        "rail": args.rail,
        "root_cap_bytes_per_rank": root_cap,
    }
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + args.timeout_seconds
            last_error = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"V service exited early: {process.returncode}")
                try:
                    coordinator = _get_json(
                        f"http://127.0.0.1:{args.coordinator_port}/health"
                    )
                    snapshots = [
                        _get_json(f"http://127.0.0.1:{args.shard_port_base + rank}"
                                  "/internal/v1/indexes")
                        for rank in (0, 1)
                    ]
                    if coordinator.get("healthy") and len(coordinator["shards"]) == 2:
                        break
                except (OSError, ValueError, KeyError) as exc:
                    last_error = str(exc)
                time.sleep(0.25)
            else:
                raise RuntimeError(f"V service did not become healthy: {last_error}")

            for rank, (shard, index) in enumerate(
                zip(coordinator["shards"], snapshots)
            ):
                transport = shard["transport"]
                preflight = shard["preflight"]
                budget = index["budget"]
                if not (
                    shard["rank"] == rank
                    and shard["device"] == f"cuda:{rank}"
                    and shard["rail"] == args.rail
                    and preflight["rail_mode"] == "single-rail-debug"
                    and preflight["gpu_memory_registered"]
                    and preflight["local_gpu_transfer"]
                    and transport["backend"] == "mooncake"
                    and transport["healthy"]
                    and transport["mooncake_version"]
                    == args.expected_mooncake_version
                    and index["backend"] == "cagra_auto"
                    and index["device"] == f"cuda:{rank}"
                    and index["enabled"]
                    and not index["quarantined"]
                    and budget["used_staging_bytes"] == root_cap
                    and budget["reservations"] == 1
                ):
                    raise AssertionError(f"V rank {rank} failed native service gate")
            report["healthy_ranks"] = [0, 1]
            report["coordinator_healthy"] = True
            report["root_reservations_per_rank"] = [
                index["budget"]["reservations"] for index in snapshots
            ]
            report["status"] = "passed"
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
                report["status"] = "failed"
                report["cleanup_error"] = "service required force kill"
            report["service_exit_code"] = process.returncode
            log.seek(0)
            if report["status"] != "passed":
                report["service_log_tail"] = log.read()[-4000:]
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
