"""Launcher for a two-rank PVD vector worker group.

This launcher is intentionally separate from ``launch_server``: V is a KV data
service, not a model-serving endpoint.  Rank 0 exposes both its private shard API
and the group coordinator API; rank 1 exposes only its private shard API.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
from typing import List

from aiohttp import web

from sglang.srt.disaggregation.pvd.control_server import (
    HttpShardClient,
    create_coordinator_app,
    create_shard_app,
)
from sglang.srt.disaggregation.pvd.coordinator import (
    LocalShardClient,
    VectorCoordinator,
)
from sglang.srt.disaggregation.pvd.mooncake_engine import MooncakePVDTransferEngine
from sglang.srt.disaggregation.pvd.preflight import (
    run_rank_preflight,
    validate_rank_rail_names,
)
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine
from sglang.srt.disaggregation.pvd.vector_store import VectorKVStore


logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch one PVD V rank")
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", "0")))
    parser.add_argument(
        "--local-rank", type=int, default=int(os.getenv("LOCAL_RANK", "0"))
    )
    parser.add_argument(
        "--world-size", type=int, default=int(os.getenv("WORLD_SIZE", "2"))
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--advertise-host", required=True)
    parser.add_argument("--coordinator-port", type=int, default=9100)
    parser.add_argument("--shard-port-base", type=int, default=9200)
    parser.add_argument(
        "--rank1-shard-url",
        help="rank-1 private URL as seen by rank 0, e.g. http://10.0.0.2:9201",
    )
    parser.add_argument(
        "--rails",
        "--pvd-rank-rails",
        dest="rails",
        default="mlx5_0,mlx5_1",
        help=(
            "rank-to-rail mapping: mlx5_0,mlx5_1 for production or "
            "mlx5_0,mlx5_0 for single-rail debug mode"
        ),
    )
    parser.add_argument(
        "--transfer-backend", choices=["mooncake", "fake"], default="mooncake"
    )
    parser.add_argument(
        "--strict-rdma-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--total-pages", type=_positive_int, required=True)
    parser.add_argument("--page-bytes", type=_positive_int, required=True)
    parser.add_argument("--entry-ttl-secs", type=float, default=300.0)
    parser.add_argument("--delivery-timeout-secs", type=float, default=300.0)
    parser.add_argument("--rank1-startup-timeout-secs", type=float, default=300.0)
    parser.add_argument("--reaper-interval-secs", type=float, default=1.0)
    parser.add_argument(
        "--allow-fake-transport",
        action="store_true",
        help="development only; fake transport is process-local and is not RDMA",
    )
    parser.add_argument(
        "--allow-cpu-for-tests",
        action="store_true",
        help="test-only override; production V storage must be CUDA",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> List[str]:
    if args.world_size != 2:
        raise ValueError("PVD v1 requires exactly 2 V ranks")
    if args.rank not in (0, 1):
        raise ValueError("PVD V rank must be 0 or 1")
    rails = [value.strip() for value in args.rails.split(",") if value.strip()]
    rail_mode = validate_rank_rail_names(rails)
    if rail_mode == "single-rail-debug":
        logger.warning(
            "PVD single-rail debug mode is active: both V ranks use mlx5_0; "
            "there is no rail redundancy or aggregate dual-rail bandwidth"
        )
    if args.rank == 0 and not args.rank1_shard_url:
        raise ValueError("rank 0 requires --rank1-shard-url")
    if args.transfer_backend == "fake" and not args.allow_fake_transport:
        raise ValueError(
            "fake transport requires --allow-fake-transport and is test-only"
        )
    if args.transfer_backend == "fake" and args.strict_rdma_preflight:
        raise ValueError("fake transport requires --no-strict-rdma-preflight")
    if args.transfer_backend == "mooncake" and args.allow_cpu_for_tests:
        raise ValueError("Mooncake PVD cannot use --allow-cpu-for-tests")
    if args.reaper_interval_secs <= 0:
        raise ValueError("--reaper-interval-secs must be positive")
    if args.rank1_startup_timeout_secs <= 0:
        raise ValueError("--rank1-startup-timeout-secs must be positive")
    return rails


async def _reaper(
    store: VectorKVStore,
    interval: float,
    coordinator: VectorCoordinator | None = None,
) -> None:
    while True:
        await asyncio.sleep(interval)
        # The coordinator is the authority for Entry TTL so its STORED view can
        # never outlive the two shard allocations. Shards still reap timed-out
        # Delivery resources locally.
        store.reap_expired(reap_entries=False)
        if coordinator is not None:
            await coordinator.reap_expired()


async def _wait_for_rank1(
    client: HttpShardClient, *, timeout: float, strict: bool, expected_rail: str
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    last_error: Exception | None = None
    while asyncio.get_running_loop().time() < deadline:
        try:
            health = await client.health()
            if health.get("rank") != 1 or health.get("world_size") != 2:
                raise RuntimeError(f"unexpected rank-1 health payload: {health}")
            if health.get("rail") != expected_rail:
                raise RuntimeError(
                    f"V rank 1 must be bound to {expected_rail}, "
                    f"got {health.get('rail')}"
                )
            if strict:
                preflight = health.get("preflight", {})
                for field in (
                    "rail_present",
                    "active_port",
                    "gpu_memory_registered",
                    "local_gpu_transfer",
                ):
                    if preflight.get(field) is not True:
                        raise RuntimeError(
                            f"V rank 1 failed strict preflight field {field}"
                        )
            return
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(0.25)
    raise RuntimeError(
        f"V rank 1 did not become healthy within {timeout}s: {last_error}"
    )


async def _serve(args: argparse.Namespace) -> None:
    rails = _validate_args(args)
    device = "cpu" if args.allow_cpu_for_tests else f"cuda:{args.local_rank}"
    shard_port = args.shard_port_base + args.rank
    endpoint = f"{args.advertise_host}:{shard_port}"
    if args.transfer_backend == "mooncake":
        engine = MooncakePVDTransferEngine(
            hostname=args.advertise_host,
            gpu_id=args.local_rank,
            rail=rails[args.rank],
        )
        preflight = run_rank_preflight(
            rank=args.rank,
            rails=rails,
            device=device,
            engine=engine,
            strict=args.strict_rdma_preflight,
        ).to_dict()
    else:
        engine = FakeTransferEngine()
        preflight = {
            "rank": args.rank,
            "rail": rails[args.rank],
            "skipped": True,
            "reason": "fake transport",
        }
    store = VectorKVStore(
        rank=args.rank,
        world_size=args.world_size,
        rail=rails[args.rank],
        device=device,
        total_pages=args.total_pages,
        page_bytes=args.page_bytes,
        endpoint=endpoint,
        transfer_engine=engine,
        entry_ttl_secs=args.entry_ttl_secs,
        delivery_timeout_secs=args.delivery_timeout_secs,
        allow_cpu_for_tests=args.allow_cpu_for_tests,
    )

    runners = []
    remote_client = None
    coordinator = None
    shard_runner = web.AppRunner(create_shard_app(store, preflight=preflight))
    await shard_runner.setup()
    await web.TCPSite(shard_runner, args.host, shard_port).start()
    runners.append(shard_runner)

    if args.rank == 0:
        remote_client = HttpShardClient(1, args.rank1_shard_url)
        await _wait_for_rank1(
            remote_client,
            timeout=args.rank1_startup_timeout_secs,
            strict=args.strict_rdma_preflight,
            expected_rail=rails[1],
        )
        coordinator = VectorCoordinator(
            [LocalShardClient(store, preflight=preflight), remote_client],
            entry_ttl_secs=args.entry_ttl_secs,
            delivery_timeout_secs=args.delivery_timeout_secs,
        )
        coordinator_runner = web.AppRunner(create_coordinator_app(coordinator))
        await coordinator_runner.setup()
        await web.TCPSite(
            coordinator_runner, args.host, args.coordinator_port
        ).start()
        runners.append(coordinator_runner)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signame in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, signame, None)
        if sig is not None:
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, stop.set)

    reaper_task = asyncio.create_task(
        _reaper(store, args.reaper_interval_secs, coordinator),
        name=f"pvd-v{args.rank}-reaper",
    )
    try:
        await stop.wait()
    finally:
        reaper_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reaper_task
        if remote_client is not None:
            await remote_client.close()
        for runner in reversed(runners):
            await runner.cleanup()
        store.close()


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(_serve(args))


if __name__ == "__main__":
    main()
