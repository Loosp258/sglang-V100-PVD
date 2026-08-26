"""Launcher for a two-shard PVD vector worker group.

V is a KV data service rather than a model-serving endpoint. The default mode
hosts both GPU shards and the coordinator in one Python process. Supplying
``--rank`` keeps the original one-process-per-rank mode available for rolling
upgrades and diagnostics.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
from typing import List, Sequence

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
    parser = argparse.ArgumentParser(description="Launch a PVD V worker group")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help=(
            "legacy mode: launch only this V rank; omit this option to launch "
            "the complete two-GPU V worker group in one process"
        ),
    )
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
        "--devices",
        "--pvd-rank-devices",
        dest="devices",
        default="0,1",
        help=(
            "process-local CUDA device id for each V shard in group mode; "
            "default: 0,1"
        ),
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
        raise ValueError("PVD requires exactly 2 V storage ranks")
    if args.rank is not None and args.rank not in (0, 1):
        raise ValueError("PVD V rank must be 0 or 1")
    rails = [value.strip() for value in args.rails.split(",") if value.strip()]
    if len(rails) != args.world_size:
        raise ValueError(
            "V requires exactly one --pvd-rank-rails value per storage rank; "
            f"got {len(rails)} values for world size {args.world_size}"
        )
    rail_mode = validate_rank_rail_names(rails)
    if rail_mode == "single-rail-debug":
        logger.warning(
            "PVD single-rail debug mode is active: both V ranks use mlx5_0; "
            "there is no rail redundancy or aggregate dual-rail bandwidth"
        )
    if args.rank == 0 and not args.rank1_shard_url:
        raise ValueError("rank 0 requires --rank1-shard-url")
    if args.rank is None:
        _parse_device_ids(args.devices, args.world_size)
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


def _parse_device_ids(value: str, world_size: int) -> List[int]:
    try:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--pvd-rank-devices must contain CUDA device integers") from exc
    if len(devices) != world_size:
        raise ValueError(
            f"--pvd-rank-devices requires {world_size} entries, got {devices}"
        )
    if any(device < 0 for device in devices):
        raise ValueError("--pvd-rank-devices cannot contain negative device ids")
    if len(set(devices)) != len(devices):
        raise ValueError("each V shard requires a distinct CUDA device")
    return devices


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


async def _group_reaper(
    stores: Sequence[VectorKVStore],
    interval: float,
    coordinator: VectorCoordinator,
) -> None:
    """Reap all local shards, then update the group-level lifecycle once."""
    while True:
        await asyncio.sleep(interval)
        for store in stores:
            store.reap_expired(reap_entries=False)
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


def _create_store(
    args: argparse.Namespace,
    *,
    rank: int,
    local_rank: int,
    rails: Sequence[str],
) -> tuple[VectorKVStore, dict]:
    """Create one logical V shard on an explicitly selected local device."""
    device = "cpu" if args.allow_cpu_for_tests else f"cuda:{local_rank}"
    shard_port = args.shard_port_base + rank
    endpoint = f"{args.advertise_host}:{shard_port}"
    if args.transfer_backend == "mooncake":
        engine = MooncakePVDTransferEngine(
            hostname=args.advertise_host,
            gpu_id=local_rank,
            rail=rails[rank],
        )
        preflight = run_rank_preflight(
            rank=rank,
            rails=rails,
            device=device,
            engine=engine,
            strict=args.strict_rdma_preflight,
        ).to_dict()
    else:
        engine = FakeTransferEngine()
        preflight = {
            "rank": rank,
            "rail": rails[rank],
            "skipped": True,
            "reason": "fake transport",
        }
    store = VectorKVStore(
        rank=rank,
        world_size=args.world_size,
        rail=rails[rank],
        device=device,
        total_pages=args.total_pages,
        page_bytes=args.page_bytes,
        endpoint=endpoint,
        transfer_engine=engine,
        entry_ttl_secs=args.entry_ttl_secs,
        delivery_timeout_secs=args.delivery_timeout_secs,
        allow_cpu_for_tests=args.allow_cpu_for_tests,
    )
    return store, preflight


def _install_signal_handlers(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for signame in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, signame, None)
        if sig is not None:
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, stop.set)


async def _serve_rank(args: argparse.Namespace) -> None:
    """Run the legacy one-process-per-rank deployment."""
    rails = _validate_args(args)
    assert args.rank is not None
    shard_port = args.shard_port_base + args.rank
    store, preflight = _create_store(
        args,
        rank=args.rank,
        local_rank=args.local_rank,
        rails=rails,
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
    _install_signal_handlers(stop)

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


async def _serve_group(args: argparse.Namespace) -> None:
    """Run both V GPU shards and the coordinator in one Python process."""
    rails = _validate_args(args)
    device_ids = _parse_device_ids(args.devices, args.world_size)
    stores: List[VectorKVStore] = []
    preflights: List[dict] = []
    runners: List[web.AppRunner] = []
    reaper_task = None
    try:
        for rank, local_rank in enumerate(device_ids):
            store, preflight = _create_store(
                args,
                rank=rank,
                local_rank=local_rank,
                rails=rails,
            )
            stores.append(store)
            preflights.append(preflight)

        for rank, store in enumerate(stores):
            runner = web.AppRunner(
                create_shard_app(store, preflight=preflights[rank])
            )
            await runner.setup()
            await web.TCPSite(
                runner, args.host, args.shard_port_base + rank
            ).start()
            runners.append(runner)

        coordinator = VectorCoordinator(
            [
                LocalShardClient(store, preflight=preflights[rank])
                for rank, store in enumerate(stores)
            ],
            entry_ttl_secs=args.entry_ttl_secs,
            delivery_timeout_secs=args.delivery_timeout_secs,
        )
        coordinator_runner = web.AppRunner(create_coordinator_app(coordinator))
        await coordinator_runner.setup()
        await web.TCPSite(
            coordinator_runner, args.host, args.coordinator_port
        ).start()
        runners.append(coordinator_runner)

        logger.info(
            "PVD V worker group is ready in one process: devices=%s rails=%s "
            "coordinator=%s:%s shard_ports=%s",
            device_ids,
            list(rails),
            args.advertise_host,
            args.coordinator_port,
            [args.shard_port_base + rank for rank in range(args.world_size)],
        )

        stop = asyncio.Event()
        _install_signal_handlers(stop)
        reaper_task = asyncio.create_task(
            _group_reaper(stores, args.reaper_interval_secs, coordinator),
            name="pvd-v-group-reaper",
        )
        await stop.wait()
    finally:
        if reaper_task is not None:
            reaper_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reaper_task
        for runner in reversed(runners):
            await runner.cleanup()
        for store in reversed(stores):
            store.close()


def main() -> None:
    args = build_parser().parse_args()
    if args.rank is None:
        asyncio.run(_serve_group(args))
    else:
        asyncio.run(_serve_rank(args))


if __name__ == "__main__":
    main()
