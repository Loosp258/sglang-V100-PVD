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
import time
from collections.abc import Awaitable, Callable, Sequence
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
from sglang.srt.disaggregation.pvd.preflight import (
    resolve_rank_rails,
    run_rank_preflight,
    validate_rank_rail_names,
)
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine
from sglang.srt.disaggregation.pvd.vector_store import VectorKVStore

logger = logging.getLogger(__name__)


class _MaintenanceReaperHealth:
    """Small event-loop-owned status record for the V maintenance task."""

    def __init__(self) -> None:
        self.rounds = 0
        self.successful_rounds = 0
        self.failed_rounds = 0
        self.consecutive_failures = 0
        self.last_round_started_unix: float | None = None
        self.last_round_completed_unix: float | None = None
        self.last_success_unix: float | None = None
        self.last_failure_unix: float | None = None
        self.last_error: str | None = None

    def record_round(self, errors: Sequence[str]) -> None:
        now = time.time()
        self.rounds += 1
        self.last_round_completed_unix = now
        if errors:
            self.failed_rounds += 1
            self.consecutive_failures += 1
            self.last_failure_unix = now
            self.last_error = "; ".join(errors)[:512]
        else:
            self.successful_rounds += 1
            self.consecutive_failures = 0
            self.last_success_unix = now

    def snapshot(self) -> dict:
        if self.rounds == 0:
            status = "starting"
            healthy = None
        elif self.consecutive_failures:
            status = "degraded"
            healthy = False
        else:
            status = "healthy"
            healthy = True
        return {
            "status": status,
            "healthy": healthy,
            "rounds": self.rounds,
            "successful_rounds": self.successful_rounds,
            "failed_rounds": self.failed_rounds,
            "consecutive_failures": self.consecutive_failures,
            "last_round_started_unix": self.last_round_started_unix,
            "last_round_completed_unix": self.last_round_completed_unix,
            "last_success_unix": self.last_success_unix,
            "last_failure_unix": self.last_failure_unix,
            "last_error": self.last_error,
        }


class _CoordinatorHealthView:
    """Add maintenance-task status to the existing coordinator health API."""

    def __init__(self, coordinator, reaper_health: _MaintenanceReaperHealth):
        self._coordinator = coordinator
        self._reaper_health = reaper_health

    def __getattr__(self, name):
        return getattr(self._coordinator, name)

    async def health(self):
        snapshot = dict(await self._coordinator.health())
        reaper = self._reaper_health.snapshot()
        snapshot["maintenance_reaper"] = reaper
        # A failed maintenance round is surfaced to health checks. Startup is
        # reported separately until the first round completes.
        if reaper["healthy"] is False:
            snapshot["healthy"] = False
        return snapshot


async def _run_reaper_round(
    health: _MaintenanceReaperHealth,
    steps: Sequence[tuple[str, Callable[[], Awaitable[object]]]],
) -> None:
    """Run all maintenance steps, recording failures without killing retries."""
    health.last_round_started_unix = time.time()
    errors = []
    for name, step in steps:
        try:
            await step()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("PVD maintenance reaper step %s failed", name)
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    health.record_round(errors)
    if errors:
        logger.error(
            "PVD maintenance reaper round failed; the next round will retry: %s",
            "; ".join(errors),
        )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a PVD V worker group")
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Use debug to trace PVD MR registration and PUT lifecycles",
    )
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
            "process-local CUDA device id for each V shard in group mode; default: 0,1"
        ),
    )
    parser.add_argument(
        "--rails",
        "--pvd-rank-rails",
        dest="rails",
        default=None,
        help=(
            "One HCA per storage rank, e.g. mlx5_2,mlx5_3. "
            "Defaults to mlx5_0,mlx5_1 if neither HCA flag is supplied."
        ),
    )
    parser.add_argument(
        "--disaggregation-ib-device",
        help=(
            "One HCA shared by all storage ranks, or a comma-separated HCA "
            "per rank, e.g. mlx5_2,mlx5_3. Must agree with --pvd-rank-rails "
            "if both are supplied."
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
    parser.add_argument(
        "--transfer-staging-budget-bytes",
        type=_positive_int,
        required=True,
        help="Byte budget for this V rank's repacking staging allocations. "
        "No default is guessed.",
    )
    parser.add_argument(
        "--transfer-max-inflight",
        type=_positive_int,
        required=True,
        help="Maximum PVD transfers this V rank may have in flight, including "
        "draining and quarantined ones.",
    )
    parser.add_argument("--total-pages", type=_positive_int, required=True)
    parser.add_argument("--page-bytes", type=_positive_int, required=True)
    parser.add_argument(
        "--prompt-index-vector-space",
        type=str,
        default=None,
        help="Enable the retrieval index over stored prompts, naming the model "
        "whose K these vectors come from. A query in another space is refused. "
        "Omitted (the default) means no index is built and V serves exactly as "
        "before; delivery never depends on the index either way.",
    )
    parser.add_argument(
        "--prompt-index-metric",
        type=str,
        default="ip",
        choices=("ip", "l2"),
        help="Similarity used by the retrieval index. Only meaningful with "
        "--prompt-index-vector-space.",
    )
    parser.add_argument(
        "--prompt-index-budget-bytes",
        type=_positive_int,
        default=None,
        help="Byte budget for this V rank's retrieval-index copies: the "
        "extracted Prompt K, whatever the backend retains for it, and the "
        "bounded scratch of a search. Required with "
        "--prompt-index-vector-space, and no default is guessed. Separate "
        "from --transfer-staging-budget-bytes on purpose: index pressure "
        "must never consume the headroom a transfer was admitted against.",
    )
    parser.add_argument("--entry-ttl-secs", type=float, default=300.0)
    parser.add_argument(
        "--max-entry-records",
        type=_positive_int,
        default=8192,
        help="Maximum Entry records retained by each V rank in one worker epoch, "
        "including terminal replay tombstones. At capacity, new Entry keys "
        "are refused without evicting old identities.",
    )
    parser.add_argument(
        "--max-delivery-records",
        type=_positive_int,
        default=65536,
        help="Maximum Delivery records retained by each V rank in one worker "
        "epoch, including terminal tombstones. New Delivery IDs are refused "
        "at capacity; existing ones can still be retried or fenced.",
    )
    parser.add_argument(
        "--max-legacy-absent-fences",
        type=_positive_int,
        default=4096,
        help="Maximum ID-only absent Delivery fences retained by each V rank. "
        "Complete-identity write fences have their separate bound.",
    )
    parser.add_argument(
        "--prompt-index-backend",
        choices=("exact", "cagra", "cagra-auto"),
        default="exact",
    )
    parser.add_argument(
        "--prompt-index-cagra-native-bytes",
        type=_positive_int,
        default=None,
        help="Per layer/head native allocation cap, reserved for its entire index lifetime, including native workspace.",
    )
    parser.add_argument(
        "--prompt-index-cagra-global-native-bytes",
        type=_positive_int,
        default=None,
        help="Optional single native RMM parent cap across all CAGRA graphs on this V rank. Reserved once from --prompt-index-budget-bytes; must cover one per-index cap. Without it, the old per-graph lifetime reservation remains.",
    )
    parser.add_argument(
        "--prompt-index-cagra-graph-degree", type=_positive_int, default=64
    )
    parser.add_argument(
        "--prompt-index-cagra-intermediate-degree", type=_positive_int, default=128
    )
    parser.add_argument(
        "--prompt-index-cagra-itopk-size", type=_positive_int, default=512
    )
    parser.add_argument(
        "--full-kv-fanin-max-slices",
        type=_positive_int,
        default=None,
        help="Opt in to V full-KV fan-in routes with this per-destination plan bound. "
        "Requires --full-kv-fanin-max-inflight; does not enable D predictive serving.",
    )
    parser.add_argument(
        "--full-kv-fanin-max-inflight",
        type=_positive_int,
        default=None,
        help="Maximum outstanding PUTs per fan-in writer. Both fan-in bounds are required.",
    )
    parser.add_argument(
        "--full-kv-fanin-max-records",
        type=_positive_int,
        default=None,
        help="Opt in to global fan-in aggregation; bound retained delivery records "
        "including terminal tombstones. Requires both shard fan-in bounds.",
    )
    parser.add_argument(
        "--experimental-cuda-sparse-packing",
        action="store_true",
        help="Explicit V-only CUDA sparse copy baseline: synchronize before PUT. "
        "Requires Mooncake, retrieval index and budgets. Does not enable D sparse "
        "attention/predictive serving or claim GPU/RDMA validation or overlap.",
    )
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
    index_mode = getattr(args, "prompt_index_backend", "exact")
    shared_native = getattr(args, "prompt_index_cagra_global_native_bytes", None)
    if index_mode in ("cagra", "cagra-auto"):
        if (
            not args.prompt_index_vector_space
            or not args.prompt_index_budget_bytes
            or not args.prompt_index_cagra_native_bytes
        ):
            raise ValueError(
                "CAGRA requires vector space, index budget and native allocation cap"
            )
        if args.allow_cpu_for_tests:
            raise ValueError("native CAGRA cannot use the CPU test override")
        if shared_native is not None and (
            type(shared_native) is not int
            or shared_native < args.prompt_index_cagra_native_bytes
            or shared_native > args.prompt_index_budget_bytes
        ):
            raise ValueError(
                "shared CAGRA native cap must cover one per-index cap and fit "
                "within the total index budget"
            )
    elif shared_native is not None:
        raise ValueError("shared CAGRA native cap requires a CAGRA backend")
    bounds = (
        getattr(args, "full_kv_fanin_max_slices", None),
        getattr(args, "full_kv_fanin_max_inflight", None),
    )
    if bounds != (None, None) and any(type(v) is not int or v <= 0 for v in bounds):
        raise ValueError("both full-KV fan-in bounds must be positive integers")
    records = getattr(args, "full_kv_fanin_max_records", None)
    if records is not None and (
        type(records) is not int or records <= 0 or bounds == (None, None)
    ):
        raise ValueError("global fan-in requires positive record and shard bounds")
    if args.world_size != 2:
        raise ValueError("PVD requires exactly 2 V storage ranks")
    if args.rank is not None and args.rank not in (0, 1):
        raise ValueError("PVD V rank must be 0 or 1")
    rails = resolve_rank_rails(
        args.rails, args.disaggregation_ib_device, args.world_size
    )
    rail_mode = validate_rank_rail_names(rails)
    if rail_mode == "single-rail-debug":
        logger.warning(
            "PVD single-rail debug mode is active: both V ranks use %s; "
            "there is no rail redundancy or aggregate dual-rail bandwidth",
            rails[0],
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
    if getattr(args, "experimental_cuda_sparse_packing", False):
        if (
            args.allow_cpu_for_tests
            or args.transfer_backend != "mooncake"
            or not args.prompt_index_vector_space
            or not args.prompt_index_budget_bytes
        ):
            raise ValueError(
                "experimental CUDA sparse packing requires Mooncake/CUDA and "
                "--prompt-index-vector-space with --prompt-index-budget-bytes"
            )
        logger.warning(
            "Experimental V CUDA sparse packing uses blocking completion before PUT; "
            "D predictive serving is not enabled and GPU/RDMA validation is pending"
        )
    if args.reaper_interval_secs <= 0:
        raise ValueError("--reaper-interval-secs must be positive")
    if args.rank1_startup_timeout_secs <= 0:
        raise ValueError("--rank1-startup-timeout-secs must be positive")
    return rails


def _parse_device_ids(value: str, world_size: int) -> List[int]:
    try:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(
            "--pvd-rank-devices must contain CUDA device integers"
        ) from exc
    if len(devices) != world_size:
        raise ValueError(
            f"--pvd-rank-devices requires {world_size} entries, got {devices}"
        )
    if any(device < 0 for device in devices):
        raise ValueError("--pvd-rank-devices cannot contain negative device ids")
    if len(set(devices)) != len(devices):
        raise ValueError("each V shard requires a distinct CUDA device")
    return devices


def _build_prompt_index(args: argparse.Namespace, *, device=None):
    """Return a PromptIndexManager, or None when retrieval is not configured.

    None is the default. Without a vector space there is nothing to compare a
    query against, so a V rank builds no index and serves exactly as before.

    When it *is* configured it is given its own budget, sized by the operator.
    Every copy the manager retains is charged against it before allocation, so
    a worker cannot be talked into holding an unbounded mirror of every prompt
    it has ever stored. The budget is a separate object from the transfer
    budget: the two are sized for different things, and letting index copies
    eat the staging headroom would stall uploads that were already admitted.
    """
    if not getattr(args, "prompt_index_vector_space", None):
        return None
    from sglang.srt.disaggregation.pvd.prompt_index import PromptIndexManager
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

    budget_bytes = getattr(args, "prompt_index_budget_bytes", None)
    if not budget_bytes:
        raise ValueError(
            "--prompt-index-budget-bytes is required with "
            "--prompt-index-vector-space: the index retains a copy of every "
            "prompt it indexes, and no size for that is worth guessing"
        )
    # Index copies occupy no transfer slots, so the slot limit is the minimum
    # the budget accepts and every reservation here asks for zero.
    budget = TransferBudget(staging_bytes=budget_bytes, max_inflight=1)
    # The exact CPU backend by default: a V rank can build and search with no
    # cuVS present, and its copies stay off the device holding the KV pool.
    # A CAGRA backend replaces it without other changes.
    backend = None
    index_mode = getattr(args, "prompt_index_backend", "exact")
    shared_native = getattr(args, "prompt_index_cagra_global_native_bytes", None)
    if index_mode in ("cagra", "cagra-auto"):
        from sglang.srt.disaggregation.pvd.cagra_backend import (
            CagraAutoIndexBackend,
            CagraIndexBackend,
        )

        if device is None:
            raise ValueError("CAGRA factory requires the V rank's actual device")
        native_kwargs = dict(
            device=device,
            native_bytes_per_index=args.prompt_index_cagra_native_bytes,
            graph_degree=args.prompt_index_cagra_graph_degree,
            intermediate_degree=args.prompt_index_cagra_intermediate_degree,
            itopk_size=args.prompt_index_cagra_itopk_size,
        )
        if shared_native is not None:
            native_kwargs["global_native_cap_bytes"] = shared_native
        native = CagraIndexBackend(**native_kwargs)
        backend = (
            CagraAutoIndexBackend(native) if index_mode == "cagra-auto" else native
        )
    elif shared_native is not None:
        raise ValueError("shared CAGRA native cap requires a CAGRA backend")
    return PromptIndexManager(
        vector_space=args.prompt_index_vector_space,
        metric=getattr(args, "prompt_index_metric", "ip"),
        budget=budget,
        backend=backend,
    )


async def _reaper(
    store: VectorKVStore,
    interval: float,
    coordinator: VectorCoordinator | None = None,
    health: _MaintenanceReaperHealth | None = None,
) -> None:
    health = health or _MaintenanceReaperHealth()

    async def reap_local_entries():
        store.reap_expired(reap_entries=False)

    async def progress_indexes():
        # Run outside the event loop so index construction cannot block HTTP.
        await asyncio.to_thread(store.progress_prompt_indexes)

    while True:
        await asyncio.sleep(interval)
        # The coordinator is the authority for Entry TTL so its STORED view can
        # never outlive the two shard allocations. Shards still reap timed-out
        # Delivery resources locally.
        steps = [
            ("shard_expired_delivery", reap_local_entries),
            # Drive one bounded round of index builds. A no-op unless a prompt
            # index is configured; per-entry build failures stay in its gate.
            ("prompt_index_progress", progress_indexes),
        ]
        if coordinator is not None:
            steps.append(("coordinator_expiration", coordinator.reap_expired))
        await _run_reaper_round(health, steps)


async def _group_reaper(
    stores: Sequence[VectorKVStore],
    interval: float,
    coordinator: VectorCoordinator,
    health: _MaintenanceReaperHealth | None = None,
) -> None:
    """Reap all local shards, then update the group-level lifecycle once."""
    health = health or _MaintenanceReaperHealth()
    while True:
        await asyncio.sleep(interval)
        steps = []
        for store in stores:

            async def reap_local_entries(store=store):
                store.reap_expired(reap_entries=False)

            async def progress_indexes(store=store):
                await asyncio.to_thread(store.progress_prompt_indexes)

            steps.extend(
                [
                    (f"shard_{store.rank}_expired_delivery", reap_local_entries),
                    (f"shard_{store.rank}_prompt_index_progress", progress_indexes),
                ]
            )
        steps.append(("coordinator_expiration", coordinator.reap_expired))
        await _run_reaper_round(health, steps)


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


def _fanin_coordinator_args(args):
    records = getattr(args, "full_kv_fanin_max_records", None)
    return {
        "full_kv_fanin_max_records": records,
        "full_kv_fanin_max_slices": (
            getattr(args, "full_kv_fanin_max_slices", None)
            if records is not None
            else None
        ),
    }


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
        from sglang.srt.disaggregation.pvd.mooncake_engine import (
            MooncakePVDTransferEngine,
        )
        from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

        engine = MooncakePVDTransferEngine(
            hostname=args.advertise_host,
            gpu_id=local_rank,
            rail=rails[rank],
            budget=TransferBudget(
                staging_bytes=args.transfer_staging_budget_bytes,
                max_inflight=args.transfer_max_inflight,
            ),
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
    prompt_index = (
        _build_prompt_index(args, device=device)
        if getattr(args, "prompt_index_backend", "exact") in ("cagra", "cagra-auto")
        else _build_prompt_index(args)
    )
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
        max_entry_records=getattr(args, "max_entry_records", 8192),
        max_delivery_records=getattr(args, "max_delivery_records", 65536),
        max_legacy_absent_fences=getattr(args, "max_legacy_absent_fences", 4096),
        delivery_timeout_secs=args.delivery_timeout_secs,
        allow_cpu_for_tests=args.allow_cpu_for_tests,
        prompt_index=prompt_index,
        full_kv_fanin_max_slices=getattr(args, "full_kv_fanin_max_slices", None),
        full_kv_fanin_max_inflight=getattr(args, "full_kv_fanin_max_inflight", None),
        allow_cuda_sparse_packing=getattr(
            args, "experimental_cuda_sparse_packing", False
        ),
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
    reaper_health = _MaintenanceReaperHealth()
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
            [
                LocalShardClient(
                    store,
                    preflight=preflight,
                    shard_url=f"http://{args.advertise_host}:{shard_port}",
                ),
                remote_client,
            ],
            entry_ttl_secs=args.entry_ttl_secs,
            delivery_timeout_secs=args.delivery_timeout_secs,
            **_fanin_coordinator_args(args),
        )
        coordinator_runner = web.AppRunner(
            create_coordinator_app(_CoordinatorHealthView(coordinator, reaper_health))
        )
        await coordinator_runner.setup()
        await web.TCPSite(coordinator_runner, args.host, args.coordinator_port).start()
        runners.append(coordinator_runner)

    stop = asyncio.Event()
    _install_signal_handlers(stop)

    reaper_task = asyncio.create_task(
        _reaper(store, args.reaper_interval_secs, coordinator, reaper_health),
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
    reaper_health = _MaintenanceReaperHealth()
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
            runner = web.AppRunner(create_shard_app(store, preflight=preflights[rank]))
            await runner.setup()
            await web.TCPSite(runner, args.host, args.shard_port_base + rank).start()
            runners.append(runner)

        coordinator = VectorCoordinator(
            [
                LocalShardClient(
                    store,
                    preflight=preflights[rank],
                    shard_url=(
                        f"http://{args.advertise_host}:{args.shard_port_base + rank}"
                    ),
                )
                for rank, store in enumerate(stores)
            ],
            entry_ttl_secs=args.entry_ttl_secs,
            delivery_timeout_secs=args.delivery_timeout_secs,
            **_fanin_coordinator_args(args),
        )
        coordinator_runner = web.AppRunner(
            create_coordinator_app(_CoordinatorHealthView(coordinator, reaper_health))
        )
        await coordinator_runner.setup()
        await web.TCPSite(coordinator_runner, args.host, args.coordinator_port).start()
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
            _group_reaper(
                stores, args.reaper_interval_secs, coordinator, reaper_health
            ),
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
    level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger().setLevel(level)
    if args.rank is None:
        asyncio.run(_serve_group(args))
    else:
        asyncio.run(_serve_rank(args))


if __name__ == "__main__":
    main()
