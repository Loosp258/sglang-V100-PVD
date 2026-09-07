"""Strict rank/rail and GPUDirect preflight checks for PVD."""

from __future__ import annotations

import json
import platform
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch
from sglang.srt.disaggregation.pvd.transfer_engine import (
    MemorySlice,
    TransferEngine,
    TransferStatus,
)


class PVDPreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class PVDPreflightReport:
    rank: int
    rail: str
    rail_mode: str
    cuda_device: str
    rail_present: bool
    active_port: bool
    gpu_memory_registered: bool
    local_gpu_transfer: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


DEFAULT_RANK_RAILS = ("mlx5_0", "mlx5_1")


def validate_rank_rail_names(rails: Iterable[str]) -> str:
    values = list(rails)
    if not values or any(
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", value) is None
        for value in values
    ):
        raise PVDPreflightError(f"Invalid PVD HCA names: {values}")
    distinct = len(set(values))
    if distinct == 1:
        return "single-rail-debug"
    return "dual-rail" if distinct == 2 else "multi-rail"


def validate_dual_rail_names(rails: Iterable[str]) -> None:
    """Backward-compatible validator for callers that require dual rail."""
    values = list(rails)
    if len(values) != 2 or validate_rank_rail_names(values) != "dual-rail":
        raise PVDPreflightError(
            f"PVD dual-rail mode requires two distinct HCA names, got {values}"
        )


def resolve_rank_rails(
    rank_rails: str | None, ib_device: str | None, world_size: int
) -> list[str]:
    """Resolve PVD rank order without passing a shared HCA list to Mooncake.

    The legacy flag requires one HCA per rank. The common IB flag also accepts
    a single HCA shared by all ranks. Defaults apply only if both are absent.
    """

    def parse(value: str, flag: str, broadcast: bool = False) -> list[str]:
        value = value.strip()
        if broadcast and (value.startswith("{") or value.endswith(".json")):
            # Also accept the normalized mapping emitted to Mooncake, so
            # revalidating serialized model arguments is idempotent.
            if value.endswith(".json"):
                value = Path(value).read_text(encoding="utf-8")
            mapping = json.loads(value)
            if not isinstance(mapping, dict) or set(mapping) != {
                str(rank) for rank in range(world_size)
            }:
                raise ValueError(
                    f"{flag} JSON requires exactly ranks 0..{world_size - 1}"
                )
            rails = [mapping[str(rank)] for rank in range(world_size)]
            rails = [rail.strip() if isinstance(rail, str) else rail for rail in rails]
        else:
            rails = [item.strip() for item in value.split(",")]
        try:
            validate_rank_rail_names(rails)
        except PVDPreflightError as exc:
            raise ValueError(f"{flag}: {exc}") from exc
        if broadcast and len(rails) == 1:
            rails *= world_size
        if len(rails) != world_size:
            raise ValueError(
                f"{flag} requires one HCA per storage rank / TP rank "
                f"({world_size} entries), got {len(rails)}"
            )
        return rails

    explicit = parse(rank_rails, "--pvd-rank-rails") if rank_rails is not None else None
    common = (
        parse(ib_device, "--disaggregation-ib-device", broadcast=True)
        if ib_device is not None
        else None
    )
    if explicit is not None and common is not None and explicit != common:
        raise ValueError(
            "conflicting --pvd-rank-rails and --disaggregation-ib-device mappings"
        )
    if explicit is not None:
        return explicit
    if common is not None:
        return common
    return parse(
        ",".join(DEFAULT_RANK_RAILS),
        "default rails; specify --disaggregation-ib-device",
    )


def _has_active_port(rail_path: Path) -> bool:
    ports = rail_path / "ports"
    if not ports.is_dir():
        return False
    for state_path in ports.glob("*/state"):
        try:
            if "ACTIVE" in state_path.read_text(encoding="utf-8").upper():
                return True
        except OSError:
            continue
    return False


def run_rank_preflight(
    *,
    rank: int,
    rails: Iterable[str],
    device: str,
    engine: TransferEngine,
    strict: bool = True,
) -> PVDPreflightReport:
    rails = list(rails)
    rail_mode = validate_rank_rail_names(rails)
    rail = rails[rank]
    if platform.system() != "Linux" and strict:
        raise PVDPreflightError("production PVD RDMA preflight requires Linux")
    if not torch.cuda.is_available() and strict:
        raise PVDPreflightError(
            "CUDA is unavailable; GPUDirect RDMA cannot be verified"
        )

    rail_path = Path("/sys/class/infiniband") / rail
    rail_present = rail_path.is_dir()
    active_port = _has_active_port(rail_path) if rail_present else False
    if strict and not rail_present:
        raise PVDPreflightError(f"RDMA rail {rail} is not present")
    if strict and not active_port:
        raise PVDPreflightError(f"RDMA rail {rail} has no ACTIVE port")

    registered = False
    local_transfer = False
    source_registration = None
    destination_registration = None
    try:
        source = torch.arange(256, dtype=torch.uint8, device=device)
        destination = torch.zeros(256, dtype=torch.uint8, device=device)
        source_registration = engine.register_memory(
            source, endpoint="preflight-source", rank=rank, rail=rail
        )
        destination_registration = engine.register_memory(
            destination, endpoint="preflight-destination", rank=rank, rail=rail
        )
        registered = True
        handle = engine.submit_put(
            MemorySlice(source_registration, 0, 256),
            destination_registration.descriptor,
        )
        local_transfer = engine.poll(handle) == TransferStatus.SUCCESS
        if local_transfer:
            torch.cuda.synchronize(torch.device(device))
            local_transfer = bool(torch.equal(source, destination))
    except Exception as exc:
        if strict:
            raise PVDPreflightError(
                f"GPUDirect registration/transfer failed on rank {rank} rail {rail}: {exc}"
            ) from exc
    finally:
        if source_registration is not None:
            engine.release_memory(source_registration)
        if destination_registration is not None:
            engine.release_memory(destination_registration)

    if strict and not (registered and local_transfer):
        raise PVDPreflightError(
            f"GPUDirect preflight did not complete on rank {rank} rail {rail}"
        )
    return PVDPreflightReport(
        rank=rank,
        rail=rail,
        rail_mode=rail_mode,
        cuda_device=device,
        rail_present=rail_present,
        active_port=active_port,
        gpu_memory_registered=registered,
        local_gpu_transfer=local_transfer,
    )
