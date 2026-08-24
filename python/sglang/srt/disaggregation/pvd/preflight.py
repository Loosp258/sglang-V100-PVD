"""Strict dual-rail and GPUDirect preflight checks for PVD v1."""

from __future__ import annotations

import platform
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
    cuda_device: str
    rail_present: bool
    active_port: bool
    gpu_memory_registered: bool
    local_gpu_transfer: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def validate_dual_rail_names(rails: Iterable[str]) -> None:
    values = list(rails)
    if values != ["mlx5_0", "mlx5_1"]:
        raise PVDPreflightError(
            f"PVD v1 requires rails ['mlx5_0', 'mlx5_1'], got {values}"
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
    validate_dual_rail_names(rails)
    rail = rails[rank]
    if platform.system() != "Linux" and strict:
        raise PVDPreflightError("production PVD RDMA preflight requires Linux")
    if not torch.cuda.is_available() and strict:
        raise PVDPreflightError("CUDA is unavailable; GPUDirect RDMA cannot be verified")

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
        cuda_device=device,
        rail_present=rail_present,
        active_port=active_port,
        gpu_memory_registered=registered,
        local_gpu_transfer=local_transfer,
    )
