"""Receive-only dispatch across explicitly owned per-HCA transfer adapters.

One D GPU may receive separate V shard writes through different HCA sessions.
Each native adapter registers its own private destination, so an endpoint/rkey
never crosses rails. This wrapper does not create native engines or assert that
two HCAs can reach the same GPU; startup preflight must prove that separately.
"""

import os
import platform
import threading
from pathlib import Path

import torch
from sglang.srt.disaggregation.pvd.transfer_engine import (
    RegisteredMemory,
    TransferEngine,
)


class MultiRailReceiveError(ValueError):
    pass


class RailMappedReceiveEngine(TransferEngine):
    """Dispatch D registrations by rail; source PUTs remain V's responsibility."""

    name = "rail_mapped_receive"

    def __init__(self, adapters):
        if (
            not isinstance(adapters, dict)
            or not 1 <= len(adapters) <= 8
            or any(
                not isinstance(rail, str)
                or not rail.strip()
                or not isinstance(engine, TransferEngine)
                or getattr(engine, "rail", rail) != rail
                for rail, engine in adapters.items()
            )
            or len({id(engine) for engine in adapters.values()}) != len(adapters)
        ):
            raise MultiRailReceiveError(
                "distinct explicit per-rail transfer adapters required"
            )
        self.adapters = dict(adapters)
        self._registrations = {}
        self._quarantined_collisions = []
        self._lock = threading.RLock()

    def supports_rail(self, rail):
        return rail in self.adapters

    def register_memory(self, buffer, *, endpoint, rank, rail, metadata=None):
        engine = self.adapters.get(rail)
        if engine is None:
            raise MultiRailReceiveError(f"unconfigured receive rail {rail!r}")
        with self._lock:
            registration = engine.register_memory(
                buffer,
                endpoint=endpoint,
                rank=rank,
                rail=rail,
                metadata=metadata,
            )
            # The child may still own native state if later validation fails;
            # retain its exact identity before returning to the registry.
            region_id = registration.descriptor.region_id
            if region_id in self._registrations:
                # Preserve the original owner. A colliding native descriptor
                # cannot be routed safely and requires process-level recovery.
                self._quarantined_collisions.append((engine, registration))
                raise MultiRailReceiveError("duplicate native region identity")
            self._registrations[region_id] = (engine, registration)
        return registration

    def release_memory(self, registration: RegisteredMemory):
        if not isinstance(registration, RegisteredMemory):
            raise MultiRailReceiveError("explicit registered destination required")
        region_id = registration.descriptor.region_id
        with self._lock:
            owner = self._registrations.get(region_id)
            if owner is None or owner[1] is not registration:
                raise MultiRailReceiveError("foreign or already retired destination")
            owner[0].release_memory(registration)
            self._registrations.pop(region_id)

    def submit_put(self, local, remote, *, remote_offset=0):
        raise MultiRailReceiveError("D receive adapter cannot submit a source PUT")

    def poll(self, handle):
        raise MultiRailReceiveError("D receive adapter owns no source PUT handle")

    def abort(self, handle):
        raise MultiRailReceiveError("D receive adapter owns no source PUT handle")

    def health(self):
        with self._lock:
            return {
                "backend": self.name,
                "rails": {
                    rail: engine.health() for rail, engine in self.adapters.items()
                },
                "registered_destinations": len(self._registrations),
                "quarantined_collisions": len(self._quarantined_collisions),
            }


def create_native_receive_group(
    *,
    hostname: str,
    gpu_id: int,
    rails: tuple[str, ...],
    transfer_budget,
    existing_adapter=None,
) -> RailMappedReceiveEngine:
    """Initialize one native Mooncake session per HCA for one D CUDA device.

    A failed startup must terminate the process. In particular, an unknown
    preflight write is retained by the process-level preflight quarantine.
    """
    from sglang.srt.disaggregation.pvd.mooncake_engine import (
        MooncakePVDTransferEngine,
    )
    from sglang.srt.disaggregation.pvd.preflight import (
        _has_active_port,
        run_rank_preflight,
        validate_rank_rail_names,
    )
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
        TransferBudget,
        budget_of,
    )

    if (
        not isinstance(hostname, str)
        or not hostname.strip()
        or type(gpu_id) is not int
        or gpu_id < 0
        or not isinstance(rails, tuple)
        or not 1 <= len(rails) <= 8
        or any(not isinstance(rail, str) for rail in rails)
        or len(set(rails)) != len(rails)
        or not isinstance(transfer_budget, TransferBudget)
    ):
        raise MultiRailReceiveError("explicit D GPU, distinct HCAs and budget required")
    validate_rank_rail_names(rails)
    if (
        platform.system() != "Linux"
        or not torch.cuda.is_available()
        or gpu_id >= torch.cuda.device_count()
        or os.environ.get("MC_FORCE_TCP") == "1"
        or os.environ.get("MOONCAKE_PROTOCOL", "rdma").lower() != "rdma"
    ):
        raise MultiRailReceiveError(
            "native multi-rail receive requires Linux CUDA RDMA"
        )
    for rail in rails:
        path = Path("/sys/class/infiniband") / rail
        if not path.is_dir() or not _has_active_port(path):
            raise MultiRailReceiveError(f"receive HCA {rail} has no ACTIVE port")
    if existing_adapter is not None and (
        not isinstance(existing_adapter, MooncakePVDTransferEngine)
        or existing_adapter.rail not in rails
        or existing_adapter._engine.get_ib_device() != existing_adapter.rail
        or existing_adapter._engine.gpu_id != gpu_id
        or budget_of(existing_adapter) is not transfer_budget
    ):
        raise MultiRailReceiveError(
            "existing D adapter has different GPU, rail or budget"
        )

    adapters = {}
    for rank, rail in enumerate(rails):
        adapter = (
            existing_adapter
            if existing_adapter is not None and existing_adapter.rail == rail
            else MooncakePVDTransferEngine(
                hostname=hostname,
                gpu_id=gpu_id,
                rail=rail,
                budget=transfer_budget,
            )
        )
        if adapter._engine.get_ib_device() != rail:
            raise MultiRailReceiveError(f"Mooncake did not select required HCA {rail}")
        # Strictly prove each native session can register this GPU and complete
        # a local GPUDirect write before exposing any destination descriptor.
        run_rank_preflight(
            rank=rank,
            rails=rails,
            device=f"cuda:{gpu_id}",
            engine=adapter,
            strict=True,
        )
        adapters[rail] = adapter
    return RailMappedReceiveEngine(adapters)
