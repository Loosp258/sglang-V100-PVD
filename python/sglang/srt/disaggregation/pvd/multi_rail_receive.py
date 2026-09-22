"""Receive-only dispatch across explicitly owned per-HCA transfer adapters.

One D GPU may receive separate V shard writes through different HCA sessions.
Each native adapter registers its own private destination, so an endpoint/rkey
never crosses rails. This wrapper does not create native engines or assert that
two HCAs can reach the same GPU; startup preflight must prove that separately.
"""

import threading

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
        self._thread = threading.get_ident()

    def _owner(self):
        if threading.get_ident() != self._thread:
            raise MultiRailReceiveError("multi-rail receive uses its owner thread")

    def supports_rail(self, rail):
        return rail in self.adapters

    def register_memory(self, buffer, *, endpoint, rank, rail, metadata=None):
        self._owner()
        engine = self.adapters.get(rail)
        if engine is None:
            raise MultiRailReceiveError(f"unconfigured receive rail {rail!r}")
        registration = engine.register_memory(
            buffer,
            endpoint=endpoint,
            rank=rank,
            rail=rail,
            metadata=metadata,
        )
        # The child may still own native state if a later validation fails;
        # store the identity immediately and leave descriptor validation to
        # CUDASparseReceiveRegistry, which retains failed registrations.
        self._registrations[registration.descriptor.region_id] = (
            engine,
            registration,
        )
        return registration

    def release_memory(self, registration: RegisteredMemory):
        self._owner()
        if not isinstance(registration, RegisteredMemory):
            raise MultiRailReceiveError("explicit registered destination required")
        region_id = registration.descriptor.region_id
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
        self._owner()
        return {
            "backend": self.name,
            "rails": {rail: engine.health() for rail, engine in self.adapters.items()},
            "registered_destinations": len(self._registrations),
        }
