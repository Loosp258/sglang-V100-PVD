"""HTTP Delivery sink for the controlled CPU prefetch loop, not Scheduler TP.

Search returns logical selections. This sink publishes owned destinations to
the chosen V shards, waits for terminal byte delivery, and stages CPU copies.
Installation remains at the request's boundary. ACK/cleanup are asynchronous
and observable; failed cleanup never drops the registry's memory owner.
"""

import asyncio
import math
import uuid
from dataclasses import dataclass

from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_install import CPUInstallGroup
from sglang.srt.disaggregation.pvd.sparse_receiver import (
    SparseReceiveError,
    SparseReceiveRegistry,
)


@dataclass(frozen=True)
class CPUReceiveRoute:
    client: object
    sender_epoch: str
    endpoint: str
    rail: str

    def __post_init__(self):
        if any(
            not isinstance(s, str) or not s.strip()
            for s in (self.sender_epoch, self.endpoint, self.rail)
        ):
            raise SparseReceiveError(
                "explicit chosen V epoch and D endpoint/rail required"
            )


class CPUSparseDelivery:
    def __init__(self, group, registry, *, key, routes, poll_interval_seconds):
        if not isinstance(group, CPUInstallGroup) or not isinstance(
            registry, SparseReceiveRegistry
        ):
            raise SparseReceiveError(
                "explicit CPU install group and receive registry required"
            )
        registry._owner()
        metadata = group.describe_banks()
        if group.coordinator.identity[2] != key.transfer_id:
            raise SparseReceiveError("Delivery Entry differs from installation Entry")
        if set(routes) != set(metadata) or any(
            type(rank) is not int or not isinstance(route, CPUReceiveRoute)
            for rank, route in routes.items()
        ):
            raise SparseReceiveError(
                "one explicit Delivery route per CPU bank required"
            )
        if (
            type(poll_interval_seconds) not in (int, float)
            or not math.isfinite(poll_interval_seconds)
            or poll_interval_seconds <= 0
        ):
            raise SparseReceiveError("finite positive delivery poll interval required")
        self.group, self.registry, self.key = group, registry, key
        self._routes, self._metadata = dict(routes), metadata
        self._interval = float(poll_interval_seconds)
        self._rounds, self._tasks, self._errors = {}, {}, {}
        self._closed = False
        self._scope = "cpu-sparse-delivery:" + uuid.uuid4().hex

    def _live(self):
        self.registry._owner()
        if self._closed:
            raise SparseReceiveError("CPU sparse Delivery sink is closed")

    async def stage(self, epoch, rank, specs):
        self._live()
        self.group.coordinator._match(epoch)
        if type(rank) is not int or rank not in self._routes:
            raise SparseReceiveError("unknown receiving rank")
        records = self._rounds.setdefault(epoch, {})
        if rank in records:
            raise SparseReceiveError("rank already has a destination for this epoch")
        manifest = SparseDeliveryManifest(
            tuple(specs), "torch.float32", self._metadata[rank]["head_dim"]
        )
        first = manifest.specs[0]
        if (
            first.request_id,
            first.incarnation,
            first.entry_transfer_id,
            first.operation_id,
            first.target_tokens,
        ) != (
            epoch.request_id,
            epoch.incarnation,
            epoch.entry_transfer_id,
            epoch.operation_id,
            epoch.target_tokens,
        ) or first.layout_fingerprint != self._metadata[rank]["identity"][3]:
            raise SparseReceiveError(
                "selection does not belong to this installation epoch"
            )
        if {(s.layer, s.kv_head) for s in manifest.specs} != self._metadata[rank][
            "groups"
        ]:
            raise SparseReceiveError(
                "selection groups do not cover the destination bank"
            )
        route = self._routes[rank]
        record = self.registry.prepare(
            manifest,
            key=self.key,
            rank=rank,
            rail=route.rail,
            endpoint=route.endpoint,
            sender_epoch=route.sender_epoch,
            client=route.client,
            owner_scope=self._scope,
        )
        records[rank] = record  # retain before any publication/await
        ready = await record.start()
        while not ready:
            self._live()
            await asyncio.sleep(self._interval)
            ready = await record.poll()
        self._live()
        self.group.coordinator._match(epoch)
        return record.stage(self.group, epoch)

    def require_installable(self, epoch):
        self._live()
        asyncio.get_running_loop()  # ACK dispatch must be available BEFORE install
        records = self._rounds.get(epoch, {})
        if set(records) != set(self._routes) or not all(
            r.snapshot()["staged"] for r in records.values()
        ):
            raise SparseReceiveError(
                "every destination must be staged before installation"
            )

    def installed(self, epoch):
        self.require_installable(epoch)
        for record in self._rounds[epoch].values():
            record.confirm_install()
        self.retry_acks(epoch)

    def retry_acks(self, epoch):
        """Explicit retry, no spin on a failed remote ACK or unregister."""
        self._live()
        prior = self._tasks.get(epoch)
        if prior is not None and not prior.done():
            return prior
        records = self._rounds.get(epoch)
        if not records or not all(r.snapshot()["installed"] for r in records.values()):
            raise SparseReceiveError("only a completed installation can be ACKed")
        self._tasks[epoch] = asyncio.create_task(self._ack_round(epoch))
        return self._tasks[epoch]

    async def _ack_round(self, epoch):
        errors = {}
        for rank, record in tuple(self._rounds[epoch].items()):
            if record.snapshot()["closed"]:
                continue
            try:
                if not record.snapshot()["acknowledged"]:
                    await record.ack()
                if not await record.close():
                    errors[rank] = "destination not fenced"
            except Exception as exc:  # noqa: BLE001 -- retain destination for retry
                errors[rank] = str(exc)
        if errors:
            self._errors[epoch] = errors
        else:
            self._errors.pop(epoch, None)
            self._rounds.pop(epoch, None)
        # No accumulation of completed Tasks, tensors or receipts over rounds.
        self._tasks.pop(epoch, None)
        return errors

    def cancel(self):
        self.registry._owner()
        self._closed = True  # no new destinations or staging; not a write fence

    async def close(self):
        self.cancel()
        tasks = tuple(self._tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # One cleanup pass over all this request's owners, including failed
        # registrations. Report only unresolved owners from that pass.
        errors = await self.registry.close(owner_scope=self._scope)
        for epoch, records in tuple(self._rounds.items()):
            if all(record.snapshot()["closed"] for record in records.values()):
                self._rounds.pop(epoch, None)
                self._errors.pop(epoch, None)
        return errors

    def snapshot(self):
        self.registry._owner()
        return {
            "closed": self._closed,
            "pending_rounds": len(self._rounds),
            "retained_destinations": len(
                self.registry.snapshot(owner_scope=self._scope)
            ),
            "ack_tasks": len(self._tasks),
            "errors": {
                e.operation_id: dict(errors) for e, errors in self._errors.items()
            },
        }
