"""Bounded V-many to D-one sparse receive aggregation for a TP1 CUDA bank.

Every V WRITE has a separate registered destination and terminal proof. Only
after all proofs and local visibility ordering do we copy into one D-owned
contiguous staging buffer. The bank receives the complete group set once.
"""

import uuid

import torch
from sglang.srt.disaggregation.pvd.cuda_runtime_group import CUDARuntimeInstallGroup
from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import (
    CUDASparseReceiveRecord,
    CUDASparseReceiveRegistry,
)
from sglang.srt.disaggregation.pvd.search_routing import RoutedShardSearchClient
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVPayload
from sglang.srt.disaggregation.pvd.sparse_receiver import SparseReceiveError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferBudget,
)


class CUDASparseFanInStage:
    """Single-use owner for all source MRs, aggregate bytes and install receipt."""

    def __init__(self, group, registry, routing, budget):
        if (
            not isinstance(group, CUDARuntimeInstallGroup)
            or not isinstance(registry, CUDASparseReceiveRegistry)
            or not isinstance(routing, RoutedShardSearchClient)
            or not isinstance(budget, TransferBudget)
        ):
            raise SparseReceiveError(
                "explicit CUDA group, registry, route and budget required"
            )
        if (
            routing.compute_rank not in group._banks
            or len(group._banks) != 1
            or group._banks[routing.compute_rank].device != registry.device
            or group._banks[routing.compute_rank].identity[2]
            != routing.entry_transfer_id
            or group._banks[routing.compute_rank].identity[3]
            != routing.compute_fingerprint
            or group._banks[routing.compute_rank].expected_groups
            != frozenset(routing.groups)
            or group._banks[routing.compute_rank].prompt_tokens
            != routing.scope.prompt_tokens
            or group._banks[routing.compute_rank].head_dim != routing.scope.head_dim
            or str(group._banks[routing.compute_rank].dtype) != routing.dtype
        ):
            raise SparseReceiveError("routed sources do not cover the exact D bank")
        self.group, self.registry, self.routing, self.budget = (
            group,
            registry,
            routing,
            budget,
        )
        self._used = self._unknown = False
        self._aggregate = self._guard = None
        self._owner = "cuda-sparse-fanin:" + uuid.uuid4().hex
        self._records = ()
        self.receipt = None

    def _synchronize(self):
        torch.cuda.synchronize(self.registry.device)

    def _allocate(self, size):
        return torch.empty(size, dtype=torch.uint8, device=self.registry.device)

    def _validate(self, epoch, plans, records):
        self.registry._owner()
        self.group._check_stage(epoch, self.routing.compute_rank)
        if self._used or self._unknown:
            raise SparseReceiveError("fan-in stage is already used or quarantined")
        if not isinstance(plans, tuple) or not isinstance(records, dict):
            raise SparseReceiveError("complete source plans and records required")
        expected = self.routing.partition_specs(
            tuple(spec for plan in plans for spec in plan.decode_specs)
        )
        if plans != expected or set(records) != {plan.storage_rank for plan in plans}:
            raise SparseReceiveError("source plans/records differ from trusted layout")
        bank = self.group._banks[self.routing.compute_rank]
        for plan in plans:
            rank = plan.storage_rank
            record = records[rank]
            if (
                not isinstance(record, CUDASparseReceiveRecord)
                or record._registry is not self.registry
                or self.registry._records.get(record.identity.transfer_id) is not record
                or record.identity.shard_rank != rank
                or record.identity.key.transfer_id != self.routing.entry_transfer_id
                or record.manifest != plan.manifest
                or getattr(record._client, "base_url", None)
                != self.routing._endpoints[rank]
                or getattr(record._client, "rank", None) != rank
                or record._lock.locked()
                or not record._ready
                or not record._safe
                or record._receipt is not None
                or getattr(record, "_fanin_stage", None) is not None
                or record._closing
                or record._local_unknown is not None
                or record._source_guard is None
                or record._source_guard.value is not record._registration
                or record._registration.buffer is not record._buffer
                or record._buffer.device != bank.device
                or plan.decode_specs[0].request_id != bank.identity[0]
                or plan.decode_specs[0].incarnation != bank.identity[1]
                or plan.decode_specs[0].operation_id != epoch.operation_id
                or plan.decode_specs[0].target_tokens != epoch.target_tokens
            ):
                raise SparseReceiveError(
                    "source lacks exact terminal proof or D identity"
                )
        return sum(plan.manifest.nbytes for plan in plans)

    def stage(self, epoch, plans, records):
        """All-or-nothing local stage; returned receipt is NOT an install ACK."""
        size = self._validate(epoch, plans, records)
        # A capacity refusal happens before source ownership or CUDA work.
        self.budget.reserve(self._owner, size, 1)
        self._used = True
        self._records = tuple(records[plan.storage_rank] for plan in plans)
        # Keep this owner discoverable from the registry's retained records even
        # if a caller drops its Python reference after a failing CUDA operation.
        for record in self._records:
            record._fanin_stage = self
        pins, views = [], []
        try:
            for record in self._records:
                token = self._owner + ":v" + str(record.identity.shard_rank)
                record._source_guard.pin(token)
                pins.append((record._source_guard, token))
            self._aggregate = self._allocate(size)
            self._guard = ResourceGuard(self._aggregate, self._release_aggregate)
            offset = 0
            for plan, record in zip(plans, self._records, strict=True):
                self.registry.ordering.after_remote_write(record._registration)
                record._ordered = True
                count = plan.manifest.nbytes
                self._aggregate[offset : offset + count].copy_(record._buffer)
                offset += count
            self._synchronize()  # All source reads and D writes are complete.
            for guard, token in pins:
                guard.unpin(token)
            pins.clear()
            offset = 0
            for plan in plans:
                count = plan.manifest.nbytes
                wire = plan.manifest.payload_views(
                    self._aggregate[offset : offset + count]
                )
                views.extend(
                    SparseKVPayload(local, item.tensor)
                    for local, item in zip(plan.decode_specs, wire, strict=True)
                )
                for item in wire:
                    item.close()
                offset += count
            receipt = self.group.stage(
                epoch, self.routing.compute_rank, tuple(views), source_guard=self._guard
            )
            if receipt.rank != self.routing.compute_rank or receipt.epoch != epoch:
                raise SparseReceiveError("D bank returned a foreign install receipt")
            self.receipt = receipt
            for record in self._records:
                record._group = self.group.runtime.exchange
                record._receipt = self.receipt
            return self.receipt
        except BaseException:
            try:
                self._synchronize()
            except BaseException:
                self._unknown = True
                for record in self._records:
                    record._local_unknown = "sparse fan-in completion unknown"
                raise
            raise
        finally:
            for item in views:
                item.close()
            views.clear()
            if not self._unknown:
                for guard, token in pins:
                    guard.unpin(token)
                if self._guard is not None:
                    self._guard.request_release()
                if self._guard is None or self._guard.value is None:
                    self._aggregate = None
                    self.budget.release(self._owner)

    def _release_aggregate(self):
        self._aggregate = None

    def snapshot(self):
        return {
            "used": self._used,
            "unknown": self._unknown,
            "sources": len(self._records),
            "staging_retained": self._aggregate is not None,
            "installed": self.receipt is not None
            and self.group.installation_complete(self.receipt),
        }
