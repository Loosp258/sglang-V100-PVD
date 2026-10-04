"""Budgeted row IDs for experimental Torch packing, never completion proof.

The caller owns Entry/index/output leases and drains the current CUDA stream
on success and exceptions before release_after_fence. Constructor UNKNOWN is
quarantined. A singleton needs no index upload or gather kernel.
"""

import torch
from sglang.srt.disaggregation.pvd.protocol import KVLayoutSignature, KVShardManifest
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_pack_plan import SparsePackCompletionUnknown
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

_QUARANTINED_WORKSPACES = []


class SparseRowIndexWorkspace:
    def __init__(self, manifest, *, layout, shard, device, budget, owner):
        if (not isinstance(manifest, SparseDeliveryManifest)
                or not isinstance(layout, KVLayoutSignature)
                or not isinstance(shard, KVShardManifest)
                or not isinstance(budget, TransferBudget)
                or not isinstance(owner, str) or not owner):
            raise SparsePayloadError("explicit row-index layout/manifest/budget required")
        self.device = torch.device(device)
        heads = layout.kv_heads_per_rank
        if (self.device.type not in ('cpu', 'cuda') or type(heads) is not int or heads <= 0
                or any(not 0 <= s.kv_head - shard.rank * heads < heads for s in manifest.specs)):
            raise SparsePayloadError("row-index device or local head mismatch")
        self.manifest, self.layout, self.shard = manifest, layout, shard
        self.budget, self.owner = budget, owner
        self.index_count = sum(len(s.token_ids) for s in manifest.specs if len(s.token_ids) > 1)
        self.index_bytes = self.index_count * 8
        # Conservative scratch admission for Python IDs/groups and both arrays.
        self.bytes = self.index_count * 64 + len(manifest.specs) * 512
        self._released = False
        self._host = self._indices = None
        self._groups = ()
        budget.reserve(owner, self.bytes, 0)
        try:
            flat, spans = [], []
            for spec in manifest.specs:
                if len(spec.token_ids) == 1:
                    spans.append(None)
                else:
                    spans.append((len(flat), len(spec.token_ids)))
                    head = spec.kv_head - shard.rank * heads
                    flat.extend(t * heads + head for t in spec.token_ids)
            if flat:
                self._host = torch.tensor(flat, dtype=torch.int64, device='cpu')
                self._indices = self._host if self.device.type == 'cpu' else self._host.to(
                    self.device, non_blocking=True)
            self._groups = tuple(None if span is None else self._indices.narrow(0, *span)
                for span in spans)
        except BaseException:
            if self.device.type == 'cuda':
                try:
                    torch.cuda.synchronize(self.device)
                except BaseException as exc:
                    _QUARANTINED_WORKSPACES.append(self)
                    raise SparsePackCompletionUnknown("row-index upload completion unknown") from exc
            self.release_after_fence()
            raise

    def check_matches(self, manifest, layout, shard, device):
        # Internal per-delivery objects, not cached layout hashes or remote proof.
        # The pack helper independently validates every live layout/spec first.
        if (self._released or self.manifest is not manifest or self.layout is not layout
                or self.shard is not shard or self.device != device
                or len(self._groups) != len(manifest.specs)):
            raise SparsePayloadError("row-index workspace is stale or belongs to another delivery")
        if self.index_count and (self._indices.device != device or self._indices.dtype != torch.int64
                or self._indices.ndim != 1 or self._indices.numel() != self.index_count
                or not self._indices.is_contiguous()):
            raise SparsePayloadError("invalid owned row-index tensor")
        offset = 0
        for spec, ids in zip(manifest.specs, self._groups, strict=True):
            count = len(spec.token_ids)
            if count == 1:
                if ids is not None: raise SparsePayloadError("singleton must not use row indexes")
            else:
                if (not isinstance(ids, torch.Tensor) or ids.device != device or ids.dtype != torch.int64
                        or ids.ndim != 1 or ids.numel() != count or not ids.is_contiguous()
                        or ids.data_ptr() != self._indices.data_ptr() + offset * 8):
                    raise SparsePayloadError("row-index group cannot resize its owned output")
                offset += count

    def index_for(self, group):
        return self._groups[group]

    def release_after_fence(self):
        """CPU work must be synchronous; CUDA caller must prove completion."""
        if not self._released:
            self.budget.release(self.owner)
            self._released = True
            self._groups = ()
            self._host = self._indices = None
