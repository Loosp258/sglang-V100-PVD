"""Experimental one-kernel V gather/pack into an owned sparse PUT buffer.

The selected token IDs have already been authorized against a pinned index.
This module neither searches CAGRA nor submits Mooncake work: the caller owns
the Entry, destination, metadata workspace, CUDA fence and RDMA lifecycle.
"""

import torch
import triton
import triton.language as tl
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_pack_plan import (
    SparsePackCompletionUnknown,
    build_sparse_pack_plan,
)
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

# A failed CUDA synchronization is not evidence that temporary metadata is
# dead. Keep its tensors and budget charged until process exit in that case.
_QUARANTINED_WORKSPACES = []


@triton.jit
def _gather_pack_sparse_bytes(
    source,
    destination,
    token_ids,
    group_meta,
    rows: tl.constexpr,
    heads: tl.constexpr,
    layers: tl.constexpr,
    head_bytes: tl.constexpr,
    block_bytes: tl.constexpr,
):
    group = tl.program_id(0)
    byte = tl.program_id(1).to(tl.int64) * block_bytes + tl.arange(0, block_bytes).to(
        tl.int64
    )
    layer = tl.load(group_meta + group * 5)
    head = tl.load(group_meta + group * 5 + 1)
    token_start = tl.load(group_meta + group * 5 + 2)
    count = tl.load(group_meta + group * 5 + 3)
    destination_start = tl.load(group_meta + group * 5 + 4)
    kind = byte // (count * head_bytes)
    token_row = (byte % (count * head_bytes)) // head_bytes
    within_head = byte % head_bytes
    valid = byte < 2 * count * head_bytes
    token = tl.load(token_ids + token_start + token_row, mask=valid, other=0)
    component = layer + kind * layers
    source_byte = (
        component * rows * heads * head_bytes
        + token * heads * head_bytes
        + head * head_bytes
        + within_head
    )
    value = tl.load(source + source_byte, mask=valid, other=0)
    tl.store(destination + destination_start + byte, value, mask=valid)


class SparsePackWorkspace:
    """Budgeted metadata kept alive until a successful caller-owned CUDA fence."""

    def __init__(self, manifest, *, shard, layout, device, budget, owner):
        if (
            not isinstance(manifest, SparseDeliveryManifest)
            or not isinstance(budget, TransferBudget)
            or not isinstance(owner, str)
            or not owner
            or torch.device(device).type != "cuda"
        ):
            raise SparsePayloadError("explicit CUDA sparse pack workspace required")
        plan = build_sparse_pack_plan(manifest, layout, shard)
        token_ids = plan.token_ids
        group_meta = tuple(value for group in plan.groups for value in group)
        self.max_group_bytes = plan.max_group_bytes
        self.bytes = plan.metadata_bytes
        self.source_bytes = plan.source_bytes
        self.destination_bytes = plan.destination_bytes
        self.device = torch.device(device)
        self.manifest_fingerprint = manifest.fingerprint
        self.layout_fingerprint = layout.fingerprint
        self.shard = shard
        self.owner = owner
        self.budget = budget
        self.group_count = len(plan.groups)
        self._released = False
        budget.reserve(owner, self.bytes, 0)
        self.token_ids = self.group_meta = None
        try:
            self.token_ids = torch.tensor(
                token_ids, dtype=torch.int64, device=self.device
            )
            self.group_meta = torch.tensor(
                group_meta, dtype=torch.int64, device=self.device
            )
        except BaseException:
            try:
                torch.cuda.synchronize(self.device)
            except BaseException as exc:
                _QUARANTINED_WORKSPACES.append(self)
                raise SparsePackCompletionUnknown(
                    "sparse metadata upload completion is unknown"
                ) from exc
            else:
                budget.release(owner)
                self._released = True
            raise

    def release_after_fence(self):
        """Caller must prove CUDA completion before invoking this method."""
        if not self._released:
            self.budget.release(self.owner)
            self._released = True
            self.token_ids = self.group_meta = None


def launch_sparse_pack(
    source,
    destination,
    workspace,
    *,
    rows,
    heads,
    layers,
    head_dim,
    element_bytes,
):
    """Enqueue one gather/pack kernel; does not imply CUDA completion."""
    if (
        not isinstance(workspace, SparsePackWorkspace)
        or workspace._released
        or source.device != workspace.device
        or destination.device != workspace.device
        or source.dtype != torch.uint8
        or destination.dtype != torch.uint8
        or any(type(n) is not int or n <= 0 for n in (rows, heads, layers, head_dim))
        or element_bytes not in (2, 4)
        or source.numel() != workspace.source_bytes
        or destination.numel() != workspace.destination_bytes
        or source.numel() != 2 * layers * rows * heads * head_dim * element_bytes
    ):
        raise SparsePayloadError("invalid fused sparse pack source/workspace")
    # A V worker group shares one process across GPUs. Triton launches against
    # the thread-current CUDA device, which need not be this shard's device.
    with torch.cuda.device(workspace.device):
        _gather_pack_sparse_bytes[
            (workspace.group_count, triton.cdiv(workspace.max_group_bytes, 1024))
        ](
            source,
            destination,
            workspace.token_ids,
            workspace.group_meta,
            rows,
            heads,
            layers,
            head_dim * element_bytes,
            1024,
        )
