"""Derive one TP1 CUDA Prompt bank from the actual D receiver receipt.

Planning validates model-pool metadata without allocating GPU memory. Creation
requires a real CUDA placement and builds the local install group/importer;
only a later, explicit install_received() may import the completed Prompt.
"""

import threading
from dataclasses import dataclass

import torch
from sglang.srt.disaggregation.pvd.cuda_prompt_bootstrap import CUDAPromptBootstrap
from sglang.srt.disaggregation.pvd.cuda_runtime_group import CUDARuntimeInstallGroup
from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import (
    CUDASparseReceiveRegistry,
)
from sglang.srt.disaggregation.pvd.cuda_working_set import CUDASparseWorkingSet
from sglang.srt.disaggregation.pvd.decode_refresh import (
    InitialPromptReceipt,
    PVDDecodeSession,
)
from sglang.srt.disaggregation.pvd.protocol import KVLayoutSignature
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

_PARTIAL_GROUP_QUARANTINE = []


@dataclass(frozen=True)
class CUDAInitialBankPlan:
    receipt: InitialPromptReceipt
    identity: tuple[str, str, str, str]
    device: torch.device
    dtype: torch.dtype
    expected_groups: tuple[tuple[int, int], ...]
    prompt_tokens: int
    head_dim: int
    max_union_tokens: int
    interval: int
    peer_epoch: str


@dataclass(frozen=True)
class CUDAInitialGroup:
    plan: CUDAInitialBankPlan
    bank: CUDASparseWorkingSet
    group: CUDARuntimeInstallGroup
    importer: CUDAPromptBootstrap


def plan_received_prompt_bank(session, *, max_union_tokens):
    """Use the receiver's immutable completion receipt, never request JSON."""
    if not isinstance(session, PVDDecodeSession):
        raise InstallProtocolError("real D Prompt receiver session required")
    receipt = session.require_initial_prompt()
    manager = session.manager
    layout = manager.layout()
    registry = getattr(manager, "sparse_receive_registry", None)
    peer_epoch = getattr(manager, "worker_epoch", None)
    if (
        not isinstance(layout, KVLayoutSignature)
        or layout.tp_size != 1
        or layout.pp_size != 1
        or layout.page_size != 1
        or layout.kv_heads_per_rank != layout.total_kv_heads
        or manager.tp_size != 1
        or manager.tp_rank != 0
        or not isinstance(registry, CUDASparseReceiveRegistry)
        or not isinstance(peer_epoch, str)
        or not peer_epoch.strip()
        or type(max_union_tokens) is not int
        or not 0 < max_union_tokens <= len(receipt.prompt)
        or not receipt.pages
        or len(receipt.pages) != len(receipt.prompt)
        or len(set(receipt.pages)) != len(receipt.pages)
        or any(type(page) is not int or page <= 0 for page in receipt.pages)
        or receipt.key.transfer_id != session.key.transfer_id
    ):
        raise InstallProtocolError(
            "exact TP1 receiver, layout and bounded Prompt group required"
        )
    device = registry.device
    pool = manager.kv_pool
    dtype = None
    for layer in range(layout.num_layers):
        for getter in (pool.get_key_buffer, pool.get_value_buffer):
            tensor = getter(layer)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 3
                or tensor.device != device
                or tensor.shape[1:] != (layout.total_kv_heads, layout.head_dim)
                or tensor.shape[0] <= max(receipt.pages)
                or not tensor.is_contiguous()
            ):
                raise InstallProtocolError(
                    "model KV pool does not match the receiver layout or device"
                )
            if dtype is None:
                dtype = tensor.dtype
            elif tensor.dtype != dtype:
                raise InstallProtocolError("model KV components have mixed dtypes")
    if (
        dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or str(dtype) != layout.kv_dtype
    ):
        raise InstallProtocolError("model KV dtype differs from receiver layout")
    return CUDAInitialBankPlan(
        receipt=receipt,
        identity=(
            receipt.request_id,
            receipt.receiver_epoch,
            receipt.key.transfer_id,
            receipt.layout,
        ),
        device=device,
        dtype=dtype,
        expected_groups=tuple(
            (layer, head)
            for layer in range(layout.num_layers)
            for head in range(layout.total_kv_heads)
        ),
        prompt_tokens=len(receipt.prompt),
        head_dim=layout.head_dim,
        max_union_tokens=max_union_tokens,
        interval=session.clock.interval,
        peer_epoch=peer_epoch,
    )


def create_received_prompt_group(
    session,
    *,
    bank_budget,
    staging_budget,
    execution_lock,
    max_union_tokens,
    lead_tokens,
    timeout_seconds,
    max_pending_events,
    max_pending_bytes,
):
    """Construct but do not install, register, or admit a CUDA request."""
    plan = plan_received_prompt_bank(session, max_union_tokens=max_union_tokens)
    if (
        plan.device.type != "cuda"
        or plan.device.index is None
        or not isinstance(bank_budget, TransferBudget)
        or not isinstance(staging_budget, TransferBudget)
        or not isinstance(execution_lock, type(threading.RLock()))
        or type(lead_tokens) is not int
        or not 0 < lead_tokens < plan.interval
        or type(max_pending_events) is not int
        or max_pending_events <= 0
        or type(max_pending_bytes) is not int
        or max_pending_bytes <= 0
        or type(timeout_seconds) not in (int, float)
        or not 0 < timeout_seconds < float("inf")
    ):
        raise InstallProtocolError(
            "explicit CUDA placement, budgets, lock and bounded install policy required"
        )
    bank = CUDASparseWorkingSet(
        device=plan.device,
        dtype=plan.dtype,
        budget=bank_budget,
        request_id=plan.identity[0],
        incarnation=plan.identity[1],
        entry_transfer_id=plan.identity[2],
        layout_fingerprint=plan.identity[3],
        expected_groups=plan.expected_groups,
        prompt_tokens=plan.prompt_tokens,
        head_dim=plan.head_dim,
        max_union_tokens=plan.max_union_tokens,
    )
    group = None
    try:
        group = CUDARuntimeInstallGroup(
            {0: bank},
            interval=plan.interval,
            lead_tokens=lead_tokens,
            peer_epochs={0: plan.peer_epoch},
            timeout_seconds=float(timeout_seconds),
            max_pending_events=max_pending_events,
            max_pending_bytes=max_pending_bytes,
        )
        importer = CUDAPromptBootstrap(
            group, execution_lock=execution_lock, staging_budget=staging_budget
        )
    except BaseException:
        # Before the importer is returned, neither model rows nor a native
        # destination have been published. Still retain the partial owner if
        # close itself cannot prove retirement.
        try:
            (bank if group is None else group).close()
        except BaseException:
            _PARTIAL_GROUP_QUARANTINE.append((bank, group))
            raise
        raise
    return CUDAInitialGroup(plan, bank, group, importer)
