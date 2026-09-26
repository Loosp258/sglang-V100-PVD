"""Import already-delivered full Prompt from D's owned model pool, no index.

The source declaration is NOT remote completion proof. The existing full-Prompt
receiver must finish identity validation, native completion and unpack ordering
BEFORE this importer runs. It never contacts V/search or changes generated KV.
"""

import threading
import traceback
import uuid
from dataclasses import dataclass

import torch
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import TargetExecutionArbiter
from sglang.srt.disaggregation.pvd.cuda_model_attention import CUDAModelPools
from sglang.srt.disaggregation.pvd.cuda_runtime_group import CUDARuntimeInstallGroup
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVPayload, SparseKVSpec
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferBudget,
)


@dataclass(frozen=True)
class CUDAPromptPoolSource:
    identity: tuple[str, str, str, str]  # request/incarnation/Entry/layout
    slot: int
    prompt_tokens: int
    pool_owner: ResourceGuard
    expected_rows: tuple | None = None


class CUDAPromptBootstrap:
    def __init__(self, group, *, execution_lock, staging_budget):
        if not isinstance(group, CUDARuntimeInstallGroup):
            raise InstallProtocolError("explicit CUDA runtime group required")
        if not isinstance(execution_lock, type(threading.RLock())) or not isinstance(
            staging_budget, TransferBudget
        ):
            raise InstallProtocolError(
                "shared reentrant lock and bootstrap budget required"
            )
        self.group, self._lock, self.budget = group, execution_lock, staging_budget
        self._rank, self._bank = next(iter(group._banks.items()))
        self._owner = f"cuda-prompt-bootstrap:{uuid.uuid4().hex}"
        self._held = {}
        self._quarantined = self._active = self._used = False
        self._received_session = self._received_receipt = self._receive_lease = None
        self._received_pool_owner = None

    def install_received(self, session, *, arbiter, pool_owner, cache):
        """Bind the shipped full receiver to this request's initial CUDA bank.

        This explicit entrypoint does not enable serving or construct a
        controller. UNKNOWN keeps the session/importer, arbiter and real pools
        alive, and poisons allocation rather than permitting default cleanup.
        """
        from sglang.srt.disaggregation.pvd.cuda_request_release import (
            _require_supported_pools,
        )
        from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeSession

        if not isinstance(session, PVDDecodeSession) or not isinstance(
            arbiter, TargetExecutionArbiter
        ):
            raise InstallProtocolError("real receiver and shared arbiter required")
        arbiter.owner()
        receipt = session.require_initial_prompt()
        _require_supported_pools(cache)
        manager = session.manager
        pools = pool_owner.value if isinstance(pool_owner, ResourceGuard) else None
        allocator = cache.token_to_kv_pool_allocator
        if (
            self._received_session is not None
            or getattr(session, "_cuda_prompt_importer", None) is not None
            or manager.tp_size != 1
            or manager.tp_rank != 0
            or manager.page_size != 1
            or not isinstance(pools, CUDAModelPools)
            or pools.req_pool is not manager.scheduler.req_to_token_pool
            or pools.kv_pool is not manager.kv_pool
            or cache is not manager.scheduler.tree_cache
            or cache.req_to_token_pool is not pools.req_pool
            or allocator.get_kvcache() is not pools.kv_pool
            or getattr(allocator, "pvd_cuda_retirement_error", None) is not None
            or getattr(pools.req_pool, "pvd_cuda_retirement_error", None) is not None
            or self._bank.identity
            != (
                session.req.rid,
                receipt.receiver_epoch,
                receipt.key.transfer_id,
                receipt.layout,
            )
        ):
            raise InstallProtocolError(
                "exact unclaimed TP1 receiver/model pools required"
            )
        lease = arbiter.acquire()
        self._receive_lease = lease
        self._received_session, self._received_receipt = session, receipt
        self._received_pool_owner = pool_owner
        session._cuda_prompt_importer = self
        try:
            return self.install(
                CUDAPromptPoolSource(
                    self._bank.identity,
                    receipt.slot,
                    len(receipt.prompt),
                    pool_owner,
                    receipt.pages,
                )
            )
        finally:
            if self._quarantined:
                reason = (
                    "CUDA initial Prompt completion unknown; worker pools quarantined"
                )
                pools.req_pool.pvd_cuda_retirement_error = reason
                allocator.pvd_cuda_retirement_error = reason
                # Keep even a reentrant target lock from admitting another forward.
            else:
                arbiter.release(lease)
                self._receive_lease = None
                if not self._used:
                    # Capacity/pre-copy refusal did not consume the bootstrap.
                    session._cuda_prompt_importer = None
                    self._received_session = self._received_receipt = None
                    self._received_pool_owner = None

    def _synchronize(self):
        torch.cuda.synchronize(self._bank.device)

    def _fence(self):
        try:
            self._synchronize()
        except BaseException:
            self._quarantined = True
            raise

    def install(self, source):
        self.group.coordinator._owner()
        if self._active or self._quarantined or self._used:
            raise InstallProtocolError("bootstrap is active, consumed or quarantined")
        if (
            not isinstance(source, CUDAPromptPoolSource)
            or source.identity != self._bank.identity
            or type(source.slot) is not int
            or source.slot <= 0
            or type(source.prompt_tokens) is not int
            or source.prompt_tokens != self._bank.prompt_tokens
            or not isinstance(source.pool_owner, ResourceGuard)
        ):
            raise InstallProtocolError(
                "exact Prompt pool identity, count and owner required"
            )
        if self.group.coordinator.snapshot()["completed"] is not None:
            raise InstallProtocolError("initial Prompt is already installed")
        if not self._lock.acquire(blocking=False):
            raise InstallProtocolError("target execution is busy")
        self._active = True
        pinned = charged = False
        failure = None
        try:
            size = (
                len(self._bank.expected_groups)
                * 2
                * source.prompt_tokens
                * self._bank.head_dim
            )
            size *= torch.finfo(self._bank.dtype).bits // 8
            self.budget.reserve(self._owner, size, 1)
            charged = True
            source.pool_owner.pin(self._owner)
            pinned = True
            self._held["source"] = source
            return self._copy_and_install(source)
        except BaseException as exc:
            failure = exc
            self._held["failure"] = exc
            if pinned:
                self._used = True  # no half-installed bootstrap replay
                self.group.cancel("initial Prompt import failed")
            raise
        finally:
            try:
                if pinned:
                    if self._quarantined:
                        raise RuntimeError(
                            "initial Prompt completion unknown; owners retained"
                        )
                    self._fence()
                    if self._bank.snapshot()["quarantine"] is not None:
                        raise InstallProtocolError("Prompt bank completion unknown")
                if failure is not None:
                    traceback.clear_frames(failure.__traceback__)
                payloads = self._held.pop("payloads", ())
                for payload in payloads:
                    payload.close()
                self._held.pop("backing", None)
                packed = self._held.get("packed_guard")
                if packed is not None:
                    packed.request_release()
                    if packed.value is not None:
                        raise InstallProtocolError(
                            "bootstrap staging still has a reader"
                        )
                if pinned:
                    source.pool_owner.unpin(self._owner)
                self._held.clear()
                if charged:
                    self.budget.release(self._owner)
                self._active = False
                self._lock.release()
            except BaseException:
                self._quarantined = True
                self.group.cancel("initial Prompt cleanup unknown")
                raise

    def _copy_and_install(self, source):
        pools = source.pool_owner.value
        if not isinstance(pools, CUDAModelPools):
            raise InstallProtocolError("allocator-backed model pools required")
        table = pools.req_pool.req_to_token
        if (
            not isinstance(table, torch.Tensor)
            or table.device != self._bank.device
            or table.dtype not in (torch.int32, torch.int64)
            or table.ndim != 2
            or source.slot >= table.shape[0]
            or source.prompt_tokens > table.shape[1]
        ):
            raise InstallProtocolError("invalid Prompt request map")
        try:
            rows = table[source.slot, : source.prompt_tokens].tolist()
        except BaseException:
            # Device readback failed; do not retry a fence to justify releasing
            # the source pool while its completion is unknown.
            self._quarantined = True
            raise
        if source.expected_rows is not None and tuple(rows) != source.expected_rows:
            raise InstallProtocolError("installed Prompt mapping changed since receive")
        # TokenToKVPoolAllocator reserves row zero for padding, not Prompt KV.
        if any(row <= 0 for row in rows) or len(set(rows)) != len(rows):
            raise InstallProtocolError("Prompt rows must be unique allocated positions")
        groups = sorted(self._bank.expected_groups)
        buffers = self._held["buffers"] = []
        for layer, head in groups:
            pair = (
                pools.kv_pool.get_key_buffer(layer),
                pools.kv_pool.get_value_buffer(layer),
            )
            for tensor in pair:
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.ndim != 3
                    or tensor.device != self._bank.device
                    or tensor.dtype != self._bank.dtype
                    or tensor.shape[-1] != self._bank.head_dim
                    or head >= tensor.shape[1]
                    or max(rows) >= tensor.shape[0]
                ):
                    raise InstallProtocolError(
                        "Prompt pool layout/device/rows mismatch"
                    )
            buffers.append(pair)
        backing = torch.empty(
            (len(groups), 2, source.prompt_tokens, self._bank.head_dim),
            dtype=self._bank.dtype,
            device=self._bank.device,
        )
        self._held["backing"] = backing
        packed = ResourceGuard(backing, lambda: None)  # guard drops tensor storage
        self._held["packed_guard"] = packed
        # Snapshot the already-validated absolute pool rows once. One gather
        # per K/V head replaces prompt_tokens tiny GPU copy_ submissions per
        # head, which otherwise blocks the Scheduler's refresh owner loop while
        # a second request enters the waiting queue.
        row_indices = torch.tensor(rows, dtype=torch.int64, device=self._bank.device)
        for i, ((layer, head), pair) in enumerate(zip(groups, buffers, strict=True)):
            for kind, tensor in enumerate(pair):
                torch.index_select(
                    tensor[:, head, :], 0, row_indices, out=backing[i, kind]
                )
        self._fence()
        epoch = self.group.begin(0)
        payloads = tuple(
            SparseKVPayload(
                SparseKVSpec(
                    *source.identity[:2],
                    epoch.operation_id,
                    0,
                    source.identity[2],
                    "full-prompt:no-index",
                    "absolute-prompt-positions:v1",
                    source.identity[3],
                    layer,
                    head,
                    tuple(range(source.prompt_tokens)),
                ),
                backing[i],
            )
            for i, (layer, head) in enumerate(groups)
        )
        self._held["payloads"] = payloads
        receipt = self.group.stage(epoch, self._rank, payloads, source_guard=packed)
        if not self.group.try_install(
            epoch, {self._rank: 0}
        ) or not self.group.installation_complete(receipt):
            raise InstallProtocolError("initial Prompt runtime did not resume")
        self._used = True
        return epoch
