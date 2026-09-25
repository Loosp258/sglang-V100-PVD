"""Explicit TP1 CUDA model-pool adapter, not automatic Scheduler activation.

The whole-forward scope owns the target execution lock, allocator lease and
Prompt readers. The caller must use that lock for EVERY target/probe forward
and tie the pool guard to actual allocator retirement (not a no-op callback).
Device synchronization is a correctness baseline, not compute/network overlap.
"""

import logging
import math
import os
import threading
import time
import traceback
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch
from sglang.srt.disaggregation.pvd.cuda_rank_install import CUDARankInstallParticipant
from sglang.srt.disaggregation.pvd.cuda_sparse_attention import (
    AttentionBuffers,
    CUDASparseAttentionWorkspace,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.rank_install_wire import RankInstallExchange
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferBudget,
)

logger = logging.getLogger(__name__)


class _ColdModelStageProfile:
    """Opt-in, first-forward CUDA-fenced timings; no tensor data is recorded."""

    def __init__(self, device):
        self.device = device
        self._last = time.perf_counter()

    def mark(self, layer, stage):
        if layer not in (0, 1):
            return
        torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        logger.info(
            "PVD cold target stage: layer=%d stage=%s elapsed_seconds=%.6f",
            layer,
            stage,
            now - self._last,
        )
        self._last = now


@dataclass(frozen=True)
class CUDAModelPools:
    req_pool: object
    kv_pool: object


@dataclass(frozen=True)
class CUDADecodeBinding:
    slot: int
    decode_tokens: int
    participant: CUDARankInstallParticipant
    exchange: RankInstallExchange


class CUDAModelSparseConsumer:
    def __init__(
        self,
        req_pool,
        kv_pool,
        *,
        layers,
        mapping,
        workspace,
        execution_lock,
        output_budget,
        max_batch_size,
    ):
        if (
            not isinstance(workspace, CUDASparseAttentionWorkspace)
            or not isinstance(mapping, QueryHeadMapping)
            or not isinstance(output_budget, TransferBudget)
            or type(max_batch_size) is not int
            or max_batch_size <= 0
            or not all(
                callable(getattr(execution_lock, n, None))
                for n in ("acquire", "release")
            )
        ):
            raise SparsePayloadError(
                "explicit CUDA workspace, lock and output budget required"
            )
        self.req_pool, self.kv_pool = req_pool, kv_pool
        self.layers, self.mapping = tuple(layers), mapping
        self.workspace, self.device, self.dtype = (
            workspace,
            workspace.device,
            workspace.dtype,
        )
        if (
            not self.layers
            or len(set(self.layers)) != len(self.layers)
            or any(type(n) is not int or n < 0 for n in self.layers)
        ):
            raise SparsePayloadError("explicit unique layer ids required")
        table = req_pool.req_to_token
        if (
            not isinstance(table, torch.Tensor)
            or table.ndim != 2
            or table.device != self.device
            or table.dtype not in (torch.int32, torch.int64)
        ):
            raise SparsePayloadError(
                "integral request map on the workspace device required"
            )
        self._lock, self._budget, self._max_batch = (
            execution_lock,
            output_budget,
            max_batch_size,
        )
        self._owner = "cuda-model-attention:" + uuid.uuid4().hex
        self._packed_owner = self._owner + ":packed-qkv"
        self._bound, self._outputs, self._pending = None, [], None
        self._packed_inputs, self._packed_charged = [], False
        self._seen, self._quarantine, self._held = set(), None, None
        self._active = False

    def _check(self):
        if threading.current_thread() is not threading.main_thread():
            raise SparsePayloadError("CUDA model adapter requires the main thread")
        if self._quarantine is not None:
            raise SparsePayloadError("CUDA model adapter is quarantined")
        self.workspace._check()

    def _synchronize(self):
        torch.cuda.synchronize(self.device)

    @contextmanager
    def bind(self, bindings, *, pool_owner):
        """Enclose the entire model forward, not just attention-layer calls."""
        self._check()
        if self._active:
            raise SparsePayloadError("CUDA model forward scopes cannot nest")
        bindings = tuple(bindings)
        if not 0 < len(bindings) <= self._max_batch:
            raise SparsePayloadError("explicit bounded request bindings required")
        if not isinstance(pool_owner, ResourceGuard):
            raise SparsePayloadError("an allocator-backed model pool lease is required")
        if not self._lock.acquire(blocking=False):
            raise SparsePayloadError("target execution is busy")
        self._active = True
        self._packed_inputs.clear()
        self._packed_charged = False
        stack, pinned, charged = ExitStack(), False, False
        bound, identities = {}, set()
        failure = None
        try:
            stack.enter_context(torch.cuda.device(self.device))
            pool_owner.pin(self._owner)
            pinned = True
            pools = pool_owner.value
            if (
                not isinstance(pools, CUDAModelPools)
                or pools.req_pool is not self.req_pool
                or pools.kv_pool is not self.kv_pool
            ):
                raise SparsePayloadError("lease must own these exact model pools")
            expected = {
                (layer, head)
                for layer in self.layers
                for head in range(self.mapping.total_kv_heads)
            }
            for binding in bindings:
                if not isinstance(binding, CUDADecodeBinding):
                    raise SparsePayloadError("explicit CUDA decode binding required")
                peer, exchange = binding.participant, binding.exchange
                if (
                    not isinstance(peer, CUDARankInstallParticipant)
                    or not isinstance(exchange, RankInstallExchange)
                    or set(exchange.peer_epochs) != {peer.rank}
                    or exchange.peer_epochs[peer.rank] != peer.peer_epoch
                    or type(binding.slot) is not int
                    or not 0 < binding.slot < self.req_pool.req_to_token.shape[0]
                    or binding.slot in bound
                    or type(binding.decode_tokens) is not int
                    or binding.decode_tokens < 0
                ):
                    raise SparsePayloadError(
                        "TP1 binding, slot or peer identity mismatch"
                    )
                bank = peer._bank
                if (
                    bank.identity[:3] != exchange.coordinator.identity
                    or bank.identity[:3] in identities
                    or bank.device != self.device
                    or bank.dtype != self.dtype
                    or bank.head_dim != self.workspace.head_dim
                    or bank.expected_groups != expected
                    or not exchange.can_decode(binding.decode_tokens)
                ):
                    raise SparsePayloadError(
                        "bank identity, layout or all-rank resume mismatch"
                    )
                installed = exchange.coordinator.snapshot()["completed"]
                groups = stack.enter_context(peer.read(binding.decode_tokens))
                if any(
                    (spec.operation_id, spec.target_tokens)
                    != (installed.operation_id, installed.target_tokens)
                    for spec, _ in groups.values()
                ):
                    raise SparsePayloadError(
                        "bank does not match the agreed installation"
                    )
                identities.add(bank.identity[:3])
                bound[binding.slot] = binding
            # Every layer output lives through the whole model forward. This
            # bound excludes the caller's model activations and native workspace.
            size = (
                len(bindings)
                * len(self.layers)
                * self.mapping.num_query_heads
                * self.workspace.head_dim
                * (torch.finfo(self.dtype).bits // 8)
            )
            self._budget.reserve(self._owner, size, 1)
            charged = True
            self._bound, self._seen = bound, set()
            yield self
            if self._seen != {(slot, layer) for slot in bound for layer in self.layers}:
                raise SparsePayloadError(
                    "forward did not consume every bound request/layer"
                )
            if any(not b.exchange.can_decode(b.decode_tokens) for b in bound.values()):
                raise SparsePayloadError(
                    "request cancelled or installation permission changed during forward"
                )
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                if self.workspace.snapshot()["quarantine"] is not None:
                    raise SparsePayloadError("attention completion remains unknown")
                self._synchronize()
                stack.close()  # Includes Prompt reader completion fences.
                self._pending = None
                self._outputs.clear()
                self._packed_inputs.clear()
                if self._packed_charged:
                    self._budget.release(self._packed_owner)
                    self._packed_charged = False
                if failure is not None:
                    traceback.clear_frames(failure.__traceback__)
                    failure = None
                self._bound = None
                self._seen.clear()
                if pinned:
                    pool_owner.unpin(self._owner)
                if charged:
                    self._budget.release(self._owner)
                self._lock.release()
                self._active = False
            except BaseException:
                self._quarantine = "model forward or lease retirement incomplete"
                self._held = (stack, pool_owner, bound)
                raise

    def _indices(self, tensor, name):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device != self.device
            or tensor.ndim != 1
            or tensor.dtype not in (torch.int32, torch.int64)
        ):
            raise SparsePayloadError(f"device integral {name} required")
        return tensor.tolist()  # Bounded host metadata; no full KV gather.

    def forward_decode(self, q, k, v, layer, batch, save_kv_cache=True):
        self._check()
        if self._bound is None:
            raise SparsePayloadError("whole-forward model pool binding required")
        # Metadata .tolist() can itself synchronize and surface a CUDA failure.
        # Publish inputs before that first operation, not only before KV writes.
        self._pending = (q, k, v, batch)
        if (
            not batch.forward_mode.is_decode()
            or batch.encoder_lens is not None
            or getattr(batch, "spec_info", None) is not None
            or (
                (spec_algorithm := getattr(batch, "spec_algorithm", None)) is not None
                and not (
                    callable(getattr(spec_algorithm, "is_none", None))
                    and spec_algorithm.is_none()
                )
            )
            or getattr(batch, "pvd_query_capture", None) is not None
            or layer.layer_id not in self.layers
            or layer.is_cross_attention
            or getattr(layer.attn_type, "value", layer.attn_type) != "decoder"
            or layer.logit_cap != 0
            or layer.sliding_window_size not in (None, -1)
            or getattr(layer, "quant_method", None) is not None
            or getattr(layer, "use_irope", False)
            or getattr(layer, "k_scale", None) is not None
            or getattr(layer, "v_scale", None) is not None
            or layer.tp_q_head_num != self.mapping.num_query_heads
            or layer.tp_k_head_num != self.mapping.total_kv_heads
            or layer.tp_v_head_num != self.mapping.total_kv_heads
            or layer.qk_head_dim != self.workspace.head_dim
            or layer.v_head_dim != self.workspace.head_dim
            or type(layer.scaling) not in (int, float)
            or not math.isfinite(layer.scaling)
            or layer.scaling <= 0
            or not save_kv_cache
            or k is None
            or v is None
        ):
            raise SparsePayloadError("unsupported CUDA sparse decode configuration")
        slots = self._indices(batch.req_pool_indices, "request slots")
        positions = self._indices(batch.positions, "positions")
        lengths = self._indices(batch.seq_lens, "lengths")
        destinations = self._indices(batch.out_cache_loc, "write locations")
        count, dim = len(slots), self.workspace.head_dim
        if (
            count != len(self._bound)
            or len(set(slots)) != count
            or set(slots) != set(self._bound)
            or not len(positions) == len(lengths) == len(destinations) == count
            or len(set(destinations)) != count
        ):
            raise SparsePayloadError("batch does not match bound requests")
        inputs = []
        for value, heads in (
            (q, self.mapping.num_query_heads),
            (k, self.mapping.total_kv_heads),
            (v, self.mapping.total_kv_heads),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.device != self.device
                or value.dtype != self.dtype
                or tuple(value.shape) not in ((count, heads * dim), (count, heads, dim))
            ):
                raise SparsePayloadError("Q/K/V shape, dtype or device mismatch")
            inputs.append(value.reshape(count, heads, dim))
        if any(not value.is_contiguous() for value in inputs):
            if not self._packed_charged:
                # Qwen's qkv.split is strided across a multi-request batch.
                # Retain every packed layer input until the whole forward's
                # CUDA completion fence, charging the worst-case live copies.
                packed_bytes = (
                    len(self.layers)
                    * count
                    * (self.mapping.num_query_heads + 2 * self.mapping.total_kv_heads)
                    * dim
                    * (torch.finfo(self.dtype).bits // 8)
                )
                self._budget.reserve(self._packed_owner, packed_bytes, 0)
                self._packed_charged = True
            inputs = [value.contiguous() for value in inputs]
            self._packed_inputs.extend(inputs)
        q, k, v = inputs
        keys, values = (
            self.kv_pool.get_key_buffer(layer.layer_id),
            self.kv_pool.get_value_buffer(layer.layer_id),
        )
        for value in (keys, values):
            if (
                value.device != self.device
                or value.dtype != self.dtype
                or value.ndim != 3
                or not value.is_contiguous()
                or tuple(value.shape[1:]) != (self.mapping.total_kv_heads, dim)
            ):
                raise SparsePayloadError("unsupported model KV pool layout")
        if keys.shape != values.shape:
            raise SparsePayloadError("K/V pool extents differ")
        plans, occupied = [], set()
        for slot, pos, length, dest in zip(
            slots, positions, lengths, destinations, strict=True
        ):
            binding = self._bound[slot]
            prompt_tokens = binding.participant._bank.prompt_tokens
            if (
                pos != prompt_tokens + binding.decode_tokens
                or length != pos + 1
                or length > self.req_pool.req_to_token.shape[1]
                or (slot, layer.layer_id) in self._seen
            ):
                raise SparsePayloadError("stale position or repeated layer")
            rows = tuple(
                self.req_pool.req_to_token[slot, prompt_tokens:length].tolist()
            )
            if (
                not rows
                or rows[-1] != dest
                or len(set(rows)) != len(rows)
                or any(r <= 0 or r >= keys.shape[0] for r in rows)
                or occupied.intersection(rows)
            ):
                raise SparsePayloadError("invalid or overlapping generated pool rows")
            occupied.update(rows)
            plans.append((binding, rows))
        output = torch.empty(
            (count, self.mapping.num_query_heads, dim),
            device=self.device,
            dtype=self.dtype,
        )
        self._outputs.append(output)
        # Retain enqueue inputs even if pool writing raises part way through.
        self._pending = (q, k, v, batch, keys, values)
        self.kv_pool.set_kv_buffer(
            layer,
            batch.out_cache_loc,
            k,
            v,
        )
        for index, (binding, rows) in enumerate(plans):
            # bind() keeps the actual allocator lease pinned through this entire
            # forward; this nested guard keeps this layer's tensor handles alive.
            resources = ResourceGuard(
                AttentionBuffers(q[index], keys, values, output[index], rows),
                lambda: None,
            )
            try:
                self.workspace.execute(
                    binding.participant,
                    decode_tokens=binding.decode_tokens,
                    layer=layer.layer_id,
                    mapping=self.mapping,
                    resources=resources,
                    scale=layer.scaling,
                )
            finally:
                resources.request_release()
            self._seen.add((binding.slot, layer.layer_id))
        self._pending = None
        return output.reshape(count, self.mapping.num_query_heads * dim)

    def snapshot(self):
        return {
            "active": self._active,
            "quarantine": self._quarantine,
            "held": self._held is not None,
            "retained_outputs": len(self._outputs),
        }


def make_cuda_sparse_backend(
    runner, *, workspace, execution_lock, output_budget, max_batch_size
):
    """Explicit factory; callers install it only inside a complete owned forward."""
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
    from sglang.srt.models.llama import LlamaForCausalLM
    from sglang.srt.models.qwen2 import Qwen2ForCausalLM

    if (
        type(runner.model) not in (LlamaForCausalLM, Qwen2ForCausalLM)
        or torch.device(runner.device).type != "cuda"
        or runner.tp_size != 1
        or runner.pp_size != 1
        or runner.attn_cp_size != 1
        or type(runner.attn_backend) is not TorchNativeAttnBackend
        or runner.server_args.enable_dp_attention
        or runner.server_args.speculative_algorithm is not None
        or runner.model.quant_config is not None
        or runner.server_args.page_size != 1
        or not runner.server_args.disable_overlap_schedule
        or not runner.server_args.disable_cuda_graph
    ):
        raise SparsePayloadError(
            "CUDA sparse backend requires TP1/PP1 unquantized Llama/Qwen2, native attention, page 1, no graphs/overlap/speculation"
        )
    if not isinstance(workspace, CUDASparseAttentionWorkspace):
        raise SparsePayloadError("explicit CUDA workspace required")
    parameters = tuple(runner.model.parameters())
    if (
        not parameters
        or any(
            p.device != workspace.device or p.dtype != workspace.dtype
            for p in parameters
        )
        or runner.model_config.head_dim != workspace.head_dim
        or getattr(runner, "gpu_id", workspace.device.index) != workspace.device.index
    ):
        raise SparsePayloadError(
            "model weights and sparse workspace placement/layout differ"
        )

    class CUDASparseBackend(TorchNativeAttnBackend):
        def __init__(self):
            super().__init__(runner)
            self._profile_first_decode = (
                os.environ.get("PVD_PROFILE_COLD_STAGES") == "1"
            )
            config = runner.model.config
            self.consumer = CUDAModelSparseConsumer(
                self.req_to_token_pool,
                self.token_to_kv_pool,
                layers=range(config.num_hidden_layers),
                mapping=QueryHeadMapping(
                    config.num_attention_heads, config.num_key_value_heads
                ),
                workspace=workspace,
                execution_lock=execution_lock,
                output_budget=output_budget,
                max_batch_size=max_batch_size,
            )

        def init_forward_metadata(self, forward_batch):
            super().init_forward_metadata(forward_batch)
            if self._profile_first_decode and forward_batch.forward_mode.is_decode():
                self._profile_first_decode = False
                if forward_batch.pvd_cold_profile is not None:
                    raise SparsePayloadError("cold target profile already attached")
                forward_batch.pvd_cold_profile = _ColdModelStageProfile(
                    workspace.device
                )

        def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
            return self.consumer.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache
            )

        def forward_extend(self, *args, **kwargs):
            raise SparsePayloadError("CUDA sparse backend supports bound decode only")

    return CUDASparseBackend()
