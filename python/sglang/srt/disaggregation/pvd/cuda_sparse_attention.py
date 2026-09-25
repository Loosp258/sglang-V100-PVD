"""Bounded explicit-tensor CUDA attention baseline, NOT a serving backend.

One Decode token, TP1 full causal MHA/GQA, post-RoPE Q/K, no dropout/logit cap.
Prompt is read through a rank participant; generated KV stays caller-owned.
Only explicit scratch tensors are budgeted here, not CUDA/cuBLAS context or
allocator caches. This is a synchronous correctness baseline, not a latency or
total GPU-memory bound. Model pool/forward integration is deliberately separate.
"""

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass

import torch
from sglang.srt.disaggregation.pvd.cuda_rank_install import CUDARankInstallParticipant
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferBudget,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttentionBuffers:
    """Guarded inputs/output. Owner must forbid pool-row reuse until unpinned."""

    q: torch.Tensor
    generated_k: torch.Tensor
    generated_v: torch.Tensor
    output: torch.Tensor
    # None means compact generated KV. Otherwise K/V are the model pool and
    # these immutable host indices select only this request's generated rows.
    # The owner must pin those rows until execution completes. No device-sized
    # gather of the generated context is performed.
    generated_rows: tuple[int, ...] | None = None


def scratch_elements(chunk_tokens, head_dim):
    if any(type(n) is not int or n <= 0 for n in (chunk_tokens, head_dim)):
        raise SparsePayloadError("positive chunk size and head dimension required")
    return 2 * chunk_tokens * head_dim + chunk_tokens + 3 * head_dim + 6


def _stream_attention(groups, buffers, mapping, layer, scale, scratch, chunk, dim):
    """Device-neutral math core; callers validate and fence resource ownership.

    Reuse a fixed FP32 tile for online softmax; never concatenate Prompt with
    generated KV or allocate a score vector proportional to total context.
    No positions are recomputed or RoPE applied. Inputs contain only past/current
    positions, as checked by the public one-token Decode wrapper.
    """
    offset = 0

    def take(n):
        nonlocal offset
        result = scratch[offset : offset + n]
        offset += n
        return result

    keys, values = (
        take(chunk * dim).view(chunk, dim),
        take(chunk * dim).view(chunk, dim),
    )
    scores, query, accum, contribution = take(chunk), take(dim), take(dim), take(dim)
    maximum, next_max, tile_max, alpha, denominator, tile_sum = (
        take(1).view(()) for _ in range(6)
    )
    with torch.inference_mode():
        for head in range(mapping.num_query_heads):
            kv_head = mapping.kv_head_for(head)
            prompt = groups[(layer, kv_head)][1]
            query.copy_(buffers.q[head])
            maximum.fill_(-float("inf"))
            denominator.zero_()
            accum.zero_()
            for source_k, source_v, rows in (
                (prompt[0], prompt[1], None),
                (
                    buffers.generated_k[:, kv_head],
                    buffers.generated_v[:, kv_head],
                    buffers.generated_rows,
                ),
            ):
                length = source_k.shape[0] if rows is None else len(rows)
                for start in range(0, length, chunk):
                    count = min(chunk, length - start)
                    if rows is None:
                        keys[:count].copy_(source_k[start : start + count])
                        values[:count].copy_(source_v[start : start + count])
                    else:
                        # Baseline avoids a dtype-sized index_select temporary:
                        # copy selected rows directly into existing FP32 tiles.
                        for tile_row in range(count):
                            source_row = rows[start + tile_row]
                            keys[tile_row].copy_(source_k[source_row])
                            values[tile_row].copy_(source_v[source_row])
                    torch.mv(keys[:count], query, out=scores[:count])
                    scores[:count].mul_(scale)
                    torch.max(scores[:count], out=tile_max)
                    torch.maximum(maximum, tile_max, out=next_max)
                    torch.sub(maximum, next_max, out=alpha)
                    alpha.exp_()
                    scores[:count].sub_(next_max).exp_()
                    torch.sum(scores[:count], dim=0, out=tile_sum)
                    torch.mv(values[:count].t(), scores[:count], out=contribution)
                    accum.mul_(alpha).add_(contribution)
                    denominator.mul_(alpha).add_(tile_sum)
                    maximum.copy_(next_max)
            accum.div_(denominator)
            buffers.output[head].copy_(accum)


def sdpa_workspace_bytes(max_tokens, kv_heads, query_heads, head_dim, dtype):
    """Conservative reservation, including a bound for opaque SDPA temporaries.

    This is not a proof of the CUDA library's actual peak allocation. The
    opt-in path additionally requires real-device peak-memory validation.
    """
    if any(
        type(n) is not int or n <= 0
        for n in (max_tokens, kv_heads, query_heads, head_dim)
    ):
        raise SparsePayloadError("positive bounded SDPA dimensions required")
    if (
        query_heads < kv_heads
        or query_heads % kv_heads
        or dtype not in (torch.float16, torch.float32)
    ):
        raise SparsePayloadError("unsupported bounded SDPA head layout or dtype")
    element_bytes = torch.empty((), dtype=dtype).element_size()
    resident = 2 * kv_heads * max_tokens * head_dim * element_bytes
    # Worst-case math fallback repeats K/V for every Q head. One-token scores,
    # output and row indices are counted too; backend internals remain opaque.
    transient = (
        2 * query_heads * max_tokens * head_dim * element_bytes
        + 4 * query_heads * max_tokens * 4
        + query_heads * head_dim * element_bytes
        + max_tokens * 8
    )
    return resident + transient


def _sdpa_attention(groups, buffers, mapping, layer, scale, keys, values, max_tokens):
    """Use only the selected Prompt workset and this request's generated rows."""
    rows = buffers.generated_rows
    generated_count = buffers.generated_k.shape[0] if rows is None else len(rows)
    if any(
        groups[(layer, head)][1].shape[1] + generated_count > max_tokens
        for head in range(mapping.total_kv_heads)
    ):
        raise SparsePayloadError("bounded SDPA workset exceeds max_sequence_tokens")
    row_ids = (
        None
        if rows is None
        else torch.tensor(rows, dtype=torch.long, device=buffers.q.device)
    )
    with torch.inference_mode():
        for kv_head in range(mapping.total_kv_heads):
            prompt = groups[(layer, kv_head)][1]
            prompt_count = prompt.shape[1]
            total = prompt_count + generated_count
            key = keys[kv_head, :total]
            value = values[kv_head, :total]
            key[:prompt_count].copy_(prompt[0])
            value[:prompt_count].copy_(prompt[1])
            for source, dest in (
                (buffers.generated_k[:, kv_head], key[prompt_count:]),
                (buffers.generated_v[:, kv_head], value[prompt_count:]),
            ):
                if row_ids is None:
                    dest.copy_(source)
                else:
                    torch.index_select(source, 0, row_ids, out=dest)
            first = kv_head * mapping.group_size
            last = first + mapping.group_size
            result = torch.nn.functional.scaled_dot_product_attention(
                buffers.q[first:last][None, :, None, :],
                key[None, None, :, :],
                value[None, None, :, :],
                scale=scale,
                enable_gqa=True,
            )
            buffers.output[first:last].copy_(result[0, :, 0, :])


class CUDASparseAttentionWorkspace:
    def __init__(
        self,
        *,
        device,
        dtype,
        head_dim,
        chunk_tokens,
        budget,
        attention_impl="online",
        max_sequence_tokens=None,
        total_kv_heads=None,
        num_query_heads=None,
    ):
        count = scratch_elements(chunk_tokens, head_dim)
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise SparsePayloadError("explicit indexed CUDA attention device required")
        if dtype not in (torch.float16, torch.float32):
            raise SparsePayloadError("attention baseline supports FP16/FP32 storage")
        if not isinstance(budget, TransferBudget) or not torch.cuda.is_available():
            raise SparsePayloadError("CUDA and explicit attention budget required")
        self.dtype, self.head_dim, self.chunk_tokens = dtype, head_dim, chunk_tokens
        if attention_impl not in ("online", "sdpa_bounded", "triton_grouped"):
            raise SparsePayloadError("unknown attention implementation")
        if attention_impl == "triton_grouped":
            if (
                dtype != torch.float16
                or head_dim not in (64, 128)
                or chunk_tokens not in (8, 16, 32, 64, 128)
                or type(total_kv_heads) is not int
                or total_kv_heads <= 0
                or type(num_query_heads) is not int
                or num_query_heads < total_kv_heads
                or num_query_heads % total_kv_heads
                or torch.cuda.get_device_capability(self.device) != (7, 0)
            ):
                raise SparsePayloadError(
                    "triton_grouped requires SM70, FP16, supported tile and GQA heads"
                )
            from sglang.srt.disaggregation.pvd.triton_sparse_attention import (
                one_token_gqa_grouped,
            )

            self._triton_execute = one_token_gqa_grouped
        self.attention_impl = attention_impl
        self.max_sequence_tokens = max_sequence_tokens
        self.total_kv_heads = total_kv_heads
        self.num_query_heads = num_query_heads
        self._budget, self._owner = budget, f"cuda-attention:{uuid.uuid4().hex}"
        self._sdpa_owner = f"{self._owner}:sdpa"
        self._sdpa_keys = self._sdpa_values = None
        self._timed_first_layers = set()
        self._thread = threading.get_ident()
        self._closed, self._active, self._quarantine, self._held = (
            False,
            False,
            None,
            None,
        )
        budget.reserve(self._owner, count * 4, 1)
        try:
            self._scratch = torch.empty(count, dtype=torch.float32, device=self.device)
        except BaseException:
            budget.release(self._owner)
            raise
        if attention_impl == "sdpa_bounded":
            try:
                if type(max_sequence_tokens) is not int or max_sequence_tokens > 256:
                    raise SparsePayloadError(
                        "bounded SDPA requires max_sequence_tokens <= 256"
                    )
                reservation = sdpa_workspace_bytes(
                    max_sequence_tokens,
                    total_kv_heads,
                    num_query_heads,
                    head_dim,
                    dtype,
                )
                budget.reserve(self._sdpa_owner, reservation, 1)
                shape = (total_kv_heads, max_sequence_tokens, head_dim)
                self._sdpa_keys = torch.empty(shape, dtype=dtype, device=self.device)
                self._sdpa_values = torch.empty(shape, dtype=dtype, device=self.device)
            except BaseException:
                self._sdpa_keys = self._sdpa_values = self._scratch = None
                budget.release(self._sdpa_owner)
                budget.release(self._owner)
                raise

    def _check(self):
        if threading.get_ident() != self._thread:
            raise SparsePayloadError("attention workspace requires its owner thread")
        if self._closed or self._quarantine is not None:
            raise SparsePayloadError("attention workspace closed or quarantined")

    def _synchronize(self):
        torch.cuda.synchronize(self.device)

    def execute(self, participant, *, decode_tokens, layer, mapping, resources, scale):
        self._check()
        if self._active:
            raise SparsePayloadError("attention workspace cannot run concurrently")
        if (
            not isinstance(participant, CUDARankInstallParticipant)
            or not isinstance(mapping, QueryHeadMapping)
            or not isinstance(resources, ResourceGuard)
            or type(layer) is not int
            or layer < 0
            or type(decode_tokens) is not int
            or decode_tokens < 0
            or type(scale) not in (float, int)
            or not math.isfinite(scale)
            or scale <= 0
        ):
            raise SparsePayloadError(
                "explicit participant, mapping, buffers and scale required"
            )
        started = time.perf_counter()
        triton_owner = triton_view = triton_rows = None
        pin = f"{self._owner}:execution"
        resources.pin(pin)
        self._active = True
        try:
            buffers = resources.value
            bank = participant._bank
            if not isinstance(buffers, AttentionBuffers):
                raise SparsePayloadError("guard must own AttentionBuffers")
            if (
                bank.device != self.device
                or bank.dtype != self.dtype
                or bank.head_dim != self.head_dim
            ):
                raise SparsePayloadError(
                    "bank and attention device/dtype/dimension mismatch"
                )
            if self.attention_impl in ("sdpa_bounded", "triton_grouped") and (
                mapping.total_kv_heads != self.total_kv_heads
                or mapping.num_query_heads != self.num_query_heads
            ):
                raise SparsePayloadError("attention head mapping changed")
            pool_rows = decode_tokens + 1
            if buffers.generated_rows is not None:
                rows = buffers.generated_rows
                if (
                    type(rows) is not tuple
                    or len(rows) != decode_tokens + 1
                    or any(type(r) is not int or r <= 0 for r in rows)
                    or len(set(rows)) != len(rows)
                    or not isinstance(buffers.generated_k, torch.Tensor)
                    or buffers.generated_k.ndim != 3
                    or max(rows) >= buffers.generated_k.shape[0]
                ):
                    raise SparsePayloadError("invalid generated pool row mapping")
                pool_rows = buffers.generated_k.shape[0]
            shape = (pool_rows, mapping.total_kv_heads, self.head_dim)
            tensors = (
                buffers.q,
                buffers.generated_k,
                buffers.generated_v,
                buffers.output,
            )
            expected = (
                (mapping.num_query_heads, self.head_dim),
                shape,
                shape,
                (mapping.num_query_heads, self.head_dim),
            )
            for tensor, wanted in zip(tensors, expected, strict=True):
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.device != self.device
                    or tensor.dtype != self.dtype
                    or tuple(tensor.shape) != wanted
                    or not tensor.is_contiguous()
                ):
                    raise SparsePayloadError(
                        "attention tensor shape/device/dtype/contiguity mismatch"
                    )
            output_storage = buffers.output.untyped_storage().data_ptr()
            if any(
                output_storage == t.untyped_storage().data_ptr()
                for t in (*tensors[:3], self._scratch)
            ):
                raise SparsePayloadError("attention output must own distinct storage")
            with participant.read(decode_tokens) as groups:
                for head in range(mapping.total_kv_heads):
                    if (layer, head) not in groups:
                        raise SparsePayloadError(
                            "all TP1 layer/KV heads must be installed"
                        )
                    spec, prompt = groups[(layer, head)]
                    if any(p >= bank.prompt_tokens for p in spec.token_ids):
                        raise SparsePayloadError(
                            "Prompt positions cannot include generated KV"
                        )
                    if output_storage == prompt.untyped_storage().data_ptr():
                        raise SparsePayloadError(
                            "attention output aliases Prompt storage"
                        )
                # The lease covers all operations and failure draining, not only
                # Python's return from an asynchronous CUDA tensor operation.
                try:
                    if self.attention_impl == "triton_grouped":
                        if not math.isclose(
                            scale,
                            self.head_dim**-0.5,
                            rel_tol=1e-8,
                            abs_tol=1e-12,
                        ):
                            raise SparsePayloadError(
                                "triton_grouped requires standard attention scale"
                            )
                        from sglang.srt.disaggregation.pvd.triton_sparse_attention import (
                            PromptPointerView,
                        )

                        owner = f"{self._owner}:tables:{uuid.uuid4().hex}"
                        # Two tiny GPU tables plus the generated-row index. Charge
                        # before allocation; release only after the reader fence.
                        table_bytes = 12 * mapping.total_kv_heads + 8 * (
                            decode_tokens + 1
                        )
                        self._budget.reserve(owner, table_bytes, 1)
                        triton_owner = owner
                        triton_view = PromptPointerView.from_groups(
                            groups,
                            layer=layer,
                            kv_heads=mapping.total_kv_heads,
                            device=self.device,
                        )
                        row_ids = (
                            buffers.generated_rows
                            if buffers.generated_rows is not None
                            else tuple(range(decode_tokens + 1))
                        )
                        triton_rows = torch.tensor(
                            row_ids, dtype=torch.int64, device=self.device
                        )
                        self._triton_execute(
                            buffers.q,
                            triton_view,
                            buffers.generated_k,
                            buffers.generated_v,
                            triton_rows,
                            buffers.output,
                            block_tokens=self.chunk_tokens,
                        )
                    elif self.attention_impl == "sdpa_bounded":
                        _sdpa_attention(
                            groups,
                            buffers,
                            mapping,
                            layer,
                            scale,
                            self._sdpa_keys,
                            self._sdpa_values,
                            self.max_sequence_tokens,
                        )
                    else:
                        _stream_attention(
                            groups,
                            buffers,
                            mapping,
                            layer,
                            scale,
                            self._scratch,
                            self.chunk_tokens,
                            self.head_dim,
                        )
                finally:
                    try:
                        self._synchronize()
                    except BaseException:
                        self._quarantine = "attention completion unknown"
                        self._held = (
                            resources,
                            pin,
                            participant,
                            triton_view,
                            triton_rows,
                        )
                        raise
        finally:
            try:
                # A later bank-reader drain can fail even after this workspace's
                # compute drain succeeded. That uncertainty must retain the
                # generated pool/output owner too, not just the Prompt bank.
                if participant._bank.snapshot()["quarantine"] is not None:
                    self._quarantine = "Prompt reader completion unknown"
                    self._held = (
                        resources,
                        pin,
                        participant,
                        triton_view,
                        triton_rows,
                    )
                if self._quarantine is None:
                    try:
                        resources.unpin(pin)
                    except BaseException:
                        self._quarantine = "attention resource release unknown"
                        self._held = (
                            resources,
                            pin,
                            participant,
                            triton_view,
                            triton_rows,
                        )
                        raise
                    if triton_owner is not None:
                        self._budget.release(triton_owner)
            finally:
                self._active = False
        if layer not in self._timed_first_layers:
            self._timed_first_layers.add(layer)
            logger.info(
                "PVD first sparse attention execution: layer=%d impl=%s "
                "decode_tokens=%d elapsed_seconds=%.6f",
                layer,
                self.attention_impl,
                decode_tokens,
                time.perf_counter() - started,
            )

    def snapshot(self):
        if threading.get_ident() != self._thread:
            raise SparsePayloadError("attention workspace requires its owner thread")
        return {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "closed": self._closed,
            "active": self._active,
            "quarantine": self._quarantine,
            "resources_held": self._held is not None,
            "explicit_scratch_bytes": 0
            if self._scratch is None
            else self._scratch.numel() * 4,
            "completion_policy": "device_synchronize",
            "attention_impl": self.attention_impl,
            "sdpa_reserved_bytes": (
                sdpa_workspace_bytes(
                    self.max_sequence_tokens,
                    self.total_kv_heads,
                    self.num_query_heads,
                    self.head_dim,
                    self.dtype,
                )
                if self.attention_impl == "sdpa_bounded" and self._sdpa_keys is not None
                else 0
            ),
        }

    def close(self):
        if self._closed:
            return
        self._check()
        if self._active:
            raise SparsePayloadError("attention execution still owns workspace")
        # Every execution drains before returning. Unknown completion has
        # already quarantined this object; close cannot override that state.
        self._scratch = None
        self._sdpa_keys = self._sdpa_values = None
        self._budget.release(self._sdpa_owner)
        self._budget.release(self._owner)
        self._closed = True
