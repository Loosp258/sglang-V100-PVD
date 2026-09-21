"""CPU-only current/next Prompt-KV reference. Not a Scheduler/GPU integration.

Single-owner synchronous use. Read scopes model a forward's ownership: install
and close refuse while a reader exists. They are not CUDA events or RDMA fences.
Generated KV is caller-owned and never installed/evicted by this Prompt bank.
"""

import uuid
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.sparse_payload import (
    SparseKVPayload,
    SparsePayloadError,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from torch.nn.functional import scaled_dot_product_attention


@dataclass
class _Bank:
    owner: str
    boundary: int
    groups: dict


@dataclass(frozen=True)
class CPUInstallCandidate:
    """Local staging identity, NOT a transport grant or GPU visibility proof."""

    identity: tuple[str, str, str, str]
    operation_id: str
    target_tokens: int
    staging_id: str


class CPUSparseWorkingSet:
    def __init__(
        self,
        *,
        request_id,
        incarnation,
        entry_transfer_id,
        layout_fingerprint,
        expected_groups,
        prompt_tokens,
        head_dim,
        max_union_tokens,
        budget: TransferBudget,
    ):
        for value in (prompt_tokens, head_dim, max_union_tokens):
            if type(value) is not int or value <= 0:
                raise SparsePayloadError(
                    "working-set bounds must be explicitly positive"
                )
        groups = tuple(expected_groups)
        if (
            not groups
            or len(set(groups)) != len(groups)
            or any(
                len(group) != 2 or any(type(i) is not int or i < 0 for i in group)
                for group in groups
            )
        ):
            raise SparsePayloadError("explicit unique (layer, KV head) groups required")
        self.identity = (request_id, incarnation, entry_transfer_id, layout_fingerprint)
        if any(not isinstance(v, str) or not v.strip() for v in self.identity):
            raise SparsePayloadError("working-set identity must be explicit")
        self.expected_groups = frozenset(groups)
        self.prompt_tokens, self.head_dim = prompt_tokens, head_dim
        self.max_union_tokens, self.budget = max_union_tokens, budget
        self._current = self._next = None
        self._readers = 0
        self._closed = False

    def _open(self):
        if self._closed:
            raise SparsePayloadError("working set closed")

    def stage(self, payloads):
        self._open()
        if self._next is not None:
            raise SparsePayloadError("a next bank is already staged")
        source, contexts, size = {}, set(), 0
        for payload in payloads:
            if not isinstance(payload, SparseKVPayload):
                raise SparsePayloadError("verified scoped sparse payload required")
            spec, tensor = payload.spec, payload.tensor
            if (
                spec.request_id,
                spec.incarnation,
                spec.entry_transfer_id,
                spec.layout_fingerprint,
            ) != self.identity:
                raise SparsePayloadError("working-set request/Entry/layout mismatch")
            key = (spec.layer, spec.kv_head)
            if key in source or key not in self.expected_groups:
                raise SparsePayloadError("unexpected or duplicate working-set group")
            if any(t >= self.prompt_tokens for t in spec.token_ids):
                raise SparsePayloadError(
                    "Prompt bank cannot contain generated positions"
                )
            if (
                tensor.device.type != "cpu"
                or tensor.dtype != torch.float32
                or tuple(tensor.shape) != (2, len(spec.token_ids), self.head_dim)
            ):
                raise SparsePayloadError(
                    "working-set reference requires CPU FP32 paired K/V"
                )
            if self._current is None:
                if spec.target_tokens != 0 or set(spec.token_ids) != set(
                    range(self.prompt_tokens)
                ):
                    raise SparsePayloadError(
                        "initial bank must contain the complete Prompt at boundary zero"
                    )
            elif (
                spec.target_tokens <= self._current.boundary
                or len(spec.token_ids) > self.max_union_tokens
            ):
                raise SparsePayloadError("stale boundary or union capacity exceeded")
            contexts.add((spec.operation_id, spec.target_tokens))
            source[key] = (spec, tensor)
            size += payload.nbytes
        if set(source) != self.expected_groups or len(contexts) != 1:
            raise SparsePayloadError("incomplete bank or mixed refresh operations")
        owner = f"pvd-cpu-working-set:{uuid.uuid4().hex}"
        self.budget.reserve(owner, size, 1)
        copies = {}
        try:
            for key, (spec, tensor) in source.items():
                copies[key] = (spec, tensor.detach().clone())
            self._next = _Bank(owner, next(iter(contexts))[1], copies)
        except BaseException:
            copies.clear()
            self.budget.release(owner)
            raise

    def install_candidate(self):
        self._open()
        if self._next is None:
            raise SparsePayloadError("no staged bank")
        spec = next(iter(self._next.groups.values()))[0]
        return CPUInstallCandidate(
            self.identity, spec.operation_id, self._next.boundary, self._next.owner
        )

    def can_install(self, candidate, committed_tokens):
        """CPU preflight only. Caller must prevent new readers until install."""
        self._open()
        if (
            not isinstance(candidate, CPUInstallCandidate)
            or candidate != self.install_candidate()
        ):
            raise SparsePayloadError("stale or foreign staged bank")
        if (
            type(committed_tokens) is not int
            or committed_tokens != candidate.target_tokens
        ):
            raise SparsePayloadError("install requires the exact prepared boundary")
        return self._readers == 0

    def install(self, committed_tokens, *, candidate=None):
        self._open()
        if candidate is not None and not self.can_install(candidate, committed_tokens):
            raise SparsePayloadError("a current forward still owns the Prompt bank")
        if (
            type(committed_tokens) is not int
            or self._next is None
            or committed_tokens != self._next.boundary
        ):
            raise SparsePayloadError("install requires the exact prepared boundary")
        if self._readers:
            raise SparsePayloadError("a current forward still owns the Prompt bank")
        previous, self._current, self._next = self._current, self._next, None
        if previous is not None:
            previous.groups.clear()
            self.budget.release(previous.owner)

    @contextmanager
    def read(self):
        self._open()
        if self._current is None:
            raise SparsePayloadError("initial Prompt bank is not installed")
        self._readers += 1
        try:
            yield self._current.groups
        finally:
            self._readers -= 1

    def discard_next(self):
        if self._next is not None:
            self._next.groups.clear()
            self.budget.release(self._next.owner)
            self._next = None

    def close(self):
        if self._readers:
            raise SparsePayloadError("cannot close a bank still owned by a forward")
        self.discard_next()
        if self._current is not None:
            self._current.groups.clear()
            self.budget.release(self._current.owner)
            self._current = None
        self._closed = True


def reference_attention(
    groups,
    q,
    *,
    layer,
    query_position,
    mapping: QueryHeadMapping,
    prompt_tokens,
    generated_positions,
    generated_k,
    generated_v,
):
    """CPU FP32 one-token attention over Prompt union + all generated KV.

    Numerical oracle only; this does not implement a serving kernel, bound
    attention workspace or mutate/install generated KV. Absolute positions,
    not compacted row indices, determine the causal mask. Inputs are post-RoPE.
    """
    if (
        type(query_position) is not int
        or query_position < 0
        or q.device.type != "cpu"
        or q.dtype != torch.float32
        or q.ndim != 2
        or q.shape[0] != mapping.num_query_heads
    ):
        raise SparsePayloadError("attention reference requires CPU FP32 [Q heads, dim]")
    if (
        not isinstance(generated_positions, tuple)
        or any(type(p) is not int or p < prompt_tokens for p in generated_positions)
        or tuple(sorted(set(generated_positions))) != generated_positions
    ):
        raise SparsePayloadError(
            "generated positions must be ordered unique post-Prompt positions"
        )
    shape = (len(generated_positions), mapping.total_kv_heads, q.shape[-1])
    if any(
        t.device.type != "cpu" or t.dtype != q.dtype or tuple(t.shape) != shape
        for t in (generated_k, generated_v)
    ):
        raise SparsePayloadError("generated KV shape/device/dtype mismatch")
    outputs = []
    for head in range(mapping.num_query_heads):
        kv_head = mapping.kv_head_for(head)
        if (layer, kv_head) not in groups:
            raise SparsePayloadError("missing Prompt group for query head")
        spec, data = groups[(layer, kv_head)]
        positions = spec.token_ids + generated_positions
        keys = torch.cat((data[0], generated_k[:, kv_head]), dim=0)
        values = torch.cat((data[1], generated_v[:, kv_head]), dim=0)
        mask = torch.tensor([p <= query_position for p in positions], dtype=torch.bool)
        if not mask.any():
            raise SparsePayloadError("query has no causally visible KV")
        outputs.append(
            scaled_dot_product_attention(
                q[head].reshape(1, 1, -1),
                keys.unsqueeze(0),
                values.unsqueeze(0),
                attn_mask=mask.reshape(1, 1, -1),
                dropout_p=0.0,
                is_causal=False,
            ).reshape(-1)
        )
    return torch.stack(outputs)
