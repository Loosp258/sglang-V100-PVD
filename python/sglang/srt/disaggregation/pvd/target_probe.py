"""Offline CPU/TP1 Llama target-Q reference path, NOT wired into serving.

Reuses target weights, never swaps target pools/backend or installs hooks.
The caller must keep the target quiescent: ForwardContext is process-global,
so this reference is main-thread-only and makes no concurrency claim.
"""

from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from sglang.srt.disaggregation.pvd.prediction import (
    CommittedPrefix,
    DraftPrediction,
    PredictionConfigError,
    ProbeConfig,
    QueryVectors,
    TargetProbe,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


class PostRopeQueryCapture:
    """Batch-owned collector; no model/global state and no borrowed Q storage."""

    def __init__(
        self,
        config: ProbeConfig,
        prefix: CommittedPrefix,
        predicted: int,
        *,
        query_heads: int,
        head_dim: int,
    ):
        for value in (predicted, query_heads, head_dim):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PredictionConfigError(
                    "prediction/head dimensions must be positive integers"
                )
        if config.head_start + config.head_count > query_heads:
            raise PredictionConfigError("probe Q heads exceed the target Q-head count")
        self.config = config
        self.prefix = prefix
        self.positions = tuple(
            range(len(prefix.tokens), len(prefix.tokens) + predicted)
        )
        self.query_heads = query_heads
        self.head_dim = head_dim
        self.version = uuid.uuid4().hex
        self._queries = {}
        self._closed = False

    def capture(self, layer: int, positions: torch.Tensor, q: torch.Tensor) -> None:
        if self._closed:
            raise PredictionConfigError("probe capture is closed")
        if layer not in self.config.layers:
            return
        if layer in self._queries:
            raise PredictionConfigError("probe layer was captured twice")
        expected = tuple(range(len(self.prefix.tokens) + len(self.positions)))
        if tuple(positions.tolist()) != expected:
            raise PredictionConfigError(
                "probe positions must cover the complete prefix and prediction"
            )
        if q.shape != (len(expected), self.query_heads * self.head_dim):
            raise PredictionConfigError("probe Q shape does not match target heads")
        selected = (
            q.reshape(len(expected), self.query_heads, self.head_dim)[
                len(self.prefix.tokens) :,
                self.config.head_start : self.config.head_start
                + self.config.head_count,
            ]
            .detach()
            .clone()
        )
        if not torch.isfinite(selected).all():
            raise PredictionConfigError("probe Q contains non-finite values")
        self._queries[layer] = QueryVectors(
            vector_space=self.config.target_model_id,
            version=self.version,
            layer=layer,
            head_start=self.config.head_start,
            head_count=self.config.head_count,
            positions=self.positions,
            valid_length=len(self.positions),
            vectors=selected,
            prefix_version=self.prefix.version,
            positional_encoding=ROPE_APPLIED,
            request_id=self.prefix.request_id,
        )

    def finish(self) -> tuple[QueryVectors, ...]:
        if self._closed or set(self._queries) != set(self.config.layers):
            raise PredictionConfigError("probe did not capture every requested layer")
        return tuple(self._queries[layer] for layer in self.config.layers)

    def close(self) -> None:
        self._closed = True
        self._queries.clear()


class OfflineLlamaTargetProbe(TargetProbe):
    """Full-prefix recomputation with private CPU pools and existing weights.

    ``target_model_id`` is an explicit deployment identity binding, not a hash
    inferred from weights. ``transient_bytes_bound`` is caller-declared extra
    headroom (backend/activation/allocator temporaries), not a measured bound.
    An entire branch is one capture. Query tensors are scoped to that branch.
    """

    def __init__(
        self,
        runner,
        config: ProbeConfig,
        *,
        target_model_id: str,
        max_tokens: int,
        max_predict_tokens: int,
        transient_bytes_bound: int,
        budget: TransferBudget,
    ):
        from sglang.srt.models.llama import LlamaForCausalLM

        if type(runner.model) is not LlamaForCausalLM:
            raise PredictionConfigError(
                "offline probe supports the exact LlamaForCausalLM class only"
            )
        if runner.device != "cpu" or runner.tp_size != 1 or runner.pp_size != 1:
            raise PredictionConfigError("offline probe requires CPU, TP1 and PP1")
        if (
            runner.server_args.attention_backend != "torch_native"
            or runner.server_args.enable_dp_attention
        ):
            raise PredictionConfigError(
                "offline probe requires non-DP torch_native attention"
            )
        if runner.attn_cp_size != 1:
            raise PredictionConfigError("context-parallel targets are not supported")
        if (
            runner.model.quant_config is not None
            or runner.server_args.speculative_algorithm is not None
        ):
            raise PredictionConfigError(
                "quantized/speculative targets are not supported"
            )
        if config.target_model_id != target_model_id or not target_model_id:
            raise PredictionConfigError(
                "probe identity differs from bound target identity"
            )
        for name, value in (
            ("max_tokens", max_tokens),
            ("max_predict_tokens", max_predict_tokens),
            ("transient_bytes_bound", transient_bytes_bound),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PredictionConfigError(f"{name} must be a positive integer")
        if (
            max_tokens > runner.model_config.context_len
            or max_predict_tokens >= max_tokens
        ):
            raise PredictionConfigError("probe bounds exceed the target context")
        if any(
            p.device.type != "cpu" or p.dtype != torch.float32
            for p in runner.model.parameters()
        ):
            raise PredictionConfigError(
                "offline probe requires CPU FP32 target weights"
            )
        hf = runner.model.config
        if any(layer >= hf.num_hidden_layers for layer in config.layers):
            raise PredictionConfigError("probe layer exceeds the target layer count")
        if config.head_start + config.head_count > hf.num_attention_heads:
            raise PredictionConfigError("probe heads exceed the target Q-head count")
        self.model = runner.model  # same object/weights, never another ModelRunner
        self.config = config
        self.max_tokens = max_tokens
        self.max_predict_tokens = max_predict_tokens
        self.budget = budget
        self.head_dim = runner.model_config.head_dim
        self.query_heads = hf.num_attention_heads
        self.kv_heads = hf.num_key_value_heads
        self.layers = hf.num_hidden_layers
        self.vocab_size = hf.vocab_size
        kv = (max_tokens + 1) * self.layers * self.kv_heads * self.head_dim * 2 * 4
        mapping = 2 * max_tokens * 4
        queries = (
            max_predict_tokens
            * len(config.layers)
            * config.head_count
            * self.head_dim
            * 4
        )
        self.reservation_bytes = kv + mapping + queries + transient_bytes_bound
        self._active = False
        self._used = False
        self._state = None
        self._quarantined = False
        self._private_state = None

    @staticmethod
    def _require_main_thread():
        if threading.current_thread() is not threading.main_thread():
            raise PredictionConfigError(
                "offline probe is main-thread-only; target must be quiescent"
            )

    @contextmanager
    def branch(self):
        self._require_main_thread()
        if self._active or self._quarantined:
            raise PredictionConfigError(
                "probe branches cannot nest or reuse a quarantined probe"
            )
        owner = f"pvd-target-probe:{uuid.uuid4().hex}"
        self.budget.reserve(owner, self.reservation_bytes, 1)
        self._active, self._used = True, False
        try:
            yield self
        finally:
            # capture() owns allocation/release and does not return before
            # releasing private rows. Branch holds the query-copy reservation.
            if self._state is not None and not self._quarantined:
                self._state.close()
                self._state = None
            self._active = False
            if not self._quarantined:
                self.budget.release(owner)

    def capture(self, prefix: CommittedPrefix, prediction: DraftPrediction):
        self._require_main_thread()
        if not self._active or self._used:
            raise PredictionConfigError("capture requires an unused probe branch")
        if (prediction.request_id, prediction.prefix_version) != (
            prefix.request_id,
            prefix.version,
        ):
            raise PredictionConfigError("prediction belongs to another request/prefix")
        tokens = prefix.tokens + prediction.tokens
        if (
            not prefix.tokens
            or len(tokens) > self.max_tokens
            or len(prediction.tokens) > self.max_predict_tokens
        ):
            raise PredictionConfigError(
                "probe input exceeds admitted bounds or has no prefix"
            )
        if any(t < 0 or t >= self.vocab_size for t in tokens):
            raise PredictionConfigError("probe token is outside the target vocabulary")
        self._used = True
        self._state = PostRopeQueryCapture(
            self.config,
            prefix,
            len(prediction.tokens),
            query_heads=self.query_heads,
            head_dim=self.head_dim,
        )
        return self._forward(tokens, self._state)

    def _forward(self, tokens, capture):
        from sglang.srt.compilation.piecewise_context_manager import get_forward_context
        from sglang.srt.disaggregation.pvd.draft_forward_adapter import (
            DraftForwardAdapter,
            PrivatePoolAllocator,
        )
        from sglang.srt.disaggregation.pvd.draft_runner_sglang import DraftForwardInputs
        from sglang.srt.layers.attention.torch_native_backend import (
            TorchNativeAttnBackend,
        )
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        if get_forward_context() is not None:
            raise PredictionConfigError(
                "probe cannot execute inside a piecewise graph context"
            )

        requests = ReqToTokenPool(1, self.max_tokens, "cpu", False)
        pool = MHATokenToKVPool(
            self.max_tokens,
            1,
            torch.float32,
            self.kv_heads,
            self.head_dim,
            self.layers,
            "cpu",
            False,
        )
        kv = TokenToKVPoolAllocator(self.max_tokens, torch.float32, "cpu", pool, False)
        allocator = PrivatePoolAllocator(requests, kv)
        backend = TorchNativeAttnBackend(
            SimpleNamespace(
                device="cpu",
                req_to_token_pool=requests,
                token_to_kv_pool=pool,
            )
        )
        self._private_state = (allocator, backend)
        builder = DraftForwardAdapter(
            None,
            architecture="LlamaForCausalLM",
            attention_backend="torch_native",
            bytes_per_token=self.layers * self.kv_heads * self.head_dim * 2 * 4,
            device="cpu",
        )
        slot, rows = allocator.alloc_request(), []
        try:
            rows = allocator.alloc_kv(len(tokens))
            allocator.write_mapping(slot, 0, rows)
            batch = builder.build_forward_batch(
                DraftForwardInputs(
                    "extend",
                    tokens,
                    tuple(range(len(tokens))),
                    (len(tokens),),
                    (slot,),
                    tuple(rows),
                    (0,),
                    (len(tokens),),
                )
            )
            batch.pvd_query_capture = capture
            backend.init_forward_metadata(batch)
            with (
                torch.inference_mode(),
                forward_context(ForwardContext(attn_backend=backend)),
            ):
                # Backbone only: no sampler, output commit, or LM-head logits.
                self.model.model(batch.input_ids, batch.positions, batch)
            return capture.finish()
        finally:
            try:
                allocator.clear_mapping(slot)
                allocator.free_kv(rows)
                allocator.free_request(slot)
            except BaseException:
                # Do not refund or reuse possibly-live rows after failed
                # cleanup. Keep the private state and reservation for diagnosis.
                self._quarantined = True
                raise
            else:
                self._private_state = None
