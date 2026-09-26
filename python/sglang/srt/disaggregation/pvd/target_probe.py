"""Shared private-pool Llama/Qwen2 target-Q core and offline CPU reference.

Reuses target weights, never swaps target pools/backend or installs hooks.
The caller must keep the target quiescent: ForwardContext is process-global,
so this reference is main-thread-only and makes no concurrency claim. The
separate CUDA subclass adds placement, completion and target-execution locking.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import traceback
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
from sglang.srt.disaggregation.pvd.draft_hf import VocabularySignature
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferBudget,
    TransferCapacityError,
)


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
        committed_positions: tuple[int, ...] | None = None,
        forward_start: int = 0,
    ):
        for value in ((predicted,) if committed_positions is None else ()) + (
            query_heads,
            head_dim,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PredictionConfigError(
                    "prediction/head dimensions must be positive integers"
                )
        if config.head_start + config.head_count > query_heads:
            raise PredictionConfigError("probe Q heads exceed the target Q-head count")
        self.config = config
        self.prefix = prefix
        if committed_positions is None:
            self.positions = tuple(
                range(len(prefix.tokens), len(prefix.tokens) + predicted)
            )
            self.sequence_length = len(prefix.tokens) + predicted
        else:
            if (
                type(predicted) is not int
                or predicted != 0
                or not isinstance(committed_positions, tuple)
                or not committed_positions
                or any(
                    type(p) is not int or not 0 <= p < len(prefix.tokens)
                    for p in committed_positions
                )
                or tuple(sorted(set(committed_positions))) != committed_positions
            ):
                raise PredictionConfigError(
                    "committed capture requires explicit in-prefix positions and no prediction"
                )
            self.positions = committed_positions
            self.sequence_length = len(prefix.tokens)
        self.query_heads = query_heads
        self.head_dim = head_dim
        if type(forward_start) is not int or not 0 <= forward_start <= min(
            self.positions
        ):
            raise PredictionConfigError(
                "probe forward start must precede every captured Q position"
            )
        self.forward_start = forward_start
        self.version = uuid.uuid4().hex
        self._queries = {}
        self._finite_flags = []
        self._bound_positions = None
        self._bound_expected_positions = None
        self._bound_position_version = None
        self._closed = False

    def bind_positions(self, positions: torch.Tensor) -> None:
        """Validate the private forward's positions once, before model execution.

        Every target layer receives this same tensor. Checking its identity and
        mutation version during capture avoids a device-to-host copy per layer.
        Standalone collectors still validate the actual values on each call.
        """
        if self._closed or self._bound_positions is not None or self._queries:
            raise PredictionConfigError("probe positions cannot be rebound")
        if (
            not isinstance(positions, torch.Tensor)
            or positions.ndim != 1
            or positions.dtype != torch.int64
            or positions.numel() != self.sequence_length - self.forward_start
        ):
            raise PredictionConfigError(
                "probe positions must cover the complete prefix and prediction"
            )
        expected = torch.arange(
            self.forward_start,
            self.sequence_length,
            dtype=torch.int64,
            device=positions.device,
        )
        if not torch.equal(positions, expected):
            raise PredictionConfigError(
                "probe positions must cover the complete prefix and prediction"
            )
        self._bound_positions = positions
        self._bound_expected_positions = expected
        # Model forwards can construct inference tensors. They intentionally
        # have no version counter, so identity is checked per layer and the
        # values are compared once at finish instead of reading ``_version``.
        self._bound_position_version = (
            None if positions.is_inference() else positions._version
        )

    def capture(self, layer: int, positions: torch.Tensor, q: torch.Tensor) -> None:
        if self._closed:
            raise PredictionConfigError("probe capture is closed")
        if layer not in self.config.layers:
            return
        if layer in self._queries:
            raise PredictionConfigError("probe layer was captured twice")
        expected = tuple(range(self.forward_start, self.sequence_length))
        if self._bound_positions is not None:
            if positions is not self._bound_positions or (
                self._bound_position_version is not None
                and positions._version != self._bound_position_version
            ):
                raise PredictionConfigError("probe positions changed after binding")
        elif tuple(positions.tolist()) != expected:
            raise PredictionConfigError(
                "probe positions must cover the complete prefix and prediction"
            )
        if q.shape != (len(expected), self.query_heads * self.head_dim):
            raise PredictionConfigError("probe Q shape does not match target heads")
        selected = (
            q.reshape(len(expected), self.query_heads, self.head_dim)[
                [position - self.forward_start for position in self.positions],
                self.config.head_start : self.config.head_start
                + self.config.head_count,
            ]
            .detach()
            .clone()
        )
        # Defer the host-visible reduction until finish(): one GPU/CPU fence
        # for all layers, not one synchronization for each layer's Q tensor.
        self._finite_flags.append(torch.isfinite(selected).all())
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
        if self._bound_positions is not None and not torch.equal(
            self._bound_positions, self._bound_expected_positions
        ):
            raise PredictionConfigError("probe positions changed after binding")
        if not bool(torch.stack(self._finite_flags).all().item()):
            raise PredictionConfigError("probe Q contains non-finite values")
        return tuple(self._queries[layer] for layer in self.config.layers)

    def close(self) -> None:
        self._closed = True
        self._queries.clear()
        self._finite_flags.clear()
        self._bound_positions = None
        self._bound_expected_positions = None


class _LlamaTargetProbeCore(TargetProbe):
    """Full-prefix recomputation with private pools and existing weights.

    ``target_model_id`` is an explicit deployment identity binding, not a hash
    inferred from weights. ``transient_bytes_bound`` is caller-declared extra
    headroom (backend/activation/allocator temporaries), not a measured bound.
    An entire branch is one capture. Query tensors are scoped to that branch.
    """

    _model_architecture = "LlamaForCausalLM"

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
        vocabulary: VocabularySignature | None = None,
        prefix_budget: TransferBudget | None = None,
    ):
        if self._model_architecture == "Qwen2ForCausalLM":
            from sglang.srt.models.qwen2 import Qwen2ForCausalLM

            model_type = Qwen2ForCausalLM
        else:
            from sglang.srt.models.llama import LlamaForCausalLM

            model_type = LlamaForCausalLM
        if type(runner.model) is not model_type:
            raise PredictionConfigError(
                f"probe requires exact {self._model_architecture} model class"
            )
        if runner.tp_size != 1 or runner.pp_size != 1:
            raise PredictionConfigError("probe requires TP1 and PP1")
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
        self._validate_placement(runner)
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
        if prefix_budget is not None and not isinstance(prefix_budget, TransferBudget):
            raise PredictionConfigError("separate prefix cache budget required")
        self.prefix_budget = prefix_budget
        self.head_dim = runner.model_config.head_dim
        self.query_heads = hf.num_attention_heads
        self.kv_heads = hf.num_key_value_heads
        self.layers = hf.num_hidden_layers
        self.vocab_size = hf.vocab_size
        if vocabulary is not None and (
            not isinstance(vocabulary, VocabularySignature)
            or not vocabulary.exact_mapping_available
            or max(vocabulary.allowed_ids) >= self.vocab_size
        ):
            raise PredictionConfigError(
                "probe needs an exact tokenizer mapping within the target embedding"
            )
        self.vocabulary = vocabulary
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
        self.prefix_cache_bytes = (
            (max_tokens + 1)
            * self.layers
            * self.kv_heads
            * self.head_dim
            * 2
            * torch.empty((), dtype=self.dtype).element_size()
            + 2 * max_tokens * 4
            + 65536
        )
        self._prefix_caches = {}
        self._active = False
        self._used = False
        self._state = None
        self._quarantined = False
        self._private_state = None
        self._private_failure = None

    def _validate_placement(self, runner):
        if runner.device != "cpu" or any(
            p.device.type != "cpu" or p.dtype != torch.float32
            for p in runner.model.parameters()
        ):
            raise PredictionConfigError(
                "offline probe requires CPU FP32 target weights"
            )
        self.device, self.dtype = "cpu", torch.float32

    def _drain_private(self):
        """CPU is synchronous; CUDA subclass fences before any pool cleanup."""

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
        if not self._tokens_valid(tokens):
            raise PredictionConfigError("probe token is outside the target vocabulary")
        self._used = True
        cached = self._cached_record(prefix)
        self._state = PostRopeQueryCapture(
            self.config,
            prefix,
            len(prediction.tokens),
            query_heads=self.query_heads,
            head_dim=self.head_dim,
            forward_start=len(prefix.tokens) if cached is not None else 0,
        )
        if cached is not None:
            return self._forward_cached(prefix, prediction.tokens, self._state, cached)
        return self._forward(tokens, self._state)

    def capture_committed(self, prefix, positions):
        self._require_main_thread()
        if not self._active or self._used:
            raise PredictionConfigError("capture requires an unused probe branch")
        if (
            not isinstance(prefix, CommittedPrefix)
            or not prefix.tokens
            or len(prefix.tokens) > self.max_tokens
        ):
            raise PredictionConfigError(
                "committed probe input exceeds admitted bounds or has no prefix"
            )
        if (
            not isinstance(positions, tuple)
            or not 1 <= len(positions) <= self.max_predict_tokens
        ):
            raise PredictionConfigError(
                "committed Q copies exceed the reserved query bound"
            )
        if not self._tokens_valid(prefix.tokens):
            raise PredictionConfigError("probe token is outside the target vocabulary")
        # Late-start Q may refer to positions already cached as K/V but not
        # retained as Q. Evict before full recomputation rather than invent a
        # query from an unrelated position or silently hold duplicate pools.
        cached = self._prefix_caches.get(prefix.request_id)
        if cached is not None:
            self._drop_prefix_cache(cached)
        self._state = PostRopeQueryCapture(
            self.config,
            prefix,
            0,
            query_heads=self.query_heads,
            head_dim=self.head_dim,
            committed_positions=positions,
        )
        self._used = True
        return self._forward(prefix.tokens, self._state)

    def _tokens_valid(self, tokens):
        if self.vocabulary is not None:
            return all(self.vocabulary.contains(token) for token in tokens)
        # Legacy CPU fixture probes have no tokenizer.  Their physical model
        # embedding check is not a claim that padded logits are valid tokens.
        return all(
            type(token) is int and 0 <= token < self.vocab_size for token in tokens
        )

    def register_cached_request(self, req) -> None:
        """Bind an optional private prefix to one live Req incarnation.

        Registration allocates nothing. Only the CUDA refresh driver may call
        this, after its own identity and admission checks. A reused rid cannot
        inherit an earlier Req's cached KV.
        """
        self._require_main_thread()
        if self.prefix_budget is None:
            return
        rid = getattr(req, "rid", None)
        if not isinstance(rid, str) or not rid or rid in self._prefix_caches:
            raise PredictionConfigError("unique request cache identity required")
        self._prefix_caches[rid] = SimpleNamespace(
            req=req,
            owner=None,
            resources=None,
            slot=None,
            rows=[],
            tokens=(),
            version=None,
            committed_position=0,
        )

    def retire_cached_request(self, req) -> None:
        """Fence and free the exact request's private prefix before Req release."""
        self._require_main_thread()
        if self.prefix_budget is None:
            return
        record = self._prefix_caches.get(getattr(req, "rid", None))
        if record is None or record.req is not req or self._active or self._quarantined:
            raise PredictionConfigError("exact inactive cached Req required")
        self._drop_prefix_cache(record)
        del self._prefix_caches[req.rid]

    @torch.inference_mode()
    def _drop_prefix_cache(self, record) -> None:
        # The owner loop can retire this cache outside the target forward's
        # inference context. Its private request map may itself be an
        # inference tensor, so all pool mutations must re-enter that mode.
        if record.owner is None:
            return
        resources = record.resources
        try:
            self._drain_private()
            if resources is not None and record.slot is not None:
                resources.allocator.clear_mapping(record.slot)
                resources.allocator.free_kv(record.rows)
                resources.allocator.free_request(record.slot)
            self._drain_private()
        except BaseException:
            # Possibly-live rows and their charge remain owned. CUDA branch
            # also retains its target execution lock after quarantine.
            self._quarantined = True
            raise
        if resources is not None:
            vars(resources).clear()
        self.prefix_budget.release(record.owner)
        record.owner = record.resources = record.slot = None
        record.rows = []
        record.tokens = ()
        record.version = None
        record.committed_position = 0

    def _cached_record(self, prefix):
        if self.prefix_budget is None:
            return None
        record = self._prefix_caches.get(prefix.request_id)
        if record is None:
            return None
        if record.owner is not None and (
            prefix.tokens[: len(record.tokens)] != record.tokens
            or prefix.committed_position < record.committed_position
            or (prefix.version == record.version and prefix.tokens != record.tokens)
        ):
            self._drop_prefix_cache(record)
        if record.owner is None:
            owner = f"pvd-target-prefix:{uuid.uuid4().hex}"
            try:
                self.prefix_budget.reserve(owner, self.prefix_cache_bytes, 1)
            except TransferCapacityError:
                # Optional cache pressure never blocks the correct full-prefix
                # path. An owner may retry on its next prediction round.
                return None
            record.owner = owner
        return record

    @torch.inference_mode()
    def _forward_cached(self, prefix, predicted_tokens, capture, record):
        """Extend only authoritative tokens, then discard speculative KV.

        This is only reachable for a request explicitly registered by the
        CUDA refresh driver with a separate persistent budget. Each record's
        pools and mapping are private; neither can alias a committed Req.
        """
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
        resources = record.resources
        if resources is None:
            resources = SimpleNamespace()
            record.resources = resources  # Published before any CUDA allocation.
            try:
                resources.requests = ReqToTokenPool(
                    1, self.max_tokens, self.device, False
                )
                resources.pool = MHATokenToKVPool(
                    self.max_tokens,
                    1,
                    self.dtype,
                    self.kv_heads,
                    self.head_dim,
                    self.layers,
                    self.device,
                    False,
                )
                resources.kv = TokenToKVPoolAllocator(
                    self.max_tokens, self.dtype, self.device, resources.pool, False
                )
                resources.allocator = PrivatePoolAllocator(
                    resources.requests, resources.kv
                )
                resources.backend = TorchNativeAttnBackend(
                    SimpleNamespace(
                        device=self.device,
                        req_to_token_pool=resources.requests,
                        token_to_kv_pool=resources.pool,
                    )
                )
                resources.builder = DraftForwardAdapter(
                    None,
                    architecture=self._model_architecture,
                    attention_backend="torch_native",
                    bytes_per_token=self.layers * self.kv_heads * self.head_dim * 2 * 4,
                    device=self.device,
                )
                record.slot = resources.allocator.alloc_request()
            except BaseException:
                # A constructor may leave tensors in its traceback. Keep all
                # charge and the execution lease until a recovery protocol can
                # prove they are no longer live.
                self._private_state = resources
                self._quarantined = True
                raise

        def forward_chunk(chunk, *, start, rows, query_capture=None):
            end = start + len(chunk)
            batch = resources.builder.build_forward_batch(
                DraftForwardInputs(
                    "extend",
                    tuple(chunk),
                    tuple(range(start, end)),
                    (end,),
                    (record.slot,),
                    tuple(rows),
                    (start,),
                    (len(chunk),),
                )
            )
            resources.batch = batch
            batch.pvd_compact_extend = start > 0
            if query_capture is not None:
                query_capture.bind_positions(batch.positions)
                batch.pvd_query_capture = query_capture
            resources.backend.init_forward_metadata(batch)
            with (
                torch.inference_mode(),
                forward_context(ForwardContext(attn_backend=resources.backend)),
            ):
                self.model.model(batch.input_ids, batch.positions, batch)
            result = query_capture.finish() if query_capture is not None else None
            self._drain_private()
            del resources.batch
            return result

        try:
            committed_start = len(record.tokens)
            new_committed = prefix.tokens[committed_start:]
            if new_committed:
                rows = resources.allocator.alloc_kv(len(new_committed))
                record.rows.extend(rows)
                resources.allocator.write_mapping(record.slot, committed_start, rows)
                forward_chunk(new_committed, start=committed_start, rows=rows)
                record.tokens = prefix.tokens
                record.version = prefix.version
                record.committed_position = prefix.committed_position
            else:
                record.version = prefix.version
                record.committed_position = prefix.committed_position

            suffix_rows = resources.allocator.alloc_kv(len(predicted_tokens))
            record.rows.extend(suffix_rows)
            try:
                resources.allocator.write_mapping(
                    record.slot, len(prefix.tokens), suffix_rows
                )
                result = forward_chunk(
                    predicted_tokens,
                    start=len(prefix.tokens),
                    rows=suffix_rows,
                    query_capture=capture,
                )
                return result
            finally:
                try:
                    self._drain_private()
                    resources.requests.req_to_token[
                        record.slot,
                        len(prefix.tokens) : len(prefix.tokens) + len(suffix_rows),
                    ] = 0
                    resources.allocator.free_kv(suffix_rows)
                    self._drain_private()
                    del record.rows[-len(suffix_rows) :]
                except BaseException:
                    self._private_state = resources
                    self._quarantined = True
                    raise
        except BaseException as exc:
            if not self._quarantined:
                self._drop_prefix_cache(record)
                traceback.clear_frames(exc.__traceback__)
            raise

    def _forward(self, tokens, capture):
        profile_started = (
            time.perf_counter()
            if os.environ.get("PVD_PROFILE_REFRESH_TIMELINE") == "1"
            else None
        )
        setup_done = forward_done = None
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

        # Publish the owner BEFORE any constructor can allocate. A constructor
        # may raise after creating tensors and leave them in its traceback.
        resources = SimpleNamespace()
        self._private_state = resources
        slot, rows = None, []
        try:
            resources.requests = ReqToTokenPool(1, self.max_tokens, self.device, False)
            resources.pool = MHATokenToKVPool(
                self.max_tokens,
                1,
                self.dtype,
                self.kv_heads,
                self.head_dim,
                self.layers,
                self.device,
                False,
            )
            resources.kv = TokenToKVPoolAllocator(
                self.max_tokens, self.dtype, self.device, resources.pool, False
            )
            resources.allocator = PrivatePoolAllocator(resources.requests, resources.kv)
            resources.backend = TorchNativeAttnBackend(
                SimpleNamespace(
                    device=self.device,
                    req_to_token_pool=resources.requests,
                    token_to_kv_pool=resources.pool,
                )
            )
            resources.builder = DraftForwardAdapter(
                None,
                architecture=self._model_architecture,
                attention_backend="torch_native",
                bytes_per_token=self.layers * self.kv_heads * self.head_dim * 2 * 4,
                device=self.device,
            )
            slot = resources.allocator.alloc_request()
            rows = resources.allocator.alloc_kv(len(tokens))
            resources.allocator.write_mapping(slot, 0, rows)
            resources.batch = resources.builder.build_forward_batch(
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
            capture.bind_positions(resources.batch.positions)
            resources.batch.pvd_query_capture = capture
            resources.backend.init_forward_metadata(resources.batch)
            if profile_started is not None:
                setup_done = time.perf_counter()
            with (
                torch.inference_mode(),
                forward_context(ForwardContext(attn_backend=resources.backend)),
            ):
                # Backbone only: no sampler, output commit, or LM-head logits.
                self.model.model(
                    resources.batch.input_ids,
                    resources.batch.positions,
                    resources.batch,
                )
            result = capture.finish()
            if profile_started is not None:
                forward_done = time.perf_counter()
            return result
        except BaseException as exc:
            self._private_failure = exc
            raise
        finally:
            try:
                self._drain_private()
                if slot is not None:
                    resources.allocator.clear_mapping(slot)
                    resources.allocator.free_kv(rows)
                    resources.allocator.free_request(slot)
                # CUDA clear_mapping/free may themselves enqueue operations.
                self._drain_private()
            except BaseException:
                # Do not refund or reuse possibly-live rows after failed
                # cleanup. Keep the private state and reservation for diagnosis.
                self._quarantined = True
                raise
            else:
                if self._private_failure is not None:
                    traceback.clear_frames(self._private_failure.__traceback__)
                    self._private_failure = None
                vars(resources).clear()
                self._private_state = None
                if forward_done is not None:
                    try:
                        retired = time.perf_counter()
                        logging.getLogger(__name__).info(
                            "PVD timeline event=probe_stage tokens=%d "
                            "setup_ms=%.3f forward_capture_ms=%.3f "
                            "retire_ms=%.3f",
                            len(tokens),
                            (setup_done - profile_started) * 1000,
                            (forward_done - setup_done) * 1000,
                            (retired - forward_done) * 1000,
                        )
                    except Exception:
                        pass  # Profiling cannot change Q ownership or result.


class OfflineLlamaTargetProbe(_LlamaTargetProbeCore):
    """CPU FP32 reference; CUDA probes use a distinct public type."""


class OfflineQwen2TargetProbe(_LlamaTargetProbeCore):
    """CPU FP32 Qwen2 reference with the same private-pool Q contract."""

    _model_architecture = "Qwen2ForCausalLM"
