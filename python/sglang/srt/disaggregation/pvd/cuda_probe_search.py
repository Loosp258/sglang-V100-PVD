"""Explicit synchronous CUDA Q -> bounded host rows -> existing HTTP search.

No serving activation or compute-overlap claim. All callers of target/draft
execution and RNG must honor the shared lock; scopes must never span an await.
"""

import logging
import threading
import time
import uuid
from contextlib import contextmanager

import torch
from sglang.srt.disaggregation.pvd.cuda_target_probe import CUDALlamaTargetProbe
from sglang.srt.disaggregation.pvd.draft_forward_adapter import DraftForwardAdapter
from sglang.srt.disaggregation.pvd.draft_runner_sglang import SGLangDraftRunnerFactory
from sglang.srt.disaggregation.pvd.draft_sglang import SGLangDraftProvider
from sglang.srt.disaggregation.pvd.prediction import (
    CommittedPrefix,
    DraftPrediction,
    PredictionConfigError,
    PredictionPipeline,
)
from sglang.srt.disaggregation.pvd.probe_search import ProbeSearchSession
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

logger = logging.getLogger(__name__)


class CUDAPredictionPipeline(PredictionPipeline):
    """Concrete greedy draft + private target probe, serialized RNG scope.

    Model/config selection remains external. One owner main thread; the target
    consumer must use this same reentrant execution lock. CUDA RNG restoration
    is not isolation from unrelated users ignoring that lock.
    """

    def __init__(self, provider, probe, draft_config, probe_config, *, execution_lock):
        if not isinstance(execution_lock, type(threading.RLock())):
            raise PredictionConfigError("shared reentrant execution lock required")
        if not isinstance(probe, CUDALlamaTargetProbe) or (
            probe._execution_lock is not execution_lock or probe.config != probe_config
        ):
            raise PredictionConfigError("exact CUDA probe and shared lock required")
        if (
            not isinstance(provider, SGLangDraftProvider)
            or provider.config != draft_config
            or not isinstance(provider.factory, SGLangDraftRunnerFactory)
            or not isinstance(provider.factory._executor, DraftForwardAdapter)
        ):
            raise PredictionConfigError(
                "concrete prediction-only draft adapter required"
            )
        vocabulary = provider.vocabulary
        if vocabulary is not None and (
            not vocabulary.exact_mapping_available
            or getattr(probe, "vocabulary", None) != vocabulary
        ):
            raise PredictionConfigError(
                "CUDA prediction requires the same exact tokenizer mapping "
                "on draft and target probe"
            )
        target = torch.device(probe.device)
        draft = torch.device(draft_config.device)
        if draft != provider.factory._executor._device or draft.type not in (
            "cpu",
            "cuda",
        ):
            raise PredictionConfigError("draft config and executor placement differ")
        if draft.type == "cuda" and draft.index is None:
            raise PredictionConfigError("indexed CUDA draft device required")
        super().__init__(provider, probe, draft_config, probe_config)
        self._lock = execution_lock
        self._rng_devices = sorted(
            {d.index for d in (target, draft) if d.type == "cuda"}
        )
        self._scope_active = False
        self._quarantined = False

    @contextmanager
    def _scope(self):
        self.probe._require_main_thread()
        if (
            self._scope_active
            or self._quarantined
            or self.probe._quarantined
            or self.provider.degraded
        ):
            raise PredictionConfigError("CUDA prediction is active or quarantined")
        if not self._lock.acquire(blocking=False):
            raise PredictionConfigError("target execution is busy")
        self._scope_active = True
        try:
            rng = torch.random.fork_rng(devices=self._rng_devices, enabled=True)
            try:
                rng.__enter__()
            except BaseException:
                self._quarantined = True
                raise
            try:
                yield
            finally:
                try:
                    rng.__exit__(None, None, None)
                except BaseException:
                    self._quarantined = True
                    raise
        finally:
            self._scope_active = False
            self._quarantined |= self.probe._quarantined or self.provider.degraded
            # Keep the outer lease too when draft/RNG completion is unknown.
            if not self._quarantined:
                self._lock.release()

    def run(self, prefix):
        if not self._scope_active:
            raise PredictionConfigError("CUDA prediction requires its branch scope")
        return super().run(prefix)

    def _record_run_timing(self, *, draft_seconds, probe_seconds):
        logger.info(
            "PVD CUDA prediction stages: draft_seconds=%.6f probe_seconds=%.6f",
            draft_seconds,
            probe_seconds,
        )

    def iter_queries(self, prefix):
        """Produce prediction-only Q in bounded same-GPU scheduler steps.

        The caller advances this generator between ordinary formal Decode
        forwards. Neither the shared model lock nor an RNG fork spans a yield.
        Close it on request cancellation to retire branch-owned KV promptly.
        """
        self.probe._require_main_thread()
        if (
            self._scope_active
            or self._quarantined
            or self.probe._quarantined
            or self.provider.degraded
        ):
            raise PredictionConfigError("CUDA prediction is active or quarantined")
        if not isinstance(prefix, CommittedPrefix):
            raise PredictionConfigError("immutable committed prefix required")
        self._scope_active = True
        try:
            with self.provider.branch():
                prediction = yield from self.provider.iter_predict(
                    prefix, self.draft_config.predict_tokens
                )
            if (
                not isinstance(prediction, DraftPrediction)
                or prediction.request_id != prefix.request_id
                or prediction.prefix_version != prefix.version
                or len(prediction.tokens) > self.draft_config.predict_tokens
            ):
                raise PredictionConfigError(
                    "draft returned a foreign or oversized branch"
                )
            queries = yield from self.probe.capture_steps(prefix, prediction)
            return self._validate_queries(
                prefix,
                queries,
                range(len(prefix.tokens), len(prefix.tokens) + len(prediction.tokens)),
            )
        finally:
            self._scope_active = False
            self._quarantined |= self.probe._quarantined or self.provider.degraded

    @contextmanager
    def query_branch(self, prefix):
        started = time.perf_counter()
        with self._scope(), super().query_branch(prefix) as queries:
            logger.info(
                "PVD CUDA prediction scope entered: enter_seconds=%.6f",
                time.perf_counter() - started,
            )
            yield queries

    @contextmanager
    def committed_query_branch(self, prefix, positions):
        started = time.perf_counter()
        with (
            self._scope(),
            super().committed_query_branch(prefix, positions) as queries,
        ):
            logger.info(
                "PVD CUDA committed probe scope entered: enter_seconds=%.6f",
                time.perf_counter() - started,
            )
            yield queries


class CUDAProbeSearchSession(ProbeSearchSession):
    """CPU session identity/version checks plus explicitly owned CUDA copies.

    Budget covers explicit host tensors, not Python/HTTP allocator internals.
    Prepared route-position rows are bounded by ProbeSearchSession; positions
    remain <=64 and head dimension has an explicit bound. Source Q is owned
    and charged by the probe.
    """

    def __init__(self, *args, device, copy_budget, max_head_dim, **kwargs):
        selected = torch.device(device)
        if selected.type != "cuda" or selected.index is None:
            raise ValueError("indexed CUDA query device required")
        if not isinstance(copy_budget, TransferBudget):
            raise TypeError("explicit query copy budget required")
        if type(max_head_dim) is not int or max_head_dim <= 0:
            raise ValueError("positive query head-dimension bound required")
        super().__init__(*args, **kwargs)
        self.device, self.copy_budget, self.max_head_dim = (
            selected,
            copy_budget,
            max_head_dim,
        )
        self._copy_owner = None
        self._copy_retained = []
        self._copy_unknown = False
        self._copy_heads = ()
        self._copy_cache = None

    def _validate_pipeline(self, pipeline):
        if not isinstance(pipeline, CUDAPredictionPipeline) or (
            torch.device(pipeline.probe.device) != self.device
        ):
            raise PredictionConfigError("matching CUDA prediction pipeline required")
        if self._copy_unknown:
            raise PredictionConfigError("query copy is quarantined")

    def _query_device(self, tensor):
        return tensor.device == self.device and tensor.dtype in (
            torch.float16,
            torch.float32,
        )

    @contextmanager
    def _prepare_scope(self, routes, window):
        if any(r.scope.head_dim > self.max_head_dim for r in routes):
            raise ValueError("query head dimension exceeds admitted bound")
        # Copy every routed head in one layer before the next layer. Both the
        # native-dtype host rows and FP32 rows fit this reservation; no GPU
        # gather buffer or unrelated query head is materialized.
        heads = tuple(sorted({r.query_head for r in routes}))
        size = (
            len(window.query_positions)
            * len(heads)
            * max(r.scope.head_dim for r in routes)
            * 8
        )
        owner = f"pvd-query-copy:{uuid.uuid4().hex}"
        self.copy_budget.reserve(owner, size, 1)
        self._copy_owner = owner
        self._copy_heads = heads
        try:
            yield
        finally:
            if not self._copy_unknown:
                self._copy_cache = None
                self._copy_retained.clear()
                self._copy_heads = ()
                self.copy_budget.release(owner)
                self._copy_owner = None

    def _query_rows(self, tensor, indices, head, pipeline):
        cached = self._copy_cache
        if cached is None or cached[0] is not tensor or cached[1] != indices:
            # The previous layer was fenced; its host storage can be retired
            # before allocating the next layer under the same reservation.
            self._copy_retained.clear()
            self._copy_cache = None
            head_start = getattr(
                getattr(pipeline, "probe_config", None), "head_start", 0
            )
            local_heads = tuple(
                global_head - head_start for global_head in self._copy_heads
            )
            if not local_heads or any(
                local_head < 0 or local_head >= tensor.shape[1]
                for local_head in local_heads
            ):
                raise ValueError("routed query head is missing from probe")
            host = torch.empty(
                (len(indices), len(local_heads), tensor.shape[-1]),
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=tensor.device.type == "cuda",
            )
            self._copy_retained = [tensor, host]
            try:
                for row, index in enumerate(indices):
                    for column, local_head in enumerate(local_heads):
                        host[row, column].copy_(
                            tensor[index, local_head].detach(),
                            non_blocking=tensor.device.type == "cuda",
                        )
            except BaseException:
                # An unsuccessful submission may still have enqueued work.
                self._finish_copy(pipeline)
                raise
            self._finish_copy(pipeline)
            converted = host.to(dtype=torch.float32)
            self._copy_retained.append(converted)
            if not torch.isfinite(converted).all():
                raise ValueError("probe query must contain finite float32 values")
            cached = (tensor, indices, converted, local_heads)
            self._copy_cache = cached
        try:
            column = cached[3].index(head)
        except ValueError as exc:
            raise ValueError("routed query head is missing from probe") from exc
        return tuple(tuple(row) for row in cached[2][:, column, :].tolist())

    def _finish_copy(self, pipeline):
        try:
            torch.cuda.synchronize(self.device)
        except BaseException:
            self._copy_unknown = True
            pipeline.probe.quarantine_query_copy()
            raise
