"""Explicit TP1/non-overlap CUDA batch -> original Decode result processor.

No sampler, token writer or automatic serving activation. The bootstrap/factory
must supply the real pool owner and registered controllers. The result hook is
valid only inside run(), after CUDARankBatchExecutor proved reader completion.
"""

import inspect
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError
from sglang.srt.disaggregation.pvd.cuda_rank_batch import (
    CUDARankBatchExecutor,
    CUDARuntimeBatchMember,
)
from sglang.srt.disaggregation.pvd.cuda_refresh_driver import CUDARefreshDriver


@dataclass(frozen=True)
class _Dispatch:
    registration: object
    controller: object
    group: object
    request_id: str
    slot: int
    outputs: tuple
    committed_tokens: int


class CUDAScheduleBridge:
    def __init__(self, executor, driver, batch, *, pool_owner):
        if not isinstance(executor, CUDARankBatchExecutor) or not isinstance(
            driver, CUDARefreshDriver
        ):
            raise LifecycleError(
                "explicit CUDA batch executor and refresh driver required"
            )
        driver._owner()
        previous = getattr(batch, "pvd_cuda_result_bridge", None)
        if (
            executor.dispatcher.arbiter is not driver.arbiter
            or executor.consumer._lock is not driver._execution_lock
            or driver.arbiter.busy
            or getattr(batch, "pvd_cpu_result_bridge", None) is not None
            or (
                previous is not None
                and (
                    not isinstance(previous, CUDAScheduleBridge)
                    or previous.state != "completed"
                    or previous.executor is not executor
                    or previous.driver is not driver
                )
            )
        ):
            raise LifecycleError(
                "one idle shared CUDA target and result owner required"
            )
        self.executor, self.driver, self.batch = executor, driver, batch
        self.pool_owner = pool_owner
        self.state = "attached"
        self._processor = self._result = None
        self._check_batch()
        reqs = tuple(batch.reqs)
        if not 0 < len(reqs) <= executor.dispatcher.max_requests or len(
            {id(r) for r in reqs}
        ) != len(reqs):
            raise LifecycleError("bounded unique request batch required")
        records = []
        for req in reqs:
            record = driver._records.get(req.rid)
            if (
                record is None
                or record.req is not req
                or record.stopping
                or record.quarantined
                or req.finished()
                or req.is_retracted
            ):
                raise LifecycleError("batch needs live registered Req incarnations")
            n = driver._observe(record)
            if not record.controller.can_decode(n):
                raise LifecycleError("whole CUDA batch must wait for refresh")
            records.append(
                _Dispatch(
                    record,
                    record.controller,
                    record.controller.group,
                    req.rid,
                    record.slot,
                    record.outputs,
                    n,
                )
            )
        self.records = tuple(records)
        # Retain a tombstone on this specific ScheduleBatch to refuse replay.
        batch.pvd_cuda_result_bridge = self

    def _check_batch(self):
        batch = self.batch
        if (
            torch.device(batch.device) != self.executor.consumer.device
            or batch.req_to_token_pool is not self.executor.consumer.req_pool
            or batch.enable_overlap
            or not batch.forward_mode.is_decode()
            or not batch.spec_algorithm.is_none()
            or batch.is_spec_v2
        ):
            raise LifecycleError(
                "exact CUDA pools, ordinary non-overlap Decode required"
            )

    def _unchanged(self):
        self._check_batch()
        if len(self.batch.reqs) != len(self.records):
            raise LifecycleError("dispatched CUDA batch membership changed")
        for req, saved in zip(self.batch.reqs, self.records, strict=True):
            record = saved.registration
            if (
                req is not record.req
                or record.controller is not saved.controller
                or record.controller.group is not saved.group
                or self.driver._records.get(saved.request_id) is not record
                or req.rid != saved.request_id
                or req.req_pool_idx != saved.slot
                or tuple(req.origin_input_ids) != record.prompt
                or tuple(req.output_ids) != saved.outputs
                or req.finished()
                or req.is_retracted
                or record.stopping
                or record.quarantined
            ):
                raise LifecycleError("Req changed during CUDA dispatch; commit nothing")

    def run(self, *, forward, processor):
        self.driver._owner()
        if self.state != "attached":
            raise LifecycleError("stale or replayed CUDA batch")
        if (
            not callable(forward)
            or inspect.iscoroutinefunction(forward)
            or processor.enable_overlap
            or processor.enable_overlap_mlx
        ):
            raise LifecycleError(
                "synchronous forward and non-overlap result processor required"
            )
        self._unchanged()
        self._processor = processor
        self.state = "executing"

        def process(result):
            # Called only inside the rank executor's drained result scope.
            self._unchanged()
            self._result, self.state = result, "awaiting_results"
            value = processor.process_batch_result_decode(self.batch, result)
            if self.state != "processed" or inspect.isawaitable(value):
                raise LifecycleError("original synchronous result hook was bypassed")
            return value

        try:
            value = self.executor.run(
                tuple(
                    CUDARuntimeBatchMember(r.group, r.slot, r.committed_tokens)
                    for r in self.records
                ),
                pool_owner=self.pool_owner,
                forward=forward,
                process_results=process,
            )
        except BaseException:
            self.state = "failed"
            # No rollback of tokens possibly committed before a callback failed.
            # Stop every member, even if another member's cancel itself fails.
            for saved in self.records:
                try:
                    self.driver._stop(
                        saved.registration, "CUDA batch execution/result failed"
                    )
                except BaseException:
                    pass  # _stop retains that exception and quarantines its owner.
            raise
        else:
            self.state = "completed"
            return value
        finally:
            if not self.executor._quarantined:
                self._result = self._processor = None

    @contextmanager
    def processing(self, processor, batch, result):
        self.driver._owner()
        if (
            self.state != "awaiting_results"
            or processor is not self._processor
            or batch is not self.batch
            or result is not self._result
            or not self.driver.arbiter.busy
            or self.executor.dispatcher._ticket is None
        ):
            raise LifecycleError(
                "CUDA result processing is outside its drained owner scope"
            )
        self._unchanged()
        if result.copy_done is not None:
            raise LifecycleError(
                "asynchronous copy result is outside non-overlap contract"
            )
        tokens = result.next_token_ids
        if isinstance(tokens, torch.Tensor):
            if (
                tokens.ndim != 1
                or tokens.dtype not in (torch.int32, torch.int64)
                or tokens.device
                not in (torch.device("cpu"), self.executor.consumer.device)
            ):
                raise LifecycleError("sampled integral token rows required")
            tokens = tokens.tolist()
        if (
            not isinstance(tokens, list)
            or len(tokens) != len(self.records)
            or any(type(t) is not int or t < 0 for t in tokens)
        ):
            raise LifecycleError("one sampled target token per dispatched Req required")
        tokens = tuple(tokens)  # The processor cannot rewrite the dispatch evidence.
        self.state = "processing"
        yield self
        # Validate ALL authoritative writes before accepting this result scope.
        for saved, token in zip(self.records, tokens, strict=True):
            record, req = saved.registration, saved.registration.req
            if (
                req.rid != saved.request_id
                or tuple(req.origin_input_ids) != record.prompt
                or tuple(req.output_ids) != saved.outputs + (token,)
                or (
                    not req.finished()
                    and (req.req_pool_idx != saved.slot or req.is_retracted)
                )
            ):
                raise LifecycleError(
                    "authoritative CUDA Req output differs from sampled dispatch"
                )
        for saved in self.records:
            saved.registration.outputs = tuple(saved.registration.req.output_ids)
        self.state = "processed"
