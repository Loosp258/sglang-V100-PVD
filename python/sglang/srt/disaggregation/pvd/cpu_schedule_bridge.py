"""Opt-in CPU ScheduleBatch result bridge, not a production sparse-mode switch.

The normal result processor is the ONLY writer of Req.output_ids. This bridge
holds immutable dispatch membership and observes that write into the offline
lifecycle ledger. A real synchronous CPUBatchForwardExecutor must have finished
before result processing. CPU/TP1/no-overlap constraints remain enforced by the
executor; this bridge adds request and result ownership, not GPU support.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from sglang.srt.disaggregation.pvd.cpu_batch_forward import CPUBatchForwardExecutor
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError


@dataclass(frozen=True)
class RequestDispatch:
    req: object
    lifecycle: object
    slot: int
    outputs: tuple[int, ...]


class CPUScheduleBridge:
    """One batch ticket; retain its tombstone to reject result callback replay.

    attach() follows dispatcher.begin(), precedes forward(), and changes no Req
    tokens. On forward failure call fail_after_execution() after readers unwind.
    A request cancelled/retracted before processing is skipped. A structural
    identity mismatch aborts the entire ticket rather than guessing ownership.
    """

    def __init__(self, executor, batch, ticket):
        if not isinstance(executor, CPUBatchForwardExecutor):
            raise LifecycleError("real supported CPU batch executor required")
        executor.dispatcher._match(ticket)
        self.executor, self.batch, self.ticket = executor, batch, ticket
        self.dispatcher = executor.dispatcher
        self.state = "attached"
        self._accepted = None
        self._tokens = None
        self._check_batch_mode(batch)
        if getattr(batch, "pvd_cpu_result_bridge", None) is not None:
            raise LifecycleError("ScheduleBatch already has a dispatch owner")
        reqs = tuple(batch.reqs)
        if len(reqs) != len(ticket.members):
            raise LifecycleError(
                "ScheduleBatch must match complete dispatch membership"
            )
        records = []
        for req, life, member in zip(
            reqs, self.dispatcher._members, ticket.members, strict=True
        ):
            slot = executor._storage.get(life)
            if (
                req.rid != member.request_id
                or tuple(req.origin_input_ids) != life.prompt
                or tuple(req.output_ids) != life.outputs
                or type(slot) is not int
                or req.req_pool_idx != slot
                or req.is_retracted
                or req.finished()
            ):
                raise LifecycleError("Req identity, prefix, slot or admission mismatch")
            records.append(RequestDispatch(req, life, slot, life.outputs))
        self.records = tuple(records)
        batch.pvd_cpu_result_bridge = self

    @staticmethod
    def _check_batch_mode(batch):
        if (
            str(batch.device) != "cpu"
            or batch.enable_overlap
            or not batch.forward_mode.is_decode()
            or not batch.spec_algorithm.is_none()
            or batch.is_spec_v2
        ):
            raise LifecycleError(
                "CPU non-overlap ordinary Decode ScheduleBatch required"
            )

    def accepts(self, req):
        self.dispatcher.arbiter.owner()
        if self.state != "processing":
            raise LifecycleError("request filtering requires active result processing")
        for record, accepted in zip(self.records, self._accepted, strict=True):
            if record.req is req:
                return accepted
        raise LifecycleError("foreign request in result processor")

    def _prepare(self, processor, batch, result):
        self.dispatcher._match(self.ticket)
        if self.state != "attached" or batch is not self.batch:
            raise LifecycleError("stale or replayed ScheduleBatch result")
        if self.executor._completed_operation != self.ticket.operation_id:
            raise LifecycleError("actual synchronous CPU forward has not completed")
        self._check_batch_mode(batch)
        if processor.enable_overlap or processor.enable_overlap_mlx:
            raise LifecycleError("overlap result processing is unsupported")
        if result.copy_done is not None:
            raise LifecycleError("CPU result must not carry asynchronous device work")
        if len(batch.reqs) != len(self.records) or any(
            req is not record.req
            for req, record in zip(batch.reqs, self.records, strict=True)
        ):
            raise LifecycleError("dispatched ScheduleBatch membership/order changed")
        tokens = result.next_token_ids
        if isinstance(tokens, torch.Tensor):
            if (
                tokens.device.type != "cpu"
                or tokens.ndim != 1
                or tokens.dtype not in (torch.int32, torch.int64)
            ):
                raise LifecycleError(
                    "complete CPU integral sampled token rows required"
                )
            tokens = tokens.tolist()
        if not isinstance(tokens, list) or len(tokens) != len(self.records):
            raise LifecycleError("complete sampled token list required")
        for token in tokens:
            # Do not argmax again: preserve the real sampler's chosen output.
            self.records[0].lifecycle._token(token)
        accepted = []
        for record in self.records:
            req, life = record.req, record.lifecycle
            if (
                req.rid != life.request_id
                or tuple(req.origin_input_ids) != life.prompt
                or tuple(req.output_ids) != record.outputs
                or req.req_pool_idx != record.slot
            ):
                raise LifecycleError(
                    "Req identity/prefix/storage changed during dispatch"
                )
            life.poll()
            if req.is_retracted or req.finished():
                life.terminate("request cancelled, finished or retracted before commit")
            accepted.append(life.state == "running")
        self._tokens, self._accepted = tuple(tokens), tuple(accepted)
        self.state = "processing"

    def _observe(self):
        # Validate ALL authoritative writes before advancing ANY mirror clock.
        # Finish processing can release req_pool_idx, so only validate slots
        # before that processing, never claim a finished request still owns one.
        ended = []
        for record, token, accepted in zip(
            self.records, self._tokens, self._accepted, strict=True
        ):
            expected = record.outputs + ((token,) if accepted else ())
            if tuple(record.req.output_ids) != expected:
                raise LifecycleError(
                    "authoritative Req output commit did not match dispatch"
                )
            ended.append(record.req.finished())
        for record, token, accepted, finished in zip(
            self.records, self._tokens, self._accepted, ended, strict=True
        ):
            if accepted:
                record.lifecycle._outputs.append(token)
                if finished:
                    record.lifecycle.terminate("EOS or output limit", finished=True)
        # No second poll here: a deadline crossed during synchronous processing
        # cannot uncommit a token the normal result processor already appended.
        self.dispatcher._retire()
        self.state = "completed"

    def fail_after_execution(self, reason):
        """Caller guarantees synchronous forward/readers have already unwound."""
        self.dispatcher.arbiter.owner()
        if self.state not in ("completed", "failed"):
            if self.dispatcher._ticket is not None:
                self.dispatcher.fail(self.ticket, reason)
            self.state = "failed"

    @contextmanager
    def processing(self, processor, batch, result):
        self.dispatcher.arbiter.owner()
        # A bad early callback is not proof that execution drained. Do not free
        # its lease here; the owner must finish/stop the actual forward first.
        if self.executor._completed_operation != self.ticket.operation_id:
            raise LifecycleError("actual synchronous CPU forward has not completed")
        if self.state != "attached":
            raise LifecycleError("stale or replayed ScheduleBatch result")
        try:
            with self.dispatcher.result_scope(self.ticket):
                self._prepare(processor, batch, result)
                yield self
                self._observe()
        except BaseException:
            # Normal output processing may already have committed some Req
            # tokens. Never roll those back; terminally stop this whole batch.
            self.fail_after_execution("ScheduleBatch result processing failed")
            raise
