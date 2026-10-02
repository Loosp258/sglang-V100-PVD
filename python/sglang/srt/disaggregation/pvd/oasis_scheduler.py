"""Explicit TP1 Oasis binding to ordinary Scheduler sampling and results.

Startup supplies the initialized EAGLE/sparse-bank admission factory. This
binding never constructs a complete-prefix prediction pipeline or substitutes
a full-KV forward for a missing Oasis request owner.
"""

from contextlib import contextmanager
import threading
import logging
import time

import torch

from sglang.srt.disaggregation.pvd.oasis_request import OasisRequestDecoder
from sglang.srt.disaggregation.pvd.oasis_sglang import SGLangQwenPairedDecode

logger = logging.getLogger(__name__)


class OasisSchedulerBinding:
    def __init__(self, scheduler, prepare_request):
        args = scheduler.server_args
        runner = scheduler.tp_worker.model_runner
        if (not callable(prepare_request) or scheduler.enable_overlap
                or args.disaggregation_topology != "pvd"
                or args.disaggregation_mode != "decode"
                or args.pvd_cuda_predictive_serving
                or args.speculative_algorithm is not None
                or not args.disable_cuda_graph or not args.disable_overlap_schedule
                or scheduler.max_running_requests != 1
                or runner.tp_size != 1 or runner.pp_size != 1
                or getattr(scheduler, "pvd_cuda_binding", None) is not None
                or getattr(scheduler, "pvd_oasis_binding", None) is not None):
            raise ValueError("explicit single-request TP1 Oasis admission factory required")
        self.scheduler, self.prepare_request = scheduler, prepare_request
        self.manager = scheduler.disagg_decode_prealloc_queue.kv_manager
        self.runner = runner
        self.records = {}
        self._thread = threading.get_ident()
        self._dispatch = None
        self._processing = False
        self._deferred_release = None
        self.quarantined = False
        scheduler.pvd_oasis_binding = self

    def _check(self):
        if threading.get_ident() != self._thread or self.quarantined:
            raise RuntimeError("Oasis Scheduler owner is foreign or quarantined")

    def owns(self, req):
        record = self.records.get(req.rid)
        return record is not None and record[0] is req

    def prepare(self, reqs):
        self._check()
        for req in reqs:
            if self.owns(req):
                continue
            if self.records or req.rid in self.records or not req.output_ids:
                raise RuntimeError("one exact live Oasis request required")
            session = self.manager.decode_sessions[self.manager.key_for(req)]
            receipt = session.require_initial_prompt()
            owner = self.prepare_request(req, receipt)
            if (not isinstance(owner, OasisRequestDecoder)
                    or not isinstance(owner.decoder, SGLangQwenPairedDecode)
                    or owner.decoder.target is not self.runner.model
                    or owner.request_id != req.rid or owner.step != 0
                    or owner.current_token != req.output_ids[-1]
                    or owner.position != len(req.origin_input_ids)
                    or owner.state != "ready"):
                # The factory retains incomplete native owners on failure; do
                # not erase evidence by silently switching to ordinary Decode.
                self.quarantined = True
                self._rejected_owner = owner
                raise RuntimeError("Oasis admission returned a mismatched request owner")
            self.records[req.rid] = (req, owner, session, tuple(req.origin_input_ids))

    @torch.inference_mode()
    def forward(self, batch):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.managers.scheduler import GenerationBatchResult
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        self._check()
        if (self._dispatch is not None or len(batch.reqs) != 1
                or not batch.forward_mode.is_decode() or not batch.spec_algorithm.is_none()
                or batch.enable_overlap or getattr(batch, "pvd_cuda_result_bridge", None)
                or getattr(batch, "pvd_cpu_result_bridge", None)):
            raise RuntimeError("ordinary single-request Oasis Decode batch required")
        req = batch.reqs[0]
        if not self.owns(req) or req.finished() or req.is_retracted:
            raise RuntimeError("missing or stopped Oasis request; no probe fallback")
        _, owner, session, prompt = self.records[req.rid]
        outputs = tuple(req.output_ids)
        if (tuple(req.origin_input_ids) != prompt or owner.step != len(outputs) - 1
                or session.lease_error or owner.state != "ready"):
            raise RuntimeError("Oasis request incarnation or committed prefix changed")
        fb = ForwardBatch.init_new(batch, self.runner)
        self._dispatch = (batch, req, owner, outputs, fb, None)
        try:
            tick = time.perf_counter()
            trace_start = len(owner.pipeline.trace)
            logits = owner.forward(outputs[-1], len(prompt) + len(outputs) - 1)
            # Keep the original formal allocator mapping useful for release,
            # canaries and request accounting. Future candidate KV is excluded.
            pool = self.runner.token_to_kv_pool_allocator.get_kvcache()
            for layer, history in enumerate(owner.decoder.generated):
                key, value = history[-1]
                pool.set_kv_buffer(self.runner.model.model.layers[layer].self_attn.attn,
                    fb.out_cache_loc, key.transpose(0, 1), value.transpose(0, 1))
            torch.cuda.current_stream(logits.device).synchronize()
            output = LogitsProcessorOutput(next_token_logits=logits.float())
            tokens = self.runner.sample(output, fb)
            result = GenerationBatchResult(logits_output=output,
                next_token_ids=tokens, can_run_cuda_graph=False)
            logger.info("PVD Oasis forward rid=%s step=%d total_ms=%.3f layer_wait_ms=%.3f",
                req.rid, owner.step, (time.perf_counter() - tick) * 1000,
                sum(t["consumer_wait_seconds"] for t in owner.pipeline.trace[trace_start:]) * 1000)
            self._dispatch = (batch, req, owner, outputs, fb, result)
            batch.pvd_oasis_result_bridge = self
            return result
        except BaseException:
            self.quarantined = True
            # Unknown formal-pool writes or sampling must keep the whole
            # dispatch, incoming futures and actual history alive.
            raise

    @contextmanager
    def processing(self, processor, batch, result):
        self._check()
        dispatch = self._dispatch
        if (dispatch is None or dispatch[0] is not batch or dispatch[-1] is not result
                or self._processing or processor is not self.scheduler.batch_result_processor):
            raise RuntimeError("Oasis result outside its drained formal forward")
        _, req, owner, outputs, fb, _ = dispatch
        if tuple(req.output_ids) != outputs or req.finished() or req.is_retracted:
            raise RuntimeError("actual request changed before sampling commit")
        tokens = result.next_token_ids.tolist()
        if len(tokens) != 1 or type(tokens[0]) is not int:
            raise RuntimeError("one ordinary sampled token required")
        self._processing = True
        try:
            yield self
            if (batch.reqs != [req] or tuple(req.output_ids) != outputs + (tokens[0],)
                    or not self.owns(req)):
                raise RuntimeError("ordinary result processor did not commit exactly one actual token")
            owner.actual_committed(tokens[0])
            self._dispatch = None
        except BaseException:
            self.quarantined = True
            raise
        finally:
            self._processing = False
            if self._deferred_release is req and not self.quarantined:
                self._deferred_release = None
                self.release(req)
                self.manager.decode_refresher.release_request(req)

    def release(self, req):
        """Return True when a live receiver release must be deferred."""
        self._check()
        if not self.owns(req):
            return False
        if self._processing:
            self._deferred_release = req
            return True
        owner = self.records[req.rid][1]
        errors = owner.close()
        if errors:
            self.quarantined = True
            raise RuntimeError("Oasis background work failed during drainage")
        del self.records[req.rid]
        return False
