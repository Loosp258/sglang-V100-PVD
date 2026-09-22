"""Real Req/ScheduleBatch/result processor; external service callbacks are spies.

Not a Scheduler event loop. Optional real finish/cache-release plumbing uses
CPU-owned deferred releases; streaming and metrics remain service spies.
"""

from types import MethodType, SimpleNamespace

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def make_req(lifecycle, slot, vocab_size):
    sampling = SamplingParams(max_new_tokens=100, ignore_eos=True)
    sampling.normalize(None)  # no stop strings/tokenizer needed by this fixture
    req = Req(
        lifecycle.request_id,
        "",
        lifecycle.prompt,
        sampling,
        vocab_size=vocab_size,
    )
    req.output_ids.extend(lifecycle.outputs)
    req.req_pool_idx = slot
    return req


def make_batch(reqs):
    return ScheduleBatch(
        reqs=list(reqs),
        device="cpu",
        enable_overlap=False,
        forward_mode=ForwardMode.DECODE,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        return_logprob=False,
    )


def make_processor():
    calls = []

    def record(name):
        return lambda *args, **kwargs: calls.append(name)

    processor = SimpleNamespace(
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(enable_metrics=False),
        model_config=SimpleNamespace(think_end_id=None),
        metrics_reporter=SimpleNamespace(
            num_generated_tokens=0,
            forward_ct_decode=0,
            report_decode_stats=record("stats"),
        ),
        token_to_kv_pool_allocator=SimpleNamespace(
            free_group_begin=record("free_begin"),
            free_group_end=record("free_end"),
        ),
        output_streamer=SimpleNamespace(stream_output=record("stream")),
        _handle_finished_req=record("finish_callback"),
        calls=calls,
    )
    for name in (
        "process_batch_result_decode",
        "_process_batch_result_decode",
        "_normalize_decode_outputs",
        "_maybe_update_reasoning_tokens",
        "_mamba_prefix_cache_update",
    ):
        setattr(
            processor,
            name,
            MethodType(getattr(SchedulerBatchResultProcessor, name), processor),
        )
    # Baseline exercises the SAME real normal entrypoint with NO bridge.
    base = Req("normal-baseline", "", (1, 2), SamplingParams(max_new_tokens=1))
    processor.process_batch_result_decode(
        make_batch([base]), GenerationBatchResult(next_token_ids=[7])
    )
    assert tuple(base.output_ids) == (7,) and base.finished()
    assert calls == ["free_begin", "finish_callback", "stream", "free_end", "stats"]
    calls.clear()
    return processor


def deliver(processor, bridge, batch, logits):
    processor.process_batch_result_decode(
        batch, GenerationBatchResult(next_token_ids=logits.argmax(-1))
    )
    assert bridge.state == "completed"
    for record in bridge.records:
        assert tuple(record.req.output_ids) == record.lifecycle.outputs
    assert not bridge.dispatcher.arbiter.busy


def bind_real_cache_release(processor, runner):
    """Use the real finish method, ChunkCache and model pool allocators."""
    from sglang.srt.mem_cache.chunk_cache import ChunkCache

    processor.tree_cache = ChunkCache(
        SimpleNamespace(
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            page_size=1,
        )
    )
    processor.token_to_kv_pool_allocator = runner.token_to_kv_pool_allocator
    processor.server_args.disaggregation_decode_enable_offload_kvcache = False
    processor.server_args.enable_hisparse = False
    processor.model_worker = SimpleNamespace()
    for name in (
        "_maybe_collect_routed_experts",
        "_maybe_collect_indexer_topk",
        "_maybe_collect_customized_info",
    ):
        setattr(processor, name, lambda *args, **kwargs: None)
    processor._handle_finished_req = MethodType(
        SchedulerBatchResultProcessor._handle_finished_req, processor
    )


def abort_real_waiting_request(req, processor):
    """Execute Scheduler.abort_request; I/O/other queues are fixture services."""
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.scheduler import Scheduler

    messages = []
    scheduler = SimpleNamespace(
        waiting_queue=[req],
        enable_hicache_storage=False,
        disaggregation_mode=DisaggregationMode.DECODE,
        server_args=SimpleNamespace(disaggregation_topology="pd"),
        tree_cache=processor.tree_cache,
        ipc_channels=SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(
                send_output=lambda message, owner: messages.append((message, owner))
            )
        ),
        grammar_manager=SimpleNamespace(abort_requests=lambda _: None),
        disagg_decode_prealloc_queue=SimpleNamespace(queue=[], retracted_queue=[]),
        disagg_decode_transfer_queue=SimpleNamespace(queue=[]),
        cur_batch=None,
        running_batch=SimpleNamespace(reqs=[]),
    )
    Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))
    assert not scheduler.waiting_queue
    assert len(messages) == 1 and messages[0][1] is req
