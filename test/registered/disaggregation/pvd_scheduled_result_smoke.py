"""Real Req/ScheduleBatch/result processor; external service callbacks are spies.

Not a Scheduler event loop or production cache-release test. The surrounding
CPU driver owns and eventually releases all pool allocations.
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
