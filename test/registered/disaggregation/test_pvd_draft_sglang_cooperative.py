"""Cooperative native draft stepping; no CUDA model is needed for this test."""

import threading

import torch

from sglang.srt.disaggregation.pvd.draft_runner_sglang import (
    DEFAULT_CAPABILITIES,
    SGLangDraftRunnerFactory,
)
from sglang.srt.disaggregation.pvd.draft_sglang import (
    DraftForwardProgress,
    DraftPlacement,
    SGLangDraftProvider,
)
from sglang.srt.disaggregation.pvd.prediction import DraftConfig, snapshot_committed


class _Allocator:
    def __init__(self):
        self.next_request = 0
        self.next_row = 0
        self.live_requests = set()
        self.live_rows = set()
        self.mapping = {}

    def fork_for_branch(self):
        return self

    def alloc_request(self):
        request = self.next_request
        self.next_request += 1
        self.live_requests.add(request)
        self.mapping[request] = []
        return request

    def free_request(self, request):
        self.live_requests.discard(request)
        self.mapping.pop(request, None)

    def alloc_kv(self, count):
        rows = tuple(range(self.next_row, self.next_row + count))
        self.next_row += count
        self.live_rows.update(rows)
        return rows

    def free_kv(self, rows):
        self.live_rows.difference_update(rows)

    def write_mapping(self, request, start, rows):
        mapping = self.mapping[request]
        end = start + len(rows)
        if len(mapping) < end:
            mapping.extend([None] * (end - len(mapping)))
        mapping[start:end] = rows

    def clear_mapping(self, request):
        self.mapping[request] = []


class _Executor:
    def __init__(self):
        self.calls = []

    def architecture(self):
        return "LlamaForCausalLM"

    def attention_backend(self):
        return "triton"

    def bytes_per_token(self):
        return 8

    def transient_bytes(self, prefix_tokens, predict_tokens):
        return 4096

    def drain(self):
        pass

    def forward(self, inputs):
        self.calls.append(inputs)
        logits = torch.zeros(64)
        logits[20 + len(self.calls)] = 1
        return logits


def _provider(executor):
    factory = SGLangDraftRunnerFactory(
        executor,
        _Allocator(),
        capabilities=DEFAULT_CAPABILITIES,
        persistent_bytes=0,
        max_tokens=4,
    )
    return SGLangDraftProvider(
        DraftConfig("test-draft", predict_tokens=3),
        DraftPlacement(
            scratch_budget_bytes=1 << 20,
            persistent_budget_bytes=0,
        ),
        factory,
    )


def _prefix():
    return snapshot_committed("request-A", [1, 2, 3], 0, "version-4")


def _formal_worker_can_take_lock(lock):
    acquired = []

    def check():
        got = lock.acquire(blocking=False)
        acquired.append(got)
        if got:
            lock.release()

    worker = threading.Thread(target=check)
    worker.start()
    worker.join()
    return acquired == [True]


def test_iter_predict_yields_after_one_forward_without_holding_runner_lock():
    executor = _Executor()
    provider = _provider(executor)
    committed = _prefix()
    formal_decode_turns = []
    iterator_result = None

    with provider.branch():
        steps = provider.iter_predict(committed, max_tokens=3)
        while True:
            forwards_before = len(executor.calls)
            try:
                progress = next(steps)
            except StopIteration as completed:
                iterator_result = completed.value
                break

            assert isinstance(progress, DraftForwardProgress)
            assert len(executor.calls) == forwards_before + 1
            assert progress.forward_index == len(executor.calls)
            assert progress.forward_mode == executor.calls[-1].forward_mode
            assert not hasattr(progress, "tokens")

            # The scheduler can run the normal batch's formal Decode turn
            # before advancing the shadow prediction again.
            assert _formal_worker_can_take_lock(provider._execution_lock)
            formal_decode_turns.append("B/C formal Decode completed")
            assert committed.tokens == (1, 2, 3)

    assert iterator_result.request_id == "request-A"
    assert iterator_result.prefix_version == "version-4"
    assert iterator_result.tokens == (21, 22, 23)
    assert len(formal_decode_turns) == 3
    assert len(executor.calls) == 3  # one prefix and two continuation forwards
    assert provider.active_branches == 0


def test_closing_a_paused_prediction_releases_its_private_rows():
    executor = _Executor()
    provider = _provider(executor)
    allocator = provider.factory._allocator

    with provider.branch():
        steps = provider.iter_predict(_prefix(), max_tokens=3)
        next(steps)
        assert provider.active_branches == 1
        assert allocator.live_requests
        assert allocator.live_rows
        steps.close()
        assert provider.active_branches == 0
        assert not allocator.live_requests
        assert not allocator.live_rows


def test_synchronous_predict_remains_available():
    executor = _Executor()
    provider = _provider(executor)

    with provider.branch():
        prediction = provider.predict(_prefix(), max_tokens=3)

    assert prediction.tokens == (21, 22, 23)
    assert len(executor.calls) == 3
    assert provider.active_branches == 0
