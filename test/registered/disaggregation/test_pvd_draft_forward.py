"""ForwardBatch construction, the private request map, and the real pools.

The doubles here are written from the source contracts in this checkout, not
from what the adapter happens to send:

* ``ReqToTokenPool`` -- alloc takes a list of request objects and assigns
  ``req_pool_idx`` in place; free takes the object and clears it; slot 0 is a
  padding row that is never handed out.
* ``TokenToKVPoolAllocator`` -- ``free_pages`` is int64 on the allocator's
  device; ``free()`` concatenates onto it, so a wrong device or dtype raises.
* ``PagedTokenToKVPoolAllocator`` -- ``alloc`` is page-aligned.
* ``ModelRunner.forward`` -- returns ``ModelRunnerOutput`` whose
  ``logits_output.next_token_logits`` is ``[#seq, vocab]`` and Optional.
* ``CaptureHiddenMode`` -- an IntEnum with ``NULL = 0`` and ``need_capture()``.

The optional real-class tests below also construct real batches and drive
real pools when serving dependencies are installed. Model execution is tested
separately by the opt-in strict ``run_pvd_draft_cpu_smoke.py``, not by doubles.
"""

import ast
from enum import IntEnum
from pathlib import Path

import pytest
import torch
from sglang.srt.disaggregation.pvd.draft_forward_adapter import (
    MUST_STAY_UNSET,
    SUPPORTED_PAGE_SIZE,
    DraftForwardAdapter,
    DraftRequestHandle,
    PrivatePoolAllocator,
)
from sglang.srt.disaggregation.pvd.draft_runner_sglang import (
    DEFAULT_CAPABILITIES,
    DraftForwardInputs,
    SGLangDraftHandle,
)
from sglang.srt.disaggregation.pvd.draft_sglang import (
    DraftCapabilityError,
    DraftLifecycleError,
)
from test_pvd_draft_sglang import FakeAllocator, FakeExecutor

SRT = Path(__file__).resolve().parents[3] / "python" / "sglang" / "srt"


# --------------------------------------------------------------------------
# Doubles built from the source contracts
# --------------------------------------------------------------------------


class RealShapedReqPool:
    """``ReqToTokenPool``: alloc(list[Req]) -> list[int], free(Req).

    Mirrors the shipped implementation, including the padding row at index 0
    that ``free_slots = list(range(1, size + 1))`` excludes.
    """

    def __init__(self, size=4, max_context_len=32, device="cpu", full=False):
        self._alloc_size = size + 1
        self.req_to_token = torch.zeros(
            (self._alloc_size, max_context_len), dtype=torch.int32, device=device
        )
        self.free_slots = [] if full else list(range(1, self._alloc_size))
        self.freed = []

    def alloc(self, reqs):
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        assert all(
            reqs[i].inflight_middle_chunks > 0 or reqs[i].kv_committed_len > 0
            for i in reusing
        ), "reusing request must be chunked or have committed KV"
        need = len(reqs) - len(reusing)
        if need > len(self.free_slots):
            return None
        select = self.free_slots[:need]
        self.free_slots = self.free_slots[need:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.freed.append(req.req_pool_idx)
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None


class RealShapedKVAllocator:
    """``TokenToKVPoolAllocator``: int64 ``free_pages`` on its own device."""

    page_size = 1

    def __init__(self, size=64, device="cpu", short_by=0):
        self.device = device
        self.size = size
        self.free_pages = torch.arange(1, size + 1, dtype=torch.int64, device=device)
        self.short_by = short_by
        self.freed = []

    def alloc(self, need_size):
        if need_size > len(self.free_pages):
            return None
        take = max(need_size - self.short_by, 0)
        select = self.free_pages[:take]
        self.free_pages = self.free_pages[take:]
        return select

    def free(self, free_index):
        # The real implementation concatenates, so device and dtype must match.
        if free_index.device != self.free_pages.device:
            raise RuntimeError(
                f"free index on {free_index.device}, free_pages on "
                f"{self.free_pages.device}"
            )
        if free_index.dtype != self.free_pages.dtype:
            raise RuntimeError(
                f"free index dtype {free_index.dtype}, free_pages "
                f"{self.free_pages.dtype}"
            )
        self.freed.append(free_index.tolist())
        self.free_pages = torch.cat((self.free_pages, free_index))


class RealShapedPagedAllocator(RealShapedKVAllocator):
    """``PagedTokenToKVPoolAllocator``: page-aligned allocation only."""

    def __init__(self, *a, page_size=16, **k):
        super().__init__(*a, **k)
        self.page_size = page_size

    def alloc(self, need_size):
        assert need_size % self.page_size == 0, (
            "The allocation size should be page-aligned"
        )
        return super().alloc(need_size)


class CaptureHiddenModeDouble(IntEnum):
    """``CaptureHiddenMode``, including the method the adapter must not break."""

    NULL = 0
    LAST = 1
    FULL = 2

    def need_capture(self):
        return self != CaptureHiddenModeDouble.NULL


class LogitsProcessorOutputDouble:
    def __init__(self, next_token_logits):
        self.next_token_logits = next_token_logits
        self.hidden_states = None


class PPProxyTensorsDouble:
    """What logits_output is under pipeline parallelism: no next_token_logits."""

    def __init__(self):
        self.tensors = {}


class ModelRunnerOutputDouble:
    def __init__(self, logits_output):
        self.logits_output = logits_output
        self.can_run_graph = False


class RealShapedModelRunner:
    """Returns a ModelRunnerOutput, as ``ModelRunner.forward`` does."""

    def __init__(self, vocab=64, device="cpu", logits_output="normal", rows=None):
        self.vocab = vocab
        self.device = device
        self.mode = logits_output
        self.rows = rows
        self.seen_shapes = []

    def forward(self, forward_batch):
        self.seen_shapes.append(
            (forward_batch.forward_mode, int(forward_batch.batch_size))
        )
        if self.mode == "pp":
            return ModelRunnerOutputDouble(PPProxyTensorsDouble())
        if self.mode == "none":
            return ModelRunnerOutputDouble(LogitsProcessorOutputDouble(None))
        if self.mode == "bare":
            bare = torch.zeros(self.vocab)
            bare[7] = 10.0
            return bare
        rows = self.rows or 1
        logits = torch.zeros(rows, self.vocab)
        logits[-1, 7] = 10.0
        return ModelRunnerOutputDouble(LogitsProcessorOutputDouble(logits))


class RecordingForwardBatch:
    """Stands in for ForwardBatch: the real one needs the serving frontend."""

    def __init__(self, **fields):
        self.fields = dict(fields)
        for name, value in fields.items():
            setattr(self, name, value)
        for name in MUST_STAY_UNSET:
            if name not in fields:
                setattr(self, name, None)
        for name in ("extend_seq_lens", "extend_prefix_lens", "extend_num_tokens"):
            if name not in fields:
                setattr(self, name, None)
        # The factory translates the adapter's marker, as the real path does.
        if self.capture_hidden_mode == "null":
            self.capture_hidden_mode = CaptureHiddenModeDouble.NULL


def adapter(runner=None, **kwargs):
    kwargs.setdefault("architecture", "LlamaForCausalLM")
    kwargs.setdefault("attention_backend", "triton")
    kwargs.setdefault("bytes_per_token", 8)
    kwargs.setdefault("forward_batch_factory", RecordingForwardBatch)
    return DraftForwardAdapter(runner or RealShapedModelRunner(), **kwargs)


def extend_inputs(length=3, request=2):
    return DraftForwardInputs(
        forward_mode="extend",
        input_ids=tuple(range(10, 10 + length)),
        positions=tuple(range(length)),
        seq_lens=(length,),
        req_pool_indices=(request,),
        out_cache_loc=tuple(range(100, 100 + length)),
        extend_prefix_lens=(0,),
        extend_seq_lens=(length,),
    )


def decode_inputs(position=3, request=2):
    return DraftForwardInputs(
        forward_mode="decode",
        input_ids=(55,),
        positions=(position,),
        seq_lens=(position + 1,),
        req_pool_indices=(request,),
        out_cache_loc=(200,),
    )


def private_pools(**kwargs):
    return PrivatePoolAllocator(
        kwargs.pop("requests", None) or RealShapedReqPool(),
        kwargs.pop("kv", None) or RealShapedKVAllocator(),
    )


# --------------------------------------------------------------------------
# 1. The real request-pool API
# --------------------------------------------------------------------------


def test_the_request_pool_is_driven_with_a_request_object():
    requests = RealShapedReqPool()
    alloc = PrivatePoolAllocator(requests, RealShapedKVAllocator())
    index = alloc.alloc_request()
    # Slot 0 is the padding row and is never handed out.
    assert index >= 1
    assert alloc._request.req_pool_idx == index
    alloc.free_request(index)
    # free() clears the attribute, which is what lets the object be reused.
    assert alloc._request.req_pool_idx is None
    assert requests.freed == [index]


def test_a_branch_never_borrows_a_committed_request():
    alloc = private_pools()
    assert isinstance(alloc._request, DraftRequestHandle)
    # It carries exactly the attributes the pool touches, and no Req behaviour.
    assert set(DraftRequestHandle.__slots__) == {
        "req_pool_idx",
        "inflight_middle_chunks",
        "kv_committed_len",
    }
    handle = DraftRequestHandle()
    assert handle.req_pool_idx is None
    assert handle.inflight_middle_chunks == 0 and handle.kv_committed_len == 0


def test_a_full_request_pool_is_refused_not_misread():
    alloc = PrivatePoolAllocator(RealShapedReqPool(full=True), RealShapedKVAllocator())
    with pytest.raises(DraftLifecycleError, match="no private request slot"):
        alloc.alloc_request()


def test_allocating_twice_without_freeing_is_refused():
    alloc = private_pools()
    alloc.alloc_request()
    with pytest.raises(DraftLifecycleError, match="one branch, one request"):
        alloc.alloc_request()


def test_freeing_a_slot_this_branch_does_not_hold_is_refused():
    alloc = private_pools()
    index = alloc.alloc_request()
    with pytest.raises(DraftLifecycleError, match="but this branch holds"):
        alloc.free_request(index + 1)


def test_freeing_twice_does_not_trip_the_pools_assertion():
    requests = RealShapedReqPool()
    alloc = PrivatePoolAllocator(requests, RealShapedKVAllocator())
    index = alloc.alloc_request()
    alloc.free_request(index)
    alloc.free_request(index)  # the real pool would assert here
    assert requests.freed == [index]


def test_a_returned_slot_is_reusable_by_the_next_branch():
    requests = RealShapedReqPool(size=1)
    first = PrivatePoolAllocator(requests, RealShapedKVAllocator())
    index = first.alloc_request()
    first.free_request(index)
    second = PrivatePoolAllocator(requests, RealShapedKVAllocator())
    assert second.alloc_request() == index


# --------------------------------------------------------------------------
# 2. The real ModelRunner output
# --------------------------------------------------------------------------


def test_logits_are_unwrapped_from_the_model_runner_output():
    runner = RealShapedModelRunner(rows=3)
    logits = adapter(runner).forward(extend_inputs(3))
    assert logits.ndim == 1 and logits.shape[0] == runner.vocab
    assert int(torch.argmax(logits).item()) == 7


def test_a_missing_logits_output_is_refused():
    class NoOutput:
        device = "cpu"

        def forward(self, batch):
            return object()

    with pytest.raises(DraftLifecycleError, match="ModelRunnerOutput"):
        adapter(NoOutput()).forward(extend_inputs())


def test_pipeline_parallel_output_is_refused_rather_than_misread():
    """logits_output can be PPProxyTensors, which has no next_token_logits."""
    with pytest.raises(DraftLifecycleError, match="pipeline parallelism"):
        adapter(RealShapedModelRunner(logits_output="pp")).forward(extend_inputs())


def test_a_none_next_token_logits_is_refused_explicitly():
    """A documented case: prefill-only requests produce no next token."""
    with pytest.raises(DraftLifecycleError, match="no next-token distribution"):
        adapter(RealShapedModelRunner(logits_output="none")).forward(extend_inputs())


def test_the_last_sequence_row_is_taken_from_a_two_dimensional_result():
    runner = RealShapedModelRunner(rows=4)
    logits = adapter(runner).forward(extend_inputs(4))
    assert logits.shape == (runner.vocab,)
    assert int(torch.argmax(logits).item()) == 7


def test_an_empty_logits_matrix_is_refused():
    class Empty(RealShapedModelRunner):
        def forward(self, batch):
            return ModelRunnerOutputDouble(
                LogitsProcessorOutputDouble(torch.zeros(0, self.vocab))
            )

    with pytest.raises(DraftLifecycleError, match="no rows"):
        adapter(Empty()).forward(extend_inputs())


# --------------------------------------------------------------------------
# 3. ForwardBatch execution semantics
# --------------------------------------------------------------------------


def test_capture_is_disabled_with_the_enum_not_with_none():
    made = adapter()
    fields = made.forward_fields(extend_inputs())
    assert "capture_hidden_mode" in fields
    batch = made.build_forward_batch(extend_inputs())
    assert batch.capture_hidden_mode is CaptureHiddenModeDouble.NULL
    # The method the logits processor calls must work on it.
    assert batch.capture_hidden_mode.need_capture() is False


def test_a_none_capture_mode_is_refused():
    made = adapter()
    batch = made.build_forward_batch(extend_inputs())
    batch.capture_hidden_mode = None
    with pytest.raises(DraftLifecycleError, match="must be CaptureHiddenMode.NULL"):
        made._assert_prediction_only(batch)


@pytest.mark.parametrize(
    "mode", [CaptureHiddenModeDouble.LAST, CaptureHiddenModeDouble.FULL]
)
def test_a_real_capture_mode_is_refused(mode):
    made = adapter()
    batch = made.build_forward_batch(extend_inputs())
    batch.capture_hidden_mode = mode
    with pytest.raises(DraftLifecycleError, match="must not capture"):
        made._assert_prediction_only(batch)


def test_the_extend_path_supplies_extend_num_tokens():
    """init_new copies it off the ScheduleBatch, which this adapter lacks."""
    fields = adapter().forward_fields(extend_inputs(5))
    assert fields["extend_num_tokens"] == 5
    assert fields["extend_seq_lens_cpu"] == [5]
    assert fields["extend_prefix_lens_cpu"] == [0]


def test_extend_start_loc_is_the_prefix_sum_not_a_constant():
    made = adapter()
    fields = made.forward_fields(extend_inputs(4))
    assert fields["extend_start_loc"].tolist() == [0]
    wide = DraftForwardInputs(
        forward_mode="extend",
        input_ids=tuple(range(7)),
        positions=tuple(range(7)),
        seq_lens=(3, 4),
        req_pool_indices=(1, 2),
        out_cache_loc=tuple(range(100, 107)),
        extend_prefix_lens=(0, 0),
        extend_seq_lens=(3, 4),
    )
    fields = made.forward_fields(wide)
    assert fields["extend_start_loc"].tolist() == [0, 3]
    assert fields["extend_num_tokens"] == 7


def test_extend_metadata_that_disagrees_with_the_ids_is_refused():
    made = adapter()
    bad = DraftForwardInputs(
        forward_mode="extend",
        input_ids=(1, 2, 3),
        positions=(0, 1, 2),
        seq_lens=(3,),
        req_pool_indices=(1,),
        out_cache_loc=(10, 11, 12),
        extend_prefix_lens=(0,),
        extend_seq_lens=(9,),
    )
    with pytest.raises(DraftLifecycleError, match="disagrees with"):
        made.forward_fields(bad)


def test_the_cpu_mirror_of_the_sequence_lengths_is_populated():
    fields = adapter().forward_fields(extend_inputs(3))
    assert fields["seq_lens_cpu"].tolist() == [3]
    assert fields["seq_lens_cpu"].device == torch.device("cpu")


def test_a_decode_batch_carries_no_extend_metadata():
    fields = adapter().forward_fields(decode_inputs(position=5))
    for name in (
        "extend_seq_lens",
        "extend_prefix_lens",
        "extend_num_tokens",
        "extend_start_loc",
    ):
        assert name not in fields
    assert fields["positions"].tolist() == [5]
    assert fields["seq_lens"].tolist() == [6]


def test_an_extend_batch_carries_the_whole_prefix_from_position_zero():
    batch = adapter().build_forward_batch(extend_inputs(3))
    assert batch.forward_mode == "extend"
    assert batch.batch_size == 1
    assert batch.input_ids.tolist() == [10, 11, 12]
    assert batch.positions.tolist() == [0, 1, 2]
    assert batch.seq_lens.tolist() == [3]
    assert batch.seq_lens_sum == 3
    assert batch.out_cache_loc.tolist() == [100, 101, 102]
    assert batch.req_pool_indices.tolist() == [2]


def test_the_batch_tensors_are_on_the_declared_device_with_the_right_dtypes():
    batch = adapter().build_forward_batch(extend_inputs())
    for name in (
        "input_ids",
        "positions",
        "seq_lens",
        "out_cache_loc",
        "req_pool_indices",
    ):
        assert getattr(batch, name).device == torch.device("cpu")
        assert getattr(batch, name).dtype is torch.int64
    assert batch.extend_seq_lens.dtype is torch.int32
    assert batch.extend_start_loc.dtype is torch.int32


@pytest.mark.parametrize("name", MUST_STAY_UNSET)
def test_a_prediction_batch_carries_no_speculative_payload(name):
    made = adapter()
    batch = made.build_forward_batch(extend_inputs())
    assert getattr(batch, name, None) is None
    setattr(batch, name, object())
    with pytest.raises(DraftLifecycleError, match="must not carry"):
        made._assert_prediction_only(batch)


def test_no_logprob_accounting_is_requested():
    assert adapter().forward_fields(extend_inputs())["return_logprob"] is False


def test_the_forward_runs_under_inference_mode():
    seen = {}

    class Checking(RealShapedModelRunner):
        def forward(self, forward_batch):
            seen["grad"] = torch.is_grad_enabled()
            return super().forward(forward_batch)

    adapter(Checking()).forward(extend_inputs())
    assert seen["grad"] is False


# --------------------------------------------------------------------------
# 4. Allocator device and page semantics
# --------------------------------------------------------------------------


def test_release_indices_use_the_allocators_device_and_dtype():
    kv = RealShapedKVAllocator()
    alloc = PrivatePoolAllocator(RealShapedReqPool(), kv)
    rows = alloc.alloc_kv(3)
    # The double raises on a device or dtype mismatch, as the real one would.
    alloc.free_kv(rows)
    assert kv.freed == [rows]
    assert kv.free_pages.dtype is torch.int64


def test_a_paged_allocator_is_refused_before_anything_is_allocated():
    paged = RealShapedPagedAllocator(page_size=16)
    with pytest.raises(DraftCapabilityError, match="page_size 16"):
        PrivatePoolAllocator(RealShapedReqPool(), paged)
    # Nothing was taken from it.
    assert len(paged.free_pages) == paged.size


def test_the_supported_page_size_is_stated_and_accepted():
    assert SUPPORTED_PAGE_SIZE == 1
    alloc = PrivatePoolAllocator(RealShapedReqPool(), RealShapedKVAllocator())
    assert alloc.page_size == 1


def test_a_partial_allocation_is_refused_and_handed_back():
    kv = RealShapedKVAllocator(short_by=1)
    alloc = PrivatePoolAllocator(RealShapedReqPool(), kv)
    before = len(kv.free_pages)
    with pytest.raises(DraftLifecycleError, match="supplied 2 of 3"):
        alloc.alloc_kv(3)
    # The rows that did arrive were released, not leaked.
    assert kv.freed, "a partial allocation was kept"
    assert len(kv.free_pages) == before


def test_an_exhausted_allocator_is_refused():
    kv = RealShapedKVAllocator(size=2)
    alloc = PrivatePoolAllocator(RealShapedReqPool(), kv)
    with pytest.raises(DraftLifecycleError, match="could not supply"):
        alloc.alloc_kv(8)


def test_freeing_nothing_is_harmless_and_repeatable():
    kv = RealShapedKVAllocator()
    alloc = PrivatePoolAllocator(RealShapedReqPool(), kv)
    alloc.free_kv([])
    alloc.free_kv(None)
    alloc.free_kv(torch.empty(0, dtype=torch.int64))
    assert kv.freed == []


def test_a_nonpositive_allocation_is_refused():
    alloc = private_pools()
    for count in (0, -1):
        with pytest.raises(DraftLifecycleError, match="must be positive"):
            alloc.alloc_kv(count)


# --------------------------------------------------------------------------
# 5. No unbounded retention
# --------------------------------------------------------------------------


def adapter_state_size(made):
    """How much the adapter is holding: containers, tensors, batch objects."""
    held = 0
    for value in vars(made).values():
        if isinstance(value, torch.Tensor):
            held += value.numel()
        elif isinstance(value, (list, tuple, set)):
            held += len(value)
        elif isinstance(value, dict):
            held += len(value)
    return held


def test_the_adapter_does_not_accumulate_anything_across_forwards():
    """A retained ForwardBatch pins its device tensors for the adapter's life.

    Deterministic by construction: the adapter's own state is measured after
    one forward and after many. Whether a particular object has been
    collected depends on the whole process, so that is deliberately not what
    is asserted here.
    """
    made = adapter()
    made.forward(extend_inputs())
    after_one = adapter_state_size(made)
    for _ in range(20):
        made.forward(extend_inputs())
    assert adapter_state_size(made) == after_one, (
        "the adapter's state grew with the number of forwards"
    )
    assert made.forward_count == 21
    # No collection for batches to accumulate in, under any name.
    assert not hasattr(made, "batches")
    for name, value in vars(made).items():
        assert not isinstance(value, torch.Tensor), name
        if isinstance(value, (list, tuple, set, dict)):
            items = value.values() if isinstance(value, dict) else value
            assert not any(
                isinstance(v, torch.Tensor) or hasattr(v, "out_cache_loc")
                for v in items
            ), f"{name} holds tensors or batches"


def test_diagnostics_are_bounded_and_hold_no_tensors():
    made = adapter()
    made.forward(extend_inputs(3))
    made.forward(decode_inputs(position=3))
    assert made.forward_count == 2
    # Only the most recent, and only plain data.
    assert made.last_forward["forward_mode"] == "decode"
    assert made.last_forward["num_tokens"] == 1
    for value in made.last_forward.values():
        assert not isinstance(value, torch.Tensor)


# --------------------------------------------------------------------------
# Field names, against upstream's source
# --------------------------------------------------------------------------


def forward_batch_fields():
    """Every ForwardBatch field, and which ones have no default.

    Read as source: importing the module pulls in triton, torchvision and the
    HTTP stack, which this suite deliberately does not require.
    """
    tree = ast.parse(
        (SRT / "model_executor" / "forward_batch_info.py").read_text(encoding="utf-8")
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ForwardBatch":
            names, required = set(), set()
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    names.add(stmt.target.id)
                    if stmt.value is None:
                        required.add(stmt.target.id)
            return names, required
    raise AssertionError("ForwardBatch not found")


def test_every_mapped_field_is_a_real_forward_batch_field():
    names, _ = forward_batch_fields()
    made = adapter()
    for inputs in (extend_inputs(), decode_inputs()):
        unknown = set(made.forward_fields(inputs)) - names
        assert not unknown, f"not ForwardBatch fields: {sorted(unknown)}"


def test_every_field_without_a_default_is_supplied():
    names, required = forward_batch_fields()
    assert required, "ForwardBatch appears to have no required fields"
    made = adapter()
    for inputs in (extend_inputs(), decode_inputs()):
        missing = required - set(made.forward_fields(inputs))
        assert not missing, f"required ForwardBatch fields missing: {sorted(missing)}"


def test_the_doubles_match_the_shipped_request_pool_contract():
    """The doubles are derived from source, not from what the adapter sends."""
    source = (SRT / "mem_cache" / "memory_pool.py").read_text(encoding="utf-8")
    assert "def alloc(self, reqs: list[Req]) -> Optional[List[int]]:" in source
    assert "def free(self, req: Req):" in source
    assert (
        'assert req.req_pool_idx is not None, "request must have req_pool_idx"'
        in source
    )
    assert "self.free_slots = list(range(1, self._alloc_size))" in source


def test_the_doubles_match_the_shipped_allocator_contract():
    token = (SRT / "mem_cache" / "allocator" / "token.py").read_text(encoding="utf-8")
    assert "1, self.size + 1, dtype=torch.int64, device=self.device" in token
    assert "torch.cat((self.free_pages, free_index))" in token
    paged = (SRT / "mem_cache" / "allocator" / "paged.py").read_text(encoding="utf-8")
    assert "need_size % self.page_size == 0" in paged


def test_the_capture_mode_contract_is_what_the_adapter_assumes():
    source = (SRT / "model_executor" / "forward_batch_info.py").read_text(
        encoding="utf-8"
    )
    assert "class CaptureHiddenMode(IntEnum):" in source
    assert "NULL = 0" in source
    assert "def need_capture(self):" in source


def test_the_model_runner_output_contract_is_what_the_adapter_unwraps():
    runner = (SRT / "model_executor" / "model_runner.py").read_text(encoding="utf-8")
    assert "class ModelRunnerOutput:" in runner
    assert "logits_output: Union[LogitsProcessorOutput, PPProxyTensors]" in runner
    logits = (SRT / "layers" / "logits_processor.py").read_text(encoding="utf-8")
    assert "next_token_logits: Optional[torch.Tensor]" in logits


# --------------------------------------------------------------------------
# The private request map, end to end over the real-shaped pools
# --------------------------------------------------------------------------


def handle(alloc=None, executor=None, **kwargs):
    kwargs.setdefault("max_prefix_tokens", 64)
    kwargs.setdefault("max_tokens", 4)
    return SGLangDraftHandle(
        "b",
        executor or FakeExecutor(sequence=[31, 32, 33]),
        alloc or FakeAllocator(),
        capabilities=DEFAULT_CAPABILITIES,
        **kwargs,
    )


def test_the_prefix_is_mapped_before_the_forward_that_reads_it():
    alloc = FakeAllocator()
    made = handle(alloc)
    made.prepare_prefix((10, 11, 12))
    index = made.request_index
    assert alloc.req_to_token[index, :3].tolist() == list(made.owned_kv[:3])
    assert made.mapped_tokens == 3
    assert alloc.req_to_token[index, 3:].sum().item() == 0


def test_each_step_extends_the_map_by_exactly_one_position():
    alloc = FakeAllocator()
    made = handle(alloc)
    made.generate(made.prepare_prefix((10, 11)), 3)
    assert made.mapped_tokens == 4
    assert [start for _, start, _ in alloc.mapping_writes] == [0, 2, 3]


def test_only_this_handles_row_is_ever_written():
    alloc = FakeAllocator()
    other = alloc.req_to_token[1].clone()
    made = handle(alloc)
    made.generate(made.prepare_prefix((10, 11, 12)), 2)
    assert made.request_index == 0
    assert torch.equal(alloc.req_to_token[1], other)


def test_releasing_clears_the_map_before_returning_the_slot():
    requests = RealShapedReqPool()
    kv = RealShapedKVAllocator()
    alloc = PrivatePoolAllocator(requests, kv)
    made = handle(alloc)
    made.generate(made.prepare_prefix((10, 11, 12)), 2)
    index = made.request_index
    assert requests.req_to_token[index].sum().item() > 0
    made.release()
    # Cleared, and only then returned to the free list.
    assert requests.req_to_token[index].sum().item() == 0
    assert requests.freed == [index]


def test_a_reused_slot_starts_clean_over_the_real_shaped_pool():
    requests = RealShapedReqPool(size=1)
    kv = RealShapedKVAllocator()
    first = handle(PrivatePoolAllocator(requests, kv))
    first.prepare_prefix((10, 11, 12))
    first.release()
    second = handle(PrivatePoolAllocator(requests, kv))
    second.prepare_prefix((20,))
    index = second.request_index
    assert requests.req_to_token[index, 1:].sum().item() == 0


def test_a_full_prediction_round_drives_the_real_shaped_pools():
    requests = RealShapedReqPool()
    kv = RealShapedKVAllocator()
    alloc = PrivatePoolAllocator(requests, kv)
    runner = RealShapedModelRunner(rows=3)
    made = SGLangDraftHandle(
        "b",
        adapter(runner),
        alloc,
        max_prefix_tokens=16,
        max_tokens=4,
        capabilities=DEFAULT_CAPABILITIES,
    )
    produced = made.generate(made.prepare_prefix((10, 11, 12)), 2)
    assert produced == [7, 7]
    assert [mode for mode, _ in runner.seen_shapes] == ["extend", "decode"]
    index = made.request_index
    assert requests.req_to_token[index, :4].tolist() == list(made.owned_kv[:4])
    made.release()
    assert requests.req_to_token[index].sum().item() == 0
    assert requests.freed == [index]
    assert kv.freed, "KV rows were never released"


# --------------------------------------------------------------------------
# Real object construction
#
# These build the ACTUAL ForwardBatch, ForwardMode and CaptureHiddenMode, not
# doubles. They skip where SGLang's serving frontend cannot be imported, and
# the skip reason names the exact dependency so "blocked" is never a guess.
#
# Even here, nothing is EXECUTED: no model, no attention backend, no forward.
# Construction succeeding is not the same as a forward succeeding.
# --------------------------------------------------------------------------


def real_forward_batch_import():
    """Import the real classes, or return the exact failure."""
    try:
        from sglang.srt.model_executor.forward_batch_info import (  # noqa: F401
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )
    except Exception as exc:  # ModuleNotFoundError, RuntimeError, ValueError...
        return None, f"{type(exc).__name__}: {exc}"
    return (ForwardBatch, ForwardMode, CaptureHiddenMode), None


REAL_IMPORT, REAL_IMPORT_ERROR = real_forward_batch_import()
needs_real = pytest.mark.skipif(
    REAL_IMPORT is None,
    reason=f"SGLang serving frontend unavailable -> {REAL_IMPORT_ERROR}",
)


class NeverRuns:
    """A runner that fails loudly if anything tries to execute a forward."""

    device = "cpu"

    def forward(self, batch):  # pragma: no cover - must not be reached
        raise AssertionError("no forward may be executed by a construction test")


@needs_real
def test_a_real_forward_batch_is_constructed_for_extend():
    ForwardBatch, ForwardMode, CaptureHiddenMode = REAL_IMPORT
    made = DraftForwardAdapter(
        NeverRuns(),
        architecture="LlamaForCausalLM",
        attention_backend="triton",
        bytes_per_token=8,
    )
    batch = made.build_forward_batch(extend_inputs(3))
    assert isinstance(batch, ForwardBatch)
    assert batch.forward_mode is ForwardMode.EXTEND
    assert batch.input_ids.tolist() == [10, 11, 12]
    assert batch.positions.tolist() == [0, 1, 2]
    assert batch.seq_lens.tolist() == [3]
    assert batch.seq_lens_sum == 3
    assert batch.extend_num_tokens == 3
    assert batch.extend_seq_lens_cpu == [3]
    assert batch.extend_start_loc.tolist() == [0]


@needs_real
def test_a_real_forward_batch_is_constructed_for_decode():
    ForwardBatch, ForwardMode, _ = REAL_IMPORT
    made = DraftForwardAdapter(
        NeverRuns(),
        architecture="LlamaForCausalLM",
        attention_backend="triton",
        bytes_per_token=8,
    )
    batch = made.build_forward_batch(decode_inputs(position=5))
    assert isinstance(batch, ForwardBatch)
    assert batch.forward_mode is ForwardMode.DECODE
    assert batch.positions.tolist() == [5]
    assert batch.seq_lens.tolist() == [6]
    # A decode carries no extend bookkeeping on the real object either.
    assert batch.extend_num_tokens is None


@needs_real
def test_the_real_capture_mode_is_the_enum_and_disables_capture():
    """The defect this fixes: None has no need_capture()."""
    _, _, CaptureHiddenMode = REAL_IMPORT
    made = DraftForwardAdapter(
        NeverRuns(),
        architecture="LlamaForCausalLM",
        attention_backend="triton",
        bytes_per_token=8,
    )
    batch = made.build_forward_batch(extend_inputs())
    assert batch.capture_hidden_mode is CaptureHiddenMode.NULL
    # The call the logits processor makes, on the real enum.
    assert batch.capture_hidden_mode.need_capture() is False
    assert batch.capture_hidden_mode.is_full() is False
    assert batch.capture_hidden_mode.is_last() is False


@needs_real
def test_a_real_forward_batch_passes_the_prediction_only_assertions():
    made = DraftForwardAdapter(
        NeverRuns(),
        architecture="LlamaForCausalLM",
        attention_backend="triton",
        bytes_per_token=8,
    )
    for inputs in (extend_inputs(), decode_inputs()):
        batch = made.build_forward_batch(inputs)
        made._assert_prediction_only(batch)
        assert batch.spec_info is None
        assert batch.return_logprob is False


def real_pool_import():
    """Import the real pools, or return the exact failure."""
    try:
        from sglang.srt.mem_cache.allocator.token import (  # noqa: F401
            TokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool  # noqa: F401
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return (ReqToTokenPool, TokenToKVPoolAllocator), None


REAL_POOLS, REAL_POOLS_ERROR = real_pool_import()
needs_real_pools = pytest.mark.skipif(
    REAL_POOLS is None,
    reason=f"SGLang memory pools unavailable -> {REAL_POOLS_ERROR}",
)


@needs_real_pools
def test_the_real_request_pool_and_allocator_drive_a_whole_round():
    """The actual ReqToTokenPool and TokenToKVPoolAllocator, not doubles."""
    ReqToTokenPool, TokenToKVPoolAllocator = REAL_POOLS

    requests = ReqToTokenPool(
        size=4, max_context_len=64, device="cpu", enable_memory_saver=False
    )
    kv = TokenToKVPoolAllocator.__new__(TokenToKVPoolAllocator)
    kv.size = 64
    kv.page_size = 1
    kv.device = "cpu"
    kv.need_sort = False
    kv.clear()

    alloc = PrivatePoolAllocator(requests, kv)
    index = alloc.alloc_request()
    assert index >= 1, "slot 0 is the padding row and must not be issued"
    rows = alloc.alloc_kv(3)
    alloc.write_mapping(index, 0, rows)
    assert requests.req_to_token[index, :3].tolist() == rows
    # free() on the real allocator concatenates onto int64 free_pages.
    alloc.free_kv(rows)
    alloc.clear_mapping(index)
    assert requests.req_to_token[index].sum().item() == 0
    alloc.free_request(index)
    assert index in requests.free_slots
    assert alloc._request.req_pool_idx is None
