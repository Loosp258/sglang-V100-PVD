"""Prediction-only reuse of SGLang's draft-worker infrastructure.

What these establish: the lifecycle, the isolation checks and the refusals,
with a fake runner standing in for the forward pass. What they do NOT
establish: that any real model loads, that CUDA RNG or GPU memory is
isolated, that predictions are any good, or that co-locating draft and target
overlaps compute. No weights are loaded and no device is used.
"""

import ast
from pathlib import Path

import pytest
import torch
from sglang.srt.disaggregation.pvd.draft_hf import VocabularySignature
from sglang.srt.disaggregation.pvd.draft_runner_sglang import (
    DEFAULT_CAPABILITIES,
    DraftForwardInputs,
    SGLangDraftHandle,
    SGLangDraftRunnerFactory,
)
from sglang.srt.disaggregation.pvd.draft_sglang import (
    ALLOWED_WORKER_METHODS,
    KNOWN_GENERATION_METHODS,
    DraftCapabilities,
    DraftCapabilityError,
    DraftLifecycleError,
    DraftPlacement,
    DraftWorkerError,
    DraftWorkerInterface,
    SGLangDraftProvider,
    build_draft_server_args,
    require_private_pools,
)
from sglang.srt.disaggregation.pvd.prediction import (
    DraftConfig,
    PredictionConfigError,
    PredictionPipeline,
    ProbeConfig,
    snapshot_committed,
)
from sglang.srt.disaggregation.pvd.probe_search import (
    ProbeSearchRoute,
    ProbeSearchSession,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferCapacityError,
)

SRT = Path(__file__).resolve().parents[3] / "python" / "sglang" / "srt"


# --------------------------------------------------------------------------
# The audit, in executable form
#
# The reuse plan rests on two facts about upstream code. If either changes,
# these fail and the plan is re-examined rather than silently invalidated.
# --------------------------------------------------------------------------


def test_sglang_standalone_worker_really_does_share_the_targets_pools():
    """Why draft() is not the reuse point, asserted against the source."""
    source = (SRT / "speculative" / "standalone_worker.py").read_text(encoding="utf-8")
    assert "target_worker.get_memory_pool()" in source
    assert "Share the allocator with a target worker" in source
    # StandaloneWorker extends EAGLEWorker, whose cache clearing is a no-op
    # precisely because of that sharing.
    assert "class StandaloneWorker(EAGLEWorker)" in source
    eagle = ast.parse(
        (SRT / "speculative" / "eagle_worker.py").read_text(encoding="utf-8")
    )
    clear = [
        node
        for node in ast.walk(eagle)
        if isinstance(node, ast.FunctionDef) and node.name == "clear_cache_pool"
    ]
    assert clear, "EAGLEWorker.clear_cache_pool disappeared"
    body = [n for n in clear[0].body if not isinstance(n, ast.Expr)]
    assert all(isinstance(stmt, ast.Pass) for stmt in body)


def test_eagle_draft_preprocess_mutates_committed_state():
    """The six mutations the reuse plan refuses to inherit."""
    source = (SRT / "speculative" / "eagle_worker.py").read_text(encoding="utf-8")
    body = source[source.index("def _draft_preprocess_decode") :]
    body = body[: body.index("def _draft_preprocess_idle")]
    for fragment in (
        "req.decode_batch_idx += 1",  # committed per-request counter
        "cumulate_output_tokens",  # committed sampler penalties
        "batch.maybe_evict_swa()",  # shared cache eviction
        "req_to_token_pool.req_to_token",  # live request -> slot mapping
        "batch.out_cache_loc =",  # live batch fields
        "batch.seq_lens_sum =",
    ):
        assert fragment in body, f"{fragment} vanished; re-audit the reuse plan"
    # The allocator is restored. Nothing restores req_to_token.
    assert "restore_state" in body


# --------------------------------------------------------------------------
# Doubles
#
# These stand in for the model runner and the allocator. They record what the
# orchestration actually asked for, which is what the tests assert against.
# No weights, no device, no ForwardBatch.
# --------------------------------------------------------------------------


class FakePool:
    def __init__(self, name, buffer=None):
        self.name = name
        self.req_to_token = (
            torch.zeros(4, 16, dtype=torch.int32) if buffer is None else buffer
        )


class FakeWorker:
    def __init__(self, req_pool, kv_pool):
        self._pools = (req_pool, kv_pool)
        self.model_config = {"architecture": "LlamaForCausalLM"}
        self.device = "cpu"
        self.generation_calls = 0

    def get_memory_pool(self):
        return self._pools

    def draft(self, *a, **k):  # pragma: no cover - must never be reached
        self.generation_calls += 1
        raise AssertionError("generation path entered")

    def verify(self, *a, **k):  # pragma: no cover
        self.generation_calls += 1
        raise AssertionError("generation path entered")

    def forward_batch_generation(self, *a, **k):  # pragma: no cover
        self.generation_calls += 1
        raise AssertionError("generation path entered")

    def some_unreviewed_method(self):  # pragma: no cover
        raise AssertionError("an unlisted method was reached")


class FakeAllocator:
    """Private slots, rows and request map, with a ledger so leaks show."""

    def __init__(self, *, fail_free=False, rows=4096, slots=8, width=64):
        self.next_request = 0
        self.next_kv = 0
        self.rows = rows
        self.live_requests = set()
        self.live_kv = set()
        self.fail_free = fail_free
        self.freed_kv = []
        # Stands in for the private req_to_token_pool's table.
        self.req_to_token = torch.zeros(slots, width, dtype=torch.int32)
        self.mapping_writes = []

    def fork_for_branch(self):
        # This double stores a per-index ledger, not one mutable request handle.
        # Real PrivatePoolAllocator forks an actual branch-local request object.
        return self

    def write_mapping(self, request_index, start, locations):
        values = torch.tensor(list(locations), dtype=torch.int32)
        self.req_to_token[request_index, start : start + len(values)] = values
        self.mapping_writes.append((request_index, start, tuple(locations)))

    def clear_mapping(self, request_index):
        self.req_to_token[request_index].fill_(0)

    def alloc_request(self):
        index = self.next_request
        self.next_request += 1
        self.live_requests.add(index)
        return index

    def free_request(self, index):
        if self.fail_free:
            raise RuntimeError("allocator refused to free the request slot")
        self.live_requests.discard(index)

    def alloc_kv(self, count):
        if self.next_kv + count > self.rows:
            raise RuntimeError("out of KV rows")
        out = list(range(self.next_kv, self.next_kv + count))
        self.next_kv += count
        self.live_kv.update(out)
        return out

    def free_kv(self, locations):
        if self.fail_free:
            raise RuntimeError("allocator refused to free KV rows")
        self.freed_kv.append(tuple(locations))
        self.live_kv.difference_update(locations)


class FakeExecutor:
    """Returns a deterministic next token and records every forward."""

    def __init__(
        self,
        *,
        vocab=64,
        arch="LlamaForCausalLM",
        backend="triton",
        bytes_per_token=8,
        sequence=None,
    ):
        self.vocab = vocab
        self.arch = arch
        self.backend = backend
        self._bytes = bytes_per_token
        self.sequence = list(sequence or [31, 32, 33, 34])
        self.calls = []

    def architecture(self):
        return self.arch

    def drain(self):
        # All work in this double is synchronous CPU computation.
        pass

    def attention_backend(self):
        return self.backend

    def bytes_per_token(self):
        return self._bytes

    def transient_bytes(self, prefix_tokens, predict_tokens):
        # Test-double forward only produces this small CPU logits vector.
        # Include simultaneous prepared/current/next logits; not a real-model bound.
        return 3 * self.vocab * 4

    def forward(self, inputs):
        self.calls.append(inputs)
        logits = torch.zeros(self.vocab)
        token = self.sequence[min(len(self.calls) - 1, len(self.sequence) - 1)]
        logits[token] = 10.0
        return logits


def factory(executor=None, allocator=None, **kwargs):
    kwargs.setdefault("persistent_bytes", 1024)
    kwargs.setdefault("max_tokens", 4)
    return SGLangDraftRunnerFactory(
        executor or FakeExecutor(), allocator or FakeAllocator(), **kwargs
    )


def provider(fac=None, **kwargs):
    config = kwargs.pop("config", None) or DraftConfig(
        "configurable/draft", predict_tokens=2
    )
    placement = kwargs.pop("placement", None) or DraftPlacement(
        scratch_budget_bytes=1 << 20, persistent_budget_bytes=1 << 20
    )
    return SGLangDraftProvider(config, placement, fac or factory(), **kwargs)


def prefix(request_id="request", version="prefix-v1", tokens=(10, 11, 12, 13)):
    return snapshot_committed(request_id, list(tokens), 2, version)


# --------------------------------------------------------------------------
# Gap 1 + 2 + 3: cleanup ordering and branch-owned resources
# --------------------------------------------------------------------------


def test_budget_and_admission_outlive_the_resources_they_stand_for():
    """The reservation must not be refunded before the handle is released."""
    observed = []
    alloc = FakeAllocator()
    made = provider(factory(allocator=alloc))

    class Watching(SGLangDraftHandle):
        def release(self):
            observed.append(
                (
                    "release.enter",
                    made.scratch_budget.snapshot()["used_staging_bytes"],
                    made.active_branches,
                )
            )
            super().release()
            observed.append(
                (
                    "release.exit",
                    made.scratch_budget.snapshot()["used_staging_bytes"],
                    made.active_branches,
                )
            )

    made.factory.open = lambda **kw: Watching(
        kw["branch_id"],
        FakeExecutor(),
        alloc,
        max_prefix_tokens=kw["prefix_tokens"],
        max_tokens=kw["max_tokens"],
        capabilities=DEFAULT_CAPABILITIES,
    )
    with made.branch():
        made.predict(prefix(), 2)
    # While release runs, both the bytes and the admission slot are still held.
    assert observed[0][1] > 0, "scratch was refunded before cleanup ran"
    assert observed[0][2] == 1, "admission reopened before cleanup ran"
    # Only afterwards are they given back.
    assert made.scratch_budget.snapshot()["used_staging_bytes"] == 0
    assert made.active_branches == 0
    assert not alloc.live_kv and not alloc.live_requests


def test_a_failed_release_quarantines_rather_than_reissuing():
    """Resources that may still be live are never handed to a second owner."""
    alloc = FakeAllocator(fail_free=True)
    made = provider(factory(allocator=alloc))
    with pytest.raises(DraftWorkerError, match="could not be released"):
        with made.branch():
            made.predict(prefix(), 2)
    assert made.degraded
    quarantined = made.quarantined
    assert len(quarantined) == 1 and quarantined[0].reserved_bytes > 0
    # The bytes stay charged: the rows may still be live.
    assert made.scratch_budget.snapshot()["used_staging_bytes"] > 0
    # And the admission slot is gone for good, so nothing reuses it.
    with pytest.raises(TransferCapacityError):
        with made.branch():
            pass  # pragma: no cover


def test_each_branch_gets_its_own_handle_and_releases_only_its_own_rows():
    alloc = FakeAllocator()
    fac = factory(allocator=alloc)
    made = provider(fac)
    with made.branch():
        made.predict(prefix(), 2)
    first = tuple(alloc.freed_kv)
    with made.branch():
        made.predict(prefix(), 2)
    second = tuple(alloc.freed_kv)[len(first) :]
    assert len(fac.opened) == 2 and len(set(fac.opened)) == 2
    assert first and second
    # Disjoint rows: no branch freed another's.
    assert not set(first[0]) & set(second[0])
    assert not alloc.live_kv and not alloc.live_requests


def test_a_handle_refuses_a_prepared_prefix_from_another_handle():
    alloc = FakeAllocator()
    executor = FakeExecutor()
    one = SGLangDraftHandle(
        "a",
        executor,
        alloc,
        max_prefix_tokens=64,
        max_tokens=4,
        capabilities=DEFAULT_CAPABILITIES,
    )
    two = SGLangDraftHandle(
        "b",
        executor,
        alloc,
        max_prefix_tokens=64,
        max_tokens=4,
        capabilities=DEFAULT_CAPABILITIES,
    )
    prepared = one.prepare_prefix((1, 2, 3))
    two.prepare_prefix((4, 5))
    with pytest.raises(DraftLifecycleError, match="another handle"):
        two.generate(prepared, 1)


def test_a_released_handle_refuses_further_execution():
    alloc = FakeAllocator()
    handle = SGLangDraftHandle(
        "a",
        FakeExecutor(),
        alloc,
        max_prefix_tokens=64,
        max_tokens=4,
        capabilities=DEFAULT_CAPABILITIES,
    )
    prepared = handle.prepare_prefix((1, 2))
    handle.release()
    assert handle.released
    handle.release()  # idempotent
    with pytest.raises(DraftLifecycleError, match="has been released"):
        handle.generate(prepared, 1)


# --------------------------------------------------------------------------
# Gap 4: the calling contract
# --------------------------------------------------------------------------


def test_predict_outside_a_branch_is_refused():
    made = provider()
    with pytest.raises(DraftLifecycleError, match="must be called inside branch"):
        made.predict(prefix(), 2)
    assert made.scratch_budget.snapshot()["used_staging_bytes"] == 0
    assert made.active_branches == 0


def test_branches_do_not_nest():
    made = provider(
        factory(),
        placement=DraftPlacement(
            scratch_budget_bytes=1 << 20,
            persistent_budget_bytes=1 << 20,
            max_concurrent_branches=2,
        ),
    )
    with made.branch():
        with pytest.raises(DraftLifecycleError, match="do not nest"):
            with made.branch():
                pass  # pragma: no cover


def test_the_branch_token_does_not_leak_to_another_thread():
    import threading

    made = provider(
        factory(),
        placement=DraftPlacement(
            scratch_budget_bytes=1 << 20,
            persistent_budget_bytes=1 << 20,
            max_concurrent_branches=2,
        ),
    )
    seen = []
    ready = threading.Event()

    def other():
        ready.wait(5)
        try:
            made.predict(prefix(), 2)
        except DraftLifecycleError as exc:
            seen.append(exc)

    worker = threading.Thread(target=other)
    worker.start()
    with made.branch():
        ready.set()
        worker.join(5)
    assert seen, "another thread predicted using this thread's branch"


# --------------------------------------------------------------------------
# Gap 5: configuration translation
# --------------------------------------------------------------------------


def draft_args(**overrides):
    from types import SimpleNamespace

    base = dict(
        model_path="target/model-8b",
        tokenizer_path="target/model-8b",
        revision="target-rev",
        device="cuda:0",
        speculative_algorithm=None,
        speculative_num_steps=3,
        disaggregation_mode="decode",
        disaggregation_topology="pvd",
        pvd_draft_model_path="configurable/draft",
        pvd_draft_revision="draft-rev",
        pvd_draft_device="cuda:1",
        mem_fraction_static=0.78,
        max_running_requests=256,
        pvd_draft_mem_fraction_static=0.08,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_the_draft_settings_are_translated_into_the_fields_the_loader_reads():
    args = draft_args()
    private = build_draft_server_args(args, DraftPlacement(scratch_budget_bytes=4096))
    assert private.model_path == "configurable/draft"
    assert private.tokenizer_path == "configurable/draft"
    assert private.revision == "draft-rev"
    assert private.device == "cuda:1"
    assert private.mem_fraction_static == 0.08
    assert private.max_running_requests == 1
    # Speculative and disaggregation settings are off inside the copy.
    assert private.speculative_algorithm is None
    assert private.speculative_num_steps is None
    assert private.disaggregation_mode is None
    assert private.disaggregation_topology is None


def test_translating_leaves_the_target_configuration_untouched():
    args = draft_args()
    before = vars(args).copy()
    build_draft_server_args(args, DraftPlacement(scratch_budget_bytes=4096))
    assert vars(args) == before, "the target's configuration was modified"
    assert args.model_path == "target/model-8b"
    assert args.device == "cuda:0"


@pytest.mark.parametrize("fraction", [None, False, 0, 1, -0.1, 1.5, "0.1"])
def test_private_draft_never_inherits_target_kv_pool_fraction(fraction):
    with pytest.raises(DraftWorkerError, match="pvd-draft-mem-fraction-static"):
        build_draft_server_args(
            draft_args(pvd_draft_mem_fraction_static=fraction),
            DraftPlacement(scratch_budget_bytes=4096),
        )


def test_stale_native_draft_loader_fields_cannot_override_pvd_model():
    args = draft_args(
        speculative_draft_model_path="stale/other-model",
        speculative_draft_model_revision="stale-revision",
        speculative_draft_model_quantization="awq",
    )
    before = vars(args).copy()
    private = build_draft_server_args(args, DraftPlacement(scratch_budget_bytes=4096))
    # TpModelWorker reads these for is_draft_worker=True. ModelConfig uses
    # the normal path/revision only when these speculative overrides are None.
    assert private.speculative_draft_model_path is None
    assert private.speculative_draft_model_revision is None
    assert private.speculative_draft_model_quantization is None
    assert private.model_path == "configurable/draft"
    assert private.revision == "draft-rev"
    assert vars(args) == before


def test_translating_without_a_draft_model_is_refused():
    with pytest.raises(DraftWorkerError, match="no draft model is configured"):
        build_draft_server_args(
            draft_args(pvd_draft_model_path=None),
            DraftPlacement(scratch_budget_bytes=4096),
        )


def test_the_worker_is_constructed_with_private_pools_and_the_draft_config():
    """Asserted through an injectable factory, so no weights are loaded."""
    from sglang.srt.disaggregation.pvd.draft_sglang import (
        build_prediction_only_worker,
    )

    captured = {}

    def fake_factory(**kwargs):
        captured.update(kwargs)
        return FakeWorker(FakePool("draft-req"), FakePool("draft-kv"))

    target = FakeWorker(FakePool("target-req"), FakePool("target-kv"))
    worker, ownership = build_prediction_only_worker(
        draft_args(),
        placement=DraftPlacement(scratch_budget_bytes=4096, gpu_id=1, tp_rank=0),
        nccl_port=1234,
        target_worker=target,
        worker_factory=fake_factory,
    )
    assert captured["is_draft_worker"] is True
    # The two arguments that make the pools private.
    assert captured["req_to_token_pool"] is None
    assert captured["token_to_kv_pool_allocator"] is None
    assert captured["server_args"].model_path == "configurable/draft"
    assert captured["server_args"] is not draft_args()
    assert captured["gpu_id"] == 1 and captured["nccl_port"] == 1234
    assert captured["server_args"].device == "cuda:1"
    assert isinstance(worker, DraftWorkerInterface)
    assert ownership.distinct_objects


@pytest.mark.parametrize("device", ["cuda:0", "cuda", "cpu", "not-a-device"])
def test_worker_refuses_a_device_that_would_disagree_with_gpu_id(device):
    from sglang.srt.disaggregation.pvd.draft_sglang import (
        build_prediction_only_worker,
    )

    called = []
    with pytest.raises(DraftWorkerError, match="pvd-draft-device"):
        build_prediction_only_worker(
            draft_args(pvd_draft_device=device),
            placement=DraftPlacement(scratch_budget_bytes=4096, gpu_id=1),
            nccl_port=1234,
            target_worker=object(),
            worker_factory=lambda **kwargs: called.append(kwargs),
        )
    assert not called, "weights would have been loaded before placement validation"


def test_worker_defaults_to_its_actual_cuda_gpu_id():
    from sglang.srt.disaggregation.pvd.draft_sglang import (
        build_prediction_only_worker,
    )

    captured = {}

    def fake_factory(**kwargs):
        captured.update(kwargs)
        return FakeWorker(FakePool("draft-req"), FakePool("draft-kv"))

    build_prediction_only_worker(
        draft_args(pvd_draft_device=None),
        placement=DraftPlacement(scratch_budget_bytes=4096, gpu_id=2),
        nccl_port=1234,
        target_worker=FakeWorker(FakePool("target-req"), FakePool("target-kv")),
        worker_factory=fake_factory,
    )
    assert captured["server_args"].device == "cuda:2"


# --------------------------------------------------------------------------
# Gap 6: persistent memory is not per-call scratch
# --------------------------------------------------------------------------


def test_weights_and_pools_are_charged_once_against_their_own_budget():
    fac = factory(persistent_bytes=4096)
    made = provider(fac)
    assert made.persistent_budget.snapshot()["used_staging_bytes"] == 4096
    before = made.scratch_budget.snapshot()["used_staging_bytes"]
    with made.branch():
        made.predict(prefix(), 2)
    # Branch scratch came and went; the persistent charge did not move.
    assert made.persistent_budget.snapshot()["used_staging_bytes"] == 4096
    assert made.scratch_budget.snapshot()["used_staging_bytes"] == before
    # And the two are different budgets, so one cannot fund the other.
    assert made.persistent_budget is not made.scratch_budget


def test_a_model_too_large_for_the_persistent_budget_is_refused():
    with pytest.raises(TransferCapacityError):
        provider(
            factory(persistent_bytes=1 << 30),
            placement=DraftPlacement(
                scratch_budget_bytes=1 << 20, persistent_budget_bytes=4096
            ),
        )


def test_branch_scratch_is_bounded_by_the_handles_declared_worst_case():
    executor = FakeExecutor(bytes_per_token=1 << 16)
    made = provider(
        factory(executor=executor),
        placement=DraftPlacement(
            scratch_budget_bytes=4096, persistent_budget_bytes=1 << 20
        ),
    )
    with pytest.raises(TransferCapacityError):
        with made.branch():
            pass  # pragma: no cover
    assert made.active_branches == 0
    assert made.scratch_budget.snapshot()["used_staging_bytes"] == 0


# --------------------------------------------------------------------------
# Gap 7: tokenizer compatibility, not vocabulary size
# --------------------------------------------------------------------------


def signature(size=64, bos=1, eos=2, fingerprint="abc"):
    return VocabularySignature(
        size=size, bos_token_id=bos, eos_token_id=eos, fingerprint=fingerprint
    )


@pytest.mark.parametrize(
    "change",
    [
        {"size": 65},
        {"bos": 9},
        {"eos": 9},
        {"fingerprint": "different-encoding"},
    ],
)
def test_two_tokenizers_that_disagree_anywhere_are_refused(change):
    with pytest.raises(PredictionConfigError, match="disagree on"):
        provider(
            draft_vocabulary=signature(),
            target_vocabulary=signature(**change),
        )


def test_matching_size_with_a_different_encoding_is_still_refused():
    """The case a vocabulary-size check misses entirely."""
    with pytest.raises(PredictionConfigError, match="fingerprint"):
        provider(
            draft_vocabulary=signature(fingerprint="encodes-one-way"),
            target_vocabulary=signature(fingerprint="encodes-another-way"),
        )


def test_a_one_sided_vocabulary_check_is_refused():
    with pytest.raises(PredictionConfigError, match="one-sided"):
        provider(draft_vocabulary=signature())


def test_both_the_prefix_and_the_returned_tokens_are_checked():
    made = provider(
        factory(executor=FakeExecutor(sequence=[31, 32])),
        draft_vocabulary=signature(size=64),
        target_vocabulary=signature(size=64),
    )
    with made.branch():
        # A prefix token outside the draft vocabulary is refused before use.
        with pytest.raises(PredictionConfigError, match="prefix token 999"):
            made.predict(prefix(tokens=(1, 999)), 2)
    made2 = provider(
        factory(executor=FakeExecutor(vocab=4096, sequence=[31, 2000])),
        draft_vocabulary=signature(size=64),
        target_vocabulary=signature(size=64),
    )
    with made2.branch():
        with pytest.raises(PredictionConfigError, match="predicted token 2000"):
            made2.predict(prefix(tokens=(1, 2)), 2)


# --------------------------------------------------------------------------
# Gap 8: the worker surface is an allowlist
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", KNOWN_GENERATION_METHODS)
def test_every_generation_entry_point_is_unreachable(name):
    worker = FakeWorker(FakePool("a"), FakePool("b"))
    wrapped = DraftWorkerInterface(worker)
    with pytest.raises(DraftWorkerError, match="not on the prediction-only allowlist"):
        getattr(wrapped, name)
    assert worker.generation_calls == 0
    assert name in wrapped.refused


def test_a_method_nobody_listed_is_refused_too():
    """The difference between an allowlist and a deny list."""
    wrapped = DraftWorkerInterface(FakeWorker(FakePool("a"), FakePool("b")))
    with pytest.raises(DraftWorkerError, match="not on the prediction-only allowlist"):
        wrapped.some_unreviewed_method
    # Including a name that does not exist upstream at all.
    with pytest.raises(DraftWorkerError):
        wrapped.a_method_added_next_release


def test_the_allowed_names_pass_through_and_writes_do_not():
    worker = FakeWorker(FakePool("a"), FakePool("b"))
    wrapped = DraftWorkerInterface(worker)
    assert wrapped.get_memory_pool() == worker.get_memory_pool()
    assert wrapped.device == "cpu"
    assert set(ALLOWED_WORKER_METHODS) == {
        "get_memory_pool",
        "model_config",
        "device",
    }
    with pytest.raises(DraftWorkerError, match="does not mutate"):
        wrapped.model_config = {}


# --------------------------------------------------------------------------
# Gap 9: pool ownership, honestly reported
# --------------------------------------------------------------------------


def test_distinct_pool_objects_over_one_buffer_are_not_private_pools():
    shared = torch.zeros(4, 16, dtype=torch.int32)
    target = FakeWorker(FakePool("t-req", shared), FakePool("t-kv"))
    mine = FakeWorker(FakePool("d-req", shared), FakePool("d-kv"))
    with pytest.raises(DraftWorkerError, match="same memory"):
        require_private_pools(mine, target)


def test_private_pools_over_separate_buffers_are_storage_verified():
    target = FakeWorker(FakePool("t-req"), FakePool("t-kv"))
    mine = FakeWorker(FakePool("d-req"), FakePool("d-kv"))
    ownership = require_private_pools(mine, target)
    assert ownership.distinct_objects
    assert ownership.storage_verified
    assert "storage-verified" in ownership.describe()


def test_pools_whose_storage_cannot_be_inspected_are_not_called_private():
    class Opaque:
        pass

    target = FakeWorker(Opaque(), Opaque())
    mine = FakeWorker(Opaque(), Opaque())
    ownership = require_private_pools(mine, target)
    assert ownership.distinct_objects
    assert not ownership.storage_verified
    assert "not verified" in ownership.describe()
    made = provider(worker=FakeWorker(Opaque(), Opaque()), target_worker=target)
    # The description must not claim more than was established.
    assert "private (storage-verified)" not in made.describe()["pools"]


def test_a_draft_worker_sharing_the_targets_pools_is_refused():
    shared_req, shared_kv = FakePool("req"), FakePool("kv")
    target = FakeWorker(shared_req, shared_kv)
    with pytest.raises(DraftWorkerError, match="shares the target's req_to_token_pool"):
        require_private_pools(FakeWorker(shared_req, shared_kv), target)
    with pytest.raises(
        DraftWorkerError, match="shares the target's token_to_kv_pool_allocator"
    ):
        require_private_pools(FakeWorker(FakePool("own"), shared_kv), target)


def test_a_draft_worker_with_no_pool_of_its_own_is_refused():
    target = FakeWorker(FakePool("req"), FakePool("kv"))
    with pytest.raises(DraftWorkerError, match="no req_to_token_pool of its own"):
        require_private_pools(FakeWorker(None, FakePool("kv")), target)


# --------------------------------------------------------------------------
# The execution path itself
#
# Asserted against doubles: what the orchestration asks the model runner and
# the allocator for. This establishes the bookkeeping is right, NOT that any
# real model accepts it -- no ForwardBatch is built and no weights are loaded.
# --------------------------------------------------------------------------


def run_once(tokens=(10, 11, 12), want=2, **kwargs):
    executor = kwargs.pop("executor", None) or FakeExecutor(sequence=[31, 32, 33])
    alloc = kwargs.pop("allocator", None) or FakeAllocator()
    handle = SGLangDraftHandle(
        "b",
        executor,
        alloc,
        max_prefix_tokens=64,
        max_tokens=4,
        capabilities=DEFAULT_CAPABILITIES,
    )
    prepared = handle.prepare_prefix(tokens)
    produced = handle.generate(prepared, want)
    return handle, executor, alloc, produced


def test_the_prefix_forward_carries_the_whole_snapshot_from_position_zero():
    handle, executor, alloc, _ = run_once(tokens=(10, 11, 12))
    first = executor.calls[0]
    assert first.forward_mode == "extend"
    assert first.input_ids == (10, 11, 12)
    # Absolute positions, from zero: a fresh computation, not a continuation.
    assert first.positions == (0, 1, 2)
    assert first.seq_lens == (3,)
    assert first.extend_prefix_lens == (0,) and first.extend_seq_lens == (3,)
    # One KV row per token, all of them allocated by this handle.
    assert len(first.out_cache_loc) == 3
    assert set(first.out_cache_loc) <= set(handle.owned_kv)
    assert first.req_pool_indices == (handle.request_index,)


def test_each_step_advances_position_and_sequence_length_by_exactly_one():
    handle, executor, alloc, produced = run_once(tokens=(10, 11, 12), want=3)
    steps = executor.calls[1:]
    assert [s.forward_mode for s in steps] == ["decode", "decode"]
    assert [s.positions for s in steps] == [(3,), (4,)]
    assert [s.seq_lens for s in steps] == [(4,), (5,)]
    # Each step feeds back the token it just produced.
    assert [s.input_ids for s in steps] == [(produced[0],), (produced[1],)]
    # And writes to exactly one fresh row it owns.
    assert all(len(s.out_cache_loc) == 1 for s in steps)
    rows = [s.out_cache_loc[0] for s in steps]
    assert len(set(rows)) == len(rows)
    assert set(rows) <= set(handle.owned_kv)


def test_every_kv_row_written_was_allocated_by_this_handle_and_freed_after():
    handle, executor, alloc, _ = run_once(tokens=(10, 11), want=3)
    written = {loc for call in executor.calls for loc in call.out_cache_loc}
    assert written == set(handle.owned_kv)
    assert written <= alloc.live_kv
    handle.release()
    assert not alloc.live_kv and not alloc.live_requests


def test_the_number_of_forwards_is_bounded_by_the_token_budget():
    _, executor, _, produced = run_once(tokens=(1, 2), want=2)
    # One prefill plus (want - 1) steps: the last token needs no forward.
    assert len(executor.calls) == 2
    assert len(produced) == 2


def test_the_handle_never_names_a_request_index_it_did_not_allocate():
    alloc = FakeAllocator()
    alloc.next_request = 7
    handle, executor, _, _ = run_once(allocator=alloc)
    assert handle.request_index == 7
    assert all(c.req_pool_indices == (7,) for c in executor.calls)


def test_a_prefix_beyond_the_supported_bound_is_refused_before_allocating():
    alloc = FakeAllocator()
    handle = SGLangDraftHandle(
        "b",
        FakeExecutor(),
        alloc,
        max_prefix_tokens=4,
        max_tokens=2,
        capabilities=DraftCapabilities(
            architectures=("LlamaForCausalLM",),
            attention_backends=("triton",),
            max_prefix_tokens=4,
            max_predict_tokens=2,
        ),
    )
    with pytest.raises(DraftCapabilityError, match="exceeds the supported"):
        handle.prepare_prefix(tuple(range(9)))
    assert not alloc.live_kv and not alloc.live_requests


def test_an_unsupported_model_or_backend_is_refused_at_construction():
    with pytest.raises(DraftCapabilityError, match="architecture"):
        factory(executor=FakeExecutor(arch="SomeExoticMoEForCausalLM"))
    with pytest.raises(DraftCapabilityError, match="attention backend"):
        factory(executor=FakeExecutor(backend="an_unaudited_backend"))


def test_capabilities_are_checked_against_the_model_not_against_themselves():
    """A self-comparison would pass by construction and prove nothing."""
    caps = DEFAULT_CAPABILITIES
    caps.require_model(architecture="LlamaForCausalLM", attention_backend="triton")
    with pytest.raises(DraftCapabilityError):
        caps.require_model(architecture="NotListed", attention_backend="triton")
    # The shape check takes a request, not a declaration.
    caps.require_shape(prefix_tokens=10, predict_tokens=2)
    with pytest.raises(DraftCapabilityError):
        caps.require_shape(prefix_tokens=10, predict_tokens=10_000)


def test_nonsense_logits_are_refused_rather_than_sampled():
    class Broken(FakeExecutor):
        def forward(self, inputs):
            self.calls.append(inputs)
            return torch.full((self.vocab,), float("nan"))

    alloc = FakeAllocator()
    handle = SGLangDraftHandle(
        "b",
        Broken(),
        alloc,
        max_prefix_tokens=64,
        max_tokens=2,
        capabilities=DEFAULT_CAPABILITIES,
    )
    prepared = handle.prepare_prefix((1, 2))
    with pytest.raises(DraftLifecycleError, match="non-finite"):
        handle.generate(prepared, 1)


@pytest.mark.parametrize(
    "inputs,error",
    [
        (dict(forward_mode="verify"), "unsupported forward mode"),
        (dict(input_ids=()), "at least one token"),
        (dict(positions=(0, 1)), "one position per token"),
        (dict(out_cache_loc=(0, 1)), "one KV location per token"),
        (dict(seq_lens=(1, 2)), "one sequence length per request"),
    ],
)
def test_malformed_forward_inputs_are_refused(inputs, error):
    base = dict(
        forward_mode="decode",
        input_ids=(5,),
        positions=(3,),
        seq_lens=(4,),
        req_pool_indices=(0,),
        out_cache_loc=(9,),
    )
    base.update(inputs)
    with pytest.raises((DraftCapabilityError, DraftLifecycleError), match=error):
        DraftForwardInputs(**base)


def test_the_prefix_is_recomputed_on_every_prediction():
    """The documented v1 behaviour: nothing is retained between rounds."""
    alloc = FakeAllocator()
    executor = FakeExecutor()
    fac = factory(executor=executor, allocator=alloc)
    made = provider(fac)
    for _ in range(3):
        with made.branch():
            made.predict(prefix(tokens=(1, 2, 3, 4)), 2)
    prefills = [c for c in executor.calls if c.forward_mode == "extend"]
    assert len(prefills) == 3, "the prefix was not recomputed each round"
    assert all(c.input_ids == (1, 2, 3, 4) for c in prefills)
    assert all(c.positions == (0, 1, 2, 3) for c in prefills)
    # And nothing survives: no rows, no slots.
    assert not alloc.live_kv and not alloc.live_requests


def test_execution_is_serialized_across_concurrent_branches():
    """Separate handles are not a claim that the model runner is reentrant."""
    import threading

    overlap = []
    inside = threading.Semaphore(0)

    class Slow(FakeExecutor):
        def forward(self, inputs):
            overlap.append("enter")
            import time

            time.sleep(0.02)
            overlap.append("leave")
            return super().forward(inputs)

    alloc = FakeAllocator()
    made = provider(
        factory(executor=Slow(), allocator=alloc),
        placement=DraftPlacement(
            scratch_budget_bytes=1 << 20,
            persistent_budget_bytes=1 << 20,
            max_concurrent_branches=2,
        ),
    )

    def run():
        with made.branch():
            made.predict(prefix(), 2)

    workers = [threading.Thread(target=run) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)
        assert not worker.is_alive()
    # Strictly alternating enter/leave: no two forwards overlapped.
    assert overlap == ["enter", "leave"] * (len(overlap) // 2)


# --------------------------------------------------------------------------
# Nothing committed, nothing live touched
# --------------------------------------------------------------------------


class LiveState:
    """Stands in for what a real batch would carry through a decode step."""

    def __init__(self):
        self.decode_batch_idx = 7
        self.req_to_token = torch.zeros(4, 16, dtype=torch.int32)
        self.seq_lens = torch.tensor([4, 4], dtype=torch.int32)
        self.penalizer_tokens = 0
        self.out_cache_loc = None
        self.committed_tokens = [10, 11, 12, 13]
        self.refresh_clock = 2

    def fingerprint(self):
        return (
            self.decode_batch_idx,
            self.req_to_token.clone(),
            self.seq_lens.clone(),
            self.penalizer_tokens,
            self.out_cache_loc,
            list(self.committed_tokens),
            self.refresh_clock,
        )


def same(a, b):
    return all(
        torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y
        for x, y in zip(a, b)
    )


def test_predicting_commits_nothing_and_touches_no_live_state():
    live = LiveState()
    before = live.fingerprint()
    worker = FakeWorker(FakePool("my-req"), FakePool("my-kv"))
    made = provider(
        worker=worker,
        target_worker=FakeWorker(FakePool("t-req"), FakePool("t-kv")),
    )
    snapshot = snapshot_committed(
        "request", live.committed_tokens, live.refresh_clock, "prefix-v1"
    )
    with made.branch():
        made.predict(snapshot, 2)
    assert same(live.fingerprint(), before)
    assert worker.generation_calls == 0
    assert made.worker.refused == ()


def test_the_snapshot_is_not_mutated_and_is_handed_over_immutably():
    made = provider()
    snapshot = prefix()
    before = (snapshot.tokens, snapshot.committed_position, snapshot.version)
    with made.branch():
        made.predict(snapshot, 2)
    assert (snapshot.tokens, snapshot.committed_position, snapshot.version) == before


def test_the_refresh_clock_is_not_advanced_by_prediction():
    live = LiveState()
    made = provider()
    for _ in range(3):
        with made.branch():
            made.predict(
                snapshot_committed(
                    "request", live.committed_tokens, live.refresh_clock, "prefix-v1"
                ),
                2,
            )
    assert live.refresh_clock == 2
    assert live.committed_tokens == [10, 11, 12, 13]


def test_a_new_request_does_not_disturb_an_existing_one():
    made = provider()
    first = ProbeSearchSession("request-a", "entry-a")
    window = first.begin(prefix("request-a"), target_tokens=4, query_positions=(5,))
    second = ProbeSearchSession("request-b", "entry-b")
    second.begin(prefix("request-b"), target_tokens=6, query_positions=(5,))
    with made.branch():
        made.predict(prefix("request-b"), 2)
    assert first.incarnation != second.incarnation
    assert first._pending is window
    assert first._observed == 2
    assert window.target_tokens == 4


def test_a_prediction_carries_its_request_and_prefix_identity():
    made = provider()
    with made.branch():
        result = made.predict(prefix(), 2)
    assert result.request_id == "request"
    assert result.prefix_version == "prefix-v1"
    assert result.source["provider"] == "sglang-standalone"
    assert result.source["prefix"] == "recomputed-per-call"
    assert result.source["execution"] == "serialized"
    assert result.source["worker_surface"] == "allowlist"


def test_a_runner_that_ignores_the_budget_cannot_set_it():
    made = provider(factory(executor=FakeExecutor(sequence=[31, 32, 33, 34])))
    with made.branch():
        assert len(made.predict(prefix(), 2).tokens) == 2
    with made.branch():
        assert len(made.predict(prefix(), 1).tokens) == 1


# --------------------------------------------------------------------------
# Composition with the existing foundation
# --------------------------------------------------------------------------


def test_the_provider_drops_into_the_existing_prediction_pipeline():
    """Phase 3 is composition: no new plumbing, the same checks apply."""
    from test_pvd_probe_search import ScratchProbe, prepare
    from test_pvd_search_client import fixture

    index, store, identity, rows, scope = fixture()
    config = DraftConfig("configurable/draft", predict_tokens=2)
    alloc = FakeAllocator()
    made = provider(factory(allocator=alloc), config=config)
    pipeline = PredictionPipeline(
        made,
        ScratchProbe(rows[0]),
        config,
        ProbeConfig(identity.vector_space, (0,), head_count=1),
    )
    session = ProbeSearchSession("request", identity.entry_transfer_id)
    window = session.begin(prefix(), target_tokens=4, query_positions=(5,))
    rng = torch.random.get_rng_state().clone()
    prepared = prepare(
        session, window, pipeline, ProbeSearchRoute(0, identity, scope, 1)
    )
    assert prepared.window is window and len(prepared.queries) == 1
    # The branch ran inside the pipeline's scope and gave everything back.
    assert made.scratch_budget.snapshot()["used_staging_bytes"] == 0
    assert made.active_branches == 0
    assert not alloc.live_kv and not alloc.live_requests
    assert torch.equal(rng, torch.random.get_rng_state())


# --------------------------------------------------------------------------
# Configuration surface
#
# PVD's own flags. Setting them must never enable SGLang's speculative
# generation loop, and the existing prohibition is not narrowed to let them in.
# --------------------------------------------------------------------------


def pvd_args(**overrides):
    from types import SimpleNamespace

    base = dict(
        disaggregation_topology="pvd",
        disaggregation_mode="decode",
        pvd_kv_refresh_interval=4,
        disable_overlap_schedule=False,
        pvd_vector_coordinator_url="http://v:9100",
        pvd_vector_groups=None,
        tp_size=2,
        dp_size=1,
        enable_dp_attention=False,
        pp_size=1,
        pvd_rank_rails="mlx5_0,mlx5_0",
        disaggregation_transfer_backend="mooncake",
        pvd_strict_rdma_preflight=True,
        speculative_algorithm=None,
        enable_hierarchical_cache=False,
        enable_hisparse=False,
        enable_prefill_context_parallel=False,
        disaggregation_decode_enable_radix_cache=False,
        pvd_model_instance_id="model",
        pvd_transfer_staging_budget_bytes=1 << 30,
        pvd_transfer_max_inflight=64,
        pvd_draft_model_path=None,
        pvd_draft_revision=None,
        pvd_draft_device=None,
        pvd_draft_predict_tokens=8,
        pvd_draft_scratch_budget_bytes=None,
        pvd_draft_persistent_budget_bytes=None,
        pvd_draft_mem_fraction_static=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_a_draft_model_needs_an_explicit_scratch_budget():
    from sglang.srt.arg_groups.pvd_disaggregation_hook import (
        handle_pvd_disaggregation,
    )

    with pytest.raises(ValueError, match="pvd-draft-scratch-budget-bytes is required"):
        handle_pvd_disaggregation(pvd_args(pvd_draft_model_path="configurable/draft"))
    args = pvd_args(
        pvd_draft_model_path="configurable/draft",
        pvd_draft_scratch_budget_bytes=1 << 20,
    )
    handle_pvd_disaggregation(args)
    assert args.pvd_draft_model_path == "configurable/draft"
    # Configuring a draft model does not turn speculative decoding on.
    assert args.speculative_algorithm is None


def test_a_scratch_budget_without_a_draft_model_is_refused():
    from sglang.srt.arg_groups.pvd_disaggregation_hook import (
        handle_pvd_disaggregation,
    )

    with pytest.raises(ValueError, match="no meaning without"):
        handle_pvd_disaggregation(pvd_args(pvd_draft_scratch_budget_bytes=1 << 20))


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"pvd_draft_scratch_budget_bytes": -1}, "positive integer"),
        (
            {"pvd_draft_scratch_budget_bytes": 1 << 20, "pvd_draft_predict_tokens": 0},
            "predict-tokens must be a positive integer",
        ),
    ],
)
def test_a_nonsense_draft_configuration_is_refused(overrides, error):
    from sglang.srt.arg_groups.pvd_disaggregation_hook import (
        handle_pvd_disaggregation,
    )

    with pytest.raises(ValueError, match=error):
        handle_pvd_disaggregation(
            pvd_args(pvd_draft_model_path="configurable/draft", **overrides)
        )


def test_the_speculative_prohibition_is_still_unconditional():
    """Configuring a PVD draft model must not create an exemption."""
    from sglang.srt.arg_groups.pvd_disaggregation_hook import (
        handle_pvd_disaggregation,
    )

    for extra in (
        {},
        {
            "pvd_draft_model_path": "configurable/draft",
            "pvd_draft_scratch_budget_bytes": 1 << 20,
        },
    ):
        with pytest.raises(ValueError, match="does not support speculative decoding"):
            handle_pvd_disaggregation(pvd_args(speculative_algorithm="EAGLE", **extra))


def test_the_flags_exist_and_default_to_no_draft_model():
    """Read from source: importing ServerArgs pulls the whole frontend in."""
    source = (SRT / "server_args.py").read_text(encoding="utf-8")
    for flag in (
        "--pvd-draft-model-path",
        "--pvd-draft-revision",
        "--pvd-draft-device",
        "--pvd-draft-predict-tokens",
        "--pvd-draft-scratch-budget-bytes",
        "--pvd-draft-persistent-budget-bytes",
        "--pvd-draft-mem-fraction-static",
    ):
        assert flag in source, f"{flag} is missing"
    defaults = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.startswith("pvd_draft_") and node.value is not None:
                defaults[node.target.id] = ast.literal_eval(node.value)
    # Off unless asked for: no model, no revision and no budget are defaulted.
    assert defaults["pvd_draft_model_path"] is None
    assert defaults["pvd_draft_revision"] is None
    assert defaults["pvd_draft_scratch_budget_bytes"] is None
    assert defaults["pvd_draft_persistent_budget_bytes"] is None
    assert defaults["pvd_draft_mem_fraction_static"] is None
    assert defaults["pvd_draft_predict_tokens"] > 0
    # None of these flags reads or writes the speculative configuration; the
    # word appears in the help text only, to say they do not enable it.
    draft_block = source[source.index("--pvd-draft-model-path") :]
    draft_block = draft_block[: draft_block.index("--pvd-strict-rdma-preflight")]
    assert "speculative_algorithm" not in draft_block
    assert "speculative_num" not in draft_block
    assert "does NOT enable speculative decoding" in draft_block


def test_the_provider_refuses_an_oversized_request_before_executing():
    """Checked at the provider, not only inside a handle that may not check.

    The handle double here validates nothing, so if the provider skipped its
    own shape check the executor would be reached with an unsupported request.
    """

    class UncheckedHandle:
        def __init__(self):
            self.calls = 0

        @property
        def branch_id(self):
            return "b"

        def scratch_bytes(self):
            return 16

        def prepare_prefix(self, tokens):
            self.calls += 1
            return object()

        def generate(self, prepared, max_tokens):  # pragma: no cover
            self.calls += 1
            return [1]

        def release(self):
            pass

    handle = UncheckedHandle()

    class TinyFactory:
        def capabilities(self):
            return DraftCapabilities(
                architectures=("LlamaForCausalLM",),
                attention_backends=("triton",),
                max_prefix_tokens=3,
                max_predict_tokens=4,
            )

        def persistent_bytes(self):
            return 0

        def open(self, **kwargs):
            return handle

    made = provider(TinyFactory())
    with made.branch():
        with pytest.raises(DraftCapabilityError, match="exceeds the supported"):
            made.predict(prefix(tokens=(1, 2, 3, 4, 5)), 2)
    assert handle.calls == 0, "execution was reached despite an unsupported shape"
