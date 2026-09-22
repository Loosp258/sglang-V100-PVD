"""Admission must not silently treat KV capacity as all transient memory."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from sglang.srt.disaggregation.pvd.draft_sglang import (
    DraftCapabilityError,
    DraftPlacement,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferCapacityError
from test_pvd_draft_forward import adapter
from test_pvd_draft_sglang import FakeExecutor, factory, prefix, provider


def test_unknown_non_kv_peak_refuses_admission_before_allocation():
    executor = adapter()
    made = provider(factory(executor=executor))
    with (
        pytest.raises(DraftCapabilityError, match="non-KV transient bound is unknown"),
        made.branch(),
    ):
        pytest.fail("unknown workspace admitted")
    assert made.active_branches == 0
    assert made.scratch_budget.snapshot()["used_staging_bytes"] == 0
    assert executor.forward_count == 0


def test_non_kv_reservation_is_included_and_refunded():
    executor = FakeExecutor()
    made = provider(factory(executor=executor))
    with made.branch():
        expected = (
            made.capabilities.max_prefix_tokens + made.config.predict_tokens
        ) * executor.bytes_per_token() + executor.transient_bytes(0, 0)
        assert made.scratch_budget.snapshot()["used_staging_bytes"] == expected
        made.predict(prefix(), 2)
    assert made.scratch_budget.snapshot()["used_staging_bytes"] == 0


def test_capacity_that_fits_only_kv_does_not_admit_a_branch():
    executor = FakeExecutor()
    fac = factory(executor=executor)
    kv_only = (fac.capabilities().max_prefix_tokens + 4) * executor.bytes_per_token()
    made = provider(
        fac,
        placement=DraftPlacement(
            scratch_budget_bytes=kv_only, persistent_budget_bytes=1024
        ),
    )
    with pytest.raises(TransferCapacityError), made.branch():
        pytest.fail("logits/workspace charge omitted")
    assert not executor.calls
    assert made.active_branches == 0


@pytest.mark.parametrize("bad", [None, -1, True, 1.25])
def test_invalid_executor_transient_estimates_are_refused(bad):
    executor = FakeExecutor()
    executor.transient_bytes = lambda *args: bad
    made = provider(factory(executor=executor))
    with (
        pytest.raises(DraftCapabilityError, match="non-KV transient bytes"),
        made.branch(),
    ):
        pytest.fail("invalid estimate admitted")


@pytest.mark.parametrize("bad", [-1, True, 1.25])
def test_adapter_rejects_invalid_workspace_declarations(bad):
    with pytest.raises(DraftCapabilityError, match="transient_bytes_bound"):
        adapter(transient_bytes_bound=bad)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sm70_backend_selection_only_queries_cuda_for_cuda_runner(device):
    # Execute the actual method's AST with dependency doubles. This is a
    # dispatch unit test, not evidence that an attention kernel runs; the
    # separate strict real-ModelRunner smoke provides CPU execution evidence.
    source = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/model_executor/model_runner.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    runner_class = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
    )
    method = next(
        n
        for n in runner_class.body
        if isinstance(n, ast.FunctionDef) and n.name == "_get_attention_backend"
    )
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    queries = []

    def capability(index):
        assert device == "cuda", "CPU path queried CUDA"
        queries.append(index)
        return (7, 0)

    args = SimpleNamespace(
        attention_backend="torch_native", speculative_draft_attention_backend=None
    )
    args.get_attention_backends = lambda: (None, None)
    globals_ = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(get_device_capability=capability)
        ),
        "get_global_server_args": lambda: args,
    }
    exec(compile(module, str(source), "exec"), globals_)  # noqa: S102 -- trusted repo AST
    runner = SimpleNamespace(
        device=device,
        gpu_id=2,
        is_draft_worker=False,
        server_args=args,
        _get_attention_backend_from_str=lambda backend, **kwargs: backend,
    )
    assert globals_["_get_attention_backend"](runner) == "torch_native"
    assert queries == ([2] if device == "cuda" else [])
