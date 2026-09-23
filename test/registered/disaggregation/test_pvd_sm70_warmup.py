"""Keep torch-native offline probes out of FlashInfer-only SM70 warmup."""

import ast
from pathlib import Path
from types import SimpleNamespace


def _kernel_warmup():
    source = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/model_executor/model_runner.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    method = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "kernel_warmup"
    )
    namespace = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(get_device_capability=lambda: (7, 0))
        )
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["kernel_warmup"]


def _runner(backend, calls):
    return SimpleNamespace(
        device="cuda",
        _should_run_flashinfer_autotune=lambda: False,
        _warmup_sm70_flashinfer_sampling=lambda: calls.append("sampling"),
        is_generation=True,
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(is_dflash_family=lambda: False),
        server_args=SimpleNamespace(max_running_requests=2),
        prefill_attention_backend_str=backend,
        _warmup_prefill_kernels_extends=lambda batch_size: calls.append(batch_size),
    )


def test_sm70_torch_native_skips_flashinfer_prefill_warmup():
    calls = []
    _kernel_warmup()(_runner("torch_native", calls))
    assert calls == ["sampling"]


def test_sm70_flashinfer_still_warms_both_batch_sizes():
    calls = []
    _kernel_warmup()(_runner("flashinfer", calls))
    assert calls == ["sampling", 1, 2]
