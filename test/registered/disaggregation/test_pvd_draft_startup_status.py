"""Configuration is not activation; bounds must be actual integers."""

import ast
import logging
from pathlib import Path

import pytest
from sglang.srt.arg_groups.pvd_disaggregation_hook import handle_pvd_disaggregation
from test_pvd_draft_sglang import pvd_args


def test_configured_draft_warns_that_production_prediction_is_not_active(caplog):
    args = pvd_args(
        pvd_draft_model_path="user/chosen-model", pvd_draft_scratch_budget_bytes=4096
    )
    with caplog.at_level(logging.WARNING):
        handle_pvd_disaggregation(args)
    assert "not active" in caplog.text and "full-Prompt" in caplog.text
    assert "CPU reference" in caplog.text
    assert args.speculative_algorithm is None


@pytest.mark.parametrize(
    "field",
    [
        "pvd_draft_scratch_budget_bytes",
        "pvd_draft_persistent_budget_bytes",
        "pvd_draft_predict_tokens",
    ],
)
@pytest.mark.parametrize("bad", [True, 1.5, "8"])
def test_draft_bounds_refuse_bool_float_and_string_values(field, bad):
    args = pvd_args(
        pvd_draft_model_path="user/chosen-model", pvd_draft_scratch_budget_bytes=4096
    )
    setattr(args, field, bad)
    with pytest.raises(ValueError, match="positive integer"):
        handle_pvd_disaggregation(args)


@pytest.mark.parametrize("field", ["scratch", "persistent"])
@pytest.mark.parametrize("orphan", [0, False])
def test_zero_or_false_budget_without_model_is_not_silently_ignored(field, orphan):
    with pytest.raises(ValueError, match="no meaning without"):
        handle_pvd_disaggregation(
            pvd_args(**{f"pvd_draft_{field}_budget_bytes": orphan})
        )


def test_persistent_budget_is_separate_optional_configuration():
    args = pvd_args(
        pvd_draft_model_path="user/chosen-model",
        pvd_draft_scratch_budget_bytes=4096,
        pvd_draft_persistent_budget_bytes=8192,
    )
    handle_pvd_disaggregation(args)
    assert args.pvd_draft_persistent_budget_bytes == 8192
    assert args.pvd_draft_scratch_budget_bytes == 4096


@pytest.mark.parametrize("fraction", [False, -0.2, 0, 1, 1.2, "0.1"])
def test_draft_fraction_refuses_invalid_values(fraction):
    with pytest.raises(ValueError, match="between 0 and 1"):
        handle_pvd_disaggregation(
            pvd_args(
                pvd_draft_model_path="user/chosen-model",
                pvd_draft_scratch_budget_bytes=4096,
                pvd_draft_mem_fraction_static=fraction,
            )
        )


def test_draft_fraction_has_no_meaning_without_model():
    with pytest.raises(ValueError, match="no meaning without"):
        handle_pvd_disaggregation(pvd_args(pvd_draft_mem_fraction_static=0.1))


def test_actual_cli_help_distinguishes_configuration_from_activation():
    source = Path("python/sglang/srt/server_args.py").read_text(encoding="utf-8")
    call = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--pvd-draft-model-path"
    )
    help_text = next(
        keyword.value.value for keyword in call.keywords if keyword.arg == "help"
    )
    assert "does not activate production" in help_text
    assert "full Prompt KV" in help_text
    assert "does NOT enable speculative decoding" in help_text


def _retrieval_config_args(**overrides):
    values = dict(
        pvd_predictive_retrieval_config=True,
        pvd_retrieval_vector_space="qwen2.5-7b@revision-a",
        pvd_retrieval_metric="ip",
        pvd_retrieval_top_k=10,
        pvd_retrieval_max_union_tokens=70,
        pvd_retrieval_bank_budget_bytes=1 << 30,
        pvd_retrieval_scratch_budget_bytes=1 << 28,
        pvd_draft_model_path="user/draft",
        pvd_draft_scratch_budget_bytes=1 << 27,
        pvd_draft_persistent_budget_bytes=1 << 29,
        pvd_draft_mem_fraction_static=0.2,
        pvd_waiting_queue_bootstrap=True,
        pvd_full_kv_fanin_max_slices=128,
        pvd_full_kv_fanin_response_bytes=1 << 30,
        tp_size=1,
        pvd_rank_rails="mlx5_0",
        device="cuda",
        page_size=1,
        attention_backend="torch_native",
        disable_cuda_graph=True,
    )
    values.update(overrides)
    return pvd_args(**values)


def test_retrieval_flags_validate_configuration_without_activating_pipeline(caplog):
    args = _retrieval_config_args()
    with caplog.at_level(logging.WARNING):
        handle_pvd_disaggregation(args)
    assert "configuration only" in caplog.text
    assert "production Scheduler does not construct" in caplog.text
    assert args.speculative_algorithm is None
    assert args.disable_overlap_schedule


@pytest.mark.parametrize(
    "field",
    [
        "pvd_retrieval_vector_space",
        "pvd_retrieval_top_k",
        "pvd_retrieval_max_union_tokens",
        "pvd_retrieval_bank_budget_bytes",
        "pvd_retrieval_scratch_budget_bytes",
    ],
)
def test_retrieval_configuration_requires_explicit_identity_bounds_and_budgets(field):
    with pytest.raises(ValueError):
        handle_pvd_disaggregation(_retrieval_config_args(**{field: None}))


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"disaggregation_mode": "prefill"}, "Decode-only"),
        ({"tp_size": 2}, "Decode TP1"),
        ({"pp_size": 2}, "--pp-size 1"),
        ({"dp_size": 2}, "DP1"),
        ({"device": "cpu"}, "CUDA device"),
        ({"page_size": 16}, "--page-size 1"),
        ({"attention_backend": "triton"}, "torch_native"),
        ({"disable_cuda_graph": False}, "CUDA graphs off"),
        ({"pvd_waiting_queue_bootstrap": False}, "waiting-queue-bootstrap"),
        ({"pvd_full_kv_fanin_max_slices": None}, "initial full-KV fan-in"),
        ({"pvd_draft_model_path": None}, "pvd-draft-model-path"),
    ],
)
def test_retrieval_configuration_fails_closed_outside_cuda_tp1_envelope(
    overrides, error
):
    with pytest.raises(ValueError, match=error):
        handle_pvd_disaggregation(_retrieval_config_args(**overrides))


@pytest.mark.parametrize("top_k", [True, 0, 513])
def test_retrieval_top_k_must_be_integer_in_supported_range(top_k):
    with pytest.raises(ValueError, match="pvd-retrieval-top-k"):
        handle_pvd_disaggregation(_retrieval_config_args(pvd_retrieval_top_k=top_k))


def test_retrieval_union_limit_must_cover_each_query_top_k():
    with pytest.raises(ValueError, match="must be at least.*top-k"):
        handle_pvd_disaggregation(
            _retrieval_config_args(
                pvd_retrieval_top_k=10, pvd_retrieval_max_union_tokens=9
            )
        )


def test_retrieval_values_without_configuration_opt_in_are_refused():
    with pytest.raises(ValueError, match="require --pvd-predictive-retrieval-config"):
        handle_pvd_disaggregation(
            pvd_args(pvd_retrieval_vector_space="ignored-by-runtime")
        )


def test_retrieval_configuration_flag_is_pvd_only():
    args = _retrieval_config_args(disaggregation_topology="pd")
    with pytest.raises(ValueError, match="requires topology pvd"):
        handle_pvd_disaggregation(args)


def test_retrieval_cli_is_explicitly_configuration_only():
    source = Path("python/sglang/srt/server_args.py").read_text(encoding="utf-8")
    for flag in (
        "--pvd-predictive-retrieval-config",
        "--pvd-retrieval-vector-space",
        "--pvd-retrieval-metric",
        "--pvd-retrieval-top-k",
        "--pvd-retrieval-max-union-tokens",
        "--pvd-retrieval-bank-budget-bytes",
        "--pvd-retrieval-scratch-budget-bytes",
    ):
        assert flag in source
    assert "does NOT activate retrieval in the serving" in source
