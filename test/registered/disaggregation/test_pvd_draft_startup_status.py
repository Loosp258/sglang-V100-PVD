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
    "field", ["pvd_draft_scratch_budget_bytes", "pvd_draft_predict_tokens"]
)
@pytest.mark.parametrize("bad", [True, 1.5, "8"])
def test_draft_bounds_refuse_bool_float_and_string_values(field, bad):
    args = pvd_args(
        pvd_draft_model_path="user/chosen-model", pvd_draft_scratch_budget_bytes=4096
    )
    setattr(args, field, bad)
    with pytest.raises(ValueError, match="positive integer"):
        handle_pvd_disaggregation(args)


@pytest.mark.parametrize("orphan", [0, False])
def test_zero_or_false_budget_without_model_is_not_silently_ignored(orphan):
    with pytest.raises(ValueError, match="no meaning without"):
        handle_pvd_disaggregation(pvd_args(pvd_draft_scratch_budget_bytes=orphan))


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
