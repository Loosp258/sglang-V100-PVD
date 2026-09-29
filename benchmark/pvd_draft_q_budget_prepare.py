"""Calibrate bounded per-head Draft-Q candidate budgets and export native data.

Calibration uses only training-window Draft trajectories. Validation fixtures
and their 7B Q/K remain separate, with two future query positions per refresh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from pvd_draft_q_multitask_probe import truncate_draft
from pvd_draft_q_readout_probe import (
    TargetQueryReadout,
    student_features,
    target_rope,
)


KS = (4, 8, 16)
LAYERS = 28
QUERY_HEADS = 28
KV_HEADS = 4
HEADS_PER_KV = QUERY_HEADS // KV_HEADS


@torch.inference_mode()
def exact_group_metrics(prompt_k, predicted_q, true_q, *, device):
    """Per-head coverage and margins for one two-position query batch."""
    if (predicted_q.shape != (LAYERS, 2, QUERY_HEADS, 128)
            or true_q.shape != predicted_q.shape
            or prompt_k.shape[:1] != (LAYERS,)
            or prompt_k.shape[2:] != (KV_HEADS, 128)):
        raise ValueError("expected two future queries and full target Q/K layout")
    coverage = torch.empty(LAYERS, QUERY_HEADS, len(KS))
    margin = torch.empty(LAYERS, QUERY_HEADS)
    for layer in range(LAYERS):
        for kv_head in range(KV_HEADS):
            keys = prompt_k[layer, :, kv_head].float().to(device)
            first = kv_head * HEADS_PER_KV
            last = first + HEADS_PER_KV
            pred = predicted_q[layer, :, first:last].float().to(device)
            true = true_q[layer, :, first:last].float().to(device)
            pred_scores = pred.reshape(-1, 128) @ keys.T
            true_scores = true.reshape(-1, 128) @ keys.T
            pred_top = pred_scores.topk(max(KS), dim=-1)
            true_top = true_scores.topk(4, dim=-1).indices
            for index, k in enumerate(KS):
                hit = (pred_top.indices[:, :k, None]
                       == true_top[:, None, :]).any(dim=1)
                coverage[layer, first:last, index] = (
                    hit.float().mean(dim=-1).reshape(2, HEADS_PER_KV)
                    .mean(dim=0).cpu()
                )
            top_values = pred_top.values
            normalized_gap = (top_values[:, 3] - top_values[:, 4]) / (
                pred_scores.std(dim=-1).clamp_min(1e-6))
            margin[layer, first:last] = normalized_gap.reshape(
                2, HEADS_PER_KV).mean(dim=0).cpu()
    return coverage, margin


def calibrate_equal_budget(mean_coverage):
    """One K16, two K4 and four K8 heads per GQA group: 56 per position."""
    policy = torch.full((LAYERS, QUERY_HEADS), 8, dtype=torch.int64)
    choices = []
    for layer in range(LAYERS):
        for kv_head in range(KV_HEADS):
            first = kv_head * HEADS_PER_KV
            group = mean_coverage[layer, first:first + HEADS_PER_KV]
            best = None
            for up in range(HEADS_PER_KV):
                lost = sorted(
                    ((float(group[head, 1] - group[head, 0]), head)
                     for head in range(HEADS_PER_KV) if head != up),
                    key=lambda item: item[0],
                )[:2]
                expected_gain = float(group[up, 2] - group[up, 1])
                expected_gain -= sum(value for value, _ in lost)
                candidate = (expected_gain, up, tuple(head for _, head in lost))
                if best is None or candidate[0] > best[0]:
                    best = candidate
            _, up, down = best
            policy[layer, first + up] = 16
            for head in down:
                policy[layer, first + head] = 4
            choices.append({
                "layer": layer, "kv_head": kv_head,
                "k16_head": first + up,
                "k4_heads": [first + head for head in down],
                "predicted_gain_vs_uniform8": best[0] / HEADS_PER_KV,
            })
    for layer in range(LAYERS):
        for kv_head in range(KV_HEADS):
            values = policy[layer, kv_head * 7:(kv_head + 1) * 7]
            if sorted(values.tolist()) != [4, 4, 8, 8, 8, 8, 16]:
                raise AssertionError("invalid equal-budget policy")
    return policy, choices


@torch.inference_mode()
def predict_train_q(student, readout, record, *, device):
    prompt = record["prompt"]
    future = (record["future"] if "future" in record else
              record["branches"]["student"]["future"])
    features = student_features(
        student, prompt + future, device, tuple(range(6)))[-2:].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        pre = readout(features)
    positions = torch.arange(len(prompt) + len(future) - 2,
                             len(prompt) + len(future), device=device)
    return target_rope(pre.reshape(2, LAYERS, QUERY_HEADS, 128),
                       1_000_000, positions=positions).permute(1, 0, 2, 3)


@torch.inference_mode()
def validation_fixture(record, *, device):
    prefix = record["prompt_tokens"]
    future = record["student_future"]
    if len(future) < 2:
        raise ValueError("validation future shorter than two tokens")
    pre = record["student_branch"]["base_pre"][-2:].float().to(device)
    positions = torch.arange(prefix + len(future) - 2,
                             prefix + len(future), device=device)
    predicted = target_rope(
        pre.reshape(2, LAYERS, QUERY_HEADS, 128), 1_000_000,
        positions=positions).permute(1, 0, 2, 3)
    return {
        "name": record["name"],
        "prompt_tokens": prefix,
        "prompt_k": record["prompt_k"].half().contiguous(),
        "predicted_q": predicted.float().cpu().contiguous(),
        "student_branch_target_q": record["student_q"][:, -2:].half().contiguous(),
        "target_branch_target_q": record["target_q"][:, -2:].half().contiguous(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--train-captures", type=Path, required=True)
    parser.add_argument("--valid-capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    train = torch.load(args.train_captures, map_location="cpu",
                       weights_only=False)
    valid = torch.load(args.valid_capture, map_location="cpu",
                       weights_only=False)
    state = torch.load(args.student_checkpoint, map_location=device,
                       weights_only=True)
    student = truncate_draft(AutoModelForCausalLM.from_pretrained(
        args.draft_model, dtype=torch.float32,
        attn_implementation="eager", local_files_only=True,
    ).to(device)).eval()
    student.load_state_dict(state["student"])
    readout = TargetQueryReadout(896, LAYERS, QUERY_HEADS * 128, 896,
                                 fusion="learned").to(device).eval()
    readout.load_state_dict(state["readout"])
    train_coverage = torch.zeros(LAYERS, QUERY_HEADS, len(KS))
    train_record_summary = []
    for record in train:
        predicted = predict_train_q(student, readout, record, device=device)
        true = (record["post_q"] if "post_q" in record else
                record["branches"]["student"]["post_q"])[-2:]
        true = true.permute(1, 0, 2, 3)
        coverage, margin = exact_group_metrics(
            record["prompt_k"], predicted, true, device=device)
        train_coverage += coverage
        train_record_summary.append({
            "name": record["name"],
            "mean_top4_coverage_k8": float(coverage[:, :, 1].mean()),
            "mean_normalized_top4_to_5_margin": float(margin.mean()),
        })
        print(json.dumps({"calibrated_train": record["name"]}), flush=True)
    mean_coverage = train_coverage / len(train)
    policy, choices = calibrate_equal_budget(mean_coverage)
    fixtures = [validation_fixture(record, device=device) for record in valid]
    torch.save(fixtures, args.output_dir / "native_fixture.pt")
    validation_summary = []
    for fixture in fixtures:
        coverage, margin = exact_group_metrics(
            fixture["prompt_k"], fixture["predicted_q"],
            fixture["student_branch_target_q"], device=device)
        selected = torch.empty(LAYERS, QUERY_HEADS)
        for index, k in enumerate(KS):
            selected[policy == k] = coverage[:, :, index][policy == k]
        validation_summary.append({
            "name": fixture["name"],
            "prompt_tokens": fixture["prompt_tokens"],
            "exact_branch_top4_coverage_uniform4": float(coverage[:, :, 0].mean()),
            "exact_branch_top4_coverage_uniform8": float(coverage[:, :, 1].mean()),
            "exact_branch_top4_coverage_adaptive": float(selected.mean()),
            "exact_branch_top4_coverage_uniform16": float(coverage[:, :, 2].mean()),
            "mean_normalized_top4_to_5_margin": float(margin.mean()),
        })
    sorted_margins = sorted(r["mean_normalized_top4_to_5_margin"]
                            for r in train_record_summary)
    threshold = sorted_margins[max(0, int(0.2 * len(sorted_margins)) - 1)]
    for row in validation_summary:
        row["low_margin_fallback"] = (
            row["mean_normalized_top4_to_5_margin"] <= threshold)
    report = {
        "calibration_capture": str(args.train_captures),
        "post_training_draft_trajectories": "future" in train[0],
        "train_windows": len(train),
        "valid_windows": len(valid),
        "query_positions_per_refresh": 2,
        "true_neighbor_k": 4,
        "candidate_ks": KS,
        "adaptive_candidates_per_gqa_group_per_position": 56,
        "worst_case_two_position_union": 112,
        "max_union_cap_tested": 128,
        "mean_train_top4_coverage": {
            str(k): float(mean_coverage[:, :, i].mean())
            for i, k in enumerate(KS)
        },
        "per_head_train_top4_coverage": mean_coverage.tolist(),
        "adaptive_policy": policy.tolist(),
        "policy_choices": choices,
        "low_margin_20pct_train_threshold": threshold,
        "train_record_summary": train_record_summary,
        "validation_exact_summary": validation_summary,
        "note": "Calibration uses training windows only. Native CAGRA and actual union counts are evaluated separately on node1. Confidence fallback is an offline flag, not a serving route.",
    }
    (args.output_dir / "calibration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"completed": True,
                      "mean_train_top4_coverage": report["mean_train_top4_coverage"],
                      "mean_validation_uniform8": sum(
                          x["exact_branch_top4_coverage_uniform8"]
                          for x in validation_summary) / len(validation_summary),
                      "mean_validation_adaptive": sum(
                          x["exact_branch_top4_coverage_adaptive"]
                          for x in validation_summary) / len(validation_summary),
                      "fallback_flags": sum(x["low_margin_fallback"]
                                            for x in validation_summary)}),
          flush=True)


if __name__ == "__main__":
    main()
