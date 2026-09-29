"""Two-position Top-16 union capacity on held-out 2155-token Prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pvd_draft_q_error_decomposition import generate_greedy, predicted_q
from pvd_draft_q_multitask_probe import truncate_draft
from pvd_draft_q_readout_probe import TargetQueryReadout, teacher_capture


def choose_capped_budgets(top, cap=128):
    """Reduce K16 heads to K8 until the two-position token union fits."""
    budgets = [16] * 7

    def union_size(candidate):
        return len(set(int(token) for position in range(2)
                       for head in range(7)
                       for token in top[position, head, :candidate[head]]))

    while union_size(budgets) > cap:
        candidates = []
        before = union_size(budgets)
        for head in range(7):
            if budgets[head] == 16:
                smaller = list(budgets)
                smaller[head] = 8
                candidates.append((before - union_size(smaller), head))
        if not candidates:
            raise AssertionError("K8 for seven heads across two positions must fit 128")
        _, head = max(candidates)
        budgets[head] = 8
    return budgets


@torch.inference_mode()
def evaluate(prompt, student, readout, target, *, device):
    future = generate_greedy(student, prompt, 8, device,
                             pad_token_id=student.config.eos_token_id)
    ids = prompt + future
    true_q, prompt_k = teacher_capture(target, ids, device)
    predicted = predicted_q(student, readout, ids, device)[-2:]
    predicted = predicted.permute(1, 0, 2, 3)
    true_q = true_q[:, -2:]
    groups = {name: [] for name in ("uniform8", "uniform16", "capped16")}
    reductions = []
    for layer in range(28):
        for kv_head in range(4):
            first = kv_head * 7
            last = first + 7
            keys = prompt_k[layer, :len(prompt), kv_head].float().to(device)
            pred = predicted[layer, :, first:last].reshape(14, 128)
            truth = true_q[layer, :, first:last].float().to(device).reshape(14, 128)
            top = (pred @ keys.T).topk(16, dim=-1).indices.reshape(2, 7, 16).cpu()
            exact = (truth @ keys.T).topk(4, dim=-1).indices.reshape(2, 7, 4).cpu()
            capped = choose_capped_budgets(top)
            reductions.append(sum(value == 8 for value in capped))
            for name, budgets in (
                ("uniform8", [8] * 7),
                ("uniform16", [16] * 7),
                ("capped16", capped),
            ):
                tokens = set()
                recalls = []
                for pos in range(2):
                    for head in range(7):
                        selected = set(top[pos, head, :budgets[head]].tolist())
                        tokens.update(selected)
                        recalls.append(len(selected.intersection(
                            exact[pos, head].tolist())) / 4)
                groups[name].append({
                    "union_tokens": len(tokens),
                    "top4_coverage": sum(recalls) / len(recalls),
                })
    return {
        "prompt_tokens": len(prompt),
        "future_tokens": future,
        "groups_reduced_from_k16": sum(value > 0 for value in reductions),
        "heads_reduced_to_k8": sum(reductions),
        "policies": {
            name: {
                "mean_true_top4_coverage": sum(x["top4_coverage"] for x in values)
                                           / len(values),
                "max_union_tokens": max(x["union_tokens"] for x in values),
                "fraction_over_128_cap": sum(
                    x["union_tokens"] > 128 for x in values) / len(values),
                "estimated_fresh_kv_mib": sum(
                    x["union_tokens"] for x in values) * 512 / 2**20,
            } for name, values in groups.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2155)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model,
                                              local_files_only=True)
    hotpot = (args.source_root / "benchmark/react/hotpotqa_100.jsonl").read_text(
        encoding="utf-8").splitlines()
    contents = {
        "case40_repeat": "Case 40. " + "EEFTRITON " * 1100,
        "hotpot_last20": "\n".join(hotpot[80:]),
    }
    prompts = {}
    for name, content in contents.items():
        ids = tokenizer.encode(content, add_special_tokens=False)
        if len(ids) < args.prompt_tokens:
            raise ValueError(f"{name} has only {len(ids)} tokens")
        prompts[name] = ids[:args.prompt_tokens]
    student = truncate_draft(AutoModelForCausalLM.from_pretrained(
        args.draft_model, dtype=torch.float32,
        attn_implementation="eager", local_files_only=True,
    ).to(device)).eval()
    readout = TargetQueryReadout(896, 28, 28 * 128, 896,
                                 fusion="learned").to(device).eval()
    state = torch.load(args.student_checkpoint, map_location=device,
                       weights_only=True)
    student.load_state_dict(state["student"])
    readout.load_state_dict(state["readout"])
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, dtype=torch.float16,
        attn_implementation="eager", local_files_only=True,
    ).to(device).eval()
    results = {}
    for name, prompt in prompts.items():
        results[name] = evaluate(prompt, student, readout, target,
                                 device=device)
        print(json.dumps({"completed": name,
                          "groups_reduced": results[name][
                              "groups_reduced_from_k16"],
                          "policies": results[name]["policies"]}),
              flush=True)
    report = {
        "prompt_tokens": args.prompt_tokens,
        "query_positions": 2,
        "gqa_groups": 112,
        "cap": 128,
        "mode": "exact Prompt-K scores on post-training Draft trajectories; no native CAGRA or transfer timing",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
