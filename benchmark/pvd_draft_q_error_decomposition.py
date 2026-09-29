"""Separate generated-token and target-Q prediction errors on Prompt K.

The full pretrained draft generates tokens. Its first six hidden layers feed a
frozen Q readout. Both target and draft token branches are replayed through the
real target model so that the token-only arm has an actual target-space Q.
This offline probe never changes the serving retrieval route.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pvd_draft_q_corpus_probe import corpus_prompts
from pvd_draft_q_readout_probe import (
    TargetQueryReadout,
    prompts,
    student_features,
    target_rope,
    teacher_capture,
)


ARM_NAMES = (
    "target_tokens_target_q",
    "target_tokens_predicted_q",
    "draft_tokens_target_q",
    "draft_tokens_predicted_q",
)
HEADS = 28
KV_HEADS = 4
GROUP_SIZE = HEADS // KV_HEADS


@torch.inference_mode()
def generate_greedy(model, prefix, count, device, *, pad_token_id):
    ids = torch.tensor(prefix, dtype=torch.long, device=device)[None]
    # Disable EOS termination so every arm compares exactly the same positions.
    generated = model.generate(
        ids,
        do_sample=False,
        max_new_tokens=count,
        min_new_tokens=count,
        eos_token_id=None,
        pad_token_id=pad_token_id,
        use_cache=True,
    )
    if generated.shape[1] != len(prefix) + count:
        raise RuntimeError("generation did not produce the requested horizon")
    return generated[0, len(prefix):].cpu().tolist()


@torch.inference_mode()
def predicted_q(model, readout, ids, device):
    features = student_features(model, ids, device, tuple(range(6))).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        raw = readout(features)
    rows = len(ids)
    return target_rope(raw.reshape(rows, 28, HEADS, 128), 1_000_000)


@torch.inference_mode()
def evaluate_arms(prompt_k, target_q, predicted_on_target,
                  target_on_draft, predicted_on_draft, *, prefix):
    """Return each arm's true-target Top-10 overlap on the same Prompt K."""
    horizon = target_q.shape[1] - prefix
    expected_q_shape = (28, horizon, HEADS, 128)
    if (target_q[:, prefix:].shape != expected_q_shape
            or target_on_draft[:, prefix:].shape != expected_q_shape
            or predicted_on_target.shape != expected_q_shape
            or predicted_on_draft.shape != expected_q_shape
            or prompt_k.shape != (28, prefix, KV_HEADS, 128)
            or prefix < 10):
        raise ValueError("Q/K shapes or Prompt length differ")
    device = predicted_on_target.device
    q_arms = torch.stack((
        target_q[:, prefix:].to(device),
        predicted_on_target,
        target_on_draft[:, prefix:].to(device),
        predicted_on_draft.to(device),
    )).float()
    all_recall = {name: [] for name in ARM_NAMES}
    draft_branch_q_recall = []
    by_layer = {layer: {name: [] for name in ARM_NAMES} for layer in range(28)}
    by_head = {head: {name: [] for name in ARM_NAMES} for head in range(HEADS)}
    by_position = {position: {name: [] for name in ARM_NAMES}
                   for position in range(horizon)}
    for layer in range(28):
        for kv_head in range(KV_HEADS):
            queries = q_arms[:, layer, :, kv_head * GROUP_SIZE:
                             (kv_head + 1) * GROUP_SIZE].reshape(4, -1, 128)
            keys = prompt_k[layer, :, kv_head].float().to(queries.device)
            top = (queries @ keys.T).topk(10, dim=-1).indices
            # [arm, horizon, Q heads per KV head, returned token]
            top = top.reshape(4, horizon, GROUP_SIZE, 10)
            reference = top[0]
            overlaps = (top[:, :, :, :, None] == reference[None, :, :, None, :])
            overlaps = overlaps.any(dim=-1).float().mean(dim=-1)
            internal = (top[3, :, :, :, None] == top[2, :, :, None, :])
            draft_branch_q_recall.extend(
                internal.any(dim=-1).float().mean(dim=-1).flatten().cpu().tolist()
            )
            for arm, name in enumerate(ARM_NAMES):
                values = overlaps[arm].cpu().tolist()
                for position, head_values in enumerate(values):
                    for local_head, value in enumerate(head_values):
                        head = kv_head * GROUP_SIZE + local_head
                        all_recall[name].append(value)
                        by_layer[layer][name].append(value)
                        by_head[head][name].append(value)
                        by_position[position][name].append(value)
    mean = lambda values: sum(values) / len(values)
    return {
        "query_cases_per_arm": len(all_recall[ARM_NAMES[0]]),
        "mean_true_top10_recall": {name: mean(all_recall[name])
                                   for name in ARM_NAMES},
        "draft_branch_predicted_q_vs_its_own_target_q": mean(draft_branch_q_recall),
        "by_layer": {str(layer): {name: mean(values[name])
                                  for name in ARM_NAMES}
                     for layer, values in by_layer.items()},
        "by_head": {str(head): {name: mean(values[name])
                                for name in ARM_NAMES}
                    for head, values in by_head.items()},
        "by_future_position": {str(pos): {name: mean(values[name])
                                           for name in ARM_NAMES}
                               for pos, values in by_position.items()},
    }


def weighted_mean(rows, key):
    denominator = sum(row["quality"]["query_cases_per_arm"] for row in rows)
    return {arm: sum(row["quality"]["query_cases_per_arm"]
                     * row["quality"][key][arm] for row in rows) / denominator
            for arm in ARM_NAMES}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--future-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.future_tokens < 1:
        raise ValueError("future-tokens must be positive")
    torch.manual_seed(20260929)
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    target_device = torch.device("cuda:0")
    draft_device = torch.device("cuda:1" if torch.cuda.device_count() > 1
                                else "cuda:0")
    target_tokenizer = AutoTokenizer.from_pretrained(
        args.target_model, local_files_only=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(
        args.draft_model, local_files_only=True)
    if target_tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("target and draft token IDs differ")
    _, old_valid = prompts(target_tokenizer, args.source_root, 512)
    _, natural_valid = corpus_prompts(target_tokenizer, args.source_root)
    samples = old_valid + natural_valid
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, dtype=torch.float16, attn_implementation="eager",
        local_files_only=True).to(target_device).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft_model, dtype=torch.float16, attn_implementation="eager",
        local_files_only=True).to(draft_device).eval()
    readout = TargetQueryReadout(896, 28, HEADS * 128, 896,
                                 fusion="learned").to(draft_device).eval()
    readout.load_state_dict(torch.load(args.readout, map_location=draft_device,
                                       weights_only=True))
    results = []
    started = time.perf_counter()
    for name, prompt in samples:
        target_future = generate_greedy(
            target, prompt, args.future_tokens, target_device,
            pad_token_id=target_tokenizer.eos_token_id)
        draft_future = generate_greedy(
            draft, prompt, args.future_tokens, draft_device,
            pad_token_id=draft_tokenizer.eos_token_id)
        target_ids = prompt + target_future
        draft_ids = prompt + draft_future
        target_q, target_k = teacher_capture(target, target_ids, target_device)
        target_on_draft, draft_branch_k = teacher_capture(
            target, draft_ids, target_device)
        if not torch.allclose(target_k[:, :len(prompt)],
                              draft_branch_k[:, :len(prompt)],
                              atol=1e-3, rtol=1e-3):
            raise RuntimeError("the two target branches disagree on Prompt K")
        predicted_on_target = predicted_q(
            draft, readout, target_ids, draft_device)[len(prompt):]
        predicted_on_draft = predicted_q(
            draft, readout, draft_ids, draft_device)[len(prompt):]
        quality = evaluate_arms(
            target_k[:, :len(prompt)], target_q,
            predicted_on_target.permute(1, 0, 2, 3), target_on_draft,
            predicted_on_draft.permute(1, 0, 2, 3), prefix=len(prompt))
        exact_matches = [left == right for left, right in
                         zip(target_future, draft_future)]
        results.append({
            "name": name,
            "prompt_tokens": len(prompt),
            "future_tokens": args.future_tokens,
            "target_future_token_ids": target_future,
            "draft_future_token_ids": draft_future,
            "same_token_at_position": exact_matches,
            "first_token_divergence": next(
                (i for i, same in enumerate(exact_matches) if not same), None),
            "quality": quality,
        })
        print(json.dumps({"record": name, "token_matches": sum(exact_matches),
                          "recall": quality["mean_true_top10_recall"]}),
              flush=True)
    if not results:
        raise RuntimeError("no validation records")
    report = {
        "method": "greedy target and full-24-layer Draft trajectories; first-six-layer frozen Draft Q readout",
        "target_model": str(args.target_model),
        "draft_model": str(args.draft_model),
        "readout": str(args.readout),
        "prompt_k_only": True,
        "records": len(results),
        "future_tokens_per_record": args.future_tokens,
        "token_agreement_fraction": sum(sum(r["same_token_at_position"])
                                        for r in results) / (len(results) * args.future_tokens),
        "mean_true_top10_recall": weighted_mean(results, "mean_true_top10_recall"),
        "seconds": time.perf_counter() - started,
        "per_record": results,
        "note": "Offline exact Prompt-K retrieval; target-token/target-Q branch is the reference. Target Q on Draft tokens is recomputed by the real target model; no native CAGRA, Decode serving, or generated-output quality claim.",
    }
    if abs(report["mean_true_top10_recall"][ARM_NAMES[0]] - 1.0) > 1e-6:
        raise AssertionError("the reference arm must have perfect overlap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"completed": True, "records": len(results),
                      "token_agreement_fraction": report["token_agreement_fraction"],
                      "mean_true_top10_recall": report["mean_true_top10_recall"]}),
          flush=True)


if __name__ == "__main__":
    main()
