"""Train a six-layer token Draft and target-Q head on generated trajectories.

The experiment compares token+Q training with and without true-Prompt-K score
distillation. All serving paths remain unchanged. Captures are reusable so the
two arms see identical initial Draft trajectories and teacher labels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from pvd_draft_q_corpus_probe import corpus_prompts
from pvd_draft_q_error_decomposition import (
    evaluate_arms,
    generate_greedy,
    predicted_q,
    weighted_mean,
)
from pvd_draft_q_readout_probe import (
    TargetQueryReadout,
    prompts,
    target_rope,
    teacher_capture,
)


def truncate_draft(model):
    model.model.layers = nn.ModuleList(list(model.model.layers[:6]))
    model.model.config.num_hidden_layers = 6
    model.config.num_hidden_layers = 6
    return model


@torch.inference_mode()
def capture_records(target, student, train_prompts, *, device, horizon,
                    eos_token_id):
    records = []
    for name, prompt in train_prompts:
        target_future = generate_greedy(
            target, prompt, horizon, device, pad_token_id=eos_token_id)
        student_future = generate_greedy(
            student, prompt, horizon, device, pad_token_id=eos_token_id)
        prefix = len(prompt)
        token_positions = sorted(set(
            list(range(64, prefix, 16))
            + list(range(prefix - 1, prefix + horizon - 1))
        ))
        branches = {}
        prompt_k = None
        for branch, future in (("target", target_future),
                               ("student", student_future)):
            ids = prompt + future
            q, k, next_ids = teacher_capture(
                target, ids, device, next_token_positions=token_positions)
            if prompt_k is None:
                prompt_k = k[:, :prefix].contiguous()
            elif not torch.allclose(prompt_k, k[:, :prefix],
                                    atol=1e-3, rtol=1e-3):
                raise RuntimeError(f"{name}: Prompt K differs across branches")
            future_q = q[:, prefix:].permute(1, 0, 2, 3).to(device)
            positions = torch.arange(prefix, prefix + horizon, device=device)
            pre_q = target_rope(future_q, 1_000_000, inverse=True,
                                positions=positions)
            branches[branch] = {
                "future": future,
                "pre_q": pre_q.half().cpu().contiguous(),
                "post_q": future_q.half().cpu().contiguous(),
                "teacher_next_ids": next_ids,
            }
        records.append({
            "name": name,
            "prompt": prompt,
            "prompt_k": prompt_k,
            "teacher_label_positions": token_positions,
            "branches": branches,
        })
        print(json.dumps({"captured": name,
                          "same_future_tokens": sum(
                              a == b for a, b in zip(
                                  target_future, student_future))}),
              flush=True)
    return records


@torch.inference_mode()
def evaluate_live(target, student, readout, valid_prompts, *, device,
                  horizon, eos_token_id):
    student.eval()
    readout.eval()
    rows = []
    for name, prompt in valid_prompts:
        target_future = generate_greedy(
            target, prompt, horizon, device, pad_token_id=eos_token_id)
        student_future = generate_greedy(
            student, prompt, horizon, device, pad_token_id=eos_token_id)
        target_ids = prompt + target_future
        student_ids = prompt + student_future
        q_target, k_target = teacher_capture(target, target_ids, device)
        q_student, k_student = teacher_capture(target, student_ids, device)
        if not torch.allclose(k_target[:, :len(prompt)],
                              k_student[:, :len(prompt)],
                              atol=1e-3, rtol=1e-3):
            raise RuntimeError(f"{name}: validation Prompt K differs")
        predicted_target = predicted_q(
            student, readout, target_ids, device)[len(prompt):]
        predicted_student = predicted_q(
            student, readout, student_ids, device)[len(prompt):]
        quality = evaluate_arms(
            k_target[:, :len(prompt)], q_target,
            predicted_target.permute(1, 0, 2, 3), q_student,
            predicted_student.permute(1, 0, 2, 3), prefix=len(prompt))
        rows.append({
            "name": name,
            "prompt_tokens": len(prompt),
            "target_future_token_ids": target_future,
            "student_future_token_ids": student_future,
            "same_token_at_position": [a == b for a, b in
                                       zip(target_future, student_future)],
            "quality": quality,
        })
    return {
        "records": len(rows),
        "query_cases_per_arm": sum(r["quality"]["query_cases_per_arm"]
                                   for r in rows),
        "token_agreement_fraction": sum(
            sum(r["same_token_at_position"]) for r in rows
        ) / (len(rows) * horizon),
        "mean_true_top10_recall": weighted_mean(
            rows, "mean_true_top10_recall"),
        "draft_branch_predicted_q_vs_its_own_target_q": sum(
            r["quality"]["draft_branch_predicted_q_vs_its_own_target_q"]
            for r in rows) / len(rows),
        "per_record": rows,
    }


def q_scale(records):
    sum_squares = torch.zeros(28)
    count = 0
    for record in records:
        for branch in record["branches"].values():
            q = branch["pre_q"].float()
            sum_squares += q.square().sum(dim=(0, 2, 3))
            count += q.shape[0] * q.shape[2] * q.shape[3]
    return (sum_squares / count).sqrt().clamp_min(1e-3)


def score_kl(predicted_pre, true_post, prompt_k, positions, *, rng,
             pair_count=8, temperature=2.0):
    horizon = predicted_pre.shape[0]
    predicted_post = target_rope(
        predicted_pre.reshape(horizon, 28, 28, 128), 1_000_000,
        positions=positions)
    pairs = [(rng.randrange(28), rng.randrange(28))
             for _ in range(pair_count)]
    losses = []
    for layer, head in pairs:
        keys = prompt_k[layer, :, head // 7].float().to(predicted_pre.device)
        divisor = math.sqrt(128) * temperature
        teacher_scores = true_post[:, layer, head].float().to(
            predicted_pre.device) @ keys.T / divisor
        student_scores = predicted_post[:, layer, head] @ keys.T / divisor
        teacher_prob = F.softmax(teacher_scores.detach(), dim=-1)
        losses.append(F.kl_div(
            F.log_softmax(student_scores, dim=-1),
            teacher_prob,
            reduction="batchmean",
        ))
    return torch.stack(losses).mean()


def train(student, readout, records, *, device, steps, score_weight,
          trunk_lr, head_lr):
    student.train()
    readout.train()
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [{"params": student.parameters(), "lr": trunk_lr},
         {"params": readout.parameters(), "lr": head_lr}],
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda")
    record_rng = random.Random(20260929)
    pair_rng = random.Random(12920962)
    scales = q_scale(records).to(device)
    slots = [None] * 6
    hooks = []
    for layer, block in enumerate(student.model.layers):
        def capture(module, inputs, output, layer=layer):
            slots[layer] = output[0] if isinstance(output, tuple) else output
        hooks.append(block.register_forward_hook(capture))
    snapshots = []
    started = time.perf_counter()
    try:
        for step in range(1, steps + 1):
            record = records[record_rng.randrange(len(records))]
            branch_name = "target" if record_rng.randrange(2) == 0 else "student"
            branch = record["branches"][branch_name]
            prefix = len(record["prompt"])
            horizon = len(branch["future"])
            ids = torch.tensor(record["prompt"] + branch["future"],
                               dtype=torch.long, device=device)[None]
            label_positions = torch.tensor(record["teacher_label_positions"],
                                           dtype=torch.long, device=device)
            teacher_next = torch.tensor(branch["teacher_next_ids"],
                                        dtype=torch.long, device=device)
            expected_pre = branch["pre_q"].to(device).reshape(horizon, 28, -1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                hidden = student.model(input_ids=ids, use_cache=False)
                token_logits = student.lm_head(
                    hidden.last_hidden_state[0, label_positions])
                features = torch.stack(
                    [slot[0, prefix:prefix + horizon] for slot in slots],
                    dim=1)
                predicted_pre = readout(features)
            token_loss = F.cross_entropy(token_logits.float(), teacher_next)
            q_loss = (((predicted_pre.float() - expected_pre.float())
                       / scales[None, :, None]).square().mean())
            if score_weight:
                absolute_positions = torch.arange(
                    prefix, prefix + horizon, device=device)
                ranking_loss = score_kl(
                    predicted_pre.float(), branch["post_q"],
                    record["prompt_k"], absolute_positions, rng=pair_rng)
            else:
                ranking_loss = token_loss.new_zeros(())
            loss = token_loss + 4.0 * q_loss + score_weight * ranking_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if step in (1, 100, 200, 400, 600, steps):
                snapshot = {
                    "step": step,
                    "branch": branch_name,
                    "token_ce": float(token_loss.detach()),
                    "normalized_q_mse": float(q_loss.detach()),
                    "score_kl": float(ranking_loss.detach()),
                    "total_loss": float(loss.detach()),
                }
                snapshots.append(snapshot)
                print(json.dumps(snapshot), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    torch.cuda.synchronize(device)
    return snapshots, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-captures", type=Path)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--future-tokens", type=int, default=8)
    parser.add_argument("--score-weight", type=float, default=0.0)
    parser.add_argument("--trunk-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    args = parser.parse_args()
    if args.steps < 1 or args.future_tokens < 1 or args.score_weight < 0:
        raise ValueError("steps/future-tokens must be positive and score weight nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.manual_seed(20260929)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model,
                                              local_files_only=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model,
                                                    local_files_only=True)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("target/draft token IDs differ")
    old_train, old_valid = prompts(tokenizer, args.source_root, 512)
    natural_train, natural_valid = corpus_prompts(tokenizer, args.source_root)
    train_prompts = old_train + natural_train
    valid_prompts = old_valid + natural_valid
    student = truncate_draft(AutoModelForCausalLM.from_pretrained(
        args.draft_model, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True).to(device))
    student.eval()
    readout = TargetQueryReadout(896, 28, 28 * 128, 896,
                                 fusion="learned").to(device).eval()
    readout.load_state_dict(torch.load(args.readout, map_location=device,
                                       weights_only=True))
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, dtype=torch.float16, attn_implementation="eager",
        local_files_only=True).to(device).eval()
    target.requires_grad_(False)
    if args.reuse_captures:
        records = torch.load(args.reuse_captures, map_location="cpu",
                             weights_only=False)
        capture_seconds = 0.0
    else:
        started = time.perf_counter()
        records = capture_records(
            target, student, train_prompts, device=device,
            horizon=args.future_tokens, eos_token_id=tokenizer.eos_token_id)
        capture_seconds = time.perf_counter() - started
        torch.save(records, args.output_dir / "captures.pt")
    if [r["name"] for r in records] != [name for name, _ in train_prompts]:
        raise ValueError("capture names differ from current train split")
    if any(r["prompt"] != prompt for r, (_, prompt) in
           zip(records, train_prompts)):
        raise ValueError("capture token IDs differ from current train split")
    before = evaluate_live(target, student, readout, valid_prompts,
                           device=device, horizon=args.future_tokens,
                           eos_token_id=tokenizer.eos_token_id)
    print(json.dumps({"baseline": before["mean_true_top10_recall"],
                      "token_agreement": before["token_agreement_fraction"]}),
          flush=True)
    snapshots, train_seconds = train(
        student, readout, records, device=device, steps=args.steps,
        score_weight=args.score_weight, trunk_lr=args.trunk_lr,
        head_lr=args.head_lr)
    after = evaluate_live(target, student, readout, valid_prompts,
                          device=device, horizon=args.future_tokens,
                          eos_token_id=tokenizer.eos_token_id)
    report = {
        "train_windows": len(records),
        "valid_windows": len(valid_prompts),
        "future_tokens": args.future_tokens,
        "steps": args.steps,
        "score_weight": args.score_weight,
        "token_ce_weight": 1.0,
        "q_mse_weight": 4.0,
        "trunk_lr": args.trunk_lr,
        "head_lr": args.head_lr,
        "capture_seconds": capture_seconds,
        "train_seconds": train_seconds,
        "baseline": before,
        "trained": after,
        "loss_snapshots": snapshots,
        "gpu_peak_mib": torch.cuda.max_memory_allocated(device) // 1048576,
        "note": "Six-layer token-generating student. Initial student-generated plus target-generated trajectories; 80 training windows. Exact Prompt-K quality, no native CAGRA or online Decode."
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    torch.save({"student": student.cpu().state_dict(),
                "readout": readout.cpu().state_dict()},
               args.output_dir / "trained.pt")
    print(json.dumps({"completed": True,
                      "trained_token_agreement": after["token_agreement_fraction"],
                      "trained_recall": after["mean_true_top10_recall"]}),
          flush=True)


if __name__ == "__main__":
    main()
