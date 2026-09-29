"""Condition a frozen Draft-Q readout on a causal target-prefix hidden state.

The selected target state is captured at the last *committed Prompt* token. No
future target token or hidden state is exposed to the adapter. This is an
offline feasibility probe; D does not currently receive this state.
"""

from __future__ import annotations

import argparse
import json
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
    weighted_mean,
)
from pvd_draft_q_multitask_probe import q_scale, score_kl, truncate_draft
from pvd_draft_q_readout_probe import (
    TargetQueryReadout,
    prompts,
    student_features,
    target_rope,
    teacher_capture,
)


class PrefixStateResidual(nn.Module):
    def __init__(self, target_width=3584, draft_width=896, rank=128):
        super().__init__()
        self.context = nn.Linear(target_width, rank)
        self.future = nn.Linear(draft_width, rank)
        self.output = nn.Linear(rank, 28 * 28 * 128, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, prefix_hidden, future_draft_features):
        fused = F.gelu(
            self.context(prefix_hidden)[None]
            + self.future(future_draft_features[:, -1])
        )
        return self.output(fused).reshape(len(future_draft_features), 28, -1)


@torch.inference_mode()
def capture_prefix_hidden(target, prompt, device, *, layer=12):
    ids = torch.tensor(prompt, dtype=torch.long, device=device)[None]
    if layer == "final":
        hidden = target.model(input_ids=ids, use_cache=False).last_hidden_state
        state = hidden[0, -1].float().cpu().contiguous()
        if not torch.isfinite(state).all():
            raise RuntimeError("target final prefix hidden is nonfinite")
        return state
    if layer != 12:
        raise ValueError("unsupported target hidden layer")
    slot = []

    def capture(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        slot.append(hidden[0, -1].float().cpu().contiguous())

    hook = target.model.layers[11].register_forward_hook(capture)
    try:
        target.model(input_ids=ids, use_cache=False)
    finally:
        hook.remove()
    if len(slot) != 1 or not torch.isfinite(slot[0]).all():
        raise RuntimeError("target layer-12 prefix hidden is missing or nonfinite")
    return slot[0]


@torch.inference_mode()
def cached_student_branch(student, readout, ids, *, prefix, device):
    features = student_features(student, ids, device, tuple(range(6)))
    future_features = features[prefix:].contiguous().to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        base_pre = readout(future_features)
    return {
        "future_features": future_features.half().cpu().contiguous(),
        "base_pre": base_pre.half().cpu().contiguous(),
    }


@torch.inference_mode()
def prepare_training(target, student, readout, records, *, device, layer):
    for record in records:
        prefix = len(record["prompt"])
        record["prefix_hidden"] = capture_prefix_hidden(
            target, record["prompt"], device, layer=layer)
        for branch in record["branches"].values():
            branch.update(cached_student_branch(
                student, readout, record["prompt"] + branch["future"],
                prefix=prefix, device=device))
        print(json.dumps({"prepared_train": record["name"]}), flush=True)


@torch.inference_mode()
def prepare_validation(target, student, readout, valid_prompts, *, device,
                       horizon, eos_token_id, layer):
    output = []
    for name, prompt in valid_prompts:
        prefix = len(prompt)
        hidden = capture_prefix_hidden(target, prompt, device, layer=layer)
        target_future = generate_greedy(
            target, prompt, horizon, device, pad_token_id=eos_token_id)
        student_future = generate_greedy(
            student, prompt, horizon, device, pad_token_id=eos_token_id)
        target_ids = prompt + target_future
        student_ids = prompt + student_future
        q_target, k_target = teacher_capture(target, target_ids, device)
        q_student, k_student = teacher_capture(target, student_ids, device)
        if not torch.allclose(k_target[:, :prefix], k_student[:, :prefix],
                              atol=1e-3, rtol=1e-3):
            raise RuntimeError(f"{name}: Prompt K differs")
        output.append({
            "name": name,
            "prompt_tokens": prefix,
            "prefix_hidden": hidden,
            "target_future": target_future,
            "student_future": student_future,
            "target_q": q_target,
            "student_q": q_student,
            "prompt_k": k_target[:, :prefix].contiguous(),
            "target_branch": cached_student_branch(
                student, readout, target_ids, prefix=prefix, device=device),
            "student_branch": cached_student_branch(
                student, readout, student_ids, prefix=prefix, device=device),
        })
        print(json.dumps({"prepared_valid": name}), flush=True)
    return output


@torch.inference_mode()
def evaluate(records, adapter, *, device, hidden_mode="real"):
    adapter.eval()
    rows = []
    for index, record in enumerate(records):
        prefix = record["prompt_tokens"]
        if hidden_mode == "real":
            hidden = record["prefix_hidden"]
        elif hidden_mode == "shuffled":
            hidden = records[(index + 1) % len(records)]["prefix_hidden"]
        elif hidden_mode == "zero":
            hidden = torch.zeros_like(record["prefix_hidden"])
        else:
            raise ValueError("invalid hidden mode")
        hidden = hidden.to(device)
        predicted = []
        for branch_name in ("target_branch", "student_branch"):
            branch = record[branch_name]
            features = branch["future_features"].float().to(device)
            pre = branch["base_pre"].float().to(device)
            pre = pre + adapter(hidden, features)
            horizon = len(features)
            post = target_rope(
                pre.reshape(horizon, 28, 28, 128), 1_000_000,
                positions=torch.arange(prefix, prefix + horizon, device=device))
            predicted.append(post.permute(1, 0, 2, 3))
        quality = evaluate_arms(
            record["prompt_k"], record["target_q"], predicted[0],
            record["student_q"], predicted[1], prefix=prefix)
        rows.append({
            "name": record["name"],
            "token_matches": sum(a == b for a, b in
                                 zip(record["target_future"],
                                     record["student_future"])),
            "quality": quality,
        })
    return {
        "hidden_mode": hidden_mode,
        "query_cases_per_arm": sum(r["quality"]["query_cases_per_arm"]
                                   for r in rows),
        "token_agreement_fraction": sum(r["token_matches"] for r in rows)
                                    / (len(rows) * len(records[0]["target_future"])),
        "mean_true_top10_recall": weighted_mean(rows,
                                                 "mean_true_top10_recall"),
        "per_record": rows,
    }


def train_adapter(adapter, records, *, device, steps, score_weight):
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=2e-4,
                                  weight_decay=0.01)
    scales = q_scale(records).to(device)
    record_rng = random.Random(20260930)
    pair_rng = random.Random(20260929)
    snapshots = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        record = records[record_rng.randrange(len(records))]
        branch = record["branches"]["target" if record_rng.randrange(2) == 0
                                    else "student"]
        prefix = len(record["prompt"])
        features = branch["future_features"].float().to(device)
        base_pre = branch["base_pre"].float().to(device)
        target_pre = branch["pre_q"].float().to(device).reshape(len(features),
                                                               28, -1)
        adapter.train()
        predicted_pre = base_pre + adapter(
            record["prefix_hidden"].to(device), features)
        mse = (((predicted_pre - target_pre)
                / scales[None, :, None]).square().mean())
        if score_weight:
            kl = score_kl(
                predicted_pre, branch["post_q"], record["prompt_k"],
                torch.arange(prefix, prefix + len(features), device=device),
                rng=pair_rng)
        else:
            kl = mse.new_zeros(())
        loss = mse + score_weight * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if step in (1, 100, 250, 500, steps):
            snapshot = {"step": step, "q_mse": float(mse.detach()),
                        "score_kl": float(kl.detach()),
                        "total_loss": float(loss.detach())}
            snapshots.append(snapshot)
            print(json.dumps(snapshot), flush=True)
    torch.cuda.synchronize(device)
    return snapshots, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--train-captures", type=Path, required=True)
    parser.add_argument("--reuse-valid-capture", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--score-weight", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--hidden-layer", choices=("12", "final"), default="12")
    parser.add_argument("--expected-baseline", type=float)
    args = parser.parse_args()
    if args.steps < 1 or args.horizon < 1 or args.score_weight < 0:
        raise ValueError("invalid steps, horizon or score weight")
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260930)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model,
                                              local_files_only=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model,
                                                    local_files_only=True)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("token IDs differ")
    old_train, old_valid = prompts(tokenizer, args.source_root, 512)
    natural_train, natural_valid = corpus_prompts(tokenizer, args.source_root)
    expected_train = old_train + natural_train
    student = truncate_draft(AutoModelForCausalLM.from_pretrained(
        args.draft_model, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True).to(device)).eval()
    readout = TargetQueryReadout(896, 28, 28 * 128, 896,
                                 fusion="learned").to(device).eval()
    state = torch.load(args.student_checkpoint, map_location=device,
                       weights_only=True)
    student.load_state_dict(state["student"])
    readout.load_state_dict(state["readout"])
    student.requires_grad_(False)
    readout.requires_grad_(False)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, dtype=torch.float16,
        attn_implementation="sdpa" if args.hidden_layer == "final" else "eager",
        local_files_only=True).to(device).eval()
    target.requires_grad_(False)
    records = torch.load(args.train_captures, map_location="cpu",
                         weights_only=False)
    if [r["name"] for r in records] != [name for name, _ in expected_train]:
        raise ValueError("train capture order differs")
    if any(r["prompt"] != ids for r, (_, ids) in
           zip(records, expected_train)):
        raise ValueError("train capture token IDs differ")
    started = time.perf_counter()
    layer = "final" if args.hidden_layer == "final" else 12
    prepare_training(target, student, readout, records, device=device,
                     layer=layer)
    if args.reuse_valid_capture:
        valid = torch.load(args.reuse_valid_capture, map_location="cpu",
                           weights_only=False)
        expected_valid = old_valid + natural_valid
        if [r["name"] for r in valid] != [name for name, _ in expected_valid]:
            raise ValueError("cached validation names differ")
        for record, (_, prompt) in zip(valid, expected_valid):
            if record["prompt_tokens"] != len(prompt):
                raise ValueError("cached validation Prompt length differs")
            record["prefix_hidden"] = capture_prefix_hidden(
                target, prompt, device, layer=layer)
    else:
        valid = prepare_validation(
            target, student, readout, old_valid + natural_valid, device=device,
            horizon=args.horizon, eos_token_id=tokenizer.eos_token_id,
            layer=layer)
    preparation_seconds = time.perf_counter() - started
    torch.save(valid, args.output_dir / "valid_capture.pt")
    del target, student, readout
    torch.cuda.empty_cache()
    adapter = PrefixStateResidual().to(device)
    baseline = evaluate(valid, adapter, device=device)
    if (args.expected_baseline is not None
            and abs(baseline["mean_true_top10_recall"][
                "draft_tokens_predicted_q"] - args.expected_baseline) > 0.01):
        raise RuntimeError("zero-initialized adapter does not reproduce the frozen Draft baseline")
    print(json.dumps({"baseline": baseline["mean_true_top10_recall"]}),
          flush=True)
    snapshots, train_seconds = train_adapter(
        adapter, records, device=device, steps=args.steps,
        score_weight=args.score_weight)
    trained = evaluate(valid, adapter, device=device)
    shuffled = evaluate(valid, adapter, device=device,
                        hidden_mode="shuffled")
    zero = evaluate(valid, adapter, device=device, hidden_mode="zero")
    report = {
        "method": f"causal target {args.hidden_layer} last-Prompt-token hidden plus frozen six-layer Draft features",
        "train_windows": len(records),
        "valid_windows": len(valid),
        "steps": args.steps,
        "score_weight": args.score_weight,
        "preparation_seconds": preparation_seconds,
        "train_seconds": train_seconds,
        "baseline": baseline,
        "trained_real_hidden": trained,
        "trained_shuffled_hidden": shuffled,
        "trained_zero_hidden": zero,
        "loss_snapshots": snapshots,
        "gpu_peak_mib": torch.cuda.max_memory_allocated(device) // 1048576,
        "note": "The target hidden is causal Prompt-only information. Offline exact Prompt-K retrieval; the current D serving route does not receive this hidden state."
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    torch.save(adapter.cpu().state_dict(), args.output_dir / "adapter.pt")
    print(json.dumps({"completed": True,
                      "baseline": baseline["mean_true_top10_recall"],
                      "real": trained["mean_true_top10_recall"],
                      "shuffled": shuffled["mean_true_top10_recall"],
                      "zero": zero["mean_true_top10_recall"]}),
          flush=True)


if __name__ == "__main__":
    main()
