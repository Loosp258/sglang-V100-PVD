"""Capture training-only, post-training Draft trajectories for budget calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pvd_draft_q_error_decomposition import generate_greedy
from pvd_draft_q_multitask_probe import truncate_draft
from pvd_draft_q_readout_probe import teacher_capture


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--original-train-captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--future-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.future_tokens < 2:
        raise ValueError("need at least two future tokens")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model,
                                              local_files_only=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model,
                                                    local_files_only=True)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("token IDs differ")
    original = torch.load(args.original_train_captures, map_location="cpu",
                          weights_only=False)
    student = truncate_draft(AutoModelForCausalLM.from_pretrained(
        args.draft_model, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True).to(device)).eval()
    state = torch.load(args.student_checkpoint, map_location=device,
                       weights_only=True)
    student.load_state_dict(state["student"])
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, dtype=torch.float16, attn_implementation="eager",
        local_files_only=True).to(device).eval()
    records = []
    for record in original:
        prompt = record["prompt"]
        future = generate_greedy(student, prompt, args.future_tokens, device,
                                 pad_token_id=tokenizer.eos_token_id)
        q, k = teacher_capture(target, prompt + future, device)
        records.append({
            "name": record["name"],
            "prompt": prompt,
            "future": future,
            "post_q": q[:, len(prompt):].permute(1, 0, 2, 3).contiguous(),
            "prompt_k": k[:, :len(prompt)].contiguous(),
        })
        print(json.dumps({"refreshed": record["name"],
                          "token_changes": sum(a != b for a, b in zip(
                              future,
                              record["branches"]["student"]["future"]))}),
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, args.output)
    print(json.dumps({"completed": True, "records": len(records),
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
