"""Diverse-text target-Q readout adaptation using real 7B Q/K captures.

Validation keeps the original held-out fixtures and adds disjoint HotpotQA
items. This remains an offline Q proxy, not a token-generating Draft.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from pvd_draft_q_readout_probe import (
    TargetQueryReadout,
    retrieve_quality,
    student_features,
    target_rope,
    teacher_capture,
)


def spaced_windows(tokenizer, text, count, label):
    ids = tokenizer.encode(text, add_special_tokens=False)
    available = len(ids) // 512
    if available < count:
        raise ValueError(f"{label}: only {available} complete windows, need {count}")
    stride = max(1, available // count)
    starts = [index * stride * 512 for index in range(count)]
    return [(f"{label}:{index}", ids[start:start + 512])
            for index, start in enumerate(starts)]


def corpus_prompts(tokenizer, root):
    hotpot = (root / "benchmark/react/hotpotqa_100.jsonl").read_text(
        encoding="utf-8").splitlines()
    articles = (root / "benchmark/llm_judge/articles.jsonl").read_text(
        encoding="utf-8")
    docs = "\n\n".join(path.read_text(encoding="utf-8") for path in
                       sorted((root / "docs").rglob("*.md")))
    examples = "\n\n".join(path.read_text(encoding="utf-8") for path in
                           sorted((root / "examples").rglob("*.py")))
    train = []
    for text, count, label in (
        ("\n".join(hotpot[:80]), 24, "train:hotpot-first80"),
        (articles, 16, "train:articles"),
        (docs, 16, "train:docs"),
        (examples, 8, "train:examples"),
    ):
        train.extend(spaced_windows(tokenizer, text, count, label))
    valid = spaced_windows(tokenizer, "\n".join(hotpot[80:]), 4,
                           "valid:hotpot-last20")
    return train, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--old-captures", type=Path, required=True)
    parser.add_argument("--base-readout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--rank", type=int, choices=(256, 512, 896), default=256)
    parser.add_argument("--reuse-corpus-captures", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(20260929)
    torch.manual_seed(20260929)
    target_device, draft_device = torch.device("cuda:0"), torch.device("cuda:1")
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model,
                                                     local_files_only=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model,
                                                    local_files_only=True)
    if target_tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("target/draft token IDs differ")
    train_prompts, natural_valid_prompts = corpus_prompts(target_tokenizer,
                                                          args.source_root)
    old_records = torch.load(args.old_captures, map_location="cpu",
                             weights_only=False)
    old_train = [record for record in old_records if record["name"].startswith("train:")]
    old_valid = [record for record in old_records if record["name"].startswith("valid:")]
    if args.reuse_corpus_captures:
        data = torch.load(args.reuse_corpus_captures, map_location="cpu",
                          weights_only=False)
        train_records = data["train"]
        natural_valid = data["natural_valid"]
        capture_seconds = 0.0
    else:
        target = AutoModelForCausalLM.from_pretrained(
            args.target_model, dtype=torch.float16, attn_implementation="eager",
            local_files_only=True,
        ).to(target_device).eval()
        draft = AutoModelForCausalLM.from_pretrained(
            args.draft_model, dtype=torch.float16, attn_implementation="eager",
            local_files_only=True,
        ).to(draft_device).eval()
        draft.model.layers = nn.ModuleList(list(draft.model.layers[:6]))
        draft.model.config.num_hidden_layers = 6
        draft.config.num_hidden_layers = 6
        positions = torch.arange(64, 512, 2, dtype=torch.long)

        def shrink(record):
            q = record["q"][:, positions].permute(1, 0, 2, 3).to(draft_device)
            pre_q = target_rope(q, 1_000_000, inverse=True,
                                positions=positions.to(draft_device))
            return {
                "name": record["name"],
                "features": record["features"][positions].contiguous(),
                "pre_q": pre_q.half().cpu().contiguous(),
            }

        train_records = [shrink(record) for record in old_train]
        natural_valid = []
        capture_started = time.perf_counter()
        for name, ids in train_prompts + natural_valid_prompts:
            q, k = teacher_capture(target, ids, target_device)
            features = student_features(draft, ids, draft_device, tuple(range(6)))
            record = {"name": name, "features": features, "q": q, "k": k}
            if name.startswith("train:"):
                train_records.append(shrink(record))
            else:
                natural_valid.append(record)
            print(json.dumps({"captured": name}), flush=True)
        capture_seconds = time.perf_counter() - capture_started
        torch.save({"train": train_records, "natural_valid": natural_valid},
                   args.output_dir / "captures.pt")
        del target, draft
        torch.cuda.empty_cache()
    expected_train = [record["name"] for record in old_train] + [name for name, _ in train_prompts]
    if [record["name"] for record in train_records] != expected_train:
        raise ValueError("cached corpus train window order differs")
    if [record["name"] for record in natural_valid] != [name for name, _ in natural_valid_prompts]:
        raise ValueError("cached natural validation window order differs")
    readout = TargetQueryReadout(896, 28, 28 * 128, args.rank,
                                 fusion="learned").to(draft_device)
    base_state = torch.load(args.base_readout, map_location=draft_device,
                            weights_only=True)
    if args.rank == 256:
        readout.load_state_dict(base_state)
    else:
        base = TargetQueryReadout(896, 28, 28 * 128, 256,
                                  fusion="learned").to(draft_device)
        base.load_state_dict(base_state)
        with torch.no_grad():
            readout.anchor[:, :, :256].copy_(base.anchor)
            readout.output[:, :256].copy_(base.output)
            readout.output[:, 256:].zero_()
            readout.bias.copy_(base.bias)
            readout.anchor_logits.copy_(base.anchor_logits)
        del base
    heads = (0, 1, 7, 8, 14, 15, 21, 22)

    def evaluate():
        readout.eval()
        return {
            "old": retrieve_quality(readout, old_valid, device=draft_device,
                                    query_heads=heads, ratio=7,
                                    q_target="pre_rope"),
            "natural": retrieve_quality(readout, natural_valid,
                                        device=draft_device, query_heads=heads,
                                        ratio=7, q_target="pre_rope"),
        }

    baseline = evaluate()
    print(json.dumps({"step": 0, "old_recall": baseline["old"]["mean_recall_at_10"],
                      "natural_recall": baseline["natural"]["mean_recall_at_10"]}),
          flush=True)
    features = torch.cat([record["features"] for record in train_records]).to(draft_device)
    labels = torch.cat([record["pre_q"].reshape(len(record["features"]), 28, -1)
                        for record in train_records]).to(draft_device)
    scale = labels.float().square().mean(dim=(0, 2)).sqrt().clamp_min(1e-3)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=2e-4,
                                  weight_decay=0.01)
    snapshots = [{"step": 0, "quality": baseline}]
    train_started = time.perf_counter()
    for step in range(1, args.steps + 1):
        readout.train()
        indices = torch.randint(len(features), (96,), device=draft_device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = readout(features[indices])
        loss = (((prediction.float() - labels[indices].float())
                 / scale[None, :, None]).square().mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0)
        optimizer.step()
        if step in (500, 1000, args.steps):
            quality = evaluate()
            snapshots.append({"step": step, "loss": float(loss.detach()),
                              "quality": quality})
            print(json.dumps({"step": step, "train_loss": float(loss.detach()),
                              "old_recall": quality["old"]["mean_recall_at_10"],
                              "natural_recall": quality["natural"]["mean_recall_at_10"]}),
                  flush=True)
    torch.cuda.synchronize(draft_device)
    report = {
        "new_train_windows": len(train_prompts),
        "old_train_windows": len(old_train),
        "train_positions": len(features),
        "natural_valid_windows": len(natural_valid),
        "steps": args.steps,
        "rank": args.rank,
        "capture_seconds": capture_seconds,
        "train_seconds": time.perf_counter() - train_started,
        "snapshots": snapshots,
        "gpu_peak_mib": torch.cuda.max_memory_allocated(draft_device) // 1048576,
        "note": "Frozen six-block student, diverse readout fine-tune only."
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    torch.save(readout.cpu().state_dict(), args.output_dir / "readout.pt")
    print(json.dumps({"completed": True, "final_old_recall":
                      snapshots[-1]["quality"]["old"]["mean_recall_at_10"],
                      "final_natural_recall": snapshots[-1]["quality"]["natural"]["mean_recall_at_10"]}),
          flush=True)


if __name__ == "__main__":
    main()
