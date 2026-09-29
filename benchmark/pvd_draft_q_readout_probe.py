"""Offline six-layer draft -> target-Q distillation feasibility probe.

This probes query prediction only. It does not alter the PVD serving path and
does not claim that a six-layer truncation is a usable token-generating draft.
Run on two GPUs: target teacher on cuda:0, narrow student on cuda:1.
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


TRAIN_FILES = (
    "python/sglang/srt/disaggregation/pvd/vector_store.py",
    "python/sglang/srt/disaggregation/pvd/coordinator.py",
    "python/sglang/srt/disaggregation/pvd/conn.py",
    "python/sglang/srt/disaggregation/pvd/target_probe.py",
    "python/sglang/srt/disaggregation/pvd/cuda_refresh_driver.py",
    "python/sglang/srt/disaggregation/pvd/draft_sglang.py",
    "python/sglang/srt/disaggregation/pvd/prompt_index.py",
    "python/sglang/srt/disaggregation/pvd/cagra_backend.py",
)
VALID_FILES = (
    "AGENTS.md",
    "python/sglang/srt/disaggregation/pvd/runtime.py",
)


def prompts(tokenizer, source_root: Path, length: int):
    train, valid = [], []
    for label, names, output in (
        ("train", TRAIN_FILES, train),
        ("valid", VALID_FILES, valid),
    ):
        for filename in names:
            content = (source_root / filename).read_text(encoding="utf-8")
            ids = tokenizer.encode(content, add_special_tokens=False)
            if len(ids) < length + 32:
                raise ValueError(f"{filename} has too few tokens for {label}")
            offsets = (0, min(length + 37, len(ids) - length)) if label == "train" else (0,)
            for offset in offsets:
                output.append((f"{label}:{filename}:{offset}", ids[offset:offset + length]))
    online = tokenizer.encode(
        "Case 40. " + "EEFTRITON " * 430, add_special_tokens=False
    )
    valid.append(("valid:online-case40", online[:2 * length]))
    rng = random.Random(20260929)
    valid.append(("valid:random", [rng.randrange(3, 1000) for _ in range(length)]))
    return train, valid


def teacher_capture(model, token_ids, device):
    """Capture the actual post-RoPE Q/K used by each HF Qwen attention layer."""
    q_slots = [None] * model.config.num_hidden_layers
    k_slots = [None] * model.config.num_hidden_layers
    handles = []

    def capture(attn, inputs, kwargs, layer):
        hidden = inputs[0] if inputs else kwargs["hidden_states"]
        shape = (hidden.shape[0], hidden.shape[1], -1, attn.head_dim)
        q = attn.q_proj(hidden).view(shape).transpose(1, 2)
        k = attn.k_proj(hidden).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *kwargs["position_embeddings"])
        q_slots[layer] = q[0].transpose(0, 1).contiguous().cpu()
        k_slots[layer] = k[0].transpose(0, 1).contiguous().cpu()

    for layer, block in enumerate(model.model.layers):
        handles.append(
            block.self_attn.register_forward_pre_hook(
                lambda module, inputs, kwargs, layer=layer: capture(
                    module, inputs, kwargs, layer
                ),
                with_kwargs=True,
            )
        )
    try:
        ids = torch.tensor(token_ids, dtype=torch.long, device=device)[None, :]
        with torch.inference_mode():
            model.model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()
    if any(item is None for item in q_slots + k_slots):
        raise RuntimeError("teacher failed to capture every Q/K layer")
    return torch.stack(q_slots), torch.stack(k_slots)


def student_features(model, token_ids, device, anchor_layers):
    slots = [None] * len(anchor_layers)
    handles = []
    for slot, layer in enumerate(anchor_layers):
        block = model.model.layers[layer]
        handles.append(
            block.register_forward_hook(
                lambda module, inputs, output, slot=slot: slots.__setitem__(
                    slot, output[0].detach().contiguous().cpu()
                )
            )
        )
    try:
        ids = torch.tensor(token_ids, dtype=torch.long, device=device)[None, :]
        with torch.inference_mode():
            model.model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()
    if any(item is None for item in slots):
        raise RuntimeError("student failed to capture all six anchors")
    return torch.stack(slots).transpose(0, 1).contiguous()


class TargetQueryReadout(nn.Module):
    def __init__(self, width: int, layers: int, q_width: int, rank: int,
                 fusion: str = "selected"):
        super().__init__()
        self.fusion = fusion
        self.anchor = nn.Parameter(torch.randn(6, width, rank) * (width ** -0.5))
        self.output = nn.Parameter(torch.randn(layers, rank, q_width) * (rank ** -0.5))
        self.bias = nn.Parameter(torch.zeros(layers, q_width))
        self.register_buffer(
            "anchor_for_layer",
            torch.arange(layers, dtype=torch.long).mul(6).div(layers, rounding_mode="floor"),
        )
        if fusion == "learned":
            logits = torch.zeros(layers, 6)
            logits.scatter_(1, self.anchor_for_layer[:, None], 2.0)
            self.anchor_logits = nn.Parameter(logits)
        elif fusion != "selected":
            raise ValueError("fusion must be selected or learned")

    def forward(self, features):
        anchors = F.gelu(torch.einsum("baw,awr->bar", features, self.anchor))
        if self.fusion == "learned":
            selected = torch.einsum(
                "bar,la->blr", anchors, self.anchor_logits.softmax(dim=-1)
            )
        else:
            selected = anchors[:, self.anchor_for_layer]
        return torch.einsum("blr,lrd->bld", selected, self.output) + self.bias


def target_rope(query, theta, inverse=False, positions=None):
    """Apply (or undo) target Qwen RoPE over [position, layer, head, dim]."""
    rows, _, _, dim = query.shape
    freq = torch.arange(0, dim, 2, device=query.device, dtype=torch.float32)
    freq = theta ** (-freq / dim)
    if positions is None:
        positions = torch.arange(rows, device=query.device)
    phase = torch.outer(positions, freq)
    cosine = torch.cat((phase.cos(), phase.cos()), dim=-1)[:, None, None]
    sine = torch.cat((phase.sin(), phase.sin()), dim=-1)[:, None, None]
    if inverse:
        sine = -sine
    half = dim // 2
    rotated = torch.cat((-query[..., half:], query[..., :half]), dim=-1)
    return query.float() * cosine + rotated.float() * sine


@torch.inference_mode()
def retrieve_quality(readout, records, *, device, query_heads, ratio,
                     q_target="post_rope", rope_theta=1_000_000,
                     exact_first_layers=0):
    all_recalls, all_mass = [], []
    by_record = []
    for record in records:
        features = record["features"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            predicted = readout(features).float().cpu()
        target_q = record["q"]
        target_k = record["k"]
        rows, layers, q_width = predicted.shape
        head_dim = target_q.shape[-1]
        predicted = predicted.reshape(rows, layers, -1, head_dim).to(device)
        if q_target == "pre_rope":
            predicted = target_rope(predicted, rope_theta)
        predicted = predicted.permute(1, 0, 2, 3)
        if exact_first_layers:
            predicted[:exact_first_layers] = record["q"][:exact_first_layers].to(device)
        positions = list(range(128, rows, 64))
        if rows - 1 not in positions:
            positions.append(rows - 1)
        recalls, masses = [], []
        for layer in range(layers):
            for head in query_heads:
                keys = target_k[layer, :, head // ratio].float().to(device)
                truth = target_q[layer, positions, head].float().to(device)
                guess = predicted[layer, positions, head].to(device)
                actual_scores = truth @ keys.T
                guessed_scores = guess @ keys.T
                for position_index, position in enumerate(positions):
                    actual_scores[position_index, position:] = -torch.inf
                    guessed_scores[position_index, position:] = -torch.inf
                true_top = actual_scores.topk(10, dim=-1).indices
                pred_top = guessed_scores.topk(10, dim=-1).indices
                recalls.extend(
                    (pred_top[:, :, None] == true_top[:, None, :])
                    .any(dim=2).float().mean(dim=1).cpu().tolist()
                )
                attention = (actual_scores / head_dim**0.5).softmax(dim=-1)
                masses.extend(
                    attention.gather(1, pred_top).sum(dim=1).cpu().tolist()
                )
        all_recalls.extend(recalls)
        all_mass.extend(masses)
        by_record.append({
            "name": record["name"], "positions": len(positions),
            "mean_recall_at_10": sum(recalls) / len(recalls),
            "min_query_recall_at_10": min(recalls),
            "mean_attention_mass_at_10": sum(masses) / len(masses),
        })
    return {
        "mean_recall_at_10": sum(all_recalls) / len(all_recalls),
        "min_query_recall_at_10": min(all_recalls),
        "mean_attention_mass_at_10": sum(all_mass) / len(all_mass),
        "records": by_record,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--reuse-capture", type=Path)
    parser.add_argument("--reuse-teacher-capture", type=Path)
    parser.add_argument("--draft-layers", type=int, choices=(6, 24), default=6)
    parser.add_argument("--q-target", choices=("pre_rope", "post_rope"), default="post_rope")
    parser.add_argument("--retrieval-loss-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-loss-weight", type=float, default=0.0)
    parser.add_argument("--sample-min-position", type=int, default=0)
    parser.add_argument("--anchor-fusion", choices=("selected", "learned"),
                        default="selected")
    args = parser.parse_args()
    if (torch.cuda.device_count() < 2 or args.rows < 256 or args.rank < 1
            or not 0 <= args.sample_min_position < args.rows):
        raise ValueError("two CUDA GPUs, rows >= 256 and positive readout rank required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(20260929)
    torch.manual_seed(20260929)
    target_device, draft_device = torch.device("cuda:0"), torch.device("cuda:1")
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model, local_files_only=True)
    if target_tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("target and draft tokenizers do not share exact IDs")
    train_prompts, valid_prompts = prompts(target_tokenizer, args.source_root, args.rows)
    started = time.perf_counter()
    target_config = AutoConfig.from_pretrained(args.target_model, local_files_only=True)
    draft_config = AutoConfig.from_pretrained(args.draft_model, local_files_only=True)
    if target_config.model_type != "qwen2" or draft_config.model_type != "qwen2":
        raise ValueError("this probe requires Qwen2 teacher and draft")
    if args.reuse_capture and args.reuse_teacher_capture:
        raise ValueError("choose either complete capture reuse or teacher-only reuse")
    if args.reuse_capture:
        records = torch.load(args.reuse_capture, map_location="cpu", weights_only=False)
        expected = [name for name, _ in train_prompts + valid_prompts]
        if [record["name"] for record in records] != expected:
            raise ValueError("cached records do not match the current train/valid prompts")
        if any(record.get("draft_layers", 6) != args.draft_layers for record in records):
            raise ValueError("cached draft features use a different layer count")
        capture_seconds = 0.0
    else:
        records = None
        if args.reuse_teacher_capture:
            records = torch.load(args.reuse_teacher_capture, map_location="cpu", weights_only=False)
            expected = [name for name, _ in train_prompts + valid_prompts]
            if [record["name"] for record in records] != expected:
                raise ValueError("cached teacher records do not match current prompts")
        else:
            target = AutoModelForCausalLM.from_pretrained(
                args.target_model, dtype=torch.float16,
                attn_implementation="eager", local_files_only=True,
            ).to(target_device).eval()
        draft = AutoModelForCausalLM.from_pretrained(
            args.draft_model, dtype=torch.float16,
            attn_implementation="eager", local_files_only=True,
        ).to(draft_device).eval()
        if args.draft_layers < draft.config.num_hidden_layers:
            draft.model.layers = nn.ModuleList(list(draft.model.layers[:args.draft_layers]))
            draft.model.config.num_hidden_layers = args.draft_layers
            draft.config.num_hidden_layers = args.draft_layers
        anchor_layers = tuple((i + 1) * args.draft_layers // 6 - 1 for i in range(6))
        if records is None:
            records = []
        for index, (name, token_ids) in enumerate(train_prompts + valid_prompts):
            if args.reuse_teacher_capture:
                q, k = records[index]["q"], records[index]["k"]
            else:
                q, k = teacher_capture(target, token_ids, target_device)
            features = student_features(draft, token_ids, draft_device, anchor_layers)
            if args.reuse_teacher_capture:
                records[index]["features"] = features
                records[index]["draft_layers"] = args.draft_layers
            else:
                records.append({"name": name, "features": features, "q": q,
                                "k": k, "draft_layers": args.draft_layers})
            print(json.dumps({"captured": name, "rows": len(token_ids)}), flush=True)
        capture_seconds = time.perf_counter() - started
        torch.save(records, args.output_dir / "captures.pt")
    train_records = records[:len(train_prompts)]
    valid_records = records[len(train_prompts):]
    q_heads = target_config.num_attention_heads
    kv_heads = target_config.num_key_value_heads
    ratio = q_heads // kv_heads
    if q_heads % kv_heads or target_config.hidden_size // q_heads != 128:
        raise ValueError("unexpected target Q/K head layout")
    rope_theta = target_config.rope_parameters["rope_theta"]
    selected_heads = tuple(k * ratio + h for k in range(kv_heads) for h in (0, 1))
    readout = TargetQueryReadout(
        draft_config.hidden_size, target_config.num_hidden_layers,
        q_heads * 128, args.rank, fusion=args.anchor_fusion,
    ).to(draft_device)
    baseline = retrieve_quality(
        readout, valid_records, device=draft_device,
        query_heads=selected_heads, ratio=ratio,
        q_target=args.q_target, rope_theta=rope_theta,
    )
    features = torch.cat([item["features"] for item in train_records]).to(draft_device)
    label_parts = []
    for item in train_records:
        q = item["q"].permute(1, 0, 2, 3).to(draft_device)
        if args.q_target == "pre_rope":
            q = target_rope(q, rope_theta, inverse=True).half()
        label_parts.append(q.reshape(len(item["features"]), target_config.num_hidden_layers, -1))
    labels = torch.cat(label_parts)
    scale = labels.float().square().mean(dim=(0, 2)).sqrt().clamp_min(1e-3)
    if args.retrieval_loss_weight and args.pairwise_loss_weight:
        raise ValueError("run attention-distribution and pairwise losses separately")
    if args.retrieval_loss_weight or args.pairwise_loss_weight:
        if args.q_target != "pre_rope":
            raise ValueError("retrieval loss expects pre-RoPE readout")
        if any(len(item["features"]) != args.rows for item in train_records):
            raise ValueError("retrieval loss expects equal-length train records")
        train_q = torch.stack([item["q"] for item in train_records]).to(draft_device)
        train_k = torch.stack([item["k"] for item in train_records]).to(draft_device)
        key_positions = torch.arange(args.rows, device=draft_device)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=1e-3, weight_decay=0.01)
    train_started = time.perf_counter()
    losses = []
    readout.train()
    for step in range(args.steps):
        if args.retrieval_loss_weight or args.pairwise_loss_weight or args.sample_min_position:
            indices = (
                torch.randint(len(train_records), (96,), device=draft_device) * args.rows
                + torch.randint(
                    max(128 if (args.retrieval_loss_weight or args.pairwise_loss_weight)
                        else 0, args.sample_min_position),
                    args.rows, (96,), device=draft_device,
                )
            )
        else:
            indices = torch.randint(len(features), (96,), device=draft_device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = readout(features[indices])
        mse = (
            (prediction.float() - labels[indices].float())
            / scale[None, :, None]
        ).square().mean()
        loss = mse
        if args.retrieval_loss_weight or args.pairwise_loss_weight:
            record_ids, local_positions = indices // args.rows, indices % args.rows
            shaped = prediction.reshape(len(indices), target_config.num_hidden_layers,
                                         q_heads, 128)
            retrieval_losses = []
            for _ in range(4):
                layer = random.randrange(target_config.num_hidden_layers)
                head = random.choice(selected_heads)
                predicted_q = target_rope(
                    shaped[:, layer:layer+1, head:head+1], rope_theta,
                    positions=local_positions,
                )[:, 0, 0]
                true_q = train_q[record_ids, layer, local_positions, head].float()
                keys = train_k[record_ids, layer, :, head // ratio].float()
                predicted_scores = torch.bmm(
                    keys, predicted_q.unsqueeze(-1)
                ).squeeze(-1) / (128 ** 0.5)
                true_scores = torch.bmm(
                    keys, true_q.unsqueeze(-1)
                ).squeeze(-1) / (128 ** 0.5)
                causal_mask = key_positions[None, :] >= local_positions[:, None]
                predicted_scores = predicted_scores.masked_fill(causal_mask, -1e4)
                true_scores = true_scores.masked_fill(causal_mask, -1e4)
                distribution = true_scores.softmax(dim=-1).detach()
                if args.retrieval_loss_weight:
                    retrieval_losses.append(
                        -(distribution * predicted_scores.log_softmax(dim=-1))
                        .sum(dim=-1).mean()
                    )
                else:
                    teacher_top = true_scores.topk(50, dim=-1).indices
                    positive_ids = teacher_top[:, :10]
                    positive_scores = predicted_scores.gather(1, positive_ids)
                    next_scores = predicted_scores.gather(1, teacher_top[:, 10:30])
                    near_boundary = F.softplus(
                        (next_scores[:, None, :] - positive_scores[:, :, None]
                         + 0.1) / 0.5
                    ).mean()
                    mistaken_ids = predicted_scores.detach().topk(10, dim=-1).indices
                    mistaken_scores = predicted_scores.gather(1, mistaken_ids)
                    is_true_positive = (
                        mistaken_ids[:, :, None] == positive_ids[:, None, :]
                    ).any(dim=-1)
                    mistaken_penalties = F.softplus(
                        (mistaken_scores[:, None, :] - positive_scores[:, :, None]
                         + 0.1) / 0.5
                    ).mean(dim=1)
                    mistaken = (
                        (mistaken_penalties * (~is_true_positive)).sum(dim=1)
                        / (~is_true_positive).sum(dim=1).clamp_min(1)
                    ).mean()
                    retrieval_losses.append(0.5 * (near_boundary + mistaken))
            retrieval_loss = torch.stack(retrieval_losses).mean()
            loss = mse + (
                args.retrieval_loss_weight + args.pairwise_loss_weight
            ) * retrieval_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0)
        optimizer.step()
        if step in (0, 24, 49, 99, 149, 199, args.steps - 1):
            value = float(loss.detach())
            losses.append((step + 1, value))
            print(json.dumps({"step": step + 1, "training_objective": value,
                              "normalized_q_mse": float(mse.detach())}), flush=True)
    torch.cuda.synchronize(draft_device)
    train_seconds = time.perf_counter() - train_started
    readout.eval()
    quality = retrieve_quality(
        readout, valid_records, device=draft_device,
        query_heads=selected_heads, ratio=ratio,
        q_target=args.q_target, rope_theta=rope_theta,
    )
    train_quality = retrieve_quality(
        readout, train_records[:1], device=draft_device,
        query_heads=selected_heads, ratio=ratio,
        q_target=args.q_target, rope_theta=rope_theta,
    )
    report = {
        "target": str(args.target_model), "draft": str(args.draft_model),
        "draft_layers_executed": args.draft_layers, "draft_width": draft_config.hidden_size,
        "target_layers": target_config.num_hidden_layers,
        "target_q_heads": q_heads, "readout_rank": args.rank,
        "q_target": args.q_target, "steps": args.steps,
        "retrieval_loss_weight": args.retrieval_loss_weight,
        "pairwise_loss_weight": args.pairwise_loss_weight,
        "sample_min_position": args.sample_min_position,
        "anchor_fusion": args.anchor_fusion,
        "sampled_query_heads": selected_heads,
        "train_prompts": [name for name, _ in train_prompts],
        "valid_prompts": [name for name, _ in valid_prompts],
        "train_positions": len(features),
        "capture_seconds": capture_seconds, "train_seconds": train_seconds,
        "losses": losses, "untrained": baseline, "trained": quality,
        "trained_first_train_record": train_quality,
        "teacher_gpu_peak_mib": torch.cuda.max_memory_allocated(target_device) // 1048576,
        "student_gpu_peak_mib": torch.cuda.max_memory_allocated(draft_device) // 1048576,
        "note": "Frozen pretrained Qwen0.5B blocks; only readout trained."
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    torch.save(readout.cpu().state_dict(), args.output_dir / "readout.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
