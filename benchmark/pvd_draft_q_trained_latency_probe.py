"""Time the jointly trained token Draft and target-Q readout on one GPU.

Measure the existing offline generate/replay evaluator and a cached forward
that emits Q directly at each generated position. Neither is a serving run.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

from pvd_draft_q_error_decomposition import generate_greedy, predicted_q
from pvd_draft_q_readout_probe import TargetQueryReadout, target_rope


def summarize(samples):
    ordered = sorted(samples)
    return {"median_ms": statistics.median(samples),
            "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            "samples_ms": samples}


@torch.inference_mode()
def cached_forward(student, readout, prompt, steps, device):
    slots = [None] * 6
    handles = []

    def capture(output, slot):
        hidden = output[0] if isinstance(output, tuple) else output
        slots[slot] = hidden[:, -1].detach()

    for slot, layer in enumerate(student.model.layers):
        handles.append(layer.register_forward_hook(
            lambda module, inputs, output, slot=slot: capture(output, slot)))
    try:
        ids = torch.tensor(prompt, device=device)[None]
        positions = torch.arange(len(prompt), len(prompt) + steps, device=device)
        kwargs = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(
            student.forward).parameters else {}
        penalty = student.generation_config.repetition_penalty
        processor = RepetitionPenaltyLogitsProcessor(penalty)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = student(input_ids=ids, use_cache=True, **kwargs)
        torch.cuda.synchronize(device)
        prefill_end = time.perf_counter()
        cache = output.past_key_values
        seen_ids = ids
        token = processor(seen_ids, output.logits[:, -1].float()).argmax(
            dim=-1, keepdim=True)
        future, queries = [], []
        for step in range(steps):
            future.append(token)
            output = student(input_ids=token, past_key_values=cache,
                             use_cache=True, **kwargs)
            cache = output.past_key_values
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                raw = readout(torch.stack(slots, dim=1))
            queries.append(target_rope(raw.reshape(1, 28, 28, 128),
                                       1_000_000,
                                       positions=positions[step:step + 1]))
            if step + 1 < steps:
                seen_ids = torch.cat((seen_ids, token), dim=-1)
                token = processor(seen_ids, output.logits[:, -1].float()).argmax(
                    dim=-1, keepdim=True)
        query = torch.cat(queries)
        torch.cuda.synchronize(device)
        end = time.perf_counter()
        return {"prefill_ms": 1000 * (prefill_end - start),
                "rollout_ms": 1000 * (end - prefill_end),
                "total_ms": 1000 * (end - start),
                "future": torch.cat(future, dim=-1)[0].cpu().tolist(),
                "query": query,
                "prefill_keeps_only_last_logit": bool(kwargs)}
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def replay_forward(student, readout, prompt, steps, device):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    future = generate_greedy(student, prompt, steps, device,
                             pad_token_id=student.config.eos_token_id)
    torch.cuda.synchronize(device)
    generated = time.perf_counter()
    query = predicted_q(student, readout, prompt + future, device)[len(prompt):]
    torch.cuda.synchronize(device)
    end = time.perf_counter()
    return {"generate_ms": 1000 * (generated - start),
            "replay_q_ms": 1000 * (end - generated),
            "total_ms": 1000 * (end - start),
            "future": future, "query": query}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 2155])
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 8])
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("invalid repetition count")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    fixture = tokenizer.encode("Case 40. " + "EEFTRITON " * 1100,
                                add_special_tokens=False)
    if len(fixture) < max(args.lengths):
        raise ValueError("fixture is too short")
    student = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True).to(device)
    student.model.layers = nn.ModuleList(list(student.model.layers[:6]))
    student.model.config.num_hidden_layers = 6
    student.config.num_hidden_layers = 6
    readout = TargetQueryReadout(896, 28, 28 * 128, 896,
                                 fusion="learned").to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    student.load_state_dict(state["student"], strict=True)
    readout.load_state_dict(state["readout"], strict=True)
    del state
    if args.dtype == "fp16":
        student.half()
    student.eval()
    readout.eval()
    results = []
    for length in args.lengths:
        prompt = fixture[:length]
        for steps in args.steps:
            last = {}
            timings = {}
            torch.cuda.reset_peak_memory_stats(device)
            for name, function, fields in (
                ("cached_emit", cached_forward,
                 ("prefill_ms", "rollout_ms", "total_ms")),
                ("existing_evaluator", replay_forward,
                 ("generate_ms", "replay_q_ms", "total_ms")),
            ):
                samples = {field: [] for field in fields}
                for repetition in range(args.warmup + args.repeats):
                    row = function(student, readout, prompt, steps, device)
                    if not torch.isfinite(row["query"]).all().item():
                        raise RuntimeError(f"nonfinite Q in {name}")
                    if repetition >= args.warmup:
                        for field in fields:
                            samples[field].append(row[field])
                    last[name] = row
                timings[name] = {field: summarize(values)
                                 for field, values in samples.items()}
            if last["cached_emit"]["future"] != last["existing_evaluator"]["future"]:
                raise RuntimeError(f"cached and evaluator greedy tokens differ: "
                                   f"{last['cached_emit']['future']} vs "
                                   f"{last['existing_evaluator']['future']}")
            a = last["cached_emit"]["query"].flatten(start_dim=1)
            b = last["existing_evaluator"]["query"].flatten(start_dim=1)
            agreement = torch.nn.functional.cosine_similarity(a, b, dim=-1)
            result = {"prompt_tokens": length, "prediction_tokens": steps,
                      "timings": timings,
                      "cached_vs_replay_min_q_cosine": agreement.min().item(),
                      "greedy_future_tokens": last["cached_emit"]["future"],
                      "prefill_keeps_only_last_logit": last["cached_emit"][
                          "prefill_keeps_only_last_logit"],
                      "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(
                          device) / 2**20}
            results.append(result)
            print(json.dumps(result), flush=True)
            del last, a, b, row
            torch.cuda.empty_cache()
    report = {"gpu": torch.cuda.get_device_name(device),
              "torch": torch.__version__, "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "student_dtype": args.dtype, "readout": "FP32 parameters, FP16 autocast",
              "layers": 6, "readout_rank": 896,
              "repetition_penalty": student.generation_config.repetition_penalty,
              "target_q_shape_per_position": [28, 28, 128],
              "warmup": args.warmup, "repeats": args.repeats,
              "method": "single idle V100S, HF eager, trained CE+Q+score-KL0.1 checkpoint; synchronized wall-clock time",
              "excluded": "model load, tokenizer, teacher, graph construction/readiness, native CAGRA, KV transport and SGLang serving contention",
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
