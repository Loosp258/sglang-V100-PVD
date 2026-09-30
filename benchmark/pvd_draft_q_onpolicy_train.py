"""Decode-focused token distillation with observed sparse-D recovery prefixes.

Same six-block/rank-896 architecture, fixed old calibration, no benchmark
questions or gold answers in training. Compare token-position weighting only.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

from pvd_draft_q_decode_train import (evaluate, forward_features, labels,
    retrieval_loss, save_json, sha)
from pvd_draft_q_error_decomposition import generate_greedy
from pvd_draft_q_multitask_probe import truncate_draft
from pvd_draft_q_readout_probe import TargetQueryReadout


def bounded(ids, eos):
    stop = next((i for i, token in enumerate(ids) if token in eos), None)
    return ids if stop is None else ids[:stop + 1]


@torch.inference_mode()
def continuation(model, prefix, count, device, eos):
    output = model.generate(torch.tensor(prefix, device=device)[None], do_sample=False,
        repetition_penalty=1.05, max_new_tokens=count, use_cache=True,
        pad_token_id=model.config.eos_token_id)[0, len(prefix):].cpu().tolist()
    return bounded(output, eos)


@torch.inference_mode()
def capture(target, student, items, observed, td, sd, eos):
    records = []
    for item in items:
        if item['split'] != 'train':
            continue
        prompt = item['prompt_ids']
        generated = continuation(target, prompt, 384 if item['benchmark'] == 'gsm8k' else 128, td, eos)
        prompt_k = None
        before = len(records)
        sources = [('teacher', generated)]
        if item['id'] in observed:
            sources.append(('sparse_d', bounded(observed[item['id']]['output_ids'], eos)))
        for source, trajectory in sources:
            for boundary in (0, 4, 32, 64, 128, 256):
                remaining = trajectory[boundary:boundary + 8]
                if not remaining:
                    continue
                prefix = prompt + trajectory[:boundary]
                teacher_future = (remaining if source == 'teacher' else
                                  continuation(target, prefix, len(remaining), td, eos))
                h = min(len(remaining), len(teacher_future), 8)
                if h == 0:
                    continue
                futures = {'target': teacher_future[:h],
                           'student': generate_greedy(student, prefix, h, sd,
                                                      pad_token_id=student.config.eos_token_id)}
                if source == 'sparse_d':
                    futures['observed'] = remaining[:h]
                positions = sorted(set(list(range(0, len(prompt), 32)) +
                                       list(range(len(prefix) - 1, len(prefix) + h - 1))))
                branches = {}
                for name, future in futures.items():
                    branch, keys = labels(target, prefix + future, len(prompt), len(prefix), h,
                                          td, positions, repetition_penalty=1.05)
                    if prompt_k is None:
                        prompt_k = keys
                    elif not torch.allclose(prompt_k, keys, atol=1e-3, rtol=1e-3):
                        raise ValueError('Prompt K changed across teacher/recovery branches')
                    branches[name] = {**branch, 'future': future}
                records.append({'id': item['id'], 'benchmark': item['benchmark'], 'split': 'train',
                    'source': source, 'boundary': boundary, 'prompt_length': len(prompt),
                    'prefix': prefix, 'horizon': h, 'prompt_k': prompt_k,
                    'label_positions': positions, 'branches': branches})
        print(json.dumps({'captured': item['id'], 'teacher_tokens': len(generated),
                          'records': len(records) - before,
                          'sparse_d_tokens': len(observed.get(item['id'], {}).get('output_ids', []))}), flush=True)
    return records


def train(student, readout, records, device, steps, decode_only):
    student.eval(); readout.train()
    student.requires_grad_(True); readout.requires_grad_(True)
    scales = torch.stack([b['pre_q'].float().square().mean((0, 2, 3))
        for r in records for b in r['branches'].values()]).mean(0).sqrt().clamp_min(1e-3).to(device)
    optimizer = torch.optim.AdamW([{'params': readout.parameters(), 'lr': 5e-5},
                                  {'params': student.parameters(), 'lr': 2e-6}], weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    rng, heads_rng = random.Random(202609302), random.Random(32920963)
    processor = RepetitionPenaltyLogitsProcessor(student.generation_config.repetition_penalty)
    start = time.perf_counter(); snapshots = []
    for step in range(1, steps + 1):
        r = records[rng.randrange(len(records))]
        names = sorted(r['branches']); name = names[rng.randrange(len(names))]
        branch = r['branches'][name]; ids = r['prefix'] + branch['future']
        positions = r['label_positions']
        selected = [i for i, p in enumerate(positions) if not decode_only or p >= len(r['prefix']) - 1]
        if not selected or len([i for i in selected if positions[i] >= len(r['prefix']) - 1]) != r['horizon']:
            raise ValueError('Decode token labels do not match horizon')
        optimizer.zero_grad(set_to_none=True)
        features, hidden = forward_features(student, ids, len(r['prefix']), r['horizon'], device)
        with torch.autocast('cuda', dtype=torch.float16):
            predicted = readout(features)
            logits = student.lm_head(hidden[[positions[i] for i in selected]])
        # Match the student's actual serving repetition processor while keeping
        # gradients through the processed logits. Teacher labels use target1.05.
        logits = torch.cat([processor(torch.tensor(ids[:positions[i] + 1], device=device)[None],
                                     logits[j:j + 1].float()) for j, i in enumerate(selected)])
        expected = branch['pre_q'].float().to(device).reshape(r['horizon'], 28, -1)
        q_loss = ((predicted.float() - expected) / scales[None, :, None]).square().mean()
        k_loss = retrieval_loss(predicted, branch, r, heads_rng, device)
        ce = F.cross_entropy(logits, branch['next_ids'][selected].to(device))
        loss = q_loss + 0.1 * k_loss + 2.0 * ce
        if not torch.isfinite(loss):
            raise ValueError('nonfinite on-policy loss')
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        if step == 1 or step % 400 == 0 or step == steps:
            row = {'step': step, 'decode_only': decode_only, 'source': r['source'],
                   'q_mse': float(q_loss.detach()), 'retrieval_loss': float(k_loss.detach()),
                   'token_ce': float(ce.detach()), 'token_labels': len(selected),
                   'seconds': time.perf_counter() - start}
            snapshots.append(row); print(json.dumps(row), flush=True)
    return {'steps': steps, 'seconds': time.perf_counter() - start, 'snapshots': snapshots}


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(args.questions.read_text(encoding='utf-8'))
    old = torch.load(args.calibration_captures, map_location='cpu', weights_only=False)
    if old['dataset_sha256'] != sha(args.questions):
        raise ValueError('frozen calibration identity mismatch')
    observed = [json.loads(line) for line in args.trajectories.read_text(encoding='utf-8').splitlines()]
    train_ids = {r['id'] for r in dataset['items'] if r['split'] == 'train'}
    if len(observed) != 12 or len({r['id'] for r in observed}) != 12 or any(
        r['id'] not in train_ids or not r.get('completed_entry_released')
        or len(r['output_ids']) != r['completion_tokens'] for r in observed):
        raise ValueError('invalid or non-training observed sparse-D trajectory')
    td, sd = torch.device('cuda:0'), torch.device('cuda:1')
    torch.manual_seed(202609302)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model, local_files_only=True)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError('tokenizer ID mismatch')
    student = truncate_draft(AutoModelForCausalLM.from_pretrained(args.draft_model, dtype=torch.float32,
        attn_implementation='eager', local_files_only=True).to(sd)).eval()
    readout = TargetQueryReadout(896, 28, 28 * 128, 896, fusion='learned').to(sd).eval()
    initial = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    student.load_state_dict(initial['student']); readout.load_state_dict(initial['readout'])
    target = AutoModelForCausalLM.from_pretrained(args.target_model, dtype=torch.float16,
        attn_implementation='sdpa', local_files_only=True).to(td).eval().requires_grad_(False)
    eos = target.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    identity = {'questions_sha256': sha(args.questions), 'checkpoint_sha256': sha(args.checkpoint),
                'trajectories_sha256': sha(args.trajectories), 'teacher_penalty': 1.05,
                'max_math_tokens': 384, 'max_reading_tokens': 128}
    capture_path = args.output_dir / 'captures.pt'
    if capture_path.exists():
        saved = torch.load(capture_path, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('capture identity differs')
        records = saved['records']
    else:
        records = capture(target, student, dataset['items'], {r['id']: r for r in observed}, td, sd, eos)
        torch.save({'identity': identity, 'records': records}, capture_path)
    calibration = [r for r in old['records'] if r['split'] == 'calibration']
    before = evaluate(target, student, readout, calibration, td, sd)
    report = {'identity': identity, 'baseline': before, 'arms': {},
              'capture_records_by_source': dict(Counter(r['source'] for r in records)),
              'calibration_note': 'Unchanged25 prefixes from the previous8 held-out questions; teacher1.0 reference retained for comparability.',
              'note': 'Single observed-policy capture at prior75a0 weights. Full-attention HF teacher recovery on real sparse-D token prefixes, not sparse-D Q labels. Original Prompt K only.'}
    save_json(args.output_dir / 'report.json', report)
    print(json.dumps({'baseline': before['mean_top10']}), flush=True)
    for name, decode_only in (('all_positions', False), ('decode_only', True)):
        student.load_state_dict(initial['student']); readout.load_state_dict(initial['readout'])
        torch.manual_seed(202609302)
        trained = train(student, readout, records, sd, args.steps, decode_only)
        quality = evaluate(target, student, readout, calibration, td, sd)
        path = args.output_dir / (name + '.pt')
        torch.save({'student': {k: v.cpu() for k, v in student.state_dict().items()},
                    'readout': {k: v.cpu() for k, v in readout.state_dict().items()}}, path)
        report['arms'][name] = {'training': trained, 'quality': quality, 'checkpoint_sha256': sha(path)}
        save_json(args.output_dir / 'report.json', report)
        print(json.dumps({'completed_arm': name, 'recall': quality['mean_top10'],
                          'token_agreement': quality['token_agreement_fraction']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('questions', 'trajectories', 'calibration-captures', 'checkpoint',
                 'target-model', 'draft-model', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4000)
    run(parser.parse_args())
