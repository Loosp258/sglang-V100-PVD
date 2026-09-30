"""Bounded six-layer Draft-Q adaptation on real, EOS-bounded Decode prefixes.

Training and calibration questions exclude the frozen serving benchmark.
The retrieval dataset contains original Prompt K, never generated Decode K.
Candidate models retain the exact six-layer rank-896 serving architecture.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from pvd_draft_q_error_decomposition import generate_greedy
from pvd_draft_q_multitask_probe import truncate_draft
from pvd_draft_q_readout_probe import TargetQueryReadout, target_rope
from pvd_draft_q_trained_latency_probe import cached_forward


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, payload):
    path.write_bytes((json.dumps(payload, indent=2, ensure_ascii=False) + '\n').encode())


def prepare(args):
    import pyarrow.parquet as pq
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    benchmark = json.loads(args.benchmark.read_text(encoding='utf-8'))
    excluded = {x['question'].strip() for x in benchmark['items']}
    for line in args.react.read_text(encoding='utf-8').splitlines():
        excluded.update(x.removeprefix('Question: ').strip() for x in json.loads(line))
    sources = {'gsm8k': [json.loads(x) for x in args.gsm.read_text(encoding='utf-8').splitlines()],
               'hotpotqa': pq.read_table(args.hotpot).to_pylist()}
    rng = random.Random(202609301)
    items = []
    for task, count in [('gsm8k', 36), ('hotpotqa', 20)]:
        indices = list(range(len(sources[task])))
        rng.shuffle(indices)
        accepted = []
        for index in indices:
            row = sources[task][index]
            if row['question'].strip() in excluded:
                continue
            if task == 'gsm8k':
                user = ('Solve this math problem. Show a brief step-by-step solution, '
                        'then write the final numeric answer on its own line as FINAL: <number>.\n\n'
                        + row['question'])
                source_id = str(index)
            else:
                context = '\n\n'.join(title + ':\n' + ''.join(sentences) for title, sentences in
                                      zip(row['context']['title'], row['context']['sentences']))
                user = ('Use the following passages to answer the question. Give a brief '
                        'explanation, then put only the short answer on its own line as '
                        'FINAL: <answer>.\n\nPassages:\n' + context + '\n\nQuestion: ' + row['question'])
                source_id = row['id']
            prompt = tokenizer.apply_chat_template([
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not 48 <= len(ids) <= 1536 or (task == 'hotpotqa' and len(ids) < 600):
                continue
            accepted.append({'id': task + ':' + source_id, 'benchmark': task,
                             'question': row['question'], 'prompt': prompt, 'prompt_ids': ids,
                             'split': 'calibration' if len(accepted) < 4 else 'train'})
            if len(accepted) == count:
                break
        if len(accepted) != count:
            raise ValueError('not enough disjoint fitting questions')
        items.extend(accepted)
    if len({x['question'].strip() for x in items}) != len(items):
        raise ValueError('duplicate questions in adaptation split')
    payload = {'schema': 'pvd.draft_q.decode_questions.v1', 'seed': 202609301,
               'sources_sha256': {'gsm8k_train': sha(args.gsm), 'hotpotqa_dev': sha(args.hotpot),
                                  'excluded_benchmark': sha(args.benchmark), 'excluded_react': sha(args.react)},
               'items': items}
    save_json(args.output, payload)
    print(json.dumps({'prepared': len(items), 'train': 48, 'calibration': 8,
                      'benchmark_overlap': 0, 'sha256': sha(args.output)}), flush=True)


@torch.inference_mode()
def labels(target, ids, original_prompt_length, start, horizon, device, label_positions):
    qs, ks, hooks = [None] * 28, [None] * 28, []
    def capture(attn, inputs, kwargs, layer):
        hidden = inputs[0] if inputs else kwargs['hidden_states']
        shape = (1, hidden.shape[1], -1, 128)
        q = attn.q_proj(hidden).view(shape).transpose(1, 2)
        k = attn.k_proj(hidden).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *kwargs['position_embeddings'])
        qs[layer] = q[0, :, start:start + horizon].transpose(0, 1).cpu().contiguous()
        ks[layer] = k[0, :, :original_prompt_length].transpose(0, 1).cpu().contiguous()
    for layer, block in enumerate(target.model.layers):
        hooks.append(block.self_attn.register_forward_pre_hook(
            lambda module, inputs, kwargs, layer=layer: capture(module, inputs, kwargs, layer),
            with_kwargs=True))
    try:
        result = target.model(input_ids=torch.tensor(ids, device=device)[None], use_cache=False)
        logits = target.lm_head(result.last_hidden_state[0, label_positions])
        if not torch.isfinite(logits).all():
            raise ValueError('nonfinite teacher next-token logits')
        next_ids = logits.argmax(-1).cpu()
    finally:
        for hook in hooks:
            hook.remove()
    post = torch.stack(qs).permute(1, 0, 2, 3).contiguous()
    pre = target_rope(post.float(), 1e6, inverse=True,
                      positions=torch.arange(start, start + horizon)).half()
    keys = torch.stack(ks)
    if not torch.isfinite(post).all() or not torch.isfinite(keys).all():
        raise ValueError('nonfinite target Q/K')
    return {'post_q': post, 'pre_q': pre, 'next_ids': next_ids}, keys


@torch.inference_mode()
def capture(target, student, items, target_device, student_device, tokenizer):
    records = []
    for item in items:
        prompt = item['prompt_ids']
        generated = target.generate(torch.tensor(prompt, device=target_device)[None],
                                    do_sample=False, repetition_penalty=1.0,
                                    max_new_tokens=192, use_cache=True,
                                    pad_token_id=tokenizer.eos_token_id)[0, len(prompt):].cpu().tolist()
        eos_ids = target.generation_config.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        stop = next((i for i, token in enumerate(generated) if token in eos_ids), len(generated))
        generated = generated[:stop]
        keys = None
        count = 0
        for boundary in (0, 4, 32, 64, 128):
            horizon = min(8, len(generated) - boundary)
            if horizon < 3:
                continue
            prefix = prompt + generated[:boundary]
            teacher_future = generated[boundary:boundary + horizon]
            student_future = generate_greedy(student, prefix, horizon, student_device,
                                             pad_token_id=tokenizer.eos_token_id)
            positions = list(range(0, len(prompt), 32)) + list(range(len(prefix) - 1,
                                                                 len(prefix) + horizon - 1))
            positions = sorted(set(positions))
            branches = {}
            for name, future in [('target', teacher_future), ('student', student_future)]:
                branch, branch_keys = labels(target, prefix + future, len(prompt), len(prefix),
                                              horizon, target_device, positions)
                if keys is None:
                    keys = branch_keys
                elif not torch.allclose(keys, branch_keys, atol=1e-3, rtol=1e-3):
                    raise ValueError('original Prompt K changed across Decode branches')
                branch['future'] = future
                branches[name] = branch
            records.append({'id': item['id'], 'benchmark': item['benchmark'], 'split': item['split'],
                            'boundary': boundary, 'prompt_length': len(prompt), 'prefix': prefix,
                            'horizon': horizon, 'prompt_k': keys, 'label_positions': positions,
                            'branches': branches})
            count += 1
        print(json.dumps({'captured': item['id'], 'split': item['split'],
                          'real_generated_tokens': len(generated), 'boundaries': count}), flush=True)
    if any(not any(r['id'] == i['id'] for r in records) for i in items):
        raise ValueError('a selected question produced fewer than three real Decode tokens')
    return records


def forward_features(student, ids, start, horizon, device):
    slots, hooks = [None] * 6, []
    for layer, block in enumerate(student.model.layers):
        def capture(module, inputs, output, layer=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            slots[layer] = hidden[0, start:start + horizon]
        hooks.append(block.register_forward_hook(capture))
    try:
        result = student.model(input_ids=torch.tensor(ids, device=device)[None], use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    return torch.stack(slots, dim=1), result.last_hidden_state[0]


@torch.inference_mode()
def recall(keys, q_arms, device):
    names = ['target_tokens_target_q', 'target_tokens_predicted_q',
             'draft_tokens_target_q', 'draft_tokens_predicted_q']
    by_layer, by_position = defaultdict(list), defaultdict(list)
    top10, coverage = [], []
    queries = torch.stack(q_arms).float().to(device)
    for layer in range(28):
        k = keys[layer].float().to(device)
        for kv_head in range(4):
            q = queries[:, :, layer, kv_head * 7:(kv_head + 1) * 7]
            scores = q @ k[:, kv_head].T
            ids = scores.topk(16, dim=-1).indices
            expected = ids[0, ..., :10]
            overlaps = (ids[..., :10, None] == expected[None, ..., None, :]).any(-1).float().mean(-1)
            top10.append(overlaps.reshape(4, -1).cpu())
            covered = (ids[..., None] == ids[0, ..., None, :4]).any(-2).float().mean(-1)
            coverage.append(covered[:, 2].flatten(1).cpu())
            by_layer[layer].append(overlaps.reshape(4, -1).cpu())
            for pos in range(q.shape[1]):
                by_position[pos].append(overlaps[:, pos].cpu())
    means = torch.cat(top10, -1).mean(-1).tolist()
    cov = torch.cat(coverage, -1).mean(-1).tolist()
    layers = {str(i): dict(zip(names, torch.cat(v, -1).mean(-1).tolist())) for i, v in by_layer.items()}
    return {'cases': torch.cat(top10, -1).shape[1], 'top10': dict(zip(names, means)),
            'top4_coverage_at_k16_position2': dict(zip(names, cov)), 'by_layer': layers,
            'by_position': {str(i): dict(zip(names, torch.cat(v, -1).mean(-1).tolist())) for i, v in by_position.items()}}


@torch.inference_mode()
def evaluate(target, student, readout, records, target_device, device):
    student.eval(); readout.eval()
    rows = []
    for record in records:
        prefix, h = record['prefix'], record['horizon']
        target_future = record['branches']['target']['future']
        rolled = cached_forward(student, readout, prefix, h, device)
        features, _ = forward_features(student, prefix + target_future, len(prefix), h, device)
        with torch.autocast('cuda', dtype=torch.float16):
            pre = readout(features)
        predicted_target = target_rope(pre.reshape(h, 28, 28, 128), 1e6,
                                       positions=torch.arange(len(prefix), len(prefix) + h, device=device))
        actual_draft, keys = labels(target, prefix + rolled['future'], record['prompt_length'],
                                    len(prefix), h, target_device, record['label_positions'])
        if not torch.allclose(keys, record['prompt_k'], atol=1e-3, rtol=1e-3):
            raise ValueError('validation Prompt K changed')
        quality = recall(keys, [record['branches']['target']['post_q'], predicted_target.cpu(),
                                actual_draft['post_q'], rolled['query'].cpu()], device)
        rows.append({'id': record['id'], 'benchmark': record['benchmark'], 'boundary': record['boundary'],
                     'horizon': h, 'quality': quality,
                     'token_agreement': sum(a == b for a, b in zip(target_future, rolled['future'])),
                     'cached_forward_ms': rolled['total_ms']})
    names = list(rows[0]['quality']['top10'])
    count = sum(r['quality']['cases'] for r in rows)
    means = {n: sum(r['quality']['top10'][n] * r['quality']['cases'] for r in rows) / count for n in names}
    coverage = {n: sum(r['quality']['top4_coverage_at_k16_position2'][n] for r in rows) / len(rows) for n in names}
    return {'records': len(rows), 'cases': count, 'mean_top10': means,
            'top4_coverage_at_k16_position2': coverage,
            'token_agreement_fraction': sum(r['token_agreement'] for r in rows) / sum(r['horizon'] for r in rows),
            'worst_layer_mean_top10': {n: min(sum(r['quality']['by_layer'][str(i)][n] * r['horizon'] for r in rows)
                                               / sum(r['horizon'] for r in rows) for i in range(28)) for n in names},
            'per_record': rows}


def retrieval_loss(predicted_pre, branch, record, rng, device):
    h = record['horizon']
    post = target_rope(predicted_pre.float().reshape(h, 28, 28, 128), 1e6,
                       positions=torch.arange(len(record['prefix']), len(record['prefix']) + h, device=device))
    losses = []
    for _ in range(16):
        layer, head = rng.randrange(28), rng.randrange(28)
        k = record['prompt_k'][layer, :, head // 7].float().to(device)
        k = k - k.mean(0)
        teacher = branch['post_q'][:, layer, head].float().to(device) @ k.T / math.sqrt(128)
        prediction = post[:, layer, head] @ k.T / math.sqrt(128)
        scale = teacher.std(-1, keepdim=True).clamp_min(0.25)
        teacher, prediction = teacher / scale, prediction / scale
        log_probs = prediction.log_softmax(-1)
        distribution = teacher.softmax(-1).detach()
        kl = F.kl_div(log_probs, distribution, reduction='batchmean')
        positives = teacher.topk(10, -1).indices
        top_ce = -log_probs.gather(-1, positives).mean()
        losses.append(kl + top_ce)
    return torch.stack(losses).mean()


def train(student, readout, records, device, steps, joint):
    student.eval(); readout.train()
    student.requires_grad_(joint)
    readout.requires_grad_(True)
    cache = {}
    if not joint:
        with torch.no_grad():
            for i, r in enumerate(records):
                for name, branch in r['branches'].items():
                    features, _ = forward_features(student, r['prefix'] + branch['future'],
                                                     len(r['prefix']), r['horizon'], device)
                    cache[i, name] = features.cpu()
    scales = torch.stack([b['pre_q'].float().square().mean((0, 2, 3))
                          for r in records for b in r['branches'].values()]).mean(0).sqrt().clamp_min(1e-3).to(device)
    groups = [{'params': readout.parameters(), 'lr': 5e-5}]
    if joint:
        groups.append({'params': student.parameters(), 'lr': 2e-6})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    rng, pair_rng = random.Random(202609301), random.Random(32920962)
    snapshots = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        index = rng.randrange(len(records)); r = records[index]
        name = 'target' if rng.randrange(2) == 0 else 'student'
        branch = r['branches'][name]
        optimizer.zero_grad(set_to_none=True)
        if joint:
            features, hidden = forward_features(student, r['prefix'] + branch['future'],
                                                 len(r['prefix']), r['horizon'], device)
        else:
            features = cache[index, name].to(device)
        with torch.autocast('cuda', dtype=torch.float16):
            predicted = readout(features)
            token_logits = student.lm_head(hidden[r['label_positions']]) if joint else None
        expected = branch['pre_q'].float().to(device).reshape(r['horizon'], 28, -1)
        q_loss = ((predicted.float() - expected) / scales[None, :, None]).square().mean()
        score_loss = retrieval_loss(predicted, branch, r, pair_rng, device)
        ce = F.cross_entropy(token_logits.float(), branch['next_ids'].to(device)) if joint else q_loss.new_zeros(())
        loss = q_loss + 0.1 * score_loss + 2.0 * ce
        if not torch.isfinite(loss):
            raise ValueError('nonfinite adaptation loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0)
        if joint:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        if step == 1 or step % 200 == 0 or step == steps:
            row = {'step': step, 'joint': joint, 'q_mse': float(q_loss.detach()),
                   'retrieval_loss': float(score_loss.detach()), 'token_ce': float(ce.detach()),
                   'seconds': time.perf_counter() - start}
            snapshots.append(row); print(json.dumps(row), flush=True)
    return {'steps': steps, 'seconds': time.perf_counter() - start, 'snapshots': snapshots}


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(202609301)
    td, device = torch.device('cuda:0'), torch.device('cuda:1')
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model, local_files_only=True)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError('target/Draft token IDs differ')
    student = truncate_draft(AutoModelForCausalLM.from_pretrained(args.draft_model, dtype=torch.float32,
                            attn_implementation='eager', local_files_only=True).to(device)).eval()
    readout = TargetQueryReadout(896, 28, 28 * 128, 896, fusion='learned').to(device).eval()
    initial = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    student.load_state_dict(initial['student']); readout.load_state_dict(initial['readout'])
    target = AutoModelForCausalLM.from_pretrained(args.target_model, dtype=torch.float16,
                            attn_implementation='sdpa', local_files_only=True).to(td).eval()
    target.requires_grad_(False)
    dataset = json.loads(args.dataset.read_text(encoding='utf-8'))
    captures = args.output_dir / 'captures.pt'
    if captures.exists():
        saved = torch.load(captures, map_location='cpu', weights_only=False)
        if saved['dataset_sha256'] != sha(args.dataset) or saved['checkpoint_sha256'] != sha(args.checkpoint):
            raise ValueError('capture identity differs')
        records = saved['records']
    else:
        records = capture(target, student, dataset['items'], td, device, tokenizer)
        torch.save({'dataset_sha256': sha(args.dataset), 'checkpoint_sha256': sha(args.checkpoint),
                    'records': records}, captures)
    training = [r for r in records if r['split'] == 'train']
    calibration = [r for r in records if r['split'] == 'calibration']
    before = evaluate(target, student, readout, calibration, td, device)
    print(json.dumps({'baseline': before['mean_top10'], 'token_agreement': before['token_agreement_fraction']}), flush=True)
    report = {'dataset_sha256': sha(args.dataset), 'baseline_checkpoint_sha256': sha(args.checkpoint),
              'train_questions': 48, 'calibration_questions': 8, 'train_records': len(training),
              'calibration_records': len(calibration), 'original_prompt_k_only': True,
              'baseline': before, 'arms': {}, 'note': 'One offline adaptation round on baseline Draft branches; validation rolls each current model afresh. EOS-bounded teacher trajectories. Exact scoring, no native CAGRA yet.'}
    for name, joint in [('frozen', False), ('joint', True)]:
        student.load_state_dict(initial['student']); readout.load_state_dict(initial['readout'])
        trained = train(student, readout, training, device, args.steps, joint)
        after = evaluate(target, student, readout, calibration, td, device)
        path = args.output_dir / (name + '.pt')
        torch.save({'student': {k: v.cpu() for k, v in student.state_dict().items()},
                    'readout': {k: v.cpu() for k, v in readout.state_dict().items()}}, path)
        report['arms'][name] = {'training': trained, 'quality': after, 'checkpoint_sha256': sha(path)}
        save_json(args.output_dir / 'report.json', report)
        print(json.dumps({'arm_completed': name, 'recall': after['mean_top10'],
                          'token_agreement': after['token_agreement_fraction']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('gsm', 'hotpot', 'benchmark', 'react', 'output', 'target-model'):
        prep.add_argument('--' + name, type=Path, required=True)
    runner = sub.add_parser('run')
    for name in ('dataset', 'checkpoint', 'target-model', 'draft-model', 'output-dir'):
        runner.add_argument('--' + name, type=Path, required=True)
    runner.add_argument('--steps', type=int, default=1200)
    args = parser.parse_args()
    prepare(args) if args.command == 'prepare' else run(args)


if __name__ == '__main__':
    main()
