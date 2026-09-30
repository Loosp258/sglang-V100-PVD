"""One-GPU token-only probe of a pinned Qwen2.5-7B/EAGLE3 pair.

The target supplies the root token and post-block auxiliary features. Neither
the root nor any target verification counts as a successful draft prediction.
Author EAGLE eager inference is used unchanged; no target-Q readout is loaded.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

from pvd_draft_q_multitask_probe import truncate_draft


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def checked_logits(logits):
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite logits: FP16 compatibility gate failed')
    return logits.float()


def pick(logits, seen, penalty):
    return penalty(seen, checked_logits(logits)).argmax(-1, keepdim=True)


@torch.inference_mode()
def target_prefix(target, prefix):
    ids = torch.tensor(prefix, device='cuda:0')[None]
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = target(input_ids=ids, use_cache=True, output_hidden_states=True,
                    logits_to_keep=1)
    torch.cuda.synchronize()
    pref = time.perf_counter()
    features = torch.cat([output.hidden_states[i] for i in (2, 14, 25)], dim=-1)
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite target auxiliary features')
    torch.cuda.synchronize()
    return ids, output, features, {'target_prefill_ms': (pref - start) * 1000,
        'feature_fusion_ms': (time.perf_counter() - pref) * 1000}


@torch.inference_mode()
def target_future(target, ids, output, root, steps, penalty, eos):
    cache, token, seen, future = output.past_key_values, root, ids, []
    for _ in range(steps):
        seen = torch.cat((seen, token), dim=-1)
        output = target(input_ids=token, past_key_values=cache, use_cache=True,
                        logits_to_keep=1)
        token = pick(output.logits[:, -1], seen, penalty)
        future.append(token.item())
        cache = output.past_key_values
        if token.item() in eos:
            break
    return future


@torch.inference_mode()
def eagle_future(draft, features, ids, root, mapping, steps, penalty):
    # Author topK_genrate shifts input_ids[:,1:] after appending the root.
    # The N target features are paired with N shifted tokens, including root.
    seen = torch.cat((ids, root), dim=-1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    hidden, cache = draft(features, input_ids=seen[:, 1:], use_cache=True)
    torch.cuda.synchronize()
    prefix_end = time.perf_counter()
    future = []
    for position in range(steps):
        logits = checked_logits(draft.lm_head(draft.norm(hidden[:, -1])))
        full_logits = logits.new_full((1, draft.config.vocab_size), float('-inf'))
        full_logits[:, mapping] = logits
        # -inf outside the reduced vocabulary is deliberate, not nonfinite output.
        token = penalty(seen, full_logits).argmax(-1, keepdim=True)
        future.append(token.item())
        seen = torch.cat((seen, token), dim=-1)
        if position + 1 < steps:
            hidden, cache = draft(hidden[:, -1:], input_ids=token,
                                  past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return future, {'prefix_ms': (prefix_end - start) * 1000,
                    'rollout_ms': (end - prefix_end) * 1000,
                    'total_ms': (end - start) * 1000}


@torch.inference_mode()
def old_future(student, ids, root, steps, penalty):
    seen = torch.cat((ids, root), dim=-1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = student(input_ids=seen, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    prefix_end = time.perf_counter()
    cache, future = output.past_key_values, []
    for position in range(steps):
        token = pick(output.logits[:, -1], seen, penalty)
        future.append(token.item())
        seen = torch.cat((seen, token), dim=-1)
        if position + 1 < steps:
            output = student(input_ids=token, past_key_values=cache, use_cache=True,
                             logits_to_keep=1)
            cache = output.past_key_values
    torch.cuda.synchronize()
    end = time.perf_counter()
    return future, {'prefix_ms': (prefix_end - start) * 1000,
                    'rollout_ms': (end - prefix_end) * 1000,
                    'total_ms': (end - start) * 1000}


def aggregate(rows, key):
    eligible = [r for r in rows if r['truth']]
    total = sum(len(r['truth']) for r in eligible)
    correct = sum(sum(a == b for a, b in zip(r[key], r['truth'])) for r in eligible)
    consecutive = []
    positions = []
    for r in eligible:
        count = 0
        for a, b in zip(r[key], r['truth']):
            if a != b:
                break
            count += 1
        consecutive.append(count)
    for i in range(8):
        sample = [r for r in eligible if i < len(r['truth'])]
        positions.append({'position': i + 1, 'n': len(sample),
                          'correct': sum(r[key][i] == r['truth'][i] for r in sample)})
    times = [t for r in eligible for t in r[key + '_times']]
    return {'predicted_positions': total, 'correct': correct,
            'agreement': correct / total, 'consecutive_prefix_mean': statistics.mean(consecutive),
            'positions': positions, 'warm_median_ms': {
                name: statistics.median(t[name] for t in times)
                for name in ('prefix_ms', 'rollout_ms', 'total_ms')}}


def main():
    parser = argparse.ArgumentParser()
    for name in ('target-model', 'draft-model', 'old-checkpoint', 'questions',
                 'eagle-source', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--alignment-only', action='store_true')
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise ValueError('Exactly one visible GPU is required')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.eagle_source))
    from eagle.model.cnets import Model
    from eagle.model.configs import EConfig
    metadata = json.loads((args.output_dir / 'checkpoint.json').read_text())
    source_commit = subprocess.check_output(['git', '-C', str(args.eagle_source),
        'rev-parse', 'HEAD'], text=True).strip()
    if source_commit != metadata['eagle_commit']:
        raise ValueError('Author inference source revision changed')
    checkpoint = args.output_dir / 'checkpoint'
    if digest(checkpoint / 'model.safetensors') != metadata['weights_sha256']:
        raise ValueError('EAGLE3 weight fingerprint changed')
    torch.manual_seed(20261001)
    target = AutoModelForCausalLM.from_pretrained(args.target_model, dtype=torch.float16,
        attn_implementation='sdpa', local_files_only=True).cuda().eval()
    config = EConfig.from_pretrained(checkpoint, local_files_only=True)
    # Transformers 5 normalizes an absent rope_scaling to a default dict;
    # the pinned author implementation expects None for unscaled RoPE.
    config.rope_scaling = metadata['config'].get('rope_scaling')
    config.rope_theta = metadata['config']['rope_theta']
    draft = Model(config, bias=False, top_k=1, depth=7, total_tokens=9).half()
    state = load_file(checkpoint / 'model.safetensors')
    loaded = draft.load_state_dict(state, strict=False)
    if loaded.missing_keys != ['embed_tokens.weight'] or loaded.unexpected_keys:
        raise ValueError(f'Unexpected checkpoint mismatch: {loaded}')
    # The public checkpoint omits embeddings and reuses target embeddings.
    draft.embed_tokens = target.model.embed_tokens
    draft = draft.cuda().eval()
    draft.reset()
    active = draft.t2d.nonzero().flatten()
    direct = draft.d2t
    offset = draft.d2t + torch.arange(config.draft_vocab_size, device='cuda:0')
    if torch.equal(offset, active):
        mapping, mapping_kind = offset, 'offset + draft_index'
    elif torch.equal(direct, active):
        mapping, mapping_kind = direct, 'absolute target IDs'
    else:
        raise ValueError('Reduced-vocabulary mapping disagrees with t2d')
    if mapping.numel() != config.draft_vocab_size:
        raise ValueError('Reduced vocabulary length mismatch')
    if any(not torch.isfinite(p).all() for p in draft.parameters()):
        raise ValueError('FP16 draft weights are not finite')
    del state
    items = [r for r in json.loads(args.questions.read_text(encoding='utf-8'))['items']
             if r['split'] == 'calibration']
    # Independent author API check: a width-one tree is one greedy chain.
    # Disable repetition penalties in both paths for this alignment check;
    # author's tree generator does not apply a repetition logits processor.
    ids, output, features, _ = target_prefix(target, items[0]['prompt_ids'])
    root = pick(output.logits[:, -1], ids, RepetitionPenaltyLogitsProcessor(1.0))
    expected, _ = eagle_future(draft, features, ids, root, mapping, 8,
                                RepetitionPenaltyLogitsProcessor(1.0))
    if mapping_kind != 'offset + draft_index':
        raise ValueError('Author API requires offset-format d2t mapping')
    draft.init_tree()
    draft.reset_kv()
    result = draft.topK_genrate(features, torch.cat((ids, root), dim=-1),
                               target.lm_head, None)
    reference = result[0][0].cpu().tolist()
    if reference != [root.item()] + expected:
        raise ValueError('Eight greedy predictions disagree with author width-one tree')
    save(args.output_dir / 'alignment.json', {'passed': True,
        'root_excluded': root.item(), 'manual_greedy': expected,
        'author_width_one_tree': reference, 'source_commit': source_commit,
        'probe_sha256': digest(Path(__file__)), 'repetition_penalty': 1.0})
    draft.reset()
    draft.reset_kv()
    del ids, output, features, root, result
    if args.alignment_only:
        print(json.dumps({'author_alignment_passed': True}), flush=True)
        return
    student = truncate_draft(AutoModelForCausalLM.from_pretrained(args.draft_model,
        dtype=torch.float32, attn_implementation='eager', local_files_only=True))
    old = torch.load(args.old_checkpoint, map_location='cpu', weights_only=True)
    student.load_state_dict(old['student'], strict=True)
    del old
    student = student.cuda().eval()
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({'loaded': True, 'visible_gpu': torch.cuda.get_device_name(),
        'mapping': mapping_kind, 'weight_sha256': metadata['weights_sha256']}), flush=True)
    penalty = RepetitionPenaltyLogitsProcessor(1.05)
    eos_value = target.generation_config.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for item in items:
        prompt = item['prompt_ids']
        with torch.inference_mode():
            ids = torch.tensor(prompt, device='cuda:0')[None]
            generated = target.generate(ids, do_sample=False, repetition_penalty=1.05,
                max_new_tokens=72, use_cache=True, pad_token_id=next(iter(eos)),
                logits_to_keep=1)[0, len(prompt):].cpu().tolist()
        for boundary in (0, 4, 32, 64):
            if boundary >= len(generated) or any(t in eos for t in generated[:boundary]):
                continue
            prefix = prompt + generated[:boundary]
            ids, output, features, target_time = target_prefix(target, prefix)
            root = pick(output.logits[:, -1], ids, penalty)
            if root.item() in eos:
                continue
            truth = target_future(target, ids, output, root, 8, penalty, eos)
            del output
            # One excluded warmup per arm/prefix. Alternate timed arm order.
            eagle_future(draft, features, ids, root, mapping, 8, penalty)
            old_future(student, ids, root, 8, penalty)
            row = {'id': item['id'], 'task': item['benchmark'], 'boundary': boundary,
                   'prefix_length': len(prefix), 'prefix_sha256': hashlib.sha256(
                       json.dumps(prefix).encode()).hexdigest(), 'root': root.item(),
                   'truth': truth, 'target_time': target_time,
                   'eagle_times': [], 'old_times': []}
            for repeat in range(args.repeats):
                order = ('eagle', 'old') if (len(rows) + repeat) % 2 == 0 else ('old', 'eagle')
                for name in order:
                    if name == 'eagle':
                        tokens, timing = eagle_future(draft, features, ids, root,
                                                      mapping, 8, penalty)
                    else:
                        tokens, timing = old_future(student, ids, root, 8, penalty)
                    if name in row and tokens != row[name]:
                        raise ValueError('Greedy predictions changed between timing repeats')
                    row[name] = tokens
                    row[name + '_times'].append(timing)
            rows.append(row)
            save(args.output_dir / 'rows.json', rows)
            print(json.dumps({'id': item['id'], 'boundary': boundary,
                'truth': truth, 'eagle': row['eagle'], 'old': row['old']}), flush=True)
            del ids, features, root
    # Same synthetic Case 40 length fixture as earlier latency probes. It is
    # a serving-shape timing observation, not an output-quality question.
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    long_prefix = tokenizer.encode('Case 40. ' + 'EEFTRITON ' * 1100,
                                    add_special_tokens=False)[:2155]
    if len(long_prefix) != 2155:
        raise ValueError('Latency fixture is too short')
    ids, output, features, _ = target_prefix(target, long_prefix)
    del ids, output, features
    ids, output, features, preparation_time = target_prefix(target, long_prefix)
    root = pick(output.logits[:, -1], ids, penalty)
    del output
    eagle_future(draft, features, ids, root, mapping, 8, penalty)
    old_future(student, ids, root, 8, penalty)
    long_times = {'eagle': [], 'old': []}
    for repeat in range(5):
        for name in (('eagle', 'old') if repeat % 2 == 0 else ('old', 'eagle')):
            if name == 'eagle':
                _, timing = eagle_future(draft, features, ids, root, mapping, 8, penalty)
            else:
                _, timing = old_future(student, ids, root, 8, penalty)
            long_times[name].append(timing)
    long_latency = {'prompt_tokens': 2155, 'known_root_tokens': 1, 'predicted_tokens': 8,
        'fixture': 'Synthetic Case 40 / repeated EEFTRITON; same as previous latency probe',
        'target_preparation': preparation_time, 'samples': long_times,
        'warm_median_ms': {name: {part: statistics.median(t[part] for t in samples)
            for part in ('prefix_ms', 'rollout_ms', 'total_ms')}
            for name, samples in long_times.items()}}
    save(args.output_dir / 'latency-2155.json', long_latency)
    del ids, features, root
    report = {'schema': 'pvd.eagle3_pair.token_probe.v1', 'questions': len(items),
        'prefixes': len(rows), 'device_count': torch.cuda.device_count(),
        'gpu': torch.cuda.get_device_name(), 'cuda_visible_devices': __import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        'eagle': aggregate(rows, 'eagle'), 'old': aggregate(rows, 'old'),
        'by_task': {task: {name: aggregate([r for r in rows if r['task'] == task], name)
            for name in ('eagle', 'old')} for task in ('gsm8k', 'hotpotqa')},
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
        'shared_repetition_penalty': 1.05, 'known_root_counted': False,
        'eos': sorted(eos), 'auxiliary_post_block_ids': [1, 13, 24],
        'hf_hidden_state_slots': [2, 14, 25], 'mapping': mapping_kind,
        'eagle_source_commit': source_commit, 'torch': torch.__version__,
        'probe_sha256': digest(Path(__file__)),
        'latency_2155': long_latency,
        'questions_sha256': digest(args.questions),
        'old_checkpoint_sha256': digest(args.old_checkpoint), 'checkpoint': metadata,
        'notes': ['Target full-attention features supplied offline; serving sparse-D drift is untested.',
                  'Author EAGLE eager inference unchanged; no native serving kernel timing.',
                  'Old six-layer token trunk FP32 matches deployed trunk; Q readout excluded.',
                  'Neither arm performs token verification or emits target Q.',
                  'Only genuine predicted positions through the first target EOS are scored.']}
    save(args.output_dir / 'report.json', report)
    print(json.dumps({'completed': True, 'eagle': report['eagle'], 'old': report['old'],
                      'peak_allocated_mib': report['peak_allocated_mib']}), flush=True)


if __name__ == '__main__':
    main()
