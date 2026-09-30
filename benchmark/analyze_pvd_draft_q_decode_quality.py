"""Compare old/new six-layer checkpoints on the same frozen output requests."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from pvd_output_quality import answer_scores, extract_qa_final_variant, numeric


def score(row, variant=False):
    if row['benchmark'] == 'gsm8k':
        answer = numeric(row['gsm_numeric_answer'])
        return float(answer is not None and answer == numeric(row['gold_answer'])), None
    answer = extract_qa_final_variant(row['text']) if variant else row['final_answer']
    return answer_scores(answer or '', row['gold_answer'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    names = {'old_joint': 'old-joint.jsonl', 'new_joint': 'joint.jsonl'}
    arms = {arm: [json.loads(line) for line in (args.folder / filename).read_text(
        encoding='utf-8').splitlines()] for arm, filename in names.items()}
    expected_hash = hashlib.sha256((args.folder / 'dataset.json').read_bytes()).hexdigest()
    for arm, rows in arms.items():
        if len(rows) != 40 or len({r['id'] for r in rows}) != 40:
            raise ValueError(f'incomplete or duplicated {arm}')
        if any(r['arm'] != 'joint' or r['dataset_sha256'] != expected_hash
               or not r.get('completed_entry_released')
               or r['finish_reason'].get('type') not in ('stop', 'length')
               or not 0 < r['completion_tokens'] <= r['max_new_tokens'] for r in rows):
            raise ValueError(f'invalid request or cleanup in {arm}')
    for old, new in zip(arms['old_joint'], arms['new_joint']):
        if any(old[k] != new[k] for k in ('id', 'benchmark', 'dataset_sha256',
               'prompt_sha256', 'prompt_tokens', 'max_new_tokens', 'gold_answer')):
            raise ValueError('unmatched questions, order or generation bounds')
    metrics, paired = {}, {}
    for task in ('gsm8k', 'hotpotqa'):
        subset = {arm: [r for r in rows if r['benchmark'] == task]
                  for arm, rows in arms.items()}
        metrics[task], paired[task] = {}, {}
        for arm, rows in subset.items():
            values = [score(r) for r in rows]
            variants = [score(r, True) for r in rows]
            metrics[task][arm] = {
                'n': len(rows), 'correct': sum(v[0] for v in values),
                'answer_em': statistics.mean(v[0] for v in values),
                'answer_f1': statistics.mean(v[1] for v in values) if task == 'hotpotqa' else None,
                'variant_correct': sum(v[0] for v in variants),
                'variant_em': statistics.mean(v[0] for v in variants),
                'variant_f1': statistics.mean(v[1] for v in variants) if task == 'hotpotqa' else None,
                'missing_final': sum(r['final_answer'] is None for r in rows),
                'truncated_ids': [r['id'] for r in rows if r['truncated']],
                'completion_tokens': sum(r['completion_tokens'] for r in rows),
                'median_client_seconds': statistics.median(r['wall_seconds'] for r in rows)}
        for variant in (False, True):
            changes = {'gains': [], 'losses': [], 'both_correct': [], 'both_wrong': []}
            for old, new in zip(subset['old_joint'], subset['new_joint']):
                a, b = score(old, variant)[0], score(new, variant)[0]
                category = ('both_correct' if a and b else 'both_wrong' if not a and not b
                            else 'gains' if b else 'losses')
                changes[category].append(old['id'])
            paired[task]['variant' if variant else 'primary'] = changes
    payload = {'dataset_sha256': expected_hash, 'metrics': metrics, 'paired_changes': paired,
               'source_sha256': {filename: hashlib.sha256((args.folder / filename).read_bytes()).hexdigest()
                                 for filename in names.values()},
               'note': 'Fixed 40-question subset; references are from the earlier run. Primary metrics and the already declared format sensitivity rule are unchanged. These are output quality scores, not repeated-request latency estimates.'}
    args.output.write_bytes((json.dumps(payload, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
