"""Freeze a training-only subset and collect exact token IDs on sparse D."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from pvd_output_quality import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'collect'))
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text(encoding='utf-8'))
    if args.command == 'prepare':
        if args.dataset.exists():
            raise FileExistsError(args.dataset)
        items = []
        for task, count in (('gsm8k', 8), ('hotpotqa', 4)):
            selected = [r for r in questions['items'] if r['split'] == 'train'
                        and r['benchmark'] == task][:count]
            if len(selected) != count:
                raise ValueError('not enough training-only requests')
            items.extend({**r, 'prompt_tokens': len(r['prompt_ids']),
                          'prompt_sha256': hashlib.sha256(r['prompt'].encode()).hexdigest(),
                          'gold_answer': '', 'max_new_tokens': 384 if task == 'gsm8k' else 128}
                         for r in selected)
        payload = {'schema': 'pvd.draft_q.observed_prefix_dataset.v1',
                   'source_questions_sha256': hashlib.sha256(args.questions.read_bytes()).hexdigest(),
                   'items': items, 'purpose': 'Training-only sparse-D token trajectories; no answer scoring.'}
        args.dataset.write_bytes((json.dumps(payload, indent=2) + '\n').encode())
        print(json.dumps({'prepared_training_requests': len(items),
                          'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest()}))
    else:
        if args.output is None:
            raise ValueError('--output required')
        dataset = json.loads(args.dataset.read_text(encoding='utf-8'))
        train_ids = {r['id'] for r in questions['items'] if r['split'] == 'train'}
        if len(dataset['items']) != 12 or any(r['id'] not in train_ids for r in dataset['items']):
            raise ValueError('observed collection includes non-training requests')
        collect(SimpleNamespace(dataset=args.dataset, output=args.output, resume=False,
            arm='joint', url='http://10.10.1.2:8001', vector_host='http://10.10.1.2',
            coordinator='http://10.10.1.2:9100', cleanup_wait_seconds=120,
            retain_output_ids=True))


if __name__ == '__main__':
    main()
