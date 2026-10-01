"""Repeat identical native queries through the live experimental V endpoint."""
import argparse
import hashlib
import io
import json
import uuid

import requests
import torch

from pvd_oasis_experiment import pack, unpack, post_json, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v-url', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=8)
    args = parser.parse_args()
    session = requests.Session()
    activation = post_json(session, args.v_url, '/activate', {'fixture': 'case40-2155.pt'})
    rows = []
    query_sha = {}
    for repeat in range(args.repeats):
        ticket = {'request_id': str(uuid.uuid4()), 'incarnation': str(uuid.uuid4()), 'step': 0}
        response = session.post(args.v_url + '/seed', data=pack({**ticket,
            'fixture_sha256': activation['fixture_sha256'], 'selection_mode': 'live'}), timeout=60)
        response.raise_for_status()
        seed = torch.load(io.BytesIO(response.content), weights_only=True)
        sample = []
        for layer in (0, 7, 14):
            array = seed['seed_q'][layer].numpy()
            current_sha = hashlib.sha256(array.tobytes()).hexdigest()
            if layer in query_sha and query_sha[layer] != current_sha:
                raise ValueError('diagnostic query changed')
            query_sha[layer] = current_sha
            response = session.post(args.v_url + '/layer', data=pack({
                'ticket': {**ticket, 'layer': layer}, 'resident': [[], [], [], []],
                'cached': [[], [], [], []], 'capacity': 128, 'max_new': 128, 'top_k': 16}, array), timeout=60)
            response.raise_for_status()
            meta, _ = unpack(response.content)
            sample.append({'layer': layer, 'selected': meta['selected'],
                'candidates': meta['candidate_ids']})
        rows.append(sample)
    changes = []
    for index, sample in enumerate(rows[1:], 1):
        for original, current in zip(rows[0], sample):
            for head, (a, b) in enumerate(zip(original['selected'], current['selected'])):
                if a != b:
                    changes.append({'repeat': index, 'layer': current['layer'], 'kv_head': head,
                        'set_symmetric_difference': len(set(a) ^ set(b)),
                        'only_first': sorted(set(a) - set(b)), 'only_current': sorted(set(b) - set(a)),
                        'order_only': set(a) == set(b)})
    save_json(args.output, {'fixture_sha256': activation['fixture_sha256'],
        'repeats': args.repeats, 'identical_query_sha256': query_sha,
        'changed_head_selections': len(changes), 'changes': changes, 'raw': rows})
    print(json.dumps({'changed_head_selections': len(changes), 'changes': changes[:10]}), flush=True)


if __name__ == '__main__':
    main()
