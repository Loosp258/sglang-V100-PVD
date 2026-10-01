"""Validate the actual experiment seed serializer on saved P fixtures."""
import argparse
import io
import json
from pathlib import Path
import sys
import threading

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'benchmark'))
from pvd_oasis_experiment import VectorExperiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    fixture = torch.load(args.fixture, weights_only=True)
    # Seed serialization only requires the immutable fixture, not a GPU index.
    store = VectorExperiment.__new__(VectorExperiment)
    store.lock = threading.Lock()
    store.fixture, store.identity = fixture, 'immutable-fixture-test'
    sizes = []
    for teacher in (False, True):
        wire = store.seed({'fixture_sha256': store.identity, 'request_id': 'r',
            'incarnation': 'g', 'teacher_trajectory': teacher})
        restored = torch.load(io.BytesIO(wire), weights_only=True)
        q = restored['seed_q']
        assert torch.equal(q, fixture['teacher_q'][0])
        assert q.untyped_storage().nbytes() == q.numel() * q.element_size()
        assert restored['root'] == fixture['root']
        assert 'keys' not in restored and 'values' not in restored
        assert restored['teacher_tokens'] == (fixture['teacher_tokens'] if teacher else None)
        raw_bytes = sum(restored[k].numel() * restored[k].element_size() for k in ('features', 'seed_q'))
        assert len(wire) < raw_bytes + 65536, 'hidden backing storage leaked into seed'
        sizes.append({'teacher_trajectory': teacher, 'wire_bytes': len(wire), 'tensor_bytes': raw_bytes})
    print(json.dumps({'fixture': args.fixture.name, 'passed': True,
        'future_q_storage_excluded': True, 'free_future_tokens_excluded': True, 'sizes': sizes}))


if __name__ == '__main__':
    main()
