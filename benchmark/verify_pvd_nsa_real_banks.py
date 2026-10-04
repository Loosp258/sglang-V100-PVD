"""CPU byte/bank equivalence on saved real-model sparse KV trajectories.

Reconstruct original interleaved storage ONLY for captured known Prompt rows;
uncaptured rows are placeholders and are never selected. Exercise actual pack,
wire manifests and monotonic CPU-cache installation, preserving bank order.
This is not CUDA/native qualification, a target-model forward, or latency data.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
package = types.ModuleType('sglang')
package.__path__ = [str(ROOT / 'python/sglang')]
sys.modules['sglang'] = package  # package bootstrap only; all PVD/Torch code is real

import torch

from sglang.srt.disaggregation.pvd.kv_packer import PVD_TENSOR_LAYOUT
from sglang.srt.disaggregation.pvd.protocol import KVLayoutSignature, KVShardManifest
from sglang.srt.disaggregation.pvd.sparse_copy import copy_sparse_kv_into
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
from sglang.srt.disaggregation.pvd.sparse_token_runs import consecutive_token_runs


def verify_case(trajectory, source):
    if (trajectory['schema'] != 'pvd-oasis-ready-replay-v1'
            or len(trajectory['prompt_ids']) != 2159 or len(trajectory['steps']) != 15):
        raise ValueError('qualified real trajectory required')
    count, pages, rows, layers, dim, heads = 2159, 135, 2160, 28, 128, 2
    layout = KVLayoutSignature(model_id='Qwen2.5-7B-captured-KV', model_revision='known-rows-only',
        kv_dtype='torch.float16', page_size=16, num_layers=layers, total_kv_heads=4,
        kv_heads_per_rank=heads, head_dim=dim, tp_size=2, pp_size=1,
        tensor_layout=PVD_TENSOR_LAYOUT, extra=dict(component_count=2 * layers,
            component_dtypes=['torch.float16'] * (2 * layers),
            component_token_shapes=[[heads, dim]] * (2 * layers),
            component_bytes_per_token=[heads * dim * 2] * (2 * layers)))
    components = [torch.zeros((2 * layers, rows, heads, dim), dtype=torch.float16) for _ in (0, 1)]
    shards = [KVShardManifest(rank=rank, rail='cpu-no-rdma', expected_bytes=components[rank].numel() * 2,
        page_count=pages, last_page_valid_tokens=count - (pages - 1) * 16,
        layer_start=0, layer_end=layers) for rank in (0, 1)]
    known = [[set() for _ in range(4)] for _ in range(layers)]
    # Independent source construction from real captured tensors; additionally
    # prove that every repeatedly captured Prompt row is immutable.
    for row in trajectory['steps']:
        for layer, bank in enumerate(row['banks']):
            for head, ids in enumerate(bank['ids']):
                if not ids or len(ids) > 32 or len(set(ids)) != len(ids) or any(not 0 <= t < count for t in ids):
                    raise ValueError('bounded unique captured IDs required')
                rank, local = divmod(head, heads)
                fresh = [(i, token) for i, token in enumerate(ids) if token not in known[layer][head]]
                if fresh:
                    indexes, tokens = map(list, zip(*fresh))
                    components[rank][layer, tokens, local] = bank['keys'][head, indexes]
                    components[rank][layer + layers, tokens, local] = bank['values'][head, indexes]
                    known[layer][head].update(tokens)
                if not (torch.equal(components[rank][layer, list(ids), local], bank['keys'][head, :len(ids)])
                        and torch.equal(components[rank][layer + layers, list(ids), local], bank['values'][head, :len(ids)])):
                    raise AssertionError('captured immutable Prompt KV changed across steps')
    packed = [part.view(torch.uint8).reshape(-1) for part in components]
    observations = {}
    for mode in ('baseline', 'contiguous'):
        cache = [[{} for _ in range(4)] for _ in range(layers)]
        totals = dict(bank_checks=0, received_token_checks=0, deliveries=0, payload_bytes=0, packing_calls=0)
        for step, row in enumerate(trajectory['steps']):
            for layer, bank in enumerate(row['banks']):
                selected = bank['ids']
                for rank in (0, 1):
                    specs = []
                    for head in range(rank * heads, (rank + 1) * heads):
                        missing = tuple(t for t in selected[head] if t not in cache[layer][head])
                        if not missing:
                            continue
                        if mode == 'contiguous':
                            missing = tuple(sorted(missing))
                        specs.append(SparseKVSpec('real-replay', 'inc', f'{step}:{layer}', step + 1,
                            'entry', 'index', 'map', layout.fingerprint, layer, head, missing))
                    if not specs:
                        continue
                    manifest = SparseDeliveryManifest(tuple(specs), 'torch.float16', dim)
                    destination = torch.empty(manifest.nbytes, dtype=torch.uint8)
                    copy_sparse_kv_into(packed[rank], destination, manifest=manifest, layout=layout,
                        shard=shards[rank], entry_transfer_id='entry', index_version='index',
                        id_mapping_version='map', contiguous_runs=mode == 'contiguous')
                    for payload in manifest.payload_views(destination):
                        head = payload.spec.kv_head
                        original_indexes = {token: i for i, token in enumerate(selected[head])}
                        for i, token in enumerate(payload.spec.token_ids):
                            original_index = original_indexes[token]
                            expected = torch.stack((bank['keys'][head, original_index], bank['values'][head, original_index]))
                            if not torch.equal(payload.tensor[:, i], expected):
                                raise AssertionError('packed real KV token differs from independent captured oracle')
                            if token in cache[layer][head]:
                                raise AssertionError('monotonic cache received a duplicate token')
                            cache[layer][head][token] = payload.tensor[:, i].clone()
                            totals['received_token_checks'] += 1
                        totals['packing_calls'] += 2 * (sum(1 for _ in consecutive_token_runs(payload.spec.token_ids))
                            if mode == 'contiguous' else len(payload.spec.token_ids))
                        payload.close()
                    totals['deliveries'] += 1
                    totals['payload_bytes'] += manifest.nbytes
                keys, values, valid = torch.zeros_like(bank['keys']), torch.zeros_like(bank['values']), torch.zeros_like(bank['valid'])
                for head, ids in enumerate(selected):
                    pairs = torch.stack([cache[layer][head][token] for token in ids])
                    keys[head, :len(ids)], values[head, :len(ids)] = pairs[:, 0], pairs[:, 1]
                    valid[head, :len(ids)] = True
                if not (torch.equal(keys, bank['keys']) and torch.equal(values, bank['values'])
                        and torch.equal(valid, bank['valid'])):
                    raise AssertionError('rebuilt real bank changed order, KV bits or mask')
                totals['bank_checks'] += 1
        observations[mode] = totals
    for name in ('bank_checks', 'received_token_checks', 'deliveries', 'payload_bytes'):
        if observations['baseline'][name] != observations['contiguous'][name]:
            raise AssertionError('changed real candidate/byte/delivery budget: ' + name)
    return dict(case=trajectory['case'], source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                observations=observations, real_bank_bits_and_order_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture, output = args.capture.resolve(), args.output.resolve()
    if not capture.is_relative_to(ROOT / 'artifacts') or not output.is_relative_to(ROOT / 'artifacts') or output.exists():
        raise ValueError('project-local captures and fresh artifacts output required')
    result = dict(schema='pvd-nsa-real-bank-equivalence-v1', cases=[], status='passed',
                  cpu_only=True, cuda_native_online_tested=False, target_forward_run=False,
                  original_storage_reconstructed_from_captured_known_rows_only=True,
                  latency_measured=False)
    for case in (99401, 99402):
        source = capture / str(case) / 'trajectory.pt'
        result['cases'].append(verify_case(torch.load(source, map_location='cpu', weights_only=True), source))
    result['source_hashes'] = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                             for path in [Path(__file__), ROOT / 'python/sglang/srt/disaggregation/pvd/sparse_copy.py',
                                          ROOT / 'python/sglang/srt/disaggregation/pvd/sparse_token_runs.py']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
