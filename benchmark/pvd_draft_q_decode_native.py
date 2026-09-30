"""Compare baseline/new cached Draft-Q through serving-shaped native CAGRA."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch


def export(args):
    from transformers import AutoModelForCausalLM
    from pvd_draft_q_multitask_probe import truncate_draft
    from pvd_draft_q_readout_probe import TargetQueryReadout
    from pvd_draft_q_trained_latency_probe import cached_forward
    captures = torch.load(args.captures, map_location='cpu', weights_only=False)
    selected = {}
    for row in captures['records']:
        if row['split'] != 'calibration':
            continue
        # Match the first actual refresh where available; very short answers
        # have only a boundary-zero offline fixture, explicitly recorded.
        if row['id'] not in selected or row['boundary'] == 4:
            selected[row['id']] = row
    fixtures = [{'id': r['id'], 'boundary': r['boundary'], 'prefix': r['prefix'],
                 'prompt_k': r['prompt_k'], 'queries': {'reference': r['branches']['target']['post_q'][2]}}
                for r in selected.values()]
    device = torch.device('cuda:0')
    student = truncate_draft(AutoModelForCausalLM.from_pretrained(args.draft_model,
                            dtype=torch.float32, attn_implementation='eager',
                            local_files_only=True).to(device)).eval()
    readout = TargetQueryReadout(896, 28, 28 * 128, 896, fusion='learned').to(device).eval()
    hashes = {}
    for name, path in [('baseline', args.baseline), ('joint', args.joint)]:
        state = torch.load(path, map_location='cpu', weights_only=True)
        student.load_state_dict(state['student']); readout.load_state_dict(state['readout'])
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for fixture in fixtures:
            prediction = cached_forward(student, readout, fixture['prefix'], 8, device)
            fixture['queries'][name] = prediction['query'][2].cpu()
            fixture.setdefault('predicted_tokens', {})[name] = prediction['future']
        del state
    torch.save({'dataset_sha256': captures['dataset_sha256'], 'checkpoint_sha256': hashes,
                'fixtures': fixtures}, args.output)
    print(json.dumps({'exported': len(fixtures), 'checkpoint_sha256': hashes}), flush=True)


def native(args):
    from sglang.srt.disaggregation.pvd.cagra_backend import CagraIndexBackend
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    saved = torch.load(args.fixture, map_location='cpu', weights_only=False)
    backend = CagraIndexBackend(device='cuda:0', native_bytes_per_index=536870912,
                               global_native_cap_bytes=671088640, graph_degree=16,
                               intermediate_degree=16, itopk_size=2048, exact_head_groups=4)
    records = []
    for fixture in saved['fixtures']:
        keys = fixture['prompt_k'].float().to(device)
        n = keys.shape[1]
        queries = {a: q.float().to(device) for a, q in fixture['queries'].items()}
        values = {a: [] for a in queries}
        unions = {a: [] for a in queries}
        timings = {a: [] for a in queries}
        builds = []
        # Each actual V shard owns two contiguous KV heads. Four-way merging
        # groups those heads from two adjacent layers, matching the manager.
        for rank in (0, 1):
            for base_layer in range(0, 28, 2):
                group = [(base_layer + offset // 2, rank * 2 + offset % 2) for offset in range(4)]
                data = torch.cat([keys[l, :, h] - keys[l, :, h].mean(0) for l, h in group]).contiguous()
                start = time.perf_counter()
                index = backend.build(data, vector_space='qwen25-7b-pvd', metric='ip')
                builds.append(time.perf_counter() - start)
                try:
                    for offset, (layer, kv_head) in enumerate(group):
                        words = [0] * ((4 * n + 31) // 32)
                        for location in range(offset * n, (offset + 1) * n):
                            words[location // 32] |= 1 << (location % 32)
                        bitset = torch.tensor(words, dtype=torch.uint32, device=device)
                        first = kv_head * 7
                        expected = (queries['reference'][layer, first:first + 7]
                                    @ keys[layer, :, kv_head].T).topk(10, -1).indices.cpu().tolist()
                        for arm, query in queries.items():
                            start = time.perf_counter()
                            ids, _ = backend.search(index, query[layer, first:first + 7].contiguous(),
                                                    top_k=16, bitset=bitset)
                            timings[arm].append(time.perf_counter() - start)
                            if not ((ids >= offset * n) & (ids < (offset + 1) * n)).all():
                                raise ValueError('native filtered search returned a foreign head ID')
                            found = (ids.to(torch.int64) - offset * n).cpu().tolist()
                            unions[arm].append(len(set(x for row in found for x in row)))
                            values[arm].extend({'layer': layer, 'head': first + head,
                                               'top10_recall': len(set(row[:10]) & set(gold)) / 10,
                                               'top4_coverage_k16': len(set(row) & set(gold[:4])) / 4}
                                              for head, (row, gold) in enumerate(zip(found, expected)))
                finally:
                    backend.dispose(index)
        record = {'id': fixture['id'], 'boundary': fixture['boundary'], 'prompt_tokens': n,
                  'build_seconds': sum(builds), 'arms': {a: {
                      'mean_top10_recall': statistics.mean(x['top10_recall'] for x in values[a]),
                      'mean_top4_coverage_k16': statistics.mean(x['top4_coverage_k16'] for x in values[a]),
                      'worst_layer_head_top10_recall': min(x['top10_recall'] for x in values[a]),
                      'max_group_union_tokens': max(unions[a]),
                      'estimated_payload_mib': sum(unions[a]) * 512 / 2**20,
                      'search_median_ms_per_head_batch': statistics.median(timings[a]) * 1000,
                      'per_head': values[a]} for a in queries}}
        if any(x['max_group_union_tokens'] > 128 for x in record['arms'].values()):
            raise ValueError('native result exceeded serving union cap')
        records.append(record)
        print(json.dumps({'native_completed': fixture['id'], 'coverage': {
            a: x['mean_top4_coverage_k16'] for a, x in record['arms'].items()}}), flush=True)
    if backend.runtime.global_allocated_bytes() != 0:
        raise ValueError('native graph owners remain allocated')
    payload = {'checkpoint_sha256': saved['checkpoint_sha256'], 'dataset_sha256': saved['dataset_sha256'],
               'fixture_sha256': hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
               'records': records, 'summary': {a: {
                   'mean_top10_recall': statistics.mean(r['arms'][a]['mean_top10_recall'] for r in records),
                   'mean_top4_coverage_k16': statistics.mean(r['arms'][a]['mean_top4_coverage_k16'] for r in records)}
                   for a in ('reference', 'baseline', 'joint')},
               'note': 'Same serving-shaped four-head exact-degree16 graph and itopk2048 for all queries; both rank layouts sequentially on one idle V100S, not concurrent serving latency.'}
    args.output.write_bytes((json.dumps(payload, indent=2) + '\n').encode())
    print(json.dumps(payload['summary']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    e = commands.add_parser('export')
    for name in ('captures', 'draft-model', 'baseline', 'joint', 'output'):
        e.add_argument('--' + name, type=Path, required=True)
    n = commands.add_parser('native')
    for name in ('fixture', 'output'):
        n.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    export(args) if args.command == 'export' else native(args)


if __name__ == '__main__':
    main()
