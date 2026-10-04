"""Replay actual bank installation on saved real-model KV, without GPU timing.

Reconstruct the monotonic CPU cache only from captured selected rows. Each
previous installed bank provides resident hits; cached but nonresident tokens
still contribute to the unchanged KV H2D byte count. No unused final prefetch.
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
sys.modules['sglang'] = package  # frontend bootstrap only

import torch
from sglang.srt.disaggregation.pvd.oasis_bank_install import install_batched_bank, install_tensor_bound
from sglang.srt.disaggregation.pvd.oasis_qwen import PromptBank


def per_head(chosen, resident, cache):
    """CPU execution of baseline resident gather + nonresident cache stack."""
    width = max(map(len, chosen))
    k = torch.zeros((4, width, 128), dtype=torch.float16)
    v, valid = torch.zeros_like(k), torch.zeros((4, width), dtype=torch.bool)
    stats = dict(selected_rows=0, resident_rows=0, cpu_rows=0, kv_h2d_bytes=0,
        kv_h2d_calls=0, resident_gather_calls=0, resident_scatter_calls=0, cpu_scatter_calls=0)
    for head, ids in enumerate(chosen):
        old = {} if resident is None else {t: i for i, t in enumerate(resident.ids[head])}
        hits = [(i, old[t]) for i, t in enumerate(ids) if t in old]
        misses = [(i, t) for i, t in enumerate(ids) if t not in old]
        if hits:
            dst, src = map(list, zip(*hits))
            k[head, dst], v[head, dst] = resident.keys[head, src], resident.values[head, src]
        if misses:
            pairs = torch.stack([cache[head][t] for _, t in misses])
            dst = [i for i, _ in misses]
            k[head, dst], v[head, dst] = pairs[:, 0], pairs[:, 1]
        valid[head, :len(ids)] = True
        stats['selected_rows'] += len(ids)
        stats['resident_rows'] += len(hits)
        stats['cpu_rows'] += len(misses)
        stats['kv_h2d_bytes'] += len(misses) * 512
        stats['kv_h2d_calls'] += int(bool(misses))
        stats['resident_gather_calls'] += 2 * int(bool(hits))
        stats['resident_scatter_calls'] += 2 * int(bool(hits))
        stats['cpu_scatter_calls'] += 2 * int(bool(misses))
    return k, v, valid, stats


def verify_case(trajectory, source):
    if (trajectory['schema'] != 'pvd-oasis-ready-replay-v1'
            or len(trajectory['prompt_ids']) != 2159 or len(trajectory['steps']) != 15):
        raise ValueError('qualified captured real trajectory required')
    caches = [[{} for _ in range(4)] for _ in range(28)]
    residents = [None] * 28
    totals = {phase: {mode: dict(bank_checks=0, selected_rows=0, resident_rows=0, cpu_rows=0,
        kv_h2d_bytes=0, kv_h2d_calls=0, resident_gather_calls=0, resident_scatter_calls=0,
        cpu_scatter_calls=0) for mode in ('baseline', 'batched')}
        for phase in ('bootstrap', 'steady')}
    remote = dict(rows=0, payload_bytes=0, deliveries=0)
    for step, row in enumerate(trajectory['steps']):
        if row['step'] != step or len(row['banks']) != 28:
            raise ValueError('ordered complete captured banks required')
        phase = 'bootstrap' if step == 0 else 'steady'
        for layer, bank in enumerate(row['banks']):
            chosen = tuple(tuple(ids) for ids in bank['ids'])
            ranks = set()
            for head, ids in enumerate(chosen):
                if not ids or len(ids) > 32 or len(set(ids)) != len(ids):
                    raise ValueError('bounded nonempty unique banks required')
                for i, token in enumerate(ids):
                    value = torch.stack((bank['keys'][head, i], bank['values'][head, i]))
                    if token in caches[layer][head]:
                        if not torch.equal(value, caches[layer][head][token]):
                            raise AssertionError('immutable captured Prompt row changed')
                    else:
                        caches[layer][head][token] = value
                        ranks.add(head // 2)
                        remote['rows'] += 1
                        remote['payload_bytes'] += 512
            remote['deliveries'] += len(ranks)
            baseline = per_head(chosen, residents[layer], caches[layer])
            retained = []
            batched = install_batched_bank(chosen, residents[layer], caches[layer], device='cpu',
                capacity=32, prompt_tokens=2159, retained=retained)
            for mode, actual in (('baseline', baseline), ('batched', batched)):
                for tensor, name in zip(actual[:3], ('keys', 'values', 'valid')):
                    if not torch.equal(tensor, bank[name]):
                        raise AssertionError(f'{mode} differs from captured {name}')
                sums = totals[phase][mode]
                sums['bank_checks'] += 1
                for name in set(sums) - {'bank_checks'}:
                    sums[name] += actual[3][name]
            for name in ('selected_rows', 'resident_rows', 'cpu_rows', 'kv_h2d_bytes'):
                if baseline[3][name] != batched[3][name]:
                    raise AssertionError('changed selected/resident/local-byte budget')
            residents[layer] = PromptBank(chosen, *batched[:3])
    return dict(case=trajectory['case'], source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        phases=totals, remote=remote, bank_bits_order_and_mask_exact=True,
        per_job_tensor_bound_bytes=install_tensor_bound(32),
        two_worker_tensor_bound_bytes=2 * install_tensor_bound(32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture, output = args.capture.resolve(), args.output.resolve()
    if not capture.is_relative_to(ROOT / 'artifacts') or not output.is_relative_to(ROOT / 'artifacts') or output.exists():
        raise ValueError('project-local captures and fresh artifacts output required')
    result = dict(schema='pvd-batched-real-bank-equivalence-v1', cases=[], cpu_only=True,
        cuda_native_online_tested=False, target_forward_run=False, latency_measured=False,
        call_counts_scope='planned Torch-level KV gather/scatter/upload calls, not CUDA kernel counts',
        terminal_unused_prefetch_included=False)
    for case in (99401, 99402):
        source = capture / str(case) / 'trajectory.pt'
        result['cases'].append(verify_case(torch.load(source, map_location='cpu', weights_only=True), source))
    result['source_hashes'] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for p in (Path(__file__), ROOT / 'python/sglang/srt/disaggregation/pvd/oasis_bank_install.py')}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
