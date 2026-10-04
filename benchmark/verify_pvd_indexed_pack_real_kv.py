"""Actual indexed packing CPU ABBA including index admission/preparation/release.

Both arms use identical selected views and full validation. Uncaptured CPU
source rows are poison, not a real Entry. No CUDA/native/network/Decode claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import verify_pvd_selected_views_real_kv as replay
import torch
from sglang.srt.disaggregation.pvd.sparse_copy import copy_sparse_kv_into
from sglang.srt.disaggregation.pvd.sparse_row_index import SparseRowIndexWorkspace
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

ROOT = Path(__file__).resolve().parents[1]


def pack(job, indexed, budget):
    owner = None
    try:
        if indexed and any(len(s.token_ids) > 1 for s in job['manifest'].specs):
            owner = SparseRowIndexWorkspace(job['manifest'], layout=job['layout'], shard=job['shard'],
                device='cpu', budget=budget, owner='cpu-only-indexed-pack')
        metrics = {}
        count = copy_sparse_kv_into(job['source'], job['target'], manifest=job['manifest'],
            layout=job['layout'], shard=job['shard'], entry_transfer_id='entry', index_version='index',
            id_mapping_version='map', selected_component_views=True, indexed_workspace=owner, copy_metrics=metrics)
        return count, metrics, owner.bytes if owner is not None else 0
    finally:
        if owner is not None: owner.release_after_fence()  # actual synchronous CPU, never GPU proof


def verify_case(path, rounds):
    sources, known, jobs = replay.prepare(torch.load(path, map_location='cpu', weights_only=True))
    before = [replay.byte_digest(s) for s in sources]
    budget = TransferBudget(1 << 20, 8)
    observations = {}
    for mode in ('baseline', 'indexed'):
        phases = {}
        for phase, items in jobs.items():
            payload = hashlib.sha256()
            manifests = hashlib.sha256()
            totals = dict(source_component_views=0, row_copy_calls=0, index_select_calls=0, row_index_bytes=0)
            rows = peak = 0
            for job in items:
                views, metrics, bound = pack(job, mode == 'indexed', budget)
                assert torch.equal(job['target'], job['oracle'])
                assert budget.snapshot()['used_staging_bytes'] == 0
                counts = [len(s.token_ids) for s in job['manifest'].specs]
                assert metrics['row_copy_calls'] == 2 * sum(n for n in counts if mode == 'baseline' or n == 1)
                assert metrics['index_select_calls'] == 2 * sum(n > 1 for n in counts) * (mode == 'indexed')
                assert metrics['row_index_bytes'] == 8 * sum(n for n in counts if n > 1) * (mode == 'indexed')
                totals['source_component_views'] += views
                for key in metrics: totals[key] += metrics[key]
                peak = max(peak, bound)
                rows += sum(counts)
                payload.update(job['target'].numpy().tobytes())
                manifests.update(job['manifest'].fingerprint.encode())
            phases[phase] = dict(deliveries=len(items), rows=rows, bytes=rows * 512,
                wire_sha256=payload.hexdigest(), manifest_sha256=manifests.hexdigest(),
                peak_index_budget_bytes=peak, **totals)
        observations[mode] = phases
    for phase in jobs:
        a,b = (observations[mode][phase] for mode in ('baseline','indexed'))
        for key in ('deliveries','rows','bytes','wire_sha256','manifest_sha256','source_component_views'):
            assert a[key] == b[key]
    trials = []
    for repeat in range(rounds):
        for arm in ('base_a','opt_a','opt_b','base_b'):
            for phase, items in jobs.items():
                started = time.perf_counter()
                for job in items: pack(job, arm.startswith('opt'), budget)
                elapsed = time.perf_counter() - started
                assert budget.snapshot()['used_staging_bytes'] == 0
                trials.append(dict(round=repeat, arm=arm, phase=phase, calls=len(items),
                    seconds=elapsed, mean_us=elapsed * 1e6 / len(items)))
    means = {phase:{mode:statistics.fmean(t['mean_us'] for t in trials
        if t['phase'] == phase and t['arm'].startswith(prefix))
        for mode,prefix in (('baseline','base'),('indexed','opt'))} for phase in jobs}
    assert before == [replay.byte_digest(s) for s in sources]
    return dict(case=int(path.parent.name), input_path=path.relative_to(ROOT).as_posix(),
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), consumed_banks=420,
        known_prompt_rows=int(known.sum()), uncaptured_prompt_rows=known.numel()-int(known.sum()),
        source_bytes_unchanged=True, selected_wire_bytes_exact=True, observations=observations,
        matched_cpu_timing=dict(rounds=rounds, per_delivery_mean_us=means, trials=trials))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    capture, output = args.capture.resolve(), args.output.resolve()
    if (not capture.is_relative_to(ROOT / 'artifacts') or not output.is_relative_to(ROOT / 'artifacts')
            or output.exists() or not 1 <= args.rounds <= 20):
        raise ValueError('project artifacts capture and fresh bounded output required')
    paths = [Path(__file__), ROOT / 'benchmark/verify_pvd_selected_views_real_kv.py'] + [
        ROOT / 'python/sglang/srt/disaggregation/pvd' / name for name in
        ('sparse_copy.py','sparse_row_index.py','sparse_payload.py','sparse_delivery.py','protocol.py','transfer_lifecycle.py')]
    hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in paths}
    torch.set_num_threads(1)
    result = dict(schema='pvd-indexed-pack-real-kv-local-v1', cpu_only=True, torch=torch.__version__,
        cuda_available=torch.cuda.is_available(), cpu_threads=torch.get_num_threads(),
        native_gpu_decode_tested=False, selected_views_identical_in_both_arms=True,
        source_is_declared_cpu_reconstruction=True, uncaptured_rows_poisoned_and_never_selected=True,
        live_speedup_demonstrated=False,
        timing_scope='actual CPU helper+dispatch+index workspace creation/admission/host tensor/group views/release; preallocated source/manifest/destination excluded; no CUDA/H2D/native/network/D wait',
        cases=[])
    for case in (99401,99402):
        row = verify_case(capture / str(case) / 'trajectory.pt', args.rounds)
        result['cases'].append(row)
        print(json.dumps(dict(case=case, mean_us=row['matched_cpu_timing']['per_delivery_mean_us'],
            steady=row['observations'])), flush=True)
    for p in paths: assert hashes[p.relative_to(ROOT).as_posix()] == hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    result['source_hashes'] = hashes
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__': main()
