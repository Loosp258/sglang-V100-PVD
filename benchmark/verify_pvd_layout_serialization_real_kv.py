"""Matched CPU layout serialization/packing with captured selected KV.

The reference temporarily substitutes only the verified previous to_dict
method. All metadata checks and fingerprints still run. Reconstructed CPU
sources are declared fixtures, not native Entries or GPU/Decode measurements.
"""

import argparse
import ast
from contextlib import contextmanager
import dataclasses
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import verify_pvd_selected_views_real_kv as replay
import torch
from sglang.srt.disaggregation.pvd.protocol import KVLayoutSignature

ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = '3df6cc9d6'
PROTOCOL = 'python/sglang/srt/disaggregation/pvd/protocol.py'
CANDIDATE = KVLayoutSignature.to_dict


def previous_to_dict(self):
    result = dataclasses.asdict(self)
    result["extra"] = dict(self.extra)
    return result


def check_reference():
    blob = subprocess.check_output(['git', 'show', BASELINE_COMMIT + ':' + PROTOCOL], cwd=ROOT)
    module = ast.parse(blob)
    owner = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'KVLayoutSignature')
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == 'to_dict')
    reference = next(n for n in ast.parse(Path(__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == 'previous_to_dict')
    assert [ast.dump(n, include_attributes=False) for n in method.body] == [
        ast.dump(n, include_attributes=False) for n in reference.body]
    return dict(commit=BASELINE_COMMIT, path=PROTOCOL,
        normalized_lf_sha256=hashlib.sha256(blob.replace(b'\r\n', b'\n')).hexdigest(),
        reference_method_ast_matches=True)


@contextmanager
def serializer(mode):
    original = KVLayoutSignature.to_dict
    KVLayoutSignature.to_dict = CANDIDATE if mode == 'candidate' else previous_to_dict
    try:
        yield
    finally:
        KVLayoutSignature.to_dict = original


def summarize(trials, phase):
    return {mode: statistics.fmean(t['mean_us'] for t in trials
        if t['phase'] == phase and t['arm'].startswith(prefix))
        for mode, prefix in (('baseline', 'base'), ('candidate', 'opt'))}


def verify_case(path, rounds):
    trajectory = torch.load(path, map_location='cpu', weights_only=True)
    sources, known, jobs = replay.prepare(trajectory)
    before = [replay.byte_digest(s) for s in sources]
    observations = {}
    for mode in ('baseline', 'candidate'):
        with serializer(mode):
            phases = {}
            for phase, items in jobs.items():
                wire = hashlib.sha256()
                manifest = hashlib.sha256()
                layouts = hashlib.sha256()
                rows = views = 0
                for job in items:
                    views += replay.pack(job, True)
                    assert torch.equal(job['target'], job['oracle'])
                    wire.update(job['target'].numpy().tobytes())
                    manifest.update(job['manifest'].fingerprint.encode())
                    layouts.update(json.dumps(job['layout'].to_dict(), sort_keys=True,
                        separators=(',', ':'), default=str).encode())
                    rows += sum(len(s.token_ids) for s in job['manifest'].specs)
                phases[phase] = dict(deliveries=len(items), rows=rows, bytes=rows * 512,
                    source_component_views=views, wire_sha256=wire.hexdigest(),
                    manifest_sha256=manifest.hexdigest(), layout_wire_sha256=layouts.hexdigest())
            observations[mode] = phases
    assert observations['baseline'] == observations['candidate']
    fingerprint = jobs['steady'][0]['layout'].fingerprint
    timings = []
    for repeat in range(rounds):
        for arm in ('base_a', 'opt_a', 'opt_b', 'base_b'):
            mode = 'candidate' if arm.startswith('opt') else 'baseline'
            with serializer(mode):
                for phase, items in jobs.items():
                    started = time.perf_counter()
                    for job in items: replay.pack(job, True)
                    elapsed = time.perf_counter() - started
                    timings.append(dict(round=repeat, arm=arm, phase=phase,
                        calls=len(items), seconds=elapsed, mean_us=elapsed * 1e6 / len(items)))
                started = time.perf_counter()
                for _ in range(2000): actual = jobs['steady'][0]['layout'].fingerprint
                elapsed = time.perf_counter() - started
                assert actual == fingerprint
                timings.append(dict(round=repeat, arm=arm, phase='fingerprint',
                    calls=2000, seconds=elapsed, mean_us=elapsed * 1e6 / 2000))
    assert before == [replay.byte_digest(s) for s in sources]
    return dict(case=int(path.parent.name), input_path=path.relative_to(ROOT).as_posix(),
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        known_prompt_rows=int(known.sum()), uncaptured_prompt_rows=known.numel() - int(known.sum()),
        reconstructed_source_bytes=sum(s.numel() for s in sources), source_bytes_unchanged=True,
        consumed_banks=420, selected_wire_bytes_exact=True, observations=observations,
        matched_cpu_timing=dict(rounds=rounds, trials=timings,
            per_delivery_mean_us={p:summarize(timings, p) for p in ('bootstrap', 'steady')},
            per_fingerprint_mean_us=summarize(timings, 'fingerprint')))


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
    reference = check_reference()
    torch.set_num_threads(1)
    paths = [Path(__file__), ROOT / PROTOCOL,
        ROOT / 'benchmark/verify_pvd_selected_views_real_kv.py',
        ROOT / 'python/sglang/srt/disaggregation/pvd/sparse_copy.py',
        ROOT / 'python/sglang/srt/disaggregation/pvd/sparse_payload.py']
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for p in paths}
    result = dict(schema='pvd-layout-serialization-real-kv-local-v1', cpu_only=True,
        torch=torch.__version__, cuda_available=torch.cuda.is_available(), cpu_threads=torch.get_num_threads(),
        selected_component_views=True, native_gpu_decode_tested=False,
        source_is_declared_cpu_reconstruction=True, uncaptured_rows_poisoned_and_never_selected=True,
        live_speedup_demonstrated=False, reference=reference,
        timing_scope='same actual CPU pack helper plus dispatch with selected views in both arms; changes only to_dict; source/manifest/destination preparation, CUDA, registration, native, network and D wait excluded',
        cases=[])
    for case in (99401, 99402):
        row = verify_case(capture / str(case) / 'trajectory.pt', args.rounds)
        result['cases'].append(row)
        print(json.dumps(dict(case=case, cpu=row['matched_cpu_timing']['per_delivery_mean_us'],
            fingerprint_us=row['matched_cpu_timing']['per_fingerprint_mean_us'])), flush=True)
    assert KVLayoutSignature.to_dict is CANDIDATE
    for p in paths:
        assert hashes[p.relative_to(ROOT).as_posix()] == hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    result['source_hashes'] = hashes
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__': main()
