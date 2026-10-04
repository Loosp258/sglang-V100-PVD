"""Real captured selected KV, reconstructed CPU source, actual Torch pack helper.

Uncaptured rows remain poisoned and are never selected. This is not a full
Prompt fixture, native PUT, CAGRA quality test or GPU/Decode timing experiment.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'python'))
for name in ('sglang', 'sglang.srt', 'sglang.srt.disaggregation', 'sglang.srt.disaggregation.pvd'):
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT / 'python' / name.replace('.', '/'))]
    sys.modules[name] = module  # frontend bootstrap only; real Torch/PVD

import torch
from sglang.srt.disaggregation.pvd.kv_packer import PVD_TENSOR_LAYOUT
from sglang.srt.disaggregation.pvd.protocol import KVLayoutSignature, KVShardManifest
from sglang.srt.disaggregation.pvd.sparse_copy import copy_sparse_kv_into
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec


def byte_digest(tensor):
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def prepare(trajectory):
    if (trajectory['schema'] != 'pvd-oasis-ready-replay-v1'
            or len(trajectory['prompt_ids']) != 2159 or len(trajectory['steps']) != 15):
        raise ValueError('qualified real captured trajectory required')
    source = [torch.full((2 * 28 * 2159 * 2 * 128 * 2,), 165, dtype=torch.uint8) for _ in (0, 1)]
    typed = [s.view(torch.float16).reshape(2, 28, 2159, 2, 128) for s in source]
    known = torch.zeros((28, 4, 2159), dtype=torch.bool)
    for step, row in enumerate(trajectory['steps']):
        if row['step'] != step or len(row['banks']) != 28:
            raise ValueError('ordered complete consumed banks required')
        for layer, bank in enumerate(row['banks']):
            for head, tokens in enumerate(bank['ids']):
                if not tokens or len(tokens) > 32 or len(set(tokens)) != len(tokens):
                    raise ValueError('bounded captured bank required')
                if any(type(t) is not int or not 0 <= t < 2159 for t in tokens):
                    raise ValueError('invalid captured Prompt token')
                values = torch.stack((bank['keys'][head, :len(tokens)], bank['values'][head, :len(tokens)]))
                if values.dtype != torch.float16 or values.shape != (2, len(tokens), 128):
                    raise ValueError('exact captured KV shape/dtype required')
                previous = known[layer, head, tokens]
                if previous.any():
                    old = typed[head // 2][:, layer, tokens, head % 2][:, previous]
                    if not torch.equal(old.view(torch.uint8), values[:, previous].contiguous().view(torch.uint8)):
                        raise AssertionError('captured immutable KV bytes changed')
                typed[head // 2][:, layer, tokens, head % 2] = values
                known[layer, head, tokens] = True
    layout = KVLayoutSignature(model_id='captured-Qwen2.5-7B', model_revision='',
        kv_dtype='torch.float16', page_size=1, num_layers=28, total_kv_heads=4,
        kv_heads_per_rank=2, head_dim=128, tp_size=2, pp_size=1, tensor_layout=PVD_TENSOR_LAYOUT,
        extra=dict(component_count=56, component_dtypes=['torch.float16'] * 56,
            component_token_shapes=[[2, 128] for _ in range(56)], component_bytes_per_token=[512] * 56))
    shards = [KVShardManifest(rank=r, rail=f'local-oracle-{r}', expected_bytes=source[r].numel(),
        page_count=2159, last_page_valid_tokens=1, layer_start=0, layer_end=28) for r in (0, 1)]
    seen = [[set() for _ in range(4)] for _ in range(28)]
    jobs = dict(bootstrap=[], steady=[])
    for step, row in enumerate(trajectory['steps']):
        phase = 'bootstrap' if step == 0 else 'steady'
        for layer, bank in enumerate(row['banks']):
            for rank in (0, 1):
                specs, parts = [], []
                for head in range(rank * 2, rank * 2 + 2):
                    missing = tuple(t for t in bank['ids'][head] if t not in seen[layer][head])
                    if missing:
                        assert known[layer, head, list(missing)].all()
                        specs.append(SparseKVSpec('captured', 'local-fixture', f'{step}:{layer}',
                            step + 1, 'entry', 'index', 'map', layout.fingerprint, layer, head, missing))
                        lookup = {token: i for i, token in enumerate(bank['ids'][head])}
                        ids = [lookup[t] for t in missing]
                        parts.append(torch.stack((bank['keys'][head, ids], bank['values'][head, ids])).view(torch.uint8).reshape(-1))
                    seen[layer][head].update(bank['ids'][head])
                if not specs: continue
                manifest = SparseDeliveryManifest(tuple(specs), 'torch.float16', 128)
                oracle = torch.cat(parts)
                assert oracle.numel() == manifest.nbytes
                jobs[phase].append(dict(source=source[rank], target=torch.empty_like(oracle), oracle=oracle,
                    manifest=manifest, layout=layout, shard=shards[rank], rank=rank, step=step, layer=layer))
    return source, known, jobs


def pack(job, selected):
    return copy_sparse_kv_into(job['source'], job['target'], manifest=job['manifest'],
        layout=job['layout'], shard=job['shard'], entry_transfer_id='entry',
        index_version='index', id_mapping_version='map', selected_component_views=selected)


def verify_case(path, *, rounds):
    trajectory = torch.load(path, map_location='cpu', weights_only=True)
    sources, known, jobs = prepare(trajectory)
    before = [byte_digest(s) for s in sources]
    observations = {}
    for mode in ('baseline', 'selected'):
        phases = {}
        for phase, items in jobs.items():
            payload = hashlib.sha256()
            manifest_digest = hashlib.sha256()
            views = rows = 0
            for job in items:
                views += pack(job, mode == 'selected')
                assert torch.equal(job['target'], job['oracle']), (job['step'], job['layer'], job['rank'])
                payload.update(job['target'].numpy().tobytes())
                manifest_digest.update(job['manifest'].fingerprint.encode())
                rows += sum(len(s.token_ids) for s in job['manifest'].specs)
            phases[phase] = dict(deliveries=len(items), rows=rows, payload_bytes=rows * 512,
                source_component_views=views, payload_sha256=payload.hexdigest(),
                manifest_sha256=manifest_digest.hexdigest())
        observations[mode] = phases
    for phase in jobs:
        a, b = observations['baseline'][phase], observations['selected'][phase]
        assert {k: v for k, v in a.items() if k != 'source_component_views'} == {
            k: v for k, v in b.items() if k != 'source_component_views'}
        assert a['source_component_views'] == a['deliveries'] * 56
        assert b['source_component_views'] == b['deliveries'] * 2
    timings = []
    # Both modes warmed by verification above. Same preallocated jobs in ABBA.
    for repeat in range(rounds):
        for arm in ('base_a', 'opt_a', 'opt_b', 'base_b'):
            for phase, items in jobs.items():
                started = time.perf_counter()
                for job in items: pack(job, arm.startswith('opt'))
                elapsed = time.perf_counter() - started
                timings.append(dict(round=repeat, arm=arm, phase=phase, calls=len(items),
                    seconds=elapsed, mean_us=elapsed * 1e6 / len(items)))
    assert before == [byte_digest(s) for s in sources], 'source or uncaptured poisoned rows changed'
    means = {phase: {mode: statistics.fmean(r['mean_us'] for r in timings
        if r['phase'] == phase and r['arm'].startswith('opt' if mode == 'selected' else 'base'))
        for mode in ('baseline', 'selected')} for phase in jobs}
    return dict(case=int(path.parent.name), input_path=path.relative_to(ROOT).as_posix(),
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        known_prompt_rows=int(known.sum()), uncaptured_prompt_rows=known.numel() - int(known.sum()),
        reconstructed_source_bytes=sum(s.numel() for s in sources), source_bytes_unchanged=True,
        consumed_banks=15 * 28, selected_wire_bytes_exact=True, observations=observations,
        matched_cpu_timing=dict(rounds=rounds, per_delivery_mean_us=means, trials=timings))


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
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    result = dict(schema='pvd-selected-component-views-local-v1', cpu_only=True,
        cuda_available=torch.cuda.is_available(), torch=torch.__version__, cpu_threads=torch.get_num_threads(),
        native_gpu_decode_tested=False, source_is_declared_cpu_reconstruction=True,
        uncaptured_rows_poisoned_and_never_selected=True, live_speedup_demonstrated=False,
        timing_scope='actual CPU pack helper plus loop dispatch; source, manifest and destination preparation excluded; no CUDA/registration/native/network/D wait',
        cases=[])
    for case in (99401, 99402):
        row = verify_case(capture / str(case) / 'trajectory.pt', rounds=args.rounds)
        result['cases'].append(row)
        print(json.dumps(dict(case=case, cpu_mean_us=row['matched_cpu_timing']['per_delivery_mean_us'])), flush=True)
    result['source_hashes'] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for p in (Path(__file__), ROOT / 'python/sglang/srt/disaggregation/pvd/sparse_copy.py',
            ROOT / 'python/sglang/srt/disaggregation/pvd/sparse_payload.py')}
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__': main()
