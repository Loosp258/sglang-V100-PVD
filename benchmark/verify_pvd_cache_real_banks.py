"""CPU replay of both actual Oasis cache installers on captured real KV.

Readiness is a declared local fixture. No network/native terminal proof or GPU
timing is manufactured. Payload bytes/order and immutable captured KV provide
the oracle, with source overwrite after every completed installation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'artifacts/combined_rpc_agent01/deps')]
package = types.ModuleType('sglang')
package.__path__ = [str(ROOT / 'python/sglang')]
sys.modules['sglang'] = package  # frontend bootstrap only

import torch
from sglang.srt.disaggregation.pvd.oasis_transport import OasisCPUReceiveRecord, _HeadCPUCache
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
from verify_pvd_batched_real_banks import per_head


def verify_case(trajectory, source):
    if (trajectory['schema'] != 'pvd-oasis-ready-replay-v1' or len(trajectory['steps']) != 15
            or len(trajectory['prompt_ids']) != 2159):
        raise ValueError('qualified real captured trajectory required')
    observations = {}
    counters = ('cache_installed_rows', 'cache_kv_bytes', 'cache_row_clones',
        'cache_kv_copy_calls', 'cache_valid_write_calls', 'cache_index_bytes')
    for mode in ('rows', 'batched'):
        storage = torch.empty((28, 4, 2159, 2, 128), dtype=torch.float16)
        masks = torch.zeros((28, 4, 2159), dtype=torch.bool)
        caches = [[_HeadCPUCache(storage[l, h], masks[l, h]) for h in range(4)] for l in range(28)]
        residents = [None] * 28
        counts = {phase: dict(bank_checks=0, deliveries=0, groups=0, kv_h2d_bytes=0,
            **{name: 0 for name in counters}) for phase in ('bootstrap', 'steady')}
        digest = hashlib.sha256()
        registry = types.SimpleNamespace(combine_reserve_start=False, _owner=lambda: None,
            batched_cache_install=mode == 'batched', cache_capacity=32)
        for step, row in enumerate(trajectory['steps']):
            if row['step'] != step or len(row['banks']) != 28:
                raise ValueError('complete ordered real banks required')
            phase = counts['bootstrap' if step == 0 else 'steady']
            for layer, bank in enumerate(row['banks']):
                chosen = tuple(tuple(ids) for ids in bank['ids'])
                for rank in (0, 1):
                    specs, parts = [], []
                    for head in range(rank * 2, (rank + 1) * 2):
                        ids = chosen[head]
                        if not ids or len(ids) > 32 or len(set(ids)) != len(ids):
                            raise ValueError('bounded unique captured heads required')
                        missing = [(i, t) for i, t in enumerate(ids) if t not in caches[layer][head]]
                        # Prove immutable reused cached rows against captured KV.
                        for i, token in enumerate(ids):
                            if token in caches[layer][head]:
                                if not torch.equal(caches[layer][head][token],
                                        torch.stack((bank['keys'][head, i], bank['values'][head, i]))):
                                    raise AssertionError('reused captured Prompt KV changed')
                        if not missing:
                            continue
                        indexes, tokens = map(tuple, zip(*missing))
                        specs.append(SparseKVSpec('replay', 'inc', f'{step}:{layer}', step + 1,
                            'entry', 'index', 'map', 'captured', layer, head, tokens))
                        parts.append(torch.stack((bank['keys'][head, list(indexes)],
                            bank['values'][head, list(indexes)])).contiguous().view(torch.uint8).reshape(-1))
                    if not specs:
                        continue
                    manifest = SparseDeliveryManifest(tuple(specs), 'torch.float16', 128)
                    wire = torch.cat(parts)
                    assert wire.numel() == manifest.nbytes
                    digest.update(manifest.fingerprint.encode())
                    digest.update(wire.numpy().tobytes())
                    record = OasisCPUReceiveRecord(registry, manifest,
                        types.SimpleNamespace(transfer_id='local-fixture'), None)
                    record._buffer, record._ready, record._safe = wire, True, True
                    record.copy_to_cache(caches[layer])  # actual baseline/optimized record method
                    assert record._installed and record.profile['cache_install_complete']
                    assert record.profile['cache_install_mode'] == mode
                    for name in counters:
                        phase[name] += record.profile[name]
                    phase['deliveries'] += 1
                    phase['groups'] += len(specs)
                    wire.zero_()  # no borrowed KV source may escape installation
                k, v, valid, install = per_head(chosen, residents[layer], caches[layer])
                if not (torch.equal(k, bank['keys']) and torch.equal(v, bank['values'])
                        and torch.equal(valid, bank['valid'])):
                    raise AssertionError('cache replay changed captured bank bits/order/mask')
                from sglang.srt.disaggregation.pvd.oasis_qwen import PromptBank
                residents[layer] = PromptBank(chosen, k, v, valid)
                phase['bank_checks'] += 1
                phase['kv_h2d_bytes'] += install['kv_h2d_bytes']
        observations[mode] = dict(phases=counts, wire_and_manifest_sha256=digest.hexdigest(),
            installed_cache_rows=int(masks.sum()))
    assert observations['rows']['wire_and_manifest_sha256'] == observations['batched']['wire_and_manifest_sha256']
    for phase in ('bootstrap', 'steady'):
        for name in ('bank_checks', 'deliveries', 'groups', 'cache_installed_rows', 'cache_kv_bytes', 'kv_h2d_bytes'):
            assert observations['rows']['phases'][phase][name] == observations['batched']['phases'][phase][name], name
    return dict(case=trajectory['case'], source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        observations=observations, cache_and_bank_bits_order_mask_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture, output = args.capture.resolve(), args.output.resolve()
    if not capture.is_relative_to(ROOT / 'artifacts') or not output.is_relative_to(ROOT / 'artifacts') or output.exists():
        raise ValueError('project-local captures and fresh artifacts output required')
    result = dict(schema='pvd-cache-real-bank-equivalence-v1', cpu_only=True,
        native_cuda_online_tested=False, latency_measured=False, target_forward_run=False,
        terminal_unused_prefetch_included=False, receive_readiness_is_explicit_local_fixture=True,
        wire_source='captured known selected KV; not reconstructed V memory or native PUT')
    result['cases'] = [verify_case(torch.load(capture / str(case) / 'trajectory.pt',
        map_location='cpu', weights_only=True), capture / str(case) / 'trajectory.pt') for case in (99401, 99402)]
    result['source_hashes'] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for p in (Path(__file__), ROOT / 'benchmark/verify_pvd_batched_real_banks.py',
            ROOT / 'python/sglang/srt/disaggregation/pvd/oasis_transport.py',
            ROOT / 'python/sglang/srt/disaggregation/pvd/oasis_cache_install.py')}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
