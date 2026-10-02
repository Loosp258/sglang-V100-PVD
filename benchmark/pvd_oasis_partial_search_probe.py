"""Same immutable fast graph and real K/Q: two-head baseline versus cache.

Native CAGRA width, logical IDs, Top4 and GQA union budget are unchanged.
This offline probe does not generate tokens or measure network latency.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import time

p = argparse.ArgumentParser()
p.add_argument('--fixtures', type=Path, nargs=2, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--focus-layer', type=int)
a = p.parse_args()
if a.focus_layer is not None and not 0 <= a.focus_layer < 28:
    p.error('focus layer must be in 0..27')

import cuvs.neighbors.cagra  # load native dependencies before the Torch bridge
import torch
from pvd_cagra_joint_manager_probe import load
from sglang.srt.disaggregation.pvd.cagra_kv_update import CagraKVUpdateBackend
from sglang.srt.disaggregation.pvd.prompt_index import PromptIndexManager, SearchRequestIdentity
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

output = dict(ranks=[])
for rank, path in enumerate(a.fixtures):
    torch.cuda.set_device(rank)
    fixture, packed, layout, manifest = load(path, rank)
    assert manifest.page_count == 2159
    b = CagraKVUpdateBackend(device=f'cuda:{rank}', native_bytes_per_index=536870912,
        global_native_cap_bytes=671088640, graph_degree=16, intermediate_degree=16,
        itopk_size=2048, exact_head_groups=4, nogil_extend=True, routing_edges=2,
        small_tail_max_rows=512, fused_prepare=True, prepared_tail=True,
        fixed_native_views=True, ahead_capture=True, reuse_scores=True)
    m = PromptIndexManager(vector_space='qwen25-7b-pvd', backend=b,
        group_heads=4, batched_k_extraction=True, fused_k_centering=True,
        prepared_tail=True, budget=TransferBudget(2147483648, 1),
        batched_group_search=True, partial_group_search=True)
    entry = f'partial-search-{rank}'
    m.open(entry)
    for boundary in (2048, 2159):
        if boundary == 2159:
            m.note_kv_readable(entry)
        m.progress_chunked(entry, packed, layout=layout, manifest=manifest,
            complete_pages=boundary, stored=boundary == 2159)
    q = fixture['local_queries'].float().cpu().contiguous()
    original_query_shape = tuple(q.shape)
    assert q.ndim == 3 and q.shape[0] == 56 and q.shape[1] > 0 and q.shape[2] == 128
    k = fixture['local_sources'].float().to(rank)
    batches, exact = [], []
    for layer in range(28):
        items, labels = [], []
        for head in range(2):
            identity = SearchRequestIdentity(vector_space=m.vector_space,
                positional_encoding='rope_applied', entry_transfer_id=entry,
                layer=layer, kv_head=rank * 2 + head)
            query = q[layer * 2 + head]
            items.append((identity, query, 4))
            wanted = (query.to(rank) @ k[layer * 2 + head].T).topk(4, dim=1).indices
            labels.append(set(wanted.cpu().flatten().tolist()))
        batches.append(tuple(items))
        exact.append(labels)
    batch = next(iter(b._owners.values())).auxiliary[0]
    graph_hash = hashlib.sha256(batch.native_graph[:, :4 * batch.count].cpu().numpy().tobytes()).hexdigest()
    trials = []
    for repetition in range(a.repeats + 2):
        for mode in ('baseline', 'optimized', 'optimized', 'baseline'):
            m.partial_group_search = mode == 'optimized'
            rows = []
            for layer, requests in enumerate(batches):
                if a.focus_layer is not None and layer != a.focus_layer:
                    continue
                torch.cuda.synchronize(rank)
                tick = time.perf_counter()
                meta = {}
                results = m.search_many(requests, metadata=meta)
                seconds = time.perf_counter() - tick
                ids = [list(r.selection.token_ids) for r in results]
                recalls = [len(set(found) & wanted) / len(wanted)
                    for found, wanted in zip(ids, exact[layer])]
                rows.append(dict(layer=layer, seconds=seconds, ids=ids,
                    recall_union_top4=recalls, path=meta['path'], stages=meta.get('stages', {})))
            trials.append(dict(repetition=repetition, warmup=repetition < 2, mode=mode, rows=rows))
    stats = {}
    for mode in ('baseline', 'optimized'):
        rows = [row for trial in trials if not trial['warmup'] and trial['mode'] == mode for row in trial['rows']]
        recalls = [v for row in rows for v in row['recall_union_top4']]
        stats[mode] = dict(median_two_head_ms=median([r['seconds'] * 1000 for r in rows]),
            mean_recall_union_top4=sum(recalls) / len(recalls), worst_recall_union_top4=min(recalls))
        stats[mode]['stage_ms'] = {name: median([r['stages'].get(name, 0.0) * 1000 for r in rows])
            for name in sorted({name for r in rows for name in r['stages']})}
    by_layer = {}
    for trial in trials:
        if not trial['warmup']:
            for row in trial['rows']:
                by_layer.setdefault(row['layer'], set()).add(json.dumps(row['ids']))
    cached_bytes = sum(ws.retained_bytes for ws, owner in m._entries[entry].search_workspaces.values())
    m.close(entry)
    assert not b._owners and m.budget.snapshot()['used_staging_bytes'] == m.shared_native_budget_bytes
    output['ranks'].append(dict(rank=rank, fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        graph_sha256=graph_hash, all_native_ids_identical=all(len(v) == 1 for v in by_layer.values()),
        original_query_shape=original_query_shape,
        focus_layer=a.focus_layer,
        query_sha256=hashlib.sha256(q.numpy().tobytes()).hexdigest(),
        cached_bytes=cached_bytes, stats=stats, trials=trials))
    a.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({k: v for k, v in output['ranks'][-1].items() if k != 'trials'}), flush=True)
