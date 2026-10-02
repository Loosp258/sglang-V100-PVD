"""V wall stages for ordered, single-request pilots with exact RPC counts.

Each completed request has 28 initial-bank jobs and 28 jobs per future step,
each searching two ranks. Reject missing/extra/retry profiles instead of using
the second-granularity D log to infer exact cross-node timing boundaries.
"""
import argparse
import ast
import json
from pathlib import Path
import re
from statistics import median

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
root = a.directory
summary = json.loads((root / 'summary.json').read_text())
result = {}
shared_offset = 0
mode_rows = {}
comparison = (json.loads((root / 'comparison.json').read_text())['comparison']
    if (root / 'comparison.json').exists() else 'pipeline')
for arm, requests in summary['requests'].items():
    path = root / f'{arm}_v.log'
    if not path.exists():
        path = root / 'shared_v.log'
    log = path.read_text()
    batches = [(path, int(items), int(nqueries), ast.literal_eval(stages))
        for path, items, nqueries, stages in re.findall(
        r'PVD V search-batch path=(\S+) items=(\d+) query_rows=(\d+) stage_ms=(\{[^\n]+\})', log)]
    for _, _, _, stages in batches:
        if 'native_submit' in stages:
            # Combine adjacent stages per observation *before* aggregating.
            # Pooling moves implicit allocator waits into the explicit fence;
            # submission time alone is not a native-search speedup metric.
            stages['native_submit_and_completion'] = stages['native_submit'] + stages['native_completion']
            stages['candidate_handling'] = sum(stages.get(name, 0.0) for name in (
                'candidate_mapping', 'score_restore', 'host_materialize',
                'backend_enqueue', 'candidate_download'))
    warmups = json.loads((root / f'{arm}_warmup.json').read_text())
    warm_count = sum((r['completion_tokens'] - 1) * 28 * 2 for r in warmups)
    formal_count = sum((r['completion_tokens'] - 1) * 28 * 2 for r in requests)
    if path.name == 'shared_v.log':
        offset = shared_offset
        shared_offset += warm_count + formal_count
    else:
        offset = 0
        assert len(batches) == warm_count + formal_count
    offset += warm_count
    selected = []
    for request in requests:
        # First 56 profiles prime initial banks; subsequent profiles are Decode.
        expected = (request['completion_tokens'] - 2) * 28 * 2
        steady = batches[offset + 56:offset + 56 + expected]
        offset += 56 + expected
        assert len(steady) == expected, (arm, request['case'], len(steady), expected)
        assert all(row[1:3] == (2, 14) for row in steady)
        if comparison == 'v-latency':
            expected_path = 'grouped_cagra_partial_batched' + ('_host' if arm.startswith('opt') else '')
            assert all(row[0] == expected_path for row in steady), 'unexpected V search fallback/path'
        mode = ('optimized' if arm.startswith('opt') else 'baseline') if comparison in ('v-search', 'v-latency') else (
            'overlap' if arm.startswith('overlap') else 'serial')
        mode_rows.setdefault(mode, []).extend(steady)
        selected.append(dict(case=request['case'], batches=len(steady),
            paths=sorted({row[0] for row in steady}),
            median_stage_ms={name: median([row[3][name] for row in steady])
                for name in steady[0][3]}))
    result[arm] = selected
if shared_offset:
    assert len(batches) == shared_offset, (len(batches), shared_offset)
aggregate = {mode: dict(batches=len(rows), paths=sorted({row[0] for row in rows}),
    median_stage_ms={name: median([row[3][name] for row in rows]) for name in rows[0][3]})
    for mode, rows in mode_rows.items()}
result = dict(aggregate=aggregate, requests=result,
    scope='ordered completed single-request pilots; exact two-rank RPC counts; V host wall timings')
(root / 'v_search_summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
