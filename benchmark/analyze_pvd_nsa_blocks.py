"""Bounded, read-only contiguous-KV analysis of saved real Decode banks.

CPU-cache misses are reconstructed from consumed banks only, with monotonic
cache contents. Final unpublished/unconsumed prefetch is absent. This measures
layout opportunities, not CAGRA recall, GPU speed or client latency.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_count(ids):
    return sum(i == 0 or token != ids[i - 1] + 1 for i, token in enumerate(ids))


def expanded_tokens(ids, block_size, prompt_tokens):
    blocks = {token // block_size for token in ids}
    return sum(min(block_size, prompt_tokens - block * block_size) for block in blocks)


def analyze_case(trajectory, source, *, capacity=32):
    if (trajectory['schema'] != 'pvd-oasis-ready-replay-v1'
            or len(trajectory['steps']) != 15 or len(trajectory['prompt_ids']) != 2159):
        raise ValueError('qualified 2159-token, 15-step real trajectory required')
    cache = [[set() for _ in range(4)] for _ in range(28)]
    counts = {phase: dict(groups=0, rows=0, ordered_runs=0, sorted_runs=0,
                         deliveries=0, payload_bytes=0) for phase in ('bootstrap', 'steady')}
    blocks = {str(size): dict(groups=0, selected_rows=0, expanded_rows=0,
                             exceeds_capacity_groups=0, maximum_expanded_rows=0)
              for size in (4, 8, 16)}
    records = []
    for step, row in enumerate(trajectory['steps']):
        if row['step'] != step or len(row['banks']) != 28:
            raise ValueError('incomplete or reordered real trajectory')
        phase = 'bootstrap' if step == 0 else 'steady'
        for layer, bank in enumerate(row['banks']):
            if len(bank['ids']) != 4:
                raise ValueError('four explicit KV heads required')
            active_ranks = set()
            for head, selected in enumerate(bank['ids']):
                selected = tuple(selected)
                if (not selected or len(selected) > capacity or len(set(selected)) != len(selected)
                        or any(type(t) is not int or not 0 <= t < 2159 for t in selected)):
                    raise ValueError('unique bounded real Prompt IDs required')
                for size, result in blocks.items():
                    expanded = expanded_tokens(selected, int(size), 2159)
                    result['groups'] += 1
                    result['selected_rows'] += len(selected)
                    result['expanded_rows'] += expanded
                    result['exceeds_capacity_groups'] += expanded > capacity
                    result['maximum_expanded_rows'] = max(result['maximum_expanded_rows'], expanded)
                missing = tuple(t for t in selected if t not in cache[layer][head])
                cache[layer][head].update(missing)
                if not missing:
                    continue
                active_ranks.add(head // 2)
                ordered, sorted_runs = run_count(missing), run_count(tuple(sorted(missing)))
                result = counts[phase]
                result['groups'] += 1
                result['rows'] += len(missing)
                result['ordered_runs'] += ordered
                result['sorted_runs'] += sorted_runs
                result['payload_bytes'] += len(missing) * 2 * 128 * 2
                records.append(dict(step=step, layer=layer, head=head, missing_ids=list(missing),
                                    ordered_runs=ordered, sorted_runs=sorted_runs))
            counts[phase]['deliveries'] += len(active_ranks)
    for result in counts.values():
        result['row_copy_calls'] = 2 * result['rows']
        result['ordered_run_copy_calls'] = 2 * result['ordered_runs']
        result['sorted_run_copy_calls'] = 2 * result['sorted_runs']
        result['sorted_copy_call_reduction_fraction'] = 1 - result['sorted_runs'] / result['rows']
    return dict(case=trajectory['case'], source=str(source.relative_to(ROOT)),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                prompt_tokens=2159, consumed_steps=15, capacity=capacity,
                counts=counts, whole_block_expansion=blocks, missing_groups=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_root, output = args.capture.resolve(), args.output.resolve()
    if not source_root.is_relative_to(ROOT / 'artifacts') or not output.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('capture and output must stay inside project artifacts')
    if output.exists():
        raise ValueError('fresh evidence output required')
    import torch
    cases = []
    for case in (99401, 99402):
        source = source_root / str(case) / 'trajectory.pt'
        trajectory = torch.load(source, map_location='cpu', weights_only=True)
        cases.append(analyze_case(trajectory, source))
    result = dict(schema='pvd-nsa-block-opportunity-v1', cases=cases,
                  scope='consumed real sparse bank/miss layout; no GPU or network performance measurement',
                  exact_tokens_and_bytes=True, cpu_cache_monotonic=True,
                  terminal_unused_prefetch_included=False, block_selection_implemented=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for case in cases:
        print(json.dumps({key: case[key] for key in ('case', 'counts', 'whole_block_expansion')}))


if __name__ == '__main__':
    main()
