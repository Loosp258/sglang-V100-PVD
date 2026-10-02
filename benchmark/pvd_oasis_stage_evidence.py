"""Actual stage intervals, deadlines and sustained bank completion cadence."""
import math
from statistics import fmean, median

from pvd_oasis_delivery_validation import validate_stage_snapshot
from verify_pvd_oasis_latency_evidence import client_token_evidence


def phase_evidence(rows, workers):
    events = sorted((value, delta) for row in rows
                    for value, delta in ((row['start'], 1), (row['end'], -1)))
    active = peak = 0
    for _, delta in events:
        active += delta
        assert 0 <= active <= workers
        peak = max(active, peak)
    assert active == 0
    return dict(phase_jobs=len(rows), workers=workers, peak_active=peak,
        mean_service_ms=fmean((row['end'] - row['start']) * 1000 for row in rows),
        mean_queue_ms=fmean((row['start'] - row['queued']) * 1000 for row in rows),
        service_seconds=sum(row['end'] - row['start'] for row in rows))


def stage_timing_evidence(full, online):
    requests, groups = {}, {'baseline': [], 'optimized': []}
    for arm, rows in full['requests'].items():
        requests[arm] = []
        for row, event in zip(rows, online[arm], strict=True):
            token = client_token_evidence(row, event)
            layers = row['trace']['layers']
            assert len(layers) == 392
            ready = sorted(layer['ready'] for layer in layers)
            waits = [forward['wait_ms'] for forward in row['forward'] if forward['step'] > 0]
            assert len(waits) == 14
            result = dict(case=row['case'], client_tpot_mean_ms=token['client_tpot_ms'],
                kv_wait_mean_ms=fmean(waits),
                bank_completion_interval_mean_ms=(ready[-1] - ready[0]) * 1000 / (len(ready) - 1),
                mean_chain_service_ms=fmean(layer['service_seconds'] * 1000 for layer in layers),
                mean_publish_to_consume_ms=fmean((layer['consumed'] - layer['published']) * 1000 for layer in layers),
                mean_publish_to_required_ms=fmean((layer['consumed'] - layer['consumer_wait_seconds'] - layer['published']) * 1000 for layer in layers),
                ready_before_consume_fraction=sum(layer['ready_before_consume'] for layer in layers) / len(layers))
            snapshot = row['trace']['io']
            if snapshot['staged_transport']:
                validate_stage_snapshot(snapshot, jobs=420)
                phases = [item for item in snapshot['stage_trace'] if not item['bootstrap']]
                assert len(phases) == 392
                result['phases'] = {stage: phase_evidence(
                    [phase for item in phases for phase in item['phases'] if phase['stage'] == stage], workers)
                    for stage, workers in snapshot['stages']['workers'].items()}
                result['minimum_ready_deadline_slack_ms'] = min((item['deadline'] - item['terminal']) * 1000 for item in phases)
            requests[arm].append(result)
            groups['optimized' if arm.startswith('opt') else 'baseline'].append(result)
    aggregate = {}
    fields = ('client_tpot_mean_ms', 'kv_wait_mean_ms', 'bank_completion_interval_mean_ms',
              'mean_chain_service_ms', 'mean_publish_to_consume_ms', 'mean_publish_to_required_ms',
              'ready_before_consume_fraction')
    for mode, rows in groups.items():
        aggregate[mode] = {field: fmean(row[field] for row in rows) for field in fields}
        if mode == 'optimized':
            aggregate[mode]['phases'] = {stage: dict(
                workers=rows[0]['phases'][stage]['workers'],
                peak_active=max(row['phases'][stage]['peak_active'] for row in rows),
                mean_service_ms=fmean(row['phases'][stage]['mean_service_ms'] for row in rows),
                mean_queue_ms=fmean(row['phases'][stage]['mean_queue_ms'] for row in rows))
                for stage in ('search', 'delivery', 'install')}
            aggregate[mode]['minimum_ready_deadline_slack_ms'] = min(row['minimum_ready_deadline_slack_ms'] for row in rows)
    return dict(aggregate=aggregate, requests=requests,
        scope='392 consumed later-token banks/request; arithmetic means; completion cadence spans actual foreground publication and stage queues; not a pure kernel capacity bound')
