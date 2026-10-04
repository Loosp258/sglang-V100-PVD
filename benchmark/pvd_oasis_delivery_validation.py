"""Actual completed delivery proofs for the independent CloudLab pilots.

These gates use successful-request counters rather than inferring reuse from a
configuration flag. The historical pilots have no delivery profiles and do not
call these gates.
"""

import math
from statistics import median


DELIVERY_COMPARISONS = ("v-combine", "v-slots", "v-workers", "v-direct-sparse", "d-gpu-bank", 'd-stages', 'd-workspace', 'v-contiguous', 'd-batch-install')
DELIVERY_TIMINGS = (
    "prepare_seconds", "allocate_seconds", "register_seconds",
    "reserve_seconds", "start_seconds", "combined_seconds", "poll_seconds",
    "ack_seconds", "close_seconds", "cache_copy_seconds",
)
DELIVERY_COUNTS = {
    "physical_register_calls": "registration_count",
    "physical_unregister_calls": "unregistration_count",
    "reserve_calls": "reserve_rpc_count",
    "start_calls": "start_rpc_count",
    "combined_calls": "combined_rpc_count",
    "poll_calls": "poll_rpc_count",
    "ack_calls": "ack_rpc_count",
}


def integer(value, name, *, minimum=0):
    assert type(value) is int and value >= minimum, (name, value)
    return value


def delivery_flags(comparison, arm):
    assert comparison in DELIVERY_COMPARISONS
    return (comparison == "v-combine" and arm.startswith("opt"),
            comparison == "v-slots" and arm.startswith("opt"))


def comparison_workers(comparison, arm):
    """The worker pilot changes only callback concurrency, from two to four."""
    return 4 if comparison == "v-workers" and arm.startswith("opt") else 2


def validate_io_snapshot(snapshot, *, reuse_io, jobs):
    assert isinstance(snapshot, dict), "missing completed request IO snapshot"
    assert snapshot["reuse_io"] is reuse_io
    assert snapshot["manager_io_loop_reused"] is reuse_io
    assert snapshot["shared_close_submitted"] is reuse_io
    assert snapshot["closed"] is True and snapshot["closing"] is True
    for name in ("job_count", "worker_loops_created", "search_clients_created",
                 "control_clients_created", "search_sessions_created",
                 "control_sessions_created"):
        integer(snapshot[name], name)
    if snapshot.get('staged_transport', False):
        assert not reuse_io
        validate_stage_snapshot(snapshot, jobs=jobs)
        return
    assert snapshot["job_count"] == snapshot["worker_loops_created"] == jobs
    clients = 2 if reuse_io else 2 * jobs
    assert snapshot["search_clients_created"] == clients
    assert snapshot["control_clients_created"] == clients
    assert snapshot["search_sessions_created"] == clients
    if reuse_io:
        assert snapshot["control_sessions_created"] == 2
    else:
        # Cache hits need no control HTTP session, but clients are per job.
        assert 0 < snapshot["control_sessions_created"] <= 2 * jobs


def validate_stage_snapshot(snapshot, *, jobs):
    stages = snapshot['stages']
    assert stages['closed'] is stages['closing'] is True
    assert stages['accepted'] == stages['completed'] == snapshot['job_count'] == jobs
    assert stages['failed'] == stages['pending'] == stages['retained_states'] == stages['retirement_errors'] == 0
    assert stages['max_pending'] == 56 and 0 < stages['peak_pending'] <= 56
    assert stages['workers'] == dict(search=2, delivery=2, install=1)
    retired = stages['retired_workers']
    for name, limit in stages['workers'].items():
        assert 0 < stages['peak_active'][name] <= limit
        assert 0 < retired[name] <= limit
    assert snapshot['worker_loops_created'] == sum(retired.values())
    assert snapshot['search_clients_created'] == snapshot['search_sessions_created'] == 2 * retired['search']
    assert snapshot['control_clients_created'] == 2 * retired['delivery']
    assert 0 < snapshot['control_sessions_created'] <= snapshot['control_clients_created']
    rows = snapshot['stage_trace']
    assert len(rows) == jobs and jobs % 28 == 0
    actual = set()
    expected = {(True, 0, layer) for layer in range(28)}
    expected.update((False, step, layer) for step in range(jobs // 28 - 1) for layer in range(28))
    for row in rows:
        key = row['bootstrap'], row['step'], row['layer']
        assert key not in actual and key in expected
        actual.add(key)
        assert row['failed'] is False
        assert row['published'] <= row['terminal'] <= row['deadline']
        assert [phase['stage'] for phase in row['phases']] == ['search', 'delivery', 'install']
        previous = row['published']
        for phase in row['phases']:
            assert previous <= phase['queued'] <= phase['start'] <= phase['end'] <= row['terminal']
            assert type(phase['worker']) is int and 0 <= phase['worker'] < stages['workers'][phase['stage']]
            previous = phase['end']
    assert actual == expected


def validate_gpu_backup(snapshot, *, enabled):
    assert snapshot.get('gpu_receive_to_bank', False) is enabled
    backup = snapshot.get('gpu_backup')
    if not enabled:
        assert backup is None
        return
    assert backup['closed'] is backup['closing'] is True
    assert backup['quarantined'] is False
    assert backup['submitted'] == backup['completed'] == snapshot['delivery_count']
    assert all(backup[name] == 0 for name in ('pending_rows', 'retained_owners', 'charged_bytes'))
    assert 0 < backup['peak_bytes'] <= 33554432


def validate_delivery_snapshot(snapshot, *, comparison, arm):
    combine, slots = delivery_flags(comparison, arm)
    assert snapshot["combine_reserve_start"] is combine
    assert snapshot["reuse_receive_slots"] is slots
    assert snapshot["closed"] is True and snapshot["closing"] is True
    deliveries = integer(snapshot["delivery_count"], "delivery_count", minimum=1)
    for name in DELIVERY_COUNTS.values():
        integer(snapshot[name], name)
    assert snapshot["ack_rpc_count"] == deliveries
    assert snapshot["combined_rpc_count"] == (deliveries if combine else 0)
    assert snapshot["reserve_rpc_count"] == (0 if combine else deliveries)
    assert snapshot["start_rpc_count"] == (0 if combine else deliveries)
    registrations = snapshot["registration_count"]
    assert snapshot["unregistration_count"] == registrations
    if slots:
        assert 0 < registrations < deliveries, "receive registrations were not reused"
        pool = snapshot["receive_pool"]
        assert isinstance(pool, dict), "missing physical receive pool inventory"
        assert pool["closed"] is True and pool["closing"] is True and pool["quarantine"] is None
        workers = comparison_workers(comparison, arm)
        assert integer(pool["slots_per_rank"], "slots_per_rank", minimum=1) == workers
        assert integer(pool["capacity_bytes"], "capacity_bytes", minimum=1) == 32768
        for name in ("physical_slots", "leased_slots", "unknown_slots", "physical_bytes"):
            assert integer(pool[name], name) == 0, (name, "receive pool did not retire")
        for name in ("physical_register_calls", "physical_registrations",
                     "physical_release_calls", "physical_releases"):
            assert integer(pool[name], name, minimum=1) == registrations
        assert registrations <= 2 * workers, "physical receive pool exceeded configured slots per rank"
        assert integer(pool["acquired_leases"], "acquired_leases", minimum=1) == deliveries
        assert integer(pool["returned_leases"], "returned_leases", minimum=1) == deliveries
    else:
        assert registrations == deliveries, "per-delivery registration mode changed"


def validate_bank_install_profile(profile, *, batched):
    """Actual fenced installation counters; a config flag alone is insufficient."""
    assert profile['mode'] == ('batched' if batched else 'per_head')
    assert profile['cuda'] is profile['completion_proven'] is True
    selected = integer(profile['selected_rows'], 'selected_rows', minimum=1)
    hits = integer(profile['resident_rows'], 'resident_rows')
    misses = integer(profile['cpu_rows'], 'cpu_rows')
    assert selected == hits + misses and selected <= 128
    assert integer(profile['kv_h2d_bytes'], 'kv_h2d_bytes') == misses * 512
    seconds = profile['install_seconds']
    assert type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0
    upload = integer(profile['kv_h2d_calls'], 'kv_h2d_calls')
    gather = integer(profile['resident_gather_calls'], 'resident_gather_calls')
    scatter = integer(profile['resident_scatter_calls'], 'resident_scatter_calls')
    miss_scatter = integer(profile['cpu_scatter_calls'], 'cpu_scatter_calls')
    if batched:
        assert upload == int(bool(misses))
        assert gather == scatter == 2 * int(bool(hits))
        assert miss_scatter == 2 * int(bool(misses))
        assert integer(profile['tensor_bound_bytes'], 'tensor_bound_bytes', minimum=1) * 2 <= 32 << 20
        assert integer(profile['index_metadata_bytes'], 'index_metadata_bytes') == (2 * hits + misses) * 8
    else:
        assert int(bool(misses)) <= upload <= min(4, misses)
        assert 2 * int(bool(hits)) <= gather == scatter <= 2 * min(4, hits)
        assert miss_scatter == 2 * upload


def validate_delivery_profiles(trace, *, comparison, arm):
    """Check both rank profiles against cumulative post-retirement counters."""
    snapshot = trace["io"]
    if comparison == 'd-batch-install':
        assert snapshot['batched_bank_install'] is arm.startswith('opt')
        assert snapshot['staged_transport'] is snapshot['sort_missing_tokens'] is False
        assert trace['attention_workspace'] is None
        validate_gpu_backup(snapshot, enabled=False)
        for row in trace['transport']:
            validate_bank_install_profile(row['bank_install'], batched=arm.startswith('opt'))
    if comparison == 'v-contiguous':
        assert snapshot['sort_missing_tokens'] is arm.startswith('opt')
        assert snapshot['staged_transport'] is False
        validate_gpu_backup(snapshot, enabled=False)
    if comparison == 'd-workspace':
        assert snapshot['staged_transport'] is False
        validate_gpu_backup(snapshot, enabled=False)
        workspace = trace['attention_workspace']
        if arm.startswith('opt'):
            assert workspace['closed'] is True and workspace['quarantined'] is False
            assert 0 < workspace['base_bytes'] <= workspace['max_bytes'] == 32 << 20
            assert workspace['graph'] is False and workspace['graph_shapes'] == []
            assert workspace['graph_allocated_bytes'] == workspace['graph_reserved_bytes'] == 0
        else:
            assert workspace is None
    if comparison == 'd-gpu-bank':
        validate_gpu_backup(snapshot, enabled=arm.startswith('opt'))
    if comparison == 'd-stages':
        assert snapshot['staged_transport'] is arm.startswith('opt')
        validate_gpu_backup(snapshot, enabled=False)
        if arm.startswith('opt'):
            validate_stage_snapshot(snapshot, jobs=420)
            timing = {(row['step'], row['layer']): row for row in snapshot['stage_trace'] if not row['bootstrap']}
            for row in trace['layers']:
                stages = timing[row['step'], row['layer']]
                assert row['published'] == stages['published']
                assert row['worker_start'] == stages['phases'][0]['start']
                assert row['ready'] == stages['terminal']
    validate_delivery_snapshot(snapshot, comparison=comparison, arm=arm)
    combine, slots = delivery_flags(comparison, arm)
    profiles = []
    for transport in trace["transport"]:
        items = transport["deliveries"]
        assert isinstance(items, list) and len(items) <= 2
        assert len({item["rank"] for item in items}) == len(items), "duplicate delivery rank"
        assert sum(item["remote_rows"] for item in items) == transport["remote_rows"]
        for item in items:
            assert type(item["rank"]) is int and item["rank"] in (0, 1)
            rows = integer(item["remote_rows"], "remote_rows", minimum=1)
            assert rows <= 64, "changed per-rank candidate budget"
            assert integer(item["nbytes"], "nbytes", minimum=1) == rows * 512
            assert item["combine_reserve_start"] is combine
            assert item["reuse_receive_slots"] is slots
            if comparison == 'd-gpu-bank':
                assert item['gpu_receive_to_bank'] is arm.startswith('opt')
            if comparison == 'v-contiguous':
                assert item['sort_missing_tokens'] is arm.startswith('opt')
                assert 1 <= integer(item['wire_runs'], 'wire_runs') <= rows
                if arm.startswith('opt'):
                    assert item['wire_ids_sorted'] is True
            for name in DELIVERY_TIMINGS:
                value = item[name]
                assert type(value) in (int, float) and math.isfinite(value) and value >= 0, (name, value)
            for name in DELIVERY_COUNTS:
                integer(item[name], name)
            assert item["reserve_calls"] == item["start_calls"] == (0 if combine else 1)
            assert item["combined_calls"] == (1 if combine else 0)
            # A successful start response can already prove terminal readiness.
            # Zero polling is valid; actual totals are matched below.
            assert item["ack_calls"] == 1
            assert item["physical_register_calls"] in (0, 1)
            assert item["physical_unregister_calls"] in (0, 1)
            if not slots:
                assert item["physical_register_calls"] == item["physical_unregister_calls"] == 1
            profiles.append(item)
    assert len(profiles) == snapshot["delivery_count"]
    if comparison == 'd-gpu-bank' and arm.startswith('opt'):
        assert snapshot['gpu_backup']['rows_copied'] == sum(item['remote_rows'] for item in profiles)
    for field, counter in DELIVERY_COUNTS.items():
        total = sum(item[field] for item in profiles)
        if slots and field == "physical_unregister_calls":
            # Request retirement closes the physical pool after all jobs drain.
            assert total <= snapshot[counter]
        else:
            assert total == snapshot[counter], (field, total, snapshot[counter])
    return dict(deliveries=len(profiles),
                physical_registrations=snapshot["registration_count"],
                physical_unregistrations=snapshot["unregistration_count"],
                median_stage_ms={name.removesuffix("_seconds"): median(item[name] * 1000 for item in profiles)
                                 for name in DELIVERY_TIMINGS},
                scope="all completed sparse deliveries including initial-bank priming; final pool unregistration included in IO snapshot")
