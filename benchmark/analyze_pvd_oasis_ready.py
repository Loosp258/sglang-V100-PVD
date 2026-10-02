"""Verify and summarize the bounded same-runner READY-bank diagnostic.

No GPU, SSH or model loading. This reports a replay counterfactual, not an
online speedup; live capture contains diagnostic overhead.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_trial(trial, outputs):
    assert trial['all_banks_preloaded'] is True
    assert trial['target_logits_features_actual_kv_bitwise'] is True
    assert trial['predicted_tokens_identical'] is True
    assert trial['formal_kv_write_count'] == 420 and trial['formal_kv_written_rows_bitwise'] is True
    assert trial['inference_mode'] is True
    assert trial['live_max_steps_preserved'] == 15
    assert trial['published_layer_callbacks'] == 392 and trial['unused_terminal_prefetch_layers'] == 0
    assert trial['finished_unix'] > trial['started_unix'] > 0
    assert trial['ordinary_sampled_outputs'] == outputs[1:]
    assert trial['consumed_layers'] == 14 * 28
    layers = trial['layers']
    assert len(layers) == 14 * 28
    assert {(row['step'], row['layer']) for row in layers} == {
        (step, layer) for step in range(14) for layer in range(28)}
    assert trial['all_layer_callbacks_ready_before_consume'] == all(
        row['ready_before_consume'] for row in layers)
    for row in layers:
        assert 0 <= row['queue_seconds'] and 0 <= row['service_seconds']
        assert 0 <= row['consumer_wait_seconds']
    totals = [row for row in trial['stages'] if row['name'] == 'foreground_total']
    assert [row['step'] for row in totals] == list(range(15))
    assert abs(trial['steady_token_wall_mean_ms'] - mean(row['wall_ms'] for row in totals[1:])) < 1e-8
    assert len(set(trial['private_formal_rows'])) == len(trial['private_formal_rows']) == 15


def verify(directory):
    scope = read_json(directory / 'scope.json')
    assert scope['comparison'] is False and scope['replays_are_not_client_latency'] is True
    assert read_json(directory / 'owned.json') == {}
    assert read_json(directory / 'cleanup_errors.json') == []
    gpu = read_json(directory / 'final_gpu_memory.json')
    assert set(gpu) == {'p', 'v', 'd'}
    for text in gpu.values():
        rows = text.splitlines()
        assert len(rows) == 2 and all(int(row.split(',')[1].split()[0]) == 0 for row in rows)
    online = read_json(directory / 'online.json')['capture']
    assert [row['case'] for row in online] == [99401, 99402]
    assert all(row['status'] == 200 and row['error'] is None and row['completion_tokens'] == 16 for row in online)
    prompt = read_json(directory / 'prompt_identities.json')
    summaries, reports = {}, []
    log = (directory / 'capture_d.log').read_text()
    assert 'Error in sitecustomize' not in log
    hook_pids = set()
    for line in log.splitlines():
        if line.startswith('{') and '"hook_installed": true' in line:
            hook = json.loads(line)
            assert hook['schema'] == 'pvd-oasis-ready-replay-v1'
            hook_pids.add(hook['pid'])
    for live in online:
        case = live['case']
        report = read_json(directory / 'capture' / str(case) / 'replay.json')
        assert report['schema'] == 'pvd-oasis-ready-replay-v1' and report['status'] == 'passed'
        assert report['case'] == case and report['target_steps'] == 15 and report['captured_layer_banks'] == 420
        assert report['pid'] in hook_pids, 'replay scheduler did not install spawn hook'
        assert re.search(r'capture_enabled case=' + str(case) + r' rid=' + re.escape(report['request_id']) +
                         r' pid=' + str(report['pid']) + r' runner=' + str(report['runner_id']), log)
        assert report['prompt_tokens'] == 2159 and report['output_tokens'] == 16
        assert report['prompt_ids_sha256'] == prompt[str(case)]['sha256']
        assert report['live_native_retired_before_replay'] and report['formal_private_rows_released']
        assert report['initial_session_close_joined'] is True and report['initial_receive_guard_released'] is True
        assert report['formal_allocator_free_group_finished'] is True
        budget = report['budget']
        assert budget['released'] is True and budget['gpu_refs_cleared'] is True and budget['limit_bytes'] == 256 << 20
        assert budget['retained_storage_bytes'] + budget['replay_scratch_bound_bytes'] <= budget['limit_bytes']
        assert all(budget['final'][name] == 0 for name in ('used_staging_bytes', 'used_inflight', 'reservations'))
        trajectory = directory / 'capture' / str(case) / 'trajectory.pt'
        assert hashlib.sha256(trajectory.read_bytes()).hexdigest() == report['trajectory_sha256']
        outputs = live['events'][-1]['event']['output_ids']
        assert len(outputs) == 16
        trials = report['trials']
        assert len(trials) == 5 and [row['warmup'] for row in trials] == [True, True, False, False, False]
        for trial in trials + [report['event_trial']]:
            check_trial(trial, outputs)
        assert all(not row['profiled'] for row in trials) and report['event_trial']['profiled']
        measured = trials[2:]
        steady = [row for trial in measured for row in trial['stages'] if row['step'] > 0]
        layers = [row for trial in measured for row in trial['layers']]
        stage_names = sorted({row['name'] for row in steady})
        summaries[str(case)] = dict(
            steady_token_mean_ms=mean(row['steady_token_wall_mean_ms'] for row in measured),
            steady_token_trial_median_ms=median(row['steady_token_wall_mean_ms'] for row in measured),
            steady_token_trial_means_ms=[row['steady_token_wall_mean_ms'] for row in measured],
            stage_wall_mean_ms={name: mean(row['wall_ms'] for row in steady if row['name'] == name) for name in stage_names},
            ready_before_consume_fraction=mean(row['ready_before_consume'] for row in layers),
            residual_callback_wait_mean_ms_per_token=sum(row['consumer_wait_seconds'] for row in layers) * 1000 / (3 * 14),
            bank_layers_checked=420, predicted_and_actual_tokens_match=True,
            trajectory_bytes=trajectory.stat().st_size,
            retained_gpu_bytes=budget['retained_storage_bytes'],
            live_capture_client_seconds=live['wall_seconds'])
        reports.append(report)
    assert len({row['runner_id'] for row in reports}) == 1
    assert len({row['target_model_id'] for row in reports}) == 1
    assert len({row['pid'] for row in reports}) == 1
    date = datetime.fromtimestamp(min(row['started_unix'] for row in online), timezone(timedelta(hours=8))).strftime('%Y%m%d')
    average = mean(row['steady_token_mean_ms'] for row in summaries.values())
    return dict(schema='pvd-oasis-ready-summary-v1', date_utc_plus_8=date, cases=summaries,
        aggregate_steady_token_mean_ms=average, per_layer_foreground_budget_ms=average / 28,
        two_worker_completion_budget_ms=average * 2 / 28,
        budget_is_capacity_diagnostic_not_latency_guarantee=True,
        same_runner_and_weights=True, live_capture_is_not_performance_comparison=True,
        scope=reports[0]['scope'], timing_policy=reports[0]['timing_policy'],
        initial_graph_and_private_eagle_seed_excluded_from_steady_replay=True,
        replay_callbacks_have_no_network_or_v_work=True,
        all_trial_callbacks_ready_before_consume=all(row['all_layer_callbacks_ready_before_consume']
            for report in reports for row in report['trials'][2:]),
        owned={}, cleanup_errors=[], all_six_gpus_empty=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    assert directory.is_relative_to(ROOT / 'artifacts')
    summary = verify(directory)
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
