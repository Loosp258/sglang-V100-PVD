"""Preserve/verify actual same-runner attention workspace and SDPA graph trials.

Verification is CPU-only. Large trajectories are hashed from the original
nested archive; no target model or synthesized trajectories are used.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
from statistics import mean
import subprocess
import tarfile

from analyze_pvd_oasis_ready import check_trial, verify as verify_ready
from preserve_pvd_oasis_ready import archive_run

ROOT = Path(__file__).resolve().parents[1]
MODES = ['original', 'workspace', 'sdpa_graph']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def summarized(reports, online, trajectory_hashes):
    assert len(reports) == len(online) == 2
    assert len({row['pid'] for row in reports.values()}) == 1
    assert len({row['runner_id'] for row in reports.values()}) == 1
    assert len({row['target_model_id'] for row in reports.values()}) == 1
    cases = {}
    for case, report in reports.items():
        live = next(row for row in online if str(row['case']) == case)
        assert report['status'] == 'passed' and report['schema'] == 'pvd-oasis-ready-replay-v1'
        assert report['trajectory_sha256'] == trajectory_hashes[case]
        assert report['config']['attention_modes'] == MODES
        assert report['budget']['released'] and report['budget']['gpu_refs_cleared']
        assert report['initial_session_close_joined'] and report['formal_private_rows_released']
        assert report['live_native_retired_before_replay']
        assert all(report['budget']['final'][key] == 0
                   for key in ('used_staging_bytes', 'used_inflight', 'reservations'))
        assert report['budget']['retained_storage_bytes'] + report['budget']['replay_scratch_bound_bytes'] <= 256 << 20
        assert list(report['attention_mode_trials']) == list(report['attention_mode_events']) == MODES
        outputs = live['events'][-1]['event']['output_ids']
        modes = {}
        for mode in MODES:
            trials = report['attention_mode_trials'][mode]
            assert len(trials) == 5
            assert [trial['warmup'] for trial in trials] == [True, True, False, False, False]
            event = report['attention_mode_events'][mode]
            assert event['profiled'] is True and all(not trial['profiled'] for trial in trials)
            for trial in trials + [event]:
                check_trial(trial, outputs)
                assert trial['attention_mode'] == mode
                before, after = trial['workspace_before'], trial['workspace_after']
                if mode == 'original':
                    assert before is after is None
                else:
                    assert not before['closed'] and after['closed']
                    assert before['quarantined'] is after['quarantined'] is False
                    assert before['graph'] is after['graph'] is (mode == 'sdpa_graph')
                    assert before['max_bytes'] == 32 << 20
                    assert 0 < before['base_bytes'] + max(before['graph_reserved_bytes'], before['graph_allocated_bytes']) <= before['max_bytes']
                    assert after['graph_shapes'] == []
                    if mode == 'workspace':
                        assert before['graph_shapes'] == []
                    else:
                        assert before['graph_shapes'] and len(before['graph_shapes']) <= 47
            measured = trials[2:]
            stages = [row for trial in measured for row in trial['stages'] if row['step'] > 0]
            layers = [row for trial in measured for row in trial['layers']]
            modes[mode] = dict(
                steady_token_mean_ms=mean(trial['steady_token_wall_mean_ms'] for trial in measured),
                trial_means_ms=[trial['steady_token_wall_mean_ms'] for trial in measured],
                stage_mean_ms={name: mean(row['wall_ms'] for row in stages if row['name'] == name)
                               for name in sorted({row['name'] for row in stages})},
                setup_mean_ms=mean(trial['workspace_setup_ms'] for trial in measured),
                ready_before_consume_fraction=mean(row['ready_before_consume'] for row in layers),
                callback_wait_ms_per_token=sum(row['consumer_wait_seconds'] for row in layers) * 1000 / 42,
                all_logits_features_actual_kv_formal_writes_tokens_exact=True,
                workspace_scope=measured[0]['workspace_before'])
        cases[case] = modes
    aggregate = {mode: dict(steady_token_mean_ms=mean(cases[case][mode]['steady_token_mean_ms'] for case in cases),
                           setup_mean_ms=mean(cases[case][mode]['setup_mean_ms'] for case in cases)) for mode in MODES}
    date = datetime.fromtimestamp(min(row['started_unix'] for row in online), timezone(timedelta(hours=8))).strftime('%Y%m%d')
    return dict(schema='pvd-oasis-attention-replay-summary-v1', date_utc_plus_8=date,
        cases=cases, aggregate=aggregate, pid=reports['99401']['pid'],
        runner_id=reports['99401']['runner_id'], target_model_id=reports['99401']['target_model_id'],
        trajectory_hashes=trajectory_hashes, same_runner_and_live_trajectory=True,
        scope='READY-KV replay; actual SGLang target/EAGLE/ordinary sampler/formal writes; no V/network/background native receive or copy contention',
        timing_policy=reports['99401']['attention_mode_policy'],
        graph_scope='SDPA only; bank waits, publication, EAGLE, sampling and formal writes outside graph',
        setup_excluded_from_steady_but_measured=True, live_capture_is_not_speedup_arm=True)


def verify_archive(folder, *, check_git=False):
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, digest in manifest.items():
        path = (folder / name).resolve()
        assert path.is_relative_to(ROOT / 'benchmark/results')
        data = path.read_bytes()
        normalized = data if name.endswith('.tar.gz') else data.replace(b'\r\n', b'\n')
        assert hashlib.sha256(normalized).hexdigest() == digest, name
        if check_git:
            blob = subprocess.run(['git', 'show', 'HEAD:' + path.relative_to(ROOT).as_posix()],
                cwd=ROOT, check=True, capture_output=True).stdout
            assert (blob if name.endswith('.tar.gz') else blob.replace(b'\r\n', b'\n')) == normalized, name
    with tarfile.open(folder / 'raw.tar.gz', 'r:gz') as archive:
        files = {item.name: archive.extractfile(item).read() for item in archive if item.isfile()}
    def get(name):
        found = [data for path, data in files.items() if path.endswith('/' + name)]
        assert len(found) == 1, name
        return found[0]
    assert json.loads(get('owned.json')) == {} and json.loads(get('cleanup_errors.json')) == []
    for text in json.loads(get('final_gpu_memory.json')).values():
        assert len(text.splitlines()) == 2 and all(int(row.split(',')[1].split()[0]) == 0 for row in text.splitlines())
    hashes = json.loads(get('source_hashes.json'))
    with tarfile.open(fileobj=io.BytesIO(get('serving_deployed.tar.gz')), mode='r:gz') as bundle:
        source = {item.name: hashlib.sha256(bundle.extractfile(item).read()).hexdigest() for item in bundle if item.isfile()}
    assert source == json.loads(get('serving_deployed_hashes.json'))
    for role in ('v', 'd'):
        assert all(source.get(name) == digest for name, digest in hashes[role].items())
    diagnostic = json.loads(get('diagnostic_hashes.json'))
    assert all(hashlib.sha256(get(name)).hexdigest() == digest for name, digest in diagnostic.items())
    reports, trajectory_hashes = {}, {}
    with tarfile.open(fileobj=io.BytesIO(get('capture.tar.gz')), mode='r:gz') as capture:
        for item in capture:
            if not item.isfile():
                continue
            case = Path(item.name).parts[-2]
            assert case in ('99401', '99402')
            if item.name.endswith('/trajectory.pt'):
                digest = hashlib.sha256()
                stream = capture.extractfile(item)
                while block := stream.read(1 << 20):
                    digest.update(block)
                trajectory_hashes[case] = digest.hexdigest()
            elif item.name.endswith('/replay.json'):
                reports[case] = json.load(capture.extractfile(item))
            else:
                raise AssertionError('unexpected captured evidence: ' + item.name)
    log = get('capture_d.log').decode()
    for case, report in reports.items():
        assert 'capture_enabled case=' + case in log
        assert '"pid": ' + str(report['pid']) in log
        assert report == json.loads((folder / ('replay_' + case + '.json')).read_text())
    actual = summarized(reports, json.loads(get('online.json'))['capture'], trajectory_hashes)
    assert actual == json.loads((folder / 'summary.json').read_text())
    return dict(passed=True, files=len(manifest), git_blobs_checked=check_git,
                real_cases=2, measured_trials_per_mode=6,
                trajectory_hashes_checked=True, actual_tokens_logits_kv_bitwise=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--check-git', action='store_true')
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.verify:
        assert directory.is_relative_to(ROOT / 'benchmark/results')
        print(json.dumps(verify_archive(directory, check_git=args.check_git)))
        return
    assert directory.is_relative_to(ROOT / 'artifacts')
    verify_ready(directory)
    reports = {case: json.loads((directory / 'capture' / case / 'replay.json').read_text()) for case in ('99401', '99402')}
    hashes = {case: hashlib.sha256((directory / 'capture' / case / 'trajectory.pt').read_bytes()).hexdigest() for case in reports}
    summary = summarized(reports, json.loads((directory / 'online.json').read_text())['capture'], hashes)
    name = 'pvd_oasis_attention_replay_cloudlab_' + summary['date_utc_plus_8']
    output = ROOT / 'benchmark/results' / name
    output.mkdir(exist_ok=False)
    dump(output / 'summary.json', summary)
    for case, report in reports.items():
        dump(output / ('replay_' + case + '.json'), report)
    archive_run(directory, output / 'raw.tar.gz')
    rows = ['# Oasis 真实轨迹：注意力工作区与纯 SDPA CUDA Graph', '',
        '两个2159-token Prompt、16实际输出。同一进程／SGLang runner／真实 EAGLE 与普通采样器；每个模式的logits、features、actual KV和420次formal KV写入均逐位一致。', '',
        '每模式两次排除warmup、三次wall trial；按case和repetition交替反序。GPU-event另测。所有KV已在GPU，后台仍保留两worker、query clone/event、ticket与handoff；本诊断不含V／网络／receive／CPU备份竞争。', '',
        '| 模式 | D稳态ms/token | 每trial准备ms（不含在稳态） |', '|---|---:|---:|']
    for mode, row in summary['aggregate'].items():
        rows.append(f"| {mode} | {row['steady_token_mean_ms']:.3f} | {row['setup_mean_ms']:.3f} |")
    rows += ['', 'CUDA Graph仅捕获SDPA，保持原变长span和mask，按实际span预先捕获；future等待、Q发布、EAGLE、采样和formal写入均在图外。每次trial新建工作区；显式32MiB上限检查实际graph pool显存增量，关闭先同步再释放。', '',
        '准备／捕获时间计在表中；初始KV与private Prompt seed不在稳态，但完整线上实验收费。本表不是客户端TPOT收益，也不能与其他独立优化相加。默认工作区关闭；没有上线CUDA Graph。', '',
        '完整轨迹、逐层时间戳、launch/config/source和清理证据在raw.tar.gz。预算、原生请求与private formal rows全部退休，六GPU归零。TP1、两个合成文本Prompt、短greedy Decode；更多质量、长Decode、负载、TP2仍开放。', '',
        f'CPU验证：`python benchmark/preserve_pvd_oasis_attention_replay.py benchmark/results/{name} --verify`。', '']
    (ROOT / 'benchmark/results' / (name + '.md')).write_text('\n'.join(rows), encoding='utf-8')
    dump(output / 'manifest.json', {path.name: hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')
         if not path.name.endswith('.tar.gz') else path.read_bytes()).hexdigest() for path in output.iterdir() if path.is_file()})
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest['../' + name + '.md'] = hashlib.sha256((ROOT / 'benchmark/results' / (name + '.md')).read_bytes()).hexdigest()
    dump(output / 'manifest.json', manifest)
    print(json.dumps(dict(output=name, verification=verify_archive(output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
