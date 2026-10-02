"""Preserve verified READY-bank replay evidence and an explicit scope report."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from analyze_pvd_oasis_ready import ROOT, verify


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def archive_run(directory, destination):
    # capture.tar.gz already contains the exact two .pt trajectories and JSON
    # reports. Do not archive its extracted copies a second time.
    with tarfile.open(destination, 'w:gz') as archive:
        for path in sorted(directory.rglob('*')):
            if path.is_symlink() or not path.resolve().is_relative_to(directory):
                raise ValueError('evidence cannot follow links outside the run directory')
            parts = path.relative_to(directory).parts
            if path.is_file() and 'capture' not in parts and not any(part.startswith('download_parts') for part in parts):
                archive.add(path, arcname=directory.name + '/' + path.relative_to(directory).as_posix(), recursive=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--failed-run', action='append', type=Path, default=[])
    args = parser.parse_args()
    directory = args.directory.resolve()
    assert directory.is_relative_to(ROOT / 'artifacts')
    summary = verify(directory)
    name = 'pvd_oasis_ready_kv_cloudlab_' + summary['date_utc_plus_8']
    result = ROOT / 'benchmark/results' / name
    result.mkdir(exist_ok=False)
    dump(result / 'summary.json', summary)
    for case in ('99401', '99402'):
        (result / ('replay_' + case + '.json')).write_bytes((directory / 'capture' / case / 'replay.json').read_bytes())
    archive_run(directory, result / 'raw.tar.gz')
    for index, failed in enumerate(args.failed_run):
        failed = failed.resolve()
        assert failed.is_relative_to(ROOT / 'artifacts') and failed != directory
        archive_run(failed, result / f'failed_{index + 1}.tar.gz')
    rows = summary['cases']
    table = '\n'.join('| ' + case + ' | ' + f"{row['steady_token_mean_ms']:.3f}" + ' | ' +
        f"{row['residual_callback_wait_mean_ms_per_token']:.3f}" + ' | ' +
        f"{100 * row['ready_before_consume_fraction']:.2f}%" + ' |' for case, row in rows.items())
    all_ready = summary['all_trial_callbacks_ready_before_consume']
    readiness = ('All measured callbacks were READY before consumption.' if all_ready else
                 'GPU bank contents were preloaded and proven complete, but some executor callbacks were not READY before consumption. The residual callback scheduling wait is measured below; this is not a strictly zero-wait future baseline.')
    report = f'''# Oasis same-runner preloaded-KV Decode diagnostic

Date: {summary['date_utc_plus_8']} (formal request timestamps, UTC+8).

## Scope and validity

Two real 2159-token Prompts and 16 output tokens were captured through the
unchanged fast-graph P/V/D/Gateway path. Each captures 15 target-paired steps
and 420 actual per-layer PromptBank snapshots. Actual and predicted tokens,
positions, logits, features and committed actual KV match bitwise during
replay. Both cases use the same scheduler process, loaded SGLang target
runner and Qwen2.5-7B weights, real dedicated EAGLE3 proposal closure and
ordinary runner.sample. No teacher-forced replacement of sampler outputs.

The replay retains query clone/event setup, two executor workers, exact layer
tickets and resident handoffs. The background callbacks provide preloaded GPU
banks. Network, V work, native receive, CPU cache backup and background GPU
copies are absent. Original attention/history semantics, 28 formal KV writes
and safety fences remain. Initial graph-gated P→V→D KV and the private EAGLE
seed are required in the live path; they are outside steady replay timing.

{readiness}

Each case has two excluded replay warmups, three unprofiled wall trials,
then a separate CUDA-event trial. Capture, priming, allocator setup, reset,
validation, D2H and save are outside measured intervals. This is a per-case
repeated diagnostic, not ABBA or an online end-to-end gain. Live capture
contains instrumentation overhead and its client time is not a speedup arm.

## Measured steady foreground

Arithmetic means over steps 1–14; step 0 uses the primed proposal.

| Case | D foreground ms/token | Callback wait ms/token | Callbacks ready before consume |
|---|---:|---:|---:|
{table}

Aggregate foreground: **{summary['aggregate_steady_token_mean_ms']:.3f} ms/token**.
With 28 layers this corresponds to **{summary['per_layer_foreground_budget_ms']:.3f} ms/layer**,
or **{summary['two_worker_completion_budget_ms']:.3f} ms/callback** at two workers
for a sustained completion budget. These are capacity diagnostics, not
measured V latency or a lower bound for client TPOT.

Stage wall measurements and a separate GPU-event trial are in the per-case
JSON files. paired_target is an envelope that includes foreground query
publication and existing owner fences. Fence subtotals overlap their parent
envelopes and must not be added again. CUDA-event spans can include launch
gaps and stream idle time; they are not a sum of exclusive kernel execution.

## Ownership and limits

Live native/request/formal owners retire before any replay. Each trial gets
fresh legal private allocator rows and an empty actual-only decoder history.
The diagnostic has its own 256 MiB admission; retained GPU storage and scratch
are checked, references released, and the reservation returns to zero.
All six GPUs returned to 0 MiB; owned process groups and cleanup errors are empty.

The first startup attempt failed because the diagnostic hook named the wrong
result-processor class. It produced no formal performance data; its logs and
source snapshot are preserved separately. The corrected hook adds a test
against the real serving class/method definitions. The successful performance
run later hit a 240-second artifact-download timeout, after both captures and
replays passed. No serving restart or repeated timing was used for recovery:
bounded SSH chunks recovered the original archive and its SHA-256 matched
CloudLab. collection_recovery.json records the exact ranges and final proof.

Two short synthetic-text Prompts, TP1 and greedy sampling only. This does not
prove quality, long Decode, TP2, concurrent requests, native failure recovery,
or an online client gain. Subsequent independent delivery comparisons must
keep the same fast graph, Q path, Top4/cap32/max_new16/workers2 and charge the
initial graph/private seed. Previously measured gains cannot be summed.

## Evidence

- summary.json and replay_99401.json/replay_99402.json: verified measurements.
- raw.tar.gz: launch/config/source identities, exact diagnostic source,
  logs, client events, cleanup and nested capture.tar.gz with hashed trajectories.
- Any failed_* archive preserves unsuccessful attempts separately.

No GitHub push. Step 2 starts after this measurement is committed locally.
'''
    (ROOT / 'benchmark/results' / (name + '.md')).write_text(report, encoding='utf-8', newline='\n')
    manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in result.iterdir() if path.is_file()}
    manifest['../' + name + '.md'] = hashlib.sha256(report.encode()).hexdigest()
    dump(result / 'manifest.json', dict(schema='pvd-ready-evidence-manifest-v1', files=manifest))
    print(result)


if __name__ == '__main__':
    main()
