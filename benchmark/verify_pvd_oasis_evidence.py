"""Check archived run integrity, matched work and per-step admission bounds."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(path):
    # Decompress fully so truncated streams/CRC errors cannot pass silently.
    raw = gzip.decompress(path.read_bytes())
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        return {member.name: tar.extractfile(member).read()
                for member in tar if member.isfile()}


def verify(folder):
    prepare = json.loads((folder / 'prepare.json').read_text())
    assert sha(folder / 'prepare-runtime.py') == prepare['code_sha256']
    for name, source in (('fixed-default-copy.tar.gz', 'default-copy-runtime.py'),
                         ('fixed-independent-copy.tar.gz', 'timed-runtime.py')):
        files = archive(folder / name)
        assert files['decode.exit'].strip() == b'0'
        report = json.loads(files['results/report.json'])
        assert report['code_sha256'] == sha(folder / source)
        assert len(report['fixtures']) == 3
        for fixture in report['fixtures']:
            stem = Path(fixture['name']).stem
            trials = [json.loads(files[f'results/{stem}-{i}-{mode}.json'])
                      for i, mode in enumerate(fixture['order'])]
            assert fixture['schedule_identity_passed']
            for trial in trials:
                bank_sha = hashlib.sha256(json.dumps(trial['banks'], separators=(',', ':')).encode()).hexdigest()
                assert bank_sha == trial['banks_sha256']
                for key in ('actual_tokens', 'sampled_next', 'predicted_tokens',
                            'banks_sha256', 'network_kv_bytes', 'h2d_kv_bytes'):
                    assert trial[key] == trials[0][key], f'{name}: changed {key}'
                for row in trial['transport_trace']:
                    if row['ticket']['step']:
                        assert row['h2d_rows'] <= 4 * 16
                        assert row['network_kv_rows'] <= row['h2d_rows']
            free = json.loads(files[f'results/{stem}-free.json'])
            assert free['selection_mode'] == 'live'
    failed = archive(folder / 'initial-native.tar.gz')
    assert failed['decode.exit'].strip() == b'1'
    assert b'paired scheduling changed predicted_tokens' in failed['decode.log']
    stability = json.loads((folder / 'stability.json').read_text())
    assert stability['changed_head_selections'] == 11
    seed = json.loads((folder / 'seed-check.json').read_text())
    assert seed['passed'] and seed['future_q_storage_excluded'] and seed['free_future_tokens_excluded']
    return {'passed': True, 'matched_fixtures_per_run': 3, 'matched_runs': 2,
        'trials_per_fixture': 4, 'initial_failure_preserved': True,
        'native_selection_changes': 11, 'admission_bound_rows_per_step_layer': 64}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--write-manifest', action='store_true')
    args = parser.parse_args()
    result = verify(args.folder)
    if args.write_manifest:
        manifest = {'timed_commit': 'd0f83e1d8', 'prepare_commit': 'b9fbf6193',
            'default_copy_commit': 'c5f2e8e05', 'native_module_commit': 'e86d6d441',
            'eagle_weights_sha256': 'f77665495c620c17d05d6672444094844866e2a27e545d6837e8b107465ddfd8',
            'eagle_source_commit': 'cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b',
            'verification': result, 'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                for p in sorted(args.folder.iterdir()) if p.is_file() and p.name != 'manifest.json'}}
        (args.folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        manifest = json.loads((args.folder / 'manifest.json').read_text())
        for name, record in manifest['files'].items():
            assert sha(args.folder / name) == record['sha256'], f'changed artifact: {name}'
    print(json.dumps(result))


if __name__ == '__main__':
    main()
