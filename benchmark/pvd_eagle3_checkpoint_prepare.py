"""Download and fingerprint one pinned, target-specific EAGLE3 checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from huggingface_hub import snapshot_download
from safetensors import safe_open

REPOSITORY = 'thoughtworks/Qwen2.5-7B-Instruct-Eagle3'
REVISION = 'ff17dda64a036cf5bd7bc56c0ab728325f1c0d0b'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--eagle-source', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / 'checkpoint'
    snapshot_download(REPOSITORY, revision=REVISION, local_dir=destination,
                      allow_patterns=['config.json', 'README.md', 'model.safetensors'])
    weights = destination / 'model.safetensors'
    digest = hashlib.sha256()
    with weights.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    with safe_open(weights, framework='pt', device='cpu') as tensors:
        shapes = {key: tensors.get_slice(key).get_shape() for key in tensors.keys()}
    report = {'repository': REPOSITORY, 'revision': REVISION,
              'weights_sha256': digest.hexdigest(), 'bytes': weights.stat().st_size,
              'config': json.loads((destination / 'config.json').read_text()),
              'tensor_shapes': shapes,
              'eagle_commit': subprocess.check_output(
                  ['git', '-C', str(args.eagle_source), 'rev-parse', 'HEAD'], text=True).strip()}
    (args.output_dir / 'checkpoint.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
