"""Run actual CPU PVD gates with package bootstrap and project-local outputs.

The bootstrap bypasses SGLang frontend initialization, which imports Linux-only
serving dependencies on Windows. Torch, PVD modules and HTTP are unmodified.
CUDA skips remain skips; no GPU/native substitutes are installed here.
"""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('tests', nargs='+', help='project-relative test files')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'artifacts') or output.exists():
        raise ValueError('fresh project artifacts directory required')
    tests = [(ROOT / name).resolve() for name in args.tests]
    if any(not path.is_relative_to(ROOT / 'test') or not path.is_file() for path in tests):
        raise ValueError('existing project tests required')
    output.mkdir(parents=True)
    os.environ['TEMP'] = os.environ['TMP'] = os.environ['TMPDIR'] = str(output)
    sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'artifacts/combined_rpc_agent01/deps')]
    for name in ('sglang', 'sglang.srt', 'sglang.srt.disaggregation', 'sglang.srt.disaggregation.pvd'):
        module = types.ModuleType(name)
        module.__path__ = [str(ROOT / 'python' / name.replace('.', '/'))]
        sys.modules[name] = module
        if '.' in name:
            parent, child = name.rsplit('.', 1)
            setattr(sys.modules[parent], child, module)
    import pytest
    import torch
    sources = list((ROOT / 'python/sglang/srt/disaggregation/pvd').glob('*.py')) + tests
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sources}
    with (output / 'unit.txt').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            status = pytest.main(['-q', '-p', 'no:cacheprovider', '--basetemp', str(output / 'tmp')]
                                 + [str(path) for path in tests])
    result = dict(exit_code=int(status), python=sys.version, torch=torch.__version__,
                  cuda_available=torch.cuda.is_available(), package_bootstrap_only=True,
                  runner_installs_cuda_or_transport_doubles=False,
                  tests_include_explicit_policy_transport_doubles=True,
                  tests=args.tests, source_hashes=hashes)
    (output / 'status.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print((output / 'unit.txt').read_text(encoding='utf-8'))
    raise SystemExit(status)


if __name__ == '__main__':
    main()
