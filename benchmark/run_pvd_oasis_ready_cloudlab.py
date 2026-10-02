"""CloudLab same-runner, captured-trajectory ready-KV diagnostic.

This is one live capture arm followed by untimed-reset local replays, not an
online speedup comparison. Only recorded process groups are terminated.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
HOSTS = {'p': ('130.127.134.34', 'clgpu020.clemson.cloudlab.us', 0),
         'v': ('130.127.134.35', 'clgpu021.clemson.cloudlab.us', 1),
         'd': ('130.127.134.33', 'clgpu019.clemson.cloudlab.us', 2)}
KEY = '/home/loosp/.ssh/cloudlab_pub_wsl'
V_COMPARISONS = FAST_V_COMPARISONS = ('ready-kv',)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tag', required=True)
parser.add_argument('--cases', default='99401,99402')
parser.add_argument('--prepare-only', action='store_true', help='Run only source/tokenizer/CPU setup')
parser.add_argument('--prepared', action='store_true', help='Resume a CPU-prepared tag after rechecking GPU/source identities')
args = parser.parse_args()
if not re.fullmatch(r'[a-z0-9_]+', args.tag):
    raise ValueError('filename-safe fresh tag required')
cases = [int(x) for x in args.cases.split(',')]
if cases != [99401, 99402]:
    raise ValueError('the diagnostic requires the two whitelisted real cases')
args.tokens = 16
args.comparison = 'ready-kv'
OUT = ROOT / 'artifacts' / args.tag
assert OUT.resolve().is_relative_to(ROOT.resolve() / 'artifacts')
if args.prepared:
    if not OUT.is_dir():
        raise ValueError('prepared artifacts are absent')
else:
    OUT.mkdir(parents=True, exist_ok=False)
OWNED, CHECKOUTS, HEADS, ASSETS = {}, {}, {}, {}

def call(role, command, *, data=None, timeout=240, binary=False):
    ip, host, _ = HOSTS[role]
    result = subprocess.run(['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o',
        'ConnectTimeout=15', '-o', 'ServerAliveInterval=10', '-o',
        'ServerAliveCountMax=3', '-o', 'HostKeyAlias=' + host, '-i', KEY,
        'Yizhzhu@' + ip, command], input=data, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{role}: {result.stdout.decode(errors="replace")}\n{result.stderr.decode(errors="replace")}')
    return result.stdout if binary else result.stdout.decode()

def upload(role, path, data):
    call(role, 'cat > ' + shlex.quote(path), data=data)

def collect(role, arm):
    remote = 'v' if role == 'gateway' else role
    path = f'{ASSETS[remote]}/logs/{role}-{args.tag}_{arm}.log'
    compressed = call(remote, 'gzip -c -- ' + shlex.quote(path), binary=True)
    (OUT / f'{arm}_{role}.log').write_bytes(gzip.decompress(compressed))

def stop(role):
    if role not in OWNED:
        return
    remote = 'v' if role == 'gateway' else role
    pid = OWNED[role]
    call(remote, f'kill -TERM -- -{pid} 2>/dev/null || true')
    if role != 'gateway':
        for _ in range(20):
            memory = call(remote, 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits')
            if all(int(x) == 0 for x in memory.split()):
                break
            time.sleep(1)
        else:
            raise RuntimeError('owned service did not drain: ' + role)
    del OWNED[role]
    (OUT / 'owned.json').write_text(json.dumps(OWNED))

def start(role, arm):
    remote = 'v' if role == 'gateway' else role
    env = dict(PVD_CHECKOUT=CHECKOUTS[remote], PVD_EXPECTED_COMMIT=HEADS[remote],
        PVD_RUN_TAG=args.tag + '_' + arm, PVD_CAGRA_EXTEND25=1,
        PVD_CHUNKED_CAGRA_UPLOAD=1, PVD_CAGRA_GROUP_HEADS=4,
        PVD_CAGRA_EXACT_HEAD_SEED=1, PVD_CAGRA_ITOPK_SIZE=2048,
        PVD_GATE_INITIAL_FANIN_ON_INDEX=1, PVD_DIRECT_PD_BOOTSTRAP=0,
        PVD_PREFILL_CHUNK_TOKENS=256, PVD_MODE='oasis',
        PVD_OASIS_CONFIG=ASSETS['d'] + '/' + arm + '_config.json',
        PVD_LOG_DIR=ASSETS[remote] + '/logs',
        TMPDIR=ASSETS[remote] + '/tmp',
        TRITON_CACHE_DIR=ASSETS[remote] + '/cache/triton',
        TORCHINDUCTOR_CACHE_DIR=ASSETS[remote] + '/cache/inductor',
        PVD_CAGRA_KV_EDGE_UPDATE=1, PVD_CAGRA_KV_ROUTING_EDGES=2,
        PVD_CAGRA_SMALL_TAIL_MAX_ROWS=512, PVD_CAGRA_FUSED_PREPARE=1,
        PVD_BATCHED_K_EXTRACTION=1, PVD_FUSED_K_CENTERING=1,
        PVD_PLANNED_TAIL=1, PVD_REUSE_SCORES=1, PVD_EARLY_FINAL_UPDATE=1,
        PVD_CAGRA_EXTEND_CONCURRENCY=1, PVD_CAGRA_NOGIL_EXTEND=1,
        PVD_NEW_TOP16=0, PVD_FUSED_EDGE_WRITE=0, PVD_STREAM_COMPLETION=0,
        PVD_PROFILE_GPU=1, PVD_PROFILE_CHUNK_STAGES=1,
        PVD_PROFILE_V_SEARCH=1, PVD_PROFILE_D_SEARCH_BATCH=1,
        PVD_PROFILE_REFRESH_TIMELINE=1, PVD_GROUPED_EXACT_SEARCH=1,
        PVD_BATCHED_GROUP_SEARCH=1, PVD_SPLIT_POLICY_FILE=ASSETS[remote] + '/split.json')
    env['PVD_PARTIAL_GROUP_SEARCH'] = int(args.comparison in ('v-latency',) + FAST_V_COMPARISONS or (
        args.comparison == 'v-search' and arm.startswith('opt')))
    env['PVD_HOST_CANDIDATES'] = int(args.comparison in FAST_V_COMPARISONS or (args.comparison == 'v-latency' and arm.startswith('opt')))
    env['PVD_NATIVE_POOL'] = int(args.comparison in FAST_V_COMPARISONS or (args.comparison == 'v-latency' and arm.startswith('opt')))
    env['PVD_HOST_QUERY_VALIDATION'] = int(args.comparison == 'v-host-query' and arm.startswith('opt'))
    env['PVD_TRITON_SPARSE_PACKING'] = int(args.comparison == 'v-pack' and arm.startswith('opt'))
    if role == 'd':
        env['PVD_OASIS_REPLAY_CONFIG'] = ASSETS['d'] + '/diagnostic.json'
    command = 'export ' + ' '.join(k + '=' + shlex.quote(str(v)) for k, v in env.items())
    command += '; bash ' + shlex.quote(ASSETS[remote] + '/launcher.sh') + ' ' + role
    (OUT / f'{arm}_{role}.launch').write_text(command)
    result = call(remote, command)
    match = re.search(r'PID (\d+)', result)
    if not match:
        raise RuntimeError('no owned PID: ' + result)
    OWNED[role] = int(match.group(1))
    (OUT / 'owned.json').write_text(json.dumps(OWNED))
    ip = {'p': '10.10.1.1', 'v': '10.10.1.2', 'd': '10.10.1.3', 'gateway': '10.10.1.2'}[role]
    port = {'p': 30002, 'v': 9100, 'd': 30003, 'gateway': 8001}[role]
    for attempt in range(75):
        status = call(remote, f'curl -s --max-time 2 -o /dev/null -w "%{{http_code}}" http://{ip}:{port}/health || true').strip()
        if status == '200':
            (OUT / f'{arm}_{role}.command').write_text(call(remote, f'ps -p {OWNED[role]} -o args='))
            print('healthy', role, arm, OWNED[role], flush=True)
            if role == 'v' and args.comparison == 'v-pack':
                prove_pack_modes(arm)
            return
        if attempt % 15 == 0:
            print('waiting', role, arm, status, flush=True)
        if call(remote, f'ps -p {OWNED[role]} -o stat= || true').strip() in ('', 'Z', 'Zs'):
            collect(role, arm)
            raise RuntimeError('service exited; see ' + str(OUT / f'{arm}_{role}.log'))
        time.sleep(2)
    collect(role, arm)
    raise RuntimeError('health timeout: ' + role)

def probe(arm, warm=False):
    rows = []
    cases = [99991, 99992] if warm else [int(x) for x in args.cases.split(',')]
    for case in cases:
        data = call('v', 'TMPDIR=' + shlex.quote(ASSETS['v'] + '/tmp') +
            ' /proj/llm-course-PG0/Yizhzhu-node1-sglang-pvd/conda-envs/sglang-v100/bin/python ' +
            shlex.quote(ASSETS['v'] + '/probe.py') + f' --case {case} --tokens {args.tokens}', timeout=300)
        row = json.loads(data.strip())
        rows.append(row)
        (OUT / (arm + ('_warmup' if warm else '') + '.json')).write_text(json.dumps(rows, indent=2))
        print('request', arm, 'warm' if warm else 'formal', case,
            row.get('completion_tokens'), round(row['wall_seconds'], 3), flush=True)
        if row['status'] != 200 or row['error'] or row['completion_tokens'] != args.tokens:
            raise RuntimeError('formal Decode failed: ' + data)
    return rows


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def prepare():
    """Verify frozen serving bytes and place diagnostic files only in artifacts."""
    for role, (_, _, node) in HOSTS.items():
        CHECKOUTS[role] = f'/proj/llm-course-PG0/Yizhzhu-node{node}-sglang-pvd/validation/pvd-direct-20260929'
    CHECKOUTS['v'] = '/proj/llm-course-PG0/Yizhzhu-node1-sglang-pvd/validation/pvd-oasis-v-search-20261002'
    CHECKOUTS['d'] = '/proj/llm-course-PG0/Yizhzhu-node2-sglang-pvd/validation/pvd-oasis-alignment-20261002'
    with ThreadPoolExecutor(max_workers=3) as pool:
        memory = dict(zip(HOSTS, pool.map(lambda role: call(role,
            'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'), HOSTS)))
    write_json(OUT / 'initial_gpu_memory.json', memory)
    for role, text in memory.items():
        if len(text.splitlines()) != 2 or any(int(row.split(',')[1].split()[0]) for row in text.splitlines()):
            raise RuntimeError('GPU occupied: ' + role)
    gate = ROOT / 'benchmark/results/pvd_oasis_workers_cloudlab_20261003/gate.tar.gz'
    with tarfile.open(gate, 'r:gz') as archive:
        source_name = 'oasis_delivery_slots_gate01/local_source_hashes.json'
        sources = json.load(archive.extractfile(source_name))
        bundle = archive.extractfile('oasis_delivery_slots_gate01/deployed.tar.gz').read()
    with tarfile.open(fileobj=io.BytesIO(bundle), mode='r:gz') as archive:
        bundle_sources = {item.name: hashlib.sha256(archive.extractfile(item).read()).hexdigest()
                          for item in archive if item.isfile()}
    assert bundle_sources == sources, 'frozen source archive differs from recorded gate'
    (OUT / 'serving_deployed.tar.gz').write_bytes(bundle)
    write_json(OUT / 'serving_deployed_hashes.json', sources)
    serving = [name for name in sources if name.startswith('python/')]
    hashes = {}
    for role in HOSTS:
        ASSETS[role] = CHECKOUTS[role] + '/artifacts/' + args.tag
        directories = [ASSETS[role] + '/' + name for name in
                       ('logs', 'tmp', 'hooks', 'capture', 'cache/triton', 'cache/inductor')]
        call(role, 'test ! -e ' + shlex.quote(ASSETS[role]) + ' && mkdir -p ' +
             ' '.join(shlex.quote(path) for path in directories))
        HEADS[role] = call(role, 'git -C ' + shlex.quote(CHECKOUTS[role]) + ' rev-parse HEAD').strip()
        # D and V have the same complete deployed serving image. P retains its
        # unchanged upload image; record its actual identities without asserting
        # that unused D/V optimizations were installed on P.
        names = serving if role != 'p' else ['python/sglang/srt/disaggregation/pvd/runtime.py',
            'python/sglang/srt/disaggregation/pvd/conn.py', 'python/sglang/srt/disaggregation/pvd/sharding.py']
        text = call(role, 'sha256sum ' + ' '.join(shlex.quote(CHECKOUTS[role] + '/' + name) for name in names))
        hashes[role] = {}
        for line in text.splitlines():
            digest, path = line.split(None, 1)
            name = path.removeprefix(CHECKOUTS[role] + '/')
            if role != 'p' and sources[name] != digest:
                raise RuntimeError('serving source changed: ' + role + ':' + name)
            hashes[role][name] = digest
        launcher = (ROOT / 'test/registered/disaggregation/cloudlab_pvd_new_lease.sh').read_bytes().replace(b'\r\n', b'\n')
        if role == 'd':
            anchor = b'export PYTHONPATH="$checkout/python${PYTHONPATH:+:$PYTHONPATH}"'
            assert launcher.count(anchor) == 1
            # Spawned schedulers inherit both the import hook and diagnostic
            # module directory. P/V/Gateway never load the hook.
            addition = ('\nexport PYTHONPATH=' + shlex.quote(ASSETS['d'] + '/hooks') +
                        ':' + shlex.quote(CHECKOUTS['d'] + '/benchmark') + ':"$PYTHONPATH"').encode()
            launcher = launcher.replace(anchor, anchor + addition)
        (OUT / (role + '_launcher.sh')).write_bytes(launcher)
        upload(role, ASSETS[role] + '/launcher.sh', launcher)
        upload(role, ASSETS[role] + '/split.json', b'{"schema":"pvd-exact16-split-policy-v1","choices":{"2159":{"prefix":2048}}}\n')
    write_json(OUT / 'source_hashes.json', hashes)
    write_json(OUT / 'checkout_heads.json', HEADS)
    diagnostics = {}
    for name, destination in [('pvd_oasis_no_wait_sitecustomize.py', ASSETS['d'] + '/hooks/sitecustomize.py'),
                               ('pvd_oasis_no_wait_replay.py', CHECKOUTS['d'] + '/benchmark/pvd_oasis_no_wait_replay.py')]:
        data = (ROOT / 'benchmark' / name).read_bytes().replace(b'\r\n', b'\n')
        upload('d', destination, data)
        (OUT / name).write_bytes(data)
        diagnostics[name] = hashlib.sha256(data).hexdigest()
    write_json(OUT / 'diagnostic_hashes.json', diagnostics)
    upload('v', ASSETS['v'] + '/probe.py', (ROOT / 'benchmark/pvd_search_decode_probe.py').read_bytes())
    tokenizer_code = '''import hashlib,json
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('/users/Yizhzhu/pvd-models/Qwen2.5-7B-Instruct',trust_remote_code=True)
out={}
for c in (99401,99402):
 ids=t.encode(f'Case {c}. '+'EEFTRITON '*430)
 assert len(ids)==2159,(c,len(ids))
 out[str(c)]={'tokens':len(ids),'sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode('utf-8')).hexdigest()}
print(json.dumps(out))
'''
    upload('d', ASSETS['d'] + '/tokenizer_probe.py', tokenizer_code.encode())
    python = '/proj/llm-course-PG0/Yizhzhu-node2-sglang-pvd/conda-envs/sglang-v100/bin/python'
    command = 'PYTHONDONTWRITEBYTECODE=1 TMPDIR=' + shlex.quote(ASSETS['d'] + '/tmp') + ' ' + python + ' ' + shlex.quote(ASSETS['d'] + '/tokenizer_probe.py')
    identities = json.loads(call('d', command, timeout=90))
    write_json(OUT / 'prompt_identities.json', identities)
    eagle = '/proj/llm-course-PG0/Yizhzhu-node2-sglang-pvd/validation/oasiskv-20261001'
    config = dict(eagle_source=eagle + '/EAGLE', eagle_checkpoint=eagle + '/checkpoint',
        eagle_manifest=eagle + '/checkpoint.json', vector_space='qwen25-7b-pvd',
        capacity=32, max_new=16, top_k=4, workers=2, timeout_seconds=60,
        max_sequence_tokens=2304, max_decode_steps=32, request_budget_bytes=268435456,
        request_scratch_bytes=33554432, bootstrap_budget_bytes=536870912,
        bootstrap_transient_bytes=268435456, overlap=True, reuse_io=False,
        combine_reserve_start=False, reuse_receive_slots=False)
    diagnostic = dict(capture_directory=ASSETS['d'] + '/capture', cases=cases,
        expected_prompt_tokens=2159, output_tokens=16, workers=2,
        warmup_replays=2, measured_replays=3, diagnostic_gpu_budget_bytes=268435456,
        expected_prompt_ids_sha256={case: row['sha256'] for case, row in identities.items()})
    for name, value in [('capture_config.json', config), ('diagnostic.json', diagnostic)]:
        write_json(OUT / name, value)
        upload('d', ASSETS['d'] + '/' + name, (OUT / name).read_bytes())
    write_json(OUT / 'scope.json', dict(schema='pvd-ready-kv-diagnostic-v1',
        comparison=False, arm='capture', cases=cases, tokens=16,
        same_runner=True, captured_real_trajectory=True, serving_source_unchanged=True,
        replay_no_v_traffic=True, includes_live_capture_overhead=True,
        replays_are_not_client_latency=True))


def collect_capture():
    data = call('d', 'tar -czf - -C ' + shlex.quote(ASSETS['d']) + ' capture', binary=True)
    (OUT / 'capture.tar.gz').write_bytes(data)
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = (OUT / member.name).resolve()
            if not path.is_relative_to(OUT.resolve() / 'capture'):
                raise ValueError('unsafe diagnostic archive path')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())


def restore_prepared():
    """Refresh diagnostic-only files before service startup; freeze serving bytes."""
    if (OUT / 'online.json').exists() or (OUT / 'owned.json').exists():
        raise RuntimeError('this prepared tag was already used for a serving run')
    HEADS.update(json.loads((OUT / 'checkout_heads.json').read_text()))
    recorded = json.loads((OUT / 'source_hashes.json').read_text())
    for role, (_, _, node) in HOSTS.items():
        CHECKOUTS[role] = f'/proj/llm-course-PG0/Yizhzhu-node{node}-sglang-pvd/validation/pvd-direct-20260929'
    CHECKOUTS['v'] = '/proj/llm-course-PG0/Yizhzhu-node1-sglang-pvd/validation/pvd-oasis-v-search-20261002'
    CHECKOUTS['d'] = '/proj/llm-course-PG0/Yizhzhu-node2-sglang-pvd/validation/pvd-oasis-alignment-20261002'
    for role in HOSTS:
        ASSETS[role] = CHECKOUTS[role] + '/artifacts/' + args.tag
        head = call(role, 'git -C ' + shlex.quote(CHECKOUTS[role]) + ' rev-parse HEAD').strip()
        assert head == HEADS[role], 'prepared checkout HEAD changed'
        text = call(role, 'sha256sum ' + ' '.join(shlex.quote(CHECKOUTS[role] + '/' + name) for name in recorded[role]))
        actual = {path.removeprefix(CHECKOUTS[role] + '/'): digest
                  for digest, path in (line.split(None, 1) for line in text.splitlines())}
        assert actual == recorded[role], 'prepared serving source changed'
        memory = call(role, 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader')
        assert len(memory.splitlines()) == 2 and all(int(row.split(',')[1].split()[0]) == 0 for row in memory.splitlines()), 'GPU occupied'
    hashes = {}
    for name, destination in [('pvd_oasis_no_wait_sitecustomize.py', ASSETS['d'] + '/hooks/sitecustomize.py'),
                               ('pvd_oasis_no_wait_replay.py', CHECKOUTS['d'] + '/benchmark/pvd_oasis_no_wait_replay.py')]:
        data = (ROOT / 'benchmark' / name).read_bytes().replace(b'\r\n', b'\n')
        upload('d', destination, data)
        (OUT / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    write_json(OUT / 'diagnostic_hashes.json', hashes)


def wait_replays():
    """The last streamed event can precede scheduler-side replay completion."""
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        errors = call('d', 'find ' + shlex.quote(ASSETS['d'] + '/capture') +
                      ' -type f -name error.json -print').splitlines()
        if errors:
            collect_capture()
            raise RuntimeError('diagnostic failed; see captured error.json')
        text = call('d', 'find ' + shlex.quote(ASSETS['d'] + '/capture') +
                    ' -type f -name replay.json -print')
        paths = text.splitlines()
        if len(paths) == len(cases):
            collect_capture()
            for case in cases:
                report = json.loads((OUT / 'capture' / str(case) / 'replay.json').read_text())
                if report.get('schema') != 'pvd-oasis-ready-replay-v1' or report.get('status') != 'passed':
                    raise RuntimeError('diagnostic did not pass: ' + str(case))
            return
        if not call('d', f'ps -p {OWNED["d"]} -o stat= || true').strip():
            raise RuntimeError('D exited before diagnostic completion')
        time.sleep(2)
    collect_capture()
    raise RuntimeError('diagnostic completion timeout; preserve partial artifacts')


def main():
    try:
        restore_prepared() if args.prepared else prepare()
        if args.prepare_only:
            return
        start('v', 'capture')
        start('p', 'capture')
        start('d', 'capture')
        start('gateway', 'capture')
        probe('capture', True)
        collect('d', 'capture')
        warm_log = (OUT / 'capture_d.log').read_text()
        if 'Error in sitecustomize' in warm_log or '"hook_installed": true' not in warm_log:
            raise RuntimeError('spawn diagnostic hook failed; preserve D startup log')
        rows = probe('capture')
        write_json(OUT / 'online.json', {'capture': rows})
        collect('d', 'capture')
        log = (OUT / 'capture_d.log').read_text()
        for case in cases:
            if 'capture_enabled case=' + str(case) not in log:
                raise RuntimeError('formal diagnostic import hook did not capture case ' + str(case))
        wait_replays()
    finally:
        errors = []
        for role in tuple(OWNED):
            try:
                collect(role, 'capture')
            except Exception as error:
                errors.append(f'collect {role}: {error!r}')
        for role in ('gateway', 'p', 'd', 'v'):
            try:
                stop(role)
            except Exception as error:
                errors.append(f'stop {role}: {error!r}')
        checks = {}
        for role in HOSTS:
            try:
                checks[role] = call(role, 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader')
            except Exception as error:
                errors.append(f'final GPU {role}: {error!r}')
        write_json(OUT / 'final_gpu_memory.json', checks)
        write_json(OUT / 'cleanup_errors.json', errors)
        print('final GPU', checks, flush=True)
        if errors:
            raise RuntimeError('cleanup incomplete; inspect cleanup_errors.json and owned.json')


if __name__ == '__main__':
    main()
