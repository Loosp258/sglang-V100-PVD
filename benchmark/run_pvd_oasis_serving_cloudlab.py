"""WSL: fair full Gateway/P/V/D serialized-paired vs overlap-paired pilot.

Only recorded process groups are terminated. Fresh tagged artifacts preserve
config, raw outputs, argv, service logs and deployed source identities.
"""
import argparse
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
DELIVERY_COMPARISONS = ('v-combine', 'v-slots')
V_COMPARISONS = ('v-search', 'v-latency', 'v-host-query', 'v-io', 'v-pack') + DELIVERY_COMPARISONS
FAST_V_COMPARISONS = ('v-host-query', 'v-io', 'v-pack') + DELIVERY_COMPARISONS
HOSTS = {'p': ('130.127.134.34', 'clgpu020.clemson.cloudlab.us', 0),
         'v': ('130.127.134.35', 'clgpu021.clemson.cloudlab.us', 1),
         'd': ('130.127.134.33', 'clgpu019.clemson.cloudlab.us', 2)}
KEY = '/home/loosp/.ssh/cloudlab_pub_wsl'
parser = argparse.ArgumentParser()
parser.add_argument('--tag', required=True)
parser.add_argument('--arms', default='serial_a,overlap_a,overlap_b,serial_b')
parser.add_argument('--cases', default='99401,99402,99403,99404')
parser.add_argument('--tokens', type=int, default=16)
parser.add_argument('--comparison', choices=('pipeline',) + V_COMPARISONS, default='pipeline')
args = parser.parse_args()
if not re.fullmatch(r'[a-z0-9_]+', args.tag):
    raise ValueError('filename-safe fresh tag required')
OUT = ROOT / 'artifacts' / args.tag
assert OUT.resolve().is_relative_to(ROOT.resolve() / 'artifacts')
if args.comparison in DELIVERY_COMPARISONS:
    assert args.arms.split(',') == ['base_a', 'opt_a', 'opt_b', 'base_b'], 'delivery comparison requires ABBA'
    cases = [int(value) for value in args.cases.split(',')]
    assert len(cases) == len(set(cases)) == 2 and args.tokens == 16, 'delivery comparison requires two cases and 16 tokens'
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


def prove_pack_modes(arm):
    """Read both actual rank stores before Prefill; save proof before asserting."""
    path = OUT / 'pack_modes.json'
    modes = json.loads(path.read_text()) if path.exists() else {}
    modes[arm] = dict(source='GET /internal/health from both V rank endpoints; full responses saved', ranks=[])
    expected = 'triton' if arm.startswith('opt') else 'torch'
    for rank in (0, 1):
        url = f'http://10.10.1.2:{9300 + rank}/internal/health'
        response = call('v', 'curl -fsS --max-time 5 ' + shlex.quote(url))
        filename = f'{arm}_v_rank{rank}_health.json'
        (OUT / filename).write_text(response)
        state = json.loads(response)
        modes[arm]['ranks'].append(dict(rank=rank,
            sparse_pack_kernel=state.get('sparse_pack_kernel'),
            device=state.get('device'), health_file=filename, url=url))
        path.write_text(json.dumps(modes, indent=2))
        assert type(state.get('rank')) is int and state['rank'] == rank, 'wrong V rank health'
        assert state.get('ready') is True, 'V rank not ready'
        assert state.get('device') == f'cuda:{rank}', 'wrong physical V rank device'
        assert state.get('sparse_packing_mode') == 'cuda_synchronous_experimental'
        assert state.get('sparse_pack_kernel') == expected, 'actual V packing kernel differs from arm'


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


def main():
    arm_running = None
    try:
        for role, (_, _, node) in HOSTS.items():
            memory = call(role, 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader')
            (OUT / (role + '_initial_gpu.txt')).write_text(memory)
            if any(int(line.split(',')[1].strip().split()[0]) for line in memory.splitlines()):
                raise RuntimeError('GPU occupied: ' + role)
            CHECKOUTS[role] = f'/proj/llm-course-PG0/Yizhzhu-node{node}-sglang-pvd/validation/pvd-direct-20260929'
        CHECKOUTS['v'] = '/proj/llm-course-PG0/Yizhzhu-node1-sglang-pvd/validation/' + (
            'pvd-oasis-v-search-20261002' if args.comparison in V_COMPARISONS else 'pvd-search-decode-20261002')
        CHECKOUTS['d'] = '/proj/llm-course-PG0/Yizhzhu-node2-sglang-pvd/validation/pvd-oasis-alignment-20261002'
        for role in HOSTS:
            ASSETS[role] = CHECKOUTS[role] + '/artifacts/' + args.tag
            directories = [ASSETS[role] + '/' + name for name in ('logs', 'tmp', 'cache/triton', 'cache/inductor')]
            call(role, 'test ! -e ' + shlex.quote(ASSETS[role]) + ' && mkdir -p ' +
                 ' '.join(shlex.quote(path) for path in directories))
        # This named isolated archive was created by deploy_pvd_oasis_stage;
        # give its actual source tree an immutable launch-gate commit.
        d = CHECKOUTS['d']
        call('d', f'git -C {d} init >/dev/null && git -C {d} add python test benchmark docs AGENTS.md && '
            f'git -C {d} -c user.name=PVD-validation -c user.email=pvd-validation@localhost commit --allow-empty -m Oasis-serving-validation >/dev/null')
        for role in HOSTS:
            HEADS[role] = call(role, 'git -C ' + CHECKOUTS[role] + ' rev-parse HEAD').strip()
            launcher = (ROOT / 'test/registered/disaggregation/cloudlab_pvd_new_lease.sh').read_bytes().replace(b'\r\n', b'\n')
            upload(role, ASSETS[role] + '/launcher.sh', launcher)
            upload(role, ASSETS[role] + '/split.json', b'{"schema":"pvd-exact16-split-policy-v1","choices":{"2159":{"prefix":2048}}}\n')
        sources = {}
        for role, relative in (('d', ['python/sglang/srt/server_args.py', 'python/sglang/srt/models/qwen2.py',
            'python/sglang/srt/managers/scheduler.py', 'python/sglang/srt/managers/scheduler_components/batch_result_processor.py',
            'python/sglang/srt/disaggregation/decode.py'] +
            ['python/sglang/srt/disaggregation/pvd/' + p.name for p in
             (ROOT / 'python/sglang/srt/disaggregation/pvd').glob('oasis*.py')]),
            ('v', ['python/sglang/srt/disaggregation/pvd/' + n for n in
             ('cagra_backend.py', 'prompt_index.py', 'control_server.py', 'vector_store.py', 'server.py',
              'cagra_kv_update.py', 'cagra_kv_prepare.py', 'cagra_search_batch.py', 'index_search.py') +
             (('sparse_copy.py', 'sparse_pack_plan.py', 'sparse_payload.py', 'sparse_delivery.py',
               'sparse_receiver.py', 'triton_sparse_pack.py') if args.comparison == 'v-pack' else ())])):
            if args.comparison in DELIVERY_COMPARISONS:
                relative += ['python/sglang/srt/disaggregation/pvd/' + name for name in
                             ('client.py', 'sparse_receiver.py', 'cuda_sparse_receiver.py',
                              'protocol.py', 'control_server.py', 'vector_store.py',
                              'mooncake_engine.py', 'transfer_engine.py', 'oasis_receive_slots.py')]
                relative = list(dict.fromkeys(relative))
            output = call(role, 'sha256sum ' + ' '.join(CHECKOUTS[role] + '/' + p for p in relative))
            sources[role] = {}
            for line in output.splitlines():
                digest, remote_path = line.split(None, 1)
                local = remote_path.removeprefix(CHECKOUTS[role] + '/')
                expected = hashlib.sha256((ROOT / local).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                if digest != expected:
                    raise RuntimeError('deployed source differs: ' + remote_path)
                sources[role][local] = digest
        (OUT / 'source_hashes.json').write_text(json.dumps(sources, indent=2))
        (OUT / 'checkout_heads.json').write_text(json.dumps(HEADS, indent=2))
        (OUT / 'comparison.json').write_text(json.dumps(dict(comparison=args.comparison,
            arms=args.arms.split(','), cases=args.cases, tokens=args.tokens), indent=2))
        upload('v', ASSETS['v'] + '/probe.py', (ROOT / 'benchmark/pvd_search_decode_probe.py').read_bytes())
        if args.comparison == 'pipeline':
            start('v', 'shared'); start('p', 'shared')
        eagle_root = '/proj/llm-course-PG0/Yizhzhu-node2-sglang-pvd/validation/oasiskv-20261001'
        results = {}
        for arm in args.arms.split(','):
            arm_running = arm
            config = dict(eagle_source=eagle_root + '/EAGLE', eagle_checkpoint=eagle_root + '/checkpoint',
                eagle_manifest=eagle_root + '/checkpoint.json', vector_space='qwen25-7b-pvd',
                capacity=32, max_new=16, top_k=4, workers=2, timeout_seconds=60,
                max_sequence_tokens=2304, max_decode_steps=32, request_budget_bytes=268435456,
                request_scratch_bytes=33554432, bootstrap_budget_bytes=536870912,
                bootstrap_transient_bytes=268435456,
                overlap=args.comparison in V_COMPARISONS or arm.startswith('overlap'))
            if args.comparison == 'v-io':
                config['reuse_io'] = arm.startswith('opt')
            elif args.comparison == 'v-pack':
                config['reuse_io'] = False
            elif args.comparison in DELIVERY_COMPARISONS:
                config.update(reuse_io=False,
                    combine_reserve_start=args.comparison == 'v-combine' and arm.startswith('opt'),
                    reuse_receive_slots=args.comparison == 'v-slots' and arm.startswith('opt'))
            encoded = json.dumps(config, indent=2).encode()
            (OUT / (arm + '_config.json')).write_bytes(encoded)
            upload('d', ASSETS['d'] + '/' + arm + '_config.json', encoded)
            if args.comparison in V_COMPARISONS:
                start('v', arm); start('p', arm)
            start('d', arm); start('gateway', arm)
            probe(arm, True)
            results[arm] = probe(arm)
            (OUT / 'online.json').write_text(json.dumps(results, indent=2))
            collect('d', arm); collect('gateway', arm)
            stop('gateway'); stop('d')
            if args.comparison in V_COMPARISONS:
                collect('v', arm); collect('p', arm)
                stop('p'); stop('v')
        if args.comparison == 'pipeline':
            collect('v', 'shared'); collect('p', 'shared')
    finally:
        cleanup_errors = []
        for role in tuple(OWNED):
            try:
                collect(role, 'shared' if args.comparison == 'pipeline' and role in ('v', 'p') else arm_running)
            except Exception as error:
                cleanup_errors.append(f'collect {role}: {error!r}')
        for role in ('gateway', 'p', 'd', 'v'):
            try:
                stop(role)
            except Exception as error:
                cleanup_errors.append(f'stop {role}: {error!r}')
        checks = {}
        for role in HOSTS:
            try:
                checks[role] = call(role, 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader')
            except Exception as error:
                cleanup_errors.append(f'final GPU {role}: {error!r}')
        (OUT / 'final_gpu_memory.json').write_text(json.dumps(checks, indent=2))
        (OUT / 'cleanup_errors.json').write_text(json.dumps(cleanup_errors, indent=2))
        print('final GPU', checks, flush=True)
        if cleanup_errors:
            raise RuntimeError('cleanup incomplete; inspect cleanup_errors.json and owned.json')


if __name__ == '__main__':
    main()
