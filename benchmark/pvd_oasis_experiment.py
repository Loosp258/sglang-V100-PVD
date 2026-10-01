"""Isolated P/V/D OasisKV-style paired-forward experiment (trusted LAN only).

Roles: prepare on P; serve on V; decode on D. This explicit runner does not
replace the serving Scheduler or Mooncake wire protocol. JSON headers + FP16
arrays make experiment traffic/counts observable; weights are never transferred.
"""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import statistics
import socket
import struct
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import requests
import torch

from sglang.srt.disaggregation.pvd.oasis_pipeline import LayerLookahead, LayerReply, select_resident
from sglang.srt.disaggregation.pvd.oasis_qwen import PromptBank, QwenPairedDecode, full_prompt_banks


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def pack(meta, array=None):
    header = json.dumps(meta, separators=(",", ":")).encode()
    return struct.pack("!I", len(header)) + header + (b"" if array is None else array.tobytes())


def unpack(data):
    size, = struct.unpack("!I", data[:4])
    if size > len(data) - 4:
        raise ValueError("truncated experiment message")
    return json.loads(data[4:4 + size]), memoryview(data)[4 + size:]


def load_target(path):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16,
        attn_implementation="sdpa", local_files_only=True).cuda().eval()


def greedy(logits, seen, penalty):
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite actual logits")
    scores = logits.float().clone()
    ids = torch.tensor(sorted(set(seen)), device=scores.device)
    selected = scores[:, ids]
    scores[:, ids] = torch.where(selected < 0, selected * penalty, selected / penalty)
    return int(scores.argmax(-1).item())


@torch.inference_mode()
def prepare(args):
    """P makes immutable Prompt KV/features and full-KV teacher trajectories."""
    from transformers import AutoTokenizer
    target = load_target(args.target_model)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    items = [item for item in json.loads(Path(args.questions).read_text())['items']
             if item['split'] == 'calibration']
    selected = [next(item for item in items if item['benchmark'] == benchmark)
                for benchmark in ('gsm8k', 'hotpotqa')]
    if args.include_2155:
        selected.append({'id': 'case40-2155', 'benchmark': 'synthetic',
            'prompt_ids': tokenizer.encode('Case40. ' + 'EEFTRITON ' * 1100)[:2155]})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    evidence = []
    for item in selected:
        tokens = item['prompt_ids']
        inputs = torch.tensor([tokens], device='cuda')
        torch.cuda.synchronize()
        start = time.perf_counter()
        prefix = target(input_ids=inputs, use_cache=True, output_hidden_states=True,
                        logits_to_keep=1)
        banks = full_prompt_banks(prefix.past_key_values)
        features = torch.cat([prefix.hidden_states[i] for i in (2, 14, 25)], dim=-1)
        root = greedy(prefix.logits[:, -1], tokens, args.penalty)
        torch.cuda.synchronize()
        prefill_seconds = time.perf_counter() - start
        # Verify the paired current row against the original HF one-token path.
        check = QwenPairedDecode(target)
        q_check = []
        logits, actual_features = check.step(root, (root + 31) % target.config.vocab_size,
            len(tokens), banks, capture=lambda layer, q: q_check.append(q.clone()))
        hf = target(input_ids=torch.tensor([[root]], device='cuda'),
            past_key_values=prefix.past_key_values, use_cache=True,
            output_hidden_states=True, logits_to_keep=1)
        feature_ref = torch.cat([hf.hidden_states[i] for i in (2, 14, 25)], dim=-1)
        error = float((logits - hf.logits[:, -1]).abs().max())
        feature_error = float((actual_features - feature_ref).abs().max())
        argmax_equal = logits.argmax(-1).item() == hf.logits[:, -1].argmax(-1).item()
        if not argmax_equal or error > 0.25 or feature_error > 0.25:
            raise ValueError(f'paired/HF actual-row mismatch: {error}, {feature_error}')
        # Change only the private candidate; committed actual states must match.
        other = QwenPairedDecode(target)
        other_logits, other_features = other.step(root, (root + 901) % target.config.vocab_size,
            len(tokens), banks)
        if not torch.equal(logits, other_logits) or not torch.equal(actual_features, other_features):
            raise ValueError('private candidate changed actual output')
        if any(len(history) != 1 or history[0][0].shape[1] != 1 for history in check.generated):
            raise ValueError('candidate KV leaked into committed history')
        del check, other, hf, prefix, inputs
        decoder = QwenPairedDecode(target)
        seen, trajectory, queries = list(tokens), [root], []
        for step in range(args.steps + 1):
            current = trajectory[-1]
            seen.append(current)
            layer_queries = []
            logits, _ = decoder.step(current, current, len(tokens) + step, banks,
                capture=lambda layer, q: layer_queries.append(q.cpu()))
            queries.append(torch.stack(layer_queries))
            trajectory.append(greedy(logits, seen, args.penalty))
        fixture = {'format': 1, 'id': item['id'], 'task': item['benchmark'], 'prompt_ids': tokens,
            'target_config': target.config.to_dict(), 'features': features.cpu(),
            'keys': torch.stack([b.keys.cpu() for b in banks]),
            'values': torch.stack([b.values.cpu() for b in banks]),
            'root': root, 'teacher_tokens': trajectory,
            'teacher_q': torch.stack(queries), 'repetition_penalty': args.penalty,
            'teacher_text': tokenizer.decode(trajectory, skip_special_tokens=True),
            'reference_answer': item.get('answer'),
            'prefill_seconds': prefill_seconds,
            'paired_validation': {'actual_logit_max_abs_error': error,
                'actual_feature_max_abs_error': feature_error,
                'argmax_equal': argmax_equal, 'candidate_isolation_bitwise': True,
                'committed_rows_per_step_per_layer': 1}}
        path = output / f"{item['id'].replace(':', '_')}.pt"
        torch.save(fixture, path)
        evidence.append({'id': item['id'], 'task': item['benchmark'], 'prompt_tokens': len(tokens),
            'sha256': digest(path), 'paired_validation': fixture['paired_validation'],
            'prefill_seconds': prefill_seconds, 'teacher_text': fixture['teacher_text']})
        print(json.dumps(evidence[-1]), flush=True)
        del banks, decoder, features, fixture, queries
    save_json(output / 'prepare.json', {'fixtures': evidence, 'steps': args.steps,
        'questions_sha256': digest(args.questions), 'code_sha256': digest(__file__)})


class VectorExperiment:
    def __init__(self, args):
        # cuVS must be loaded before the Torch allocator bridge on this host.
        from sglang.srt.disaggregation.pvd.cagra_backend import CagraIndexBackend
        self.args = args
        self.backend = CagraIndexBackend(device='cuda:0', native_bytes_per_index=256 << 20,
            global_native_cap_bytes=4 << 30, graph_degree=16, intermediate_degree=16,
            itopk_size=2048, exact_head_groups=4)
        self.lock = threading.Lock()
        self.fixture = None
        self.indexes = []
        self.session = None
        self.expected = []
        self.recorded = {}

    def activate(self, name):
        if Path(name).name != name or not name.endswith('.pt'):
            raise ValueError('fixture basename required')
        with self.lock, torch.inference_mode():
            for index in self.indexes:
                self.backend.dispose(index)
            self.indexes.clear()
            self.fixture = torch.load(Path(self.args.fixtures) / name, weights_only=True)
            self.identity = digest(Path(self.args.fixtures) / name)
            layers, heads, rows, dim = self.fixture['keys'].shape
            if heads != 4 or layers != 28 or not 16 < rows <= 2304 or dim != 128:
                raise ValueError('bounded Qwen Prompt fixture required')
            self.keys = self.fixture['keys'].cuda()
            self.values = self.fixture['values'].cuda()
            self.centered = self.keys.float() - self.keys.float().mean(dim=2, keepdim=True)
            self.filters = []
            for head in range(heads):
                bits = np.zeros((heads * rows + 31) // 32, dtype=np.uint32)
                for row in range(head * rows, (head + 1) * rows):
                    bits[row // 32] |= np.uint32(1 << (row % 32))
                self.filters.append(torch.from_numpy(bits).cuda())
            start = time.perf_counter()
            for layer in range(layers):
                self.indexes.append(self.backend.build(self.centered[layer].reshape(-1, dim).contiguous(),
                    vector_space='qwen25-7b-oasis', metric='ip'))
            torch.cuda.synchronize()
            self.build_seconds = time.perf_counter() - start
            self.session = None
            self.recorded.clear()
            return {'fixture_sha256': self.identity, 'build_seconds': self.build_seconds,
                'prompt_tokens': rows, 'layers': layers, 'graphs': layers,
                'native_retained_bytes': self.backend.runtime.global_allocated_bytes()}

    def seed(self, meta):
        with self.lock:
            if self.fixture is None or meta['fixture_sha256'] != self.identity:
                raise ValueError('fixture identity mismatch')
            self.session = (meta['request_id'], meta['incarnation'])
            self.expected = [0] * 28
            self.selection_mode = meta.get('selection_mode', 'live')
            if self.selection_mode not in ('live', 'record', 'replay'):
                raise ValueError('explicit live/record/replay selection required')
            fixture = self.fixture
            data = {'features': fixture['features'], 'prompt_ids': fixture['prompt_ids'],
                'root': fixture['root'], 'teacher_tokens': fixture['teacher_tokens'],
                'prompt_tokens': len(fixture['prompt_ids']), 'fixture_id': fixture['id'],
                'teacher_text': fixture['teacher_text'], 'repetition_penalty': fixture['repetition_penalty'],
                'eos_token_id': fixture['target_config']['eos_token_id'],
                'seed_q': fixture['teacher_q'][0]}
            stream = io.BytesIO()
            torch.save(data, stream)
            return stream.getvalue()

    @torch.inference_mode()
    def retrieve(self, meta, query_data):
        started = time.perf_counter()
        with self.lock:
            acquired = time.perf_counter()
            ticket = meta['ticket']
            if (ticket['request_id'], ticket['incarnation']) != self.session:
                raise ValueError('foreign request incarnation')
            layer, step = ticket['layer'], ticket['step']
            if not 0 <= layer < 28 or step != self.expected[layer]:
                raise ValueError('replayed or gapped layer query')
            capacity, max_new = meta['capacity'], meta['max_new']
            if not 1 <= capacity <= 2048 or not 0 <= max_new <= capacity:
                raise ValueError('bounded resident capacity required')
            query = np.frombuffer(query_data, dtype=np.float16).reshape(28, 128).copy()
            query_sha = hashlib.sha256(query.tobytes()).hexdigest()
            query = torch.from_numpy(query).cuda().float().contiguous()
            if not torch.isfinite(query).all():
                raise ValueError('nonfinite predicted Q')
            rows = self.keys.shape[2]
            search_start = time.perf_counter()
            candidates, chosen, missing = [], [], []
            for head in range(4):
                found, _ = self.backend.search(self.indexes[layer], query[head * 7:(head + 1) * 7],
                    top_k=min(meta['top_k'], rows), bitset=self.filters[head])
                found = found.long().cpu().numpy() - head * rows
                if (found < 0).any() or (found >= rows).any():
                    raise ValueError('invalid CAGRA filtered ID')
                ranked = list(dict.fromkeys(int(t) for t in found.T.flat))
                resident, cached = meta['resident'][head], set(meta['cached'][head])
                if any(not 0 <= t < rows for t in (*resident, *cached)):
                    raise ValueError('out-of-range resident or cache row')
                selected = select_resident(ranked, resident, capacity=capacity, max_new=max_new)
                chosen.append(selected)
                missing.append([t for t in selected if t not in cached])
                candidates.append(ranked)
            # Native search can choose different equal-score IDs even on the
            # same query. A timing replay still executes every native search,
            # then uses the first arm's selection/traffic and verifies exact Q.
            key = (step, layer)
            if self.selection_mode == 'record':
                self.recorded[key] = (query_sha, tuple(chosen))
            elif self.selection_mode == 'replay':
                recorded_sha, recorded_ids = self.recorded[key]
                if query_sha != recorded_sha:
                    raise ValueError(f'fixed-selection replay Q changed at {key}')
                chosen = recorded_ids
                missing = [[t for t in chosen[head] if t not in set(meta['cached'][head])]
                           for head in range(4)]
            search_end = time.perf_counter()
            payload = []
            for head in range(4):
                for token in missing[head]:
                    payload.append(torch.stack((self.keys[layer, head, token], self.values[layer, head, token])))
            array = torch.stack(payload).cpu().numpy() if payload else np.empty((0, 2, 128), np.float16)
            self.expected[layer] += 1
            ended = time.perf_counter()
            return pack({'ticket': ticket, 'selected': chosen, 'missing': missing,
                'candidate_ids': candidates, 'v_queue_ms': (acquired - started) * 1000,
                'v_search_ms': (search_end - search_start) * 1000,
                'v_pack_ms': (ended - search_end) * 1000, 'v_total_ms': (ended - started) * 1000}, array)

    @torch.inference_mode()
    def quality(self, meta):
        """Full-attention teacher-Q coverage, entirely outside the timed path."""
        with self.lock:
            if meta['fixture_sha256'] != self.identity:
                raise ValueError('quality fixture changed')
            samples = []
            for step, layer, banks in meta['banks']:
                q = self.fixture['teacher_q'][step, layer].cuda().float()
                for head in range(4):
                    exact = torch.topk(q[head * 7:(head + 1) * 7] @ self.keys[layer, head].float().T,
                        10, dim=-1).indices.cpu().tolist()
                    selected = set(banks[head])
                    samples.extend(len(set(row) & selected) / 10 for row in exact)
            return {'true_full_q_top10_working_set_coverage': statistics.mean(samples),
                'worst_query_coverage': min(samples), 'query_count': len(samples)}


def serve(args):
    store = VectorExperiment(args)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def do_POST(self):
            try:
                length = int(self.headers['Content-Length'])
                if not 0 < length <= 16 << 20:
                    raise ValueError('bounded request required')
                meta, data = unpack(self.rfile.read(length))
                if self.path == '/activate':
                    response = pack(store.activate(meta['fixture']))
                elif self.path == '/seed':
                    response = store.seed(meta)
                elif self.path == '/layer':
                    response = store.retrieve(meta, data)
                elif self.path == '/quality':
                    response = pack(store.quality(meta))
                else:
                    raise ValueError('unknown endpoint')
                status = 200
            except Exception as exc:
                status, response = 409, pack({'error': str(exc)})
            self.send_response(status)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(json.dumps({'ready': True, 'bind': args.bind, 'port': args.port}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        for index in store.indexes:
            store.backend.dispose(index)


class LayerTransport:
    def __init__(self, url, seed, capacity, max_new, top_k):
        self.url, self.capacity, self.max_new, self.top_k = url, capacity, max_new, top_k
        self.cache = [[{} for _ in range(4)] for _ in range(28)]
        self.cache_rows = 0
        self.max_cache_rows = seed['prompt_tokens'] * 28 * 4
        self.local = threading.local()
        self.trace = []
        self.lock = threading.Lock()

    def session(self):
        if not hasattr(self.local, 'session'):
            self.local.session = requests.Session()
            self.local.stream = torch.cuda.Stream()
        return self.local.session

    def job(self, query, bank, *, bootstrap=False):
        # The worker owns these references until its event/HTTP/H2D is terminal.
        query = query.clone()
        event = torch.cuda.Event()
        event.record()

        @torch.inference_mode()
        def run(ticket):
            session = self.session()
            query_start = time.perf_counter()
            host_query = torch.empty(query.shape, dtype=query.dtype, device='cpu', pin_memory=True)
            with torch.cuda.stream(self.local.stream):
                self.local.stream.wait_event(event)
                host_query.copy_(query, non_blocking=True)
                query.record_stream(self.local.stream)
                query_complete = torch.cuda.Event()
                query_complete.record()
            query_complete.synchronize()
            q = host_query.numpy()
            query_ready = time.perf_counter()
            layer_cache = self.cache[ticket.layer]
            body = pack({'ticket': asdict(ticket), 'resident': bank.ids if bank else [[], [], [], []],
                'cached': [list(c) for c in layer_cache], 'capacity': self.capacity,
                'max_new': self.capacity if bootstrap else self.max_new, 'top_k': self.top_k}, q)
            started = time.perf_counter()
            response = session.post(self.url + '/layer', data=body, timeout=60)
            response.raise_for_status()
            meta, raw = unpack(response.content)
            if meta['ticket'] != asdict(ticket):
                raise ValueError('stale network reply')
            received = time.perf_counter()
            array = np.frombuffer(raw, dtype=np.float16).reshape(-1, 2, 128)
            offset = 0
            for head, missing in enumerate(meta['missing']):
                for token in missing:
                    if token in layer_cache[head]:
                        raise ValueError('duplicate remote KV')
                    layer_cache[head][token] = array[offset].copy()
                    offset += 1
            if offset != len(array):
                raise ValueError('remote KV byte count mismatch')
            with self.lock:
                self.cache_rows += offset
                if self.cache_rows > self.max_cache_rows:
                    raise ValueError('D CPU cache budget exceeded')
            selected = tuple(tuple(ids) for ids in meta['selected'])
            width = max(map(len, selected))
            new_gpu_rows = 0
            copy_start = time.perf_counter()
            with torch.cuda.stream(self.local.stream):
                keys = torch.zeros((4, width, 128), device='cuda', dtype=torch.float16)
                values = torch.zeros_like(keys)
                valid = torch.zeros((4, width), device='cuda', dtype=torch.bool)
                retained = []
                for head, ids in enumerate(selected):
                    old = {} if bank is None else {t: i for i, t in enumerate(bank.ids[head])}
                    hits = [(i, old[t]) for i, t in enumerate(ids) if t in old]
                    misses = [(i, t) for i, t in enumerate(ids) if t not in old]
                    if hits:
                        dst, src = zip(*hits)
                        keys[head, list(dst)] = bank.keys[head, list(src)]
                        values[head, list(dst)] = bank.values[head, list(src)]
                    if misses:
                        rows = np.stack([layer_cache[head][t] for _, t in misses])
                        host = torch.from_numpy(rows).pin_memory()
                        device = host.to('cuda', non_blocking=True)
                        indices = [i for i, _ in misses]
                        keys[head, indices] = device[:, 0]
                        values[head, indices] = device[:, 1]
                        retained.extend((host, device))
                        new_gpu_rows += len(misses)
                    valid[head, :len(ids)] = True
                complete = torch.cuda.Event()
                complete.record()
            # Prove H2D completion before releasing source allocations. The
            # worker waits; target model's CUDA stream remains independent.
            complete.synchronize()
            ended = time.perf_counter()
            trace = {'ticket': asdict(ticket), 'request_bytes': len(body),
                'response_bytes': len(response.content), 'network_kv_rows': offset,
                'network_kv_bytes': offset * 2 * 128 * 2,
                'h2d_rows': new_gpu_rows, 'h2d_kv_bytes': new_gpu_rows * 2 * 128 * 2,
                'cpu_cache_hit_rows': new_gpu_rows - offset,
                'query_d2h_ms': (query_ready - query_start) * 1000,
                'rpc_ms': (received - started) * 1000,
                'h2d_bank_ms': (ended - copy_start) * 1000,
                **{k: v for k, v in meta.items() if k.startswith('v_')}}
            with self.lock:
                self.trace.append(trace)
            return LayerReply(ticket, PromptBank(selected, keys, values, valid, complete))
        return run


def post_json(session, url, path, meta):
    response = session.post(url + path, data=pack(meta), timeout=180)
    response.raise_for_status()
    return unpack(response.content)[0]


def load_eagle(args, target):
    from safetensors.torch import load_file
    import subprocess
    sys.path.insert(0, str(args.eagle_source))
    from eagle.model.cnets import Model
    from eagle.model.configs import EConfig
    metadata = json.loads((Path(args.eagle_checkpoint).parent / 'checkpoint.json').read_text())
    if digest(Path(args.eagle_checkpoint) / 'model.safetensors') != metadata['weights_sha256']:
        raise ValueError('EAGLE3 weight identity changed')
    if subprocess.check_output(['git', '-C', str(args.eagle_source), 'rev-parse', 'HEAD'],
        text=True).strip() != metadata['eagle_commit']:
        raise ValueError('EAGLE author source revision changed')
    config = EConfig.from_pretrained(args.eagle_checkpoint, local_files_only=True)
    config.rope_scaling = metadata['config'].get('rope_scaling')
    config.rope_theta = metadata['config']['rope_theta']
    model = Model(config, bias=False, top_k=1, depth=1, total_tokens=2).half()
    loaded = model.load_state_dict(load_file(Path(args.eagle_checkpoint) / 'model.safetensors'), strict=False)
    if loaded.missing_keys != ['embed_tokens.weight'] or loaded.unexpected_keys:
        raise ValueError(f'EAGLE weight shape mismatch: {loaded}')
    model.embed_tokens = target.model.embed_tokens
    model = model.cuda().eval()
    mapping = model.d2t + torch.arange(config.draft_vocab_size, device='cuda')
    if not torch.equal(mapping, model.t2d.nonzero().flatten()):
        raise ValueError('EAGLE reduced-vocabulary mapping changed')
    return model, mapping


@torch.inference_mode()
def draft_one(draft, mapping, features, inputs, cache, seen, penalty):
    hidden, cache = draft(features, input_ids=inputs, past_key_values=cache, use_cache=True)
    logits = draft.lm_head(draft.norm(hidden[:, -1])).float()
    if not torch.isfinite(logits).all():
        raise ValueError('nonfinite EAGLE output')
    scores = logits.clone()
    # Repetition penalty applies to reduced-vocabulary entries present in history.
    repeated = torch.isin(mapping, torch.tensor(list(set(seen)), device=mapping.device))
    values = scores[:, repeated]
    scores[:, repeated] = torch.where(values < 0, values * penalty, values / penalty)
    return int(mapping[scores.argmax(-1)].item()), cache


@torch.inference_mode()
def run_trial(args, target, draft, mapping, seed, session, fixture_sha, mode, trajectory, steps,
              selection_mode='live'):
    request_id, incarnation = str(uuid.uuid4()), str(uuid.uuid4())
    body = pack({'request_id': request_id, 'incarnation': incarnation,
        'fixture_sha256': fixture_sha, 'selection_mode': selection_mode})
    started = time.perf_counter()
    response = session.post(args.v_url + '/seed', data=body, timeout=60)
    response.raise_for_status()
    seed = torch.load(io.BytesIO(response.content), weights_only=True)
    feature_seed = seed['features'].cuda()
    ids, root, penalty = seed['prompt_ids'], seed['root'], seed['repetition_penalty']
    eos = seed['eos_token_id']
    eos = set(eos if isinstance(eos, list) else [eos])
    seen = list(ids) + [root]
    shifted = torch.tensor([seen[1:]], device='cuda')
    predicted, cache = draft_one(draft, mapping, feature_seed, shifted, None, seen, penalty)
    transport = LayerTransport(args.v_url, seed, args.capacity, args.max_new, args.top_k)
    banks = [None] * 28
    bootstrap = LayerLookahead(request_id, incarnation, layers=28, workers=args.workers)
    for layer in range(28):
        bootstrap.publish(0, layer, transport.job(seed['seed_q'][layer].cuda(), None, bootstrap=True))
    for layer in range(28):
        banks[layer] = bootstrap.consume(0, layer)
    bootstrap.close()
    torch.cuda.synchronize()
    bootstrap_seconds = time.perf_counter() - started
    decoder = QwenPairedDecode(target)
    layer_banks, actual_tokens, sampled_next, predictions, step_trace = [], [root], [], [], []
    pipeline = LayerLookahead(request_id, incarnation, layers=28, workers=args.workers)
    # /layer V counters include the seed query (step zero). The pipeline itself
    # starts at zero; translate its tickets by one only at the transport boundary.
    def callback_for(q, bank):
        job = transport.job(q, bank)
        def run(ticket):
            wire = type(ticket)(ticket.request_id, ticket.incarnation, ticket.step + 1, ticket.layer)
            result = job(wire)
            return LayerReply(ticket, result.value)
        return run

    torch.cuda.synchronize()
    run_start = time.perf_counter()
    try:
        for step in range(steps):
            step_start = time.perf_counter()
            current = actual_tokens[-1]
            predictions.append(predicted)
            deferred = []
            def current_bank(layer):
                if mode == 'overlap' and step:
                    banks[layer] = pipeline.consume(step - 1, layer)
                layer_banks.append((step, layer, [list(t) for t in banks[layer].ids]))
                return banks[layer]

            def publish(layer, q, bank):
                if step + 1 == steps:
                    return
                job = callback_for(q, bank)
                if mode == 'serial':
                    deferred.append((layer, job))
                else:
                    pipeline.publish(step, layer, job)

            logits, features = decoder.step(current, predicted, len(ids) + step,
                current_bank, publish=publish)
            next_token = greedy(logits, seen, penalty)
            sampled_next.append(next_token)
            actual = seed['teacher_tokens'][step + 1] if trajectory == 'teacher' else next_token
            actual_tokens.append(actual)
            seen.append(actual)
            draft_start = time.perf_counter()
            if step + 1 < steps:
                predicted, cache = draft_one(draft, mapping, features,
                    torch.tensor([[actual]], device='cuda'), cache, seen, penalty)
            draft_end = time.perf_counter()
            if mode == 'serial':
                for layer, job in deferred:
                    pipeline.publish(step, layer, job)
                if step + 1 < steps:
                    for layer in range(28):
                        banks[layer] = pipeline.consume(step, layer)
            step_trace.append({'step': step, 'wall_ms': (time.perf_counter() - step_start) * 1000,
                'draft_one_ms': (draft_end - draft_start) * 1000})
            if trajectory == 'free' and actual in eos:
                break
        torch.cuda.synchronize()
    finally:
        pending_errors = pipeline.close()
        if pending_errors:
            raise RuntimeError(f'prefetch close encountered errors: {pending_errors}')
    elapsed = time.perf_counter() - run_start
    actual_steps = len(sampled_next)
    # Nothing below participates in Decode timing.
    quality = post_json(session, args.v_url, '/quality',
        {'fixture_sha256': fixture_sha, 'banks': layer_banks}) if trajectory == 'teacher' else None
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    trace = transport.trace
    measured = [r for r in trace if r['ticket']['step'] > 0]
    bank_sha = hashlib.sha256(json.dumps(layer_banks, separators=(',', ':')).encode()).hexdigest()
    return {'mode': mode, 'trajectory': trajectory, 'steps': actual_steps,
        'selection_mode': selection_mode, 'banks_sha256': bank_sha,
        'decode_seconds': elapsed, 'ms_per_step': elapsed * 1000 / actual_steps,
        'bootstrap_seconds': bootstrap_seconds, 'feature_seed_bytes': feature_seed.numel() * 2,
        'seed_response_bytes': len(response.content), 'actual_tokens': actual_tokens,
        'sampled_next': sampled_next, 'predicted_tokens': predictions,
        'draft_first_token_agreement': sum(a == b for a, b in zip(predictions, actual_tokens[1:])) / actual_steps,
        'teacher_next_argmax_agreement': sum(a == b for a, b in zip(sampled_next,
            seed['teacher_tokens'][1:])) / actual_steps if trajectory == 'teacher' else None,
        'output_text': tokenizer.decode(actual_tokens, skip_special_tokens=True),
        'true_q_quality': quality,
        'consumer_wait_ms': sum(r['consumer_wait_seconds'] for r in pipeline.trace) * 1000,
        'ready_layer_fraction': sum(r['ready_before_consume'] for r in pipeline.trace) / max(1, len(pipeline.trace)),
        'network_kv_bytes': sum(r['network_kv_bytes'] for r in measured),
        'h2d_kv_bytes': sum(r['h2d_kv_bytes'] for r in measured),
        'all_request_bytes': sum(r['request_bytes'] for r in measured),
        'all_response_bytes': sum(r['response_bytes'] for r in measured),
        'cpu_cache_rows': transport.cache_rows, 'cpu_cache_bytes': transport.cache_rows * 512,
        'banks': layer_banks, 'layer_trace': pipeline.trace, 'transport_trace': trace,
        'step_trace': step_trace}


def decode(args):
    target = load_target(args.target_model)
    draft, mapping = load_eagle(args, target)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    report = {'protocol': 'isolated HTTP binary experiment, not Mooncake RDMA',
        'target': 'Qwen2.5-7B-Instruct', 'lookahead': 'one dedicated EAGLE3 token',
        'selector': 'V native centered four-head exact-degree16 CAGRA, itopk2048',
        'resources': 'one V100S for V, one V100S for D; P prepare runs separately',
        'capacity_per_kv_head': args.capacity, 'max_new_per_head_step': args.max_new,
        'top_k_per_q_head': args.top_k, 'workers': args.workers, 'fixtures': [],
        'timing_comparison': 'first arm records selections; later arms replay identical IDs/bytes after native searches and require bitwise identical Q',
        'teacher_timing_eos': 'fixed-length forced trajectory may continue beyond EOS; free generation stops at EOS',
        'code_sha256': digest(__file__)}
    for name in args.fixture:
        activation = post_json(session, args.v_url, '/activate', {'fixture': name})
        trials = []
        # Warm the model, queries, selection, HTTP and H2D in both arms.
        for mode in ('serial', 'overlap'):
            run_trial(args, target, draft, mapping, None, session, activation['fixture_sha256'],
                mode, 'teacher', min(3, args.steps))
        order = ['serial', 'overlap', 'overlap', 'serial']
        for index, mode in enumerate(order):
            trial = run_trial(args, target, draft, mapping, None, session,
                activation['fixture_sha256'], mode, 'teacher', args.steps,
                'record' if index == 0 else 'replay')
            filename = f'{Path(name).stem}-{index}-{mode}.json'
            save_json(output / filename, trial)
            trials.append({k: v for k, v in trial.items() if k not in (
                'layer_trace', 'transport_trace', 'step_trace', 'banks')})
            print(json.dumps({'fixture': name, 'mode': mode,
                'ms_per_step': trial['ms_per_step'], 'consumer_wait_ms': trial['consumer_wait_ms'],
                'network_kv_bytes': trial['network_kv_bytes'], 'quality': trial['true_q_quality']}), flush=True)
        # Same trajectory/banks/outputs must hold independently of schedule.
        for key in ('actual_tokens', 'sampled_next', 'predicted_tokens', 'banks_sha256'):
            if any(trial[key] != trials[0][key] for trial in trials):
                raise ValueError(f'paired scheduling changed {key}')
        free = run_trial(args, target, draft, mapping, None, session, activation['fixture_sha256'],
            'overlap', 'free', args.steps)
        save_json(output / f'{Path(name).stem}-free.json', free)
        report['fixtures'].append({'name': name, 'activation': activation, 'order': order,
            'trials': trials, 'serial_median_ms_per_step': statistics.median(
                t['ms_per_step'] for t in trials if t['mode'] == 'serial'),
            'overlap_median_ms_per_step': statistics.median(
                t['ms_per_step'] for t in trials if t['mode'] == 'overlap'),
            'schedule_identity_passed': True, 'free_output_text': free['output_text'],
            'free_draft_agreement': free['draft_first_token_agreement']})
        save_json(output / 'report.json', report)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='role', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--target-model', required=True)
    p.add_argument('--questions', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--penalty', type=float, default=1.05)
    p.add_argument('--include-2155', action='store_true')
    v = commands.add_parser('serve')
    v.add_argument('--fixtures', required=True)
    v.add_argument('--bind', default='127.0.0.1')
    v.add_argument('--port', type=int, default=38931)
    d = commands.add_parser('decode')
    d.add_argument('--target-model', required=True)
    d.add_argument('--eagle-source', type=Path, required=True)
    d.add_argument('--eagle-checkpoint', type=Path, required=True)
    d.add_argument('--v-url', required=True)
    d.add_argument('--fixture', action='append', required=True)
    d.add_argument('--output', required=True)
    d.add_argument('--steps', type=int, default=16)
    d.add_argument('--capacity', type=int, default=128)
    d.add_argument('--max-new', type=int, default=16)
    d.add_argument('--top-k', type=int, default=16)
    d.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise ValueError('one visible GPU per experiment process required')
    if hasattr(args, 'steps') and not 2 <= args.steps <= 64:
        raise ValueError('bounded 2..64 Decode steps required')
    if args.role == 'serve':
        serve(args)
    elif args.role == 'prepare':
        prepare(args)
    else:
        decode(args)


if __name__ == '__main__':
    main()
