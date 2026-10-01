"""Summarize matched traces; optionally recompute quality only before EOS."""
import argparse
import json
from pathlib import Path
import statistics
import struct


def post(session, url, path, value):
    header = json.dumps(value, separators=(',', ':')).encode()
    response = session.post(url + path, data=struct.pack('!I', len(header)) + header, timeout=180)
    response.raise_for_status()
    data = response.content
    count, = struct.unpack('!I', data[:4])
    return json.loads(data[4:4 + count])


def summarize(folder, v_url=None):
    report = json.loads((folder / 'report.json').read_text())
    if v_url:
        import requests
        session = requests.Session()
    summaries = []
    for item in report['fixtures']:
        stem = Path(item['name']).stem
        trials = [json.loads((folder / f'{stem}-{i}-{mode}.json').read_text())
                  for i, mode in enumerate(item['order'])]
        free = json.loads((folder / f'{stem}-free.json').read_text())
        valid_steps = free['steps'] if free['steps'] < trials[0]['steps'] else trials[0]['steps']
        quality = None
        if v_url:
            post(session, v_url, '/activate', {'fixture': item['name']})
            quality = post(session, v_url, '/quality', {
                'fixture_sha256': item['activation']['fixture_sha256'],
                'banks': [row for row in trials[0]['banks'] if row[0] < valid_steps]})
        identity = ('actual_tokens', 'sampled_next', 'predicted_tokens', 'banks_sha256',
                    'network_kv_bytes', 'h2d_kv_bytes')
        if any(t[key] != trials[0][key] for t in trials for key in identity):
            raise ValueError('matched timing identity failed')
        stages = {}
        for mode in ('serial', 'overlap'):
            selected = [t for t in trials if t['mode'] == mode]
            traces = [r for t in selected for r in t['transport_trace'] if r['ticket']['step'] > 0]
            stages[mode] = {
                'ms_per_step_runs': [t['ms_per_step'] for t in selected],
                'median_ms_per_step': statistics.median(t['ms_per_step'] for t in selected),
                'median_consumer_wait_ms_per_step': statistics.median(
                    t['consumer_wait_ms'] / t['steps'] for t in selected),
                'ready_layer_fraction': statistics.mean(t['ready_layer_fraction'] for t in selected),
                'mean_layer_stage_ms': {key: statistics.mean(r[key] for r in traces)
                    for key in ('query_d2h_ms', 'rpc_ms', 'v_search_ms', 'v_queue_ms', 'h2d_bank_ms')},
                'mean_draft_one_ms': statistics.mean(r['draft_one_ms'] for t in selected for r in t['step_trace'][:-1])}
        serial, overlap = (stages[m]['median_ms_per_step'] for m in ('serial', 'overlap'))
        summaries.append({'fixture': item['name'], 'prompt_tokens': item['activation']['prompt_tokens'],
            'fixture_sha256': item['activation']['fixture_sha256'], 'graph_build_seconds': item['activation']['build_seconds'],
            'timing_steps': trials[0]['steps'], 'before_eos_steps': valid_steps,
            'serial': stages['serial'], 'overlap': stages['overlap'],
            'decode_time_reduction_fraction': 1 - overlap / serial,
            'saved_seconds_over_16_steps': (serial - overlap) * trials[0]['steps'] / 1000,
            'bootstrap_median_seconds': statistics.median(t['bootstrap_seconds'] for t in trials),
            'feature_seed_bytes': trials[0]['feature_seed_bytes'],
            'decode_network_kv_bytes': trials[0]['network_kv_bytes'],
            'decode_h2d_kv_bytes': trials[0]['h2d_kv_bytes'],
            'working_set_quality_before_eos': quality,
            'teacher_argmax_agreement_16_steps': trials[0]['teacher_next_argmax_agreement'],
            'draft_token_agreement_before_eos': sum(a == b for a, b in zip(
                trials[0]['predicted_tokens'][:valid_steps], trials[0]['actual_tokens'][1:valid_steps + 1])) / valid_steps,
            'free_output_text': free['output_text'], 'free_steps': free['steps'],
            'free_native_selection': True, 'schedule_identity_passed': True})
    return {'fixtures': summaries, 'timed_code_sha256': report['code_sha256'],
        'comparison': report['timing_comparison'], 'quality_note': 'working-set coverage of full-attention teacher Q Top10; not ANN recall or answer accuracy'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--v-url')
    args = parser.parse_args()
    output = summarize(args.results, args.v_url)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
