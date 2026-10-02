"""Token-event timings for a real Gateway/P/V/D request."""
import argparse
import hashlib
import json
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--case', type=int, required=True)
p.add_argument('--tokens', type=int, default=16)
p.add_argument('--url', default='http://10.10.1.2:8001/generate')
a = p.parse_args()
prompt = f'Case {a.case}. ' + 'EEFTRITON ' * 430
payload = json.dumps(dict(text=prompt, sampling_params=dict(temperature=0,
    max_new_tokens=a.tokens, ignore_eos=True), stream=True)).encode()
request = urllib.request.Request(a.url, data=payload, headers={'Content-Type':'application/json'})
started_unix, started = time.time(), time.perf_counter()
events = []
with urllib.request.urlopen(request, timeout=180) as response:
    status = response.status
    for raw in response:
        if raw.startswith(b'data:') and raw.strip() != b'data: [DONE]':
            events.append(dict(seconds=time.perf_counter()-started, event=json.loads(raw[5:])))
wall = time.perf_counter()-started
final = events[-1]['event'] if events else {}
print(json.dumps(dict(case=a.case, tokens=a.tokens, started_unix=started_unix,
    wall_seconds=wall, first_event_seconds=events[0]['seconds'] if events else None,
    status=status, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    output_sha256=hashlib.sha256(final.get('text','').encode()).hexdigest(),
    completion_tokens=final.get('meta_info',{}).get('completion_tokens'),
    error=final.get('error'), events=events)), flush=True)
