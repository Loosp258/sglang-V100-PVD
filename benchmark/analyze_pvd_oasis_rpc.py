"""Exact completed RPC counts and worker throughput for ordered live pilots."""
import argparse
import json
from pathlib import Path
import re
from statistics import median

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
a=p.parse_args()
root=a.directory
summary=json.loads((root/'summary.json').read_text())
per_mode={}
requests={}
for arm, rows in summary['requests'].items():
    mode='optimized' if arm.startswith('opt') else 'baseline'
    samples=per_mode.setdefault(mode,dict(search=[],delivery=[],without_miss=[],with_miss=[],workers=[],io=[]))
    raw=(root/f'{arm}_d.log').read_text()
    matches=re.findall(r'PVD D search-batch items=(\d+) query_rows=(\d+) bytes=(\d+) prepare_ms=([\d.]+) encode_ms=([\d.]+) http_ms=([\d.]+) validate_ms=([\d.]+) total_ms=([\d.]+)',raw)
    profiles=[]
    for items,nq,nbytes,prep,enc,http,val,total in matches:
        assert (int(items),int(nq))==(2,14)
        profiles.append(dict(prepare=float(prep),encode=float(enc),http=float(http),validate=float(val),total=float(total)))
    warm=json.loads((root/f'{arm}_warmup.json').read_text())
    offset=sum((r['completion_tokens']-1)*56 for r in warm)
    assert len(profiles)==offset+sum((r['completion_tokens']-1)*56 for r in rows), 'missing/extra/retry D RPCs'
    requests[arm]=[]
    for row in rows:
        n=(row['completion_tokens']-2)*56
        samples['search'].extend(profiles[offset+56:offset+56+n])
        offset+=56+n
        trace=row['trace']
        layers=trace['layers']
        transport=trace['transport'][28:]
        assert len(layers)==len(transport)==n//2
        samples['without_miss'].extend(t['rpc_seconds']*1000 for t in transport if t['remote_rows']==0)
        samples['with_miss'].extend(t['rpc_seconds']*1000 for t in transport if t['remote_rows']>0)
        # Pair by step/layer; row ordering can differ under two workers.
        rpc={(t['step'],t['layer']):t['rpc_seconds'] for t in transport}
        service=sum(t['service_seconds'] for t in layers)
        cfg=json.loads((root/f'{arm}_config.json').read_text())
        workers=cfg['workers']
        span=max(t['ready'] for t in layers)-min(t['worker_start'] for t in layers)
        samples['delivery'].extend((t['service_seconds']-rpc[t['step'],t['layer']])*1000 for t in layers)
        item=dict(case=row['case'],consumed_layers=len(layers),search_rpcs=n,workers=workers,
                  service_seconds=service,worker_window_seconds=span,
                  worker_utilization=service/(workers*span),
                  service_floor_seconds=service/workers,
                  window_above_service_floor_seconds=span-service/workers)
        if 'io' in trace:
            item['io']=trace['io']
            samples['io'].append(trace['io'])
        requests[arm].append(item)
        samples['workers'].append(item)
out={}
for mode, s in per_mode.items():
    out[mode]=dict(search_rpc_count=len(s['search']),
                  search_ms={k:median(t[k] for t in s['search']) for k in s['search'][0]},
                  service_outside_rpc_ms=median(s['delivery']),
                  no_remote_miss_layers=len(s['without_miss']),
                  no_remote_miss_rpc_ms=median(s['without_miss']) if s['without_miss'] else None,
                  remote_miss_layers=len(s['with_miss']),
                  remote_miss_rpc_ms=median(s['with_miss']) if s['with_miss'] else None,
                  worker_utilization=median(t['worker_utilization'] for t in s['workers']),
                  io=s['io'])
result=dict(aggregate=out,requests=requests,
            scope='exact ordered completed RPC blocks; warmups/bootstrap excluded; miss groups differ and are not causal subtraction')
(root/'rpc_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(out,indent=2))
