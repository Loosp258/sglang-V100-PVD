"""Native scatter dispatched on the real bounded stage executor.

Search/installation here are dispatch/proof checks; this gate does not run
CAGRA or serving bank H2D. The fair live trial separately exercises those.
Delivery performs real GPU MR/PUT/ordering/retirement with exact CPU oracles.
"""
import argparse
import json
from pathlib import Path
import threading
import time

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--current-device', default='cuda:0')
    parser.add_argument('--hostname', required=True)
    parser.add_argument('--rails', nargs=2, required=True)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to((ROOT / 'artifacts').resolve())
    import pvd_direct_sparse_batch_native as producer
    from sglang.srt.disaggregation.pvd.oasis_pipeline import LayerReply, LayerTicket
    from sglang.srt.disaggregation.pvd.oasis_stages import BoundedLayerStages
    observed = []
    class StageExecutor:
        def __init__(self, *, max_workers):
            assert max_workers == 2
            self.sequence = 0
            self.resources, self.retired = [], []
            def initialize(stage, index):
                state = dict(stage=stage, index=index, thread=threading.get_ident(),
                             stream=torch.cuda.Stream(device=args.device))
                self.resources.append(state)
                return state
            def retire(stage, state):
                assert stage == state['stage'] and state['thread'] == threading.get_ident()
                state['stream'].synchronize()
                self.retired.append(state)
            def deliver(job, state):
                job['observations'] = job['callback'](*job['arguments'])
                return job
            def install(job, state):
                assert all(row['terminal_success'] and row['exact_bytes'] and row['cleanup_complete']
                           for row in job['observations'])
                return LayerReply(job['ticket'], job['observations'])
            self.stages = BoundedLayerStages((lambda job, state: job, deliver, install),
                initialize=initialize, retire=retire)
        def submit(self, callback, *arguments):
            ticket = LayerTicket('native-stage', 'native-inc', self.sequence, 0)
            self.sequence += 1
            future = self.stages.submit(ticket, ticket, dict(ticket=ticket, callback=callback,
                arguments=arguments), published=time.monotonic(), timeout=args.timeout,
                cleanup=lambda job: None)
            class Result:
                def result(self, timeout=None):
                    return future.result(timeout)[0].value
            return Result()
        def __enter__(self):
            return self
        def __exit__(self, *error):
            self.stages.close()
            assert len(self.resources) == len(self.retired)
            observed.append(dict(snapshot=self.stages.snapshot(), trace=self.stages.trace,
                resources=[{key: value for key, value in state.items() if key != 'stream'}
                           for state in self.resources],
                retired_resources=len(self.retired)))
    args.executor_factory = StageExecutor
    try:
        result = producer.run(args)
        result['producer_mode'] = result['mode']
        result['mode'] = 'bounded_stages_native_scatter'
        result['stage_runs'] = observed
        result['stage_scope'] = 'real bounded dispatch/thread-affine CUDA resources and native scatter delivery; no CAGRA or serving bank install in this isolated gate'
    except BaseException as error:
        args.output.write_text(json.dumps(dict(status='failed', error=str(error),
            observations=producer._PROGRESS.get('observations', []), stage_runs=observed), indent=2))
        raise
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], mode=result['mode'],
        exact_byte_cases=result['exact_byte_cases'], stage_runs=len(observed))))


if __name__ == '__main__':
    main()
