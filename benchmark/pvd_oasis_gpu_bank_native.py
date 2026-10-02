"""Native receive MR retirement before forced-late CPU backup, exact GPU bank.

Reuses the independent scatter fixture only as a native producer. The consumer
is the serving GPU backup owner, with the original destination poisoned after
safe lease release. This bounded local-session gate is separate from live RDMA.
"""
import argparse
import json
from pathlib import Path
import threading

import torch

ROOT = Path(__file__).resolve().parents[1]


def consume_received_rows(lease, manifest, oracle, budget, device, timeout):
    from sglang.srt.disaggregation.pvd.oasis_gpu_backup import OasisGPUBackupPool
    from sglang.srt.disaggregation.pvd.oasis_transport import _HeadCPUCache
    layer = manifest.specs[0].layer
    rows = torch.empty((28, 4, 83, 2, 128), dtype=torch.float16)
    valid = torch.zeros((28, 4, 83), dtype=torch.bool)
    cache = [[_HeadCPUCache(rows[l, h], valid[l, h]) for h in range(4)] for l in range(28)]
    gate, entered = threading.Event(), threading.Event()
    def delay_publish():
        entered.set()
        if not gate.wait(timeout):
            raise TimeoutError('native test backup publication gate expired')
    pool = OasisGPUBackupPool(cache, budget, device=device, request_id='native-req',
        incarnation='native-inc', workers=1, before_publish=delay_publish)
    reader = pool.copy_and_enqueue(manifest, lease.buffer, layer)
    if not entered.wait(timeout):
        raise TimeoutError('native backup did not reach forced publication delay')
    ordered_rows = [(spec.kv_head, token) for spec in manifest.specs for token in spec.token_ids]
    bank = torch.stack([reader.rows[key] for key in ordered_rows])
    completion = torch.cuda.Event()
    completion.record(torch.cuda.current_stream(device))
    reader.release_after_copy(completion)
    if valid.any() or pool.snapshot()['completed']:
        raise AssertionError('CPU cache became readable before its publication proof')
    # The surrounding fixture has already acquired every worker slot and no
    # new acquire happens until both callbacks finish. Poison this free slot.
    lease.release_after_proof()
    lease._slot.buffer.fill_(199)
    torch.cuda.synchronize(device)
    expected_views = manifest.payload_views(oracle)
    expected = torch.cat([payload.tensor.transpose(0, 1) for payload in expected_views])
    for payload in expected_views:
        payload.close()
    if not torch.equal(bank.cpu(), expected):
        raise AssertionError('GPU bank changed when original receive MR was reused')
    before = pool.snapshot()
    gate.set()
    pool.close()
    for index, (head, token) in enumerate(ordered_rows):
        if not torch.equal(cache[layer][head][token], expected[index]):
            raise AssertionError('historical backup bytes differ after MR retirement')
    after = pool.snapshot()
    if not after['closed'] or after['charged_bytes'] or after['retained_owners'] or after['pending_rows']:
        raise AssertionError('GPU/CPU backup owners did not retire')
    return dict(gpu_bank_exact=True, cpu_backup_exact=True,
        original_mr_retired_before_cpu_publication=True,
        cache_valid_before_publication=False, backup_before=before, backup_after=after)


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
    args.gpu_bank_consumer = consume_received_rows
    try:
        result = producer.run(args)
        result['producer_mode'] = result['mode']
        result['mode'] = 'gpu_receive_to_bank_with_async_backup'
    except BaseException as error:
        args.output.write_text(json.dumps(dict(status='failed', error=str(error),
            observations=producer._PROGRESS.get('observations', [])), indent=2))
        raise
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], mode=result['mode'],
        exact_byte_cases=result['exact_byte_cases'], after_close=result['after_close'])))


if __name__ == '__main__':
    main()
