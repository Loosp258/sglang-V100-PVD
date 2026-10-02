"""Native received bytes -> serving stage bank -> exact paired workspace SDPA.

Proxy Q/K/V here test CUDA layout/masking equivalence, not real-model quality.
The separate same-runner trajectory replay validates target logits and tokens.
"""
import argparse
import json
from pathlib import Path
import threading
import uuid

import torch

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def consume(lease, manifest, oracle, budget, device, timeout):
    from sglang.srt.disaggregation.pvd.oasis_attention import PairedLayerAttention
    from sglang.srt.disaggregation.pvd.oasis_attention_workspace import PairedAttentionWorkspace
    from sglang.srt.disaggregation.pvd.oasis_pipeline import LayerTicket
    from sglang.srt.disaggregation.pvd.oasis_transport import OasisLayerTransport
    charge = 'native-workspace:' + uuid.uuid4().hex
    budget.reserve(charge, 8 << 20, 0)
    transport = object.__new__(OasisLayerTransport)
    transport.device, transport.lock, transport.trace = device, threading.Lock(), []
    transport.cache = [[{} for _ in range(4)] for _ in range(28)]
    layer = manifest.specs[0].layer
    chosen = [()] * 4
    received = manifest.payload_views(lease.buffer.cpu())
    width = max(len(spec.token_ids) for spec in manifest.specs)
    expected_k = torch.zeros((4, width, 128), dtype=torch.float16)
    expected_v, expected_valid = torch.zeros_like(expected_k), torch.zeros((4, width), dtype=torch.bool)
    expected = manifest.payload_views(oracle)
    for payload, reference in zip(received, expected, strict=True):
        head = payload.spec.kv_head
        chosen[head] = payload.spec.token_ids
        for index, token in enumerate(payload.spec.token_ids):
            transport.cache[layer][head][token] = payload.tensor[:, index].clone()
        count = len(payload.spec.token_ids)
        expected_k[head, :count], expected_v[head, :count] = reference.tensor[0], reference.tensor[1]
        expected_valid[head, :count] = True
        payload.close()
        reference.close()
    ticket = LayerTicket('native-req', 'native-inc', 0, layer)
    context = dict(ticket=ticket, chosen=tuple(chosen), bank=None, retained=[],
                   remote_rows=sum(map(len, chosen)), rpc_seconds=0.0, deliveries=[])
    state = dict(stream=torch.cuda.Stream(device=device))
    bank = transport._stage_install(context, state).value
    assert torch.equal(bank.keys.cpu(), expected_k) and torch.equal(bank.values.cpu(), expected_v)
    assert torch.equal(bank.valid.cpu(), expected_valid)
    generator = torch.Generator().manual_seed(1951 + layer + width)
    q = torch.randn((2, 28, 128), generator=generator).to(device=device, dtype=torch.float16)
    k, v = [torch.randn((2, 4, 128), generator=generator).to(device=device, dtype=torch.float16) for _ in range(2)]
    workspace = PairedAttentionWorkspace(device=device, dtype=torch.float16, q_heads=28,
        kv_heads=4, head_dim=128, max_bank_rows=64, max_history=14, max_bytes=8 << 20)
    for history_length in (0, 3, 14):
        history = [(torch.randn((4, 1, 128), generator=generator).to(device=device, dtype=torch.float16),
                    torch.randn((4, 1, 128), generator=generator).to(device=device, dtype=torch.float16))
                   for _ in range(history_length)]
        original = PairedLayerAttention([history], [bank], q_heads=28, kv_heads=4,
            head_dim=128, feature_layers=()).attention(0, q, k, v)
        actual = PairedLayerAttention([history], [bank], q_heads=28, kv_heads=4,
            head_dim=128, feature_layers=(), workspace=workspace).attention(0, q, k, v)
        if not torch.equal(original, actual):
            raise AssertionError('CUDA workspace changed original paired attention output')
    workspace.close()
    torch.cuda.synchronize(device)
    lease.release_after_proof()
    budget.release(charge)
    return dict(stage_install_bank_exact=True, workspace_attention_bitwise=True,
                tested_history_lengths=[0, 3, 14], workspace=workspace.snapshot())


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
    args.gpu_bank_consumer = consume
    try:
        result = producer.run(args)
        result['producer_mode'], result['mode'] = result['mode'], 'attention_workspace_native'
    except BaseException as error:
        args.output.write_text(json.dumps(dict(status='failed', error=str(error),
            observations=producer._PROGRESS.get('observations', [])), indent=2))
        raise
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], mode=result['mode'], exact_byte_cases=result['exact_byte_cases'])))


if __name__ == '__main__':
    main()
