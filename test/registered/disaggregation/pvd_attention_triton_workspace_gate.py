"""Real-GPU gate for the opt-in Triton workspace lease and budget path."""

import json
import math
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from sglang.srt.disaggregation.pvd.cuda_rank_install import CUDARankInstallParticipant
from sglang.srt.disaggregation.pvd.cuda_sparse_attention import (
    AttentionBuffers,
    CUDASparseAttentionWorkspace,
    scratch_elements,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferBudget,
)


def run() -> dict:
    device = torch.device("cuda:0")
    generator = torch.Generator().manual_seed(6000)
    q = torch.randn(28, 128, generator=generator).to(device, torch.float16)
    generated_k = torch.randn(32, 4, 128, generator=generator).to(device, torch.float16)
    generated_v = torch.randn(32, 4, 128, generator=generator).to(device, torch.float16)
    rows = (19, 3, 17, 11)
    lengths = (65, 64, 32, 65)
    groups = {
        (0, head): (
            SimpleNamespace(token_ids=tuple(range(length))),
            torch.randn(2, length, 128, generator=generator).to(device, torch.float16),
        )
        for head, length in enumerate(lengths)
    }
    expected = torch.empty_like(q)
    for head in range(28):
        kv_head = head // 7
        prompt = groups[(0, kv_head)][1]
        keys = torch.cat((prompt[0], generated_k[list(rows), kv_head]), dim=0).float()
        values = torch.cat((prompt[1], generated_v[list(rows), kv_head]), dim=0).float()
        scores = torch.mv(keys, q[head].float()) / math.sqrt(128)
        expected[head] = torch.mv(values.T, torch.softmax(scores, dim=0))

    class Peer(CUDARankInstallParticipant):
        def __init__(self):
            self._bank = SimpleNamespace(
                device=device,
                dtype=torch.float16,
                head_dim=128,
                prompt_tokens=65,
                snapshot=lambda: {"quarantine": None},
            )
            self.readers = 0

        @contextmanager
        def read(self, decode_tokens):
            assert decode_tokens == len(rows) - 1
            self.readers += 1
            try:
                yield groups
            finally:
                torch.cuda.synchronize(device)
                self.readers -= 1

    budget = TransferBudget(1 << 20, 4)
    workspace = CUDASparseAttentionWorkspace(
        device=device,
        dtype=torch.float16,
        head_dim=128,
        chunk_tokens=64,
        budget=budget,
        attention_impl="triton_grouped",
        max_sequence_tokens=2304,
        total_kv_heads=4,
        num_query_heads=28,
    )
    baseline_bytes = scratch_elements(64, 128) * 4
    assert budget.snapshot()["used_staging_bytes"] == baseline_bytes
    output = torch.empty_like(q)
    released = []
    guard = ResourceGuard(
        AttentionBuffers(q, generated_k, generated_v, output, rows),
        lambda: released.append(True),
    )
    peer = Peer()
    original_sync = workspace._synchronize
    table_bytes = 12 * 4 + 8 * len(rows)

    def fence():
        assert peer.readers == 1
        guard.request_release()
        assert not released
        assert budget.snapshot()["used_staging_bytes"] == baseline_bytes + table_bytes
        original_sync()

    workspace._synchronize = fence
    workspace.execute(
        peer,
        decode_tokens=len(rows) - 1,
        layer=0,
        mapping=QueryHeadMapping(28, 4),
        resources=guard,
        scale=1 / math.sqrt(128),
    )
    assert peer.readers == 0 and released == [True]
    assert budget.snapshot()["used_staging_bytes"] == baseline_bytes
    torch.testing.assert_close(output, expected, atol=0.025, rtol=0.002)
    error = (output.float() - expected.float()).abs().max().item()
    workspace.close()
    assert budget.snapshot()["used_staging_bytes"] == 0
    return {"max_abs_error": error, "table_bytes": table_bytes, "budget_refunded": True}


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("real CUDA device required")
    print(json.dumps(run(), sort_keys=True))
