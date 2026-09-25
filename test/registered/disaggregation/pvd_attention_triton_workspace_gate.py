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
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferBudget,
    TransferCapacityError,
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

    # A real-model-style shadow call executes both kernels under one lease.
    shadow_budget = TransferBudget(1 << 20, 4)
    shadow_workspace = CUDASparseAttentionWorkspace(
        device=device,
        dtype=torch.float16,
        head_dim=128,
        chunk_tokens=64,
        budget=shadow_budget,
        attention_impl="triton_shadow",
        total_kv_heads=4,
        num_query_heads=28,
    )
    shadow_output = torch.empty_like(q)
    shadow_guard = ResourceGuard(
        AttentionBuffers(q, generated_k, generated_v, shadow_output, rows),
        lambda: None,
    )
    shadow_workspace.execute(
        Peer(),
        decode_tokens=len(rows) - 1,
        layer=0,
        mapping=QueryHeadMapping(28, 4),
        resources=shadow_guard,
        scale=1 / math.sqrt(128),
    )
    shadow_snapshot = shadow_workspace.snapshot()
    assert shadow_snapshot["shadow_comparisons"] == 1
    assert shadow_snapshot["shadow_max_abs_error"] <= 0.025
    torch.testing.assert_close(shadow_output, expected, atol=0.025, rtol=0.002)
    shadow_guard.request_release()
    shadow_workspace.close()
    assert shadow_budget.snapshot()["used_staging_bytes"] == 0

    # A deliberately corrupted Triton result must quarantine instead of
    # silently allowing later forwards to use an unverified implementation.
    bad_budget = TransferBudget(1 << 20, 4)
    bad_workspace = CUDASparseAttentionWorkspace(
        device=device,
        dtype=torch.float16,
        head_dim=128,
        chunk_tokens=64,
        budget=bad_budget,
        attention_impl="triton_shadow",
        total_kv_heads=4,
        num_query_heads=28,
    )
    bad_released = []
    bad_guard = ResourceGuard(
        AttentionBuffers(q, generated_k, generated_v, torch.empty_like(q), rows),
        lambda: bad_released.append(True),
    )
    bad_workspace._triton_execute = lambda *args, **kwargs: args[5].zero_()
    try:
        bad_workspace.execute(
            Peer(),
            decode_tokens=len(rows) - 1,
            layer=0,
            mapping=QueryHeadMapping(28, 4),
            resources=bad_guard,
            scale=1 / math.sqrt(128),
        )
    except SparsePayloadError as exc:
        assert "numerical disagreement" in str(exc)
    else:
        raise AssertionError("corrupted Triton output must fail closed")
    bad_guard.request_release()
    assert bad_released == []
    assert (
        bad_workspace.snapshot()["quarantine"] == "Triton/online numerical disagreement"
    )
    assert bad_workspace.snapshot()["resources_held"]

    # A capacity refusal must happen before either device table is allocated.
    tight_budget = TransferBudget(baseline_bytes + table_bytes - 1, 4)
    tight_workspace = CUDASparseAttentionWorkspace(
        device=device,
        dtype=torch.float16,
        head_dim=128,
        chunk_tokens=64,
        budget=tight_budget,
        attention_impl="triton_grouped",
        total_kv_heads=4,
        num_query_heads=28,
    )
    tight_guard = ResourceGuard(
        AttentionBuffers(q, generated_k, generated_v, torch.empty_like(q), rows),
        lambda: None,
    )
    try:
        tight_workspace.execute(
            Peer(),
            decode_tokens=len(rows) - 1,
            layer=0,
            mapping=QueryHeadMapping(28, 4),
            resources=tight_guard,
            scale=1 / math.sqrt(128),
        )
    except TransferCapacityError:
        pass
    else:
        raise AssertionError("table capacity must refuse before allocation")
    assert tight_budget.snapshot()["used_staging_bytes"] == baseline_bytes
    tight_workspace.close()
    assert tight_budget.snapshot()["used_staging_bytes"] == 0

    # Unknown completion is terminal: do not return the table reservation or
    # unpin the output owner, even if the fake peer later drains successfully.
    unknown_budget = TransferBudget(1 << 20, 4)
    unknown_workspace = CUDASparseAttentionWorkspace(
        device=device,
        dtype=torch.float16,
        head_dim=128,
        chunk_tokens=64,
        budget=unknown_budget,
        attention_impl="triton_grouped",
        total_kv_heads=4,
        num_query_heads=28,
    )
    unknown_released = []
    unknown_guard = ResourceGuard(
        AttentionBuffers(q, generated_k, generated_v, torch.empty_like(q), rows),
        lambda: unknown_released.append(True),
    )

    def uncertain():
        unknown_guard.request_release()
        raise RuntimeError("injected CUDA completion uncertainty")

    unknown_workspace._synchronize = uncertain
    try:
        unknown_workspace.execute(
            Peer(),
            decode_tokens=len(rows) - 1,
            layer=0,
            mapping=QueryHeadMapping(28, 4),
            resources=unknown_guard,
            scale=1 / math.sqrt(128),
        )
    except RuntimeError as exc:
        assert "completion uncertainty" in str(exc)
    else:
        raise AssertionError("injected completion uncertainty must propagate")
    assert unknown_released == []
    assert unknown_workspace.snapshot()["quarantine"] == "attention completion unknown"
    assert unknown_workspace.snapshot()["resources_held"]
    assert (
        unknown_budget.snapshot()["used_staging_bytes"] == baseline_bytes + table_bytes
    )
    return {
        "max_abs_error": error,
        "table_bytes": table_bytes,
        "budget_refunded": True,
        "capacity_refused_before_allocation": True,
        "unknown_completion_quarantined": True,
        "shadow_comparisons": shadow_snapshot["shadow_comparisons"],
        "shadow_max_abs_error": shadow_snapshot["shadow_max_abs_error"],
        "mismatch_quarantined": True,
    }


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("real CUDA device required")
    print(json.dumps(run(), sort_keys=True))
