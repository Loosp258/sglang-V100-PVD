"""Synthetic fixture exercises the same HTTP/oracle helper as the real smoke."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import torch
from pvd_search_roundtrip import verify_exact_roundtrip
from sglang.srt.disaggregation.pvd.prediction import (
    CommittedPrefix,
    DraftConfig,
    FakeDraftProvider,
    FakeTargetProbe,
    PredictionPipeline,
    ProbeConfig,
)


def test_two_shard_http_gqa_mapping_padding_versions_and_score_oracle():
    rng = torch.Generator().manual_seed(745)
    keys = [torch.randn(6, 2, 8, generator=rng) for _ in range(2)]
    values = [torch.randn(6, 2, 8, generator=rng) for _ in range(2)]
    for tensor in keys + values:
        tensor[-1].fill_(1e6)
    pool = SimpleNamespace(k_buffer=keys, v_buffer=values)
    prefix = CommittedPrefix("req", (1, 2, 3, 4, 5), 0, "v1")
    config = ProbeConfig("target", (0, 1), head_count=4)
    draft = DraftConfig("draft", predict_tokens=2)

    class NonzeroProbe(FakeTargetProbe):
        def capture(self, prefix, prediction):
            generator = torch.Generator().manual_seed(81)
            return tuple(
                replace(q, vectors=torch.randn(q.vectors.shape, generator=generator))
                for q in super().capture(prefix, prediction)
            )

    pipeline = PredictionPipeline(
        FakeDraftProvider(draft, tokens=(6, 7)),
        NonzeroProbe(config, head_dim=8),
        draft,
        config,
    )
    result = asyncio.run(verify_exact_roundtrip(pool, prefix, pipeline))
    assert result["layer_head_results_checked"] == 8
    assert result["negative_checks"] == 14
    assert result["sparse_kv_payloads_checked"] == 8
    assert result["cpu_union_groups_installed"] == 4
