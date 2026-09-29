"""V100S acceptance of the V manager's provisional build, extend and publish."""

# ruff: noqa: I001 -- The CloudLab cuVS wheel must load before Torch.
from cuvs.neighbors import cagra as _cagra  # noqa: F401

import time

import torch

from sglang.srt.disaggregation.pvd.cagra_backend import CagraIndexBackend
from sglang.srt.disaggregation.pvd.kv_packer import PVD_TENSOR_LAYOUT
from sglang.srt.disaggregation.pvd.prompt_index import PromptIndexManager, SearchRequestIdentity
from sglang.srt.disaggregation.pvd.protocol import KVLayoutSignature, KVShardManifest
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


def main():
    rows, dim, page_size = 1024, 128, 4
    k = torch.randn(rows, 1, dim, dtype=torch.float16)
    v = torch.zeros_like(k)
    packed = torch.cat((k.view(torch.uint8).flatten(), v.view(torch.uint8).flatten()))
    layout = KVLayoutSignature(
        model_id="test-model", model_revision="rev", kv_dtype="torch.float16",
        page_size=page_size, num_layers=1, total_kv_heads=2,
        kv_heads_per_rank=1, head_dim=dim, tp_size=2, pp_size=1,
        tensor_layout=PVD_TENSOR_LAYOUT, extra={
            "component_count": 2,
            "component_dtypes": ["torch.float16", "torch.float16"],
            "component_token_shapes": [[1, dim], [1, dim]],
            "component_bytes_per_token": [dim * 2, dim * 2],
        },
    )
    shard = KVShardManifest(
        rank=0, rail="mlx5_0", expected_bytes=packed.numel(),
        page_count=rows // page_size, last_page_valid_tokens=page_size,
        layer_start=0, layer_end=1,
    )
    backend = CagraIndexBackend(
        device="cuda:0", native_bytes_per_index=536870912,
        global_native_cap_bytes=671088640,
        graph_degree=8, intermediate_degree=16, itopk_size=64,
    )
    assert backend.supports_extend
    budget = TransferBudget(staging_bytes=1073741824, max_inflight=1)
    manager = PromptIndexManager(vector_space="test/stream", backend=backend, budget=budget)
    transfer_id = "native-chunked-acceptance"
    manager.open(transfer_id)
    timeline = {}
    for pages in (128, 256):
        started = time.perf_counter()
        outcome = manager.progress_chunked(
            transfer_id, packed, layout=layout, manifest=shard,
            complete_pages=pages, stored=False,
        )
        timeline[f"pages_{pages}"] = [outcome, time.perf_counter() - started]
        assert outcome in ("built_prefix", "extended")
        assert not manager.gate_for(transfer_id).searchable
    manager.note_kv_readable(transfer_id)
    assert manager.progress_chunked(
        transfer_id, packed, layout=layout, manifest=shard,
        complete_pages=256, stored=True,
    ) == "ready"
    query = k[700:701, 0].to("cuda:0", dtype=torch.float32).contiguous()
    result = manager.search(
        SearchRequestIdentity(
            vector_space="test/stream", positional_encoding="rope_applied",
            entry_transfer_id=transfer_id, layer=0, kv_head=0,
        ),
        queries=query, top_k=4,
    )
    assert len(result.selection.token_ids) == 4
    assert manager._entries[transfer_id].indexes[(0, 0)].count == 1024
    manager.close(transfer_id)
    assert backend.runtime.global_allocated_bytes() == 0
    print({"timeline": timeline, "top4": result.selection.token_ids}, flush=True)

    # One native graph for both KV heads. Three proven chunks produce three
    # disjoint ID intervals per head, including two separate extend calls.
    k = torch.randn(rows, 2, dim, dtype=torch.float16)
    v = torch.zeros_like(k)
    packed = torch.cat((k.view(torch.uint8).flatten(), v.view(torch.uint8).flatten()))
    pair_layout = KVLayoutSignature(
        model_id="test-model", model_revision="rev", kv_dtype="torch.float16",
        page_size=page_size, num_layers=1, total_kv_heads=2,
        kv_heads_per_rank=2, head_dim=dim, tp_size=1, pp_size=1,
        tensor_layout=PVD_TENSOR_LAYOUT, extra={
            "component_count": 2,
            "component_dtypes": ["torch.float16", "torch.float16"],
            "component_token_shapes": [[2, dim], [2, dim]],
            "component_bytes_per_token": [dim * 4, dim * 4],
        },
    )
    pair_shard = KVShardManifest(
        rank=0, rail="mlx5_0", expected_bytes=packed.numel(),
        page_count=rows // page_size, last_page_valid_tokens=page_size,
        layer_start=0, layer_end=1,
    )
    pair_manager = PromptIndexManager(
        vector_space="test/stream", backend=backend,
        budget=TransferBudget(staging_bytes=1073741824, max_inflight=1),
        group_heads=2,
    )
    transfer_id = "native-grouped-chunked-acceptance"
    pair_manager.open(transfer_id)
    for pages in (128, 192, 256):
        assert pair_manager.progress_chunked(
            transfer_id, packed, layout=pair_layout, manifest=pair_shard,
            complete_pages=pages, stored=False,
        ) in ("built_prefix", "extended")
    pair_manager.note_kv_readable(transfer_id)
    assert pair_manager.progress_chunked(
        transfer_id, packed, layout=pair_layout, manifest=pair_shard,
        complete_pages=256, stored=True,
    ) == "ready"
    for head in (0, 1):
        query = k[700:701, head].to("cuda:0", dtype=torch.float32).contiguous()
        result = pair_manager.search(
            SearchRequestIdentity(
                vector_space="test/stream", positional_encoding="rope_applied",
                entry_transfer_id=transfer_id, layer=0, kv_head=head,
            ),
            queries=query, top_k=4,
        )
        assert all(0 <= token < rows for token in result.selection.token_ids)
        for token, score in zip(
            result.selection.token_ids, result.selection.scores
        ):
            expected = float(torch.dot(query[0].cpu(), k[token, head].float()))
            assert abs(score - expected) < 0.03
    requests = tuple(
        (
            SearchRequestIdentity(
                vector_space="test/stream", positional_encoding="rope_applied",
                entry_transfer_id=transfer_id, layer=0, kv_head=head,
            ),
            k[700:701, head].to("cuda:0", dtype=torch.float32).contiguous(),
            4,
        )
        for head in (0, 1)
    )
    metadata = {}
    assert len(pair_manager.search_many(requests, metadata=metadata)) == 2
    assert metadata["path"] == "grouped_cagra"
    assert pair_manager._entries[transfer_id].indexes[(0, -1)].count == 2 * rows
    pair_manager.close(transfer_id)
    assert backend.runtime.global_allocated_bytes() == 0
    print({"grouped_multi_extend": "passed"}, flush=True)


if __name__ == "__main__":
    main()
