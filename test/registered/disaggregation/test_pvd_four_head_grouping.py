"""Four-head chunk layout and full-build fallback use exact per-head IDs."""

import pytest
import torch

from sglang.srt.disaggregation.pvd.index_search import BruteForceIndexBackend
from sglang.srt.disaggregation.pvd.prompt_index import PromptIndexManager
from test_pvd_prompt_index import SPACE, ident
from test_pvd_prompt_vectors import FakePool, pack_shard, storage_layout


class FilteredExact(BruteForceIndexBackend):
    name = "cagra"
    supports_extend = True

    def extend(self, index, additional_vectors):
        return self.build(
            torch.cat((index.handle, additional_vectors)),
            vector_space=index.vector_space, metric=index.metric,
        )

    def search(self, index, queries, *, top_k, bitset=None):
        if bitset is None:
            return super().search(index, queries, top_k=top_k)
        scores = queries.float() @ index.handle.T
        words = bitset.cpu().tolist()
        allowed = torch.tensor(
            [bool(int(words[row // 32]) & (1 << (row % 32)))
             for row in range(index.count)],
            dtype=torch.bool,
        )
        scores[:, ~allowed] = -torch.inf
        values, rows = scores.topk(top_k, dim=1)
        return rows, values


@pytest.mark.parametrize("rank", (0, 1))
@pytest.mark.parametrize("incremental", (False, True))
def test_four_head_grouping_maps_two_layers_and_two_heads(rank, incremental):
    pool = FakePool(layers=4)
    layout = storage_layout(pool)
    packed, shard, _ = pack_shard(pool, layout, rank=rank, prompt_tokens=12)
    manager = PromptIndexManager(
        vector_space=SPACE, backend=FilteredExact(), metric="ip", group_heads=4,
    )
    transfer_id = f"four-head-{rank}-{incremental}"
    manager.open(transfer_id)
    if incremental:
        for pages in (1, 2, 3):
            assert manager.progress_chunked(
                transfer_id, packed.tensor, layout=layout, manifest=shard,
                complete_pages=pages, stored=False,
            ) in ("built_prefix", "extended")
        manager.note_kv_readable(transfer_id)
        assert manager.progress_chunked(
            transfer_id, packed.tensor, layout=layout, manifest=shard,
            complete_pages=3, stored=True,
        ) == "ready"
    else:
        manager.note_kv_readable(transfer_id)
        assert manager.build(
            transfer_id, packed.tensor, layout=layout, manifest=shard,
        )
    record = manager._entries[transfer_id]
    assert len(record.indexes) == 2
    assert record.group_boundaries == ([0, 4, 8, 12] if incremental else [0, 12])
    for layer in range(4):
        for head in (rank * 2, rank * 2 + 1):
            query = pool.k_buffer[layer][5, head].float().reshape(1, -1).clone()
            result = manager.search(
                ident(transfer_id, layer=layer, kv_head=head),
                queries=query, top_k=3,
            )
            assert all(0 <= token < 12 for token in result.selection.token_ids)
            for token, score in zip(
                result.selection.token_ids, result.selection.scores
            ):
                original = pool.k_buffer[layer][token, head].float()
                assert abs(score - float(torch.dot(query[0], original))) < 1e-3
    manager.close(transfer_id)
