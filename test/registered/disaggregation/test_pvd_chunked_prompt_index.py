"""A proven Prompt prefix is indexed before the full Entry is published."""

import pytest
import torch
from sglang.srt.disaggregation.pvd.index_lifecycle import IndexState
from sglang.srt.disaggregation.pvd.index_search import BruteForceIndexBackend
from sglang.srt.disaggregation.pvd.prompt_index import PromptIndexManager
from test_pvd_prompt_index import SPACE, build_entry, ident


class ExtendableExact(BruteForceIndexBackend):
    """CPU contract stand-in for native CAGRA build/extend."""

    supports_extend = True

    def extend(self, index, additional_vectors):
        return self.build(
            torch.cat((index.handle, additional_vectors), dim=0),
            vector_space=index.vector_space,
            metric=index.metric,
        )


def test_provisional_graph_is_private_until_complete_kv_is_readable():
    _, layout, manifest, packed, shard = build_entry(prompt_tokens=12)
    index = PromptIndexManager(vector_space=SPACE, backend=ExtendableExact())
    transfer_id = manifest.key.transfer_id
    gate = index.open(transfer_id)
    assert shard.page_count >= 2
    for pages in range(1, shard.page_count + 1):
        result = index.progress_chunked(
            transfer_id, packed.tensor,
            layout=layout, manifest=shard,
            complete_pages=pages, stored=False,
        )
        assert result in ("built_prefix", "extended")
        assert gate.state is IndexState.ABSENT
        with pytest.raises(Exception, match="not ready"):
            index.search(ident(transfer_id), queries=torch.ones(1, layout.head_dim), top_k=1)
    index.note_kv_readable(transfer_id)
    assert index.progress_chunked(
        transfer_id, packed.tensor,
        layout=layout, manifest=shard,
        complete_pages=shard.page_count, stored=True,
    ) == "ready"
    record = index._entries[transfer_id]
    key = next(iter(record.vectors))
    query = record.indexes[key].handle[-1:].clone()
    result = index.search(
        ident(transfer_id, layer=key[0], kv_head=key[1]),
        queries=query, top_k=1,
    )
    assert result.selection.token_ids[0] == 11
    assert all(built.count == 12 for built in record.indexes.values())
    index.close(transfer_id)
