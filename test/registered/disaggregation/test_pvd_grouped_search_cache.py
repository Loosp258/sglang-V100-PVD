"""Entry lease, cache budget and logical results using a CPU native double."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from test_pvd_cagra_kv_update import Runtime
from sglang.srt.disaggregation.pvd.cagra_kv_update import CagraKVUpdateBackend
from sglang.srt.disaggregation.pvd.index_search import IndexCompletionUnknown, IndexSearchError
from sglang.srt.disaggregation.pvd.prompt_index import PromptIndexManager, SearchRequestIdentity
from sglang.srt.disaggregation.pvd.protocol import KVLayoutSignature, KVShardManifest
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


class NativeDouble(Runtime):
    def __init__(self):
        self.cp = SimpleNamespace(from_dlpack=lambda x: x)
        self.filters = SimpleNamespace(from_bitset=lambda x: x)
        self.cagra = SimpleNamespace(SearchParams=lambda **kw: kw, search=self.submit)
        self.submits = 0
        self.on_submit = None
        self.fail_sync = False

    def scope(self, owner):
        return nullcontext()

    def import_owned_graph(self, owner, data, graph):
        super().import_owned_graph(owner, data, graph)
        owner.native.owner = owner

    def submit(self, params, native, q, k, *, neighbors, distances, filter, **kw):
        self.submits += 1
        if self.on_submit:
            hook, self.on_submit = self.on_submit, None
            hook()
        ids, scores = self.search(native.owner, q, top_k=k, itopk_size=params['itopk_size'], bitset=filter)
        neighbors.copy_(ids)
        distances.copy_(scores)

    def synchronize(self):
        if self.fail_sync:
            raise RuntimeError('unknown fence')
        super().synchronize()


def setup(rank=0, *, partial=False):
    rt = NativeDouble()
    b = CagraKVUpdateBackend(device='cpu', native_bytes_per_index=1 << 20,
        graph_degree=16, intermediate_degree=16, itopk_size=32, exact_head_groups=4,
        routing_edges=2, _runtime=rt)
    budget = TransferBudget(128 << 20, 1)
    manager = PromptIndexManager(vector_space='test', backend=b, budget=budget,
        group_heads=4, batched_group_search=True, partial_group_search=partial)
    k = torch.randn(2, 32, 2, 8, generator=torch.Generator().manual_seed(13)).half()
    packed = torch.cat((k.flatten(), torch.zeros_like(k).flatten())).view(torch.uint8)
    layout = KVLayoutSignature(model_id='test', model_revision='rev', kv_dtype='float16',
        page_size=1, num_layers=2, total_kv_heads=4, kv_heads_per_rank=2,
        head_dim=8, tp_size=2, pp_size=1, tensor_layout='component-major', extra={
            'component_count':4, 'component_dtypes':['float16']*4,
            'component_token_shapes':[[2,8]]*4, 'component_bytes_per_token':[32]*4})
    shard = KVShardManifest(rank=rank, rail='test', expected_bytes=packed.numel(),
        page_count=32, last_page_valid_tokens=1, layer_start=0, layer_end=2)
    manager.open('entry')
    manager.progress_chunked('entry', packed, layout=layout, manifest=shard, complete_pages=24, stored=False)
    manager.note_kv_readable('entry')
    assert manager.progress_chunked('entry', packed, layout=layout, manifest=shard,
        complete_pages=32, stored=True) == 'ready'
    requests = tuple((SearchRequestIdentity(vector_space='test', positional_encoding='rope_applied',
        entry_transfer_id='entry', layer=layer, kv_head=head + rank * 2),
        k[layer, 27:29, head].float().contiguous(), 4) for layer in range(2) for head in range(2))
    return manager, rt, requests


@pytest.mark.parametrize('rank', [0, 1])
def test_matches_baseline_reuses_workspace_and_refunds(rank):
    m, rt, requests = setup(rank)
    m.batched_group_search = False
    baseline = m.search_many(requests)
    before = m.budget.snapshot()['used_staging_bytes']
    m.batched_group_search = True
    meta = {}
    result = m.search_many(requests[::-1], metadata=meta)
    assert tuple(r.selection for r in result[::-1]) == tuple(r.selection for r in baseline)
    assert meta['path'] == 'grouped_cagra_batched' and rt.submits == 4
    charged = m.budget.snapshot()['used_staging_bytes']
    ws = next(iter(m._entries['entry'].search_workspaces.values()))[0]
    assert charged - before == ws.retained_bytes
    assert m.search_many(requests) == baseline
    assert rt.submits == 8 and m.budget.snapshot()['used_staging_bytes'] == charged
    m.close('entry')
    assert ws.closed and not m.backend._owners
    assert m.budget.snapshot()['used_staging_bytes'] == m.shared_native_budget_bytes


def test_wrong_identity_is_rejected_before_native_or_cache_allocation():
    m, rt, req = setup()
    bad = SearchRequestIdentity(vector_space='foreign', positional_encoding='rope_applied',
        entry_transfer_id='entry', layer=1, kv_head=1)
    with pytest.raises(ValueError):
        m.search_many(req[:3] + ((bad, req[3][1], 4),))
    assert rt.submits == 0 and not m._entries['entry'].search_workspaces
    m.close('entry')


def test_partial_group_falls_back_without_waiting():
    m, rt, req = setup()
    meta = {}
    assert len(m.search_many(req[:2], metadata=meta)) == 2
    assert meta['path'] == 'grouped_cagra' and rt.submits == 0
    m.close('entry')


def test_shape_change_replaces_one_cache_and_refunds_old_charge():
    m, rt, req = setup()
    m.search_many(req)
    first = next(iter(m._entries['entry'].search_workspaces.values()))[0]
    before = m.budget.snapshot()['used_staging_bytes']
    changed = tuple((identity, q[:1].contiguous(), top_k) for identity, q, top_k in req)
    m.search_many(changed)
    caches = m._entries['entry'].search_workspaces
    assert len(caches) == 1 and first.closed
    replacement = next(iter(caches.values()))[0]
    assert m.budget.snapshot()['used_staging_bytes'] == before - first.retained_bytes + replacement.retained_bytes
    m.close('entry')


def test_close_during_native_search_defers_workspace_and_index_retirement():
    m, rt, req = setup()
    record = m._entries['entry']
    rt.on_submit = lambda: m.close('entry')
    assert len(m.search_many(req)) == 4
    assert not m.backend._owners and not record.search_workspaces
    assert m.budget.snapshot()['used_staging_bytes'] == m.shared_native_budget_bytes


def test_unknown_completion_keeps_reader_buffers_budget_and_index():
    m, rt, req = setup()
    rt.on_submit = lambda: setattr(rt, 'fail_sync', True)
    with pytest.raises(IndexCompletionUnknown):
        m.search_many(req)
    record = m._entries['entry']
    assert m.quarantined and record.users == 1
    ws = next(iter(record.search_workspaces.values()))[0]
    assert ws.pending_queries is not None
    before = m.budget.snapshot()['used_staging_bytes']
    m.close('entry')
    assert m.backend._owners and record.search_workspaces
    assert m.budget.snapshot()['used_staging_bytes'] == before


@pytest.mark.parametrize('rank', [0, 1])
def test_partial_layers_share_cache_and_preserve_reversed_head_mapping(rank):
    m, rt, req = setup(rank, partial=True)
    m.batched_group_search = False
    expected = m.search_many(req)
    m.batched_group_search = True
    meta = {}
    first = m.search_many(req[:2][::-1], metadata=meta)
    assert tuple(r.selection for r in first[::-1]) == tuple(r.selection for r in expected[:2])
    assert meta['path'] == 'grouped_cagra_partial_batched' and rt.submits == 2
    ws = next(iter(m._entries['entry'].search_workspaces.values()))[0]
    charged = m.budget.snapshot()['used_staging_bytes']
    second = m.search_many(req[2:][::-1], metadata=meta)
    assert tuple(r.selection for r in second[::-1]) == tuple(r.selection for r in expected[2:])
    assert rt.submits == 4
    assert next(iter(m._entries['entry'].search_workspaces.values()))[0] is ws
    assert m.budget.snapshot()['used_staging_bytes'] == charged
    # Nonadjacent native heads also preserve their own Q and filter.
    mixed = m.search_many((req[3], req[0]))
    assert tuple(r.selection for r in mixed) == (expected[3].selection, expected[0].selection)
    assert rt.submits == 6
    m.close('entry')
    assert ws.closed and m.budget.snapshot()['used_staging_bytes'] == m.shared_native_budget_bytes


def test_partial_bad_version_never_submits_or_allocates_cache():
    from dataclasses import replace
    m, rt, req = setup(partial=True)
    bad = replace(req[1][0], expected_index_version='stale')
    with pytest.raises(ValueError):
        m.search_many((req[0], (bad, req[1][1], 4)))
    assert rt.submits == 0 and not m._entries['entry'].search_workspaces
    m.close('entry')


@pytest.mark.parametrize('unknown', [False, True])
def test_partial_retirement_preserves_completion_contract(unknown):
    m, rt, req = setup(partial=True)
    record = m._entries['entry']
    if unknown:
        rt.on_submit = lambda: setattr(rt, 'fail_sync', True)
        with pytest.raises(IndexCompletionUnknown):
            m.search_many(req[2:])
        ws = next(iter(record.search_workspaces.values()))[0]
        assert record.users == 1 and ws.pending_queries.shape[0] == 2
        charged = m.budget.snapshot()['used_staging_bytes']
        m.close('entry')
        assert m.quarantined and record.search_workspaces and m.backend._owners
        assert m.budget.snapshot()['used_staging_bytes'] == charged
    else:
        rt.on_submit = lambda: m.close('entry')
        assert len(m.search_many(req[2:])) == 2
        assert not record.search_workspaces and not m.backend._owners
        assert m.budget.snapshot()['used_staging_bytes'] == m.shared_native_budget_bytes


def test_partial_shape_change_replaces_only_one_budgeted_workspace():
    m, rt, req = setup(partial=True)
    m.search_many(req[:2])
    first = next(iter(m._entries['entry'].search_workspaces.values()))[0]
    charged = m.budget.snapshot()['used_staging_bytes']
    changed = tuple((identity, q[:1].contiguous(), k) for identity, q, k in req[2:])
    m.search_many(changed)
    assert first.closed and len(m._entries['entry'].search_workspaces) == 1
    second = next(iter(m._entries['entry'].search_workspaces.values()))[0]
    assert m.budget.snapshot()['used_staging_bytes'] == charged - first.retained_bytes + second.retained_bytes
    m.close('entry')
