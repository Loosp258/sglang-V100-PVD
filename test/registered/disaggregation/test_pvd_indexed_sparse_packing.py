"""Actual CPU bytes/out storage; labelled CUDA policy doubles; real CUDA skips."""

from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd import server, vector_store, sparse_row_index
from sglang.srt.disaggregation.pvd.sparse_copy import copy_sparse_kv_into
from sglang.srt.disaggregation.pvd.sparse_pack_plan import SparsePackCompletionUnknown
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from sglang.srt.disaggregation.pvd.sparse_row_index import SparseRowIndexWorkspace
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget, TransferCapacityError, TransportState
from sglang.srt.disaggregation.pvd.v_source_profile import VSourceProfile, copy_source_profile
from test_pvd_cuda_sparse_packing import args, cuda_policy
from test_pvd_selected_component_views import component_case


def workspace(kwargs, budget, device='cpu', owner='row-index'):
    return SparseRowIndexWorkspace(kwargs['manifest'], layout=kwargs['layout'], shard=kwargs['shard'],
        device=device, budget=budget, owner=owner)


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('shape', ['single', 'multi', 'offset', 'singleton', 'mixed'])
def test_actual_index_calls_exact_bytes_and_destination_storage(monkeypatch, rank, dtype, shape):
    source, kwargs, oracle = component_case(rank, dtype, shape if shape in ('multi','offset') else 'single')
    if shape in ('singleton', 'mixed'):
        specs = kwargs['manifest'].specs
        kwargs['manifest'] = replace(kwargs['manifest'], specs=tuple(replace(s,
            token_ids=(s.token_ids[0],)) if shape == 'singleton' or i == 0 else s for i,s in enumerate(specs)))
        oracle = tuple(o[:, :1] if shape == 'singleton' or i == 0 else o for i,o in enumerate(oracle))
    budget = TransferBudget(1 << 20, 8)
    owner = workspace(kwargs, budget)
    before = source.clone()
    target = torch.empty(kwargs['manifest'].nbytes, dtype=torch.uint8)
    metrics, calls, copies = {}, [], []
    original_select, original_copy = torch.index_select, torch.Tensor.copy_
    def select(input, dim, index, *, out):
        assert dim == 0 and index.ndim == 1 and out.shape == (index.numel(), input.shape[1])
        address, storage = out.data_ptr(), out.untyped_storage().data_ptr()
        assert storage == target.untyped_storage().data_ptr()
        result = original_select(input, dim, index, out=out)
        assert result.data_ptr() == address and result.untyped_storage().data_ptr() == storage
        calls.append(index.numel())
        return result
    def copy(tensor, other, **kw):
        copies.append(other.numel())
        return original_copy(tensor, other, **kw)
    def forbidden(*a, **kw): raise AssertionError('no extra KV/gather allocation inside helper')
    with monkeypatch.context() as patch:
        for name in ('empty','empty_like','zeros','cat','stack','gather','tensor'):
            patch.setattr(torch, name, forbidden)
        patch.setattr(torch, 'index_select', select)
        patch.setattr(torch.Tensor, 'copy_', copy)
        views = copy_sparse_kv_into(source, target, **kwargs, selected_component_views=True,
            indexed_workspace=owner, copy_metrics=metrics)
    counts = [len(s.token_ids) for s in kwargs['manifest'].specs]
    assert views == 2 * len({s.layer for s in kwargs['manifest'].specs})
    assert len(calls) == metrics['index_select_calls'] == 2 * sum(n > 1 for n in counts)
    assert len(copies) == metrics['row_copy_calls'] == 2 * sum(n == 1 for n in counts)
    assert metrics['row_index_bytes'] == 8 * sum(n for n in counts if n > 1)
    assert torch.equal(source, before)
    for payload, expected in zip(kwargs['manifest'].payload_views(target), oracle, strict=True):
        assert torch.equal(payload.tensor.view(torch.uint8), expected.contiguous().view(torch.uint8))
    assert budget.snapshot()['used_staging_bytes'] == owner.bytes
    owner.release_after_fence()
    owner.release_after_fence()
    assert budget.snapshot()['used_staging_bytes'] == 0 and owner._host is owner._indices is None


@pytest.mark.parametrize('bad', ['layer','padding','metadata','alias','alignment','manifest','layout','released','metrics','mix','index_shape','index_storage'])
def test_all_groups_and_owner_rejected_before_writes(bad):
    source, kwargs, _ = component_case()
    if bad in ('layer','padding'):
        specs = kwargs['manifest'].specs
        kwargs['manifest'] = replace(kwargs['manifest'], specs=(specs[0], replace(specs[1],
            **({'layer':99} if bad == 'layer' else {'token_ids':(10,)}))))
    budget = TransferBudget(1 << 20, 4)
    owner = workspace(kwargs, budget)
    target = torch.full((kwargs['manifest'].nbytes,), 217, dtype=torch.uint8)
    options = dict(selected_component_views=True, indexed_workspace=owner, copy_metrics={})
    if bad == 'metadata': kwargs['layout'].extra['component_bytes_per_token'][0] += 1
    if bad == 'alias': target = source[:kwargs['manifest'].nbytes]
    if bad == 'alignment': target = torch.full((kwargs['manifest'].nbytes + 1,), 217, dtype=torch.uint8)[1:]
    if bad == 'manifest': kwargs['manifest'] = replace(kwargs['manifest'])
    if bad == 'layout': kwargs['layout'] = replace(kwargs['layout'])
    if bad == 'released': owner.release_after_fence()
    if bad == 'metrics': options['copy_metrics'] = {'from_previous':1}
    if bad == 'mix': options['contiguous_runs'] = True
    if bad == 'index_shape': owner._groups = (owner._groups[0], owner._groups[1][:1])
    if bad == 'index_storage': owner._groups = (owner._groups[0], owner._groups[1].clone())
    before = target.clone()
    with pytest.raises(SparsePayloadError): copy_sparse_kv_into(source, target, **kwargs, **options)
    assert torch.equal(target, before)
    owner.release_after_fence()
    assert budget.snapshot()['used_staging_bytes'] == 0


def test_partial_index_copy_retains_owner_and_has_no_success_diagnostics(monkeypatch):
    source, kwargs, _ = component_case()
    budget = TransferBudget(1 << 20, 4)
    owner = workspace(kwargs, budget)
    target = torch.full((kwargs['manifest'].nbytes,), 215, dtype=torch.uint8)
    before, metrics, calls = source.clone(), {}, []
    original = torch.index_select
    def select(*a, **kw):
        calls.append(1)
        if len(calls) == 2: raise RuntimeError('partial indexed copy')
        return original(*a, **kw)
    monkeypatch.setattr(torch, 'index_select', select)
    with pytest.raises(RuntimeError, match='partial'):
        copy_sparse_kv_into(source, target, **kwargs, selected_component_views=True,
            indexed_workspace=owner, copy_metrics=metrics)
    assert metrics == {} and torch.equal(source, before)
    assert target.eq(215).any() and not target.eq(215).all()
    assert owner._indices is not None and budget.snapshot()['used_staging_bytes'] == owner.bytes
    owner.release_after_fence()  # synchronous CPU exception; never a CUDA proof


def test_budget_admission_precedes_index_allocation(monkeypatch):
    _, kwargs, _ = component_case()
    def forbidden(*a, **kw): raise AssertionError('allocation before budget admission')
    monkeypatch.setattr(torch, 'tensor', forbidden)
    budget = TransferBudget(1, 1)
    with pytest.raises(TransferCapacityError): workspace(kwargs, budget)
    assert budget.snapshot()['used_staging_bytes'] == 0


@pytest.mark.parametrize('cuda_unknown', [False, True])
def test_constructor_failure_drains_or_quarantines_explicit_cuda_policy(monkeypatch, cuda_unknown):
    _, kwargs, _ = component_case()
    budget = TransferBudget(1 << 20, 4)
    retained = []
    monkeypatch.setattr(sparse_row_index, '_QUARANTINED_WORKSPACES', retained)
    def upload(*a, **kw): raise RuntimeError('upload may have submitted')
    def sync(device):
        assert device == torch.device('cuda:1')
        if cuda_unknown: raise RuntimeError('CUDA drain unknown')
    monkeypatch.setattr(torch.Tensor, 'to', upload)
    monkeypatch.setattr(torch.cuda, 'synchronize', sync)
    with pytest.raises(SparsePackCompletionUnknown if cuda_unknown else RuntimeError):
        workspace(kwargs, budget, device='cuda:1')
    if cuda_unknown:
        assert len(retained) == 1 and retained[0]._host is not None
        assert not retained[0]._released and budget.snapshot()['used_staging_bytes'] == retained[0].bytes
    else:
        assert not retained and budget.snapshot()['used_staging_bytes'] == 0


@pytest.mark.parametrize('failure', [None,'copy','cancel','pack_fence','register','submit','metadata'])
def test_store_keeps_original_fences_and_unknown_owners(monkeypatch, failure):
    store, entry, manifest, budget, target, delivery, _ = cuda_policy(monkeypatch)
    store.selected_sparse_component_views = store.indexed_sparse_packing = True
    created, fences = [], []
    def prepare(manifest, **kw):
        if failure == 'metadata': raise SparsePackCompletionUnknown('controlled metadata UNKNOWN')
        assert kw.pop('device') == torch.device('cuda:1')
        result = SparseRowIndexWorkspace(manifest, device='cpu', **kw)  # labelled policy boundary
        created.append(result)
        return result
    monkeypatch.setattr(sparse_row_index, 'SparseRowIndexWorkspace', prepare)
    # The helper type must still be the real class, rather than the factory.
    class FactoryMeta(type):
        def __instancecheck__(cls, obj): return isinstance(obj, SparseRowIndexWorkspace)
    class Factory(metaclass=FactoryMeta):
        def __new__(cls, *a, **kw): return prepare(*a, **kw)
    monkeypatch.setattr(sparse_row_index, 'SparseRowIndexWorkspace', Factory)
    copy = vector_store.copy_sparse_kv_into
    def pack(*a, **kw):
        result = copy(*a, **kw)
        if failure == 'copy': raise RuntimeError('indexed copy failed after launch')
        if failure == 'cancel': store.cancel_entry(entry.key, 'cancel indexed pack')
        return result
    def fence(device):
        fences.append(device)
        if len(fences) == 1:
            assert delivery.packing_index_lease is not None
            if created: assert not created[0]._released
            if failure == 'pack_fence': raise RuntimeError('indexed fence UNKNOWN')
    if failure == 'register':
        monkeypatch.setattr(store.transfer_engine, 'register_memory', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('MR UNKNOWN')))
    if failure == 'submit':
        monkeypatch.setattr(store.transfer_engine, 'submit_put', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('PUT UNKNOWN')))
    monkeypatch.setattr(vector_store, 'copy_sparse_kv_into', pack)
    monkeypatch.setattr(torch.cuda, 'synchronize', fence)
    store.start_delivery(entry.key, delivery.delivery_id)
    assert len(fences) == 2
    if failure in ('pack_fence','metadata','register','submit'):
        assert delivery.local_terminal is TransportState.UNKNOWN
        assert not store.fence_write(delivery.authorization.identity)['fenced']
        assert delivery.staging_guard.value is not None
        if failure in ('pack_fence','metadata'):
            assert delivery.packing_index_lease is not None
            if created: assert budget.snapshot()['used_staging_bytes'] == manifest.nbytes + created[0].bytes
        else: assert created[0]._released and budget.snapshot()['used_staging_bytes'] == manifest.nbytes
    elif failure in ('copy','cancel'):
        assert delivery.local_terminal is TransportState.NOT_SUBMITTED
        assert created[0]._released and budget.snapshot()['used_staging_bytes'] == 0
    else:
        assert created[0]._released and delivery.packing_workspace is None
        value = delivery.source_profile.snapshot()
        assert value['kernel'] == 'torch_indexed_rows'
        assert value['index_select_calls'] == 2 * len(manifest.specs)
        assert copy_source_profile(value, nbytes=manifest.nbytes) == value
        assert store.start_delivery(entry.key, delivery.delivery_id) is delivery and len(created) == 1
        store.transfer_engine.finish(delivery.transfer_handle)
        store.progress_transfers()
        store.ack_delivery(entry.key, delivery.delivery_id)
        assert budget.snapshot()['used_staging_bytes'] == 0
    store.close()
    store.transfer_engine.release_memory(target)


@pytest.mark.parametrize('rank', [0, 1])
def test_default_off_and_rank_wiring(monkeypatch, rank):
    assert not args().experimental_indexed_sparse_packing
    configured = args('--experimental-cuda-sparse-packing', '--experimental-selected-sparse-component-views',
        '--experimental-indexed-sparse-packing')
    configured.transfer_backend = 'fake'  # wiring only
    monkeypatch.setattr(server, '_build_prompt_index', lambda _: object())
    monkeypatch.setattr(server, 'VectorKVStore', lambda **kw: SimpleNamespace(**kw))
    store, _ = server._create_store(configured, rank=rank, local_rank=rank, rails=['mlx5_2','mlx5_3'])
    assert store.indexed_sparse_packing and store.selected_sparse_component_views


@pytest.mark.parametrize('bad', [None,'no_selected','no_cuda','reuse','triton','contiguous','direct'])
def test_startup_isolation(bad):
    configured = args('--experimental-cuda-sparse-packing','--experimental-selected-sparse-component-views',
        '--experimental-indexed-sparse-packing','--prompt-index-vector-space','test','--prompt-index-budget-bytes','1024')
    fields = {'no_selected':'experimental_selected_sparse_component_views', 'no_cuda':'experimental_cuda_sparse_packing',
        'reuse':'experimental_reuse_sparse_pack_fence','triton':'experimental_triton_sparse_packing',
        'contiguous':'experimental_contiguous_sparse_packing','direct':'experimental_direct_sparse_batch_put'}
    if bad: setattr(configured, fields[bad], bad not in ('no_selected','no_cuda'))
    if bad:
        with pytest.raises(ValueError): server._validate_args(configured)
    else: server._validate_args(configured)


@pytest.mark.parametrize('bad', [None,'row_calls','index_calls','index_bytes','views','kernel','fence'])
def test_live_proof_checks_actual_copy_counts_and_index_cost(bad):
    path = Path(__file__).resolve().parents[3] / 'benchmark/pvd_oasis_delivery_validation.py'
    spec = importlib.util.spec_from_file_location('indexed_delivery_validation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = VSourceProfile(nbytes=2048, cuda=True, kernel='torch_indexed_rows', selected_component_views=True)
    profile.record_component_views(2)
    profile.record_copy_metrics(dict(row_copy_calls=2, index_select_calls=2, row_index_bytes=24))
    for name in profile.snapshot()['phases']:
        with profile.measure(name): pass
    value = profile.snapshot()
    fields = {'row_calls':'row_copy_calls','index_calls':'index_select_calls','index_bytes':'row_index_bytes','views':'source_component_views'}
    if bad in fields: value[fields[bad]] += 1
    if bad == 'kernel': value['kernel'] = 'torch'
    if bad == 'fence': value['phases']['pack_fence']['successes'] = 0
    if bad:
        with pytest.raises(AssertionError): module.validate_indexed_source_profile(value, indexed=True, nbytes=2048, group_rows=[1,3])
    else: module.validate_indexed_source_profile(value, indexed=True, nbytes=2048, group_rows=[1,3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real CUDA required; unverified')
@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('partial_failure', [False, True])
def test_real_cuda_nondefault_stream_bytes_and_owned_metadata(monkeypatch, rank, dtype, partial_failure):
    host, kwargs, oracle = component_case(rank, dtype)
    device = torch.device('cuda:0')
    source = host.to(device)
    torch.cuda.synchronize(device)
    stream = torch.cuda.Stream(device=device)
    target = torch.empty(kwargs['manifest'].nbytes, dtype=torch.uint8, device=device)
    budget = TransferBudget(1 << 20, 4)
    calls, original = [], torch.index_select
    def select(*a, **kw):
        calls.append(1)
        if partial_failure and len(calls) == 2: raise RuntimeError('partial real CUDA launch')
        return original(*a, **kw)
    monkeypatch.setattr(torch, 'index_select', select)
    with torch.cuda.stream(stream):
        owner = workspace(kwargs, budget, device=device)
        try:
            if partial_failure:
                with pytest.raises(RuntimeError, match='partial real CUDA'):
                    copy_sparse_kv_into(source, target, **kwargs, allow_cuda=True,
                        selected_component_views=True, indexed_workspace=owner)
            else:
                copy_sparse_kv_into(source, target, **kwargs, allow_cuda=True,
                    selected_component_views=True, indexed_workspace=owner)
        finally:
            assert owner._indices is not None and budget.snapshot()['used_staging_bytes'] == owner.bytes
            stream.synchronize()  # caller's actual completion proof, including exceptions
            owner.release_after_fence()
    assert budget.snapshot()['used_staging_bytes'] == 0 and torch.equal(source.cpu(), host)
    if not partial_failure:
        for payload, expected in zip(kwargs['manifest'].payload_views(target.cpu()), oracle, strict=True):
            assert torch.equal(payload.tensor.view(torch.uint8), expected.view(torch.uint8))
