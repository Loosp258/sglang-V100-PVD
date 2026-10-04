"""CPU byte oracles and explicit policy doubles; CUDA cases remain real skips."""

from dataclasses import replace
from types import SimpleNamespace
import importlib.util
from pathlib import Path

import pytest
import torch
from sglang.srt.disaggregation.pvd import server, vector_store
from sglang.srt.disaggregation.pvd.sparse_copy import copy_sparse_kv_into
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec, SparsePayloadError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransportState
from test_pvd_cuda_sparse_packing import args, cuda_policy
from test_pvd_prompt_vectors import FakePool, pack_shard, storage_layout


def component_case(rank=0, dtype=torch.float16, shape="single"):
    count, start, selected = (3, 5, (2,)) if shape == "offset" else (
        28, 0, (2, 17, 27) if shape == "multi" else (17,))
    pool = FakePool(layers=count, dim=128, dtype=dtype)
    pool.start_layer, pool.end_layer = start, start + count
    layout = replace(storage_layout(pool), num_layers=start + count)
    pages = (2, 0, 1)  # physical page order differs from logical Prompt IDs
    packed, shard, _ = pack_shard(pool, layout, rank=rank, prompt_tokens=10, pages=pages)
    specs = tuple(SparseKVSpec("req", "inc", "op", 4, "entry", "index", "map",
        layout.fingerprint, layer + start, rank * 2 + head,
        (9, 0, 7) if head == 0 else (6, 3)) for layer in selected for head in (0, 1))
    manifest = SparseDeliveryManifest(specs, str(dtype), 128)
    expected = []
    for spec in specs:
        physical = [pages[token // 4] * 4 + token % 4 for token in spec.token_ids]
        expected.append(torch.stack([values[spec.layer - start][physical, spec.kv_head]
            for values in (pool.k_buffer, pool.v_buffer)]))
    return packed.tensor, dict(layout=layout, shard=shard, manifest=manifest,
        entry_transfer_id="entry", index_version="index", id_mapping_version="map"), tuple(expected)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", ["single", "multi", "offset"])
def test_actual_view_counts_and_exact_bytes_without_gather_allocations(monkeypatch, rank, dtype, shape):
    source, kwargs, expected = component_case(rank, dtype, shape)
    before = source.clone()
    manifest, shard = kwargs["manifest"], kwargs["shard"]
    actual_calls = []
    reshape = torch.Tensor.reshape
    def observe(tensor, *shape):
        if tensor.untyped_storage().data_ptr() == source.untyped_storage().data_ptr():
            actual_calls.append(shape)
        return reshape(tensor, *shape)
    def forbidden(*args, **kwargs): raise AssertionError("no new gather/KV allocations permitted")
    for selected in (False, True):
        target = torch.empty(manifest.nbytes, dtype=torch.uint8)
        actual_calls.clear()
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "reshape", observe)
            for name in ("empty", "empty_like", "zeros", "stack", "cat", "index_select", "gather"):
                patch.setattr(torch, name, forbidden)
            reported = copy_sparse_kv_into(source, target, **kwargs, selected_component_views=selected)
        wanted = 2 * len({s.layer for s in manifest.specs}) if selected else 2 * (shard.layer_end - shard.layer_start)
        assert reported == len(actual_calls) == wanted
        assert torch.equal(source, before)
        for payload, oracle in zip(manifest.payload_views(target), expected, strict=True):
            assert torch.equal(payload.tensor.view(torch.uint8), oracle.view(torch.uint8))


@pytest.mark.parametrize("bad", ["unselected_dtype", "unselected_shape", "unselected_bytes", "later_layer",
    "later_head", "padding", "alias", "alignment", "nonbool", "fused", "contiguous", "layout_type", "shard_type"])
def test_all_metadata_and_later_groups_still_fail_before_any_write(bad):
    source, kwargs, _ = component_case()
    manifest = kwargs["manifest"]
    target = torch.full((manifest.nbytes,), 219, dtype=torch.uint8)
    options = dict(selected_component_views=True)
    if bad.startswith("unselected_"):
        # First call succeeds; mutable layout metadata must be revalidated.
        copy_sparse_kv_into(source, target, **kwargs, **options)
        extra = kwargs["layout"].extra
        if bad == "unselected_dtype": extra["component_dtypes"][0] = "torch.float32"
        if bad == "unselected_shape": extra["component_token_shapes"][0] = [2, 127]
        if bad == "unselected_bytes": extra["component_bytes_per_token"][0] += 1
        kwargs["manifest"] = replace(manifest, specs=tuple(replace(s,
            layout_fingerprint=kwargs["layout"].fingerprint) for s in manifest.specs))
    elif bad.startswith("later_") or bad == "padding":
        changed = {"later_layer": dict(layer=99), "later_head": dict(kv_head=3),
            "padding": dict(token_ids=(10,))}[bad]
        kwargs["manifest"] = replace(manifest, specs=(manifest.specs[0], replace(manifest.specs[1], **changed)))
        target = torch.full((kwargs["manifest"].nbytes,), 219, dtype=torch.uint8)
    elif bad == "alias": target = source[:manifest.nbytes]
    elif bad == "alignment": target = torch.full((manifest.nbytes + 1,), 219, dtype=torch.uint8)[1:]
    elif bad == "nonbool": options["selected_component_views"] = 1
    elif bad == "fused": options["fused_workspace"] = object()
    elif bad == "contiguous": options["contiguous_runs"] = True
    elif bad == "layout_type": kwargs["layout"] = object()
    elif bad == "shard_type": kwargs["shard"] = object()
    before = target.clone()
    with pytest.raises(SparsePayloadError):
        copy_sparse_kv_into(source, target, **kwargs, **options)
    assert torch.equal(target, before)


@pytest.mark.parametrize("failure", [None, "copy", "sync", "register"])
def test_selected_store_keeps_fences_and_unknown_owners(monkeypatch, failure):
    store, entry, manifest, budget, target, delivery, _ = cuda_policy(monkeypatch)
    store.selected_sparse_component_views = True  # explicit CPU CUDA policy
    copy = vector_store.copy_sparse_kv_into
    def copy_rows(*a, **kw):
        assert kw["selected_component_views"] is True
        count = copy(*a, **kw)
        if failure == "copy": raise RuntimeError("partly failed copy")
        return count
    calls = []
    def sync(device):
        calls.append(device)
        if failure == "sync" and len(calls) == 1: raise RuntimeError("completion unknown")
    if failure == "register":
        def fail(*a, **kw): raise RuntimeError("registration unknown")
        monkeypatch.setattr(store.transfer_engine, "register_memory", fail)
    monkeypatch.setattr(vector_store, "copy_sparse_kv_into", copy_rows)
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    store.start_delivery(entry.key, delivery.delivery_id)
    assert len(calls) == 2  # every original V store fence remains
    if failure in ("sync", "register"):
        assert delivery.local_terminal is TransportState.UNKNOWN
        assert budget.snapshot()["used_staging_bytes"] == manifest.nbytes
        assert delivery.staging_guard.value is not None
        assert not store.fence_write(delivery.authorization.identity)["fenced"]
        if failure == "sync": assert delivery.packing_index_lease is not None
    elif failure == "copy":
        assert delivery.local_terminal is TransportState.NOT_SUBMITTED
        assert budget.snapshot()["used_staging_bytes"] == 0
    else:
        profile = delivery.source_profile.snapshot()
        assert profile["selected_component_views"] is True
        assert profile["source_component_views"] == 4  # fixture selects two layers
        store.transfer_engine.finish(delivery.transfer_handle)
        store.progress_transfers()
        store.ack_delivery(entry.key, delivery.delivery_id)
        assert budget.snapshot()["used_staging_bytes"] == 0
    store.close()
    store.transfer_engine.release_memory(target)


@pytest.mark.parametrize("rank", [0, 1])
def test_default_off_and_both_rank_option_wiring(monkeypatch, rank):
    assert not args().experimental_selected_sparse_component_views
    configured = args("--experimental-cuda-sparse-packing", "--experimental-selected-sparse-component-views")
    configured.transfer_backend = "fake"  # constructor wiring only
    monkeypatch.setattr(server, "_build_prompt_index", lambda _: object())
    monkeypatch.setattr(server, "VectorKVStore", lambda **kw: SimpleNamespace(**kw))
    store, _ = server._create_store(configured, rank=rank, local_rank=rank, rails=["mlx5_2", "mlx5_3"])
    assert store.selected_sparse_component_views is True


@pytest.mark.parametrize("bad", [None, "no_cuda", "direct", "contiguous", "reuse_fence"])
def test_startup_requires_isolated_cuda_staging(bad):
    configured = args("--experimental-cuda-sparse-packing", "--experimental-selected-sparse-component-views",
        "--prompt-index-vector-space", "test", "--prompt-index-budget-bytes", "1024")
    if bad == "no_cuda": configured.experimental_cuda_sparse_packing = False
    if bad == "direct": configured.experimental_direct_sparse_batch_put = True
    if bad == "contiguous": configured.experimental_contiguous_sparse_packing = True
    if bad == "reuse_fence": configured.experimental_reuse_sparse_pack_fence = True
    if bad:
        with pytest.raises(ValueError): server._validate_args(configured)
    else:
        server._validate_args(configured)


def test_partial_copy_keeps_source_and_never_returns_success_count(monkeypatch):
    source, kwargs, _ = component_case()
    target = torch.full((kwargs['manifest'].nbytes,), 219, dtype=torch.uint8)
    before = source.clone()
    original = torch.Tensor.copy_
    calls = []
    def copy(tensor, other, **options):
        calls.append(tuple(other.shape))
        if len(calls) == 2: raise RuntimeError("selected partial copy")
        return original(tensor, other, **options)
    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    with pytest.raises(RuntimeError, match="partial copy"):
        copy_sparse_kv_into(source, target, **kwargs, selected_component_views=True)
    assert len(calls) == 2 and torch.equal(source, before)
    assert target.eq(219).any() and not target.eq(219).all()


@pytest.mark.parametrize("bad", [None, "count", "mode", "fence", "reuse", "bytes", "cpu"])
def test_live_profile_gate_refuses_unproven_preparation(bad):
    from sglang.srt.disaggregation.pvd.v_source_profile import VSourceProfile
    path = Path(__file__).resolve().parents[3] / 'benchmark/pvd_oasis_delivery_validation.py'
    spec = importlib.util.spec_from_file_location('component_delivery_validation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = VSourceProfile(nbytes=512, cuda=True, kernel='torch', selected_component_views=True)
    profile.record_component_views(2)
    for name in profile.snapshot()['phases']:
        with profile.measure(name): pass
    value = profile.snapshot()
    if bad == 'count': value['source_component_views'] = 56
    if bad == 'mode': value['selected_component_views'] = False
    if bad == 'fence': value['phases']['pack_fence']['successes'] = 0
    if bad == 'reuse': value['outer_fence_reused'] = True
    if bad == 'bytes': value['nbytes'] = 256
    if bad == 'cpu': value['cuda'] = False
    if bad:
        with pytest.raises(AssertionError): module.validate_selected_source_profile(value, selected=True, nbytes=512)
    else:
        module.validate_selected_source_profile(value, selected=True, nbytes=512)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA required; unverified")
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_real_cuda_nondefault_stream_selected_bytes(rank, dtype):
    host, kwargs, expected = component_case(rank, dtype)
    device = torch.device("cuda:0")
    source = host.to(device)
    torch.cuda.synchronize(device)
    stream = torch.cuda.Stream(device=device)
    for selected in (False, True):
        target = torch.empty(kwargs["manifest"].nbytes, dtype=torch.uint8, device=device)
        with torch.cuda.stream(stream):
            count = copy_sparse_kv_into(source, target, **kwargs, allow_cuda=True, selected_component_views=selected)
        stream.synchronize()  # caller owns source and destination through proof
        assert count == (2 if selected else 56)
        for payload, oracle in zip(kwargs["manifest"].payload_views(target.cpu()), expected, strict=True):
            assert torch.equal(payload.tensor.view(torch.uint8), oracle.view(torch.uint8))
