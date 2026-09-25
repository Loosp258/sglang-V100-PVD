"""Independent CPU byte oracle for V's optional one-kernel sparse pack path."""

from dataclasses import replace

import pytest
import torch
from sglang.srt.disaggregation.pvd.sparse_copy import copy_sparse_kv_into
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_pack_plan import build_sparse_pack_plan
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_sparse_copy import setup


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_pack_byte_mapping_matches_existing_packer(rank, dtype):
    _, source, kwargs = setup(rank, dtype)
    manifest = kwargs["manifest"]
    layout, shard = kwargs["layout"], kwargs["shard"]
    plan = build_sparse_pack_plan(manifest, layout, shard)
    expected = torch.empty(manifest.nbytes, dtype=torch.uint8)
    copy_sparse_kv_into(source, expected, **kwargs)
    actual = torch.empty_like(expected)
    rows = shard.page_count * layout.page_size
    heads = layout.kv_heads_per_rank
    layers = shard.layer_end - shard.layer_start
    head_bytes = (
        layout.head_dim
        * source.element_size()
        * (2 if dtype in (torch.float16, torch.bfloat16) else 4)
    )
    assert plan.source_bytes == source.numel()
    assert plan.destination_bytes == expected.numel()
    assert plan.metadata_bytes == 8 * (len(plan.token_ids) + 5 * len(plan.groups))
    for layer, head, token_start, count, dest_start in plan.groups:
        for kind in range(2):
            for row in range(count):
                token = plan.token_ids[token_start + row]
                src = (
                    (layer + kind * layers) * rows * heads * head_bytes
                    + token * heads * head_bytes
                    + head * head_bytes
                )
                dst = dest_start + (kind * count + row) * head_bytes
                actual[dst : dst + head_bytes] = source[src : src + head_bytes]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("bad", ["head", "layer", "padding", "layout"])
def test_fused_pack_plan_refuses_foreign_or_stale_selection(bad):
    _, _, kwargs = setup()
    manifest = kwargs["manifest"]
    spec = manifest.specs[-1]
    changes = {
        "head": {"kv_head": 999},
        "layer": {"layer": 999},
        "padding": {"token_ids": (999,)},
        "layout": {"layout_fingerprint": "stale"},
    }[bad]
    specs = list(manifest.specs)
    if bad == "layout":
        specs = [replace(s, **changes) for s in specs]
    else:
        specs[-1] = replace(spec, **changes)
    invalid = SparseDeliveryManifest(tuple(specs), manifest.dtype, manifest.head_dim)
    with pytest.raises(SparsePayloadError):
        build_sparse_pack_plan(invalid, kwargs["layout"], kwargs["shard"])


def test_unbudgeted_fused_workspace_is_refused_before_any_copy():
    _, source, kwargs = setup()
    target = torch.full((kwargs["manifest"].nbytes,), 211, dtype=torch.uint8)
    with pytest.raises(SparsePayloadError, match="matching CUDA workspace"):
        copy_sparse_kv_into(source, target, **kwargs, fused_workspace=object())
    assert torch.all(target == 211)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_real_cuda_fused_sparse_pack_matches_reference(dtype):
    pytest.importorskip("triton")
    from sglang.srt.disaggregation.pvd.triton_sparse_pack import SparsePackWorkspace

    _, host, kwargs = setup(dtype=dtype)
    device = torch.device("cuda:0")
    source = host.to(device)
    actual = torch.empty(kwargs["manifest"].nbytes, dtype=torch.uint8, device=device)
    expected = torch.empty_like(actual)
    plan = build_sparse_pack_plan(kwargs["manifest"], kwargs["layout"], kwargs["shard"])
    budget = TransferBudget(plan.metadata_bytes, 1)
    workspace = SparsePackWorkspace(
        kwargs["manifest"],
        shard=kwargs["shard"],
        layout=kwargs["layout"],
        device=device,
        budget=budget,
        owner="test-v-fused-pack",
    )
    try:
        copy_sparse_kv_into(
            source, actual, **kwargs, allow_cuda=True, fused_workspace=workspace
        )
        copy_sparse_kv_into(source, expected, **kwargs, allow_cuda=True)
        torch.cuda.synchronize(device)
        assert torch.equal(actual, expected)
    finally:
        torch.cuda.synchronize(device)
        workspace.release_after_fence()
    assert budget.snapshot()["used_staging_bytes"] == 0
