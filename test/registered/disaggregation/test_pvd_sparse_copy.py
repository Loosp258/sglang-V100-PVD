"""Direct sparse copy contracts; CUDA cases are skipped, not emulated, without CUDA."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.sparse_copy import copy_sparse_kv_into
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_sparse_payload import fixture
from test_pvd_sparse_store_delivery import destination, ready


def setup(rank=0, dtype=torch.float16):
    pool, packed, kwargs = fixture(rank, dtype)
    spec = kwargs.pop("spec")
    kwargs.pop("budget")
    manifest = SparseDeliveryManifest(
        (spec, replace(spec, layer=0, token_ids=(4, 1))),
        str(dtype),
        kwargs["layout"].head_dim,
    )
    return pool, packed.tensor, {**kwargs, "manifest": manifest}


def expected(pool, manifest):
    return tuple(
        torch.stack(
            (
                pool.k_buffer[s.layer][list(s.token_ids), s.kv_head],
                pool.v_buffer[s.layer][list(s.token_ids), s.kv_head],
            )
        )
        for s in manifest.specs
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_exact_direct_copy_without_payload_allocations(rank, dtype, monkeypatch):
    pool, source, kwargs = setup(rank, dtype)
    manifest = kwargs["manifest"]
    target = torch.empty(manifest.nbytes, dtype=torch.uint8)
    wanted, original = expected(pool, manifest), source.clone()

    def allocate(*_, **__):
        raise AssertionError(
            "direct sparse copy cannot allocate payload/gather tensors"
        )

    with monkeypatch.context() as patch:
        for name in (
            "empty",
            "empty_like",
            "zeros",
            "cat",
            "stack",
            "index_select",
            "gather",
        ):
            patch.setattr(torch, name, allocate)
        copy_sparse_kv_into(source, target, **kwargs)
    torch.testing.assert_close(source, original, rtol=0, atol=0)
    for payload, value in zip(manifest.payload_views(target), wanted, strict=True):
        torch.testing.assert_close(payload.tensor, value, rtol=0, atol=0)
    target.zero_()
    torch.testing.assert_close(source, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("layer", 99),
        ("kv_head", 3),
        ("token_ids", (10,)),
        ("token_ids", (11,)),
    ],
)
def test_invalid_later_group_cannot_partly_write_destination(field, value):
    _, source, kwargs = setup()
    manifest = kwargs["manifest"]
    kwargs["manifest"] = replace(
        manifest,
        specs=(manifest.specs[0], replace(manifest.specs[1], **{field: value})),
    )
    target = torch.full((kwargs["manifest"].nbytes,), 211, dtype=torch.uint8)
    with pytest.raises(SparsePayloadError):
        copy_sparse_kv_into(source, target, **kwargs)
    assert target.eq(211).all()


@pytest.mark.parametrize(
    "mode",
    [
        "entry",
        "index",
        "mapping",
        "layout",
        "dtype",
        "dim",
        "extent",
        "strided",
        "device",
        "alignment",
        "cuda_flag",
    ],
)
def test_bad_copy_contract_refused_before_writes(mode):
    _, source, kwargs = setup()
    manifest = kwargs["manifest"]
    target = torch.full((manifest.nbytes,), 211, dtype=torch.uint8)
    if mode in ("entry", "index", "mapping"):
        field = {
            "entry": "entry_transfer_id",
            "index": "index_version",
            "mapping": "id_mapping_version",
        }[mode]
        kwargs[field] = "old"
    elif mode == "layout":
        kwargs["manifest"] = replace(
            manifest,
            specs=tuple(replace(s, layout_fingerprint="other") for s in manifest.specs),
        )
    elif mode in ("dtype", "dim"):
        kwargs["manifest"] = replace(
            manifest,
            **({"dtype": "torch.float32"} if mode == "dtype" else {"head_dim": 4}),
        )
        target = torch.full((kwargs["manifest"].nbytes,), 211, dtype=torch.uint8)
    elif mode == "extent":
        target = target[:-1]
    elif mode == "strided":
        target = torch.full((manifest.nbytes * 2,), 211, dtype=torch.uint8)[::2]
    elif mode == "alignment":
        target = torch.full((manifest.nbytes + 1,), 211, dtype=torch.uint8)[1:]
    elif mode == "device":
        target = torch.empty(manifest.nbytes, device="meta", dtype=torch.uint8)
    else:
        kwargs["allow_cuda"] = "yes"
    with pytest.raises(SparsePayloadError):
        copy_sparse_kv_into(source, target, **kwargs)
    if mode != "device":
        assert target.eq(211).all()


@pytest.mark.parametrize("overlap", [True, False])
def test_even_disjoint_views_cannot_use_entry_storage_as_staging(overlap):
    _, original, kwargs = setup()
    length = kwargs["manifest"].nbytes
    storage = torch.empty(original.numel() + length, dtype=torch.uint8)
    source = storage[: original.numel()]
    source.copy_(original)
    target = storage[:length] if overlap else storage[-length:]
    with pytest.raises(SparsePayloadError, match="distinct storage"):
        copy_sparse_kv_into(source, target, **kwargs)
    torch.testing.assert_close(source, original)


def test_copy_failure_keeps_caller_buffer_owned_and_does_not_claim_completion(
    monkeypatch,
):
    _, source, kwargs = setup()
    target = torch.full((kwargs["manifest"].nbytes,), 211, dtype=torch.uint8)
    original = torch.Tensor.copy_
    calls = []

    def fail_after_one(tensor, *args, **options):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("copy failed after earlier work was queued")
        return original(tensor, *args, **options)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "copy_", fail_after_one)
        with pytest.raises(RuntimeError, match="copy failed"):
            copy_sparse_kv_into(source, target, **kwargs)
    assert len(calls) == 2
    assert target.numel() == kwargs["manifest"].nbytes
    # Caller must discard the partly written generation; the buffer is not freed.
    assert target.eq(211).any() and not target.eq(211).all()


def test_v_delivery_fits_exact_final_buffer_budget():
    store, entry, pool, manifest, _ = ready()
    engine = store.transfer_engine
    budget = TransferBudget(manifest.nbytes, 1)
    engine.lifecycle_manager = SimpleNamespace(budget=budget)
    target = destination(store, manifest)
    try:
        delivery = store.reserve_delivery(entry.key, "exact-budget", target.descriptor)
        store.start_delivery(entry.key, delivery.delivery_id)
        assert delivery.transfer_handle is not None, delivery.error
        assert budget.snapshot()["used_staging_bytes"] == manifest.nbytes
        assert budget.snapshot()["reservations"] == 1
        engine.finish(delivery.transfer_handle)
        store.poll_delivery(entry.key, delivery.delivery_id)
        store.ack_delivery(entry.key, delivery.delivery_id)
        assert budget.snapshot()["used_staging_bytes"] == 0
        for payload, value in zip(
            manifest.payload_views(target.buffer), expected(pool, manifest), strict=True
        ):
            torch.testing.assert_close(payload.tensor, value, rtol=0, atol=0)
    finally:
        store.close()
        engine.release_memory(target)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="real CUDA unavailable; stream/copy behavior unverified",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_real_cuda_nondefault_stream_requires_opt_in_and_owned_completion(dtype):
    pool, host, kwargs = setup(dtype=dtype)
    device = torch.device("cuda", 0)
    source = host.to(device)
    target = torch.empty(kwargs["manifest"].nbytes, dtype=torch.uint8, device=device)
    with pytest.raises(SparsePayloadError, match="explicit CUDA"):
        copy_sparse_kv_into(source, target, **kwargs)
    torch.cuda.synchronize(device)  # source placement precedes nondefault-stream copy
    stream = torch.cuda.Stream(device=device)
    event = torch.cuda.Event()
    try:
        with torch.cuda.stream(stream):
            copy_sparse_kv_into(source, target, **kwargs, allow_cuda=True)
            event.record(stream)
        event.synchronize()  # caller holds BOTH buffers until real completion
        for payload, value in zip(
            kwargs["manifest"].payload_views(target.cpu()),
            expected(pool, kwargs["manifest"]),
            strict=True,
        ):
            torch.testing.assert_close(payload.tensor, value, rtol=0, atol=0)
        torch.testing.assert_close(source.cpu(), host, rtol=0, atol=0)
    finally:
        # Includes failure after a subset of copies has already been enqueued.
        stream.synchronize()
