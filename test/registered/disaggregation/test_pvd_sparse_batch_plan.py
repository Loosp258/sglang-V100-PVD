"""Independent CPU bytes for direct reads from the original pool MR."""

from dataclasses import replace

import pytest
import torch
from sglang.srt.disaggregation.pvd.sparse_batch_plan import build_sparse_batch_plan
from sglang.srt.disaggregation.pvd.sparse_copy import copy_sparse_kv_into
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine
from test_pvd_sparse_copy import setup


def fixture(rank=0, dtype=torch.float16):
    _, source, kwargs = setup(rank, dtype)
    manifest = kwargs["manifest"]
    kwargs["manifest"] = replace(manifest, specs=(
        manifest.specs[0],
        replace(manifest.specs[1], kv_head=rank * 2),
    ))
    shard = kwargs["shard"]
    page_bytes = shard.expected_bytes // shard.page_count
    offset = 2 * page_bytes
    pool = torch.full((offset + source.numel() + page_bytes,), 213, dtype=torch.uint8)
    pool[offset:offset + source.numel()] = source
    engine = FakeTransferEngine()
    registration = engine.register_memory(
        pool, endpoint="V", rank=rank, rail=shard.rail,
    )
    return engine, registration, offset, source, kwargs


def plan(registration, offset, kwargs, **changes):
    arguments = dict(kwargs)
    manifest, layout, shard = [arguments.pop(k) for k in ("manifest", "layout", "shard")]
    arguments.update(allocation_offset=offset, registration=registration)
    arguments.update(changes)
    return build_sparse_batch_plan(manifest, layout, shard, **arguments)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_original_pool_slices_match_independent_paired_byte_oracle(rank, dtype, monkeypatch):
    engine, registration, offset, source, kwargs = fixture(rank, dtype)
    expected = torch.empty(kwargs["manifest"].nbytes, dtype=torch.uint8)
    copy_sparse_kv_into(source, expected, **kwargs)
    original = registration.buffer.clone()
    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", lambda *a, **k: pytest.fail("planner allocated staging"))
        result = plan(registration, offset, kwargs)
    actual = torch.full_like(expected, 201)
    for local, remote in zip(result.slices, result.remote_offsets, strict=True):
        assert local.registration is registration
        assert offset <= local.offset < local.offset + local.length <= offset + source.numel()
        actual[remote:remote + local.length] = registration.buffer[local.offset:local.offset + local.length]
    assert torch.equal(actual, expected)
    assert torch.equal(original, registration.buffer)
    assert sum(local.length for local in result.slices) == result.nbytes == expected.numel()
    engine.release_memory(registration)


@pytest.mark.parametrize("field", ["entry_transfer_id", "index_version", "id_mapping_version"])
def test_current_identity_is_not_taken_from_untrusted_manifest(field):
    engine, registration, offset, _, kwargs = fixture()
    with pytest.raises(SparsePayloadError, match="current Entry/index/mapping"):
        plan(registration, offset, kwargs, **{field: "stale"})
    engine.release_memory(registration)


@pytest.mark.parametrize("field,value", [
    ("component_count", 2), ("component_dtypes", ["torch.float32"] * 6),
    ("component_token_shapes", [[1, 16]] * 6),
    ("component_bytes_per_token", [17] * 6),
])
def test_component_metadata_is_fully_validated(field, value):
    engine, registration, offset, _, kwargs = fixture()
    layout = kwargs["layout"]
    kwargs["layout"] = replace(layout, extra={**layout.extra, field: value})
    with pytest.raises(SparsePayloadError):
        plan(registration, offset, kwargs)
    engine.release_memory(registration)


@pytest.mark.parametrize("change", ["address", "length", "rank", "rail", "allocation", "alignment", "backing"])
def test_pool_registration_and_entry_extent_are_independent_bounds(change):
    engine, registration, offset, _, kwargs = fixture()
    original = registration
    if change in ("address", "length", "rank", "rail"):
        values = {"address": registration.descriptor.address + 1,
                  "length": registration.descriptor.length - 1,
                  "rank": 1, "rail": "foreign"}
        registration = replace(registration, descriptor=replace(registration.descriptor, **{change: values[change]}))
    elif change == "allocation":
        offset = registration.buffer.numel()
    elif change == "alignment":
        offset += 1
    else:
        registration = replace(registration, buffer=registration.buffer.clone())
    with pytest.raises(SparsePayloadError):
        plan(registration, offset, kwargs)
    engine.release_memory(original)


@pytest.mark.parametrize("change", ["padding", "head", "layer", "tensor_layout", "slice_bound"])
def test_foreign_rows_and_unbounded_fragment_lists_are_refused(change):
    engine, registration, offset, _, kwargs = fixture()
    manifest = kwargs["manifest"]
    if change in ("padding", "head", "layer"):
        updates = {"padding": {"token_ids": (10,)}, "head": {"kv_head": 99}, "layer": {"layer": 99}}[change]
        kwargs["manifest"] = replace(manifest, specs=(manifest.specs[0], replace(manifest.specs[1], **updates)))
    elif change == "tensor_layout":
        kwargs["layout"] = replace(kwargs["layout"], tensor_layout="foreign")
    with pytest.raises(SparsePayloadError):
        plan(registration, offset, kwargs, **({"max_slices": 2} if change == "slice_bound" else {}))
    engine.release_memory(registration)
