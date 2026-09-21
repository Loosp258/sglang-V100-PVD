"""Offline sparse byte contract, not delivery/installation/attention tests."""

from dataclasses import replace

import pytest
import torch
from sglang.srt.disaggregation.pvd.sparse_payload import (
    SparseKVSpec,
    SparsePayloadError,
    pack_sparse_kv,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferBudget,
    TransferCapacityError,
)
from test_pvd_prompt_vectors import FakePool, pack_shard, storage_layout


def fixture(rank=0, dtype=torch.float16):
    pool = FakePool(dtype=dtype)
    layout = storage_layout(pool)
    packed, shard, _ = pack_shard(pool, layout, rank=rank, prompt_tokens=10)
    spec = SparseKVSpec(
        "req",
        "inc",
        "op",
        16,
        "entry",
        "index-v1",
        "map-v1",
        layout.fingerprint,
        1,
        rank * 2 + 1,
        (7, 0, 9),
    )
    budget = TransferBudget(4096, 1)
    return (
        pool,
        packed,
        {
            "layout": layout,
            "shard": shard,
            "spec": spec,
            "entry_transfer_id": "entry",
            "index_version": "index-v1",
            "id_mapping_version": "map-v1",
            "budget": budget,
        },
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_exact_kv_pairing_order_global_heads_and_no_alias(rank, dtype):
    pool, packed, kwargs = fixture(rank, dtype)
    spec = kwargs["spec"]
    expected = torch.stack(
        [
            source[list(spec.token_ids), spec.kv_head]
            for source in (pool.k_buffer[spec.layer], pool.v_buffer[spec.layer])
        ]
    )
    with pack_sparse_kv(packed.tensor, **kwargs) as payload:
        assert payload.tensor.dtype == dtype
        assert payload.spec.token_ids == (
            7,
            0,
            9,
        )  # preserve absolute positions and declared order
        assert kwargs["budget"].snapshot()["used_staging_bytes"] == payload.nbytes
        packed.tensor.zero_()  # output must own its bytes
        torch.testing.assert_close(payload.tensor, expected, rtol=0, atol=0)
    assert kwargs["budget"].snapshot()["used_staging_bytes"] == 0
    with pytest.raises(SparsePayloadError, match="scope has ended"):
        _ = payload.tensor


@pytest.mark.parametrize(
    "field,value",
    [
        ("entry_transfer_id", "other"),
        ("index_version", "old"),
        ("id_mapping_version", "old"),
        ("layout_fingerprint", "wrong"),
        ("layer", 99),
        ("kv_head", 3),
        ("token_ids", (10,)),
        ("token_ids", (11,)),
    ],
)
def test_bad_or_stale_selection_refused_before_reservation(field, value):
    _, packed, kwargs = fixture()
    kwargs["spec"] = replace(kwargs["spec"], **{field: value})
    with pytest.raises(SparsePayloadError), pack_sparse_kv(packed.tensor, **kwargs):
        pytest.fail("invalid selection admitted")
    assert kwargs["budget"].snapshot()["used_staging_bytes"] == 0


@pytest.mark.parametrize("ids", [(), (0, 0), (-1,), (True,), [0, 1], (0.5,)])
def test_invalid_logical_token_ids_rejected(ids):
    _, _, kwargs = fixture()
    with pytest.raises(SparsePayloadError, match="token ids"):
        replace(kwargs["spec"], token_ids=ids)


def test_short_buffer_refused_and_budget_exhaustion_does_not_allocate():
    _, packed, kwargs = fixture()
    with (
        pytest.raises(SparsePayloadError, match="byte count"),
        pack_sparse_kv(packed.tensor[:-1], **kwargs),
    ):
        pytest.fail("short buffer admitted")
    kwargs["budget"] = TransferBudget(1, 1)
    with pytest.raises(TransferCapacityError), pack_sparse_kv(packed.tensor, **kwargs):
        pytest.fail("over-budget payload admitted")
    assert kwargs["budget"].snapshot()["used_staging_bytes"] == 0


def test_exception_inside_scope_releases_budget_and_closes_payload():
    _, packed, kwargs = fixture()
    with (
        pytest.raises(RuntimeError, match="consumer failure"),
        pack_sparse_kv(packed.tensor, **kwargs) as payload,
    ):
        raise RuntimeError("consumer failure")
    assert kwargs["budget"].snapshot()["used_staging_bytes"] == 0
    with pytest.raises(SparsePayloadError, match="scope has ended"):
        _ = payload.tensor


def test_corrupt_component_metadata_refused():
    _, packed, kwargs = fixture()
    kwargs["layout"].extra["component_bytes_per_token"][0] += 1
    kwargs["spec"] = replace(
        kwargs["spec"], layout_fingerprint=kwargs["layout"].fingerprint
    )
    with (
        pytest.raises(SparsePayloadError, match="metadata disagree"),
        pack_sparse_kv(packed.tensor, **kwargs),
    ):
        pytest.fail("corrupt metadata admitted")


def test_nonzero_global_layer_start_maps_to_local_component():
    pool, packed, kwargs = fixture()
    kwargs["layout"] = replace(kwargs["layout"], num_layers=5)
    kwargs["shard"] = replace(kwargs["shard"], layer_start=2, layer_end=5)
    kwargs["spec"] = replace(
        kwargs["spec"], layer=3, layout_fingerprint=kwargs["layout"].fingerprint
    )
    with pack_sparse_kv(packed.tensor, **kwargs) as payload:
        torch.testing.assert_close(payload.tensor[0], pool.k_buffer[1][[7, 0, 9], 1])
        torch.testing.assert_close(payload.tensor[1], pool.v_buffer[1][[7, 0, 9], 1])


def test_output_allocation_failure_refunds_its_reservation(monkeypatch):
    _, packed, kwargs = fixture()
    original = torch.empty

    def fail_output(size, *args, **options):
        if isinstance(size, tuple) and size == (2, 3, 8):
            assert kwargs["budget"].snapshot()["used_staging_bytes"] > 0
            raise RuntimeError("injected allocation failure")
        return original(size, *args, **options)

    monkeypatch.setattr(torch, "empty", fail_output)
    with (
        pytest.raises(RuntimeError, match="injected allocation failure"),
        pack_sparse_kv(packed.tensor, **kwargs),
    ):
        pytest.fail("output allocation should fail")
    assert kwargs["budget"].snapshot()["used_staging_bytes"] == 0
