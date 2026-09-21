"""Wire contract and leased current index; no RDMA completion claims."""

import json
from dataclasses import replace

import pytest
import torch
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_prompt_index import ident, manager, stored_entry


def manifest_for(index, key, layout):
    result = index.search(
        ident(key.transfer_id), queries=torch.ones(1, layout.head_dim), top_k=1
    )
    spec = SparseKVSpec(
        "consumer",
        "inc",
        "op",
        4,
        key.transfer_id,
        result.index_version,
        result.id_mapping_version,
        layout.fingerprint,
        0,
        0,
        (3, 0, 7),
    )
    return SparseDeliveryManifest(
        (spec, replace(spec, layer=1, token_ids=(1, 4))),
        layout.kv_dtype,
        layout.head_dim,
    )


def basic():
    spec = SparseKVSpec(
        "r", "inc", "op", 4, "entry", "idx", "map", "layout", 0, 0, (3, 1)
    )
    return SparseDeliveryManifest((spec, replace(spec, layer=1)), "torch.float16", 8)


def test_canonical_wire_roundtrip_offsets_and_operation_fingerprint():
    manifest = basic()
    assert (
        SparseDeliveryManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
        == manifest
    )
    assert manifest.nbytes == 128
    data = torch.arange(64, dtype=torch.float16).view(torch.uint8)
    payloads = manifest.payload_views(data)
    assert [p.nbytes for p in payloads] == [64, 64]
    assert payloads[0].tensor.flatten().tolist() == list(range(32))
    assert payloads[1].tensor.flatten().tolist() == list(range(32, 64))
    assert (
        replace(
            manifest,
            specs=tuple(replace(s, operation_id="new") for s in manifest.specs),
        ).fingerprint
        != manifest.fingerprint
    )


@pytest.mark.parametrize(
    "change",
    [
        "protocol",
        "extra",
        "missing",
        "unknown_spec",
        "bool",
        "duplicate",
        "mixed",
        "dtype",
        "size",
    ],
)
def test_bad_wire_rejected(change):
    value = basic().to_dict()
    if change == "protocol":
        value["protocol"] = "future"
    elif change == "extra":
        value["address"] = 123
    elif change == "missing":
        del value["specs"][0]["index_version"]
    elif change == "unknown_spec":
        value["specs"][0]["offset"] = 1
    elif change == "bool":
        value["specs"][0]["token_ids"] = [True]
    elif change == "duplicate":
        value["specs"][1] = value["specs"][0]
    elif change == "mixed":
        value["specs"][1]["request_id"] = "other"
    elif change == "dtype":
        value["dtype"] = "torch.int8"
    else:
        value["head_dim"] = True
    with pytest.raises(ValueError):
        SparseDeliveryManifest.from_dict(value)


@pytest.mark.parametrize(
    "data",
    [
        torch.zeros(127, dtype=torch.uint8),
        torch.zeros(128),
        torch.zeros(256, dtype=torch.uint8)[::2],
    ],
)
def test_exact_byte_shape_is_required(data):
    with pytest.raises(ValueError):
        basic().payload_views(data)


def test_index_lease_survives_close_and_rebuild_without_refunding_new_record():
    budget = TransferBudget(1 << 20, 32)
    index = manager(budget=budget)
    store, manifest, _, layout = stored_entry(index)
    try:
        store.progress_prompt_indexes()
        selection = manifest_for(index, manifest.key, layout)
        charged = budget.snapshot()["used_staging_bytes"]
        with index.pin_selection(selection):
            index.close(manifest.key.transfer_id)
            assert budget.snapshot()["used_staging_bytes"] == charged
            index.open(manifest.key.transfer_id)
            index.note_kv_readable(manifest.key.transfer_id)
            store.progress_prompt_indexes()
            assert budget.snapshot()["used_staging_bytes"] == charged * 2
            with pytest.raises(ValueError), index.pin_selection(selection):
                pytest.fail("stale version accepted")
        assert budget.snapshot()["used_staging_bytes"] == charged
    finally:
        store.close()
    assert budget.snapshot()["used_staging_bytes"] == 0


@pytest.mark.parametrize(
    "mutation", ["head", "token", "index", "mapping", "dtype", "dim", "entry"]
)
def test_lease_checks_authoritative_built_metadata(mutation):
    index = manager(budget=TransferBudget(1 << 20, 32))
    store, manifest, _, layout = stored_entry(index)
    try:
        store.progress_prompt_indexes()
        selection = manifest_for(index, manifest.key, layout)
        changes = {
            "head": {"kv_head": 99},
            "token": {"token_ids": (999,)},
            "index": {"index_version": "old"},
            "mapping": {"id_mapping_version": "old"},
            "entry": {"entry_transfer_id": "other"},
        }
        if mutation in changes:
            selection = replace(
                selection,
                specs=tuple(replace(s, **changes[mutation]) for s in selection.specs),
            )
        elif mutation == "dtype":
            selection = replace(selection, dtype="torch.float32")
        else:
            selection = replace(selection, head_dim=256)
        with pytest.raises(ValueError), index.pin_selection(selection):
            pytest.fail("invalid selection leased")
    finally:
        store.close()
