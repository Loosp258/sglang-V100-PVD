"""Protocol/hash compatibility and mutable-layout guards, using the old serializer."""

import dataclasses
import hashlib
import json
from collections import UserDict
from dataclasses import dataclass, field, replace

import pytest
import torch
from sglang.srt.disaggregation.pvd import protocol
from sglang.srt.disaggregation.pvd.protocol import KVLayoutSignature
from sglang.srt.disaggregation.pvd.sparse_copy import copy_sparse_kv_into
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from test_pvd_selected_component_views import component_case


def previous_dict(layout):
    result = dataclasses.asdict(layout)
    result['extra'] = dict(layout.extra)
    return result


def previous_wire(layout):
    return json.dumps(previous_dict(layout), sort_keys=True, separators=(',', ':'), default=str)


def layout(extra):
    return KVLayoutSignature(model_id='Qwen-测试', model_revision='rev', kv_dtype='torch.float16',
        page_size=4, num_layers=28, total_kv_heads=4, kv_heads_per_rank=2,
        head_dim=128, tp_size=2, pp_size=1, tensor_layout='paired', extra=extra)


@pytest.mark.parametrize('extra', [
    {}, {'component_count': 56, 'component_dtypes': ['torch.float16'] * 56,
        'component_token_shapes': [[2, 128] for _ in range(56)],
        'component_bytes_per_token': [512] * 56},
    {'nested': {'lists': [None, True, 3.5, '中文'], 'tuple': (1, (2, 3))}},
    UserDict({'nested': {'shape': [2, 128]}}),
])
def test_dictionary_wire_and_hash_match_old_serializer(extra):
    source = layout(extra)
    expected = previous_dict(source)
    actual = source.to_dict()
    assert actual == expected and list(actual) == list(expected)
    assert json.dumps(actual, default=str) == json.dumps(expected, default=str)
    encoded = json.dumps(actual, sort_keys=True, separators=(',', ':'), default=str)
    assert encoded == previous_wire(source)
    assert source.fingerprint == hashlib.sha256(encoded.encode('utf-8')).hexdigest()
    roundtrip = KVLayoutSignature.from_dict(json.loads(encoded))
    assert roundtrip.fingerprint == source.fingerprint


def test_metadata_is_still_shallow_and_fingerprint_reads_every_mutation():
    source = layout({'nested': {'shape': [2, 128]}, 'tag': 'a'})
    exported = source.to_dict()
    first = source.fingerprint
    assert exported['extra'] is not source.extra
    assert exported['extra']['nested'] is source.extra['nested']
    exported['extra']['tag'] = 'export-only'
    assert source.extra['tag'] == 'a' and source.fingerprint == first
    source.extra['nested']['shape'][1] = 64
    assert source.fingerprint != first
    assert source.fingerprint == hashlib.sha256(previous_wire(source).encode()).hexdigest()


def test_normal_component_metadata_does_not_run_discarded_asdict(monkeypatch):
    source = layout({'component_token_shapes': [[2, 128] for _ in range(56)]})
    expected = previous_dict(source)
    def forbidden(*args, **kwargs):
        raise AssertionError('recursive component copy must not run for atomic layout fields')
    monkeypatch.setattr(protocol.dataclasses, 'asdict', forbidden)
    assert source.to_dict() == expected
    assert source.fingerprint == hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


@dataclass
class NestedRevision:
    values: list = field(default_factory=lambda: [1, {'a': 2}])


@dataclass(frozen=True)
class ExtendedLayout(KVLayoutSignature):
    extra_field: object = 'scalar'


class RevisionObject:
    def __init__(self): self.values = [1, 2]
    def __eq__(self, other): return isinstance(other, RevisionObject) and self.values == other.values
    def __str__(self): return 'revision:' + str(self.values)


@pytest.mark.parametrize('value', [NestedRevision(), [1, {'x': [2]}], RevisionObject()])
def test_non_atomic_revision_retains_conversion_and_deep_copy(value):
    source = replace(layout({'shape': [2, 128]}), model_revision=value)
    actual, expected = source.to_dict(), previous_dict(source)
    assert actual == expected and actual['model_revision'] is not value
    assert json.dumps(actual, default=str) == json.dumps(expected, default=str)
    assert source.fingerprint == hashlib.sha256(previous_wire(source).encode()).hexdigest()
    if isinstance(value, NestedRevision): assert isinstance(actual['model_revision'], dict)


@pytest.mark.parametrize('value', ['scalar', 3.5, None, NestedRevision(), {'items': [1, 2]}])
def test_subclass_fields_order_wire_and_fallback_are_preserved(value):
    source = ExtendedLayout(**layout({'shape': [2, 128]}).__dict__, extra_field=value)
    actual, expected = source.to_dict(), previous_dict(source)
    assert actual == expected and list(actual) == list(expected)
    assert json.dumps(actual, default=str) == json.dumps(expected, default=str)
    assert source.fingerprint == hashlib.sha256(previous_wire(source).encode()).hexdigest()
    if isinstance(value, (dict, NestedRevision)): assert actual['extra_field'] is not value


@pytest.mark.parametrize('selected', [False, True])
@pytest.mark.parametrize('change', ['invalid_component', 'unselected_tag'])
def test_layout_mutation_still_rejects_before_sparse_destination_write(selected, change):
    packed, kwargs, _ = component_case()
    destination = torch.full((kwargs['manifest'].nbytes,), 229, dtype=torch.uint8)
    copy_sparse_kv_into(packed, destination, **kwargs, selected_component_views=selected)
    before = destination.clone()
    if change == 'invalid_component': kwargs['layout'].extra['component_token_shapes'][0][1] -= 1
    else: kwargs['layout'].extra['tag'] = 'changed even though not a queried layer'
    with pytest.raises(SparsePayloadError):
        copy_sparse_kv_into(packed, destination, **kwargs, selected_component_views=selected)
    assert torch.equal(destination, before)
