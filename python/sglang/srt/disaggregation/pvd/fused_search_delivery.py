"""Bounded dynamic selection capability for one rank/layer.

Only a prefix of a D-owned physical registration may be written. The original
WriteIdentity exists before search; ordinary absent-write fences tombstone it
before any late reserve, even when the dynamic manifest response is lost.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import time
import uuid
from sglang.srt.disaggregation.pvd.oasis_pipeline import select_resident
from sglang.srt.disaggregation.pvd.protocol import WriteIdentity, PVD_TRANSFER_LIFECYCLE_PROTOCOL
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest, SPARSE_DELIVERY_KEY
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
from sglang.srt.disaggregation.pvd.search_wire import unpack_query_rows, PACKED_QUERY_ENCODING
from sglang.srt.disaggregation.pvd.search_wire import (
    BINARY_QUERY_ENCODING, BINARY_QUERY_CONTENT_TYPE, binary_search_snapshot,
    unpack_binary_batch, pack_binary_fused, BinarySearchSnapshot, freeze_binary_search)
from sglang.srt.disaggregation.pvd.cache_snapshot import cache_ids

FUSED_PROTOCOL = 'pvd.search-delivery.v1'
_SIZES={'torch.float16':2,'torch.bfloat16':2,'torch.float32':4}
_FIELDS={'request_id','incarnation','operation_id','target_tokens','entry_transfer_id',
         'layout_fingerprint','layer','heads','capacity','max_new','prompt_tokens','dtype',
         'head_dim','resident','cached'}


def validate_selection(scope):
    if not isinstance(scope,dict) or set(scope) != _FIELDS:
        raise ValueError('exact fused selection scope required')
    for name in ('request_id','incarnation','operation_id','entry_transfer_id','layout_fingerprint'):
        if not isinstance(scope[name],str) or not 1 <= len(scope[name]) <= 256:
            raise ValueError('bounded selection identity required')
    for name in ('target_tokens','layer','capacity','max_new','prompt_tokens','head_dim'):
        if type(scope[name]) is not int or scope[name] < 0: raise ValueError('integer selection bound required')
    if (not 1 <= scope['capacity'] <= 32 or not 0 <= scope['max_new'] <= scope['capacity']
            or not 1 <= scope['prompt_tokens'] <= 32768 or not 1 <= scope['head_dim'] <= 128
            or scope['dtype'] not in _SIZES or scope['heads'] not in ([0,1],[2,3])):
        raise ValueError('unsupported bounded rank selection')
    for name in ('resident','cached'):
        if not isinstance(scope[name],list) or len(scope[name]) != 2: raise ValueError('two head snapshots required')
        for ids in scope[name]:
            if name == 'cached':
                cache_ids(ids, scope['prompt_tokens'])
                continue
            if (not isinstance(ids,list) or len(ids) > (scope['capacity'] if name=='resident' else scope['prompt_tokens'])
                    or any(type(t) is not int or not 0 <= t < scope['prompt_tokens'] for t in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError('invalid resident/cache snapshot')
    return scope


def selection_digest(scope, search):
    snapshot=search if isinstance(search,BinarySearchSnapshot) else None
    if snapshot is not None: search=snapshot.search
    validate_selection(scope)
    if not isinstance(search,dict) or not isinstance(search.get('items'),list) or len(search['items']) != 2:
        raise ValueError('two head searches required')
    for head,item in zip(scope['heads'],search['items'],strict=True):
        if item.get('query_encoding') in (PACKED_QUERY_ENCODING, BINARY_QUERY_ENCODING):
            if 'queries' in item: raise ValueError('mixed fused query encodings')
            values=unpack_query_rows(item)
            shape=values.shape
        else:
            values=item.get('queries')
            if not isinstance(values,list) or not values or any(not isinstance(row,list) for row in values):
                raise ValueError('bounded fused query representation required')
            shape=(len(values),len(values[0]))
            if any(len(row) != shape[1] for row in values): raise ValueError('ragged fused queries')
        if (item.get('kv_head') != head or item.get('layer') != scope['layer']
                or item.get('transfer_id') != scope['entry_transfer_id']
                or not 1 <= shape[0] <= 7 or shape[1] != scope['head_dim']):
            raise ValueError('search differs from authorized rank/layer/shape')
    if any(item.get('query_encoding') == BINARY_QUERY_ENCODING for item in search['items']):
        if not all(item.get('query_encoding') == BINARY_QUERY_ENCODING for item in search['items']):
            raise ValueError('mixed fused binary encodings')
        encoded=(json.dumps(scope,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
                 + b'\x00pvd-fused-binary-v1\x00' + (snapshot.wire if snapshot else binary_search_snapshot(search)))
    else:
        encoded=json.dumps([scope,search],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(encoded) > 262144: raise ValueError('fused selection snapshot too large')
    return hashlib.sha256(encoded).hexdigest()


def allocation_bytes(scope):
    validate_selection(scope)
    return 2*scope['capacity']*2*scope['head_dim']*_SIZES[scope['dtype']]


def choose_wire(scope, results):
    """Same score ordering, resident policy and CPU cache misses as D baseline."""
    validate_selection(scope)
    if not isinstance(results,(list,tuple)) or len(results) != 2: raise ValueError('two search results required')
    versions={(r['index_version'],r['id_mapping_version']) for r in results}
    if len(versions) != 1: raise ValueError('mixed immutable index versions')
    pair=next(iter(versions)); chosen=[];specs=[]
    for i,(head,result) in enumerate(zip(scope['heads'],results,strict=True)):
        if result['kv_head'] != head or result['layer'] != scope['layer']:
            raise ValueError('foreign selection result')
        ranked=tuple(t for _,t in sorted(zip(result['scores'],result['token_ids'],strict=True),reverse=True))
        ids=select_resident(ranked,scope['resident'][i],capacity=scope['capacity'],max_new=scope['max_new'])
        if not ids or any(t >= scope['prompt_tokens'] for t in ids): raise ValueError('invalid selected IDs')
        chosen.append(ids);cached=set(cache_ids(scope['cached'][i],scope['prompt_tokens']));missing=tuple(t for t in ids if t not in cached)
        if missing:
            specs.append(SparseKVSpec(scope['request_id'],scope['incarnation'],scope['operation_id'],
                scope['target_tokens'],scope['entry_transfer_id'],*pair,scope['layout_fingerprint'],
                scope['layer'],head,missing))
    wire=SparseDeliveryManifest(tuple(specs),scope['dtype'],scope['head_dim']) if specs else None
    if wire is not None and wire.nbytes > allocation_bytes(scope): raise ValueError('dynamic wire exceeds authorization')
    return tuple(chosen),wire


def wire_destination(physical, wire):
    if wire.nbytes > physical.length: raise ValueError('wire exceeds physical registration')
    return replace(physical,length=wire.nbytes,backend_metadata={**physical.backend_metadata,
        SPARSE_DELIVERY_KEY:wire.to_dict()})


@dataclass(frozen=True)
class _AllocationOnly:
    nbytes: int
    digest: str
    scope: dict
    def to_dict(self):
        return dict(protocol='pvd-fused-allocation-only-v1',nbytes=self.nbytes,selection_digest=self.digest)


def prepare_fused(registry, scope, search, *, key, rank, rail, endpoint, sender_epoch, client, owner_scope=None,
                  binary_queries=False, zero_miss_proof=False):
    registry._owner();started=time.perf_counter()
    if any(not isinstance(value,str) or not value.strip() for value in (rail,endpoint)):
        raise ValueError('explicit fused rail/endpoint required')
    if owner_scope is not None and (not isinstance(owner_scope,str) or not owner_scope.strip()):
        raise ValueError('explicit fused owner scope required')
    # Freeze the producer snapshots before registration and the first await.
    scope=json.loads(json.dumps(scope,allow_nan=False))
    if type(binary_queries) is not bool: raise ValueError('binary fused option must be bool')
    if type(zero_miss_proof) is not bool: raise ValueError('zero-miss proof option must be bool')
    snapshot=freeze_binary_search(search) if binary_queries else None
    search=snapshot.search if snapshot else json.loads(json.dumps(search,allow_nan=False))
    digest=selection_digest(scope,snapshot or search)
    if scope['heads'] != [rank*2,rank*2+1] or scope['entry_transfer_id'] != key.transfer_id:
        raise ValueError('fused allocation rank/Entry mismatch')
    identity=WriteIdentity(PVD_TRANSFER_LIFECYCLE_PROTOCOL,sender_epoch,registry.receiver_epoch,
        uuid.uuid4().hex,'pending-registration',uuid.uuid4().hex,rank,key)
    record=registry._new_record(_AllocationOnly(allocation_bytes(scope),digest,scope),identity,client)
    record._scope=owner_scope
    record.fused_scope,record.fused_search,record.fused_digest=scope,search,digest
    record.fused_binary_snapshot=snapshot
    record.fused_zero_miss_proof=zero_miss_proof
    registry.budget.reserve(record.owner,registry._destination_charge(record.manifest),1)
    registry._records[identity.transfer_id]=record
    registry._prepare_registration(record,endpoint=endpoint,rank=rank,rail=rail,generation=identity.generation)
    descriptor=record._registration.descriptor
    record.identity=replace(identity,region_id=descriptor.region_id)
    record.identity.validate_destination(descriptor)
    if descriptor.length != record.manifest.nbytes: raise ValueError('fused physical extent mismatch')
    registry._after_register(record);record._registration_unknown=False
    record.profile.update(prepare_seconds=time.perf_counter()-started,fused_calls=0,fused_seconds=0.0,
        fused_search_delivery=True,authorized_bytes=descriptor.length)
    return record


async def start_fused(record, search_client, requests):
    async with record._lock:
        record._live()
        if record._published: raise ValueError('fused destination already published; poll or fence')
        record._published=True
        payload=dict(protocol=FUSED_PROTOCOL,selection=record.fused_scope,search=record.fused_search,
                     identity=record.identity.to_dict(),destination=record._registration.descriptor.to_dict())
        if record.fused_zero_miss_proof: payload['zero_miss_proof']=True
        binary=record.fused_search['items'][0].get('query_encoding') == BINARY_QUERY_ENCODING
        options=(dict(encoded_payload=pack_binary_fused(payload,snapshot=record.fused_binary_snapshot),content_type=BINARY_QUERY_CONTENT_TYPE)
                 if binary else {})
        if search_client.binary_control_channel:
            if not binary:raise ValueError('binary channel requires frozen binary Q')
            operation=search_client._post_fused_channel(options['encoded_payload'])
        else:
            operation=search_client._post_json(
                '/internal/v1/indexes/search-deliver-binary' if binary else '/internal/v1/indexes/search-deliver',
                payload,**options)
        reply=await record._timed_rpc('fused',operation)
        if reply.get('protocol') != FUSED_PROTOCOL or reply.get('selection_digest') != record.fused_digest:
            raise ValueError('fused response capability mismatch')
        if reply.get('identity') != record.identity.to_dict(): raise ValueError('fused response identity mismatch')
        results=reply.get('results')
        if not isinstance(results,list) or len(results) != 2: raise ValueError('missing fused search results')
        for result, request, item in zip(results,requests,record.fused_search['items'],strict=True):
            identity,queries,top_k,scope=request
            search_client._validate_reply(result,identity,scope,item['search_id'],len(queries),top_k)
        record.fused_results=results
        chosen,wire=choose_wire(record.fused_scope,results)
        if reply.get('chosen') != [list(ids) for ids in chosen] or reply.get('manifest') != (wire.to_dict() if wire else None):
            raise ValueError('V selection/cache misses differ from D policy')
        if wire is None:
            if reply.get('delivery') is not None: raise ValueError('unexpected zero-miss writer')
            if record.fused_zero_miss_proof:
                record.accept_absent_write_fence(reply.get('absent_write_fence'))
            record.fused_no_miss=True
            return chosen,True
        record.manifest=wire
        record._wire_destination=wire_destination(record._registration.descriptor,wire)
        record._buffer=record._buffer[:wire.nbytes]
        ready=record._observe(reply['delivery'])
        record._source_started=reply['delivery'].get('state') in ('v_writing','delivered','acked','released')
        return chosen,ready
