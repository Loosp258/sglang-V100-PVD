"""Bounded float32 query rows inside the existing PVD search-batch envelope.

Only Q travels here. Prompt KV still uses the owned Mooncake data path. The
packed representation is opt-in so older V shards keep receiving JSON rows.
"""

import base64
import binascii
import json
import struct

import numpy as np

PACKED_QUERY_ENCODING = "f32le-base64-v1"
MAX_PACKED_QUERY_CELLS = 100_000
BINARY_QUERY_ENCODING = 'f32le-binary-v1'
BINARY_QUERY_CONTENT_TYPE = 'application/x-pvd-q-f32'
_MAGIC = b'PVDQF32\x01'
_MAX_META = 262144


def binary_search_snapshot(search):
    """Freeze either producer arrays or locally parsed rows in one wire form."""
    items = []
    for item in search['items']:
        if item.get('query_encoding') == BINARY_QUERY_ENCODING:
            values = unpack_query_rows(item)
            meta = {k: v for k, v in item.items() if not k.startswith('query_')}
            meta['queries'] = values
        else:
            meta = dict(item)
        items.append(meta)
    return pack_binary_batch({**search, 'items': items})


def pack_binary_fused(payload):
    search = payload['search']
    parsed = unpack_binary_batch(binary_search_snapshot(search))
    items = []
    for item in parsed['items']:
        items.append({**{k: v for k, v in item.items() if not k.startswith('query_')},
                      'queries': item['query_values']})
    return pack_binary_batch({**{k: v for k, v in payload.items() if k != 'search'},
        'search': {k: v for k, v in search.items() if k != 'items'}, 'items': items})


def unpack_binary_fused(raw):
    payload = unpack_binary_batch(raw)
    if not isinstance(payload.get('search'), dict) or 'items' in payload['search']:
        raise ValueError('exact nested binary search metadata required')
    payload['search'] = {**payload['search'], 'items': payload.pop('items')}
    return payload


def pack_binary_batch(payload):
    """Bounded metadata plus exact f32 bytes; never materialize Python Q floats."""
    items, chunks, cells = [], [], 0
    if not isinstance(payload, dict) or not 1 <= len(payload.get('items', ())) <= 64:
        raise ValueError('bounded binary batch required')
    for item in payload['items']:
        rows = np.asarray(item['queries'])
        if (rows.ndim != 2 or not 1 <= rows.shape[0] <= 64 or rows.shape[1] < 1
                or rows.dtype.kind not in 'fi' or rows.size > MAX_PACKED_QUERY_CELLS):
            raise ValueError('bounded numeric binary query rows required')
        rows = np.array(rows, dtype='<f4', order='C', copy=True)
        if not np.isfinite(rows).all(): raise ValueError('finite f32 query required')
        cells += rows.size
        if cells > MAX_PACKED_QUERY_CELLS: raise ValueError('binary batch cell bound exceeded')
        meta = {k:v for k,v in item.items() if k != 'queries'}
        if any(k.startswith('query_') for k in meta): raise ValueError('mixed query encodings')
        meta.update(query_encoding=BINARY_QUERY_ENCODING,query_rows=int(rows.shape[0]),query_dim=int(rows.shape[1]))
        items.append(meta);chunks.append(rows.tobytes(order='C'))
    meta = json.dumps({**payload,'items':items},allow_nan=False,separators=(',',':')).encode('utf-8')
    if len(meta) > _MAX_META: raise ValueError('binary batch metadata too large')
    return _MAGIC + struct.pack('<I',len(meta)) + meta + b''.join(chunks)


def unpack_binary_batch(raw):
    if not isinstance(raw, bytes) or not 12 <= len(raw) <= 12+_MAX_META+4*MAX_PACKED_QUERY_CELLS or raw[:8] != _MAGIC:
        raise ValueError('invalid binary query envelope')
    size = struct.unpack_from('<I',raw,8)[0]
    if not 1 <= size <= _MAX_META or 12+size > len(raw): raise ValueError('invalid binary metadata length')
    payload = json.loads(raw[12:12+size])
    if not isinstance(payload,dict) or not isinstance(payload.get('items'),list) or not 1 <= len(payload['items']) <= 64:
        raise ValueError('invalid binary batch metadata')
    offset,cells=12+size,0
    for item in payload['items']:
        if not isinstance(item,dict): raise ValueError('binary item must be object')
        rows,dim=item.get('query_rows'),item.get('query_dim')
        if (item.get('query_encoding') != BINARY_QUERY_ENCODING or type(rows) is not int
                or type(dim) is not int or not 1 <= rows <= 64 or dim < 1
                or any(k in item for k in ('queries','query_data','query_values'))):
            raise ValueError('invalid binary query shape/representation')
        cells += rows*dim
        if cells > MAX_PACKED_QUERY_CELLS or offset+rows*dim*4 > len(raw):
            raise ValueError('binary query size differs from bound')
        # Owned copy independent of HTTP storage; only parser creates this ndarray.
        values=np.frombuffer(raw,dtype='<f4',count=rows*dim,offset=offset).copy().reshape(rows,dim)
        if not np.isfinite(values).all(): raise ValueError('nonfinite binary query')
        item['query_values']=values;offset+=rows*dim*4
    if offset != len(raw): raise ValueError('trailing binary bytes')
    return payload


def pack_query_rows(rows):
    """Make an owned, little-endian float32 snapshot of validated host rows."""
    values = np.asarray(rows, dtype="<f4")
    if (
        values.ndim != 2
        or not 1 <= values.shape[0] <= 64
        or not 1 <= values.shape[1]
        or values.size > MAX_PACKED_QUERY_CELLS
        or not np.isfinite(values).all()
    ):
        raise ValueError("packed queries require bounded finite float32 rows")
    return {
        "query_encoding": PACKED_QUERY_ENCODING,
        "query_rows": int(values.shape[0]),
        "query_dim": int(values.shape[1]),
        "query_data": base64.b64encode(values.tobytes(order="C")).decode("ascii"),
    }


def unpack_query_rows(item):
    """Decode only the exact canonical encoding, with bounds before allocation."""
    if item.get('query_encoding') == BINARY_QUERY_ENCODING:
        values = item.get('query_values')
        if (not isinstance(values,np.ndarray) or values.dtype != np.dtype('<f4')
                or values.ndim != 2 or values.shape != (item.get('query_rows'),item.get('query_dim'))
                or not 1 <= values.shape[0] <= 64 or values.size > MAX_PACKED_QUERY_CELLS
                or values.shape[1] < 1 or not np.isfinite(values).all()):
            raise ValueError('binary rows require locally parsed owned f32 data')
        return values
    rows, dim, encoded = (
        item.get("query_rows"),
        item.get("query_dim"),
        item.get("query_data"),
    )
    if (
        item.get("query_encoding") != PACKED_QUERY_ENCODING
        or type(rows) is not int
        or not 1 <= rows <= 64
        or type(dim) is not int
        or not 1 <= dim
        or rows * dim > MAX_PACKED_QUERY_CELLS
        or not isinstance(encoded, str)
        or len(encoded) != 4 * ((rows * dim * 4 + 2) // 3)
    ):
        raise ValueError("invalid packed query shape or encoding")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid packed query bytes") from exc
    if len(raw) != rows * dim * 4:
        raise ValueError("packed query byte length differs from shape")
    # The copy makes the tensor view writable and gives this request ownership
    # independent of the transient HTTP body.
    values = np.frombuffer(raw, dtype="<f4").copy().reshape(rows, dim)
    if not np.isfinite(values).all():
        raise ValueError("packed queries must be finite float32 numbers")
    return values
