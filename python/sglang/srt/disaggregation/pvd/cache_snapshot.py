"""Bounded exact Prompt cache sets, without cross-message delta state."""
import base64
import binascii
import numpy as np

_SPARSE = 'u16le-base64-v1'
_BITSET = 'bitset-le-base64-v1'


def pack_cache_snapshot(valid):
    values = np.asarray(valid)
    if values.dtype != np.bool_ or values.ndim != 1 or not 1 <= len(values) <= 32768:
        raise ValueError('bounded CPU boolean Prompt cache required')
    ids = np.flatnonzero(values)
    # Small lists avoid encoding-envelope overhead; larger sets never become
    # Python integer lists on the producer.
    if len(ids) <= 8:
        return ids.tolist()
    sparse_bytes = len(ids)*2
    if sparse_bytes <= (len(values)+7)//8:
        encoding, raw = _SPARSE, ids.astype('<u2').tobytes()
    else:
        encoding, raw = _BITSET, np.packbits(values, bitorder='little').tobytes()
    return dict(encoding=encoding, data=base64.b64encode(raw).decode('ascii'))


def cache_ids(snapshot, prompt_tokens):
    if type(prompt_tokens) is not int or not 1 <= prompt_tokens <= 32768:
        raise ValueError('bounded Prompt cache length required')
    if isinstance(snapshot, list):
        if (len(snapshot) > prompt_tokens or any(type(t) is not int or not 0 <= t < prompt_tokens for t in snapshot)
                or len(set(snapshot)) != len(snapshot)):
            raise ValueError('invalid cache ID list')
        return snapshot
    if (not isinstance(snapshot, dict) or set(snapshot) != {'encoding', 'data'}
            or snapshot['encoding'] not in (_SPARSE, _BITSET) or not isinstance(snapshot['data'], str)):
        raise ValueError('exact compact cache encoding required')
    max_bytes = prompt_tokens*2 if snapshot['encoding'] == _SPARSE else (prompt_tokens+7)//8
    encoded = snapshot['data']
    if len(encoded) > 4*((max_bytes+2)//3):
        raise ValueError('cache encoding exceeds Prompt bound')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError('malformed cache bytes') from exc
    if base64.b64encode(raw).decode('ascii') != encoded:
        raise ValueError('noncanonical cache bytes')
    if snapshot['encoding'] == _SPARSE:
        if len(raw) % 2 or len(raw) > max_bytes:
            raise ValueError('invalid sparse cache extent')
        ids = np.frombuffer(raw, dtype='<u2')
        if len(ids) and (int(ids[-1]) >= prompt_tokens or np.any(ids[1:] <= ids[:-1])):
            raise ValueError('sparse cache IDs must be increasing and in Prompt')
        return ids
    if len(raw) != max_bytes or (prompt_tokens % 8 and raw[-1] >> (prompt_tokens % 8)):
        raise ValueError('invalid bitmap extent or padding')
    return np.flatnonzero(np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder='little')[:prompt_tokens])
