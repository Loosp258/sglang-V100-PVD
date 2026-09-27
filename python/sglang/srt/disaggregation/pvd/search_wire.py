"""Bounded float32 query rows inside the existing PVD search-batch envelope.

Only Q travels here. Prompt KV still uses the owned Mooncake data path. The
packed representation is opt-in so older V shards keep receiving JSON rows.
"""

import base64
import binascii

import numpy as np

PACKED_QUERY_ENCODING = "f32le-base64-v1"
MAX_PACKED_QUERY_CELLS = 100_000


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
