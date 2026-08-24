"""Prefill-Vector-Decode (PVD) disaggregation primitives.

PVD is intentionally isolated from the existing PD implementation. The
package contains the versioned control protocol, lifecycle state machines and
the vector-node GPU page store. Runtime integration can select this package
without changing the default P -> D path.
"""

from sglang.srt.disaggregation.pvd.protocol import (
    FirstTokenMetadata,
    KVEntryKey,
    KVEntryManifest,
    KVLayoutSignature,
    KVShardManifest,
    PVD_PROTOCOL_VERSION,
    RemoteRegionDescriptor,
)
from sglang.srt.disaggregation.pvd.request_state import (
    DeliveryState,
    EntryShardState,
    EntryState,
    InvalidStateTransition,
)

__all__ = [
    "DeliveryState",
    "EntryShardState",
    "EntryState",
    "FirstTokenMetadata",
    "InvalidStateTransition",
    "KVEntryKey",
    "KVEntryManifest",
    "KVLayoutSignature",
    "KVShardManifest",
    "PVD_PROTOCOL_VERSION",
    "RemoteRegionDescriptor",
]
