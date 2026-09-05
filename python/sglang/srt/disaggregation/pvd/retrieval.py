"""Versioned retrieval contract and per-sequence Decode refresh clock.

Selection is deliberately explicit: full_prompt is the only implementation.
Future sparse selectors must return logical token ranges and update the D
unpacker; silently treating an unknown selector as full_prompt is forbidden.
"""

from dataclasses import dataclass

from sglang.srt.disaggregation.pvd.protocol import KVEntryKey, RemoteRegionDescriptor


@dataclass(frozen=True)
class RetrievalRequest:
    key: KVEntryKey
    sequence_id: str
    delivery_id: str
    destinations: dict[int, RemoteRegionDescriptor]
    selection: str = "full_prompt"

    @classmethod
    def from_dict(cls, value):
        if value.get("selection") != "full_prompt":
            raise ValueError(
                "unsupported PVD selection; only full_prompt is implemented"
            )
        key = KVEntryKey.from_dict(value["key"])
        sequence_id = value["sequence_id"]
        delivery_id = value["delivery_id"]
        if sequence_id != key.req_id:
            raise ValueError("sequence_id must match the cross-role Entry request ID")
        if not isinstance(delivery_id, str) or not delivery_id.strip():
            raise ValueError("delivery_id must be non-empty")
        return cls(
            key,
            sequence_id,
            delivery_id,
            {
                int(rank): RemoteRegionDescriptor.from_dict(region)
                for rank, region in value["destinations"].items()
            },
        )


class RefreshClock:
    """Advance only after a matching delivery, local unpack and ACK complete."""

    def __init__(self, delivery_prefix: str, interval: int):
        if (
            not delivery_prefix
            or isinstance(interval, bool)
            or not isinstance(interval, int)
            or interval <= 0
        ):
            raise ValueError("refresh interval and delivery prefix must be valid")
        self.prefix = delivery_prefix
        self.interval = interval
        self.round = 0
        self.last_tokens = None
        self.pending = None

    def due(self, decode_tokens: int) -> bool:
        if decode_tokens < 0 or (
            self.last_tokens is not None and decode_tokens < self.last_tokens
        ):
            raise ValueError("Decode token count regressed")
        return (
            self.last_tokens is None
            or decode_tokens - self.last_tokens >= self.interval
        )

    def begin(self, decode_tokens: int) -> str:
        if self.pending is not None:
            raise ValueError("refresh already in flight")
        if not self.due(decode_tokens):
            raise ValueError("refresh is not due")
        delivery_id = f"{self.prefix}:refresh:{self.round}"
        self.pending = (delivery_id, decode_tokens)
        return delivery_id

    def complete(self, delivery_id: str) -> None:
        if self.pending is None or self.pending[0] != delivery_id:
            raise ValueError("stale refresh completion")
        self.last_tokens = self.pending[1]
        self.pending = None
        self.round += 1
