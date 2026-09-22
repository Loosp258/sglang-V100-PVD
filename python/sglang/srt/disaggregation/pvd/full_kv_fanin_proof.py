"""Shared coordinator/receiver validation; never turn cancellation into proof."""

from typing import Mapping

from sglang.srt.disaggregation.pvd.full_kv_fanin_plan import FULL_KV_FANIN_PROTOCOL
from sglang.srt.disaggregation.pvd.protocol import (
    ProtocolValidationError,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransportState


def validate_fanin_proof(reply, *, fingerprint, identities, byte_counts):
    fields = {
        "protocol",
        "plan_fingerprint",
        "source_rank",
        "identity",
        "fenced",
        "transport_state",
        "transferred_bytes",
    }
    if not isinstance(reply, Mapping) or set(reply) != fields:
        raise ProtocolValidationError("complete fan-in closure reply required")
    rank = reply["source_rank"]
    if type(rank) is not int or rank not in identities:
        raise ProtocolValidationError("unknown or unadopted V writer")
    identity = WriteIdentity.from_dict(reply["identity"])
    if (
        reply["protocol"] != FULL_KV_FANIN_PROTOCOL
        or reply["plan_fingerprint"] != fingerprint
        or identity != identities[rank]
        or type(reply["fenced"]) is not bool
    ):
        raise ProtocolValidationError("fan-in closure identity/plan mismatch")
    try:
        state = TransportState(reply["transport_state"])
    except (ValueError, TypeError) as exc:
        raise ProtocolValidationError("unknown transport state") from exc
    transferred = reply["transferred_bytes"]
    if type(transferred) is not int or not 0 <= transferred <= byte_counts[rank]:
        raise ProtocolValidationError("invalid writer transferred byte count")
    if not reply["fenced"]:
        return rank, None
    if not state.is_locally_safe_to_release:
        raise ProtocolValidationError("fenced reply lacks terminal transport proof")
    if (state == TransportState.NOT_SUBMITTED and transferred != 0) or (
        state == TransportState.TERMINAL_SUCCESS and transferred != byte_counts[rank]
    ):
        raise ProtocolValidationError("terminal state contradicts byte coverage")
    return rank, (state, transferred)
