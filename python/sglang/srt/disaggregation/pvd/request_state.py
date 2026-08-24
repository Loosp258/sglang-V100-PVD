"""Strict lifecycle state machines for PVD entries and deliveries."""

from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, TypeVar


class InvalidStateTransition(RuntimeError):
    pass


class EntryState(str, Enum):
    CREATED = "created"
    ALLOCATING = "allocating"
    P_WRITING = "p_writing"
    STORED = "stored"
    RELEASING = "releasing"
    RELEASED = "released"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class EntryShardState(str, Enum):
    CREATED = "created"
    ALLOCATED = "allocated"
    P_WRITING = "p_writing"
    STORED = "stored"
    RELEASING = "releasing"
    RELEASED = "released"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class DeliveryState(str, Enum):
    WAITING_SOURCE = "waiting_source"
    D_RESERVED = "d_reserved"
    V_WRITING = "v_writing"
    DELIVERED = "delivered"
    ACKED = "acked"
    RELEASED = "released"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


ENTRY_TERMINAL_STATES = frozenset(
    {EntryState.RELEASED, EntryState.FAILED, EntryState.CANCELLED, EntryState.EXPIRED}
)
ENTRY_SHARD_TERMINAL_STATES = frozenset(
    {
        EntryShardState.RELEASED,
        EntryShardState.FAILED,
        EntryShardState.CANCELLED,
        EntryShardState.EXPIRED,
    }
)
DELIVERY_TERMINAL_STATES = frozenset(
    {
        DeliveryState.RELEASED,
        DeliveryState.FAILED,
        DeliveryState.CANCELLED,
        DeliveryState.EXPIRED,
    }
)


_ENTRY_TRANSITIONS: Dict[EntryState, FrozenSet[EntryState]] = {
    EntryState.CREATED: frozenset(
        {EntryState.ALLOCATING, EntryState.FAILED, EntryState.CANCELLED}
    ),
    EntryState.ALLOCATING: frozenset(
        {EntryState.P_WRITING, EntryState.FAILED, EntryState.CANCELLED}
    ),
    EntryState.P_WRITING: frozenset(
        {EntryState.STORED, EntryState.FAILED, EntryState.CANCELLED, EntryState.EXPIRED}
    ),
    EntryState.STORED: frozenset(
        {EntryState.RELEASING, EntryState.FAILED, EntryState.CANCELLED, EntryState.EXPIRED}
    ),
    EntryState.RELEASING: frozenset({EntryState.RELEASED, EntryState.FAILED}),
}

_ENTRY_SHARD_TRANSITIONS: Dict[EntryShardState, FrozenSet[EntryShardState]] = {
    EntryShardState.CREATED: frozenset(
        {EntryShardState.ALLOCATED, EntryShardState.FAILED, EntryShardState.CANCELLED}
    ),
    EntryShardState.ALLOCATED: frozenset(
        {EntryShardState.P_WRITING, EntryShardState.FAILED, EntryShardState.CANCELLED}
    ),
    EntryShardState.P_WRITING: frozenset(
        {
            EntryShardState.STORED,
            EntryShardState.FAILED,
            EntryShardState.CANCELLED,
            EntryShardState.EXPIRED,
        }
    ),
    EntryShardState.STORED: frozenset(
        {
            EntryShardState.RELEASING,
            EntryShardState.FAILED,
            EntryShardState.CANCELLED,
            EntryShardState.EXPIRED,
        }
    ),
    EntryShardState.RELEASING: frozenset(
        {EntryShardState.RELEASED, EntryShardState.FAILED}
    ),
}

_DELIVERY_TRANSITIONS: Dict[DeliveryState, FrozenSet[DeliveryState]] = {
    DeliveryState.WAITING_SOURCE: frozenset(
        {
            DeliveryState.D_RESERVED,
            DeliveryState.FAILED,
            DeliveryState.CANCELLED,
            DeliveryState.EXPIRED,
        }
    ),
    DeliveryState.D_RESERVED: frozenset(
        {
            DeliveryState.V_WRITING,
            DeliveryState.FAILED,
            DeliveryState.CANCELLED,
            DeliveryState.EXPIRED,
        }
    ),
    DeliveryState.V_WRITING: frozenset(
        {
            DeliveryState.DELIVERED,
            DeliveryState.FAILED,
            DeliveryState.CANCELLED,
            DeliveryState.EXPIRED,
        }
    ),
    DeliveryState.DELIVERED: frozenset(
        {
            DeliveryState.ACKED,
            DeliveryState.FAILED,
            DeliveryState.CANCELLED,
            DeliveryState.EXPIRED,
        }
    ),
    DeliveryState.ACKED: frozenset({DeliveryState.RELEASED}),
}


S = TypeVar("S", bound=Enum)


def transition(current: S, target: S) -> S:
    if current == target:
        return current
    if isinstance(current, EntryState) and isinstance(target, EntryState):
        allowed = _ENTRY_TRANSITIONS.get(current, frozenset())
    elif isinstance(current, EntryShardState) and isinstance(target, EntryShardState):
        allowed = _ENTRY_SHARD_TRANSITIONS.get(current, frozenset())
    elif isinstance(current, DeliveryState) and isinstance(target, DeliveryState):
        allowed = _DELIVERY_TRANSITIONS.get(current, frozenset())
    else:
        raise InvalidStateTransition(
            f"state type mismatch: {type(current).__name__} -> {type(target).__name__}"
        )
    if target not in allowed:
        raise InvalidStateTransition(f"invalid transition: {current.value} -> {target.value}")
    return target
