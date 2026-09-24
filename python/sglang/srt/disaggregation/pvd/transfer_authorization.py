"""Identity-complete authorization gates for PVD remote writes."""

from __future__ import annotations

import threading
import uuid
from typing import Any, ClassVar

from sglang.srt.disaggregation.pvd.protocol import (
    PVD_TRANSFER_LIFECYCLE_PROTOCOL,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    GuardUnpinOutcome,
    ResourceGuard,
    TransportState,
)

__all__ = [
    "PVD_TRANSFER_LIFECYCLE_PROTOCOL",
    "WriteAuthorization",
    "WriteIdentity",
]


class WriteAuthorization:
    """A one-shot write permission whose pin survives until terminal closure."""

    _TERMINAL_STATES: ClassVar[frozenset[TransportState]] = frozenset(
        {
            TransportState.NOT_SUBMITTED,
            TransportState.TERMINAL_SUCCESS,
            TransportState.TERMINAL_FAILED,
        }
    )

    def __init__(self, identity: WriteIdentity, guard: ResourceGuard) -> None:
        if not isinstance(identity, WriteIdentity):
            raise ValueError("identity must be a WriteIdentity")  # noqa: TRY004
        if not isinstance(guard, ResourceGuard):
            raise ValueError("guard must be a ResourceGuard")  # noqa: TRY004
        self._identity = identity
        self._guard = guard
        # ResourceGuard counts owners, not identities. Even an accidentally
        # duplicated authorization must not drop another object's resource pin.
        # The sender's registry must separately enforce one gate per identity.
        self._owner = "pvd-write:" + uuid.uuid4().hex
        self._guard.pin(self._owner)
        self._closed = False
        self._begun = False
        self._terminal_state: TransportState | None = None
        self._unpinned = False
        self._lock = threading.Lock()
        self._cleanup_lock = threading.Lock()

    @property
    def identity(self) -> WriteIdentity:
        return self._identity

    def begin(self, identity: WriteIdentity) -> None:
        with self._lock:
            self._require_identity(identity)
            if self._closed:
                raise ValueError("write authorization is closed")
            if self._begun:
                raise ValueError("write authorization has already begun")
            self._begun = True

    def close(self) -> None:
        with self._lock:
            self._closed = True

    @property
    def cleanup_complete(self) -> bool:
        with self._lock:
            if self._terminal_state is None:
                return False
        with self._cleanup_lock:
            return self._unpinned

    def observe_terminal(self, identity: WriteIdentity, state: TransportState) -> None:
        with self._lock:
            self._require_identity(identity)
            if not self._closed:
                raise ValueError(
                    "write authorization must be closed before terminal state"
                )
            if (
                not isinstance(state, TransportState)
                or state not in self._TERMINAL_STATES
            ):
                raise ValueError(
                    "write authorization requires a transport terminal state"
                )
            if state == TransportState.NOT_SUBMITTED and self._begun:
                raise ValueError(
                    "NOT_SUBMITTED cannot terminate an authorization that began"
                )
            if self._terminal_state is not None and self._terminal_state != state:
                raise ValueError(
                    "write authorization already has a different terminal state"
                )
            self._terminal_state = state
        # Callbacks run outside the authorization lock; they may inspect fence.
        # Retrying a terminal report must also retry a failed local unregister.
        with self._cleanup_lock:
            if not self._unpinned:
                outcome = self._guard.unpin(self._owner)
                self._unpinned = outcome != GuardUnpinOutcome.RELEASE_IN_PROGRESS

    def fence(self, identity: WriteIdentity) -> dict[str, Any]:
        with self._lock:
            matches = isinstance(identity, WriteIdentity) and identity == self._identity
            return {
                **self._identity.to_dict(),
                "fenced": bool(
                    matches and self._closed and self._terminal_state is not None
                ),
            }

    def _require_identity(self, identity: WriteIdentity) -> None:
        if not isinstance(identity, WriteIdentity) or identity != self._identity:
            raise ValueError("write authorization identity does not match")
