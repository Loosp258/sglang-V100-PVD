"""V-side retrieval-index lifecycle, separate from the Entry's KV lifecycle.

This module is NOT wired into the V control server. It owns no vectors, no
graph storage and no GPU memory; it refuses illegal orderings and records
which index a search is allowed to use. Building, searching and freeing are
the owner's job, and every transition here is the owner asserting a fact it
has already established.

The rules it enforces, from the design:

* An index may only be built after the complete Prompt KV upload and safe GPU
  visibility. A partially written Entry must never be indexed.
* ABSENT, BUILDING, READY and FAILED are distinguishable. "Not ready" is not
  "failed", and a failed build is not a missing one.
* **Full initial delivery must not acquire an INDEX_READY dependency.** A D
  worker pulling complete Prompt KV needs KV_STORED and nothing else, so
  ``deliverable`` is deliberately independent of index state.
* A Prompt index is immutable and reused across Delivery rounds. Readiness is
  not consumed by a search.
* Model revision, index version, query version and memory generation are
  distinct identities and are never conflated. A search is refused unless the
  query's vector space and the index's agree, and the vector space compared
  is **the caller's own**: an identity this gate filled in for the caller
  would prove nothing.
* A build that never started is not a build that failed. ``abandon_build``
  exists so that transient pressure -- no capacity to place the copies right
  now -- is backpressure, retried later, rather than one of the few
  permanent attempts an Entry is allowed.

Closing refuses further use and retains the descriptor. It frees nothing:
vectors, graph storage and id mappings are released by whoever allocated
them, after their own safety checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class IndexState(str, Enum):
    ABSENT = "absent"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class IndexDescriptor:
    """What a built index is, and what it may be searched with.

    ``index_version`` identifies this build. ``vector_space`` is the model
    whose K the vectors came from; a query in another space is refused.
    ``id_mapping_version`` is tracked separately because the mapping from
    graph ids back to Entry tokens/pages can be rebuilt without rebuilding
    the vectors, and conflating the two would let stale ids be trusted.
    """

    index_version: str
    entry_transfer_id: str
    vector_space: str
    id_mapping_version: str
    vector_count: int
    metric: str

    def __post_init__(self) -> None:
        for name in (
            "index_version",
            "entry_transfer_id",
            "vector_space",
            "id_mapping_version",
            "metric",
        ):
            _require_text(name, getattr(self, name))
        if (
            isinstance(self.vector_count, bool)
            or not isinstance(self.vector_count, int)
            or self.vector_count <= 0
        ):
            raise ValueError("vector_count must be a positive integer")


class IndexGate:
    """One immutable Prompt index per Entry shard."""

    def __init__(self, entry_transfer_id: str, *, max_build_attempts: int = 3):
        _require_text("entry_transfer_id", entry_transfer_id)
        if (
            isinstance(max_build_attempts, bool)
            or not isinstance(max_build_attempts, int)
            or max_build_attempts <= 0
        ):
            raise ValueError("max_build_attempts must be a positive integer")
        self.entry_transfer_id = entry_transfer_id
        self.max_build_attempts = max_build_attempts
        self._state = IndexState.ABSENT
        self._kv_readable = False
        self._descriptor: Optional[IndexDescriptor] = None
        self._attempts = 0
        self._error: Optional[str] = None
        # What to return to if an attempt turns out never to have started.
        self._state_before_build = IndexState.ABSENT
        self._deferrals = 0
        self._deferred_reason: Optional[str] = None

    # -- observation --------------------------------------------------------

    @property
    def state(self) -> IndexState:
        return self._state

    @property
    def descriptor(self) -> Optional[IndexDescriptor]:
        return self._descriptor

    @property
    def attempts(self) -> int:
        return self._attempts

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def deferrals(self) -> int:
        """How often an attempt was abandoned before it allocated anything.

        Deliberately not ``attempts``: these cost the Entry nothing and are
        retried, so they must not push it towards ``exhausted``.
        """
        return self._deferrals

    @property
    def deferred_reason(self) -> Optional[str]:
        return self._deferred_reason

    @property
    def kv_readable(self) -> bool:
        return self._kv_readable

    @property
    def deliverable(self) -> bool:
        """Whether complete Prompt KV may be delivered from this Entry.

        Index state is deliberately absent from this answer. Bootstrap pulls
        the whole prompt and needs no index; making delivery wait for
        INDEX_READY would add a dependency the design forbids.
        """
        return self._kv_readable and self._state is not IndexState.CLOSED

    @property
    def searchable(self) -> bool:
        """Only a completed build may be searched. Not consumed by a search."""
        return self._state is IndexState.READY

    @property
    def exhausted(self) -> bool:
        return (
            self._state is IndexState.FAILED
            and self._attempts >= self.max_build_attempts
        )

    def _require_open(self) -> None:
        if self._state is IndexState.CLOSED:
            raise ValueError("index gate is closed")

    # -- transitions --------------------------------------------------------

    def mark_kv_readable(self) -> None:
        """The complete Prompt KV is stored and safely visible on this device.

        The caller has already established both. This only records it, and it
        is what unblocks building; delivery does not wait on the index.
        """
        self._require_open()
        self._kv_readable = True

    def begin_build(self) -> None:
        self._require_open()
        if not self._kv_readable:
            raise ValueError(
                "an index may not be built before the complete Prompt KV is "
                "stored and visible"
            )
        if self._state is IndexState.BUILDING:
            raise ValueError("an index build is already in progress")
        if self._state is IndexState.READY:
            raise ValueError("the Prompt index is immutable and already built")
        if self.exhausted:
            raise ValueError(
                f"index build for {self.entry_transfer_id} has failed "
                f"{self._attempts} times; refusing another attempt"
            )
        self._state_before_build = self._state
        self._attempts += 1
        self._error = None
        self._state = IndexState.BUILDING

    def mark_ready(self, descriptor: IndexDescriptor) -> None:
        self._require_open()
        if self._state is not IndexState.BUILDING:
            raise ValueError("no index build is in progress")
        if not isinstance(descriptor, IndexDescriptor):
            raise ValueError("an index descriptor is required")
        if descriptor.entry_transfer_id != self.entry_transfer_id:
            raise ValueError("index descriptor belongs to a different Entry")
        self._descriptor = descriptor
        self._error = None
        self._state = IndexState.READY

    def abandon_build(self, reason: str) -> None:
        """Undo an attempt that never allocated anything. Not a failure.

        The gate returns to the state it was in before ``begin_build``, and
        the attempt is given back. Use this only when nothing was built and
        nothing is held -- typically because the owner could not reserve the
        memory the copies would need. Calling it for a genuine build error
        would let a permanently broken Entry be retried forever.
        """
        self._require_open()
        if self._state is not IndexState.BUILDING:
            raise ValueError("no index build is in progress")
        self._deferred_reason = _require_text("reason", reason)
        self._deferrals += 1
        self._attempts -= 1
        self._error = None
        self._state = self._state_before_build

    def mark_failed(self, reason: str) -> None:
        """Record a failed build. Distinct from 'not built yet'."""
        self._require_open()
        if self._state is not IndexState.BUILDING:
            raise ValueError("no index build is in progress")
        self._error = _require_text("reason", reason)
        self._deferred_reason = None
        self._state = IndexState.FAILED

    def authorize_search(
        self,
        vector_space: str,
        *,
        expected_id_mapping_version: Optional[str] = None,
        expected_index_version: Optional[str] = None,
        entry_transfer_id: Optional[str] = None,
    ) -> Tuple[IndexDescriptor, Tuple[str, ...]]:
        """Return the descriptor a search may use and what was checked.

        ``vector_space`` is the **caller's** claim about the Q it is holding
        and is always required: an index built from another model would
        happily answer a same-shaped query with confident nonsense. The two
        version pins are optional because a first-time caller has not seen a
        version yet -- but they are never filled in from this gate's own
        descriptor, because comparing a value against itself is not a check.
        The returned tuple names exactly which comparisons were made, so a
        caller that did not pin can see that it did not.
        """
        if self._state is IndexState.CLOSED:
            raise ValueError("index gate is closed")
        if not self.searchable or self._descriptor is None:
            raise ValueError(
                f"index for {self.entry_transfer_id} is {self._state.value}, "
                "not ready to search"
            )
        _require_text("vector_space", vector_space)
        checked = ["vector_space"]
        if self._descriptor.vector_space != vector_space:
            raise ValueError(
                f"query is in {vector_space!r} but the index holds "
                f"{self._descriptor.vector_space!r}"
            )
        if entry_transfer_id is not None:
            checked.append("entry_transfer_id")
            if self._descriptor.entry_transfer_id != entry_transfer_id:
                raise ValueError(
                    f"query names Entry {entry_transfer_id!r} but this index "
                    f"holds {self._descriptor.entry_transfer_id!r}"
                )
        if expected_id_mapping_version is not None:
            checked.append("id_mapping_version")
            if self._descriptor.id_mapping_version != expected_id_mapping_version:
                raise ValueError("index id mapping has been rebuilt since this query")
        if expected_index_version is not None:
            checked.append("index_version")
            if self._descriptor.index_version != expected_index_version:
                raise ValueError("the index has been rebuilt since this query")
        return self._descriptor, tuple(checked)

    def close(self) -> None:
        """Refuse further use. Frees no vectors, graph storage or mappings."""
        self._state = IndexState.CLOSED

    def snapshot(self) -> dict:
        return {
            "entry_transfer_id": self.entry_transfer_id,
            "state": self._state.value,
            "kv_readable": self._kv_readable,
            "deliverable": self.deliverable,
            "searchable": self.searchable,
            "attempts": self._attempts,
            "deferrals": self._deferrals,
            "exhausted": self.exhausted,
            "error": self._error,
            "deferred_reason": self._deferred_reason,
            "index_version": (
                self._descriptor.index_version if self._descriptor else None
            ),
        }
