"""Process incarnation identity shared by every PVD role in one worker.

An epoch identifies the OS process that owns a transfer engine and its
registered memory.  It is deliberately *not* derived from a request, an
adapter instance, a role or a rank: two adapters inside one worker process
share one incarnation, and two worker processes must never share one, because
the epoch is what lets a receiver refuse a write authorization that belonged
to a previous incarnation of the sender.

A P worker with compute TP1 legitimately uploads to several V storage shards
under the same sender epoch.  Those uploads are still distinguished by their
own write identities (see ``protocol.upload_transfer_id``); the epoch is not
the uniqueness mechanism.
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import Optional

__all__ = ["worker_epoch", "reset_worker_epoch", "current_worker_epoch"]

_LOCK = threading.Lock()
_EPOCH: Optional[str] = None
_EPOCH_PID: Optional[int] = None


def _new_epoch_locked(pid: int) -> str:
    global _EPOCH, _EPOCH_PID
    _EPOCH_PID = pid
    _EPOCH = f"{pid}-{uuid.uuid4().hex}"
    return _EPOCH


def worker_epoch() -> str:
    """Return this worker process's incarnation id, creating it on first use.

    The recorded PID is re-checked on every call, so a process that inherited
    module state through ``fork`` mints a new incarnation instead of presenting
    its parent's.  ``spawn`` re-imports this module and therefore starts empty.
    """
    global _EPOCH, _EPOCH_PID
    pid = os.getpid()
    with _LOCK:
        if _EPOCH is None or _EPOCH_PID != pid:
            return _new_epoch_locked(pid)
        return _EPOCH


def current_worker_epoch() -> Optional[str]:
    """Return the epoch only if one was already minted. Never creates one."""
    with _LOCK:
        if _EPOCH is not None and _EPOCH_PID == os.getpid():
            return _EPOCH
        return None


def reset_worker_epoch() -> str:
    """Mint a fresh incarnation.

    Used by the fork hook and by tests.  Production code must not call this:
    changing the epoch while writes are outstanding orphans their authorizations
    on the receiver, which is exactly the condition that keeps remote memory
    pinned until a coordinated restart.
    """
    with _LOCK:
        return _new_epoch_locked(os.getpid())


def _before_fork() -> None:
    _LOCK.acquire()


def _after_fork_in_parent() -> None:
    _LOCK.release()


def _after_fork_in_child() -> None:
    global _EPOCH, _EPOCH_PID
    # The child inherits a copy of the parent's state, including a lock that
    # another parent thread may have been holding at fork time.  Clear both so
    # the child neither deadlocks nor reuses the parent's incarnation.
    _EPOCH = None
    _EPOCH_PID = None
    try:
        _LOCK.release()
    except RuntimeError:  # pragma: no cover - already released
        pass


if hasattr(os, "register_at_fork"):  # pragma: no branch - POSIX only
    os.register_at_fork(
        before=_before_fork,
        after_in_parent=_after_fork_in_parent,
        after_in_child=_after_fork_in_child,
    )
