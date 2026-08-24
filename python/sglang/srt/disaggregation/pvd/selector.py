"""Selector abstraction for future vector search implementations."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterable, List, Optional

from sglang.srt.disaggregation.pvd.protocol import KVEntryKey
from sglang.srt.disaggregation.pvd.request_state import EntryState


@dataclass(frozen=True)
class SelectionResult:
    key: KVEntryKey
    state: Optional[EntryState]
    found: bool
    message: Optional[str] = None

    def to_dict(self):
        return {
            "key": self.key.to_dict(),
            "state": self.state.value if self.state is not None else None,
            "found": self.found,
            "message": self.message,
        }


class Selector(abc.ABC):
    @abc.abstractmethod
    def select(self, keys: Iterable[KVEntryKey]) -> List[SelectionResult]: ...


class PassThroughSelector(Selector):
    """Identity lookup: every requested key remains an independent result."""

    def __init__(self, state_lookup):
        self._state_lookup = state_lookup

    def select(self, keys: Iterable[KVEntryKey]) -> List[SelectionResult]:
        results = []
        for key in keys:
            state = self._state_lookup(key)
            results.append(
                SelectionResult(
                    key=key,
                    state=state,
                    found=state is not None,
                    message=None if state is not None else "entry not found",
                )
            )
        return results
