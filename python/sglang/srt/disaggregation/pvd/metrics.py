"""Small dependency-free metrics registry for the standalone V service."""

from __future__ import annotations

import threading
from collections import Counter
from typing import Dict


class PVDMetrics:
    def __init__(self) -> None:
        self._counters = Counter()
        self._gauges: Dict[str, int] = {}
        self._lock = threading.Lock()

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: int) -> None:
        with self._lock:
            self._gauges[name] = value

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
            }
