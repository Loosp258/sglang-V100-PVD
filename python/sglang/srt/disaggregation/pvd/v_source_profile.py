"""Bounded wall-clock diagnostics, never a CUDA or transport release proof."""

import math
import threading
import time
from contextlib import contextmanager


SOURCE_PHASES = (
    "allocate", "pin_index", "pack", "pack_fence", "register", "outer_fence", "submit",
)


class VSourceProfile:
    def __init__(self, *, nbytes, cuda, kernel, reuse_pack_fence=False, selected_component_views=False):
        self.nbytes, self.cuda, self.kernel = nbytes, cuda, kernel
        self.reuse_pack_fence = reuse_pack_fence
        self._outer_fence_reused = False
        self.selected_component_views = selected_component_views
        self._source_component_views = 0
        self._lock = threading.Lock()
        self._phases = {
            name: dict(calls=0, successes=0, seconds=0.0) for name in SOURCE_PHASES
        }

    @contextmanager
    def measure(self, phase):
        if phase not in SOURCE_PHASES:
            raise ValueError("unknown V source phase")
        started = time.perf_counter()
        success = False
        try:
            yield
            success = True
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                record = self._phases[phase]
                record["calls"] += 1
                record["successes"] += int(success)
                record["seconds"] += elapsed

    def snapshot(self):
        with self._lock:
            return dict(
                schema=1, nbytes=self.nbytes, cuda=self.cuda, kernel=self.kernel,
                reuse_pack_fence=self.reuse_pack_fence,
                outer_fence_reused=self._outer_fence_reused,
                selected_component_views=self.selected_component_views,
                source_component_views=self._source_component_views,
                phases={name: dict(record) for name, record in self._phases.items()},
            )

    def record_outer_fence_reuse(self):
        with self._lock:
            self._outer_fence_reused = True

    def record_component_views(self, count):
        with self._lock:
            self._source_component_views = count


def copy_source_profile(value, *, nbytes):
    """Copy only fixed diagnostic fields; do not trust them as terminal proof."""
    if (not isinstance(value, dict) or type(value.get("schema")) is not int
            or value["schema"] != 1):
        raise ValueError("unknown V source profile schema")
    if type(value.get("nbytes")) is not int or value["nbytes"] != nbytes:
        raise ValueError("V source profile byte count mismatch")
    if type(value.get("cuda")) is not bool or value.get("kernel") not in (
        "torch", "triton", "torch_contiguous_runs",
    ):
        raise ValueError("invalid V source profile mode")
    if any(type(value.get(name)) is not bool for name in (
        "reuse_pack_fence", "outer_fence_reused",
    )):
        raise ValueError("invalid V source fence diagnostic")
    selected = value.get("selected_component_views", False)
    count = value.get("source_component_views", 0)
    if type(selected) is not bool or type(count) is not int or not 0 <= count <= 65536:
        raise ValueError("invalid V source component diagnostic")
    phases = value.get("phases")
    if not isinstance(phases, dict) or set(phases) != set(SOURCE_PHASES):
        raise ValueError("invalid V source phases")
    copied = {}
    for name in SOURCE_PHASES:
        record = phases[name]
        if not isinstance(record, dict):
            raise ValueError("invalid V source phase record")
        calls, successes, seconds = (record.get(key) for key in ("calls", "successes", "seconds"))
        if (type(calls) is not int or type(successes) is not int
                or not 0 <= successes <= calls <= 1
                or type(seconds) not in (int, float)
                or not math.isfinite(seconds) or seconds < 0):
            raise ValueError("invalid V source timing or count")
        copied[name] = dict(calls=calls, successes=successes, seconds=seconds)
    return dict(schema=1, nbytes=nbytes, cuda=value["cuda"], kernel=value["kernel"],
                reuse_pack_fence=value["reuse_pack_fence"],
                outer_fence_reused=value["outer_fence_reused"], phases=copied,
                selected_component_views=selected, source_component_views=count)
