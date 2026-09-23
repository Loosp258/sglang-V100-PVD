"""cuVS CAGRA with a dedicated bounded RMM resource per layer/head index.

The native cap is reserved for the entire index lifetime, including workspace.
It is a conservative capacity reservation, not an estimate of final graph size.
No GPU compatibility or recall claim follows from importing this module.
"""

import ctypes
import importlib
import threading
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from sglang.srt.disaggregation.pvd.index_search import (
    BruteForceIndexBackend,
    BuiltIndex,
    IndexBackend,
    IndexCompletionUnknown,
    IndexSearchError,
)

_LOCKS = {}
_LOCKS_LOCK = threading.Lock()


@dataclass(eq=False)
class _NativeIndex:
    limit: object
    resources: object = None
    native: object = None
    vectors: object = None
    disposed: bool = False
    probe_allocations: list = field(default_factory=list)


class CagraNativeRuntime:
    """Actual cuVS calls; import only when a CUDA backend is explicitly chosen."""

    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("CAGRA requires an explicit CUDA device index")
        self.cuvs = importlib.import_module("cuvs")
        self.cagra = importlib.import_module("cuvs.neighbors.cagra")
        self.cp = importlib.import_module("cupy")
        self.mr = importlib.import_module("rmm.mr")
        self.Resources = importlib.import_module("cuvs.common").Resources
        for name in ("IndexParams", "SearchParams", "build", "search"):
            if not callable(getattr(self.cagra, name, None)):
                raise ValueError(f"CAGRA API missing {name}")
        for name in (
            "CudaMemoryResource",
            "LimitingResourceAdaptor",
            "get_current_device_resource",
            "set_current_device_resource",
        ):
            if not callable(getattr(self.mr, name, None)):
                raise ValueError(f"RMM API missing {name}")
        # Resolve the C symbols from the very extension the Python API uses.
        extension = importlib.import_module(self.cagra.Index.__module__)
        self.library = ctypes.CDLL(extension.__file__)
        self.alloc = self.library.cuvsRMMAlloc
        self.alloc.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.alloc.restype = ctypes.c_int
        self.free = self.library.cuvsRMMFree
        self.free.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        self.free.restype = ctypes.c_int
        with _LOCKS_LOCK:
            self.lock = _LOCKS.setdefault(self.device.index, threading.RLock())

    def synchronize(self):
        torch.cuda.synchronize(self.device)

    @contextmanager
    def scope(self, owner):
        # This lock covers every PVD CAGRA operation on this device. Other RMM
        # users must not replace its resource concurrently in the V process.
        with self.lock, torch.cuda.device(self.device):
            previous = self.mr.get_current_device_resource()
            self.mr.set_current_device_resource(owner.limit)
            try:
                yield
            finally:
                self.mr.set_current_device_resource(previous)

    def create(self, byte_cap):
        with self.lock, torch.cuda.device(self.device):
            limit = self.mr.LimitingResourceAdaptor(
                self.mr.CudaMemoryResource(), byte_cap
            )
        return _NativeIndex(limit)

    def _verify_allocator_bridge(self, owner):
        """A small allocation proves cuVS uses this exact Python RMM limiter."""
        handle = owner.resources.get_c_obj()
        pointer = ctypes.c_void_p()
        status = self.alloc(handle, ctypes.byref(pointer), 256)
        if pointer.value:
            owner.probe_allocations.append((pointer, 256))
        if status != 1 or not pointer.value:
            raise IndexCompletionUnknown("cuVS allocator bridge probe failed")
        seen = owner.limit.get_allocated_bytes()
        if self.free(handle, pointer, 256) != 1:
            raise IndexCompletionUnknown("cuVS allocator probe could not free memory")
        owner.probe_allocations.clear()
        owner.resources.sync()
        if seen != 256 or owner.limit.get_allocated_bytes() != 0:
            raise IndexCompletionUnknown(
                "cuVS and Python RMM do not share the bounded resource"
            )
        # Once sharing is proved, an over-cap allocation must fail before any
        # memory is returned. This exercises the bound used by native builds.
        pointer = ctypes.c_void_p()
        over_cap = owner.limit.get_allocation_limit() + 256
        status = self.alloc(handle, ctypes.byref(pointer), over_cap)
        if pointer.value:
            owner.probe_allocations.append((pointer, over_cap))
        if status == 1 or pointer.value:
            raise IndexCompletionUnknown("cuVS bypassed the native allocation cap")
        owner.resources.sync()
        if owner.limit.get_allocated_bytes() != 0:
            raise IndexCompletionUnknown("cuVS cap probe left allocations alive")

    def build(self, owner, vectors, *, metric, graph_degree, intermediate_degree):
        with self.scope(owner):
            owner.resources = self.Resources(
                stream=torch.cuda.current_stream(self.device).cuda_stream
            )
            self._verify_allocator_bridge(owner)
            owner.vectors = vectors  # Already an owned, budgeted extraction copy.
            params = self.cagra.IndexParams(
                metric="inner_product" if metric == "ip" else "sqeuclidean",
                graph_degree=graph_degree,
                intermediate_graph_degree=intermediate_degree,
                build_algo="ivf_pq",
            )
            owner.native = self.cagra.build(
                params, self.cp.from_dlpack(vectors), resources=owner.resources
            )
            self.synchronize()
            if owner.native.trained is not True or owner.native.dim != vectors.shape[1]:
                raise IndexSearchError(
                    "CAGRA returned an untrained or mismatched index"
                )

    def search(self, owner, queries, *, top_k, itopk_size):
        with self.scope(owner):
            rows = torch.empty(
                (len(queries), top_k), device=self.device, dtype=torch.uint32
            )
            scores = torch.empty(
                (len(queries), top_k), device=self.device, dtype=torch.float32
            )
            # Pass buffers explicitly so the wrapper cannot allocate uncharged
            # device_ndarray outputs through a different allocator.
            self.cagra.search(
                self.cagra.SearchParams(itopk_size=itopk_size),
                owner.native,
                self.cp.from_dlpack(queries),
                top_k,
                neighbors=self.cp.from_dlpack(rows),
                distances=self.cp.from_dlpack(scores),
                resources=owner.resources,
            )
            self.synchronize()
            return rows, scores

    def dispose(self, owner):
        if owner.disposed:
            return
        with self.scope(owner):
            self.synchronize()
            if owner.probe_allocations:
                # A C API failure may have returned a pointer without a reliable
                # ownership outcome. Do not guess whether it is safe to free it
                # again, or destroy the resources it might still use.
                raise IndexCompletionUnknown("cuVS allocator probe remains unresolved")
            # cuVS's Cython __dealloc__ destroys the native index. Keep its MR
            # and Resources alive until destruction and all streams complete.
            owner.native = None
            owner.resources = None
            self.synchronize()
            if owner.limit.get_allocated_bytes() != 0:
                raise IndexCompletionUnknown(
                    "CAGRA disposal left native allocations alive"
                )
            owner.vectors = None
            owner.disposed = True


class CagraIndexBackend(IndexBackend):
    name = "cagra"

    def __init__(
        self,
        *,
        device,
        native_bytes_per_index,
        graph_degree,
        intermediate_degree,
        itopk_size,
        _runtime=None,
    ):
        self._device = torch.device(device)
        for value in (
            native_bytes_per_index,
            graph_degree,
            intermediate_degree,
            itopk_size,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(
                    "explicit positive CAGRA capacity/graph/search bounds required"
                )
        if native_bytes_per_index < 256 or intermediate_degree < graph_degree:
            raise ValueError("invalid CAGRA allocation cap or graph degrees")
        if self._device.type != "cuda" and _runtime is None:
            raise ValueError("native CAGRA requires CUDA")
        self.cap = native_bytes_per_index
        self.graph_degree, self.intermediate_degree = graph_degree, intermediate_degree
        self.itopk_size = itopk_size
        self.runtime = (
            _runtime if _runtime is not None else CagraNativeRuntime(self._device)
        )
        if torch.device(self.runtime.device) != self._device:
            raise ValueError("CAGRA runtime device mismatch")
        self._lock = threading.RLock()
        self._owners = {}
        self._unknown = None

    @property
    def device(self):
        return self._device

    def _check(self):
        if self._unknown is not None:
            raise IndexCompletionUnknown(self._unknown)

    def _shape(self, rows, dim, metric):
        if (
            type(rows) is not int
            or type(dim) is not int
            or rows <= self.intermediate_degree
            or dim <= 0
            or metric not in {"ip", "l2"}
        ):
            raise IndexSearchError(
                "CAGRA requires rows above intermediate graph degree, positive dim and ip/l2"
            )

    def build_footprint(self, rows, dim, *, metric):
        self._shape(rows, dim, metric)
        return self.cap

    def build_scratch_footprint(self, rows, dim, *, metric):
        self._shape(rows, dim, metric)
        # Native temporary bytes fit within the retained cap. This is the
        # additional Torch finite-input checks, including temporary masks/abs
        # tensors (not just the final bool mask), and scalar reductions.
        return rows * dim * 8 + 64

    def search_footprint(self, rows, dim, num_queries, top_k):
        if any(
            type(v) is not int or v <= 0 for v in (rows, dim, num_queries, top_k)
        ) or top_k > min(rows, self.itopk_size):
            raise IndexSearchError("CAGRA query/top-k exceeds configured bounds")
        return num_queries * top_k * 9 + num_queries * dim * 8 + 64

    def _matrix(self, tensor):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 2
            or tensor.dtype != torch.float32
            or tensor.device != self.device
            or not tensor.is_contiguous()
        ):
            raise IndexSearchError(
                "CAGRA requires contiguous float32 matrices on its declared device"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise IndexSearchError("CAGRA vectors/queries must be finite")

    def synchronize(self):
        with self._lock:
            self._check()
            try:
                self.runtime.synchronize()
            except BaseException as exc:
                self._unknown = str(exc)
                raise IndexCompletionUnknown(str(exc)) from exc

    def build(self, vectors, *, vector_space, metric):
        with self._lock:
            self._check()
            if not isinstance(vector_space, str) or not vector_space.strip():
                raise IndexSearchError("explicit CAGRA vector space required")
            self._matrix(vectors)
            rows, dim = vectors.shape
            self._shape(rows, dim, metric)
            owner = self.runtime.create(self.cap)
            self._owners[id(owner)] = owner
            try:
                self.runtime.build(
                    owner,
                    vectors,
                    metric=metric,
                    graph_degree=self.graph_degree,
                    intermediate_degree=self.intermediate_degree,
                )
                self.runtime.synchronize()
                return BuiltIndex(vector_space, metric, dim, rows, owner)
            except BaseException as exc:
                try:
                    self.runtime.synchronize()
                    traceback.clear_frames(exc.__traceback__)
                    self.runtime.dispose(owner)
                except BaseException as cleanup:
                    self._unknown = str(cleanup)
                    raise IndexCompletionUnknown(str(cleanup)) from cleanup
                self._owners.pop(id(owner))
                if isinstance(exc, IndexCompletionUnknown):
                    self._unknown = str(exc)
                raise

    def search(self, index, queries, *, top_k):
        with self._lock:
            self._check()
            if (
                self._owners.get(id(index.handle)) is not index.handle
                or index.handle.disposed
            ):
                raise IndexSearchError("unknown or disposed CAGRA index")
            self._matrix(queries)
            if queries.shape[1] != index.dim:
                raise IndexSearchError("CAGRA query dimension mismatch")
            self.search_footprint(index.count, index.dim, len(queries), top_k)
            try:
                rows, scores = self.runtime.search(
                    index.handle, queries, top_k=top_k, itopk_size=self.itopk_size
                )
                self.runtime.synchronize()
                if index.metric == "l2":
                    if bool((scores < 0).any()):
                        raise IndexSearchError(
                            "CAGRA returned negative squared-L2 distance"
                        )
                    scores.sqrt_().neg_()
                self.runtime.synchronize()
                return rows, scores
            except BaseException as exc:
                try:
                    self.runtime.synchronize()
                except BaseException as cleanup:
                    self._unknown = str(cleanup)
                    raise IndexCompletionUnknown(str(cleanup)) from cleanup
                if isinstance(exc, IndexCompletionUnknown):
                    self._unknown = str(exc)
                raise

    def dispose(self, index):
        with self._lock:
            self._check()
            owner = index.handle
            if getattr(owner, "disposed", False):
                return
            if self._owners.get(id(owner)) is not owner:
                raise IndexSearchError("foreign CAGRA index")
            try:
                self.runtime.dispose(owner)
            except BaseException as exc:
                self._unknown = str(exc)
                raise IndexCompletionUnknown(str(exc)) from exc
            self._owners.pop(id(owner))


class CagraAutoIndexBackend(IndexBackend):
    """CAGRA for supported row counts, exact on-device search for short K.

    The mode is explicit: pure ``cagra`` still refuses short prompts. Both
    paths declare the same device, while their actual retained and scratch
    footprints are charged independently by PromptIndexManager. A native
    UNKNOWN poisons this whole mode, including its exact fallback, because
    the shared CUDA device/resource state can no longer be trusted.
    """

    name = "cagra_auto"

    def __init__(self, cagra: CagraIndexBackend):
        if not isinstance(cagra, CagraIndexBackend):
            raise TypeError("a configured CAGRA backend is required")
        self.cagra = cagra
        self.exact = BruteForceIndexBackend(device=cagra.device)
        self._lock = threading.RLock()
        self._built = {}

    @property
    def device(self):
        return self.cagra.device

    def _for_rows(self, rows):
        if type(rows) is not int or rows <= 0:
            raise IndexSearchError("index rows must be a positive integer")
        return self.exact if rows <= self.cagra.intermediate_degree else self.cagra

    def build_footprint(self, rows, dim, *, metric):
        return self._for_rows(rows).build_footprint(rows, dim, metric=metric)

    def build_scratch_footprint(self, rows, dim, *, metric):
        return self._for_rows(rows).build_scratch_footprint(rows, dim, metric=metric)

    def search_footprint(self, rows, dim, num_queries, top_k):
        return self._for_rows(rows).search_footprint(
            rows, dim, num_queries, top_k
        )

    def synchronize(self):
        with self._lock:
            self.cagra.synchronize()
            self.exact.synchronize()

    def build(self, vectors, *, vector_space, metric):
        if not isinstance(vectors, torch.Tensor) or vectors.ndim != 2:
            raise IndexSearchError("index vectors must be a 2-D tensor")
        with self._lock:
            self.cagra._check()
            selected = self._for_rows(int(vectors.shape[0]))
            if vectors.device != self.device:
                raise IndexSearchError("index vectors must be on the declared device")
            index = selected.build(vectors, vector_space=vector_space, metric=metric)
            self._built[id(index)] = (index, selected)
            return index

    def _owner(self, index):
        entry = self._built.get(id(index))
        if entry is None or entry[0] is not index:
            raise IndexSearchError("unknown or disposed CAGRA-auto index")
        return entry[1]

    def search(self, index, queries, *, top_k):
        with self._lock:
            self.cagra._check()
            return self._owner(index).search(index, queries, top_k=top_k)

    def dispose(self, index):
        with self._lock:
            self.cagra._check()
            selected = self._owner(index)
            selected.dispose(index)
            self._built.pop(id(index))
