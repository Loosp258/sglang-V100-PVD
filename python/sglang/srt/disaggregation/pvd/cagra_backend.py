"""cuVS CAGRA with a dedicated bounded RMM resource per layer/head index.

The native cap is reserved for the entire index lifetime, including workspace.
It is a conservative capacity reservation, not an estimate of final graph size.
No GPU compatibility or recall claim follows from importing this module.
"""

import ctypes
import hashlib
import importlib
import threading
import traceback
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

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
    graph: object = None
    rows: int = 0
    disposed: bool = False
    probe_allocations: list = field(default_factory=list)
    pending_extends: list = field(default_factory=list)
    stream: object = None
    auxiliary: object = None


class CagraNativeRuntime:
    """Actual cuVS calls; import only when a CUDA backend is explicitly chosen."""

    def __init__(
        self,
        device,
        *,
        global_native_cap_bytes=None,
        extend_concurrency=1,
        nogil_extend=False,
    ):
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("CAGRA requires an explicit CUDA device index")
        if global_native_cap_bytes is not None and (
            type(global_native_cap_bytes) is not int or global_native_cap_bytes <= 0
        ):
            raise ValueError("global CAGRA native cap must be a positive integer")
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
        self.stream_set = self.library.cuvsStreamSet
        self.stream_set.argtypes = [ctypes.c_size_t, ctypes.c_void_p]
        self.stream_set.restype = ctypes.c_int
        self.nogil_extend = nogil_extend
        if nogil_extend:
            if not self.supports_extend:
                raise ValueError("GIL-free CAGRA extend requires validated cuVS 25.10")
            adapter = importlib.import_module("pvd_cagra_index_handle")
            pxd = Path(extension.__file__).with_name("cagra.pxd")
            if (
                adapter.CUVS_VERSION != str(self.cuvs.__version__)
                or adapter.INDEX_PXD_SHA256
                != hashlib.sha256(pxd.read_bytes()).hexdigest()
            ):
                raise ValueError(
                    "CAGRA handle adapter does not match the installed cuVS ABI"
                )
            self.index_address = adapter.index_address
            self.params_create = self.library.cuvsCagraExtendParamsCreate
            self.params_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            self.params_create.restype = ctypes.c_int
            self.params_destroy = self.library.cuvsCagraExtendParamsDestroy
            self.params_destroy.argtypes = [ctypes.c_void_p]
            self.params_destroy.restype = ctypes.c_int
            self.native_extend = self.library.cuvsCagraExtend
            self.native_extend.argtypes = [
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self.native_extend.restype = ctypes.c_int
            self.capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
            self.capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            self.capsule_pointer.restype = ctypes.c_void_p
            self.last_error = self.library.cuvsGetLastErrorText
            self.last_error.argtypes = []
            self.last_error.restype = ctypes.c_char_p
        with _LOCKS_LOCK:
            self.lock = _LOCKS.setdefault(self.device.index, threading.RLock())
        # Keep each graph on one persistent Resources/stream pair.
        with torch.cuda.device(self.device):
            self.extend_streams = (
                [
                    torch.cuda.Stream(device=self.device)
                    for _ in range(extend_concurrency)
                ]
                if extend_concurrency > 1
                else []
            )
        self._next_extend_stream = 0
        # An optional parent limiter covers the sum of all child indexes,
        # including transient build/search allocations. Child limits still
        # enforce the per-index cap. The serving budget must reserve this
        # parent cap once before enabling it; runtime nesting alone is only a
        # native allocator capability, not a complete admission policy.
        self.global_native_cap_bytes = global_native_cap_bytes
        self.global_limit = None
        if global_native_cap_bytes is not None:
            with self.lock, torch.cuda.device(self.device):
                self.global_limit = self.mr.LimitingResourceAdaptor(
                    self.mr.CudaMemoryResource(), global_native_cap_bytes
                )

    def synchronize(self):
        torch.cuda.synchronize(self.device)

    @property
    def supports_extend(self):
        # This call shape was validated with cuVS 25.10 on V100S. Newer cuVS
        # releases may require a caller-owned padded full dataset instead of
        # the 25.10 additional-dataset argument; do not claim compatibility
        # merely because they export a function with the same name.
        return (
            str(getattr(self.cuvs, "__version__", "")).startswith("25.10.")
            and callable(getattr(self.cagra, "extend", None))
            and callable(getattr(self.cagra, "ExtendParams", None))
        )

    @contextmanager
    def scope(self, owner):
        # This lock covers every PVD CAGRA operation on this device. Other RMM
        # users must not replace its resource concurrently in the V process.
        with self.lock, torch.cuda.device(self.device):
            previous = self.mr.get_current_device_resource()
            self.mr.set_current_device_resource(owner.limit)
            try:
                with ExitStack() as contexts:
                    if owner.stream is not None:
                        owner.stream.wait_stream(torch.cuda.current_stream(self.device))
                        contexts.enter_context(torch.cuda.stream(owner.stream))
                        contexts.enter_context(
                            self.cp.cuda.ExternalStream(owner.stream.cuda_stream)
                        )
                    yield
            finally:
                self.mr.set_current_device_resource(previous)

    def create(self, byte_cap):
        with self.lock, torch.cuda.device(self.device):
            if (
                self.global_limit is not None
                and byte_cap > self.global_native_cap_bytes
            ):
                raise ValueError("per-index CAGRA cap exceeds shared native cap")
            limit = self.mr.LimitingResourceAdaptor(
                (
                    self.global_limit
                    if self.global_limit is not None
                    else self.mr.CudaMemoryResource()
                ),
                byte_cap,
            )
            streams = getattr(self, "extend_streams", [])
            stream = None
            if streams:
                stream = streams[self._next_extend_stream % len(streams)]
                self._next_extend_stream += 1
        return _NativeIndex(limit, stream=stream)

    def global_allocated_bytes(self):
        if self.global_limit is None:
            return None
        with self.lock, torch.cuda.device(self.device):
            return int(self.global_limit.get_allocated_bytes())

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

    def build(
        self,
        owner,
        vectors,
        *,
        metric,
        graph_degree,
        intermediate_degree,
        exact_head_groups=0,
    ):
        with self.scope(owner):
            # The installed 25.10 Cython Resources(stream=...) casts a Python
            # object to cudaStream_t incorrectly for nonzero handles. Use the
            # C API with an explicit pointer signature instead.
            owner.resources = self.Resources()
            if (
                self.stream_set(
                    owner.resources.get_c_obj(),
                    torch.cuda.current_stream(self.device).cuda_stream,
                )
                != 1
            ):
                raise IndexCompletionUnknown("cuVS could not bind the CUDA stream")
            self._verify_allocator_bridge(owner)
            owner.vectors = vectors  # Already an owned, budgeted extraction copy.
            if exact_head_groups:
                if metric != "ip" or len(vectors) % exact_head_groups:
                    raise IndexSearchError(
                        "exact grouped CAGRA needs inner product and equal heads"
                    )
                per_head = len(vectors) // exact_head_groups
                if per_head <= graph_degree:
                    raise IndexSearchError(
                        "exact head has fewer rows than graph degree"
                    )
                from rmm.allocators.cupy import rmm_cupy_allocator

                with self.cp.cuda.using_allocator(rmm_cupy_allocator):
                    dataset = self.cp.from_dlpack(vectors)
                    graph = self.cp.empty(
                        (len(vectors), graph_degree), dtype=self.cp.uint32
                    )
                    for head in range(exact_head_groups):
                        begin, end = head * per_head, (head + 1) * per_head
                        matrix = dataset[begin:end]
                        scores = matrix @ matrix.T
                        self.cp.fill_diagonal(scores, -self.cp.inf)
                        neighbors = self.cp.argpartition(scores, -graph_degree, axis=1)[
                            :, -graph_degree:
                        ]
                        values = self.cp.take_along_axis(scores, neighbors, axis=1)
                        neighbors = self.cp.take_along_axis(
                            neighbors, self.cp.argsort(-values, axis=1), axis=1
                        )
                        graph[begin:end] = neighbors.astype(self.cp.uint32) + begin
                        del scores, neighbors, values
                    owner.native = self.cagra.from_graph(
                        graph,
                        dataset,
                        metric="inner_product",
                        resources=owner.resources,
                    )
                    owner.graph = graph
            else:
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

    def extend(self, owner, additional_vectors):
        if not self.supports_extend:
            raise IndexSearchError("this cuVS Python CAGRA has no extend binding")
        with self.scope(owner):
            # Retain the input before submission: even a raised native call may
            # have queued work whose completion cannot yet be proved.
            old = owner.vectors
            owner.vectors = (
                (*old, additional_vectors)
                if isinstance(old, tuple)
                else (old, additional_vectors)
            )
            self._call_extend(owner, additional_vectors, owner.resources)
            self.synchronize()
            if (
                owner.native.trained is not True
                or owner.native.dim != additional_vectors.shape[1]
            ):
                raise IndexCompletionUnknown("CAGRA extend returned an invalid index")

    def _call_extend(self, owner, vectors, resources):
        if not getattr(self, "nogil_extend", False):
            self.cagra.extend(
                self.cagra.ExtendParams(),
                owner.native,
                self.cp.from_dlpack(vectors),
                resources=resources,
            )
            return
        # CDLL releases the GIL around this exact native call. Device allocator
        # and index locks remain held, allowing the other V GPU to progress.
        # The ABI-checked Cython adapter exposes the opaque handle; no guessed
        # object offsets or algorithm changes are involved.
        capsule = torch.utils.dlpack.to_dlpack(vectors)
        pointer = self.capsule_pointer(capsule, b"dltensor")
        params = ctypes.c_void_p()
        if self.params_create(ctypes.byref(params)) != 1 or not params.value:
            raise IndexCompletionUnknown("could not create native CAGRA extend params")
        try:
            status = self.native_extend(
                resources.get_c_obj(), params, pointer, self.index_address(owner.native)
            )
            if status != 1:
                error = self.last_error()
                raise IndexCompletionUnknown(f"native CAGRA extend failed: {error!r}")
        finally:
            if self.params_destroy(params) != 1:
                raise IndexCompletionUnknown(
                    "could not destroy native CAGRA extend params"
                )

    def search(self, owner, queries, *, top_k, itopk_size, bitset=None):
        with self.scope(owner):
            rows = torch.empty(
                (len(queries), top_k), device=self.device, dtype=torch.uint32
            )
            scores = torch.empty(
                (len(queries), top_k), device=self.device, dtype=torch.float32
            )
            # Pass buffers explicitly so the wrapper cannot allocate uncharged
            # device_ndarray outputs through a different allocator.
            filter_arg = None
            if bitset is not None:
                filters = importlib.import_module("cuvs.neighbors.filters")
                filter_arg = filters.from_bitset(self.cp.from_dlpack(bitset))
            self.cagra.search(
                self.cagra.SearchParams(itopk_size=itopk_size),
                owner.native,
                self.cp.from_dlpack(queries),
                top_k,
                neighbors=self.cp.from_dlpack(rows),
                distances=self.cp.from_dlpack(scores),
                filter=filter_arg,
                resources=owner.resources,
            )
            self.synchronize()
            return rows, scores

    def extend_many(self, items, *, concurrency):
        """Submit independent indexes on bounded streams, then drain each wave.

        Host submissions remain serialized under the device RMM lock. CUDA
        streams may overlap work if the native call returns asynchronously;
        cuVS 25.10's extend currently blocks in the measured shapes.
        Changing the current allocator from competing threads
        would invalidate the per-index allocation contract. Keep temporary
        resources, streams and inputs on their owners until completion proof.
        """
        if not self.supports_extend:
            raise IndexSearchError("this cuVS Python CAGRA has no extend binding")
        with self.lock, torch.cuda.device(self.device):
            for offset in range(0, len(items), concurrency):
                wave = items[offset : offset + concurrency]
                failure = None
                submitted = []
                try:
                    for owner, additional in wave:
                        if owner.stream is None:
                            raise IndexSearchError(
                                "concurrent extend needs a stream assigned before build"
                            )
                        with self.scope(owner):
                            resources, stream = owner.resources, owner.stream
                            owner.pending_extends.append(
                                (resources, stream, additional)
                            )
                            submitted.append(owner)
                            old = owner.vectors
                            owner.vectors = (
                                (*old, additional)
                                if isinstance(old, tuple)
                                else (old, additional)
                            )
                            self._call_extend(owner, additional, resources)
                except BaseException as exc:
                    failure = exc
                # Drain every submitted stream, including a call that raised.
                # An unknown completion must retain all of that call's owners.
                for owner in submitted:
                    try:
                        for resources, stream, _ in owner.pending_extends:
                            resources.sync()
                            stream.synchronize()
                        owner.pending_extends.clear()
                    except BaseException as exc:
                        failure = failure or exc
                if failure is not None:
                    raise IndexCompletionUnknown(
                        f"CAGRA batch extend completion/state is unknown: {failure}"
                    ) from failure
                for owner, additional in wave:
                    if (
                        owner.native.trained is not True
                        or owner.native.dim != additional.shape[1]
                    ):
                        raise IndexCompletionUnknown(
                            "CAGRA batch extend returned an invalid index"
                        )

    def dispose(self, owner):
        if owner.disposed:
            return
        with self.scope(owner):
            self.synchronize()
            for resources, stream, _ in owner.pending_extends:
                resources.sync()
                stream.synchronize()
            owner.pending_extends.clear()
            if owner.probe_allocations:
                # A C API failure may have returned a pointer without a reliable
                # ownership outcome. Do not guess whether it is safe to free it
                # again, or destroy the resources it might still use.
                raise IndexCompletionUnknown("cuVS allocator probe remains unresolved")
            # cuVS's Cython __dealloc__ destroys the native index. Keep its MR
            # and Resources alive until destruction and all streams complete.
            owner.native = None
            owner.graph = None
            owner.resources = None
            self.synchronize()
            if owner.limit.get_allocated_bytes() != 0:
                raise IndexCompletionUnknown(
                    "CAGRA disposal left native allocations alive"
                )
            owner.vectors = None
            owner.auxiliary = None
            owner.disposed = True

    def import_owned_graph(self, owner, vectors, graph):
        """Import caller-owned, budgeted buffers without building their edges."""
        with self.scope(owner):
            owner.resources = self.Resources()
            if (
                self.stream_set(
                    owner.resources.get_c_obj(),
                    torch.cuda.current_stream(self.device).cuda_stream,
                )
                != 1
            ):
                raise IndexCompletionUnknown("cuVS could not bind the CUDA stream")
            self._verify_allocator_bridge(owner)
            owner.vectors, owner.graph = vectors, graph
            owner.native = self.cagra.from_graph(
                self.cp.from_dlpack(graph).view(self.cp.uint32),
                self.cp.from_dlpack(vectors),
                metric="inner_product",
                resources=owner.resources,
            )

    def replace_owned_graph_views(self, owner, vectors, graph):
        """Resize the public C++ index views, retaining the same native index."""
        with self.scope(owner):
            self._replace_graph_views(
                owner.native,
                owner.resources.get_c_obj(),
                vectors.data_ptr(),
                graph.data_ptr(),
                len(vectors),
                vectors.shape[1],
                graph.shape[1],
            )

    def require_graph_view_adapter(self):
        """Check the compiled adapter before accepting an experimental upload."""
        if not self.supports_extend:
            raise ValueError("KV graph views require validated cuVS 25.10")
        adapter = importlib.import_module("pvd_cagra_index_handle")
        extension = importlib.import_module(self.cagra.Index.__module__)
        pxd = Path(extension.__file__).with_name("cagra.pxd")
        header = (
            Path(extension.__file__).parents[3]
            / "libcuvs/include/cuvs/neighbors/cagra.hpp"
        )
        if (
            getattr(adapter, "CUVS_VERSION", None) != str(self.cuvs.__version__)
            or getattr(adapter, "INDEX_PXD_SHA256", None)
            != hashlib.sha256(pxd.read_bytes()).hexdigest()
            or getattr(adapter, "CAGRA_HPP_SHA256", None)
            != hashlib.sha256(header.read_bytes()).hexdigest()
            or not callable(getattr(adapter, "replace_views", None))
        ):
            raise ValueError("CAGRA view adapter does not match installed cuVS headers")
        self._replace_graph_views = adapter.replace_views


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
        global_native_cap_bytes=None,
        exact_head_groups=0,
        extend_concurrency=1,
        nogil_extend=False,
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
        if global_native_cap_bytes is not None and (
            type(global_native_cap_bytes) is not int
            or global_native_cap_bytes < native_bytes_per_index
        ):
            raise ValueError("shared native cap must cover one full per-index cap")
        if self._device.type != "cuda" and _runtime is None:
            raise ValueError("native CAGRA requires CUDA")
        self.cap = native_bytes_per_index
        self.graph_degree, self.intermediate_degree = graph_degree, intermediate_degree
        if exact_head_groups not in (0, 4):
            raise ValueError("exact CAGRA seed requires four grouped heads")
        if exact_head_groups and graph_degree != 16:
            raise ValueError("calibrated exact CAGRA seed requires degree 16")
        self.exact_head_groups = exact_head_groups
        if type(extend_concurrency) is not int or extend_concurrency not in (1, 2, 4):
            raise ValueError("CAGRA extend concurrency must be 1, 2 or 4")
        self.extend_concurrency = extend_concurrency
        self.itopk_size = itopk_size
        self.runtime = (
            _runtime
            if _runtime is not None
            else CagraNativeRuntime(
                self._device,
                **(
                    {"global_native_cap_bytes": global_native_cap_bytes}
                    if global_native_cap_bytes is not None
                    else {}
                ),
                **(
                    {"extend_concurrency": extend_concurrency}
                    if extend_concurrency > 1
                    else {}
                ),
                **({"nogil_extend": True} if nogil_extend else {}),
            )
        )
        if torch.device(self.runtime.device) != self._device:
            raise ValueError("CAGRA runtime device mismatch")
        runtime_cap = getattr(self.runtime, "global_native_cap_bytes", None)
        if (
            global_native_cap_bytes is not None
            and runtime_cap != global_native_cap_bytes
        ):
            raise ValueError("CAGRA runtime shared cap differs from configured cap")
        if runtime_cap is not None and (
            type(runtime_cap) is not int or runtime_cap < native_bytes_per_index
        ):
            raise ValueError("CAGRA runtime shared cap cannot cover an index")
        if (
            runtime_cap is not None
            and getattr(self.runtime, "global_limit", None) is None
        ):
            raise ValueError("CAGRA runtime shared cap lacks a native root limiter")
        self._shared_native_cap = runtime_cap
        self._lock = threading.RLock()
        self._owners = {}
        self._unknown = None

    @property
    def device(self):
        return self._device

    @property
    def shared_footprint(self):
        return self._shared_native_cap or 0

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
        # In shared mode the parent's hard native cap was reserved once at
        # manager construction. Charging it again per graph would restore the
        # 56*cap V100S failure while providing no additional safety.
        return 0 if self._shared_native_cap is not None else self.cap

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

    def _matrix(self, tensor, *, check_finite=True):
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
        if check_finite and not bool(torch.isfinite(tensor).all()):
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
                kwargs = dict(
                    metric=metric,
                    graph_degree=self.graph_degree,
                    intermediate_degree=self.intermediate_degree,
                )
                if self.exact_head_groups:
                    kwargs["exact_head_groups"] = self.exact_head_groups
                self.runtime.build(owner, vectors, **kwargs)
                self.runtime.synchronize()
                owner.rows = rows
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

    @property
    def supports_extend(self):
        return bool(getattr(self.runtime, "supports_extend", False)) and callable(
            getattr(self.runtime, "extend", None)
        )

    def extend(self, index, additional_vectors):
        """Extend one private graph; return the new count/versioned handle view.

        A native failure may leave the graph partially changed. Quarantine the
        backend and retain its owner until completion can be proved; callers
        must never publish the provisional graph after such a failure.
        """
        with self._lock:
            self._check()
            if not self.supports_extend:
                raise IndexSearchError("CAGRA extend requires a supported cuVS binding")
            owner = index.handle
            if self._owners.get(id(owner)) is not owner or owner.disposed:
                raise IndexSearchError("unknown or disposed CAGRA index")
            if index.count != owner.rows:
                raise IndexSearchError("stale CAGRA index count after extend")
            self._matrix(additional_vectors)
            if additional_vectors.shape[1] != index.dim:
                raise IndexSearchError("CAGRA extension dimension mismatch")
            new_count = index.count + int(additional_vectors.shape[0])
            self._shape(new_count, index.dim, index.metric)
            try:
                self.runtime.extend(owner, additional_vectors)
                self.runtime.synchronize()
            except BaseException as exc:
                # Even a raised native call may have modified its index. A
                # successful synchronize does not restore the old graph.
                self._unknown = str(exc)
                raise IndexCompletionUnknown(
                    f"CAGRA extend completion/state is unknown: {exc}"
                ) from exc
            owner.rows = new_count
            return BuiltIndex(
                index.vector_space, index.metric, index.dim, new_count, owner
            )

    def extend_many(self, items):
        """Extend distinct private graphs with one bounded GPU batch."""
        items = tuple(items)
        with self._lock:
            self._check()
            if not self.supports_extend:
                raise IndexSearchError("CAGRA extend requires a supported cuVS binding")
            if self.extend_concurrency == 1:
                return [self.extend(index, vectors) for index, vectors in items]
            native_items, counts, seen = [], [], set()
            # Validate the entire batch before any native index can change.
            for index, vectors in items:
                owner = index.handle
                if id(owner) in seen:
                    raise IndexSearchError("batch extend requires distinct indexes")
                seen.add(id(owner))
                if self._owners.get(id(owner)) is not owner or owner.disposed:
                    raise IndexSearchError("unknown or disposed CAGRA index")
                if index.count != owner.rows:
                    raise IndexSearchError("stale CAGRA index count after extend")
                self._matrix(vectors, check_finite=False)
                if vectors.shape[1] != index.dim:
                    raise IndexSearchError("CAGRA extension dimension mismatch")
                count = index.count + int(vectors.shape[0])
                self._shape(count, index.dim, index.metric)
                native_items.append((owner, vectors))
                counts.append(count)
            # One host-visible reduction for the whole batch, rather than a
            # CPU/GPU rendezvous for every graph. Masks are reduced and freed
            # one at a time on the preparation stream; only scalar flags stay.
            if items and not bool(
                torch.stack(
                    [torch.isfinite(vectors).all() for _, vectors in items]
                ).all()
            ):
                raise IndexSearchError("CAGRA vectors/queries must be finite")
            try:
                self.runtime.extend_many(
                    native_items, concurrency=self.extend_concurrency
                )
            except BaseException as exc:
                self._unknown = str(exc)
                raise IndexCompletionUnknown(
                    f"CAGRA batch extend completion/state is unknown: {exc}"
                ) from exc
            result = []
            for (index, _), count in zip(items, counts):
                index.handle.rows = count
                result.append(
                    BuiltIndex(
                        index.vector_space, index.metric, index.dim, count, index.handle
                    )
                )
            return result

    def search(self, index, queries, *, top_k, bitset=None):
        with self._lock:
            self._check()
            if (
                self._owners.get(id(index.handle)) is not index.handle
                or index.handle.disposed
            ):
                raise IndexSearchError("unknown or disposed CAGRA index")
            if index.count != index.handle.rows:
                raise IndexSearchError("stale CAGRA index count after extend")
            self._matrix(queries)
            if bitset is not None and (
                not isinstance(bitset, torch.Tensor)
                or bitset.dtype != torch.uint32
                or bitset.device != self.device
                or bitset.ndim != 1
                or not bitset.is_contiguous()
                or bitset.numel() != (index.count + 31) // 32
            ):
                raise IndexSearchError(
                    "CAGRA filter bitset has an invalid shape/device"
                )
            if queries.shape[1] != index.dim:
                raise IndexSearchError("CAGRA query dimension mismatch")
            self.search_footprint(index.count, index.dim, len(queries), top_k)
            try:
                rows, scores = self.runtime.search(
                    index.handle,
                    queries,
                    top_k=top_k,
                    itopk_size=self.itopk_size,
                    **({"bitset": bitset} if bitset is not None else {}),
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

    def __init__(self, cagra: CagraIndexBackend, *, exact_max_rows: int | None = None):
        if not isinstance(cagra, CagraIndexBackend):
            raise TypeError("a configured CAGRA backend is required")
        if exact_max_rows is None:
            exact_max_rows = cagra.intermediate_degree
        if (
            type(exact_max_rows) is not int
            or exact_max_rows < cagra.intermediate_degree
        ):
            raise ValueError(
                "exact_max_rows must be at least the CAGRA intermediate degree"
            )
        self.cagra = cagra
        self.exact_max_rows = exact_max_rows
        self.exact = BruteForceIndexBackend(device=cagra.device)
        self._lock = threading.RLock()
        self._built = {}

    @property
    def device(self):
        return self.cagra.device

    @property
    def shared_footprint(self):
        return self.cagra.shared_footprint

    def _for_rows(self, rows):
        if type(rows) is not int or rows <= 0:
            raise IndexSearchError("index rows must be a positive integer")
        return self.exact if rows <= self.exact_max_rows else self.cagra

    def build_path(self, rows):
        """Expose the selected path for index-build diagnostics."""
        return "exact" if self._for_rows(rows) is self.exact else "cagra"

    def build_footprint(self, rows, dim, *, metric):
        return self._for_rows(rows).build_footprint(rows, dim, metric=metric)

    def build_scratch_footprint(self, rows, dim, *, metric):
        return self._for_rows(rows).build_scratch_footprint(rows, dim, metric=metric)

    def search_footprint(self, rows, dim, num_queries, top_k):
        return self._for_rows(rows).search_footprint(rows, dim, num_queries, top_k)

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

    @property
    def supports_extend(self):
        return self.cagra.supports_extend

    def extend(self, index, additional_vectors):
        """Replace the provisional CAGRA count view after a completed extend."""
        with self._lock:
            self.cagra._check()
            if self._owner(index) is not self.cagra:
                raise IndexSearchError("the exact short-Prompt path cannot extend")
            replacement = self.cagra.extend(index, additional_vectors)
            self._built.pop(id(index))
            self._built[id(replacement)] = (replacement, self.cagra)
            return replacement

    def _owner(self, index):
        entry = self._built.get(id(index))
        if entry is None or entry[0] is not index:
            raise IndexSearchError("unknown or disposed CAGRA-auto index")
        return entry[1]

    def search(self, index, queries, *, top_k):
        with self._lock:
            self.cagra._check()
            return self._owner(index).search(index, queries, top_k=top_k)

    def supports_grouped_exact(self, indexes, *, num_queries, top_k):
        """Only uniform, small exact indexes may share a batched GEMM."""
        with self._lock:
            self.cagra._check()
            if not indexes or any(
                self._owner(index) is not self.exact for index in indexes
            ):
                return False
            try:
                self.exact._group_shape(indexes, num_queries, top_k)
            except IndexSearchError:
                return False
            return True

    def grouped_exact_footprint(self, indexes, *, num_queries, top_k):
        with self._lock:
            if not self.supports_grouped_exact(
                indexes, num_queries=num_queries, top_k=top_k
            ):
                raise IndexSearchError("indexes are not a uniform exact group")
            return self.exact.grouped_search_footprint(
                indexes=indexes, num_queries=num_queries, top_k=top_k
            )

    def search_grouped_exact(self, indexes, queries, *, top_k):
        with self._lock:
            self.cagra._check()
            if not queries or not self.supports_grouped_exact(
                indexes, num_queries=int(queries[0].shape[0]), top_k=top_k
            ):
                raise IndexSearchError("indexes are not a uniform exact group")
            return self.exact.search_grouped(indexes, queries, top_k=top_k)

    def dispose(self, index):
        with self._lock:
            self.cagra._check()
            selected = self._owner(index)
            selected.dispose(index)
            self._built.pop(id(index))
