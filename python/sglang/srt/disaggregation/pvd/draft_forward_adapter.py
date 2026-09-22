"""Turn ``DraftForwardInputs`` into a real ``ForwardBatch`` and run it.

This is the seam the reuse audit left open. Everything around it -- private
slots, positions, sequence lengths, KV rows, the private request map, bounded
stepping, cleanup -- is in ``draft_runner_sglang``. This module does the one
remaining step: build the object SGLang's ``ModelRunner`` expects, drive the
real pools, and unwrap what comes back.

Why the ForwardBatch is constructed field by field
--------------------------------------------------
``ForwardBatch.init_new(batch, model_runner)`` takes a ``ScheduleBatch``: the
live scheduler object, carrying the tree cache, the sampling info and the
committed request list. Reaching for it here would reintroduce precisely the
coupling the audit exists to avoid, and ``init_new`` also *mutates* what it is
given (it consumes one-shot per-forward overrides off the batch). So the
``ForwardBatch`` is built directly from values this branch owns.

``ForwardBatch`` carries no pools and no attention backend -- the
``ModelRunner`` supplies those from its own state -- which is what makes
direct construction viable, and also what makes the private-pool decision
load-bearing: the runner this executor is given must be the draft runner,
whose pools are its own.

Interfaces this module matches, verbatim from the checkout
-----------------------------------------------------------
Each of these was got wrong first by assuming a plausible API instead of
reading one, so each is named here with what it actually is:

* ``ReqToTokenPool.alloc(reqs: list[Req]) -> Optional[List[int]]`` takes
  request *objects*, assigns ``r.req_pool_idx`` in place and returns the
  indices; ``free(req: Req)`` asserts ``req_pool_idx is not None``, returns
  the slot and sets the attribute back to ``None``. It is not
  ``alloc(count)``/``free(index)``. Slot 0 is a padding row and is never
  handed out (``free_slots = list(range(1, size + 1))``).
* ``ModelRunner.forward`` returns ``ModelRunnerOutput``, whose
  ``logits_output`` is a ``LogitsProcessorOutput`` with
  ``next_token_logits`` of shape ``[#seq, vocab]`` -- and that field is
  ``Optional``, so its absence is a real case, not a defensive branch. It can
  also be a ``PPProxyTensors`` under pipeline parallelism, which this adapter
  does not support and refuses.
* ``CaptureHiddenMode`` is an ``IntEnum`` whose ``NULL = 0`` means "capture
  nothing". The logits processor calls ``.need_capture()`` on it, so ``None``
  is not "disabled" -- it is an ``AttributeError`` waiting to happen.
  ``init_new`` derives ``NULL`` when nothing asks for capture; this adapter
  sets it explicitly.
* ``TokenToKVPoolAllocator.free_pages`` is ``int64`` on the allocator's own
  device, and ``free()`` does ``torch.cat((free_pages, free_index))``, so a
  release tensor built on the CPU with a guessed dtype fails on a CUDA
  allocator.
* ``PagedTokenToKVPoolAllocator.alloc(need_size)`` is **page-aligned**: it
  asserts ``need_size % page_size == 0`` and returns whole pages. Allocating
  one token at a time is only valid at ``page_size == 1``; anything larger
  needs ``alloc_extend``/``alloc_decode``, which this adapter does not
  implement and therefore refuses up front.

What is and is not established
------------------------------
``forward_fields`` produces the mapping as plain data, and construction is
split from it so minimal CPU environments need not import the full serving
dependency chain. Two lightweight checks remain useful:

* the **values** -- names, numbers, dtypes, devices, shapes -- against
  doubles built from the contracts above;
* the **field names** -- that every key corresponds to a real ``ForwardBatch``
  field, and that every field without a default is supplied -- by reading
  ``forward_batch_info.py`` as source, so a rename upstream fails here.

The opt-in ``run_pvd_draft_cpu_smoke.py`` additionally executes real
``ModelRunner`` forwards with a random tiny Llama, real pools, and
``TorchNativeAttnBackend`` on CPU. It compares incremental decode to full
prefix recomputation and checks a deliberately corrupted mapping is detected.
This is not GPU, RDMA, production-checkpoint, peak-memory or latency evidence.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from sglang.srt.disaggregation.pvd.draft_runner_sglang import (
    DraftForwardInputs,
    ModelExecutor,
)
from sglang.srt.disaggregation.pvd.draft_sglang import (
    DraftCapabilityError,
    DraftLifecycleError,
)

#: ``ForwardBatch`` fields that would pull this path into speculative
#: decoding, logprob accounting or hidden-state capture. Asserted, rather
#: than merely left unset, so an edit that starts populating one fails here
#: instead of quietly changing what a prediction is.
MUST_STAY_UNSET = (
    "spec_info",
    "spec_algorithm",
    "input_embeds",
    "token_type_ids",
    "pvd_query_capture",
)

#: The only page size the token-wise allocation below is valid for. A paged
#: allocator asserts page alignment and returns whole pages, so one-token
#: allocation is not a smaller version of the same thing.
SUPPORTED_PAGE_SIZE = 1


class DraftRequestHandle:
    """A branch-owned stand-in for a ``Req``, for the request pool only.

    ``ReqToTokenPool`` reads and writes ``req_pool_idx`` on the objects it is
    given, and reads ``inflight_middle_chunks`` / ``kv_committed_len`` when a
    request arrives already holding a slot. A committed request is never
    borrowed for this: passing one in would let the pool clear the slot of a
    request that is still decoding.

    Nothing here is a ``Req``. It carries the three attributes the pool
    touches and nothing else, so it cannot be mistaken for one or handed to
    code that expects the rest of that class.
    """

    __slots__ = ("req_pool_idx", "inflight_middle_chunks", "kv_committed_len")

    def __init__(self) -> None:
        # None means "no slot yet", which is what makes the pool allocate one
        # rather than treat this as a request reusing its own.
        self.req_pool_idx: Optional[int] = None
        self.inflight_middle_chunks = 0
        self.kv_committed_len = 0


class DraftForwardAdapter(ModelExecutor):
    """Runs one prediction forward on a draft ``ModelRunner``.

    The runner passed in must be the **draft** model runner -- the one built
    with private pools. Nothing here can check that on its own (a runner does
    not know whose it is), so the provider's pool-ownership check is what
    establishes it, and this class documents the dependency rather than
    implying a guarantee it cannot make.

    It retains no batches and no tensors between calls; see ``last_forward``.
    """

    def __init__(
        self,
        model_runner: Any,
        *,
        architecture: str,
        attention_backend: str,
        bytes_per_token: int,
        device: Optional[Any] = None,
        forward_batch_factory: Optional[Any] = None,
        transient_bytes_bound: Optional[int] = None,
    ) -> None:
        if not architecture or not attention_backend:
            raise DraftCapabilityError(
                "a draft executor must state the architecture and attention "
                "backend it was built for; they are checked against the "
                "supported subset, never inferred"
            )
        if (
            isinstance(bytes_per_token, bool)
            or not isinstance(bytes_per_token, int)
            or bytes_per_token <= 0
        ):
            raise DraftCapabilityError("bytes_per_token must be a positive integer")
        self._runner = model_runner
        self._architecture = str(architecture)
        self._attention_backend = str(attention_backend)
        self._bytes_per_token = bytes_per_token
        self._device = torch.device(
            device if device is not None else getattr(model_runner, "device", "cpu")
        )
        self._factory = forward_batch_factory
        # No default derived from KV bytes: model intermediates and backend
        # workspace are not KV. A deployment must supply a bound for its
        # admitted shapes before provider-level admission is allowed.
        if transient_bytes_bound is not None and (
            isinstance(transient_bytes_bound, bool)
            or not isinstance(transient_bytes_bound, int)
            or transient_bytes_bound < 0
        ):
            raise DraftCapabilityError("transient_bytes_bound must be non-negative")
        self._transient_bytes_bound = transient_bytes_bound
        #: Bounded diagnostics: shapes and counts from the most recent
        #: forward, never the tensors. Retaining the batches would pin their
        #: device memory for the life of the adapter, which on a GPU is the
        #: difference between a bounded branch and a leak.
        self.last_forward: Optional[dict] = None
        self.forward_count = 0

    # -- what the capability check reads ------------------------------------

    def architecture(self) -> str:
        return self._architecture

    def attention_backend(self) -> str:
        return self._attention_backend

    def bytes_per_token(self) -> int:
        return self._bytes_per_token

    def transient_bytes(self, prefix_tokens: int, predict_tokens: int) -> int:
        if self._transient_bytes_bound is None:
            raise DraftCapabilityError(
                "non-KV transient bound is unknown; KV bytes alone cannot "
                "bound logits, activations and backend workspace"
            )
        return self._transient_bytes_bound

    # -- construction -------------------------------------------------------

    def _tensor(self, values, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(list(values), dtype=dtype, device=self._device)

    def forward_fields(self, inputs: DraftForwardInputs) -> dict:
        """The ``ForwardBatch`` mapping as plain data.

        Separated from construction so the mapping can be checked without
        importing SGLang's serving frontend. ``forward_mode`` and
        ``capture_hidden_mode`` are left as this module's own markers; the
        caller translates them, because both enums live behind that import.
        """
        fields = {
            "forward_mode": inputs.forward_mode,
            "batch_size": len(inputs.seq_lens),
            "input_ids": self._tensor(inputs.input_ids, torch.int64),
            "req_pool_indices": self._tensor(inputs.req_pool_indices, torch.int64),
            "seq_lens": self._tensor(inputs.seq_lens, torch.int64),
            "out_cache_loc": self._tensor(inputs.out_cache_loc, torch.int64),
            "seq_lens_sum": int(sum(inputs.seq_lens)),
            # Absolute sequence positions, supplied rather than derived: the
            # prefix is recomputed from zero every call, so nothing may infer
            # positions from a prior forward's state.
            "positions": self._tensor(inputs.positions, torch.int64),
            # init_new populates this mirror for every non-gpu_only path;
            # backends read it to avoid a device sync.
            "seq_lens_cpu": torch.tensor(list(inputs.seq_lens), dtype=torch.int64),
            "return_logprob": False,
            # NULL, not None: the logits processor calls need_capture() on it.
            "capture_hidden_mode": "null",
        }
        if inputs.forward_mode == "extend":
            extend_num_tokens = int(sum(inputs.extend_seq_lens))
            # extend_start_loc is the exclusive prefix sum of extend_seq_lens,
            # which is what compute_position() derives in init_new. One
            # request per prediction makes it (0,), but it is computed rather
            # than hardcoded so a wider batch is not silently wrong.
            starts, running = [], 0
            for length in inputs.extend_seq_lens:
                starts.append(running)
                running += int(length)
            fields.update(
                {
                    "extend_prefix_lens": self._tensor(
                        inputs.extend_prefix_lens, torch.int32
                    ),
                    "extend_seq_lens": self._tensor(
                        inputs.extend_seq_lens, torch.int32
                    ),
                    "extend_prefix_lens_cpu": list(inputs.extend_prefix_lens),
                    "extend_seq_lens_cpu": list(inputs.extend_seq_lens),
                    "extend_start_loc": self._tensor(starts, torch.int32),
                    # Required by the extend path; init_new copies it off the
                    # ScheduleBatch, which this adapter does not have.
                    "extend_num_tokens": extend_num_tokens,
                }
            )
            if extend_num_tokens != len(inputs.input_ids):
                raise DraftLifecycleError(
                    f"extend_num_tokens {extend_num_tokens} disagrees with "
                    f"{len(inputs.input_ids)} input ids"
                )
        return fields

    def build_forward_batch(self, inputs: DraftForwardInputs):
        """Construct the real ``ForwardBatch``. Imported lazily.

        The import needs the serving dependencies. ``forward_batch_factory``
        allows minimal-environment unit tests; the strict smoke uses real types.
        """
        fields = self.forward_fields(inputs)
        factory = self._factory
        if factory is None:  # pragma: no cover - needs the serving frontend
            from sglang.srt.model_executor.forward_batch_info import (
                CaptureHiddenMode,
                ForwardBatch,
                ForwardMode,
            )

            fields["forward_mode"] = (
                ForwardMode.EXTEND
                if inputs.forward_mode == "extend"
                else ForwardMode.DECODE
            )
            fields["capture_hidden_mode"] = CaptureHiddenMode.NULL
            factory = ForwardBatch
        batch = factory(**fields)
        self._assert_prediction_only(batch)
        return batch

    @staticmethod
    def _assert_prediction_only(batch: Any) -> None:
        """A prediction forward carries no speculative or capture payload."""
        for name in MUST_STAY_UNSET:
            if getattr(batch, name, None) is not None:
                raise DraftLifecycleError(
                    f"a prediction forward must not carry {name}; this batch "
                    "would enter a speculative or capture path"
                )
        mode = getattr(batch, "capture_hidden_mode", None)
        if mode is None:
            raise DraftLifecycleError(
                "capture_hidden_mode must be CaptureHiddenMode.NULL, not "
                "None: the logits processor calls need_capture() on it"
            )
        # Accept the disabled value in either spelling, refuse a real capture.
        capturing = (
            mode.need_capture()
            if hasattr(mode, "need_capture")
            else (str(mode).lower() not in ("null", "capturehiddenmode.null", "0"))
        )
        if capturing:
            raise DraftLifecycleError(
                f"a prediction forward must not capture hidden states, got {mode!r}"
            )

    # -- execution ----------------------------------------------------------

    def forward(self, inputs: DraftForwardInputs) -> torch.Tensor:
        """Run one forward and return logits for the final position.

        No sampler, no logprob accounting, no hidden-state capture. The batch
        is dropped as soon as the forward returns, so its device tensors are
        not kept alive by this adapter.
        """
        batch = self.build_forward_batch(inputs)
        self.forward_count += 1
        self.last_forward = {
            "forward_mode": inputs.forward_mode,
            "num_tokens": len(inputs.input_ids),
            "seq_lens": tuple(inputs.seq_lens),
            "batch_size": len(inputs.seq_lens),
        }
        with torch.inference_mode():
            output = self._runner.forward(batch)
        del batch
        return self._last_position_logits(output)

    @staticmethod
    def _last_position_logits(output: Any) -> torch.Tensor:
        """Unwrap ``ModelRunnerOutput.logits_output.next_token_logits``.

        Written against the real structure: a ``ModelRunnerOutput`` whose
        ``logits_output`` is a ``LogitsProcessorOutput``. A bare tensor is
        accepted only because a caller may legitimately wrap the runner, and
        it is the shape check below -- not the container -- that decides
        whether the result is usable.
        """
        logits_output = getattr(output, "logits_output", None)
        if logits_output is None:
            if isinstance(output, torch.Tensor):
                logits = output
            else:
                raise DraftLifecycleError(
                    "the draft model runner returned no logits_output; "
                    "ModelRunner.forward returns a ModelRunnerOutput"
                )
        else:
            if not hasattr(logits_output, "next_token_logits"):
                # PPProxyTensors under pipeline parallelism, or a future type.
                raise DraftLifecycleError(
                    f"logits_output is a {type(logits_output).__name__}, which "
                    "carries no next_token_logits; pipeline parallelism is "
                    "outside the supported subset"
                )
            logits = logits_output.next_token_logits
            if logits is None:
                # A real, documented case: prefill-only requests that need no
                # next token. It means this forward cannot drive a prediction.
                raise DraftLifecycleError(
                    "next_token_logits is None; this forward produced no "
                    "next-token distribution to predict from"
                )
        if not isinstance(logits, torch.Tensor):
            raise DraftLifecycleError(
                f"expected next-token logits, got {type(logits).__name__}"
            )
        if logits.ndim == 2:
            # [#seq, vocab]: one row per sequence. One request per prediction,
            # so the last row is this request's.
            if logits.shape[0] < 1:
                raise DraftLifecycleError("next_token_logits has no rows")
            logits = logits[-1]
        elif logits.ndim != 1:
            raise DraftLifecycleError(
                f"expected 1-D or 2-D logits, got shape {tuple(logits.shape)}"
            )
        return logits


class PrivatePoolAllocator:
    """``SlotAllocator`` over a draft worker's own pools.

    Every index it hands out comes from the draft model runner's own
    ``req_to_token_pool`` and KV allocator. It has no reference to the
    target's pools and so cannot return one of their rows.

    Release tensors are built on the allocator's own device and dtype,
    because ``free()`` concatenates them onto ``free_pages`` and a mismatched
    device or dtype raises there rather than here.
    """

    def __init__(self, req_to_token_pool: Any, kv_allocator: Any) -> None:
        if req_to_token_pool is None or kv_allocator is None:
            raise DraftCapabilityError(
                "a private request pool and KV allocator are required; this "
                "allocator never falls back to a shared one"
            )
        page_size = getattr(kv_allocator, "page_size", SUPPORTED_PAGE_SIZE)
        if page_size != SUPPORTED_PAGE_SIZE:
            # A paged allocator asserts need_size % page_size == 0 and returns
            # whole pages; one-token allocation is not a smaller version of
            # that. Supporting it means alloc_extend/alloc_decode, which this
            # adapter does not implement -- so it is refused before anything
            # is allocated rather than discovered mid-prediction.
            raise DraftCapabilityError(
                f"page_size {page_size} is outside the supported subset "
                f"(page_size == {SUPPORTED_PAGE_SIZE}); paged allocation "
                "needs alloc_extend/alloc_decode, which this adapter does "
                "not implement"
            )
        self._requests = req_to_token_pool
        self._kv = kv_allocator
        self.page_size = page_size
        #: One request object per branch, created here and never borrowed.
        self._request = DraftRequestHandle()

    def fork_for_branch(self) -> PrivatePoolAllocator:
        """Independent request owner, same private KV and request pool storage."""
        return type(self)(self._requests, self._kv)

    # -- request slots ------------------------------------------------------

    def alloc_request(self) -> int:
        if self._request.req_pool_idx is not None:
            raise DraftLifecycleError(
                "this allocator already holds a request slot; one branch, one request"
            )
        indices = self._requests.alloc([self._request])
        # The pool returns None when it is full, and assigns req_pool_idx in
        # place when it is not.
        if not indices or self._request.req_pool_idx is None:
            raise DraftLifecycleError("no private request slot is available")
        return int(self._request.req_pool_idx)

    def free_request(self, index: int) -> None:
        if self._request.req_pool_idx is None:
            # Already returned. Freeing again would trip the pool's assert.
            return
        if int(self._request.req_pool_idx) != int(index):
            raise DraftLifecycleError(
                f"asked to free slot {index} but this branch holds "
                f"{self._request.req_pool_idx}"
            )
        # free() sets req_pool_idx back to None, which is what makes a later
        # alloc treat this object as needing a fresh slot rather than reusing.
        self._requests.free(self._request)

    # -- KV rows ------------------------------------------------------------

    def alloc_kv(self, count: int):
        count = int(count)
        if count <= 0:
            raise DraftLifecycleError("a KV allocation must be positive")
        locations = self._kv.alloc(count)
        if locations is None:
            raise DraftLifecycleError(
                f"the private KV allocator could not supply {count} rows"
            )
        if len(locations) != count:
            # A partial allocation is not usable and must not be silently
            # kept: hand back whatever arrived before refusing.
            self._free_indices(locations)
            raise DraftLifecycleError(
                f"the private KV allocator supplied {len(locations)} of {count} rows"
            )
        return [int(loc) for loc in locations]

    def free_kv(self, locations) -> None:
        if locations is None:
            return
        self._free_indices(locations)

    def _free_indices(self, locations) -> None:
        """Release on the allocator's own device and dtype."""
        if isinstance(locations, torch.Tensor):
            tensor = locations
        else:
            values = [int(loc) for loc in locations]
            if not values:
                return
            tensor = torch.tensor(
                values, dtype=self._index_dtype(), device=self._index_device()
            )
        if tensor.numel() == 0:
            return
        tensor = tensor.to(device=self._index_device(), dtype=self._index_dtype())
        self._kv.free(tensor)

    def _index_device(self) -> torch.device:
        free_pages = getattr(self._kv, "free_pages", None)
        if isinstance(free_pages, torch.Tensor):
            return free_pages.device
        return torch.device(getattr(self._kv, "device", "cpu"))

    def _index_dtype(self) -> torch.dtype:
        free_pages = getattr(self._kv, "free_pages", None)
        if isinstance(free_pages, torch.Tensor):
            return free_pages.dtype
        # free_pages is int64 in both shipped allocators; used only when the
        # allocator has not been cleared yet and exposes no tensor.
        return torch.int64

    # -- the private request map -------------------------------------------

    def write_mapping(self, request_index: int, start: int, locations) -> None:
        table = self._requests.req_to_token
        values = torch.tensor(
            [int(loc) for loc in locations], dtype=table.dtype, device=table.device
        )
        table[int(request_index), int(start) : int(start) + len(values)] = values

    def clear_mapping(self, request_index: int) -> None:
        self._requests.req_to_token[int(request_index)].fill_(0)
