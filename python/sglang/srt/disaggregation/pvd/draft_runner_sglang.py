"""The prediction-only execution path over SGLang's model-runner machinery.

This is the orchestration ``draft_sglang.SGLangDraftProvider`` drives: private
request slots, private KV rows, positions and sequence lengths maintained by
hand, and bounded continuation steps. It reuses SGLang's loading, model
registry and forward machinery; it reimplements no model's forward function
and never touches a ``ScheduleBatch``.

The seam and why it is here
---------------------------
``ForwardBatch.init_new(batch, model_runner)`` takes a ``ScheduleBatch`` -- the
live scheduler object, with tree cache, sampling info and the committed
request list attached. Building the prediction path on it would reintroduce
exactly the coupling the reuse audit exists to avoid.

So this module produces ``DraftForwardInputs``: the tensors a forward actually
needs -- ids, positions, sequence lengths, request-pool indices and KV write
locations -- and hands them to a ``ModelExecutor``. In production that
executor maps them onto a ``ForwardBatch`` for the chosen model and attention
backend; that mapping is architecture-specific and is the one piece this
module does not implement. ``draft_forward_adapter`` supplies that mapping;
the strict CPU smoke now exercises it and this handle with a real tiny model.
Other model/backend combinations still require execution validation.

Prefix handling
---------------
**The prefix is recomputed on every call.** ``prepare_prefix`` runs one
EXTEND forward over the whole committed snapshot into rows this branch owns,
and ``release`` frees them. Nothing survives the branch, so retained-KV
ownership, budgeting and invalidation do not arise.

That is a correctness baseline, not the latency answer. A full prefill per
prediction round costs O(prefix) work, and whether it fits has to be measured
against the prefetch window before anyone calls it sufficient. Persistent
prefix caching would replace ``prepare_prefix``, and would first need: a
named owner for the retained KV, a persistent budget distinct from per-branch
scratch, incremental extension as the committed prefix grows, and
invalidation on prefix replacement, token retraction, Entry replacement and
request close. None of that is implemented, and none of it is assumed.

Nothing here samples into committed state, verifies, accepts or commits.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Sequence, Tuple

import torch
from sglang.srt.disaggregation.pvd.draft_sglang import (
    DraftCapabilities,
    DraftCapabilityError,
    DraftLifecycleError,
    DraftWorkerError,
)

#: The initial supported subset. Stated narrowly on purpose: these are the
#: shapes the bookkeeping below is written for, not a survey of what might
#: happen to work.
DEFAULT_CAPABILITIES = DraftCapabilities(
    architectures=("LlamaForCausalLM", "Qwen2ForCausalLM"),
    attention_backends=("flashinfer", "triton", "torch_native"),
    max_prefix_tokens=8192,
    max_predict_tokens=16,
)


@dataclass(frozen=True)
class DraftForwardInputs:
    """Exactly what one forward needs, and nothing borrowed from a batch.

    ``req_pool_indices`` and ``out_cache_loc`` name rows this branch owns.
    They are produced here rather than taken from a ``ScheduleBatch`` so that
    no committed request's mapping is read or written.
    """

    forward_mode: str
    input_ids: Tuple[int, ...]
    positions: Tuple[int, ...]
    seq_lens: Tuple[int, ...]
    req_pool_indices: Tuple[int, ...]
    out_cache_loc: Tuple[int, ...]
    extend_prefix_lens: Tuple[int, ...] = ()
    extend_seq_lens: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.forward_mode not in ("extend", "decode"):
            raise DraftCapabilityError(
                f"unsupported forward mode {self.forward_mode!r}"
            )
        if not self.input_ids:
            raise DraftLifecycleError("a forward needs at least one token")
        if len(self.input_ids) != len(self.positions):
            raise DraftLifecycleError("one position per token is required")
        if len(self.input_ids) != len(self.out_cache_loc):
            raise DraftLifecycleError("one KV location per token is required")
        if len(self.seq_lens) != len(self.req_pool_indices):
            raise DraftLifecycleError("one sequence length per request")


class ModelExecutor(Protocol):
    """Runs one forward and returns next-token logits for the last position.

    The production implementation maps ``DraftForwardInputs`` onto a
    ``ForwardBatch`` and calls ``ModelRunner.forward``. That mapping depends
    on the model and the attention backend and is deliberately not written
    here; this protocol is where it plugs in.
    """

    def architecture(self) -> str: ...

    def attention_backend(self) -> str: ...

    def bytes_per_token(self) -> int:
        """KV bytes one token occupies in this model's private pool."""

    def transient_bytes(self, prefix_tokens: int, predict_tokens: int) -> int:
        """Declared peak non-KV bytes: inputs, logits, activations and workspace.

        Must cover the whole branch, including logits retained during a later
        forward. Unknown is an error, never zero. This is an admission contract,
        not a claim that PyTorch allocations are intercepted or hard-capped.
        """

    def forward(self, inputs: DraftForwardInputs) -> torch.Tensor:
        """Return logits for the final position, shape [vocab]."""


class SlotAllocator(Protocol):
    """Private request slots, KV rows, and the mapping between them.

    Never the target's allocator, and never the target's request map. The
    attention backend finds a request's KV by reading
    ``req_to_token_pool.req_to_token[req_index, :seq_len]``, so a forward that
    never wrote that mapping would read whatever the row happened to contain.
    Writing it is therefore part of the orchestration, not an optimisation --
    and the row written is always one this branch allocated.
    """

    def alloc_request(self) -> int: ...

    def free_request(self, index: int) -> None: ...

    def alloc_kv(self, count: int) -> Sequence[int]: ...

    def free_kv(self, locations: Sequence[int]) -> None: ...

    def write_mapping(
        self, request_index: int, start: int, locations: Sequence[int]
    ) -> None:
        """Record where this request's tokens ``start..`` live in the KV pool."""

    def clear_mapping(self, request_index: int) -> None:
        """Forget the mapping, so a reused slot never inherits stale rows."""


@dataclass
class PreparedDraftPrefix:
    """What the prefill produced, carried into the continuation steps."""

    length: int
    last_logits: Any
    request_index: int

    @property
    def length_(self) -> int:  # pragma: no cover - Protocol compatibility
        return self.length


class SGLangDraftHandle:
    """One branch's execution state: its slots, its KV rows, its cleanup.

    Every row this handle writes is one it allocated. It holds no reference
    to a committed request, a live batch, or the target's pools, so "does not
    mutate committed state" is a property of what it can reach rather than of
    how carefully it behaves.
    """

    def __init__(
        self,
        branch_id: str,
        executor: ModelExecutor,
        allocator: SlotAllocator,
        *,
        max_prefix_tokens: int,
        max_tokens: int,
        capabilities: DraftCapabilities,
    ) -> None:
        self._branch_id = branch_id
        self._executor = executor
        self._allocator = allocator
        self._max_prefix_tokens = max_prefix_tokens
        self._max_tokens = max_tokens
        self._capabilities = capabilities
        self._request_index: Optional[int] = None
        self._mapped = 0
        self._kv: List[int] = []
        self._released = False
        #: Everything handed to the executor, for inspection in tests.
        self.forwards: List[DraftForwardInputs] = []

    @property
    def branch_id(self) -> str:
        return self._branch_id

    @property
    def owned_kv(self) -> Tuple[int, ...]:
        return tuple(self._kv)

    @property
    def mapped_tokens(self) -> int:
        """How many positions of this request's map this handle has written."""
        return self._mapped

    @property
    def request_index(self) -> Optional[int]:
        return self._request_index

    def scratch_bytes(self) -> int:
        """KV-capacity credits plus an explicit non-KV peak reservation.

        Private pools are physically allocated once (persistent budget); KV
        credits here bound branch occupancy, not a second physical allocation.
        """
        per_token = self._executor.bytes_per_token()
        if (
            isinstance(per_token, bool)
            or not isinstance(per_token, int)
            or per_token < 0
        ):
            raise DraftCapabilityError("bytes_per_token must be non-negative")
        estimate = getattr(self._executor, "transient_bytes", None)
        if not callable(estimate):
            raise DraftCapabilityError("executor must declare non-KV transient bytes")
        transient = estimate(self._max_prefix_tokens, self._max_tokens)
        if (
            isinstance(transient, bool)
            or not isinstance(transient, int)
            or transient < 0
        ):
            raise DraftCapabilityError(
                "non-KV transient bytes must be a non-negative integer"
            )
        return (self._max_prefix_tokens + self._max_tokens) * per_token + transient

    def _require_open(self) -> None:
        if self._released:
            raise DraftLifecycleError(
                "this execution handle has been released; its rows may belong "
                "to another branch now"
            )

    # -- prefix -------------------------------------------------------------

    def prepare_prefix(self, tokens: Tuple[int, ...]) -> PreparedDraftPrefix:
        """One EXTEND forward over the whole snapshot, into private rows.

        Recomputed every call; see the module docstring for why, and for what
        a caching implementation would have to establish first.
        """
        self._require_open()
        if self._request_index is not None:
            raise DraftLifecycleError(
                "this handle has already prepared a prefix; open a new branch"
            )
        length = len(tokens)
        self._capabilities.require_shape(
            prefix_tokens=length, predict_tokens=self._max_tokens
        )
        # Allocated here, so every row written below is one this branch owns.
        self._request_index = int(self._allocator.alloc_request())
        locations = [int(loc) for loc in self._allocator.alloc_kv(length)]
        if len(locations) != length:
            raise DraftLifecycleError(
                f"allocator returned {len(locations)} KV rows for {length} tokens"
            )
        self._kv.extend(locations)
        # The attention backend reads KV locations out of this map, so the
        # forward below would be meaningless without it. Private row, private
        # allocator: no committed request's mapping is read or written.
        self._allocator.write_mapping(self._request_index, 0, locations)
        self._mapped = length
        inputs = DraftForwardInputs(
            forward_mode="extend",
            input_ids=tuple(int(t) for t in tokens),
            # Absolute sequence positions, from zero: this is a fresh
            # computation of the whole prefix, not a continuation of one.
            positions=tuple(range(length)),
            seq_lens=(length,),
            req_pool_indices=(self._request_index,),
            out_cache_loc=tuple(locations),
            extend_prefix_lens=(0,),
            extend_seq_lens=(length,),
        )
        self.forwards.append(inputs)
        logits = self._executor.forward(inputs)
        return PreparedDraftPrefix(
            length=length, last_logits=logits, request_index=self._request_index
        )

    # -- bounded continuation ----------------------------------------------

    def generate(self, prepared: PreparedDraftPrefix, max_tokens: int) -> Sequence[int]:
        """Step at most ``max_tokens`` times. No verification, no commit."""
        self._require_open()
        if not isinstance(prepared, PreparedDraftPrefix):
            raise DraftLifecycleError("generate needs a prepared prefix")
        if prepared.request_index != self._request_index:
            raise DraftLifecycleError("this prepared prefix belongs to another handle")
        if max_tokens > self._max_tokens:
            raise DraftCapabilityError(
                f"{max_tokens} exceeds this handle's bound of {self._max_tokens}"
            )
        produced: List[int] = []
        logits = prepared.last_logits
        position = prepared.length
        for _ in range(max_tokens):
            token = self._pick(logits)
            produced.append(token)
            if len(produced) == max_tokens:
                break
            # The token just produced becomes the next step's input. Its KV
            # row is allocated now, so the sequence length grows by exactly
            # one per step and never borrows a row from anywhere else.
            locations = [int(loc) for loc in self._allocator.alloc_kv(1)]
            self._kv.extend(locations)
            # Appended at this step's position, so the map grows by exactly
            # one row per step and stays consistent with seq_lens.
            self._allocator.write_mapping(self._request_index, position, locations)
            self._mapped = position + 1
            inputs = DraftForwardInputs(
                forward_mode="decode",
                input_ids=(token,),
                positions=(position,),
                seq_lens=(position + 1,),
                req_pool_indices=(self._request_index,),
                out_cache_loc=tuple(locations),
            )
            self.forwards.append(inputs)
            logits = self._executor.forward(inputs)
            position += 1
        return produced

    @staticmethod
    def _pick(logits: Any) -> int:
        """Greedy. Sampling would need an RNG whose isolation is established."""
        if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
            raise DraftLifecycleError("the executor must return 1-D logits")
        if not torch.isfinite(logits).all():
            raise DraftLifecycleError("the executor returned non-finite logits")
        return int(torch.argmax(logits).item())

    # -- cleanup ------------------------------------------------------------

    def release(self) -> None:
        """Free this handle's rows and slot. Idempotent; partial-safe.

        If freeing fails, the handle stays un-released and raises, so the
        provider quarantines the branch rather than reissuing memory that may
        still be live.
        """
        if self._released:
            return
        errors = []
        if self._kv:
            try:
                self._allocator.free_kv(tuple(self._kv))
            except BaseException as exc:
                errors.append(f"KV rows: {exc}")
            else:
                self._kv.clear()
        if self._request_index is not None:
            try:
                # Cleared before the slot is returned, so a later branch that
                # is handed this index cannot read rows it does not own.
                self._allocator.clear_mapping(self._request_index)
                self._mapped = 0
                self._allocator.free_request(self._request_index)
            except BaseException as exc:
                errors.append(f"request slot: {exc}")
            else:
                self._request_index = None
        if errors:
            raise DraftWorkerError(
                f"branch {self._branch_id} could not be released: " + "; ".join(errors)
            )
        self._released = True

    @property
    def released(self) -> bool:
        return self._released


class SGLangDraftRunnerFactory:
    """Mints one handle per branch over one shared model and one pool set.

    The model is loaded once. Handles are cheap: they hold indices, not
    weights. ``persistent_bytes`` is what the weights and the private pools
    cost, charged once against the persistent budget, never per branch.
    """

    def __init__(
        self,
        executor: ModelExecutor,
        allocator: SlotAllocator,
        *,
        capabilities: DraftCapabilities = DEFAULT_CAPABILITIES,
        persistent_bytes: int = 0,
        max_tokens: int = 8,
    ) -> None:
        if not isinstance(capabilities, DraftCapabilities):
            raise DraftCapabilityError("explicit DraftCapabilities are required")
        # Checked against what the loaded model actually reports, not against
        # the capability declaration itself.
        capabilities.require_model(
            architecture=str(executor.architecture()),
            attention_backend=str(executor.attention_backend()),
        )
        if (
            isinstance(persistent_bytes, bool)
            or not isinstance(persistent_bytes, int)
            or persistent_bytes < 0
        ):
            raise DraftCapabilityError(
                "persistent_bytes must be a non-negative integer"
            )
        self._executor = executor
        self._allocator = allocator
        self._capabilities = capabilities
        self._persistent_bytes = persistent_bytes
        self._max_tokens = max_tokens
        self._opened = deque(maxlen=64)
        self._opened_count = 0
        self._diagnostics_lock = threading.Lock()

    @property
    def opened(self) -> tuple[str, ...]:
        """Bounded diagnostic ids, never a lifetime branch ledger."""
        with self._diagnostics_lock:
            return tuple(self._opened)

    @property
    def opened_count(self) -> int:
        with self._diagnostics_lock:
            return self._opened_count

    def capabilities(self) -> DraftCapabilities:
        return self._capabilities

    def persistent_bytes(self) -> int:
        return self._persistent_bytes

    def open(
        self, *, branch_id: str, prefix_tokens: int, max_tokens: int
    ) -> SGLangDraftHandle:
        self._capabilities.require_shape(
            prefix_tokens=prefix_tokens, predict_tokens=max_tokens
        )
        with self._diagnostics_lock:
            self._opened.append(branch_id)
            self._opened_count += 1
        return SGLangDraftHandle(
            branch_id,
            self._executor,
            self._allocator,
            max_prefix_tokens=prefix_tokens,
            max_tokens=max_tokens,
            capabilities=self._capabilities,
        )
