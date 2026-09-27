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
By default, **the prefix is recomputed on every call**. An explicitly
configured sidecar cache may retain one draft-owned prefix slot, under a
separate byte/slot budget and explicit request-incarnation identity. It
extends only when the new token tuple begins with the complete cached tuple;
replacement, retraction, incarnation change and sidecar retirement fence and
free old rows before they can be reused. Calls without identity, budget, or
token capacity keep the full-prefill path.

Caching avoids repeated O(prefix) prefill work when the opt-in sidecar owner
has budget and the same incarnation presents an append-only token prefix.
Otherwise the correctness-preserving full-prefix path remains. A full prefill per
prediction round costs O(prefix) work, and whether it fits has to be measured
against the prefetch window before anyone calls it sufficient. The cache
owner, budget, append checks and retirement fences are implemented below.

Nothing here samples into committed state, verifies, accepts or commits.
"""

from __future__ import annotations

import threading
import uuid
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
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferBudget,
    TransferCapacityError,
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

    def drain(self) -> None:
        """Prove all work on the private pools' device complete, or raise.

        Includes failed forwards, map writes and allocator bookkeeping. A
        synchronous CPU executor may explicitly implement a no-op. Missing
        support must never silently stand in for a CUDA completion fence.
        """


class SlotAllocator(Protocol):
    """Private request slots, KV rows, and the mapping between them.

    Never the target's allocator, and never the target's request map. The
    attention backend finds a request's KV by reading
    ``req_to_token_pool.req_to_token[req_index, :seq_len]``, so a forward that
    never wrote that mapping would read whatever the row happened to contain.
    Writing it is therefore part of the orchestration, not an optimisation --
    and the row written is always one this branch allocated.
    """

    def fork_for_branch(self) -> SlotAllocator:
        """Return branch-local ownership metadata over shared private pools.

        Must not allocate device storage; admission has not reserved scratch yet.
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


@dataclass(frozen=True)
class _PrefixCacheLease:
    """A serialized branch's temporary use of the one retained cache slot."""

    request_index: int
    prefix_length: int
    new_rows: Tuple[int, ...]
    cached_logits: Any = None


class _SidecarPrefixCache:
    """One incarnation-scoped, append-only draft KV owner.

    The provider serializes every call into this object. It owns one request
    slot, one reservation, and every retained prefix row. Branch handles may
    append committed tokens and temporarily append prediction rows, but only
    the cache frees the retained prefix. A failed fence or cleanup quarantines
    the cache and keeps its budget reservation.
    """

    _FIXED_OVERHEAD = 1 * 1024 * 1024

    def __init__(
        self,
        executor: ModelExecutor,
        allocator: SlotAllocator,
        budget: TransferBudget,
        *,
        max_prefix_tokens: int,
    ) -> None:
        self._executor = executor
        self._allocator = allocator.fork_for_branch()
        self._budget = budget
        self._owner = f"pvd-draft-prefix:{uuid.uuid4().hex}"
        capacity = budget.snapshot()["staging_bytes"]
        per_token = executor.bytes_per_token()
        if type(per_token) is not int or per_token <= 0:
            raise DraftCapabilityError(
                "prefix caching requires positive executor bytes_per_token"
            )
        # The sidecar prefix budget is independent from per-branch scratch and
        # the resident model/pool budget. Reserve one bounded slot, including
        # the private request map and a small fixed metadata allowance.
        row_bytes = per_token + 4  # KV row plus one request-map index
        usable = max(0, capacity - self._FIXED_OVERHEAD)
        self._max_tokens = min(max_prefix_tokens, usable // row_bytes)
        self._reservation_bytes = min(
            capacity,
            self._FIXED_OVERHEAD + self._max_tokens * row_bytes,
        )
        self._identity: Optional[Tuple[str, str]] = None
        self._tokens: Tuple[int, ...] = ()
        self._slot: Optional[int] = None
        self._rows: List[int] = []
        self._last_logits: Any = None
        self._invalid = False
        self._quarantined = False

    @property
    def enabled(self) -> bool:
        return self._max_tokens > 0

    @property
    def quarantined(self) -> bool:
        return self._quarantined

    @property
    def invalid(self) -> bool:
        return self._invalid

    @property
    def retained_tokens(self) -> Tuple[int, ...]:
        return self._tokens

    def prepare(
        self, identity: Tuple[str, str], tokens: Tuple[int, ...]
    ) -> Optional[_PrefixCacheLease]:
        if self._quarantined:
            raise DraftLifecycleError("draft prefix cache is quarantined")
        if not self.enabled or len(tokens) > self._max_tokens:
            if self._slot is not None:
                self.retire()
            return None

        if self._slot is not None and (
            self._identity != identity
            or len(tokens) < len(self._tokens)
            or tokens[: len(self._tokens)] != self._tokens
            or self._invalid
        ):
            self.retire()

        if self._slot is None:
            try:
                self._budget.reserve(self._owner, self._reservation_bytes, 1)
            except TransferCapacityError:
                # The optional cache never blocks the full-prefix path.
                return None
            try:
                self._slot = int(self._allocator.alloc_request())
                self._identity = identity
            except BaseException as exc:
                # The allocator may assign a slot before raising. Without an
                # exact returned index, neither its slot nor this reservation
                # can safely be recycled.
                self._quarantined = True
                raise DraftWorkerError(
                    f"draft prefix request-slot allocation is uncertain: {exc}"
                ) from exc

        start = len(self._tokens)
        if start == len(tokens):
            if self._last_logits is None:
                self.retire()
                return self.prepare(identity, tokens)
            return _PrefixCacheLease(
                self._slot, start, (), cached_logits=self._last_logits
            )

        try:
            allocated = self._allocator.alloc_kv(len(tokens) - start)
            rows = [int(row) for row in allocated]
            # Record every returned row before validating the count so a
            # partial result can be fenced and freed by retire().
            self._rows.extend(rows)
            if len(rows) != len(tokens) - start:
                raise DraftLifecycleError(
                    "allocator returned an incomplete cache suffix"
                )
            self._allocator.write_mapping(self._slot, start, rows)
        except BaseException as exc:
            # If allocation or mapping failed, retire the whole cache before
            # falling back. An allocation exception can hide a partial result,
            # so quarantine instead of refunding unknown rows.
            if len(self._rows) == len(self._tokens):
                self._quarantined = True
                raise DraftWorkerError(
                    f"draft prefix KV allocation is uncertain: {exc}"
                ) from exc
            self.retire()
            return None
        return _PrefixCacheLease(self._slot, start, tuple(rows))

    def publish(self, identity: Tuple[str, str], tokens: Tuple[int, ...], logits: Any):
        if (
            self._quarantined
            or self._slot is None
            or self._identity != identity
            or len(self._rows) != len(tokens)
        ):
            raise DraftLifecycleError("draft prefix cache publication lost its owner")
        if not torch.is_tensor(logits) or logits.ndim != 1:
            raise DraftLifecycleError(
                "draft prefix cache requires one-dimensional next-token logits"
            )
        logits_bytes = logits.numel() * logits.element_size()
        if logits_bytes > self._FIXED_OVERHEAD:
            raise DraftLifecycleError(
                "draft next-token logits exceed the cache's fixed byte allowance"
            )
        if not torch.isfinite(logits).all():
            raise DraftLifecycleError("draft prefix cache received non-finite logits")
        # Retain the small next-token distribution on CPU. The charged cache
        # owns the GPU KV and request map; keeping logits on the model device
        # would add an unbounded-by-prefix-model output tensor to that budget.
        if torch.is_tensor(logits) and logits.device.type != "cpu":
            logits = logits.detach().to(device="cpu")
        self._tokens = tokens
        self._last_logits = logits
        self._invalid = False

    def mark_invalid(self) -> None:
        self._invalid = True

    def retire(self) -> None:
        """Fence, clear and free the exact persistent owner, then refund it."""
        if self._quarantined:
            raise DraftWorkerError("draft prefix cache is quarantined")
        if self._slot is None:
            if self._identity is not None:
                self._budget.release(self._owner)
            self._identity = None
            self._tokens = ()
            self._rows.clear()
            self._last_logits = None
            self._invalid = False
            return
        try:
            self._executor.drain()
            self._allocator.clear_mapping(self._slot)
            self._executor.drain()
            if self._rows:
                self._allocator.free_kv(tuple(self._rows))
            self._allocator.free_request(self._slot)
            self._executor.drain()
        except BaseException as exc:
            self._quarantined = True
            raise DraftWorkerError(
                f"draft prefix cache could not be retired: {exc}"
            ) from exc
        self._budget.release(self._owner)
        self._identity = None
        self._tokens = ()
        self._slot = None
        self._rows.clear()
        self._last_logits = None
        self._invalid = False

    def restore_mapping(self, slot: int) -> None:
        """Drop a branch's temporary map suffix and restore retained rows."""
        if self._slot != slot or self._quarantined or self._invalid:
            raise DraftLifecycleError("draft prefix cache cannot restore its mapping")
        self._allocator.clear_mapping(slot)
        if self._rows:
            self._allocator.write_mapping(slot, 0, self._rows)

    def snapshot(self) -> dict:
        return {
            "enabled": self.enabled,
            "retained_tokens": len(self._tokens),
            "max_tokens": self._max_tokens,
            "owner_reserved": self._slot is not None,
            "quarantined": self._quarantined,
        }


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
        prefix_cache: Optional[_SidecarPrefixCache] = None,
        *,
        max_prefix_tokens: int,
        max_tokens: int,
        capabilities: DraftCapabilities,
    ) -> None:
        if not callable(getattr(executor, "drain", None)):
            raise DraftCapabilityError("executor must implement completion drain")
        self._branch_id = branch_id
        self._executor = executor
        self._allocator = allocator
        self._prefix_cache = prefix_cache
        self._cache_identity: Optional[Tuple[str, str]] = None
        self._cache_slot: Optional[int] = None
        self._max_prefix_tokens = max_prefix_tokens
        self._max_tokens = max_tokens
        self._capabilities = capabilities
        self._request_index: Optional[int] = None
        self._mapped = 0
        self._kv: List[int] = []
        self._released = False
        self._release_error: Optional[str] = None
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

    @property
    def prefix_cache_action(self) -> str:
        return getattr(self, "_prefix_cache_action", "recomputed")

    def set_prefix_cache_identity(self, identity: Optional[Tuple[str, str]]) -> None:
        """Set this branch's already-admitted sidecar request incarnation."""
        if identity is not None and (
            not isinstance(identity, tuple)
            or len(identity) != 2
            or any(not isinstance(value, str) or not value for value in identity)
        ):
            raise DraftLifecycleError(
                "exact request/incarnation cache identity required"
            )
        self._cache_identity = identity

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
        if self._release_error is not None:
            raise DraftLifecycleError("this execution handle is quarantined")

    # -- prefix -------------------------------------------------------------

    def prepare_prefix(self, tokens: Tuple[int, ...]) -> PreparedDraftPrefix:
        """Prepare a cache append for an identified sidecar Req, else prefill."""
        self._require_open()
        if self._request_index is not None:
            raise DraftLifecycleError(
                "this handle has already prepared a prefix; open a new branch"
            )
        length = len(tokens)
        self._capabilities.require_shape(
            prefix_tokens=length, predict_tokens=self._max_tokens
        )
        if self._prefix_cache is not None and self._cache_identity is not None:
            lease = self._prefix_cache.prepare(self._cache_identity, tokens)
            if lease is not None:
                self._request_index = lease.request_index
                self._cache_slot = lease.request_index
                if lease.cached_logits is not None and not lease.new_rows:
                    self._mapped = length
                    self._prefix_cache_action = "hit"
                    return PreparedDraftPrefix(
                        length=length,
                        last_logits=lease.cached_logits,
                        request_index=self._request_index,
                    )
                inputs = DraftForwardInputs(
                    forward_mode="extend",
                    input_ids=tuple(int(t) for t in tokens[lease.prefix_length :]),
                    positions=tuple(range(lease.prefix_length, length)),
                    seq_lens=(length,),
                    req_pool_indices=(self._request_index,),
                    out_cache_loc=lease.new_rows,
                    extend_prefix_lens=(lease.prefix_length,),
                    extend_seq_lens=(length - lease.prefix_length,),
                )
                self.forwards.append(inputs)
                try:
                    logits = self._executor.forward(inputs)
                    self._prefix_cache.publish(self._cache_identity, tokens, logits)
                except BaseException:
                    self._prefix_cache.mark_invalid()
                    raise
                self._mapped = length
                self._prefix_cache_action = (
                    "prefill" if lease.prefix_length == 0 else "append"
                )
                return PreparedDraftPrefix(
                    length=length, last_logits=logits, request_index=self._request_index
                )
        # Allocated here, so every row written below is one this branch owns.
        self._request_index = int(self._allocator.alloc_request())
        locations = [int(loc) for loc in self._allocator.alloc_kv(length)]
        self._kv.extend(locations)
        if len(locations) != length:
            raise DraftLifecycleError(
                f"allocator returned {len(locations)} KV rows for {length} tokens"
            )
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
        self._prefix_cache_action = "recomputed"
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
            try:
                token = self._pick(logits)
            except BaseException:
                if self._cache_slot is not None and self._prefix_cache is not None:
                    self._prefix_cache.mark_invalid()
                raise
            produced.append(token)
            if len(produced) == max_tokens:
                break
            # The token just produced becomes the next step's input. Its KV
            # row is allocated now, so the sequence length grows by exactly
            # one per step and never borrows a row from anywhere else.
            locations = [int(loc) for loc in self._allocator.alloc_kv(1)]
            self._kv.extend(locations)
            if len(locations) != 1:
                raise DraftLifecycleError("allocator must return exactly one KV row")
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
        if self._release_error is not None:
            raise DraftWorkerError(self._release_error)
        try:
            # A failed forward may have launched work without returning logits.
            # Never clear the map or publish free rows until that work is done.
            self._executor.drain()
            if self._cache_slot is not None:
                if self._prefix_cache is None:
                    raise DraftLifecycleError("cached slot lost its cache owner")
                if self._prefix_cache.invalid:
                    # A failed prefill may still have allocated speculative
                    # continuation rows. Clear the shared map first, then
                    # free those branch rows before retiring retained rows.
                    self._prefix_cache._allocator.clear_mapping(self._cache_slot)
                    if self._kv:
                        self._allocator.free_kv(tuple(self._kv))
                        self._executor.drain()
                    self._prefix_cache.retire()
                elif self._kv:
                    self._prefix_cache.restore_mapping(self._cache_slot)
                    self._executor.drain()
                    self._allocator.free_kv(tuple(self._kv))
                    self._executor.drain()
                self._kv.clear()
                self._request_index = None
                self._cache_slot = None
                self._mapped = 0
                self._released = True
                return
            if self._request_index is not None:
                self._allocator.clear_mapping(self._request_index)
            self._executor.drain()
            if self._kv:
                self._allocator.free_kv(tuple(self._kv))
            if self._request_index is not None:
                self._allocator.free_request(self._request_index)
            # free() may itself enqueue CUDA bookkeeping. The provider holds
            # its shared execution lock until this fence and all accounting.
            self._executor.drain()
        except BaseException as exc:
            # Even a partially successful free is not retryable: the provider
            # must quarantine the shared allocator, not just lose one slot.
            self._release_error = (
                f"branch {self._branch_id} could not be released: {exc}"
            )
            raise DraftWorkerError(self._release_error) from exc
        self._kv.clear()
        self._request_index = None
        self._mapped = 0
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
        prefix_cache_budget: Optional[TransferBudget] = None,
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
        if not callable(getattr(allocator, "fork_for_branch", None)):
            raise DraftCapabilityError("allocator must provide branch-local ownership")
        if prefix_cache_budget is not None and not isinstance(
            prefix_cache_budget, TransferBudget
        ):
            raise DraftCapabilityError(
                "prefix_cache_budget must be a dedicated TransferBudget"
            )
        self._allocator = allocator
        self._capabilities = capabilities
        self._persistent_bytes = persistent_bytes
        self._max_tokens = max_tokens
        self.prefix_cache_budget = prefix_cache_budget
        self._prefix_cache = (
            None
            if prefix_cache_budget is None
            else _SidecarPrefixCache(
                executor,
                allocator,
                prefix_cache_budget,
                max_prefix_tokens=capabilities.max_prefix_tokens,
            )
        )
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

    @property
    def prefix_cache_enabled(self) -> bool:
        return self._prefix_cache is not None and self._prefix_cache.enabled

    def prefix_cache_snapshot(self) -> Optional[dict]:
        return None if self._prefix_cache is None else self._prefix_cache.snapshot()

    def retire_prefix_cache(self) -> None:
        if self._prefix_cache is not None:
            self._prefix_cache.retire()

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
            self._allocator.fork_for_branch(),
            self._prefix_cache,
            max_prefix_tokens=prefix_tokens,
            max_tokens=max_tokens,
            capabilities=self._capabilities,
        )
