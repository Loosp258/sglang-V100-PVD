"""Prediction-only reuse of SGLang's draft-model execution infrastructure.

The project already has a ``DraftProvider`` contract and a Hugging Face
implementation of it. This module supplies the other implementation: one that
drives the draft model SGLang itself would have loaded, through SGLang's own
worker construction, without ever entering SGLang's speculative *generation*
path.

Why not simply call ``draft()``
-------------------------------
``StandaloneWorker.draft()`` is not a pure function of a prefix. Audited
against ``speculative/eagle_worker.py`` and ``speculative/standalone_worker.py``
at the current HEAD, one call to it mutates committed state in at least six
ways, and the mutations are not all undone:

* ``standalone_worker.py`` takes the request/token map and the KV allocator
  **from the target worker** (``target_worker.get_memory_pool()``), and
  ``clear_cache_pool()`` is a deliberate no-op with the comment "allocator and
  kv cache pool are shared with target worker".
* ``_draft_preprocess_decode`` increments ``req.decode_batch_idx`` for every
  request in the batch.
* It feeds the committed sampler: ``sampling_info.penalizer_orchestrator
  .cumulate_output_tokens(...)``.
* It calls ``batch.maybe_evict_swa()``, which evicts from the shared cache.
* It allocates from the shared allocator, and the ``assign_draft_cache_locs``
  Triton kernel **writes into ``req_to_token_pool.req_to_token``** -- the live
  request-to-slot mapping.
* It overwrites ``batch.out_cache_loc``, ``batch.seq_lens_sum``,
  ``batch.return_hidden_states`` and ``spec_info.positions``.

The allocator is snapshotted and restored (``backup_state=True`` /
``restore_state``). The ``req_to_token`` writes, the counter increments, the
penalizer accumulation and the batch-field overwrites are **not**. And the
return value is an ``EagleVerifyInput``: an object built to be verified and
committed, which is the path this project must never enter.

So ``draft()`` is not the reuse point. The reuse points are the pieces
underneath it: SGLang's model loading, its worker construction, and its
forward machinery -- driven from a snapshot, against pools nobody else owns.

Ownership model
---------------
Every prediction runs inside a **branch**, and a branch owns an
``DraftExecutionHandle`` minted for it by a factory. The handle owns that
branch's request slots, KV rows and scratch, and is the only thing whose
``release`` can free them; a runner shared between branches with an
unqualified ``release()`` cannot say whose resources it just freed.

Weights are **not** per branch. One model, one worker, one set of pools,
shared by every handle. Their bytes are a persistent reservation made once,
against a different budget from the per-branch scratch, because they have
different lifetimes and conflating them makes a full model look like a
transient allocation.

Owning resources separately is not the same as being safe to execute
concurrently. ``ModelRunner`` and its attention backend carry per-forward
state, so **execution is serialized** behind one lock until concurrent use is
positively established. The branch limit bounds how many handles may exist,
not how many forwards may run at once.

Release ordering
----------------
Budget and admission capacity are released **after** the resources they stand
for, never before. A branch that has returned but whose handle has not been
released yet still holds both. If a handle's ``release`` fails, its
reservation is **not** refunded and its admission slot is **not** returned:
the resources may still be live, so making them reusable would hand them to a
second owner. The provider records the failure and reports itself degraded.

None of this makes SGLang's speculative decoding available under PVD. The
startup prohibition in ``arg_groups/pvd_disaggregation_hook.py`` is untouched:
this path is configured through PVD's own flags, translated into a private
configuration copy, and never sets ``speculative_algorithm`` anywhere.
"""

from __future__ import annotations

import copy
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from sglang.srt.disaggregation.pvd.draft_hf import VocabularySignature
from sglang.srt.disaggregation.pvd.prediction import (
    CommittedPrefix,
    DraftConfig,
    DraftPrediction,
    DraftProvider,
    PredictionConfigError,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    TransferBudget,
    TransferCapacityError,
)

#: The only names a prediction path may reach on a draft worker. This is an
#: allowlist: a name that is not here raises, whether or not it exists
#: upstream, so a new upstream method is unreachable until it is reviewed.
ALLOWED_WORKER_METHODS = ("get_memory_pool", "model_config", "device")

#: Named so a refusal can say what was reached for. Not the mechanism: the
#: mechanism is that everything outside ALLOWED_WORKER_METHODS is refused.
KNOWN_GENERATION_METHODS = (
    "draft",
    "draft_extend",
    "verify",
    "forward_batch_generation",
    "forward_target_extend",
    "capture_for_decode",
    "on_verify_complete_cpu",
)


class DraftWorkerError(RuntimeError):
    """The draft worker cannot be used for prediction-only execution."""


class DraftCapabilityError(DraftWorkerError):
    """This model, backend or request shape is outside the supported subset."""


class DraftLifecycleError(DraftWorkerError):
    """A prediction was attempted outside its branch, or after a failed one."""


# --------------------------------------------------------------------------
# The worker surface
# --------------------------------------------------------------------------


class DraftWorkerInterface:
    """A narrow, explicit view of a draft worker.

    **Allowlist, not deny list.** Only the names in
    ``ALLOWED_WORKER_METHODS`` are reachable. Everything else raises on
    attribute access, including names that do not exist upstream today, so a
    method added to ``TpModelWorker`` tomorrow is unreachable from here until
    someone adds it to the list on purpose.

    What this does **not** guarantee: that the underlying worker is
    unreachable by other means. Anything already holding the worker object can
    still call whatever it likes. This bounds what *this* module can do.
    """

    __slots__ = ("_worker", "_refused")

    def __init__(self, worker: Any):
        object.__setattr__(self, "_worker", worker)
        object.__setattr__(self, "_refused", [])

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in ALLOWED_WORKER_METHODS:
            object.__getattribute__(self, "_refused").append(name)
            known = (
                " (a speculative generation entry point)"
                if (name in KNOWN_GENERATION_METHODS)
                else ""
            )
            raise DraftWorkerError(
                f"{name!r} is not on the prediction-only allowlist{known}; "
                f"reachable names are {ALLOWED_WORKER_METHODS}"
            )
        return getattr(object.__getattribute__(self, "_worker"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise DraftWorkerError(
            "the prediction path does not mutate the draft worker; "
            f"refused to set {name!r}"
        )

    @property
    def refused(self) -> Tuple[str, ...]:
        """Names that were reached for and refused. Empty is the only pass."""
        return tuple(object.__getattribute__(self, "_refused"))


# --------------------------------------------------------------------------
# Pool ownership
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolOwnership:
    """What was actually established about pool separation.

    ``storage_verified`` is the claim that matters and the one that is easy to
    fake: two distinct pool *objects* can wrap the same buffer, so object
    inequality proves nothing about the memory. When the underlying storage
    cannot be inspected this stays ``False`` and ``note`` says why -- the
    provider then reports "not storage-verified" rather than "private".
    """

    distinct_objects: bool
    storage_verified: bool
    note: str = ""

    def describe(self) -> str:
        if self.storage_verified:
            return "private (storage-verified)"
        if self.distinct_objects:
            return f"distinct objects, storage not verified: {self.note}"
        return "shared"


def _storage_keys(pool: Any) -> Tuple[int, ...]:
    """Identify the memory a pool stands on, when it can be identified."""
    keys: List[int] = []
    seen = set()
    candidates: List[Any] = [pool]
    while candidates:
        item = candidates.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, (list, tuple)):
            candidates.extend(item)
            continue
        # Real SGLang allocators own indices, not the K/V tensors directly.
        # Follow their backing cache; inspecting only allocator identity misses
        # two allocators writing the same physical target pool.
        for name in (
            "req_to_token",
            "_kvcache",
            "kv_buffer",
            "k_buffer",
            "v_buffer",
            "buffer",
            "_kv_buffer",
            "data",
        ):
            # Tensor.data creates another tensor view on every access. Inspect
            # tensor storage directly instead of recursively chasing that view.
            if callable(getattr(item, "untyped_storage", None)):
                break
            value = getattr(item, name, None)
            if value is not None:
                candidates.append(value)
        storage = getattr(item, "untyped_storage", None)
        if callable(storage):
            try:
                keys.append(storage().data_ptr())
            except Exception:  # pragma: no cover - exotic tensors
                continue
    return tuple(sorted(set(keys)))


def require_private_pools(draft_worker: Any, target_worker: Any) -> PoolOwnership:
    """Refuse a draft worker that shares the target's pools, and say how sure.

    SGLang's own ``StandaloneWorker`` deliberately shares both, so a worker
    built the usual way fails here -- which is the point. What this returns is
    as important as that it does not raise: it reports whether the *storage*
    was actually compared or only the objects.
    """
    try:
        draft_pools = draft_worker.get_memory_pool()
        target_pools = target_worker.get_memory_pool()
    except AttributeError as exc:  # pragma: no cover - defensive
        raise DraftWorkerError(
            "a draft worker must expose get_memory_pool() to be checked"
        ) from exc
    names = ("req_to_token_pool", "token_to_kv_pool_allocator")
    notes: List[str] = []
    verified = True
    for name, mine, theirs in zip(names, draft_pools, target_pools):
        if mine is None:
            raise DraftWorkerError(f"the draft worker has no {name} of its own")
        if mine is theirs:
            raise DraftWorkerError(
                f"the draft worker shares the target's {name}; prediction "
                "would write into committed request state. Build it with "
                f"{name}=None so ModelRunner allocates a private one"
            )
        mine_keys, their_keys = _storage_keys(mine), _storage_keys(theirs)
        if not mine_keys or not their_keys:
            verified = False
            notes.append(f"{name} exposes no inspectable storage")
            continue
        overlap = set(mine_keys) & set(their_keys)
        if overlap:
            raise DraftWorkerError(
                f"the draft worker's {name} stands on the same memory as the "
                f"target's ({len(overlap)} shared buffer(s)); distinct pool "
                "objects over one buffer are not private pools"
            )
    return PoolOwnership(
        distinct_objects=True,
        storage_verified=verified,
        note="; ".join(notes),
    )


# --------------------------------------------------------------------------
# Capability subset
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftCapabilities:
    """What this runner claims to support. Everything else is refused.

    Stated positively and narrowly on purpose. The first runner targets
    standard autoregressive causal LMs with a full-attention KV layout; no
    claim is made about anything else, and an unsupported combination is
    rejected before any allocation rather than discovered during a forward.
    """

    architectures: Tuple[str, ...]
    attention_backends: Tuple[str, ...]
    max_prefix_tokens: int
    max_predict_tokens: int
    paged_kv: bool = False

    def __post_init__(self) -> None:
        for name in ("architectures", "attention_backends"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not value:
                raise DraftCapabilityError(f"{name} must be a non-empty tuple")
        for name in ("max_prefix_tokens", "max_predict_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DraftCapabilityError(f"{name} must be a positive integer")

    def require_shape(self, *, prefix_tokens: int, predict_tokens: int) -> None:
        """Check the two things a caller legitimately knows about a request.

        Deliberately separate from ``require_model``: a provider holds a
        prefix and a token budget, not a model config. Checking a declared
        architecture against the declaration it came from would pass by
        construction and prove nothing.
        """
        if prefix_tokens > self.max_prefix_tokens:
            raise DraftCapabilityError(
                f"prefix of {prefix_tokens} tokens exceeds the supported "
                f"{self.max_prefix_tokens}"
            )
        if predict_tokens > self.max_predict_tokens:
            raise DraftCapabilityError(
                f"{predict_tokens} predicted tokens exceeds the supported "
                f"{self.max_predict_tokens}"
            )

    def require_model(self, *, architecture: str, attention_backend: str) -> None:
        """Check a model against this subset. Called by whoever loaded it,
        with the values the loaded model actually reports."""
        if architecture not in self.architectures:
            raise DraftCapabilityError(
                f"architecture {architecture!r} is outside the supported "
                f"subset {self.architectures}; this runner makes no claim "
                "about it and will not guess"
            )
        if attention_backend not in self.attention_backends:
            raise DraftCapabilityError(
                f"attention backend {attention_backend!r} is outside the "
                f"supported subset {self.attention_backends}"
            )


# --------------------------------------------------------------------------
# Branch-owned execution
# --------------------------------------------------------------------------


class PreparedPrefix(Protocol):
    """Whatever a handle needs to carry from prefix preparation into steps."""

    @property
    def length(self) -> int: ...


class DraftExecutionHandle(Protocol):
    """One branch's execution state. The only thing that can free it.

    A handle owns its request slots, KV rows and scratch. ``release`` frees
    exactly those, and nothing else; it is never a global cleanup.
    """

    @property
    def branch_id(self) -> str: ...

    def scratch_bytes(self) -> int:
        """Bytes this handle will hold. Reserved before it allocates."""

    def prepare_prefix(self, tokens: Tuple[int, ...]) -> PreparedPrefix:
        """Compute whatever the continuation steps need from the prefix.

        An explicit seam. The first runner **recomputes the prefix on every
        call** and keeps nothing between predictions: ownership, budgeting and
        invalidation are then trivial, because nothing outlives the branch.
        That is a correctness baseline, not the final latency answer -- the
        prefill cost has to be measured against the available prefetch window
        before anyone calls it good enough.

        Persistent prefix caching would replace this method, and would first
        need: an owner for the retained KV, a persistent budget separate from
        per-branch scratch, incremental extension as the prefix grows, and
        invalidation on prefix replacement, retraction, Entry change and
        request close. None of that exists, so none of it is assumed.
        """

    def generate(self, prepared: PreparedPrefix, max_tokens: int) -> Sequence[int]:
        """Run bounded continuation steps and return candidate token ids."""

    def release(self) -> None:
        """Free this handle's slots, KV rows and scratch. Idempotent."""


class DraftRunnerFactory(Protocol):
    """Mints one execution handle per branch over shared model weights."""

    def capabilities(self) -> DraftCapabilities:
        """What this runner supports. Checked before anything is allocated."""

    def persistent_bytes(self) -> int:
        """Weights plus private pools: charged once, not per branch."""

    def open(
        self, *, branch_id: str, prefix_tokens: int, max_tokens: int
    ) -> DraftExecutionHandle:
        """Mint branch-local metadata without allocating execution/device storage.

        The provider reserves scratch before prepare_prefix may allocate it.
        """


@dataclass
class _Branch:
    """Bookkeeping for one admitted branch."""

    branch_id: str
    handle: Any
    owner: str
    reserved_bytes: int


@dataclass(frozen=True)
class QuarantinedBranch:
    """A branch whose cleanup failed. Its resources are never reissued."""

    branch_id: str
    owner: str
    reserved_bytes: int
    reason: str


@dataclass(frozen=True)
class DraftPlacement:
    """Where the prediction-only draft worker runs and what it may cost.

    Separate from ``DraftConfig`` because these are deployment facts, not
    model identity, and because none of them has a defensible default: a
    budget that fits one GPU can be fatal on another.

    ``scratch_budget_bytes`` bounds **per-branch** execution scratch only.
    ``persistent_budget_bytes`` bounds the model weights and the private
    pools, which are allocated once and live as long as the provider. They
    are separate numbers against separate budgets because they have separate
    lifetimes; charging a model's weights to a per-call budget would make a
    permanent allocation look transient and would let scratch accounting
    "succeed" while the device is already full.
    """

    gpu_id: int = 0
    tp_rank: int = 0
    scratch_budget_bytes: int = 0
    persistent_budget_bytes: int = 0
    max_concurrent_branches: int = 1

    def __post_init__(self) -> None:
        for name in ("gpu_id", "tp_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PredictionConfigError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.scratch_budget_bytes, bool)
            or not isinstance(self.scratch_budget_bytes, int)
            or self.scratch_budget_bytes <= 0
        ):
            raise PredictionConfigError(
                "scratch_budget_bytes must be a positive integer; a prediction "
                "branch that is not bounded can starve the committed path"
            )
        if (
            isinstance(self.persistent_budget_bytes, bool)
            or not isinstance(self.persistent_budget_bytes, int)
            or self.persistent_budget_bytes < 0
        ):
            raise PredictionConfigError(
                "persistent_budget_bytes must be a non-negative integer"
            )
        if (
            isinstance(self.max_concurrent_branches, bool)
            or not isinstance(self.max_concurrent_branches, int)
            or self.max_concurrent_branches <= 0
        ):
            raise PredictionConfigError(
                "max_concurrent_branches must be a positive integer"
            )


# --------------------------------------------------------------------------
# Configuration translation
# --------------------------------------------------------------------------


def build_draft_server_args(server_args: Any, placement: DraftPlacement) -> Any:
    """A private configuration for the draft model. The target's is untouched.

    The loader reads ``model_path``, ``revision`` and ``device``; PVD's flags
    are ``pvd_draft_*``. Passing the target's configuration through unchanged
    would load the *target* model a second time, which is the opposite of
    what a separate draft model is for. So this makes a deep copy, maps the
    fields the loader actually reads, and forces the speculative
    configuration off inside the copy.

    Returns the copy. The caller's object is not modified; a test asserts it.
    """
    model_path = getattr(server_args, "pvd_draft_model_path", None)
    if not model_path:
        raise DraftWorkerError(
            "no draft model is configured; set --pvd-draft-model-path"
        )
    private = copy.deepcopy(server_args)
    private.model_path = model_path
    fraction = getattr(server_args, "pvd_draft_mem_fraction_static", None)
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0 < fraction < 1
    ):
        raise DraftWorkerError(
            "a private draft worker requires --pvd-draft-mem-fraction-static "
            "between 0 and 1; reusing the target's KV-pool fraction can "
            "exhaust its device"
        )
    private.mem_fraction_static = float(fraction)
    private.max_running_requests = placement.max_concurrent_branches
    # A tokenizer path that still pointed at the target would silently pair
    # the draft weights with the wrong vocabulary.
    if hasattr(private, "tokenizer_path"):
        private.tokenizer_path = model_path
    private.revision = getattr(server_args, "pvd_draft_revision", None)
    # SGLang's ModelRunner expects a platform type here ("cuda"), while
    # TpModelWorker selects the actual GPU with gpu_id.  Passing "cuda:0"
    # through makes get_available_gpu_memory() reject the device at startup.
    # Keep the indexed PVD request as an assertion about that separate gpu_id.
    import torch

    requested_device = getattr(server_args, "pvd_draft_device", None)
    if requested_device is not None:
        try:
            parsed_device = torch.device(requested_device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise DraftWorkerError("invalid --pvd-draft-device") from exc
        if parsed_device.type != "cuda" or parsed_device.index != placement.gpu_id:
            raise DraftWorkerError(
                "--pvd-draft-device must be an indexed CUDA device matching "
                f"the draft worker gpu_id ({placement.gpu_id})"
            )
    private.device = "cuda"
    # Belt and braces: the copy is what the draft worker is built from, and it
    # must not carry a speculative configuration into that construction. The
    # target's own configuration is a different object and is not touched.
    private.speculative_algorithm = None
    for name in (
        "speculative_num_steps",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
        "speculative_draft_model_path",
        "speculative_draft_model_revision",
        "speculative_draft_model_quantization",
        "speculative_draft_load_format",
        "speculative_token_map",
    ):
        if hasattr(private, name):
            setattr(private, name, None)
    # ModelRunner treats every value except the literal "null" as an active
    # disaggregation mode when deciding whether to initialise Mooncake.  None
    # is *not* disabled there; it can try to create a second native engine.
    if hasattr(private, "disaggregation_mode"):
        private.disaggregation_mode = "null"
    if hasattr(private, "disaggregation_topology"):
        private.disaggregation_topology = "pd"
    return private


# --------------------------------------------------------------------------
# The provider
# --------------------------------------------------------------------------


class SGLangDraftProvider(DraftProvider):
    """``DraftProvider`` backed by an SGLang-loaded draft model.

    Prediction-only: it produces candidate token ids and the identity of the
    snapshot they came from. It has no way to return them as output, and the
    worker's generation surface is not reachable from here.

    ``predict`` must be called inside ``branch()``, on the same thread. That
    is the calling contract, and it is enforced rather than documented: a
    direct call has no handle to execute against, no reservation and no
    cleanup, so it is refused.
    """

    def __init__(
        self,
        config: DraftConfig,
        placement: DraftPlacement,
        factory: DraftRunnerFactory,
        *,
        worker: Optional[Any] = None,
        target_worker: Optional[Any] = None,
        draft_vocabulary: Optional[VocabularySignature] = None,
        target_vocabulary: Optional[VocabularySignature] = None,
        scratch_budget: Optional[TransferBudget] = None,
        persistent_budget: Optional[TransferBudget] = None,
    ) -> None:
        if not isinstance(config, DraftConfig):
            raise PredictionConfigError("a DraftConfig is required")
        if not isinstance(placement, DraftPlacement):
            raise PredictionConfigError("a DraftPlacement is required")
        self.config = config
        self.placement = placement
        self.factory = factory
        self.capabilities = factory.capabilities()
        if not isinstance(self.capabilities, DraftCapabilities):
            raise DraftCapabilityError(
                "a runner factory must declare DraftCapabilities"
            )
        if config.predict_tokens > self.capabilities.max_predict_tokens:
            raise DraftCapabilityError(
                f"predict_tokens={config.predict_tokens} exceeds what this "
                f"runner supports ({self.capabilities.max_predict_tokens})"
            )
        self.worker = None if worker is None else DraftWorkerInterface(worker)
        self.pool_ownership: Optional[PoolOwnership] = None
        if worker is not None and target_worker is not None:
            self.pool_ownership = require_private_pools(worker, target_worker)
        self.vocabulary = self._check_vocabularies(draft_vocabulary, target_vocabulary)
        self.scratch_budget = scratch_budget or TransferBudget(
            staging_bytes=placement.scratch_budget_bytes, max_inflight=1
        )
        # Weights and pools are charged once, against their own budget.
        persistent_bytes = factory.persistent_bytes()
        if type(persistent_bytes) is not int or persistent_bytes < 0:
            raise PredictionConfigError(
                "factory persistent bytes must be a non-negative integer"
            )
        self.persistent_budget = persistent_budget
        if placement.persistent_budget_bytes:
            self.persistent_budget = self.persistent_budget or TransferBudget(
                staging_bytes=placement.persistent_budget_bytes, max_inflight=1
            )
        if self.persistent_budget is self.scratch_budget:
            raise PredictionConfigError(
                "persistent and scratch budgets must be separate"
            )
        if persistent_bytes and self.persistent_budget is None:
            raise PredictionConfigError(
                "positive persistent bytes require an explicit persistent budget"
            )
        if (
            placement.persistent_budget_bytes
            and persistent_bytes > placement.persistent_budget_bytes
        ):
            raise TransferCapacityError(
                "draft exceeds its per-provider persistent budget"
            )
        # TransferBudget.reserve is idempotent by owner. A process-global name
        # silently undercharges independent providers sharing a global budget.
        self._persistent_owner = f"pvd-draft:persistent:{uuid.uuid4().hex}"
        if self.persistent_budget is not None:
            self.persistent_budget.reserve(self._persistent_owner, persistent_bytes, 0)
        self._lock = threading.Lock()
        # Owning resources separately is not the same as being safe to run
        # concurrently: ModelRunner and its attention backend carry
        # per-forward state. Execution is serialized until that is proven.
        self._execution_lock = threading.Lock()
        self._active: Dict[str, _Branch] = {}
        self._quarantined: List[QuarantinedBranch] = []
        # Keep actual handles/pools alive, not only a textual diagnostic.
        self._retained: Dict[str, _Branch] = {}
        self._current = threading.local()

    @staticmethod
    def _check_vocabularies(
        draft: Optional[VocabularySignature], target: Optional[VocabularySignature]
    ) -> Optional[VocabularySignature]:
        """Token ids only travel between models that agree on what they mean.

        A vocabulary *size* is not compatibility: two tokenizers of the same
        size can encode the same text differently, and the predicted ids are
        handed to the target model's probe. So the full signature is compared
        -- size, special ids and the encoded-probe fingerprint.
        """
        if draft is None and target is None:
            return None
        if draft is None or target is None:
            raise PredictionConfigError(
                "supply both the draft and target vocabulary signatures, or "
                "neither; a one-sided check establishes nothing"
            )
        for name in (
            "size",
            "bos_token_id",
            "eos_token_id",
            "fingerprint",
            "mapping_fingerprint",
        ):
            mine, theirs = getattr(draft, name), getattr(target, name)
            if mine != theirs:
                raise PredictionConfigError(
                    f"draft and target tokenizers disagree on {name} "
                    f"({mine!r} vs {theirs!r}); predicted ids would mean "
                    "different text to the probe than to the draft model"
                )
        if draft.allowed_ids != target.allowed_ids:
            raise PredictionConfigError(
                "draft and target tokenizers disagree on allowed token IDs; "
                "predicted ids could name different or padded tokens"
            )
        return draft

    # -- description --------------------------------------------------------

    def describe(self) -> Dict[str, str]:
        described = dict(self.config.resolved_source())
        described.update(
            {
                "provider": "sglang-standalone",
                "device": self.config.device,
                "gpu_id": str(self.placement.gpu_id),
                "tp_rank": str(self.placement.tp_rank),
                "pools": (
                    "unchecked"
                    if self.pool_ownership is None
                    else self.pool_ownership.describe()
                ),
                "worker_surface": "allowlist",
                "prefix": "recomputed-per-call",
                "execution": "serialized",
                "tokenizer": (
                    "unchecked" if self.vocabulary is None else "signature-matched"
                ),
                "degraded": str(bool(self._quarantined)),
            }
        )
        return described

    @property
    def active_branches(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def quarantined(self) -> Tuple[QuarantinedBranch, ...]:
        with self._lock:
            return tuple(self._quarantined)

    @property
    def degraded(self) -> bool:
        return bool(self.quarantined)

    def _capacity(self) -> int:
        """Admission slots still issuable: quarantined ones never come back."""
        return self.placement.max_concurrent_branches - len(self._quarantined)

    # -- isolation scope ----------------------------------------------------

    @contextmanager
    def branch(self):
        """Admit one branch, mint its handle, and clean up in the right order.

        Admission and the byte reservation are both taken before the handle
        allocates, and both are given back only after it has been released.
        A branch whose release fails keeps them: its resources may still be
        live, and reissuing them would hand the same memory to a second owner.
        """
        branch_id = uuid.uuid4().hex[:12]
        owner = f"pvd-draft:{branch_id}"
        with self._lock:
            if self._quarantined:
                raise TransferCapacityError("shared draft worker is quarantined")
            if getattr(self._current, "branch", None) is not None:
                raise DraftLifecycleError(
                    "this thread already has an open prediction branch; "
                    "branches do not nest"
                )
            if len(self._active) >= self._capacity():
                raise TransferCapacityError(
                    f"{len(self._active)} prediction branches already running "
                    f"and {len(self._quarantined)} quarantined; the limit is "
                    f"{self.placement.max_concurrent_branches}"
                )
            # Reserve the slot before anything is allocated against it.
            self._active[branch_id] = _Branch(branch_id, None, owner, 0)
        handle = None
        reserved = False
        wanted = 0
        try:
            # Sized for the worst case this runner admits, because the
            # branch is opened before any prefix is seen and a reservation
            # made after the allocation is not a reservation.
            handle = self.factory.open(
                branch_id=branch_id,
                prefix_tokens=self.capabilities.max_prefix_tokens,
                max_tokens=self.config.predict_tokens,
            )
            wanted = handle.scratch_bytes()
            if type(wanted) is not int or wanted < 0:
                raise PredictionConfigError(
                    "a handle must declare scratch bytes as a non-negative integer"
                )
            self.scratch_budget.reserve(owner, wanted, 0)
            reserved = True
            with self._lock:
                if self._quarantined:
                    raise TransferCapacityError("shared draft worker is quarantined")
                record = self._active[branch_id]
                record.handle = handle
                record.reserved_bytes = wanted
                self._current.branch = branch_id
        except BaseException as admission_error:
            # open() may already have minted branch-owned metadata. Its
            # release must precede returning the admission slot, even when
            # scratch sizing or reservation itself failed. Shared pool/backend
            # cleanup is serialized with forwards and normal retirement.
            if handle is not None:
                with self._execution_lock:
                    try:
                        handle.release()
                    except BaseException as cleanup_error:
                        with self._lock:
                            record = self._active.pop(branch_id)
                            record.handle = handle
                            record.reserved_bytes = wanted if reserved else 0
                            self._retained[branch_id] = record
                            self._quarantined.append(
                                QuarantinedBranch(
                                    branch_id=branch_id,
                                    owner=owner,
                                    reserved_bytes=record.reserved_bytes,
                                    reason=(
                                        f"admission: {type(admission_error).__name__}: "
                                        f"{admission_error}; cleanup: "
                                        f"{type(cleanup_error).__name__}: "
                                        f"{cleanup_error}"
                                    ),
                                )
                            )
                        raise DraftWorkerError(
                            "draft branch admission failed and its opened "
                            "handle could not be released; provider quarantined"
                        ) from cleanup_error
            with self._lock:
                self._active.pop(branch_id, None)
            if reserved:
                self.scratch_budget.release(owner)
            raise
        try:
            yield self
        finally:
            self._current.branch = None
            self._retire(branch_id)

    def _retire(self, branch_id: str) -> None:
        """Release the handle first; only then give the accounting back."""
        with self._execution_lock:
            self._retire_locked(branch_id)

    def _retire_locked(self, branch_id: str) -> None:
        """Caller owns the shared execution lock through quarantine/accounting."""
        with self._lock:
            record = self._active.get(branch_id)
        if record is None:  # pragma: no cover - defensive
            return
        try:
            # Allocation/forward and cleanup share pool metadata and backend
            # state. Distinct request handles do not make alloc/free reentrant.
            if self._quarantined:
                raise DraftWorkerError("shared draft worker is quarantined")
            record.handle.release()
        except BaseException as exc:
            # The resources may still be live. Keep the reservation and the
            # admission slot so nothing else is handed the same memory.
            with self._lock:
                self._active.pop(branch_id, None)
                self._retained[branch_id] = record
                self._quarantined.append(
                    QuarantinedBranch(
                        branch_id=branch_id,
                        owner=record.owner,
                        reserved_bytes=record.reserved_bytes,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )
            raise
        with self._lock:
            self._active.pop(branch_id, None)
        self.scratch_budget.release(record.owner)

    def _require_branch(self) -> _Branch:
        branch_id = getattr(self._current, "branch", None)
        if branch_id is None:
            raise DraftLifecycleError(
                "predict() must be called inside branch(); a direct call has "
                "no execution handle, no reservation and no cleanup"
            )
        with self._lock:
            record = self._active.get(branch_id)
        if record is None or record.handle is None:  # pragma: no cover
            raise DraftLifecycleError("this branch is no longer open")
        return record

    # -- prediction ---------------------------------------------------------

    def predict(self, prefix: CommittedPrefix, max_tokens: int) -> DraftPrediction:
        record = self._require_branch()
        if not isinstance(prefix, CommittedPrefix):
            raise PredictionConfigError("prediction requires a committed snapshot")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise PredictionConfigError("max_tokens must be an integer")
        if max_tokens <= 0:
            raise PredictionConfigError("max_tokens must be positive")
        if not prefix.tokens:
            raise PredictionConfigError("an empty prefix predicts nothing")
        self._check_tokens(prefix.tokens, "prefix")
        budgeted = min(max_tokens, self.config.predict_tokens)
        # Refused before execution, not discovered during a forward. The
        # model and backend were checked by the factory against what the
        # loaded model reports; this checks the request shape.
        self.capabilities.require_shape(
            prefix_tokens=len(prefix.tokens), predict_tokens=budgeted
        )
        # Serialized: the handles are separate, the model runner is not.
        with self._execution_lock:
            try:
                if self._quarantined:
                    raise DraftWorkerError("shared draft worker is quarantined")
                prepared = record.handle.prepare_prefix(prefix.tokens)
                produced = record.handle.generate(prepared, budgeted)
            except BaseException:
                # Do not open a gap between a failed launch and its fence:
                # another admitted branch uses the same model/pool/backend.
                self._retire_locked(record.branch_id)
                raise
        tokens = self._validate(produced, budgeted)
        return DraftPrediction(
            request_id=prefix.request_id,
            prefix_version=prefix.version,
            tokens=tokens,
            source=self.describe(),
        )

    def _check_tokens(self, tokens: Sequence[int], what: str) -> None:
        if self.vocabulary is None:
            return
        for token in tokens:
            if not self.vocabulary.contains(token):
                raise PredictionConfigError(
                    f"{what} token {token} is outside the shared tokenizer's "
                    "declared token IDs"
                )

    def _validate(self, produced: Any, budgeted: int) -> Tuple[int, ...]:
        if isinstance(produced, (str, bytes)) or not isinstance(
            produced, (list, tuple)
        ):
            raise PredictionConfigError("a handle must return a sequence of token ids")
        tokens = tuple(produced)
        if not tokens:
            raise PredictionConfigError("a prediction must carry at least one token")
        if any(isinstance(t, bool) or not isinstance(t, int) for t in tokens):
            raise PredictionConfigError("predicted tokens must be integers")
        # Enforced on what came back, not only on what was asked for: a runner
        # that ignores the budget must not be able to set it.
        if len(tokens) > budgeted:
            tokens = tokens[:budgeted]
        self._check_tokens(tokens, "predicted")
        return tokens


def build_prediction_only_worker(
    server_args: Any,
    *,
    placement: DraftPlacement,
    nccl_port: int,
    target_worker: Any,
    worker_factory: Optional[Any] = None,
):
    """Construct an SGLang draft worker that owns its pools.

    Reuses ``TpModelWorker`` exactly as ``StandaloneWorker`` does, with two
    deliberate differences: it is built from a **private configuration copy**
    carrying the ``--pvd-draft-*`` settings, and the pools are not taken from
    the target, so the prediction path has no handle on committed request
    state. The result is wrapped in the allowlist interface.

    ``worker_factory`` exists so the construction *arguments* can be tested
    without loading weights. The default loads a model and needs a device.
    """
    if getattr(server_args, "speculative_algorithm", None) is not None:
        raise DraftWorkerError(
            "PVD prediction does not run under a speculative algorithm; the "
            "startup prohibition stays in force and this path is configured "
            "through the --pvd-draft-* flags instead"
        )
    draft_args = build_draft_server_args(server_args, placement)

    # build_draft_server_args already checked the requested indexed placement.
    # ModelRunner receives the platform type; TpModelWorker receives gpu_id.

    if worker_factory is None:  # pragma: no cover - loads weights

        def worker_factory(**kwargs):
            from sglang.srt.managers.tp_worker import TpModelWorker

            return TpModelWorker(**kwargs)

    worker = worker_factory(
        server_args=draft_args,
        gpu_id=placement.gpu_id,
        tp_rank=placement.tp_rank,
        pp_rank=0,
        dp_rank=None,
        moe_ep_rank=0,
        attn_cp_rank=0,
        moe_dp_rank=0,
        nccl_port=nccl_port,
        is_draft_worker=True,
        # The whole point: private pools, so ModelRunner allocates its own
        # req_to_token map and KV allocator instead of borrowing the target's.
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    )
    ownership = require_private_pools(worker, target_worker)
    return DraftWorkerInterface(worker), ownership
