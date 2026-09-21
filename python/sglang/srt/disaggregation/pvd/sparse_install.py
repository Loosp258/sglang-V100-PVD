"""Request-local rank agreement, plus an IN-PROCESS CPU contract driver.

Not a collective, wire transport, crash-recovery protocol, or GPU fence. The
owner thread consumes notifications; callbacks must enqueue, never mutate here.
Prepared/parked/applied receipts are assertions from trusted participants, not
remote-memory evidence. No resource is freed by the logical coordinator.
"""

import threading
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from enum import Enum

from sglang.srt.disaggregation.pvd.prefetch import PrefetchClock
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError


class InstallProtocolError(ValueError):
    pass


class InstallState(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    INSTALLING = "installing"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class InstallEpoch:
    request_id: str
    incarnation: str
    entry_transfer_id: str
    operation_id: str
    round: int
    target_tokens: int  # committed D tokens, excluding P's first token

    def __post_init__(self):
        if any(
            not isinstance(v, str) or not v.strip()
            for v in (
                self.request_id,
                self.incarnation,
                self.entry_transfer_id,
                self.operation_id,
            )
        ) or any(type(n) is not int or n < 0 for n in (self.round, self.target_tokens)):
            raise InstallProtocolError("invalid installation epoch identity/counts")


@dataclass(frozen=True)
class RankInstallReceipt:
    epoch: InstallEpoch
    rank: int
    staging_id: str
    layout_fingerprint: str

    def __post_init__(self):
        if (
            not isinstance(self.epoch, InstallEpoch)
            or type(self.rank) is not int
            or self.rank < 0
            or any(
                not isinstance(v, str) or not v.strip()
                for v in (self.staging_id, self.layout_fingerprint)
            )
        ):
            raise InstallProtocolError("invalid rank receipt")


class RankInstallCoordinator:
    """An install decision is not permission to decode; ALL applied ACKs are.

    One in-flight round; bounded metadata retains only the last completion.
    Failure/cancellation is terminal for this request incarnation. No automatic
    retry, failover, rollback or membership changes. Missing ranks cause a wait;
    a caller timeout must explicitly fail/close the request, not release memory.
    """

    def __init__(
        self,
        request_id,
        incarnation,
        entry_transfer_id,
        *,
        rank_layouts,
        interval,
        lead_tokens,
    ):
        self.identity = (request_id, incarnation, entry_transfer_id)
        if any(not isinstance(v, str) or not v.strip() for v in self.identity):
            raise InstallProtocolError("explicit request/incarnation/Entry required")
        self._layouts = dict(rank_layouts)
        if (
            not self._layouts
            or any(type(r) is not int or r < 0 for r in self._layouts)
            or any(
                not isinstance(v, str) or not v.strip() for v in self._layouts.values()
            )
        ):
            raise InstallProtocolError("explicit rank membership and layouts required")
        self._thread = threading.get_ident()
        self._clock = PrefetchClock(
            f"rank-install:{uuid.uuid4().hex}", interval, lead_tokens
        )
        self._state = InstallState.IDLE
        self._epoch = self._timing = self._completed = None
        self._prepared, self._parked, self._applied = {}, set(), set()
        self._last_receipts = {}
        self._reason = None

    def _owner(self):
        if threading.get_ident() != self._thread:
            raise InstallProtocolError("rank coordinator must run on its owner thread")

    def _live(self):
        self._owner()
        if self._state in (InstallState.FAILED, InstallState.CANCELLED):
            raise InstallProtocolError("request install coordinator is terminal")

    def _match(self, epoch):
        self._live()
        if not isinstance(epoch, InstallEpoch) or epoch != self._epoch:
            raise InstallProtocolError("stale or foreign installation epoch")

    def _receipt(self, receipt):
        if not isinstance(receipt, RankInstallReceipt):
            raise InstallProtocolError("explicit rank receipt required")
        self._match(receipt.epoch)
        if (
            type(receipt.rank) is not int
            or receipt.rank not in self._layouts
            or self._layouts[receipt.rank] != receipt.layout_fingerprint
            or not isinstance(receipt.staging_id, str)
            or not receipt.staging_id.strip()
        ):
            raise InstallProtocolError("rank membership/layout/staging mismatch")

    def begin(self, decode_tokens):
        self._live()
        if self._state != InstallState.IDLE:
            raise InstallProtocolError("an install round is already pending")
        timing = self._clock.begin(decode_tokens)
        epoch = InstallEpoch(
            *self.identity, uuid.uuid4().hex, timing.round, timing.target_tokens
        )
        self._epoch, self._timing = epoch, timing
        self._prepared, self._parked, self._applied = {}, set(), set()
        self._state = InstallState.PREPARING
        return epoch

    def prepared(self, receipt):
        self._receipt(receipt)
        old = self._prepared.get(receipt.rank)
        if old is not None:
            if old != receipt:
                raise InstallProtocolError("conflicting rank preparation")
            return  # matching duplicate, including delayed delivery while installing
        if self._state != InstallState.PREPARING:
            raise InstallProtocolError("preparation after install decision")
        self._prepared[receipt.rank] = receipt

    def parked(self, receipt, decode_tokens):
        """Participant asserts it reached the boundary and drained old readers."""
        self._receipt(receipt)
        if self._prepared.get(receipt.rank) != receipt:
            raise InstallProtocolError("rank is not prepared for this staged bank")
        if (
            type(decode_tokens) is not int
            or decode_tokens != receipt.epoch.target_tokens
        ):
            raise InstallProtocolError("rank must park at the exact boundary")
        self._clock.requires_install(decode_tokens)
        self._parked.add(receipt.rank)

    def decide_install(self, epoch):
        self._match(epoch)
        if self._state == InstallState.INSTALLING:
            return True
        if set(self._prepared) != set(self._layouts) or self._parked != set(
            self._layouts
        ):
            return False
        self._clock.mark_ready(self._timing)
        self._state = InstallState.INSTALLING
        return True

    def require_decision(self, epoch):
        self._match(epoch)
        if self._state != InstallState.INSTALLING:
            raise InstallProtocolError("installation has not been decided")

    def applied(self, receipt):
        self._live()
        # Only the most recently completed round is retained for duplicate ACKs.
        if (
            self._state == InstallState.IDLE
            and isinstance(receipt, RankInstallReceipt)
            and receipt.epoch == self._completed
            and self._last_receipts.get(receipt.rank) == receipt
        ):
            return
        self._receipt(receipt)
        self.require_decision(receipt.epoch)
        if self._prepared.get(receipt.rank) != receipt:
            raise InstallProtocolError("applied ACK differs from prepared bank")
        self._applied.add(receipt.rank)
        if self._applied != set(self._layouts):
            return
        self._clock.commit_install(self._timing, receipt.epoch.target_tokens)
        self._completed, self._last_receipts = self._epoch, dict(self._prepared)
        self._epoch = self._timing = None
        self._state = InstallState.IDLE

    def can_decode(self, decode_tokens):
        self._owner()
        if self._state in (InstallState.FAILED, InstallState.CANCELLED):
            return False
        due = self._clock.requires_install(decode_tokens)
        return self._state != InstallState.INSTALLING and not due

    def fail(self, epoch, reason):
        self._match(epoch)
        self._terminate(InstallState.FAILED, reason)

    def cancel(self, reason="request cancelled"):
        self._owner()
        if self._state in (InstallState.FAILED, InstallState.CANCELLED):
            return
        self._terminate(InstallState.CANCELLED, reason)

    def _terminate(self, state, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise InstallProtocolError("explicit failure/cancellation reason required")
        self._clock.close()
        self._state, self._reason = state, reason

    def snapshot(self):
        self._owner()
        return {
            "state": self._state.value,
            "epoch": self._epoch,
            "completed": self._completed,
            "expected_ranks": tuple(sorted(self._layouts)),
            "prepared": tuple(sorted(self._prepared)),
            "parked": tuple(sorted(self._parked)),
            "applied": tuple(sorted(self._applied)),
            "reason": self._reason,
            "installed_tokens": self._clock.installed_tokens,
            "next_boundary": self._clock.boundary,
            "round": self._clock.round,
            "interval": self._clock.interval,
            "lead_tokens": self._clock.lead_tokens,
        }


class CPUInstallGroup:
    """Single-process simulation owning all banks; NOT an actual TP runtime.

    All reads MUST go through read(), not raw bank.read()/the model consumer.
    External owners must not stage, install or mutate these banks. This driver
    uses CPU reader scopes as local evidence, never as a remote/GPU fence.
    Partial install failure closes the request; no rollback is attempted.
    """

    def __init__(self, banks, *, interval, lead_tokens):
        from sglang.srt.disaggregation.pvd.sparse_working_set import CPUSparseWorkingSet

        self._banks = dict(banks)
        if not self._banks or any(
            not isinstance(b, CPUSparseWorkingSet) for b in self._banks.values()
        ):
            raise InstallProtocolError("explicit CPU rank banks required")
        identities = {b.identity[:3] for b in self._banks.values()}
        if len(identities) != 1 or len({id(b) for b in self._banks.values()}) != len(
            self._banks
        ):
            raise InstallProtocolError(
                "rank banks must be distinct and share request identity"
            )
        self.coordinator = RankInstallCoordinator(
            *next(iter(identities)),
            rank_layouts={r: b.identity[3] for r, b in self._banks.items()},
            interval=interval,
            lead_tokens=lead_tokens,
        )
        self._candidates, self._receipts = {}, {}

    def begin(self, decode_tokens):
        epoch = self.coordinator.begin(decode_tokens)
        self._candidates.clear()
        self._receipts.clear()
        return epoch

    def describe_banks(self):
        """Immutable metadata copies, no tensors or mutable bank access."""
        self.coordinator._owner()
        return {
            rank: {
                "identity": bank.identity,
                "groups": bank.expected_groups,
                "prompt_tokens": bank.prompt_tokens,
                "head_dim": bank.head_dim,
            }
            for rank, bank in self._banks.items()
        }

    def can_decode(self, decode_tokens):
        return self.coordinator.can_decode(decode_tokens)

    def cancel(self, reason="request cancelled"):
        self.coordinator.cancel(reason)

    def stage(self, epoch, rank, payloads):
        self.coordinator._match(epoch)
        if type(rank) is not int or rank not in self._banks or rank in self._receipts:
            raise InstallProtocolError("unknown rank or bank already staged")
        payloads = tuple(payloads)
        for payload in payloads:
            s = payload.spec
            if (
                s.request_id,
                s.incarnation,
                s.entry_transfer_id,
                s.operation_id,
                s.target_tokens,
            ) != (
                epoch.request_id,
                epoch.incarnation,
                epoch.entry_transfer_id,
                epoch.operation_id,
                epoch.target_tokens,
            ):
                raise InstallProtocolError("payload is not for the active epoch")
        bank = self._banks[rank]
        bank.stage(payloads)
        candidate = bank.install_candidate()
        receipt = RankInstallReceipt(
            epoch, rank, candidate.staging_id, bank.identity[3]
        )
        self.coordinator.prepared(receipt)
        self._candidates[rank], self._receipts[rank] = candidate, receipt
        return receipt

    def try_install(self, epoch, rank_counts):
        self.coordinator._match(epoch)
        if set(rank_counts) != set(self._banks) or any(
            type(r) is not int or type(n) is not int or n != epoch.target_tokens
            for r, n in rank_counts.items()
        ):
            raise InstallProtocolError("all ranks must report the exact same boundary")
        if set(self._receipts) != set(self._banks):
            return False
        try:
            for rank, bank in self._banks.items():
                if not bank.can_install(self._candidates[rank], rank_counts[rank]):
                    return False  # old forward still owns a bank; nothing installed
            for receipt in self._receipts.values():
                self.coordinator.parked(receipt, epoch.target_tokens)
            if not self.coordinator.decide_install(epoch):
                return False
            for rank in sorted(self._banks):
                self.coordinator.require_decision(epoch)
                self._banks[rank].install(
                    epoch.target_tokens, candidate=self._candidates[rank]
                )
                self.coordinator.applied(self._receipts[rank])
            return True
        except BaseException as exc:
            # Installation may have happened on some ranks. Never reopen reads.
            state = self.coordinator.snapshot()
            if state["state"] not in (
                InstallState.FAILED.value,
                InstallState.CANCELLED.value,
            ):
                if state["epoch"] == epoch:
                    self.coordinator.fail(epoch, f"CPU rank install failed: {exc}")
                else:
                    self.coordinator.cancel(
                        f"CPU install exception after completion: {exc}"
                    )
            raise

    @contextmanager
    def read(self, rank, decode_tokens):
        if type(rank) is not int or rank not in self._banks:
            raise InstallProtocolError("unknown rank")
        if not self.coordinator.can_decode(decode_tokens):
            raise InstallProtocolError(
                "request must wait or abort; rank install incomplete"
            )
        with self._banks[rank].read() as groups:
            yield groups

    def installation_complete(self, receipt):
        """Exact last all-rank completion, not merely a staged local bank.

        The consumer must observe this before starting another round. A wire
        receiver may latch this fact to retry an ACK whose response was lost.
        """
        state = self.coordinator.snapshot()
        return (
            isinstance(receipt, RankInstallReceipt)
            and state["state"] == InstallState.IDLE.value
            and state["completed"] == receipt.epoch
            and self._receipts.get(receipt.rank) == receipt
        )

    def close(self):
        self.coordinator.cancel()
        errors = []
        for bank in self._banks.values():
            try:
                bank.close()
            except SparsePayloadError as exc:
                errors.append(exc)
        if errors:
            raise InstallProtocolError(
                "CPU readers must drain before closing rank banks"
            ) from errors[0]


class CPUInstalledPromptView:
    """Read-only projection of all local shards into a TP1 CPU model forward.

    NOT a TP gather: no copy/transport, all shards already live in this process.
    Every bank read goes through the group's gate and remains held until the
    whole forward ends. The consumer checks complete layer/head coverage.
    """

    def __init__(self, group, decode_tokens):
        if not isinstance(group, CPUInstallGroup):
            raise InstallProtocolError("an explicit CPU install group is required")
        if type(decode_tokens) is not int or decode_tokens < 0:
            raise InstallProtocolError("explicit committed D-token count required")
        metadata = group.describe_banks()
        first = next(iter(metadata.values()))
        self.identity = first["identity"]
        self.prompt_tokens, self.head_dim = first["prompt_tokens"], first["head_dim"]
        groups = set()
        for item in metadata.values():
            if (
                item["identity"] != self.identity
                or item["prompt_tokens"] != self.prompt_tokens
                or item["head_dim"] != self.head_dim
                or groups.intersection(item["groups"])
            ):
                raise InstallProtocolError(
                    "shard metadata differ or layer/head ownership overlaps"
                )
            groups.update(item["groups"])
        self.expected_groups = frozenset(groups)
        self.decode_tokens = decode_tokens
        self._group, self._ranks = group, tuple(sorted(metadata))

    @contextmanager
    def read(self):
        merged, generations = {}, set()
        with ExitStack() as stack:
            for rank in self._ranks:
                groups = stack.enter_context(self._group.read(rank, self.decode_tokens))
                for key, (spec, tensor) in groups.items():
                    if key in merged:
                        raise InstallProtocolError(
                            "duplicate layer/head in installed shards"
                        )
                    generations.add((spec.operation_id, spec.target_tokens))
                    merged[key] = (spec, tensor)
            if set(merged) != self.expected_groups or len(generations) != 1:
                raise InstallProtocolError("incomplete or mixed-generation Prompt view")
            if (
                next(iter(generations))[1]
                != self._group.coordinator.snapshot()["installed_tokens"]
            ):
                raise InstallProtocolError(
                    "Prompt generation is not the coordinator's installed generation"
                )
            yield merged
