"""Bounded control messages for request-local rank installation.

No tensors, addresses, transport completion or resource release cross this
interface. Peer epochs must come from a trusted, incarnation-bound channel;
echoing a caller-provided epoch is NOT authentication. Call on the coordinator
owner thread. This is not a collective transport or a production TP integration.
"""

import json
from dataclasses import asdict, dataclass
from types import MappingProxyType

from sglang.srt.disaggregation.pvd.sparse_install import (
    InstallEpoch,
    InstallProtocolError,
    InstallState,
    RankInstallCoordinator,
    RankInstallReceipt,
)

PROTOCOL = "pvd-rank-install-v1"
MAX_FRAME_BYTES = 16384
MAX_TEXT_BYTES = 1024
MAX_COUNT = (1 << 63) - 1
_EPOCH_FIELDS = frozenset(
    (
        "request_id",
        "incarnation",
        "entry_transfer_id",
        "operation_id",
        "round",
        "target_tokens",
    )
)
_RECEIPT_FIELDS = frozenset(("epoch", "rank", "staging_id", "layout_fingerprint"))
_MESSAGE_FIELDS = frozenset(
    ("protocol", "kind", "peer_epoch", "receipt", "decode_tokens", "reason")
)
_KINDS = frozenset(
    ("prepared", "parked", "applied", "install", "resume", "resumed", "failed")
)


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise InstallProtocolError("bounded nonempty text required")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as exc:
        raise InstallProtocolError("invalid UTF-8 text") from exc
    if size > MAX_TEXT_BYTES:
        raise InstallProtocolError("text exceeds rank-control limit")


def _count(value):
    if type(value) is not int or not 0 <= value <= MAX_COUNT:
        raise InstallProtocolError("bounded nonnegative integer required")


def _fields(value, fields):
    if not isinstance(value, dict) or set(value) != fields:
        raise InstallProtocolError("rank-control fields do not match schema")


def _object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise InstallProtocolError("duplicate JSON field")
        result[name] = value
    return result


def _constant(_):
    raise InstallProtocolError("nonfinite JSON value")


@dataclass(frozen=True)
class RankInstallMessage:
    kind: str
    peer_epoch: str
    receipt: RankInstallReceipt
    decode_tokens: int | None = None
    reason: str | None = None

    def __post_init__(self):
        if not isinstance(self.kind, str) or self.kind not in _KINDS:
            raise InstallProtocolError("unknown rank-control kind")
        _text(self.peer_epoch)
        if not isinstance(self.receipt, RankInstallReceipt):
            raise InstallProtocolError("rank receipt required")
        r, e = self.receipt, self.receipt.epoch
        for value in (
            e.request_id,
            e.incarnation,
            e.entry_transfer_id,
            e.operation_id,
            r.staging_id,
            r.layout_fingerprint,
        ):
            _text(value)
        for value in (r.rank, e.round, e.target_tokens):
            _count(value)
        if self.kind == "parked":
            _count(self.decode_tokens)
            if self.decode_tokens != e.target_tokens:
                raise InstallProtocolError("parked count must equal target boundary")
        elif self.decode_tokens is not None:
            raise InstallProtocolError("decode_tokens is only valid for parked")
        if self.kind == "failed":
            _text(self.reason)
        elif self.reason is not None:
            raise InstallProtocolError("reason is only valid for failed")

    def encode(self):
        value = {"protocol": PROTOCOL, **asdict(self)}
        raw = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if len(raw) > MAX_FRAME_BYTES:
            raise InstallProtocolError("rank-control frame exceeds limit")
        return raw

    @classmethod
    def decode(cls, raw):
        if type(raw) is not bytes or not 0 < len(raw) <= MAX_FRAME_BYTES:
            raise InstallProtocolError("bounded bytes frame required")
        try:
            value = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=_constant
            )
        except (UnicodeError, ValueError, RecursionError) as exc:
            raise InstallProtocolError(f"invalid rank-control JSON: {exc}") from exc
        _fields(value, _MESSAGE_FIELDS)
        if value["protocol"] != PROTOCOL:
            raise InstallProtocolError("unsupported rank-control protocol")
        r = value["receipt"]
        _fields(r, _RECEIPT_FIELDS)
        _fields(r["epoch"], _EPOCH_FIELDS)
        receipt = RankInstallReceipt(
            InstallEpoch(**r["epoch"]),
            r["rank"],
            r["staging_id"],
            r["layout_fingerprint"],
        )
        return cls(
            value["kind"],
            value["peer_epoch"],
            receipt,
            value["decode_tokens"],
            value["reason"],
        )


class RankInstallExchange:
    """Owner-thread bridge from bound peer messages to the logical coordinator.

    It issues per-rank INSTALL only after every PREPARED/PARKED, and RESUME only
    after every APPLIED. Both commands carry the exact prepared staging identity.
    One round is retained. Transport loss must explicitly fail/cancel the request;
    it is never authority to free a GPU/MR or resume the remaining ranks.
    """

    def __init__(self, coordinator, *, peer_epochs):
        if not isinstance(coordinator, RankInstallCoordinator):
            raise InstallProtocolError("rank coordinator required")
        expected = coordinator.snapshot()["expected_ranks"]
        self.peer_epochs = MappingProxyType(dict(peer_epochs))
        if any(type(r) is not int for r in self.peer_epochs) or set(
            self.peer_epochs
        ) != set(expected):
            raise InstallProtocolError("peer membership must match coordinator")
        for epoch in self.peer_epochs.values():
            _text(epoch)
        self.coordinator = coordinator
        self._receipts = {}
        self._resume_epoch = None
        self._resume_receipts, self._resumed = {}, set()

    def begin(self, decode_tokens):
        completed = self.coordinator.snapshot()["completed"]
        if completed is not None and not self.resume_complete(completed):
            raise InstallProtocolError(
                "all ranks must acknowledge resume before next round"
            )
        epoch = self.coordinator.begin(decode_tokens)
        self._receipts.clear()
        return epoch

    def receive(self, raw, *, peer_rank):
        self.coordinator._owner()
        message = RankInstallMessage.decode(raw)
        receipt = message.receipt
        if (
            type(peer_rank) is not int
            or peer_rank != receipt.rank
            or self.peer_epochs.get(peer_rank) != message.peer_epoch
        ):
            raise InstallProtocolError("rank channel/incarnation mismatch")
        if message.kind == "resumed":
            state = self.coordinator.snapshot()
            if (
                state["state"]
                in (InstallState.FAILED.value, InstallState.CANCELLED.value)
                or self._resume_receipts.get(peer_rank) != receipt
            ):
                raise InstallProtocolError(
                    "resume ACK must name the issued resume command"
                )
            self._resumed.add(peer_rank)
        elif message.kind == "prepared":
            self.coordinator.prepared(receipt)
            self._receipts[peer_rank] = receipt
        elif message.kind == "parked":
            self.coordinator.parked(receipt, message.decode_tokens)
        elif message.kind == "applied":
            self.coordinator.applied(receipt)
        elif message.kind == "failed":
            self.coordinator._receipt(receipt)
            if self._receipts.get(peer_rank) != receipt:
                raise InstallProtocolError("failure must name the prepared bank")
            self.coordinator.fail(receipt.epoch, message.reason)
        else:
            raise InstallProtocolError("peer cannot send coordinator commands")

    def _commands(self, kind, reason=None):
        return {
            rank: RankInstallMessage(
                kind, self.peer_epochs[rank], receipt, reason=reason
            ).encode()
            for rank, receipt in self._receipts.items()
        }

    def install_commands(self, epoch):
        if not self.coordinator.decide_install(epoch):
            return {}
        return self._commands("install")

    def resume_commands(self, epoch):
        state = self.coordinator.snapshot()
        if (
            state["state"] != InstallState.IDLE.value
            or state["completed"] != epoch
            or set(self._receipts) != set(self.peer_epochs)
        ):
            raise InstallProtocolError("all ranks must apply before resume")
        commands = self._commands("resume")
        if self._resume_epoch != epoch:
            self._resume_epoch = epoch
            self._resume_receipts = dict(self._receipts)
            self._resumed.clear()
        return commands

    def resume_complete(self, epoch):
        state = self.coordinator.snapshot()
        return (
            state["state"]
            not in (InstallState.FAILED.value, InstallState.CANCELLED.value)
            and self._resume_epoch == epoch
            and self._resumed == set(self.peer_epochs)
        )

    def can_decode(self, decode_tokens):
        """Wire users must use this gate, not the logical coordinator directly.

        APPLIED proves local swaps; RESUMED acknowledges release of local read
        gates. Neither is native GPU/MR completion or a model forward receipt.
        """
        state = self.coordinator.snapshot()
        completed = state["completed"]
        return (
            completed is not None
            and self.resume_complete(completed)
            and self.coordinator.can_decode(decode_tokens)
        )

    def cancel_commands(self, reason):
        # Only known prepared receipts can be addressed. The transport owner
        # must also stop/close peers whose PREPARED reply was lost, not infer
        # that their absence here means they own no resources.
        _text(reason)
        self.coordinator.cancel(reason)
        return self._commands("failed", reason)
