"""Rank-local CPU participant for cross-process installation validation.

Each process owns only its bank. A local install ACK does NOT reopen reads:
an exact RESUME from the coordinator is required. Reader scopes are synchronous
CPU ownership, not CUDA events. No production backend/TP launcher registers this.
"""

import threading
from contextlib import contextmanager

from sglang.srt.disaggregation.pvd.rank_install_wire import RankInstallMessage
from sglang.srt.disaggregation.pvd.sparse_install import (
    InstallEpoch,
    InstallProtocolError,
    RankInstallReceipt,
)
from sglang.srt.disaggregation.pvd.sparse_working_set import CPUSparseWorkingSet


class CPURankInstallParticipant:
    def __init__(self, bank, *, rank, peer_epoch, interval):
        if (
            not isinstance(bank, CPUSparseWorkingSet)
            or type(rank) is not int
            or rank < 0
            or type(interval) is not int
            or interval <= 0
            or not isinstance(peer_epoch, str)
            or not peer_epoch.strip()
        ):
            raise InstallProtocolError(
                "explicit CPU bank/rank/incarnation/interval required"
            )
        self._bank, self.rank, self.peer_epoch, self.interval = (
            bank,
            rank,
            peer_epoch,
            interval,
        )
        self._thread = threading.get_ident()
        self._round, self._boundary = 0, 0
        self._pending = self._candidate = self._last = None
        self._phase, self._terminal = None, False

    def _owner(self):
        if threading.get_ident() != self._thread:
            raise InstallProtocolError("rank participant must run on its owner thread")

    def _live(self):
        self._owner()
        if self._terminal:
            raise InstallProtocolError("rank participant is terminal")

    def _message(self, kind, *, decode_tokens=None, reason=None):
        return RankInstallMessage(
            kind, self.peer_epoch, self._pending, decode_tokens, reason
        ).encode()

    def stage(self, epoch, payloads):
        self._live()
        if self._pending is not None:
            raise InstallProtocolError("rank already has a pending bank")
        if (
            not isinstance(epoch, InstallEpoch)
            or (epoch.request_id, epoch.incarnation, epoch.entry_transfer_id)
            != self._bank.identity[:3]
            or epoch.round != self._round
            or epoch.target_tokens != self._boundary
        ):
            raise InstallProtocolError("unexpected rank installation epoch")
        payloads = tuple(payloads)
        if any(
            (p.spec.operation_id, p.spec.target_tokens)
            != (epoch.operation_id, epoch.target_tokens)
            for p in payloads
        ):
            raise InstallProtocolError("payload does not match installation epoch")
        self._bank.stage(payloads)
        try:
            candidate = self._bank.install_candidate()
            receipt = RankInstallReceipt(
                epoch, self.rank, candidate.staging_id, self._bank.identity[3]
            )
            prepared = RankInstallMessage("prepared", self.peer_epoch, receipt).encode()
        except BaseException:
            self._bank.discard_next()
            raise
        self._pending, self._candidate, self._phase = receipt, candidate, "prepared"
        return prepared

    def park(self, decode_tokens):
        self._live()
        if self._phase not in ("prepared", "parked"):
            raise InstallProtocolError("rank has no prepared bank to park")
        if not self._bank.can_install(self._candidate, decode_tokens):
            return None  # old CPU forward still owns the bank; no parked receipt
        self._phase = "parked"
        return self._message("parked", decode_tokens=decode_tokens)

    def command(self, raw):
        self._live()
        message = RankInstallMessage.decode(raw)
        if message.peer_epoch != self.peer_epoch or message.receipt.rank != self.rank:
            raise InstallProtocolError("rank channel/incarnation mismatch")
        if self._pending is None:
            if message.receipt == self._last and message.kind == "resume":
                return None  # retry only the last completion; never advance twice
            if message.receipt == self._last and message.kind == "failed":
                self._terminal = True
                return None
            raise InstallProtocolError("no matching pending rank installation")
        if message.receipt != self._pending:
            raise InstallProtocolError("stale or foreign rank installation command")
        if message.kind == "failed":
            self._terminal = True  # no bank release: readers still have ownership
            return None
        if message.kind == "install":
            if self._phase == "applied":
                return self._message("applied")  # duplicate command, no second swap
            if self._phase != "parked":
                raise InstallProtocolError("rank must park before installing")
            try:
                self._bank.install(self._boundary, candidate=self._candidate)
            except BaseException:
                self._terminal = True  # a partial install must never resume reads
                raise
            self._phase = "applied"
            return self._message("applied")
        if message.kind == "resume":
            if self._phase != "applied":
                raise InstallProtocolError("rank must apply before resuming")
            self._last = self._pending
            self._pending = self._candidate = self._phase = None
            self._round += 1
            self._boundary += self.interval
            return None
        raise InstallProtocolError("rank cannot receive participant events as commands")

    @contextmanager
    def read(self, decode_tokens):
        self._live()
        if (
            self._last is None
            or self._phase in ("parked", "applied")
            or type(decode_tokens) is not int
            or not self._last.epoch.target_tokens <= decode_tokens < self._boundary
        ):
            raise InstallProtocolError("rank must wait for global resume at boundary")
        with self._bank.read() as groups:
            yield groups

    def close(self):
        self._owner()
        self._terminal = True
        self._bank.close()  # may refuse until existing CPU reader scopes drain

    def snapshot(self):
        self._owner()
        return {
            "rank": self.rank,
            "peer_epoch": self.peer_epoch,
            "round": self._round,
            "boundary": self._boundary,
            "phase": self._phase,
            "terminal": self._terminal,
        }
