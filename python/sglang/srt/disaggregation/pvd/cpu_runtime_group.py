"""In-process CPU bank binding for the wire runtime; NOT distributed model TP.

One coordinator, the exact same banks, encoded messages and real reader scopes.
Only the runtime drives installation. Local queues are pumped explicitly; this
is not a network transport, a production Scheduler or a CUDA completion fence.
"""

import time
from collections import deque
from contextlib import contextmanager

from sglang.srt.disaggregation.pvd.cpu_rank_install import CPURankInstallParticipant
from sglang.srt.disaggregation.pvd.rank_install_runtime import RankInstallRuntime
from sglang.srt.disaggregation.pvd.rank_install_wire import (
    RankInstallExchange,
    RankInstallMessage,
)
from sglang.srt.disaggregation.pvd.sparse_install import (
    CPUInstallGroup,
    InstallProtocolError,
)


class CPURuntimeInstallGroup(CPUInstallGroup):
    def __init__(
        self,
        banks,
        *,
        interval,
        lead_tokens,
        peer_epochs,
        timeout_seconds,
        max_pending_events,
        max_pending_bytes,
        clock=time.monotonic,
    ):
        super().__init__(banks, interval=interval, lead_tokens=lead_tokens)
        self._commands, self._stops = deque(), set()
        self._timeout = timeout_seconds
        exchange = RankInstallExchange(self.coordinator, peer_epochs=peer_epochs)
        self._peers = {
            rank: CPURankInstallParticipant(
                bank,
                rank=rank,
                peer_epoch=exchange.peer_epochs[rank],
                interval=interval,
            )
            for rank, bank in self._banks.items()
        }
        self.runtime = RankInstallRuntime(
            exchange,
            send=self._send,
            stop_peer=self._stop,
            max_pending_events=max_pending_events,
            max_pending_bytes=max_pending_bytes,
            clock=clock,
        )

    def _send(self, rank, raw):
        # At most one command per rank per phase; progression must drain first.
        if len(self._commands) >= len(self._peers):
            raise InstallProtocolError("local rank command queue full")
        self._commands.append((rank, raw))

    def _stop(self, rank, identity, reason):
        if identity != self.coordinator.identity or rank not in self._peers:
            raise InstallProtocolError("foreign request stop")
        self._stops.add(rank)

    def _post(self, rank, raw):
        if not self.runtime.post(
            raw, peer_rank=rank, peer_epoch=self.runtime.exchange.peer_epochs[rank]
        ):
            self.runtime.progress()
            raise InstallProtocolError("local rank event refused")

    def progress(self):
        """Bounded local message turns; never sleep or infer native completion."""
        for _ in range(4):
            self.runtime.progress()
            for rank in tuple(self._stops):
                self._peers[rank].stop()
                self._stops.remove(rank)
            if self.runtime.snapshot()["phase"] == "failed":
                self._commands.clear()
                return
            if not self._commands:
                return
            while self._commands:
                rank, raw = self._commands.popleft()
                try:
                    self._post(rank, self._peers[rank].command(raw))
                except BaseException:
                    self.cancel("local rank installation failed")
                    raise

    def begin(self, decode_tokens):
        epoch = self.runtime.begin(decode_tokens, timeout_seconds=self._timeout)
        self._receipts.clear()
        return epoch

    def stage(self, epoch, rank, payloads):
        self.coordinator._match(epoch)
        if type(rank) is not int or rank not in self._peers or rank in self._receipts:
            raise InstallProtocolError("unknown rank or bank already staged")
        raw = self._peers[rank].stage(epoch, payloads)
        receipt = RankInstallMessage.decode(raw).receipt
        self._receipts[rank] = receipt
        self._post(rank, raw)
        self.progress()
        return receipt

    def try_install(self, epoch, rank_counts):
        if set(rank_counts) != set(self._peers) or any(
            type(r) is not int or type(n) is not int or n != epoch.target_tokens
            for r, n in rank_counts.items()
        ):
            raise InstallProtocolError("all ranks must report the exact same boundary")
        self.progress()
        if self.runtime.exchange.resume_complete(epoch):
            return True
        self.coordinator._match(epoch)
        if set(self._receipts) != set(self._peers):
            return False
        for rank, peer in self._peers.items():
            if peer.snapshot()["phase"] in ("prepared", "parked"):
                parked = peer.park(rank_counts[rank])
                if parked is not None:
                    self._post(rank, parked)
        self.progress()
        return self.runtime.exchange.resume_complete(epoch)

    def can_decode(self, decode_tokens):
        self.progress()
        return self.runtime.can_decode(decode_tokens)

    @contextmanager
    def read(self, rank, decode_tokens):
        self.coordinator._owner()
        permit = self.runtime._forward
        if (
            type(rank) is not int
            or rank not in self._peers
            or permit is None
            or permit.committed_tokens != decode_tokens
            or permit.identity != self.coordinator.identity
            or permit.installed_epoch != self.coordinator.snapshot()["completed"]
            or not self.runtime._guard()
        ):
            raise InstallProtocolError(
                "exact runtime forward permit required for bank read"
            )
        with self._peers[rank].read(decode_tokens) as groups:
            if any(
                (spec.operation_id, spec.target_tokens)
                != (
                    permit.installed_epoch.operation_id,
                    permit.installed_epoch.target_tokens,
                )
                for spec, _ in groups.values()
            ):
                raise InstallProtocolError(
                    "actual bank differs from forward installation epoch"
                )
            yield groups

    def installation_complete(self, receipt):
        return super().installation_complete(
            receipt
        ) and self.runtime.exchange.resume_complete(receipt.epoch)

    def cancel(self, reason="request cancelled"):
        self.runtime.cancel()
        for peer in self._peers.values():
            peer.stop()  # no release; live reader ownership survives cancellation

    def close(self):
        self.cancel()
        if self.runtime._forward is not None:
            raise InstallProtocolError(
                "forward/result processing must drain before close"
            )
        super().close()
