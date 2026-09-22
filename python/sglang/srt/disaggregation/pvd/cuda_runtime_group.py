"""Owned TP1 CUDA install runtime; no emulated multi-rank GPU execution.

Local queues connect the exact receive record, participant and runtime. Remote
payload completion remains the receive record's job; queue progress is no fence.
"""

from contextlib import contextmanager

from sglang.srt.disaggregation.pvd.cpu_runtime_group import _LocalRuntimeInstallCore
from sglang.srt.disaggregation.pvd.cuda_rank_install import CUDARankInstallParticipant
from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import CUDASparseReceiveRecord
from sglang.srt.disaggregation.pvd.cuda_working_set import CUDASparseWorkingSet
from sglang.srt.disaggregation.pvd.sparse_install import (
    InstallProtocolError,
    RankInstallCoordinator,
)


class CUDARuntimeInstallGroup(_LocalRuntimeInstallCore):
    _participant_type = CUDARankInstallParticipant

    def _initialize_banks(self, banks, *, interval, lead_tokens):
        self._banks = dict(banks)
        if len(self._banks) != 1 or any(
            type(rank) is not int
            or rank < 0
            or not isinstance(bank, CUDASparseWorkingSet)
            for rank, bank in self._banks.items()
        ):
            raise InstallProtocolError(
                "CUDA local runtime requires exactly one TP1 rank bank"
            )
        bank = next(iter(self._banks.values()))
        bank._open()
        self.coordinator = RankInstallCoordinator(
            *bank.identity[:3],
            rank_layouts={rank: b.identity[3] for rank, b in self._banks.items()},
            interval=interval,
            lead_tokens=lead_tokens,
        )
        self._receipts = {}

    def describe_banks(self):
        self.coordinator._owner()
        return {
            rank: {
                "identity": bank.identity,
                "groups": bank.expected_groups,
                "prompt_tokens": bank.prompt_tokens,
                "head_dim": bank.head_dim,
                "device": str(bank.device),
                "dtype": str(bank.dtype),
            }
            for rank, bank in self._banks.items()
        }

    def stage(self, epoch, rank, payloads, *, source_guard):
        """Explicit local source path; caller already proved CUDA visibility."""
        self._check_stage(epoch, rank)
        raw = self._peers[rank].stage(epoch, payloads, source_guard=source_guard)
        return self._accept_prepared(epoch, rank, raw)

    def stage_received(self, record, epoch):
        """Remote success -> local ordering -> copy -> PREPARED, exactly once."""
        if not isinstance(record, CUDASparseReceiveRecord):
            raise InstallProtocolError("explicit CUDA receive record required")
        rank = record.identity.shard_rank
        self._check_stage(epoch, rank)
        raw = record.stage(self._peers[rank], epoch, exchange=self.runtime.exchange)
        return self._accept_prepared(epoch, rank, raw)

    def _close_banks(self):
        for bank in self._banks.values():
            bank.close()  # Refuses live readers/UNKNOWN; never force-releases.

    @contextmanager
    def model_forward(self, consumer, *, slot, decode_tokens, pool_owner):
        """Commit model output only AFTER successful exit from this scope.

        Unknown device completion retains the runtime permit as well as the
        consumer's owners. Cancellation/timeout cannot manufacture retirement.
        """
        from sglang.srt.disaggregation.pvd.cuda_model_attention import (
            CUDADecodeBinding,
            CUDAModelSparseConsumer,
        )

        if not isinstance(consumer, CUDAModelSparseConsumer):
            raise InstallProtocolError("explicit CUDA model consumer required")
        self.progress()
        permit = self.runtime.begin_forward(decode_tokens)
        peer = next(iter(self._peers.values()))
        try:
            with consumer.bind(
                [CUDADecodeBinding(slot, decode_tokens, peer, self.runtime.exchange)],
                pool_owner=pool_owner,
            ):
                yield consumer
        except BaseException:
            if consumer.snapshot()["quarantine"] is None:
                self.runtime.finish_forward(
                    permit, readers_drained=True, succeeded=False
                )
            else:
                self.cancel("CUDA model completion unknown")
            raise
        else:
            if not self.runtime.finish_forward(
                permit, readers_drained=True, succeeded=True
            ):
                raise InstallProtocolError("runtime refused completed model result")
