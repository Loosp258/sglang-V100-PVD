"""Opt-in CUDA bank participant for the existing logical rank-install protocol.

This connects local CUDA-bank completion to PREPARED/PARKED/APPLIED/RESUMED.
It does not supply TP transport, remote WRITE visibility or a model attention
backend. Production CPU registries remain CPU-only. A trusted coordinator owns
the all-rank agreement; a received RESUME is not itself a memory fence.
"""

from sglang.srt.disaggregation.pvd.cpu_rank_install import _RankInstallParticipantCore
from sglang.srt.disaggregation.pvd.cuda_working_set import CUDASparseWorkingSet


class CUDARankInstallParticipant(_RankInstallParticipantCore):
    _bank_type = CUDASparseWorkingSet
    _bank_description = "CUDA"

    def _live(self):
        super()._live()
        # Check even duplicate commands. A previous ACK cannot turn a bank with
        # unknown device completion into a usable bank again.
        self._bank._open()

    def stage(self, epoch, payloads, *, source_guard):
        """Receive completion/visibility must already be proven by the caller."""
        return self._stage(epoch, payloads, source_guard=source_guard)

    def snapshot(self):
        result = super().snapshot()
        result["bank"] = self._bank.snapshot()
        result["execution_policy"] = "cuda_synchronous_experimental"
        return result
