"""CUDA receive -> bank staging -> exact all-rank resume -> Delivery ACK.

Explicit component assembly, not yet the production Scheduler factory. The
same protocol can use Mooncake; fake transport in CPU tests is not RDMA proof.
Each destination is private: nobody reads it until remote success and local
CUDA ordering. Unknown CUDA/native state retains registration and budget.
"""

import torch
from sglang.srt.disaggregation.pvd.cuda_rank_install import CUDARankInstallParticipant
from sglang.srt.disaggregation.pvd.cuda_receive_ordering import CUDAReceiveOrdering
from sglang.srt.disaggregation.pvd.rank_install_wire import (
    RankInstallExchange,
    RankInstallMessage,
)
from sglang.srt.disaggregation.pvd.sparse_receiver import (
    SparseReceiveError,
    SparseReceiveRecord,
    SparseReceiveRegistry,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import ResourceGuard


class CUDASparseReceiveRegistry(SparseReceiveRegistry):
    def __init__(self, engine, budget, *, receiver_epoch, device):
        super().__init__(engine, budget, receiver_epoch=receiver_epoch)
        self.device = torch.device(device)
        self.ordering = CUDAReceiveOrdering(self.device)
        if not torch.cuda.is_available():
            raise SparseReceiveError("CUDA receive requested but CUDA is unavailable")

    def _new_record(self, manifest, identity, client):
        return CUDASparseReceiveRecord(self, manifest, identity, client)

    def _allocate_buffer(self, byte_count):
        return torch.empty(byte_count, dtype=torch.uint8, device=self.device)

    def _before_register(self, record):
        self.ordering.prepare(record._buffer)

    def _after_register(self, record):
        descriptor = record._registration.descriptor
        if (
            descriptor.device != str(self.device)
            or descriptor.address != record._buffer.data_ptr()
            or record._registration.buffer.data_ptr() != record._buffer.data_ptr()
            or record._registration.buffer.device != self.device
            or record._registration.buffer.numel()
            * record._registration.buffer.element_size()
            != record.manifest.nbytes
        ):
            raise SparseReceiveError(
                "registered CUDA destination does not match allocation"
            )
        record._source_guard = ResourceGuard(
            record._registration,
            record._unregister_destination,
        )


class CUDASparseReceiveRecord(SparseReceiveRecord):
    def __init__(self, *args):
        super().__init__(*args)
        self._source_guard = None
        self._local_unknown = None
        self._ordered = False

    def _live(self):
        super()._live()
        if self._local_unknown is not None:
            raise SparseReceiveError("CUDA receive destination is quarantined")

    def stage(self, participant, epoch, *, exchange):
        self._live()
        if self._lock.locked() or not self._ready:
            raise SparseReceiveError("wait for successful delivery before staging")
        if self._receipt is not None:
            raise SparseReceiveError("destination already staged")
        if (
            not isinstance(participant, CUDARankInstallParticipant)
            or not isinstance(exchange, RankInstallExchange)
            or participant.rank != self.identity.shard_rank
            or exchange.peer_epochs.get(participant.rank) != participant.peer_epoch
            or participant._bank.device != self._registry.device
        ):
            raise SparseReceiveError(
                "matching CUDA rank participant and exchange required"
            )
        # _ready only latches after exact terminal-success/fence/extent/identity
        # validation. This synchronization is not used to manufacture that proof.
        try:
            self._registry.ordering.after_remote_write(self._registration)
        except BaseException:
            self._local_unknown = "CUDA receive ordering unknown"
            raise
        self._ordered = True
        views = self.manifest.payload_views(self._buffer)
        try:
            prepared = participant.stage(epoch, views, source_guard=self._source_guard)
        except BaseException:
            if participant._bank.snapshot()["quarantine"] is not None:
                self._local_unknown = "CUDA bank copy completion unknown"
            raise
        finally:
            for payload in views:
                payload.close()
        message = RankInstallMessage.decode(prepared)
        self._group, self._receipt = exchange, message.receipt
        return prepared

    def _release_destination(self):
        if self._local_unknown is None:
            # Release callback retires MR/storage/budget only after the bank's
            # source pin is gone. A failed native unregister remains retryable.
            self._source_guard.request_release()
            if self._source_guard.value is None:
                # ResourceGuard drops its RegisteredMemory only AFTER the
                # callback returns. Refund after that drop, not inside it.
                self._finish_close()

    def _unregister_destination(self):
        self._registry._owner()
        self._registry.engine.release_memory(self._registration)
        self._registration = self._buffer = None

    def snapshot(self):
        state = super().snapshot()
        state.update(
            device=str(self._registry.device),
            local_ordering_complete=self._ordered,
            local_completion_unknown=self._local_unknown,
            transport=self._registry.engine.name,
        )
        return state
