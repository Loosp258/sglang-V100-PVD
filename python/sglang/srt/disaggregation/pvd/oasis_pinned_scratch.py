"""Request-owned pinned host scratch; no recycling before local completion."""
import threading
import uuid
import torch
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


def scratch_bytes(capacity):
    if type(capacity) is not int or not 1 <= capacity <= 32:
        raise ValueError('bounded sparse bank capacity required')
    return 28*128*4 + 4*capacity*2*128*2 + 2*capacity*2*128*2


class PinnedScratchLease:
    def __init__(self, pool, slot):
        self.pool, self.slot = pool, slot
        self.thread = threading.get_ident()
        self.query, self.bank, self.receive = slot['tensors']
        self.active, self.pending = True, False
        self.event = None

    def begin(self):
        self._owner()
        self.pending = True  # BEFORE the first local copy is enqueued.

    def _owner(self):
        if not self.active or threading.get_ident() != self.thread:
            raise RuntimeError('exact active scratch owner required')

    def finish(self, event):
        self._owner()
        self.event = event
        try:
            event.synchronize()
        except BaseException:
            self.pool._quarantine(self, 'pinned scratch local event completion unknown')
            raise
        self.pending = False
        self.pool._return(self)

    def abort(self, stream):
        self._owner()
        try:
            if self.pending:
                stream.synchronize()
        except BaseException:
            self.pool._quarantine(self, 'pinned scratch local stream completion unknown')
            raise
        self.pending = False
        self.pool._return(self)


class PinnedScratchPool:
    def __init__(self, budget, *, capacity, slots=2):
        if not isinstance(budget, TransferBudget) or type(slots) is not int or not 1 <= slots <= 2:
            raise ValueError('explicit bounded pinned scratch budget required')
        self.budget, self.capacity, self.slots = budget, capacity, slots
        self.bytes_per_slot = scratch_bytes(capacity)
        self._lock = threading.RLock()
        self._slots, self._closed, self._unknown = [], False, None
        self._allocations, self._acquired, self._returned = 0, 0, 0
        self._prefix = 'oasis-pinned:' + uuid.uuid4().hex

    def _allocate(self):
        return (torch.empty((28,128),dtype=torch.float32,pin_memory=True),
                torch.empty((4,self.capacity,2,128),dtype=torch.float16,pin_memory=True),
                torch.empty(2*self.capacity*512,dtype=torch.uint8,pin_memory=True))

    def acquire(self):
        with self._lock:
            if self._closed or self._unknown:
                raise RuntimeError('pinned scratch closed or quarantined')
            slot = next((s for s in self._slots if s['lease'] is None), None)
            if slot is None:
                if len(self._slots) >= self.slots:
                    raise RuntimeError('pinned scratch capacity exhausted')
                slot = dict(owner=f'{self._prefix}:{len(self._slots)}', lease=None)
                self.budget.reserve(slot['owner'], self.bytes_per_slot, 0)
                try:
                    slot['tensors'] = self._allocate()
                except BaseException:
                    self.budget.release(slot['owner'])
                    raise
                self._slots.append(slot)
                self._allocations += 1
            lease = PinnedScratchLease(self, slot)
            slot['lease'] = lease
            self._acquired += 1
            return lease

    def _return(self, lease):
        with self._lock:
            lease._owner()
            if self._unknown or lease.pending or lease.slot['lease'] is not lease:
                raise RuntimeError('retain pending or unknown pinned scratch')
            lease.slot['lease'] = None
            lease.active = False
            lease.query = lease.bank = lease.receive = lease.event = None
            self._returned += 1

    def _quarantine(self, lease, reason):
        with self._lock:
            lease._owner()
            self._unknown = reason

    def close(self):
        with self._lock:
            if self._unknown or any(s['lease'] for s in self._slots):
                raise RuntimeError('retain live or unknown pinned scratch owners')
            for slot in self._slots:
                slot['tensors'] = None
                self.budget.release(slot['owner'])
            self._slots.clear()
            self._closed = True

    def snapshot(self):
        with self._lock:
            return dict(physical_slots=len(self._slots), physical_bytes=len(self._slots)*self.bytes_per_slot,
                        allocations=self._allocations, acquired=self._acquired, returned=self._returned,
                        live=sum(s['lease'] is not None for s in self._slots),
                        unknown=self._unknown, closed=self._closed)
