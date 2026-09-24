"""Wait-all TP1 CUDA batch execution with rank-owned result processing.

No sampler or Req writer: callers supply the existing synchronous forward and
authoritative result processor. These callbacks must not await or reenter decode.
"""

import inspect
import threading
import uuid
from dataclasses import dataclass

from sglang.srt.disaggregation.pvd.cuda_model_attention import (
    CUDADecodeBinding,
    CUDAModelSparseConsumer,
)
from sglang.srt.disaggregation.pvd.cuda_runtime_group import CUDARuntimeInstallGroup
from sglang.srt.disaggregation.pvd.rank_batch_dispatch import (
    RankBatchDispatcher,
    RankBatchMember,
)
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import ResourceGuard


@dataclass(frozen=True)
class CUDARuntimeBatchMember:
    group: CUDARuntimeInstallGroup
    slot: int
    committed_tokens: int


class CUDARankBatchExecutor:
    """Keep target lock and ALL permits until the result callback completes.

    Conservative all-or-nothing admission/result acceptance. A canceled member
    discards the whole uncommitted model result; tokens already committed by a
    callback that subsequently fails are never rolled back/replayed.
    """

    def __init__(self, consumer, arbiter, *, max_requests):
        if not isinstance(consumer, CUDAModelSparseConsumer) or not isinstance(
            consumer._lock, type(threading.RLock())
        ):
            raise InstallProtocolError(
                "CUDA consumer with shared reentrant lock required"
            )
        self.consumer = consumer
        self.dispatcher = RankBatchDispatcher(arbiter, max_requests=max_requests)
        self._active = self._quarantined = False
        self._retained = None

    def run(self, members, *, pool_owner, forward, process_results):
        """No model/sampler outputs may be committed inside forward()."""
        self.dispatcher._owner()
        if self._active or self._quarantined:
            raise InstallProtocolError("CUDA batch is active or quarantined")
        if (
            type(members) is not tuple
            or not 0
            < len(members)
            <= min(self.dispatcher.max_requests, self.consumer._max_batch)
            or not callable(forward)
            or not callable(process_results)
            or inspect.iscoroutinefunction(forward)
            or inspect.iscoroutinefunction(process_results)
            or not isinstance(pool_owner, ResourceGuard)
        ):
            raise InstallProtocolError(
                "bounded members and synchronous execution callbacks required"
            )
        slots = set()
        for member in members:
            if (
                not isinstance(member, CUDARuntimeBatchMember)
                or not isinstance(member.group, CUDARuntimeInstallGroup)
                or type(member.slot) is not int
                or member.slot <= 0
                or member.slot in slots
            ):
                raise InstallProtocolError("unique CUDA group/slot bindings required")
            slots.add(member.slot)
            member.group.progress()
        if not self.consumer._lock.acquire(blocking=False):
            raise InstallProtocolError("target execution is busy")
        self._active = True
        ticket = None
        pin = f"cuda-rank-batch:{uuid.uuid4().hex}"
        pinned = False
        try:
            ticket = self.dispatcher.begin(
                tuple(
                    RankBatchMember(m.group.runtime, m.committed_tokens)
                    for m in members
                )
            )
            bindings = tuple(
                CUDADecodeBinding(
                    m.slot,
                    m.committed_tokens,
                    next(iter(m.group._peers.values())),
                    m.group.runtime.exchange,
                )
                for m in members
            )
            self._retained = (members, pool_owner, None)
            try:
                pool_owner.pin(pin)
                pinned = True
                with self.consumer.bind(bindings, pool_owner=pool_owner):
                    result = forward()
                    self._retained = (members, pool_owner, result)
            except BaseException:
                if self.consumer.snapshot()["quarantine"] is not None:
                    self._quarantined = True
                    for member in members:
                        member.group.cancel("CUDA batch completion unknown")
                else:
                    with self.dispatcher.processing(
                        ticket, readers_drained=True, succeeded=False
                    ):
                        pass
                raise
            with self.dispatcher.processing(
                ticket, readers_drained=True, succeeded=True
            ) as decisions:
                if not all(d.accepted for d in decisions):
                    refused = tuple(
                        (
                            index,
                            member.group.runtime.snapshot()["phase"],
                            member.group.runtime.snapshot()["reason"],
                        )
                        for index, (member, decision) in enumerate(
                            zip(members, decisions, strict=True)
                        )
                        if not decision.accepted
                    )
                    raise InstallProtocolError(
                        f"CUDA batch result refused; commit nothing; refused={refused}"
                    )
                # Runtime tickets and outer target lock remain held here.
                return process_results(result)
        finally:
            # A failure before/inside result processing may also retain a ticket.
            if ticket is not None and self.dispatcher._ticket is not None:
                self._quarantined = True
            if not self._quarantined:
                if pinned:
                    try:
                        pool_owner.unpin(pin)
                    except BaseException:
                        self._quarantined = True
                        raise
                self._retained = None
                self._active = False
                self.consumer._lock.release()
