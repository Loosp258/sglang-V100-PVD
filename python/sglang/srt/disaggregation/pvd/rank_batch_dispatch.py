"""Wait-all rank admission and result scope; not a production Scheduler hook.

No tensors, sampling or token writes. The execution owner supplies the SAME
shared target arbiter used by probe/forward and must drain real readers before
processing. Control tickets never stand in for bank leases or native fences.
"""

import uuid
from contextlib import contextmanager
from dataclasses import dataclass

from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import TargetExecutionArbiter
from sglang.srt.disaggregation.pvd.rank_install_runtime import (
    RankForwardPermit,
    RankInstallRuntime,
)
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError


@dataclass(frozen=True)
class RankBatchMember:
    runtime: RankInstallRuntime
    committed_tokens: int


@dataclass(frozen=True)
class RankBatchPermit:
    operation_id: str
    members: tuple[RankForwardPermit, ...]


@dataclass(frozen=True)
class RankBatchDecision:
    permit: RankForwardPermit
    accepted: bool


class RankBatchDispatcher:
    """One immutable batch, one target lease, one authoritative result writer.

    Membership/counts come from the result owner's committed-token snapshots;
    this class never maintains a second token clock. Failed members are not
    silently removed to form a smaller dispatch. Different rounds/counts are
    allowed, but all members must refer to the same bound worker group.
    """

    def __init__(self, arbiter, *, max_requests):
        if not isinstance(arbiter, TargetExecutionArbiter):
            raise InstallProtocolError("explicit shared target arbiter required")
        arbiter.owner()
        if type(max_requests) is not int or max_requests <= 0:
            raise InstallProtocolError("positive batch capacity required")
        self.arbiter, self.max_requests = arbiter, max_requests
        self._ticket = self._lease = None
        self._members = ()
        self._transitioning = False

    def _owner(self):
        self.arbiter.owner()
        if self._transitioning:
            raise InstallProtocolError("batch operation cannot be reentered")

    def begin(self, members):
        self._owner()
        if self._ticket is not None or self.arbiter.busy:
            raise InstallProtocolError("target execution is already owned")
        if type(members) is not tuple or not 0 < len(members) <= self.max_requests:
            raise InstallProtocolError(
                "bounded nonempty tuple of batch members required"
            )
        requests, runtimes, peers = set(), set(), None
        for member in members:
            if (
                not isinstance(member, RankBatchMember)
                or not isinstance(member.runtime, RankInstallRuntime)
                or type(member.committed_tokens) is not int
                or member.committed_tokens < 0
            ):
                raise InstallProtocolError("runtime and committed-token count required")
            runtime = member.runtime
            runtime.exchange.coordinator._owner()
            request = runtime.exchange.coordinator.identity[0]
            if runtime in runtimes or request in requests:
                raise InstallProtocolError("duplicate batch request or runtime")
            requests.add(request)
            runtimes.add(runtime)
            bound_peers = dict(runtime.exchange.peer_epochs)
            if peers is not None and peers != bound_peers:
                raise InstallProtocolError("batch members must share bound workers")
            peers = bound_peers

        self._transitioning = True
        acquired = []
        lease = None
        try:
            # Evaluate ALL gates before acquiring ANY permit. No subset dispatch.
            ready = [m.runtime.can_decode(m.committed_tokens) for m in members]
            if not all(ready):
                raise InstallProtocolError(
                    "selected batch must wait or abort as a whole"
                )
            lease = self.arbiter.acquire()
            for member in members:
                permit = member.runtime.begin_forward(member.committed_tokens)
                member.runtime._forward_batch = self
                acquired.append((member.runtime, permit))
            ticket = RankBatchPermit(uuid.uuid4().hex, tuple(p for _, p in acquired))
            self._members, self._lease, self._ticket = members, lease, ticket
            return ticket
        except BaseException:
            # begin() has NOT returned: no execution could have been dispatched.
            # This is metadata rollback, not an assertion of native completion.
            for runtime, permit in acquired:
                runtime._retire_forward(permit, batch=self)
            if lease is not None:
                self.arbiter.release(lease)
            raise
        finally:
            self._transitioning = False

    def _match(self, ticket):
        if ticket is None or ticket is not self._ticket:
            raise InstallProtocolError("stale or foreign batch completion")
        for member, permit in zip(self._members, ticket.members, strict=True):
            member.runtime._check_forward(permit, batch=self)

    def _retire(self, ticket):
        self._match(ticket)
        for member, permit in zip(self._members, ticket.members, strict=True):
            member.runtime._retire_forward(permit, batch=self)
        self.arbiter.release(self._lease)
        self._members, self._ticket, self._lease = (), None, None

    @contextmanager
    def processing(self, ticket, *, readers_drained, succeeded):
        """Apply decisions synchronously with the existing result processor.

        Call only after the WHOLE batch execution and all actual readers drain.
        Invalid completion assertions retain every ticket and the target lease.
        ``succeeded=False`` rejects every row (e.g. a failed shared forward).
        Otherwise request-local failure rejects only that request's row.

        The yielded tuple is in original dispatch order. No second sampler or
        Req writer is introduced. Do not await, pump control or run arbitrary
        callbacks while applying it: it is a synchronous result-commit scope,
        not a reusable authorization. Later notifications apply on the next
        owner poll; already committed tokens are never rolled back.
        """
        self._owner()
        self._match(ticket)
        if readers_drained is not True or type(succeeded) is not bool:
            raise InstallProtocolError(
                "explicit whole-batch drain and success required"
            )
        self._transitioning = True
        try:
            decisions = tuple(
                RankBatchDecision(
                    permit,
                    member.runtime._prepare_forward_result(
                        permit, succeeded=succeeded, batch=self
                    ),
                )
                for member, permit in zip(self._members, ticket.members, strict=True)
            )
            yield decisions
        except BaseException:
            # No rollback of Req tokens possibly written by the authoritative
            # processor. Abort all members; never retry a half-processed batch.
            for member in self._members:
                member.runtime.cancel()
            raise
        finally:
            try:
                self._retire(ticket)
            finally:
                self._transitioning = False
