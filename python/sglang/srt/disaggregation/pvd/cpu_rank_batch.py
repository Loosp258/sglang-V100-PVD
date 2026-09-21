"""Bind existing CPU model/Req execution to the exact rank-owned prompt banks.

Opt-in CPU fixture path, no production Scheduler or GPU admission. Reuses the
existing executor, sampler/result writer and lifecycle clock, not a second one.
"""

from contextlib import contextmanager

from sglang.srt.disaggregation.pvd.cpu_batch_dispatch import CPUBatchDispatcher
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import LifecycleError
from sglang.srt.disaggregation.pvd.cpu_runtime_group import CPURuntimeInstallGroup
from sglang.srt.disaggregation.pvd.rank_batch_dispatch import (
    RankBatchDispatcher,
    RankBatchMember,
)


class CPURankBatchDispatcher(CPUBatchDispatcher):
    def __init__(self, arbiter, *, max_requests):
        super().__init__(arbiter)
        self.rank_dispatcher = RankBatchDispatcher(arbiter, max_requests=max_requests)
        self._rank_ticket = None
        self._bindings = ()
        self._in_results = False

    def _acquire(self, members):
        bindings = tuple((life, life.controller.group) for life in members)
        if any(
            not isinstance(group, CPURuntimeInstallGroup)
            or group.runtime.exchange.coordinator is not group.coordinator
            or group.coordinator.identity[0] != life.request_id
            for life, group in bindings
        ):
            raise LifecycleError("every request must own its exact CPU runtime group")
        rank_ticket = self.rank_dispatcher.begin(
            tuple(
                RankBatchMember(g.runtime, life.committed_tokens)
                for life, g in bindings
            )
        )
        self._rank_ticket, self._bindings = rank_ticket, bindings
        # No second lease: the rank dispatcher owns the shared target arbiter.

    def _match(self, ticket):
        super()._match(ticket)
        self.rank_dispatcher._match(self._rank_ticket)
        for (life, group), member, permit in zip(
            self._bindings, ticket.members, self._rank_ticket.members, strict=True
        ):
            if (
                life.controller.group is not group
                or group.runtime.exchange.coordinator is not group.coordinator
                or permit.identity != group.coordinator.identity
                or permit.committed_tokens != member.permit.committed_tokens
                or member.permit.incarnation != life.incarnation
            ):
                raise LifecycleError(
                    "runtime, lifecycle or installed-bank binding changed"
                )

    def _retire(self):
        if not self._in_results:
            raise LifecycleError("rank-owned execution requires a drained result scope")
        for life in self._members:
            life._permit = life._batch_owner = None
        self._members, self._ticket, self._lease = (), None, None
        # Rank tickets/target lease remain held until the result scope exits.

    @contextmanager
    def result_scope(self, ticket, *, succeeded=True):
        """Only after actual synchronous execution/readers unwind.

        CPUScheduleBridge enforces executor completion before entering. Direct
        complete()/fail() callers retain the same drain obligation as the base
        CPU dispatcher; this scope cannot manufacture hardware completion.
        """
        self._match(ticket)
        if self._in_results:
            raise LifecycleError("rank result processing cannot nest")
        self._in_results = True
        try:
            with self.rank_dispatcher.processing(
                self._rank_ticket, readers_drained=True, succeeded=succeeded
            ) as decisions:
                try:
                    for life, decision in zip(self._members, decisions, strict=True):
                        if not decision.accepted:
                            life.terminate("rank runtime refused target result")
                    yield
                    if self._ticket is not None:
                        raise LifecycleError(
                            "result owner did not complete the CPU batch"
                        )
                except BaseException:
                    for life in self._members:
                        life.terminate("rank-bound result processing failed")
                    if self._ticket is not None:
                        self._retire()
                    raise
        except BaseException:
            # Preparation can fail before yielding any decisions. Retire CPU
            # metadata only if the rank result scope already retired its own.
            if self.rank_dispatcher._ticket is None and self._ticket is not None:
                for life in self._members:
                    life.terminate("rank result preparation failed")
                self._retire()
            raise
        finally:
            self._in_results = False
            if self.rank_dispatcher._ticket is None:
                self._rank_ticket, self._bindings = None, ()

    def complete(self, ticket, results):
        with self.result_scope(ticket):
            return super().complete(ticket, results)

    def fail(self, ticket, reason):
        if self._in_results:
            return super().fail(ticket, reason)
        with self.result_scope(ticket, succeeded=False):
            return super().fail(ticket, reason)
