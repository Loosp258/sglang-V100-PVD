"""Whole-batch CPU execution ownership; no ScheduleBatch/production wiring.

One target lease, immutable dispatch membership, identity-keyed completion.
A selected batch waits as a whole if ANY member cannot run. A failed forward
aborts all its members; there is no rollback of partially written generated KV.
"""

import uuid
from dataclasses import dataclass

from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import (
    CPUDecodeLifecycle,
    DecodePermit,
    LifecycleError,
    TargetExecutionArbiter,
)


@dataclass(frozen=True)
class BatchMember:
    request_id: str
    permit: DecodePermit


@dataclass(frozen=True)
class BatchPermit:
    operation_id: str
    members: tuple[BatchMember, ...]


@dataclass(frozen=True)
class BatchTokenResult:
    request_id: str
    permit: DecodePermit
    token: int
    finished: bool = False


class CPUBatchDispatcher:
    def __init__(self, arbiter):
        if not isinstance(arbiter, TargetExecutionArbiter):
            raise LifecycleError("shared target arbiter required")
        arbiter.owner()
        self.arbiter = arbiter
        self._ticket = self._lease = None
        self._members = ()

    def begin(self, members):
        self.arbiter.owner()
        if self._ticket is not None:
            raise LifecycleError("batch already in flight")
        members = tuple(members)
        if not members or any(
            not isinstance(m, CPUDecodeLifecycle) or m.arbiter is not self.arbiter
            for m in members
        ):
            raise LifecycleError("nonempty batch must share one target arbiter")
        if len({m.request_id for m in members}) != len(members):
            raise LifecycleError("duplicate request identity in selected batch")
        # Validate EVERY participant before assigning ANY permit. Do not filter
        # waiting members and silently change the user's wait-all policy.
        ready = [m.can_decode() for m in members]
        if not all(ready):
            raise LifecycleError(
                "selected batch must wait or abort; no partial dispatch"
            )
        ticket = BatchPermit(
            uuid.uuid4().hex,
            tuple(
                BatchMember(
                    m.request_id,
                    DecodePermit(
                        m.incarnation,
                        m.committed_tokens,
                        m.outputs[-1],
                        len(m.prompt) + m.committed_tokens,
                        uuid.uuid4().hex,
                    ),
                )
                for m in members
            ),
        )
        self._lease = self.arbiter.acquire()
        self._members, self._ticket = members, ticket
        for lifecycle, member in zip(members, ticket.members, strict=True):
            lifecycle._permit = member.permit
            lifecycle._batch_owner = self
        return ticket

    def _match(self, ticket):
        self.arbiter.owner()
        if ticket is None or ticket is not self._ticket:
            raise LifecycleError("stale or foreign batch completion")
        for lifecycle, member in zip(self._members, ticket.members, strict=True):
            if (
                lifecycle._batch_owner is not self
                or lifecycle._permit is not member.permit
            ):
                raise LifecycleError("batch member ownership changed")

    def _retire(self):
        for lifecycle in self._members:
            lifecycle._permit = lifecycle._batch_owner = None
        self._members = ()
        self._ticket = None
        self.arbiter.release(self._lease)
        self._lease = None

    def complete(self, ticket, results):
        """Call only after execution/readers drained; validate all before commit.

        Missing/duplicate/foreign/malformed results leave the batch owned for
        explicit fail(). Cancelled members discard output; others may commit.
        Reordered results are fine; positional matching against a newer batch
        is never used. This CPU API returns committed results, not a Req write.
        """
        self._match(ticket)
        results = tuple(results)
        expected = {m.request_id: m.permit for m in ticket.members}
        by_id = {}
        for result in results:
            if not isinstance(result, BatchTokenResult):
                raise LifecycleError("explicit batch token results required")
            if (
                result.request_id not in expected
                or result.request_id in by_id
                or result.permit is not expected[result.request_id]
            ):
                raise LifecycleError("duplicate, stale or foreign member result")
            CPUDecodeLifecycle._token(result.token)
            if type(result.finished) is not bool:
                raise LifecycleError("finished must be an explicit bool")
            by_id[result.request_id] = result
        if set(by_id) != set(expected):
            raise LifecycleError("incomplete batch result set")
        for lifecycle in self._members:
            lifecycle.poll()
        committed = []
        for lifecycle in self._members:
            result = by_id[lifecycle.request_id]
            if lifecycle.state == "running":
                lifecycle._outputs.append(result.token)
                committed.append(result)
                if result.finished:
                    lifecycle.terminate("EOS or output limit", finished=True)
        self._retire()
        return tuple(committed)

    def fail(self, ticket, reason):
        """Actual execution has stopped; discard this whole batch's outputs."""
        self._match(ticket)
        if not isinstance(reason, str) or not reason.strip():
            raise LifecycleError("explicit batch failure reason required")
        for lifecycle in self._members:
            lifecycle.terminate(reason)
        self._retire()


def batch_results_from_logits(ticket, logits, *, finished):
    """Offline greedy test adapter, explicit complete output rows, no sampling.

    `finished` is supplied by the driver in dispatch order. No guessed EOS ids.
    The production sampler/output commit stays in the real scheduler.
    """
    import torch

    if not isinstance(ticket, BatchPermit) or not ticket.members:
        raise LifecycleError("actual nonempty dispatch ticket required")
    if (
        not isinstance(logits, torch.Tensor)
        or logits.device.type != "cpu"
        or logits.ndim != 2
        or logits.shape[0] != len(ticket.members)
        or logits.shape[1] <= 0
        or not logits.is_floating_point()
        or not torch.isfinite(logits).all()
    ):
        raise LifecycleError(
            "complete finite CPU logits [dispatch rows, vocabulary] required"
        )
    finished = tuple(finished)
    if len(finished) != len(ticket.members) or any(
        type(f) is not bool for f in finished
    ):
        raise LifecycleError("one explicit finish flag per dispatched member required")
    return tuple(
        BatchTokenResult(m.request_id, m.permit, int(token), end)
        for m, token, end in zip(
            ticket.members, logits.argmax(-1).tolist(), finished, strict=True
        )
    )
