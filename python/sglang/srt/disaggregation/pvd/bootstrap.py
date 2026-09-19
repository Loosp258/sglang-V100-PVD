"""Request-local gating for the waiting-queue-triggered initial KV pull.

This module is NOT wired into Decode yet. It owns no tensors, registrations,
native handles or transport permissions, and it never releases anything.
Every transition is an assertion by the caller after it has validated the
corresponding real-world fact; this object only refuses illegal orderings.

The confirmed bootstrap shape it encodes:

  prealloc queue -> transfer queue -> FINAL WAITING QUEUE  (the trigger)
      -> D publishes an authorization over the request's already-preallocated
         final KV pages and requests delivery
      -> V performs the authorized RDMA WRITE (D is the initiator, V is still
         the writer; there is no RDMA READ and no new transport direction)
      -> native terminal + identity checks            -> RECEIVED
      -> GPU visibility + installation + TP agreement -> INSTALLED (RUNNABLE)
      -> admission into the running batch

Entering the waiting queue and observing KV_STORED on V may happen in either
order; both are required before a delivery request may go out. While pulling,
the request sits inside ``scheduler.waiting_queue`` and is NOT runnable: the
batch builder must skip it and must not charge it against the batch token
budget. It is never a completion barrier for a running request.

All transitions belong to the scheduler thread. Background callbacks must
enqueue results to that thread rather than mutating this gate directly.
"""

from dataclasses import dataclass
from enum import Enum


class BootstrapState(str, Enum):
    QUEUED = "queued"
    AUTHORIZED = "authorized"
    RECEIVED = "received"
    INSTALLED = "installed"
    CLOSED = "closed"


@dataclass(frozen=True)
class BootstrapTicket:
    """Local identity of the single initial pull; NOT a wire authorization.

    A real authorization must additionally bind Entry, per-rank destination
    regions, byte ranges, epoch and generation. This carries only what the
    scheduler needs to reject stale or foreign completions.
    """

    delivery_id: str
    receiver_epoch: str
    prompt_tokens: int


class BootstrapGate:
    """One initial Prompt KV pull per request, gated on the waiting queue.

    A request is runnable only in INSTALLED. Arrival is not readiness, and a
    published destination address is not installation. Bootstrap happens
    exactly once: after ``handoff()`` the periodic refresh clock owns the
    request, starting from zero committed Decode tokens, and round 0 is never
    fetched again.

    ``close()`` refuses further progress and retains the pending identity. It
    releases nothing, fences nothing and stops no submitted RDMA. Resource
    owners must drain native work separately; a closed gate is not permission
    to reuse the destination pages.
    """

    def __init__(self, delivery_id: str, receiver_epoch: str, prompt_tokens: int):
        if not isinstance(delivery_id, str) or not delivery_id.strip():
            raise ValueError("delivery_id must be a non-empty string")
        if not isinstance(receiver_epoch, str) or not receiver_epoch.strip():
            raise ValueError("receiver_epoch must be a non-empty string")
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be a positive integer")
        self.delivery_id = delivery_id
        self.receiver_epoch = receiver_epoch
        self.prompt_tokens = prompt_tokens
        self._state = BootstrapState.QUEUED
        self._in_waiting_queue = False
        self._source_ready = False
        self._ticket = None
        self._handed_off = False

    @property
    def state(self) -> BootstrapState:
        return self._state

    @property
    def ticket(self) -> BootstrapTicket | None:
        return self._ticket

    @property
    def in_waiting_queue(self) -> bool:
        return self._in_waiting_queue

    @property
    def source_ready(self) -> bool:
        return self._source_ready

    @property
    def is_runnable(self) -> bool:
        """The only question the batch builder should ask about a newcomer."""
        return self._state == BootstrapState.INSTALLED

    def _require_open(self) -> None:
        if self._state == BootstrapState.CLOSED:
            raise ValueError("bootstrap gate is closed")

    # -- preconditions, in either order ------------------------------------

    def enter_waiting_queue(self) -> None:
        """The scheduler has placed this request into the final waiting queue.

        This is the trigger. Nothing earlier in the Decode queue chain may
        publish an authorization or start a transfer.
        """
        self._require_open()
        if self._state != BootstrapState.QUEUED:
            raise ValueError("bootstrap has already left the waiting-queue stage")
        self._in_waiting_queue = True

    def mark_source_ready(self) -> None:
        """V reports KV_STORED for this Entry; index readiness is not required."""
        self._require_open()
        self._source_ready = True

    def can_request(self) -> bool:
        return (
            self._state == BootstrapState.QUEUED
            and self._in_waiting_queue
            and self._source_ready
        )

    # -- the single pull ----------------------------------------------------

    def begin(self) -> BootstrapTicket:
        """Publish the authorization and request delivery, exactly once.

        The caller must already have registered and pinned the destination
        pages: a descriptor may only become visible to V after that.
        """
        self._require_open()
        if self._ticket is not None:
            raise ValueError("initial pull has already been authorized")
        if not self._in_waiting_queue:
            raise ValueError("initial pull requires the final waiting queue")
        if not self._source_ready:
            raise ValueError("initial pull requires a stored Entry on V")
        self._ticket = BootstrapTicket(
            delivery_id=self.delivery_id,
            receiver_epoch=self.receiver_epoch,
            prompt_tokens=self.prompt_tokens,
        )
        self._state = BootstrapState.AUTHORIZED
        return self._ticket

    def _match(self, ticket: BootstrapTicket) -> None:
        if not isinstance(ticket, BootstrapTicket) or self._ticket != ticket:
            raise ValueError("stale or foreign bootstrap ticket")

    def mark_received(self, ticket: BootstrapTicket) -> None:
        """Native terminal state and identity checks passed. Not yet readable.

        Refused once closed: a late completion never re-opens a cancelled
        request, and it is never permission to reuse the destination.
        """
        self._require_open()
        self._match(ticket)
        if self._state not in (BootstrapState.AUTHORIZED, BootstrapState.RECEIVED):
            raise ValueError("bootstrap is not awaiting delivery")
        self._state = BootstrapState.RECEIVED

    def mark_installed(self, ticket: BootstrapTicket) -> None:
        """GPU visibility, installation and required TP agreement all succeeded.

        Arrival alone is not enough: AUTHORIZED cannot jump to INSTALLED.
        """
        self._require_open()
        self._match(ticket)
        if self._state != BootstrapState.RECEIVED:
            raise ValueError("bootstrap KV has not been received yet")
        self._state = BootstrapState.INSTALLED

    # -- handoff to the periodic clock --------------------------------------

    def handoff(self) -> int:
        """Return the committed Decode token count the refresh clock starts at.

        Always zero: P sampled the first output token, and D has committed
        nothing. Calling this twice would mean bootstrapping the same request
        again, which is exactly what must not happen after admission.
        """
        self._require_open()
        if self._state != BootstrapState.INSTALLED:
            raise ValueError("bootstrap KV is not installed")
        if self._handed_off:
            raise ValueError("bootstrap has already been handed off")
        self._handed_off = True
        return 0

    @property
    def handed_off(self) -> bool:
        return self._handed_off

    def close(self) -> None:
        """Refuse further progress; release, fence and drain nothing."""
        self._state = BootstrapState.CLOSED
