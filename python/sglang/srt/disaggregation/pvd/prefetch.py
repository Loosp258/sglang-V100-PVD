"""Request-local timing for the future predictive KV prefetch path.

This module is NOT wired into Decode yet. It owns no tensors, registrations,
native handles or transport permissions. In particular, READY is an assertion
by the caller after validating delivery/visibility, not transport evidence.
The existing RefreshClock/full_prompt path remains unchanged.

All transitions belong to the scheduler thread. Background callbacks must
enqueue results to that thread, never mutate this clock directly. Only counts
of committed D tokens (excluding P's first token) may enter this API.
"""

from dataclasses import dataclass
from enum import Enum


class PrefetchState(str, Enum):
    IDLE = "idle"
    IN_FLIGHT = "in_flight"
    READY = "ready"
    CLOSED = "closed"


@dataclass(frozen=True)
class PrefetchTicket:
    """Local timing identity; NOT an RDMA grant or a complete wire request."""

    delivery_id: str
    round: int
    prefix_tokens: int
    target_tokens: int


class PrefetchClock:
    """One next generation per request, with a fixed installation boundary.

    Initial installation is required at D count zero. A periodic round can
    begin at boundary - lead_tokens, but can only be installed at boundary.
    Wall time, other requests, and early readiness never move the boundary.
    Reaching it without issuing a prefetch allows a synchronous late start.
    Advancing committed tokens past it is an error, not an implicit refresh.

    Cancellation closes the clock permanently and retains its pending identity.
    Resource owners must separately fence/drain native work. No retry/reuse or
    buffer-release semantics are inferred from this purely logical state.

    The caller must use an incarnation-unique delivery_prefix; reconstructing
    a new clock with the same prefix is not a safe restart/retry mechanism.
    Wire identities must additionally bind query, Entry, epoch and generation.
    """

    def __init__(self, delivery_prefix: str, interval: int, lead_tokens: int):
        if not isinstance(delivery_prefix, str) or not delivery_prefix.strip():
            raise ValueError("delivery_prefix must be a non-empty string")
        if type(interval) is not int or interval <= 0:
            raise ValueError("interval must be a positive integer")
        if type(lead_tokens) is not int or not 0 <= lead_tokens < interval:
            raise ValueError("lead_tokens must be an integer in [0, interval)")
        self.prefix = delivery_prefix
        self.interval = interval
        self.lead_tokens = lead_tokens
        self._round = 0
        self._installed_tokens = None
        self._observed_tokens = 0
        self._pending = None
        self._state = PrefetchState.IDLE

    @property
    def round(self) -> int:
        return self._round

    @property
    def installed_tokens(self) -> int | None:
        return self._installed_tokens

    @property
    def boundary(self) -> int:
        if self._installed_tokens is None:
            return 0
        return self._installed_tokens + self.interval

    @property
    def pending(self) -> PrefetchTicket | None:
        return self._pending

    @property
    def state(self) -> PrefetchState:
        return self._state

    def _require_open(self) -> None:
        if self._state == PrefetchState.CLOSED:
            raise ValueError("prefetch clock is closed")

    def _observe(self, decode_tokens: int) -> None:
        self._require_open()
        if type(decode_tokens) is not int or decode_tokens < 0:
            raise ValueError("decode_tokens must be a non-negative integer")
        if decode_tokens < self._observed_tokens:
            raise ValueError("committed Decode token count regressed")
        if decode_tokens > self.boundary:
            raise ValueError("Decode advanced past an uninstalled KV boundary")
        self._observed_tokens = decode_tokens

    def needs_prefetch(self, decode_tokens: int) -> bool:
        self._observe(decode_tokens)
        return self._pending is None and decode_tokens >= max(
            0, self.boundary - self.lead_tokens
        )

    def requires_install(self, decode_tokens: int) -> bool:
        """True means this request must not execute its next forward yet."""
        self._observe(decode_tokens)
        return decode_tokens == self.boundary

    def begin(self, decode_tokens: int) -> PrefetchTicket:
        self._observe(decode_tokens)
        if self._pending is not None:
            raise ValueError("prefetch already in flight or awaiting installation")
        if not self.needs_prefetch(decode_tokens):
            raise ValueError("prefetch lead window has not started")
        ticket = PrefetchTicket(
            delivery_id=f"{self.prefix}:prefetch:{self._round}",
            round=self._round,
            prefix_tokens=decode_tokens,
            target_tokens=self.boundary,
        )
        self._pending = ticket
        self._state = PrefetchState.IN_FLIGHT
        return ticket

    def _match(self, ticket: PrefetchTicket) -> None:
        self._require_open()
        if not isinstance(ticket, PrefetchTicket) or self._pending != ticket:
            raise ValueError("stale or foreign prefetch ticket")

    def mark_ready(self, ticket: PrefetchTicket) -> None:
        """Caller has verified data identity, completion and GPU visibility.

        Does not install KV or advance the clock. Matching duplicate readiness
        notifications are harmless while this ticket is still pending.
        """
        self._match(ticket)
        self._state = PrefetchState.READY

    def can_install(self, ticket: PrefetchTicket, decode_tokens: int) -> bool:
        self._match(ticket)
        self._observe(decode_tokens)
        return self._state == PrefetchState.READY and decode_tokens == self.boundary

    def commit_install(self, ticket: PrefetchTicket, decode_tokens: int) -> None:
        """Call AFTER local installation and required TP/ACK agreement succeed.

        An integration must first check can_install(), install safely, then
        commit on all participating ranks. This method performs no GPU work.
        """
        if not self.can_install(ticket, decode_tokens):
            raise ValueError("KV is not ready for installation at this boundary")
        self._installed_tokens = ticket.target_tokens
        self._round += 1
        self._pending = None
        self._state = PrefetchState.IDLE

    def close(self) -> None:
        """Reject future completions; does NOT release or fence any resource."""
        self._state = PrefetchState.CLOSED
