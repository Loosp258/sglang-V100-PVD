"""Main-thread CUDA target-Q handler for an isolated probe lane process.

Model loading, process startup and Scheduler activation are separate. The
Unix server reserves host reply capacity before invoking this handler.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import torch
from sglang.srt.disaggregation.pvd.cuda_probe_search import CUDAPredictionPipeline
from sglang.srt.disaggregation.pvd.prediction import QueryVectors
from sglang.srt.disaggregation.pvd.probe_lane_protocol import (
    ProbeLaneProtocolError,
    ProbeLaneReply,
    ProbeLaneTicket,
    verify_reply,
)
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED

_DEFAULT_CACHE_IDLE_SECONDS = 5.0
_EARLY_PROMPT_CACHE_IDLE_SECONDS = 60.0


def _materialize_reply(ticket: ProbeLaneTicket, queries) -> ProbeLaneReply:
    """Select exact positions/heads without copying unused Q to host."""
    if not isinstance(queries, tuple) or len(queries) != len(ticket.layers):
        raise ProbeLaneProtocolError("target probe omitted a requested layer")
    if any(not isinstance(query, QueryVectors) for query in queries):
        raise ProbeLaneProtocolError("target probe returned a non-query object")
    by_layer = {query.layer: query for query in queries}
    if len(by_layer) != len(ticket.layers) or set(by_layer) != set(ticket.layers):
        raise ProbeLaneProtocolError("target probe layer coverage differs")
    prefix = ticket.window.prefix
    selected = []
    for layer in ticket.layers:
        query = by_layer[layer]
        if (
            not isinstance(query, QueryVectors)
            or query.vector_space != ticket.target_model_id
            or query.request_id != prefix.request_id
            or query.prefix_version != prefix.version
            or query.positional_encoding != ROPE_APPLIED
            or not isinstance(query.vectors, torch.Tensor)
            or query.head_start > ticket.head_start
            or query.head_start + query.head_count
            < ticket.head_start + ticket.head_count
            or query.vectors.ndim != 3
            or query.vectors.shape[2] != ticket.head_dim
        ):
            raise ProbeLaneProtocolError("target Q identity or head layout differs")
        valid = query.positions[: query.valid_length]
        if any(position not in valid for position in ticket.window.query_positions):
            raise ProbeLaneProtocolError("draft did not produce requested Q positions")
        start = ticket.head_start - query.head_start
        host_rows = []
        for position in ticket.window.query_positions:
            index = valid.index(position)
            host_rows.append(
                query.vectors[index, start : start + ticket.head_count]
                .detach()
                .to(device="cpu", dtype=torch.float32)
            )
        selected.append(
            QueryVectors(
                ticket.target_model_id,
                f"{prefix.version}:probe:{ticket.nonce}",
                layer,
                ticket.head_start,
                ticket.head_count,
                ticket.window.query_positions,
                len(ticket.window.query_positions),
                torch.stack(host_rows).contiguous(),
                prefix_version=prefix.version,
                positional_encoding=ROPE_APPLIED,
                request_id=prefix.request_id,
            )
        )
    return ProbeLaneReply(
        ticket.nonce,
        ticket.window.operation_id,
        ticket.prefix_digest,
        ticket.weights_sha256,
        ticket.tokenizer_sha256,
        tuple(selected),
    )


class ProbeLaneCUDAHandler:
    """One owner thread, one private target/draft pipeline and CUDA context."""

    def __init__(
        self,
        pipeline: CUDAPredictionPipeline,
        *,
        weights_sha256: str,
        tokenizer_sha256: str,
    ):
        if not isinstance(pipeline, CUDAPredictionPipeline):
            raise TypeError("private CUDA prediction pipeline required")
        if not all(
            isinstance(value, str) and len(value) == 64
            for value in (weights_sha256, tokenizer_sha256)
        ):
            raise ValueError("exact model and tokenizer digests required")
        self.pipeline = pipeline
        self.weights_sha256 = weights_sha256
        self.tokenizer_sha256 = tokenizer_sha256
        self.owner_thread = threading.get_ident()
        self.device = torch.device(pipeline.probe.device)
        self._cached_req = None
        self._cached_incarnation = None
        self._cached_entry_transfer_id = None
        self._cached_used_at = None
        self._cached_idle_seconds = _DEFAULT_CACHE_IDLE_SECONDS

    def retire_idle_cache(self, *, max_idle_seconds: float | None = None) -> None:
        if threading.get_ident() != self.owner_thread:
            raise ProbeLaneProtocolError("CUDA sidecar handler changed owner thread")
        idle_seconds = (
            getattr(self, "_cached_idle_seconds", _DEFAULT_CACHE_IDLE_SECONDS)
            if max_idle_seconds is None
            else max_idle_seconds
        )
        if (
            self._cached_incarnation is not None
            and self._cached_used_at is not None
            and time.monotonic() - self._cached_used_at >= idle_seconds
        ):
            self.close()

    def close(self) -> None:
        if threading.get_ident() != self.owner_thread:
            raise ProbeLaneProtocolError("CUDA sidecar handler changed owner thread")
        if self._cached_incarnation is not None:
            provider = self.pipeline.provider
            if provider.factory.prefix_cache_enabled:
                provider.retire_sidecar_cache()
            if self._cached_req is not None:
                self.pipeline.probe.retire_cached_request(self._cached_req)
            self._cached_req = None
            self._cached_incarnation = None
            self._cached_entry_transfer_id = None
            self._cached_used_at = None
            self._cached_idle_seconds = _DEFAULT_CACHE_IDLE_SECONDS

    def _prepare_cache(self, window) -> None:
        provider = self.pipeline.provider
        if (
            self.pipeline.probe.prefix_budget is None
            and not provider.factory.prefix_cache_enabled
        ):
            return
        identity = (window.prefix.request_id, window.incarnation)
        same_cache = (
            self._cached_incarnation == identity
            and getattr(self, "_cached_entry_transfer_id", None)
            == window.entry_transfer_id
        )
        if not same_cache:
            self.close()
            if provider.factory.prefix_cache_enabled:
                provider.set_sidecar_cache_identity(*identity)
            if self.pipeline.probe.prefix_budget is not None:
                req = SimpleNamespace(rid=window.prefix.request_id)
                self.pipeline.probe.register_cached_request(req)
                self._cached_req = req
            self._cached_incarnation = identity
            self._cached_entry_transfer_id = window.entry_transfer_id
            same_cache = False
        early_prompt = ":prompt-only:" in window.prefix.version
        # The regular n=0 ticket appends P's first token to the early prompt
        # prefix. Keep that cache under its bounded lease until the first
        # positive target-token refresh actually consumes it.
        keep_early_lease = (
            same_cache
            and window.target_tokens == 0
            and getattr(self, "_cached_idle_seconds", _DEFAULT_CACHE_IDLE_SECONDS)
            == _EARLY_PROMPT_CACHE_IDLE_SECONDS
        )
        self._cached_idle_seconds = (
            _EARLY_PROMPT_CACHE_IDLE_SECONDS
            if early_prompt or keep_early_lease
            else _DEFAULT_CACHE_IDLE_SECONDS
        )
        self._cached_used_at = time.monotonic()

    def __call__(self, ticket: ProbeLaneTicket) -> ProbeLaneReply:
        if threading.get_ident() != self.owner_thread:
            raise ProbeLaneProtocolError("CUDA sidecar handler changed owner thread")
        if (
            not isinstance(ticket, ProbeLaneTicket)
            or ticket.target_model_id != self.pipeline.probe_config.target_model_id
            or ticket.weights_sha256 != self.weights_sha256
            or ticket.tokenizer_sha256 != self.tokenizer_sha256
            or any(
                layer not in self.pipeline.probe_config.layers
                for layer in ticket.layers
            )
            or ticket.head_start < self.pipeline.probe_config.head_start
            or ticket.head_start + ticket.head_count
            > self.pipeline.probe_config.head_start
            + self.pipeline.probe_config.head_count
            or time.monotonic() >= ticket.deadline_monotonic
        ):
            raise ProbeLaneProtocolError("CUDA sidecar ticket/model identity differs")
        window = ticket.window
        self._prepare_cache(window)
        branch = (
            self.pipeline.committed_query_branch(window.prefix, window.query_positions)
            if window.query_source == "committed"
            else self.pipeline.query_branch(window.prefix)
        )
        with branch as queries:
            try:
                reply = _materialize_reply(ticket, queries)
                torch.cuda.synchronize(self.device)
            except BaseException:
                try:
                    torch.cuda.synchronize(self.device)
                except BaseException:
                    self.pipeline.probe.quarantine_query_copy()
                raise
        # No branch-owned GPU tensor is reachable from the returned reply.
        verify_reply(ticket, reply)
        # The idle TTL starts after the CUDA work finishes. A long first
        # prefill can exceed the TTL while the handler is still busy.
        self._cached_used_at = time.monotonic()
        return reply
