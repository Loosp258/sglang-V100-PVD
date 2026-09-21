"""Reusable real CPU batched sparse forward, deliberately not a serving backend.

Caller owns allocated destination rows and their request mappings. Execution
does not allocate, sample, commit output, free KV, or infer native completion.
Supports only the exact offline sparse backend's CPU FP32 TP1 Llama subset.
"""

from dataclasses import dataclass

import torch
from sglang.srt.disaggregation.pvd.cpu_batch_dispatch import (
    CPUBatchDispatcher,
    batch_results_from_logits,
)
from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import (
    CPUDecodeLifecycle,
    LifecycleError,
)
from sglang.srt.disaggregation.pvd.draft_forward_adapter import DraftForwardAdapter
from sglang.srt.disaggregation.pvd.draft_runner_sglang import DraftForwardInputs
from sglang.srt.disaggregation.pvd.sparse_cpu_backend import (
    SparseDecodeBinding,
    make_offline_sparse_backend,
)
from sglang.srt.disaggregation.pvd.sparse_install import CPUInstalledPromptView


@dataclass(frozen=True)
class CPUForwardDestination:
    lifecycle: CPUDecodeLifecycle
    slot: int
    kv_row: int


def bind_batch_destinations(dispatcher, ticket, destinations):
    """Validate immutable dispatch-to-storage mapping before any model writes."""
    dispatcher._match(ticket)
    destinations = tuple(destinations)
    expected = {m.request_id: m.permit for m in ticket.members}
    rows, slots, by_id = set(), set(), {}
    for dest in destinations:
        if not isinstance(dest, CPUForwardDestination) or not isinstance(
            dest.lifecycle, CPUDecodeLifecycle
        ):
            raise LifecycleError("explicit CPU forward destinations required")
        life = dest.lifecycle
        if (
            life.request_id not in expected
            or life.request_id in by_id
            or life._permit is not expected[life.request_id]
            or life._batch_owner is not dispatcher
            or life.state != "running"
        ):
            raise LifecycleError("stale, cancelled or foreign destination owner")
        if (
            type(dest.slot) is not int
            or dest.slot <= 0
            or dest.slot in slots
            or type(dest.kv_row) is not int
            or dest.kv_row <= 0
            or dest.kv_row in rows
        ):
            raise LifecycleError("unique positive request slots and KV rows required")
        rows.add(dest.kv_row)
        slots.add(dest.slot)
        by_id[life.request_id] = dest
    if set(by_id) != set(expected):
        raise LifecycleError("destinations must cover every dispatched member")
    return tuple(by_id[m.request_id] for m in ticket.members)


class CPUBatchForwardExecutor:
    def __init__(self, runner, dispatcher):
        if not isinstance(dispatcher, CPUBatchDispatcher):
            raise LifecycleError("explicit batch dispatcher required")
        dispatcher.arbiter.owner()
        self.runner, self.dispatcher = runner, dispatcher
        self.native = runner.attn_backend
        self.backend = make_offline_sparse_backend(runner)
        config = runner.model.config
        kv_bytes = (
            config.num_hidden_layers
            * config.num_key_value_heads
            * runner.model_config.head_dim
            * 2
            * 4
        )
        self.builder = DraftForwardAdapter(
            runner,
            architecture="LlamaForCausalLM",
            attention_backend="torch_native",
            bytes_per_token=kv_bytes,
            device="cpu",
        )
        # Builder is used for fields only, never for admission or its KV budget.
        self._last_operation = None
        self._storage = {}
        self.forward_count = 0

    def register_storage(self, lifecycle, slot):
        """Bind a caller-owned request slot once; does not allocate or free it."""
        self.dispatcher.arbiter.owner()
        if (
            not isinstance(lifecycle, CPUDecodeLifecycle)
            or lifecycle.arbiter is not self.dispatcher.arbiter
            or lifecycle._permit is not None
            or lifecycle.state not in ("waiting", "running")
            or lifecycle in self._storage
            or type(slot) is not int
            or not 0 < slot < self.runner.req_to_token_pool.req_to_token.shape[0]
            or slot in self._storage.values()
        ):
            raise LifecycleError(
                "request storage must be uniquely bound before execution"
            )
        self._storage[lifecycle] = slot

    def unregister_storage(self, lifecycle):
        self.dispatcher.arbiter.owner()
        if lifecycle._permit is not None or lifecycle.state not in (
            "finished",
            "aborted",
        ):
            raise LifecycleError(
                "terminal drained request required before slot unbinding"
            )
        self._storage.pop(lifecycle, None)

    def forward(self, ticket, destinations):
        """Return ALL dispatch-order logits rows; retain lease until completion.

        Refuses replay. On any execution/validation failure, caller must report
        dispatcher.fail() AFTER this synchronous method unwinds, then retire
        its generated rows. Restoring the backend is not a KV rollback.
        """
        ordered = bind_batch_destinations(self.dispatcher, ticket, destinations)
        if any(self._storage.get(d.lifecycle) != d.slot for d in ordered):
            raise LifecycleError(
                "destination slot differs from registered request ownership"
            )
        if self._last_operation == ticket.operation_id:
            raise LifecycleError("batch forward cannot execute the same dispatch twice")
        if self.runner.attn_backend is not self.native:
            raise LifecycleError(
                "target backend changed or another execution is active"
            )
        inputs = DraftForwardInputs(
            "decode",
            tuple(m.permit.input_token for m in ticket.members),
            tuple(m.permit.query_position for m in ticket.members),
            tuple(m.permit.query_position + 1 for m in ticket.members),
            tuple(d.slot for d in ordered),
            tuple(d.kv_row for d in ordered),
        )
        bindings = [
            SparseDecodeBinding(
                d.slot,
                d.lifecycle.request_id,
                d.lifecycle.controller.group.coordinator.identity[1],
                m.permit.query_position,
                CPUInstalledPromptView(
                    d.lifecycle.controller.group, m.permit.committed_tokens
                ),
            )
            for m, d in zip(ticket.members, ordered, strict=True)
        ]
        batch = self.builder.build_forward_batch(inputs)
        self._last_operation = ticket.operation_id
        self.runner.attn_backend = self.backend
        try:
            with self.backend.consumer.bind(bindings), torch.inference_mode():
                self.forward_count += 1
                output = self.runner.forward(batch)
            logits = output.logits_output.next_token_logits
            # Validate complete rows, not the draft adapter's last-row shortcut.
            batch_results_from_logits(ticket, logits, finished=(False,) * len(ordered))
            return logits
        finally:
            self.runner.attn_backend = self.native
