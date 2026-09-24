"""Explicit Scheduler-owner admission of one routed, received CUDA Prompt.

Startup supplies request-specific model resources through ``prepare``. Route
discovery and the ordinary full-Prompt receiver have already completed before
this coordinator allocates a sparse bank or claims the request in the driver.
"""

from dataclasses import dataclass

from sglang.srt.disaggregation.pvd.cuda_prompt_group_factory import (
    create_received_prompt_group,
)
from sglang.srt.disaggregation.pvd.cuda_request_admission import (
    admit_received_cuda_request,
    bounds_for_received_cuda_prompt,
    preflight_received_cuda_admission,
)
from sglang.srt.disaggregation.pvd.cuda_routed_request import (
    CUDARoutedRequestAssembly,
    partial_assembly_quarantined,
)
from sglang.srt.disaggregation.pvd.decode_refresh import PVDDecodeSession

_UNCLAIMED_GROUP_QUARANTINE = []


@dataclass(frozen=True)
class CUDAWaitingAdmissionResources:
    """Per-request inputs supplied by an explicitly installed startup factory."""

    pipeline: object
    head_mapping: object
    vector_space: str
    metric: str
    top_k: int
    max_union_tokens: int
    max_head_dim: int
    bank_budget: object
    staging_budget: object
    copy_budget: object
    aggregate_budget: object
    execution_lock: object
    lead_tokens: int
    timeout_seconds: float
    max_pending_events: int
    max_pending_bytes: int
    poll_interval_seconds: float
    release: object = None


class CUDAWaitingAdmissionCoordinator:
    """Consume a selected route after the Req reaches its final waiting queue.

    This synchronous owner path only supports the private loop used by the
    ordinary Scheduler. An existing async loop would require an async serving
    integration with equally explicit ownership and retirement semantics.
    """

    def __init__(self, manager, driver, pool_owner, prepare):
        if not callable(prepare) or not driver._owns_loop or driver._loop.is_running():
            raise ValueError(
                "synchronous CUDA waiting admission requires a private driver loop and factory"
            )
        driver._owner()
        self.manager, self.driver = manager, driver
        self.pool_owner, self.prepare = pool_owner, prepare

    def admit(self, req, selected_binding):
        self.driver._owner()
        if (
            not any(queued is req for queued in self.manager.scheduler.waiting_queue)
            or self.manager.scheduler.disagg_decode_prealloc_queue.kv_manager
            is not self.manager
        ):
            raise ValueError(
                "CUDA admission requires the final Scheduler waiting queue"
            )
        session = self.manager.decode_sessions.get(self.manager.key_for(req))
        if not isinstance(session, PVDDecodeSession) or session.req is not req:
            raise ValueError("exact completed full-Prompt receiver session required")
        preflight = preflight_received_cuda_admission(
            session, selected_binding, self.driver
        )
        resources = self.prepare(preflight)
        if not isinstance(resources, CUDAWaitingAdmissionResources):
            raise TypeError("explicit CUDA waiting admission resources required")
        bounds = bounds_for_received_cuda_prompt(
            preflight,
            top_k=resources.top_k,
            max_union_tokens=resources.max_union_tokens,
        )
        preflight.revalidate()
        initial = create_received_prompt_group(
            session,
            bank_budget=resources.bank_budget,
            staging_budget=resources.staging_budget,
            execution_lock=resources.execution_lock,
            max_union_tokens=bounds.max_union_tokens,
            lead_tokens=resources.lead_tokens,
            timeout_seconds=resources.timeout_seconds,
            max_pending_events=resources.max_pending_events,
            max_pending_bytes=resources.max_pending_bytes,
        )
        assembly = None
        try:
            assembly = self.manager.assemble_selected_cuda_request(
                req,
                selected_binding,
                group=initial.group,
                pipeline=resources.pipeline,
                head_mapping=resources.head_mapping,
                vector_space=resources.vector_space,
                metric=resources.metric,
                top_k=bounds.top_k,
                max_union_tokens=bounds.max_union_tokens,
                max_head_dim=resources.max_head_dim,
                copy_budget=resources.copy_budget,
                aggregate_budget=resources.aggregate_budget,
                poll_interval_seconds=resources.poll_interval_seconds,
                initial_import_pending=True,
            )
            if not isinstance(assembly, CUDARoutedRequestAssembly):
                raise TypeError("selected CUDA assembly must own its routed clients")
            preflight.revalidate()
            return admit_received_cuda_request(
                preflight,
                controller=assembly.controller,
                clients=assembly.clients,
                importer=initial.importer,
                pool_owner=self.pool_owner,
                timeout_seconds=resources.timeout_seconds,
                release=resources.release,
            )
        except BaseException:
            if assembly is not None and isinstance(assembly, CUDARoutedRequestAssembly):
                if not getattr(assembly.controller, "_refresh_driver_claimed", False):
                    self.driver._loop.run_until_complete(assembly.discard_unstarted())
                # The driver owns every claimed controller, including provisional
                # records that must be quarantined rather than directly closed.
            else:
                if assembly is not None or partial_assembly_quarantined(initial.group):
                    _UNCLAIMED_GROUP_QUARANTINE.append((initial, assembly))
                    raise
                try:
                    initial.group.close()
                except BaseException:
                    _UNCLAIMED_GROUP_QUARANTINE.append(initial)
                    raise
            raise
