"""Received CUDA Prompt admission, with a read-only preflight and owned handoff.

This is intentionally not a serving switch. The caller must prepare the exact
controller, clients, importer and pool owner, then invoke the transaction on
the scheduler owner thread while the Req remains in its final waiting queue.

The preflight allocates nothing. The transaction does not create a group or
HTTP client: before registration the caller still owns those resources. After
registration, it uses the driver's ordered close/quarantine path on failure;
an unclaimed receiver must never use ordinary scheduler release.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from sglang.srt.disaggregation.pvd.client import PVDSelectedShardRoutes
from sglang.srt.disaggregation.pvd.conn import PVDSelectedRouteBinding
from sglang.srt.disaggregation.pvd.cuda_model_attention import CUDAModelPools
from sglang.srt.disaggregation.pvd.cuda_prefetch_request import CUDAPrefetchRequest
from sglang.srt.disaggregation.pvd.cuda_prompt_bootstrap import CUDAPromptBootstrap
from sglang.srt.disaggregation.pvd.cuda_refresh_driver import CUDARefreshDriver
from sglang.srt.disaggregation.pvd.cuda_request_release import CUDARequestRelease
from sglang.srt.disaggregation.pvd.decode_refresh import (
    InitialPromptReceipt,
    PVDDecodeSession,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import ResourceGuard


class CUDAAdmissionBlocked(RuntimeError):
    """Refusal before any resource is created or ownership is transferred."""


@dataclass(frozen=True)
class CUDAReceivedRetrievalBounds:
    top_k: int
    max_union_tokens: int


@dataclass(frozen=True)
class CUDAAdmissionPreflight:
    session: PVDDecodeSession
    binding: PVDSelectedRouteBinding
    driver: CUDARefreshDriver
    receipt: InitialPromptReceipt

    def revalidate(self):
        """Reject changes between route discovery and a future admission step."""
        current = preflight_received_cuda_admission(
            self.session, self.binding, self.driver
        )
        if current.receipt is not self.receipt:
            raise CUDAAdmissionBlocked("initial Prompt receipt was replaced")
        return current


def preflight_received_cuda_admission(session, binding, driver):
    """Check exact receiver, selected V routes and idle admission owner.

    This is local control-plane evidence, not a pin or a transport fence. Call
    again immediately before any future provisional registration; the caller
    must not assume the returned snapshot keeps a waiting Req alive.
    """
    if (
        not isinstance(session, PVDDecodeSession)
        or not isinstance(binding, PVDSelectedRouteBinding)
        or not isinstance(driver, CUDARefreshDriver)
    ):
        raise CUDAAdmissionBlocked(
            "real receiver, route binding and CUDA driver required"
        )
    driver._owner()
    receipt = session.require_initial_prompt()
    manager, req = session.manager, session.req
    selected = binding.selected
    if (
        binding.manager is not manager
        or binding.req is not req
        or binding.rid != receipt.request_id
        or binding.key != receipt.key
        or binding.key != manager.key_for(req)
        or binding.group_id != manager.vector_group_for(req)
        or binding.delivery_id != req.pvd_delivery_id
        or not isinstance(selected, PVDSelectedShardRoutes)
        or selected.manifest.key != binding.key
        or not selected.shards
        or len({route.rank for route in selected.shards}) != len(selected.shards)
        or any(
            not route.url.startswith(("http://", "https://"))
            or not route.sender_epoch
            or not route.rail
            for route in selected.shards
        )
    ):
        raise CUDAAdmissionBlocked(
            "Gateway-selected V routes changed or are incomplete"
        )
    if (
        driver._closing
        or driver._source_quarantine is not None
        or driver._loop.is_closed()
        or driver.arbiter.busy
        or len(driver._records) >= driver.max_requests
        or req.rid in driver._records
        or any(record.slot == receipt.slot for record in driver._records.values())
        or getattr(session, "_cuda_prompt_importer", None) is not None
        or session._cuda_refresh_driver is not None
        or getattr(req, "pvd_cuda_kv_release", None) is not None
    ):
        raise CUDAAdmissionBlocked("CUDA admission owner is busy, duplicate or closing")
    return CUDAAdmissionPreflight(session, binding, driver, receipt)


def bounds_for_received_cuda_prompt(preflight, *, top_k, max_union_tokens):
    """Clamp configured search/workset limits to the actual Prompt length.

    The two explicit settings remain hard upper bounds. A one-token Prompt is
    legal: it gets Top-1 and a one-token bank rather than failing construction
    of the V search request or D initial bank.
    """
    if not isinstance(preflight, CUDAAdmissionPreflight):
        raise CUDAAdmissionBlocked("validated CUDA admission preflight required")
    receipt = preflight.revalidate().receipt
    if (
        type(top_k) is not int
        or not 1 <= top_k <= 512
        or type(max_union_tokens) is not int
        or max_union_tokens < top_k
    ):
        raise CUDAAdmissionBlocked("positive ordered retrieval limits required")
    count = len(receipt.prompt)
    if count <= 0:
        raise CUDAAdmissionBlocked("nonempty completed Prompt required")
    return CUDAReceivedRetrievalBounds(min(top_k, count), min(max_union_tokens, count))


def admit_received_cuda_request(
    preflight,
    *,
    controller=None,
    clients=None,
    importer=None,
    pool_owner=None,
    timeout_seconds=None,
    release=None,
):
    """Claim a prepared CUDA request in one owner-thread transaction.

    The caller owns the controller and its HTTP clients until registration
    succeeds. After registration, the driver owns them. An import/claim error
    schedules ordered driver drain; UNKNOWN poisons and retains all owners.
    The caller must keep the Req in its final waiting queue until this returns.
    This is not a serving startup factory or an automatic Scheduler switch.
    """
    if not isinstance(preflight, CUDAAdmissionPreflight):
        raise CUDAAdmissionBlocked("validated CUDA admission preflight required")
    preflight.revalidate()
    session, driver = preflight.session, preflight.driver
    cache = session.manager.scheduler.tree_cache
    if (
        not isinstance(controller, CUDAPrefetchRequest)
        or not isinstance(importer, CUDAPromptBootstrap)
        or not isinstance(pool_owner, ResourceGuard)
        or controller.group is not importer.group
        or controller.pipeline._lock is not importer._lock
        or not isinstance(clients, Mapping)
        or set(clients) != set(controller._routes)
        or not isinstance(pool_owner.value, CUDAModelPools)
        or pool_owner.value.req_pool is not cache.req_to_token_pool
        or pool_owner.value.kv_pool
        is not cache.token_to_kv_pool_allocator.get_kvcache()
        or type(timeout_seconds) not in (int, float)
        or timeout_seconds <= 0
    ):
        raise CUDAAdmissionBlocked(
            "exact prepared controller, importer, clients and pool owner required"
        )
    req = session.req
    driver.register(
        req,
        controller,
        clients=clients,
        timeout_seconds=timeout_seconds,
        initial_import_pending=True,
        initial_session=session,
        pool_owner=pool_owner,
    )
    try:
        retirement = CUDARequestRelease(
            req, driver, cache, pool_owner=pool_owner, release=release
        )
    except BaseException as exc:
        # Registration already redirected release_request() to the driver.
        # Without a deferred allocator owner there is no safe rollback.
        driver.quarantine_provisional(req, "release attachment failed: " + str(exc))
        raise
    try:
        importer.install_received(
            session, arbiter=driver.arbiter, pool_owner=pool_owner, cache=cache
        )
        driver.claim_received_session(session)
    except BaseException as exc:
        if (
            importer._quarantined
            or getattr(cache.req_to_token_pool, "pvd_cuda_retirement_error", None)
            is not None
            or getattr(
                cache.token_to_kv_pool_allocator,
                "pvd_cuda_retirement_error",
                None,
            )
            is not None
        ):
            driver.quarantine_provisional(req, "initial import unknown: " + str(exc))
        else:
            driver.cancel(req, "initial import or receiver claim failed")
        raise
    return retirement
