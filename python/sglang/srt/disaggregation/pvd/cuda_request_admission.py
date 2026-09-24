"""Read-only admission preflight for a received CUDA Prompt request.

This is intentionally not a serving switch. Provisional driver registration
now exists, but admission still needs a scheduler-owned transaction that
registers the driver, attaches deferred Req/KV retirement, imports the full
Prompt, and claims the receiver before making the Req runnable.

No group, HTTP client, GPU bank, or native destination is allocated here. The
transaction must use the driver's ordered close/quarantine path on failure;
an unclaimed receiver must never use ordinary scheduler release.
"""

from dataclasses import dataclass

from sglang.srt.disaggregation.pvd.client import PVDSelectedShardRoutes
from sglang.srt.disaggregation.pvd.conn import PVDSelectedRouteBinding
from sglang.srt.disaggregation.pvd.cuda_refresh_driver import CUDARefreshDriver
from sglang.srt.disaggregation.pvd.decode_refresh import (
    InitialPromptReceipt,
    PVDDecodeSession,
)


class CUDAAdmissionBlocked(RuntimeError):
    """Refusal before any resource is created or ownership is transferred."""


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


def admit_received_cuda_request(preflight):
    """Refuse the currently unsupported transaction before its first mutation.

    The provisional driver seam exists, but this function must not claim to
    admit a request until the Scheduler constructs all exact CUDA resources
    and owns the register/retirement/import/claim transaction.
    """
    if not isinstance(preflight, CUDAAdmissionPreflight):
        raise CUDAAdmissionBlocked("validated CUDA admission preflight required")
    preflight.revalidate()
    raise CUDAAdmissionBlocked(
        "CUDA admission requires the scheduler-owned provisional transaction: "
        "register, attach deferred release, import Prompt, then claim receiver"
    )
