"""Read-only admission preflight for a received CUDA Prompt request.

This is intentionally not a serving switch. The requested transaction order is
currently impossible: ``CUDARefreshDriver.register`` requires ``can_decode(0)``,
which becomes true only after ``CUDAPromptBootstrap.install_received``. Moving
the install before registration is not a safe workaround: a failed registration
would leave an installed bank without a driver or a deferred allocator owner.

No group, HTTP client, GPU bank, or native destination is allocated here. A
future provisional driver registration must also provide an explicit abort that
drains the controller and the full-Prompt receiver before releasing either MR.
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

    Required upstream seam: reserve a provisional driver record without an
    installed bank, prevent driver polling/decode until final claim, and expose
    an abort that *successfully fences both source and destination* or retains
    every owner and quarantines the worker on UNKNOWN. Current ``cancel()``
    closes an unclaimed controller without closing the full-Prompt receiver;
    treating it as rollback would permit stale RDMA against freed GPU memory.
    """
    if not isinstance(preflight, CUDAAdmissionPreflight):
        raise CUDAAdmissionBlocked("validated CUDA admission preflight required")
    preflight.revalidate()
    raise CUDAAdmissionBlocked(
        "CUDA admission requires a provisional driver register/abort API: "
        "register currently requires the initial Prompt bank installed, "
        "but safe release binding must precede that installation"
    )
