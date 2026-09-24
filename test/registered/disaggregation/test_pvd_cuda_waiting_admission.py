"""Scheduler-owner admission ordering with a real receiver and fake GPU factory."""

from types import SimpleNamespace as NS

import pytest
from sglang.srt.disaggregation.pvd import cuda_waiting_admission as module
from sglang.srt.disaggregation.pvd.cuda_routed_request import (
    CUDARoutedRequestAssembly,
)
from test_pvd_cuda_request_admission import close_driver, setup_admission


def resources():
    return module.CUDAWaitingAdmissionResources(
        pipeline=object(),
        head_mapping=object(),
        vector_space="model",
        metric="inner_product",
        top_k=10,
        max_union_tokens=20,
        max_head_dim=8,
        bank_budget=object(),
        staging_budget=object(),
        copy_budget=object(),
        aggregate_budget=object(),
        execution_lock=object(),
        lead_tokens=2,
        timeout_seconds=4.0,
        max_pending_events=8,
        max_pending_bytes=1024,
        poll_interval_seconds=0.01,
    )


def setup(monkeypatch):
    context, binding, driver = setup_admission(monkeypatch)
    context.manager.scheduler.disagg_decode_prealloc_queue = NS(
        kv_manager=context.manager
    )
    return context, binding, driver


def test_received_route_builds_pending_group_then_claims_clamped_request(monkeypatch):
    context, selected, driver = setup(monkeypatch)
    events = []
    prepared = NS(group=NS(close=lambda: events.append("close")), importer=object())
    assembly = CUDARoutedRequestAssembly(NS(_refresh_driver_claimed=False), {0: "V"})
    monkeypatch.setattr(
        module,
        "create_received_prompt_group",
        lambda session, **kwargs: events.append(("group", session, kwargs)) or prepared,
    )
    context.manager.assemble_selected_cuda_request = lambda req, route, **kwargs: (
        events.append(("assembly", req, route, kwargs)) or assembly
    )
    monkeypatch.setattr(
        module,
        "admit_received_cuda_request",
        lambda preflight, **kwargs: (
            events.append(("admit", preflight, kwargs)) or "claimed"
        ),
    )
    coordinator = module.CUDAWaitingAdmissionCoordinator(
        context.manager, driver, context.c.owner, lambda _: resources()
    )
    try:
        assert coordinator.admit(context.request, selected) == "claimed"
        assert [event[0] for event in events] == ["group", "assembly", "admit"]
        assert events[0][1] is context.session
        assert events[0][2]["max_union_tokens"] == 4
        assert events[1][3]["top_k"] == 4
        assert events[1][3]["max_union_tokens"] == 4
        assert events[1][3]["initial_import_pending"] is True
        assert events[2][2]["controller"] is assembly.controller
        assert events[2][2]["clients"] is assembly.clients
        assert events[2][2]["importer"] is prepared.importer
        assert events[2][2]["pool_owner"] is context.c.owner
    finally:
        close_driver(driver)
        context.group.close()


@pytest.mark.parametrize("claimed", [False, True])
def test_failed_admission_only_discards_unclaimed_assembly(monkeypatch, claimed):
    context, selected, driver = setup(monkeypatch)
    events = []
    prepared = NS(
        group=NS(close=lambda: events.append("group-close")), importer=object()
    )
    assembly = CUDARoutedRequestAssembly(NS(_refresh_driver_claimed=False), {})

    async def discard(self):
        assert self is assembly
        events.append("discard")

    def fail(preflight, **kwargs):
        assembly.controller._refresh_driver_claimed = claimed
        raise RuntimeError("admission failed")

    monkeypatch.setattr(CUDARoutedRequestAssembly, "discard_unstarted", discard)
    monkeypatch.setattr(
        module, "create_received_prompt_group", lambda *a, **k: prepared
    )
    context.manager.assemble_selected_cuda_request = lambda *a, **k: assembly
    monkeypatch.setattr(module, "admit_received_cuda_request", fail)
    coordinator = module.CUDAWaitingAdmissionCoordinator(
        context.manager, driver, context.c.owner, lambda _: resources()
    )
    try:
        with pytest.raises(RuntimeError, match="admission failed"):
            coordinator.admit(context.request, selected)
        assert events == ([] if claimed else ["discard"])
    finally:
        close_driver(driver)
        context.group.close()


def test_assembly_failure_closes_unclaimed_group(monkeypatch):
    context, selected, driver = setup(monkeypatch)
    closed = []
    monkeypatch.setattr(
        module,
        "create_received_prompt_group",
        lambda *a, **k: NS(group=NS(close=lambda: closed.append(True))),
    )
    context.manager.assemble_selected_cuda_request = lambda *a, **k: (
        _ for _ in ()
    ).throw(RuntimeError("assembly failed"))
    coordinator = module.CUDAWaitingAdmissionCoordinator(
        context.manager, driver, context.c.owner, lambda _: resources()
    )
    try:
        with pytest.raises(RuntimeError, match="assembly failed"):
            coordinator.admit(context.request, selected)
        assert closed == [True]
    finally:
        close_driver(driver)
        context.group.close()


def test_quarantined_partial_assembly_keeps_its_group(monkeypatch):
    context, selected, driver = setup(monkeypatch)
    closed = []
    prepared = NS(group=NS(close=lambda: closed.append(True)))
    monkeypatch.setattr(
        module, "create_received_prompt_group", lambda *a, **k: prepared
    )
    monkeypatch.setattr(
        module, "partial_assembly_quarantined", lambda group: group is prepared.group
    )
    context.manager.assemble_selected_cuda_request = lambda *a, **k: (
        _ for _ in ()
    ).throw(RuntimeError("partial cleanup unknown"))
    coordinator = module.CUDAWaitingAdmissionCoordinator(
        context.manager, driver, context.c.owner, lambda _: resources()
    )
    try:
        with pytest.raises(RuntimeError, match="partial cleanup unknown"):
            coordinator.admit(context.request, selected)
        assert not closed
        assert module._UNCLAIMED_GROUP_QUARANTINE[-1] == (prepared, None)
    finally:
        module._UNCLAIMED_GROUP_QUARANTINE.pop()
        close_driver(driver)
        context.group.close()


def test_request_must_still_be_in_final_waiting_queue(monkeypatch):
    context, selected, driver = setup(monkeypatch)
    context.manager.scheduler.waiting_queue.clear()
    calls = []
    coordinator = module.CUDAWaitingAdmissionCoordinator(
        context.manager, driver, context.c.owner, lambda _: calls.append(True)
    )
    try:
        with pytest.raises(ValueError, match="final Scheduler waiting queue"):
            coordinator.admit(context.request, selected)
        assert calls == []
    finally:
        close_driver(driver)
        context.group.close()


def test_equal_but_foreign_waiting_request_cannot_authorize_admission(monkeypatch):
    context, selected, driver = setup(monkeypatch)

    class EqualRequest:
        def __eq__(self, other):
            return True

    context.manager.scheduler.waiting_queue[:] = [EqualRequest()]
    coordinator = module.CUDAWaitingAdmissionCoordinator(
        context.manager, driver, context.c.owner, lambda _: resources()
    )
    try:
        with pytest.raises(ValueError, match="final Scheduler waiting queue"):
            coordinator.admit(context.request, selected)
    finally:
        close_driver(driver)
        context.group.close()
