"""Admission preflight fails before allocating a group or remote destination."""

from types import SimpleNamespace as NS

import pytest
from sglang.srt.disaggregation.pvd.client import (
    PVDSelectedShardRoute,
    PVDSelectedShardRoutes,
)
from sglang.srt.disaggregation.pvd.conn import PVDSelectedRouteBinding
from sglang.srt.disaggregation.pvd.cuda_refresh_driver import CUDARefreshDriver
from sglang.srt.disaggregation.pvd.cuda_request_admission import (
    CUDAAdmissionBlocked,
    admit_received_cuda_request,
    bounds_for_received_cuda_prompt,
    preflight_received_cuda_admission,
)
from test_pvd_cuda_received_prompt import received


def setup_admission(monkeypatch):
    context = received(monkeypatch)
    context.manager.vector_group_for = lambda req: "chosen"
    context.request.pvd_vector_group_id = "chosen"
    selected = PVDSelectedShardRoutes(
        NS(key=context.key),
        (
            PVDSelectedShardRoute(0, "http://v0", "epoch0", "mlx5_0"),
            PVDSelectedShardRoute(1, "http://v1", "epoch1", "mlx5_1"),
        ),
    )
    binding = PVDSelectedRouteBinding(
        context.manager,
        context.request,
        context.request.rid,
        context.key,
        "chosen",
        context.request.pvd_delivery_id,
        selected,
    )
    driver = CUDARefreshDriver(context.arbiter, max_requests=2, max_prefix_tokens=32)
    return context, binding, driver


def close_driver(driver):
    driver.begin_shutdown()
    driver.close_loop()


def test_valid_receiver_and_selected_routes_produce_read_only_preflight(monkeypatch):
    context, binding, driver = setup_admission(monkeypatch)
    try:
        plan = preflight_received_cuda_admission(context.session, binding, driver)
        assert plan.receipt is context.session.require_initial_prompt()
        assert plan.revalidate().receipt is plan.receipt
        assert not driver._records and not driver.arbiter.busy
        assert context.session._cuda_refresh_driver is None
        assert getattr(context.session, "_cuda_prompt_importer", None) is None
        assert not context.group.can_decode(0)
    finally:
        close_driver(driver)
        context.group.close()


@pytest.mark.parametrize(
    "top_k,max_union_tokens,expected",
    [(1, 1, (1, 1)), (2, 3, (2, 3)), (10, 70, (4, 4))],
)
def test_short_prompt_clamps_both_retrieval_limits(
    monkeypatch, top_k, max_union_tokens, expected
):
    context, binding, driver = setup_admission(monkeypatch)
    try:
        preflight = preflight_received_cuda_admission(context.session, binding, driver)
        limits = bounds_for_received_cuda_prompt(
            preflight, top_k=top_k, max_union_tokens=max_union_tokens
        )
        assert (limits.top_k, limits.max_union_tokens) == expected
        assert not driver._records
    finally:
        close_driver(driver)
        context.group.close()


@pytest.mark.parametrize(
    "top_k,max_union_tokens", [(True, 10), (0, 10), (513, 513), (10, 9)]
)
def test_retrieval_limits_refuse_invalid_config_before_mutation(
    monkeypatch, top_k, max_union_tokens
):
    context, binding, driver = setup_admission(monkeypatch)
    try:
        preflight = preflight_received_cuda_admission(context.session, binding, driver)
        with pytest.raises(CUDAAdmissionBlocked, match="ordered retrieval limits"):
            bounds_for_received_cuda_prompt(
                preflight, top_k=top_k, max_union_tokens=max_union_tokens
            )
        assert not driver._records
    finally:
        close_driver(driver)
        context.group.close()


def test_missing_prepared_resources_refused_before_mutation_or_import(monkeypatch):
    context, binding, driver = setup_admission(monkeypatch)
    try:
        plan = preflight_received_cuda_admission(context.session, binding, driver)
        with pytest.raises(CUDAAdmissionBlocked, match="exact prepared controller"):
            admit_received_cuda_request(plan)
        assert not driver._records and not driver.arbiter.busy
        assert getattr(context.session, "_cuda_prompt_importer", None) is None
        assert not context.importer._used
        assert not context.group.can_decode(0)
    finally:
        close_driver(driver)
        context.group.close()


@pytest.mark.parametrize("fault", ["entry", "group", "delivery", "rank", "url"])
def test_stale_gateway_route_refused_before_mutation(monkeypatch, fault):
    context, binding, driver = setup_admission(monkeypatch)
    selected = binding.selected
    if fault == "entry":
        selected = PVDSelectedShardRoutes(NS(key=object()), selected.shards)
    elif fault == "rank":
        selected = PVDSelectedShardRoutes(
            selected.manifest,
            (
                selected.shards[0],
                PVDSelectedShardRoute(0, "http://v1", "epoch1", "mlx5_1"),
            ),
        )
    elif fault == "url":
        selected = PVDSelectedShardRoutes(
            selected.manifest,
            (
                PVDSelectedShardRoute(0, "not-http", "epoch0", "mlx5_0"),
                selected.shards[1],
            ),
        )
    elif fault == "group":
        binding = PVDSelectedRouteBinding(
            binding.manager,
            binding.req,
            binding.rid,
            binding.key,
            "other",
            binding.delivery_id,
            selected,
        )
    elif fault == "delivery":
        binding = PVDSelectedRouteBinding(
            binding.manager,
            binding.req,
            binding.rid,
            binding.key,
            binding.group_id,
            "other",
            selected,
        )
    if fault in ("entry", "rank", "url"):
        binding = PVDSelectedRouteBinding(
            binding.manager,
            binding.req,
            binding.rid,
            binding.key,
            binding.group_id,
            binding.delivery_id,
            selected,
        )
    try:
        with pytest.raises(CUDAAdmissionBlocked, match="selected V routes"):
            preflight_received_cuda_admission(context.session, binding, driver)
        assert not driver._records and not driver.arbiter.busy
    finally:
        close_driver(driver)
        context.group.close()


def test_receipt_or_owner_change_invalidates_existing_preflight(monkeypatch):
    context, binding, driver = setup_admission(monkeypatch)
    try:
        plan = preflight_received_cuda_admission(context.session, binding, driver)
        context.request.output_ids.append(6)
        with pytest.raises(RuntimeError, match="initial Prompt"):
            admit_received_cuda_request(plan)
        assert not driver._records
    finally:
        close_driver(driver)
        context.group.close()


def test_busy_driver_cannot_issue_preflight(monkeypatch):
    context, binding, driver = setup_admission(monkeypatch)
    lease = driver.arbiter.acquire()
    try:
        with pytest.raises(CUDAAdmissionBlocked, match="owner is busy"):
            preflight_received_cuda_admission(context.session, binding, driver)
        assert not driver._records
    finally:
        driver.arbiter.release(lease)
        close_driver(driver)
        context.group.close()


def test_duplicate_request_registration_refused_by_preflight(monkeypatch):
    context, binding, driver = setup_admission(monkeypatch)
    driver._records[context.request.rid] = NS(slot=context.request.req_pool_idx)
    try:
        with pytest.raises(CUDAAdmissionBlocked, match="duplicate"):
            preflight_received_cuda_admission(context.session, binding, driver)
    finally:
        driver._records.clear()
        close_driver(driver)
        context.group.close()


def test_preflight_refuses_receiver_replaced_in_manager(monkeypatch):
    context, binding, driver = setup_admission(monkeypatch)
    context.manager.decode_sessions[context.key] = object()
    try:
        with pytest.raises(RuntimeError, match="initial Prompt"):
            preflight_received_cuda_admission(context.session, binding, driver)
        assert not driver._records and not driver.arbiter.busy
    finally:
        close_driver(driver)
        context.group.close()
