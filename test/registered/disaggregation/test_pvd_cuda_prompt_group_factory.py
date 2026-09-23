"""Receiver-derived TP1 bank metadata, not an actual CUDA Prompt import."""

import threading
from dataclasses import replace

import pytest
import torch
from sglang.srt.disaggregation.pvd import cuda_prompt_group_factory as module
from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import (
    CUDASparseReceiveRegistry,
)
from sglang.srt.disaggregation.pvd.sparse_install import InstallProtocolError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_cuda_received_prompt import received
from test_pvd_prompt_vectors import FakePool, storage_layout
from test_pvd_search_routing import layout


def receiver(monkeypatch):
    c = received(monkeypatch)
    storage = storage_layout(
        FakePool(layers=1, heads=2, dim=3, dtype=torch.float32),
        total_kv_heads=2,
        page_size=1,
    )
    compute = layout(storage, 1)
    c.manager.layout = lambda: compute
    c.session._initial_receipt = replace(
        c.session._initial_receipt, layout=compute.fingerprint
    )
    registry = object.__new__(CUDASparseReceiveRegistry)
    registry.device = torch.device("cpu")  # Metadata-only policy test.
    c.manager.sparse_receive_registry = registry
    c.manager.worker_epoch = "worker"
    return c, compute


def test_plan_comes_from_completed_receiver_not_route_json(monkeypatch):
    c, compute = receiver(monkeypatch)
    plan = module.plan_received_prompt_bank(c.session, max_union_tokens=2)
    receipt = c.session.require_initial_prompt()
    assert plan.receipt is receipt
    assert plan.identity == (
        c.request.rid,
        receipt.receiver_epoch,
        c.key.transfer_id,
        compute.fingerprint,
    )
    assert plan.expected_groups == ((0, 0), (0, 1))
    assert (plan.prompt_tokens, plan.head_dim, plan.interval) == (4, 3, 4)
    assert plan.dtype == torch.float32 and plan.peer_epoch == "worker"
    c.group.close()


@pytest.mark.parametrize(
    "fault",
    (
        "unclaimed",
        "tp2",
        "dtype",
        "device",
        "registry",
        "epoch",
        "union",
        "pages",
        "duplicate-page",
        "padding-page",
    ),
)
def test_factory_refuses_wrong_receiver_pool_or_bounds(monkeypatch, fault):
    c, compute = receiver(monkeypatch)
    if fault == "unclaimed":
        c.session._initial_receipt = None
    elif fault == "tp2":
        c.manager.layout = lambda: replace(compute, tp_size=2, kv_heads_per_rank=1)
        c.session._initial_receipt = replace(
            c.session._initial_receipt,
            layout=c.manager.layout().fingerprint,
        )
    elif fault == "dtype":
        c.c.pool.k = c.c.pool.k.to(torch.float16)
    elif fault == "device":
        c.manager.sparse_receive_registry.device = torch.device("cuda:0")
    elif fault == "registry":
        c.manager.sparse_receive_registry = None
    elif fault == "epoch":
        c.manager.worker_epoch = ""
    elif fault == "pages":
        c.session.pages = torch.empty(0, dtype=torch.int64)
        c.session._initial_receipt = replace(c.session._initial_receipt, pages=())
    elif fault == "duplicate-page":
        c.session.pages = torch.tensor([8, 8, 6, 1])
        c.session._initial_receipt = replace(
            c.session._initial_receipt, pages=(8, 8, 6, 1)
        )
    elif fault == "padding-page":
        c.session.pages = torch.tensor([8, 0, 6, 1])
        c.session._initial_receipt = replace(
            c.session._initial_receipt, pages=(8, 0, 6, 1)
        )
    with pytest.raises((InstallProtocolError, RuntimeError)):
        module.plan_received_prompt_bank(
            c.session, max_union_tokens=5 if fault == "union" else 2
        )
    c.group.close()


def test_creation_requires_real_cuda_but_does_not_install(monkeypatch):
    c, _ = receiver(monkeypatch)
    budgets = (TransferBudget(4096, 2), TransferBudget(4096, 2))
    kwargs = dict(
        bank_budget=budgets[0],
        staging_budget=budgets[1],
        execution_lock=threading.RLock(),
        max_union_tokens=2,
        lead_tokens=1,
        timeout_seconds=10,
        max_pending_events=8,
        max_pending_bytes=65536,
    )
    with pytest.raises(InstallProtocolError, match="CUDA placement"):
        module.create_received_prompt_group(c.session, **kwargs)

    plan = module.plan_received_prompt_bank(c.session, max_union_tokens=2)
    original = module.plan_received_prompt_bank
    monkeypatch.setattr(
        module,
        "plan_received_prompt_bank",
        lambda *a, **k: replace(original(*a, **k), device=torch.device("cuda:0")),
    )
    built = []

    class Bank:
        def __init__(self, **options):
            built.append(("bank", options))

    class Group:
        def __init__(self, banks, **options):
            built.append(("group", banks, options))

    class Importer:
        def __init__(self, group, **options):
            built.append(("importer", group, options))

    monkeypatch.setattr(module, "CUDASparseWorkingSet", Bank)
    monkeypatch.setattr(module, "CUDARuntimeInstallGroup", Group)
    monkeypatch.setattr(module, "CUDAPromptBootstrap", Importer)
    created = module.create_received_prompt_group(c.session, **kwargs)
    assert created.plan.identity == plan.identity
    assert built[0][1]["expected_groups"] == plan.expected_groups
    assert built[0][1]["request_id"] == c.request.rid
    assert built[1][1] == {0: created.bank}
    assert built[1][2]["peer_epochs"] == {0: "worker"}
    assert built[2][1] is created.group
    assert built[2][2]["execution_lock"] is kwargs["execution_lock"]
    assert all(b.snapshot()["reservations"] == 0 for b in budgets)
    c.group.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA device required")
def test_real_cuda_bank_and_group_construction_without_prompt_install(monkeypatch):
    c, _ = receiver(monkeypatch)
    c.c.pool.k = c.c.pool.k.to("cuda:0")
    c.c.pool.v = c.c.pool.v.to("cuda:0")
    c.manager.sparse_receive_registry.device = torch.device("cuda:0")
    budgets = (TransferBudget(4096, 2), TransferBudget(4096, 2))
    created = module.create_received_prompt_group(
        c.session,
        bank_budget=budgets[0],
        staging_budget=budgets[1],
        execution_lock=threading.RLock(),
        max_union_tokens=2,
        lead_tokens=1,
        timeout_seconds=10,
        max_pending_events=8,
        max_pending_bytes=65536,
    )
    assert created.bank.device == torch.device("cuda:0")
    assert created.group.coordinator.identity == created.plan.identity[:3]
    assert not created.group.can_decode(0)
    assert all(b.snapshot()["reservations"] == 0 for b in budgets)
    created.group.close()
    c.group.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA device required")
def test_real_cuda_imports_receipted_prompt_without_mutating_model_kv(monkeypatch):
    c, _ = receiver(monkeypatch)
    c.c.pool.k = c.c.pool.k.to("cuda:0")
    c.c.pool.v = c.c.pool.v.to("cuda:0")
    c.c.req.req_to_token = c.c.req.req_to_token.to("cuda:0")
    c.manager.sparse_receive_registry.device = torch.device("cuda:0")
    old_k, old_v = c.c.pool.k.clone(), c.c.pool.v.clone()
    budgets = (TransferBudget(4096, 2), TransferBudget(4096, 2))
    created = module.create_received_prompt_group(
        c.session,
        bank_budget=budgets[0],
        staging_budget=budgets[1],
        execution_lock=threading.RLock(),
        max_union_tokens=2,
        lead_tokens=1,
        timeout_seconds=10,
        max_pending_events=8,
        max_pending_bytes=65536,
    )
    epoch = created.importer.install_received(
        c.session, arbiter=c.arbiter, pool_owner=c.c.owner, cache=c.cache
    )
    assert epoch.target_tokens == 0 and created.group.can_decode(0)
    assert created.bank.snapshot()["current_boundary"] == 0
    torch.testing.assert_close(c.c.pool.k, old_k, atol=0, rtol=0)
    torch.testing.assert_close(c.c.pool.v, old_v, atol=0, rtol=0)
    assert budgets[1].snapshot()["reservations"] == 0
    created.group.close()
    assert budgets[0].snapshot()["reservations"] == 0
    c.group.close()
