"""Real CPU tensors/math with CUDA placement/fences replaced, not GPU evidence."""

import json
import math
import sys
import threading
import types
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.cuda_model_attention import (
    CUDADecodeBinding,
    CUDAModelPools,
    CUDAModelSparseConsumer,
    make_cuda_sparse_backend,
)
from sglang.srt.disaggregation.pvd.cuda_rank_install import CUDARankInstallParticipant
from sglang.srt.disaggregation.pvd.cuda_working_set import CUDASparseWorkingSet
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.rank_install_wire import RankInstallExchange
from sglang.srt.disaggregation.pvd.sparse_install import RankInstallCoordinator
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferBudget,
    TransferCapacityError,
)
from test_pvd_cuda_sparse_attention import policy_workspace
from test_pvd_cuda_working_set import options, packed_payloads
from test_pvd_sparse_cpu_backend import Pool


def fixture(monkeypatch, *, capacity=4096, resume=True, request_id="r"):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    bank = CUDASparseWorkingSet(
        device="cuda:0",
        dtype=torch.float32,
        budget=TransferBudget(4096, 3),
        **{**options(), "request_id": request_id},
    )
    bank.device = torch.device("cpu")
    monkeypatch.setattr(bank, "_synchronize", lambda: None)
    peer = CUDARankInstallParticipant(bank, rank=0, peer_epoch="p", interval=4)
    exchange = RankInstallExchange(
        RankInstallCoordinator(
            request_id,
            "inc",
            "entry",
            rank_layouts={0: "layout"},
            interval=4,
            lead_tokens=1,
        ),
        peer_epochs={0: "p"},
    )
    epoch = exchange.begin(0)
    payloads, source, _ = packed_payloads()
    for payload in payloads:
        payload.spec = replace(
            payload.spec, request_id=request_id, operation_id=epoch.operation_id
        )
    exchange.receive(peer.stage(epoch, payloads, source_guard=source), peer_rank=0)
    source.request_release()
    exchange.receive(peer.park(0), peer_rank=0)
    exchange.receive(peer.command(exchange.install_commands(epoch)[0]), peer_rank=0)
    if resume:
        exchange.receive(peer.command(exchange.resume_commands(epoch)[0]), peer_rank=0)
    workspace, _, _ = policy_workspace(monkeypatch)
    pool = Pool()
    req = SimpleNamespace(req_to_token=torch.zeros(3, 16, dtype=torch.int64))
    req.req_to_token[1, :4] = -999
    req.req_to_token[1, 4:6] = torch.tensor([10, 11])
    lock, budget, released, drains = (
        threading.RLock(),
        TransferBudget(capacity, 2),
        [],
        [],
    )
    consumer = CUDAModelSparseConsumer(
        req,
        pool,
        layers=(0,),
        mapping=QueryHeadMapping(4, 2),
        workspace=workspace,
        execution_lock=lock,
        output_budget=budget,
        max_batch_size=2,
    )
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(consumer, "_synchronize", lambda: drains.append(True))
    guard = ResourceGuard(CUDAModelPools(req, pool), lambda: released.append(True))
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        encoder_lens=None,
        req_pool_indices=torch.tensor([1]),
        positions=torch.tensor([5]),
        seq_lens=torch.tensor([6]),
        out_cache_loc=torch.tensor([11]),
        spec_info=None,
    )
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=4,
        tp_k_head_num=2,
        tp_v_head_num=2,
        qk_head_dim=3,
        v_head_dim=3,
        scaling=1 / math.sqrt(3),
        is_cross_attention=False,
        attn_type="decoder",
        logit_cap=0,
        sliding_window_size=-1,
    )
    return SimpleNamespace(
        consumer=consumer,
        binding=CUDADecodeBinding(1, 1, peer, exchange),
        owner=guard,
        pool=pool,
        req=req,
        batch=batch,
        layer=layer,
        lock=lock,
        budget=budget,
        released=released,
        drains=drains,
        workspace=workspace,
        q=torch.arange(12, dtype=torch.float32).reshape(1, 12) / 9,
        k=pool.k[12:13].clone(),
        v=pool.v[13:14].clone(),
    )


def run(c):
    return c.consumer.forward_decode(c.q, c.k, c.v, c.layer, c.batch)


def test_model_pool_mapping_matches_dense_oracle_and_retirement_is_fenced(monkeypatch):
    c = fixture(monkeypatch)
    before = c.pool.k.clone(), c.pool.v.clone()
    with c.consumer.bind([c.binding], pool_owner=c.owner):
        output = run(c)
        with c.binding.participant.read(1) as groups:
            expected = []
            for head in range(4):
                prompt = groups[(0, head // 2)][1]
                keys = torch.cat(
                    (prompt[0], before[0][10, head // 2].view(1, 3), c.k[:, head // 2])
                )
                values = torch.cat(
                    (prompt[1], before[1][10, head // 2].view(1, 3), c.v[:, head // 2])
                )
                expected.append(
                    ((keys @ c.q.reshape(4, 3)[head]) * c.layer.scaling).softmax(0)
                    @ values
                )
        torch.testing.assert_close(output.reshape(4, 3), torch.stack(expected))
        c.owner.request_release()
        assert not c.released and c.budget.snapshot()["used_staging_bytes"] == 48
        assert c.consumer.snapshot()["retained_outputs"] == 1
    assert c.released == [True] and c.drains == [True]
    assert c.budget.snapshot()["used_staging_bytes"] == 0
    assert not c.consumer.snapshot()["active"]
    assert c.consumer.snapshot()["retained_outputs"] == 0
    torch.testing.assert_close(c.pool.k[10], before[0][10], rtol=0, atol=0)
    torch.testing.assert_close(c.pool.k[11], c.k[0], rtol=0, atol=0)


@pytest.mark.parametrize(
    "bad",
    [
        "position",
        "length",
        "destination",
        "repeat_row",
        "zero_row",
        "spec",
        "head",
        "swa",
        "cap",
        "cross",
        "dtype",
        "extend",
        "slot",
        "scale",
    ],
)
def test_invalid_batch_refused_before_any_pool_write(monkeypatch, bad):
    c = fixture(monkeypatch)
    if bad == "position":
        c.batch.positions[0] = 4
    elif bad == "length":
        c.batch.seq_lens[0] = 7
    elif bad == "destination":
        c.batch.out_cache_loc[0] = 15
    elif bad == "repeat_row":
        c.req.req_to_token[1, 4] = 11
    elif bad == "zero_row":
        c.req.req_to_token[1, 4] = 0
    elif bad == "spec":
        c.batch.spec_info = object()
    elif bad == "head":
        c.layer.tp_q_head_num = 8
    elif bad == "swa":
        c.layer.sliding_window_size = 3
    elif bad == "cap":
        c.layer.logit_cap = 2
    elif bad == "cross":
        c.layer.is_cross_attention = True
    elif bad == "dtype":
        c.q = c.q.half()
    elif bad == "extend":
        c.batch.forward_mode = SimpleNamespace(is_decode=lambda: False)
    elif bad == "slot":
        c.batch.req_pool_indices[0] = 2
    elif bad == "scale":
        c.layer.scaling = float("nan")
    with (
        pytest.raises(SparsePayloadError),
        c.consumer.bind([c.binding], pool_owner=c.owner),
    ):
        run(c)
    assert c.pool.writes == 0 and not c.consumer.snapshot()["quarantine"]
    assert c.budget.snapshot()["used_staging_bytes"] == 0


@pytest.mark.parametrize("failure", ["model", "write", "drain", "reader", "release"])
def test_failure_fences_or_quarantines_whole_forward_lease(monkeypatch, failure):
    c = fixture(monkeypatch)

    def fail():
        raise RuntimeError("injected")

    if failure == "write":

        def fail_write(*args):
            c.pool.writes += 1
            fail()

        monkeypatch.setattr(c.pool, "set_kv_buffer", fail_write)
    with (
        pytest.raises((RuntimeError, SparsePayloadError)),
        c.consumer.bind([c.binding], pool_owner=c.owner),
    ):
        c.owner.request_release()
        run(c)
        if failure == "model":
            fail()
        elif failure == "drain":
            monkeypatch.setattr(c.consumer, "_synchronize", fail)
        elif failure == "reader":
            monkeypatch.setattr(c.binding.participant._bank, "_drain_reader", fail)
        elif failure == "release":
            c.owner._release = fail
    if failure in ("drain", "reader", "release"):
        assert c.consumer.snapshot()["quarantine"] and c.consumer.snapshot()["held"]
        assert not c.released and c.budget.snapshot()["used_staging_bytes"] == 48
        with (
            pytest.raises(SparsePayloadError, match="quarantined"),
            c.consumer.bind([c.binding], pool_owner=c.owner),
        ):
            pass
    else:
        assert c.released == [True] and c.budget.snapshot()["used_staging_bytes"] == 0
    # Dispose CPU-only generator fixtures before monkeypatch restores CUDA
    # methods. This is test teardown, not a production quarantine recovery API.
    if c.consumer._held is not None:
        monkeypatch.setattr(c.binding.participant._bank, "_drain_reader", lambda: None)
        c.consumer._held[0].close()


def test_output_capacity_and_missing_resume_refused_before_execution(monkeypatch):
    c = fixture(monkeypatch, capacity=1)
    with (
        pytest.raises(TransferCapacityError),
        c.consumer.bind([c.binding], pool_owner=c.owner),
    ):
        run(c)
    assert c.pool.writes == 0 and not c.consumer.snapshot()["active"]
    c = fixture(monkeypatch, resume=False)
    with (
        pytest.raises(SparsePayloadError, match="resume"),
        c.consumer.bind([c.binding], pool_owner=c.owner),
    ):
        run(c)
    assert c.pool.writes == 0 and c.budget.snapshot()["used_staging_bytes"] == 0


def test_whole_forward_lease_rejects_missing_layers_and_reentrancy(monkeypatch):
    c = fixture(monkeypatch)
    with (
        pytest.raises(SparsePayloadError, match="every bound"),
        c.consumer.bind([c.binding], pool_owner=c.owner),
    ):
        pass

    def release():
        with (
            pytest.raises(SparsePayloadError, match="nest"),
            c.consumer.bind([c.binding], pool_owner=c.owner),
        ):
            pass
        c.released.append(True)

    c.owner._release = release
    with c.consumer.bind([c.binding], pool_owner=c.owner):
        run(c)
        c.owner.request_release()
    assert c.released == [True]


@pytest.mark.parametrize("bad", ["lease", "peer", "slot", "count", "duplicate"])
def test_binding_identity_is_not_inferred(monkeypatch, bad):
    c = fixture(monkeypatch)
    bindings = [c.binding]
    if bad == "lease":
        c.owner = ResourceGuard(CUDAModelPools(object(), c.pool), lambda: None)
    elif bad == "peer":
        bindings = [
            replace(
                c.binding,
                participant=CUDARankInstallParticipant(
                    c.binding.participant._bank, rank=0, peer_epoch="other", interval=4
                ),
            )
        ]
    elif bad == "slot":
        bindings = [replace(c.binding, slot=0)]
    elif bad == "count":
        bindings = [replace(c.binding, decode_tokens=True)]
    elif bad == "duplicate":
        bindings *= 2
    with (
        pytest.raises(SparsePayloadError),
        c.consumer.bind(bindings, pool_owner=c.owner),
    ):
        run(c)
    assert c.pool.writes == 0


def test_cancellation_during_forward_cannot_be_accepted(monkeypatch):
    c = fixture(monkeypatch)
    with (
        pytest.raises(SparsePayloadError, match="cancelled"),
        c.consumer.bind([c.binding], pool_owner=c.owner),
    ):
        run(c)
        c.binding.exchange.coordinator.cancel("user cancel")
    assert not c.consumer.snapshot()["quarantine"]
    assert c.budget.snapshot()["used_staging_bytes"] == 0


def test_second_request_cannot_alias_first_requests_generated_rows(monkeypatch):
    c = fixture(monkeypatch)
    c2 = fixture(monkeypatch, request_id="other")
    binding2 = replace(c2.binding, slot=2)
    c.req.req_to_token[2, 4:6] = c.req.req_to_token[1, 4:6]
    c.req.req_to_token[2, 5] = 12
    c.batch.req_pool_indices = torch.tensor([1, 2])
    c.batch.positions = torch.tensor([5, 5])
    c.batch.seq_lens = torch.tensor([6, 6])
    c.batch.out_cache_loc = torch.tensor([11, 12])
    c.q, c.k, c.v = (
        tensor.repeat(2, *([1] * (tensor.ndim - 1))) for tensor in (c.q, c.k, c.v)
    )
    with (
        pytest.raises(SparsePayloadError, match="overlapping generated"),
        c.consumer.bind([c.binding, binding2], pool_owner=c.owner),
    ):
        run(c)
    assert c.pool.writes == 0


@pytest.mark.parametrize(
    "fault", [None, "tp", "graph", "weights", "head_dim", "device", "page"]
)
def test_explicit_backend_factory_gates_and_delegates(monkeypatch, fault):
    c = fixture(monkeypatch)

    class Model:
        quant_config = None
        config = SimpleNamespace(
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2
        )

        def parameters(self):
            return [
                SimpleNamespace(
                    device=torch.device("cpu"),
                    dtype=torch.float16 if fault == "weights" else torch.float32,
                )
            ]

    class Backend:
        def __init__(self, runner):
            self.req_to_token_pool = runner.req_to_token_pool
            self.token_to_kv_pool = runner.token_to_kv_pool

    for name, attribute, cls in (
        ("sglang.srt.models.llama", "LlamaForCausalLM", Model),
        (
            "sglang.srt.layers.attention.torch_native_backend",
            "TorchNativeAttnBackend",
            Backend,
        ),
    ):
        module = types.ModuleType(name)
        setattr(module, attribute, cls)
        monkeypatch.setitem(sys.modules, name, module)
    runner = SimpleNamespace(
        model=Model(),
        device="cuda",
        gpu_id=None,
        tp_size=2 if fault == "tp" else 1,
        pp_size=1,
        attn_cp_size=1,
        model_config=SimpleNamespace(head_dim=5 if fault == "head_dim" else 3),
        req_to_token_pool=c.req,
        token_to_kv_pool=c.pool,
        server_args=SimpleNamespace(
            enable_dp_attention=False,
            speculative_algorithm=None,
            page_size=2 if fault == "page" else 1,
            disable_overlap_schedule=True,
            disable_cuda_graph=fault != "graph",
        ),
    )
    if fault == "device":
        runner.device = "cpu"
    runner.attn_backend = Backend(runner)

    def build():
        return make_cuda_sparse_backend(
            runner,
            workspace=c.workspace,
            execution_lock=c.lock,
            output_budget=c.budget,
            max_batch_size=2,
        )

    if fault is not None:
        with pytest.raises(SparsePayloadError):
            build()
        assert c.pool.writes == 0
    else:
        backend = build()
        monkeypatch.setattr(backend.consumer, "_synchronize", lambda: None)
        with backend.consumer.bind([c.binding], pool_owner=c.owner):
            assert backend.forward_decode(c.q, c.k, c.v, c.layer, c.batch).shape == (
                1,
                12,
            )
        with pytest.raises(SparsePayloadError, match="bound decode"):
            backend.forward_extend()


def test_real_cuda_model_smoke_is_blocked_not_passed_without_device(
    monkeypatch, capsys
):
    import run_pvd_cuda_model_smoke

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert run_pvd_cuda_model_smoke.main([]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "pvd-cuda-sparse-model-v1"
    assert report["status"] == "blocked" and "evidence" not in report
