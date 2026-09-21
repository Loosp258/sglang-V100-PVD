"""Pool-consumer contracts; real ModelRunner evidence is in the strict smoke."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from enum import Enum
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.sparse_cpu_backend import (
    CPUSparseDecodeConsumer,
    SparseDecodeBinding,
)
from sglang.srt.disaggregation.pvd.sparse_payload import (
    SparseKVPayload,
    SparseKVSpec,
    SparsePayloadError,
)
from sglang.srt.disaggregation.pvd.sparse_working_set import CPUSparseWorkingSet
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


class Pool:
    def __init__(self):
        self.k = torch.randn(32, 2, 3, generator=torch.Generator().manual_seed(7))
        self.v = self.k + 3
        self.writes = 0

    def get_key_buffer(self, layer):
        return self.k

    def get_value_buffer(self, layer):
        return self.v

    def set_kv_buffer(self, layer, locations, k, v):
        self.writes += 1
        self.k[locations] = k
        self.v[locations] = v


class AttentionTypeFixture(Enum):
    # Match RadixAttention's Enum contract, not a string-valued double.
    DECODER = "decoder"


def fixture(request_id="r"):
    budget = TransferBudget(4096, 4)
    bank = CPUSparseWorkingSet(
        request_id=request_id,
        incarnation="inc",
        entry_transfer_id="entry",
        layout_fingerprint="layout",
        expected_groups=((0, 0), (0, 1)),
        prompt_tokens=4,
        head_dim=3,
        max_union_tokens=3,
        budget=budget,
    )
    pool = Pool()

    def rows(boundary, tokens):
        return [
            SparseKVPayload(
                SparseKVSpec(
                    request_id,
                    "inc",
                    f"op-{boundary}",
                    boundary,
                    "entry",
                    "idx",
                    "map",
                    "layout",
                    0,
                    head,
                    tokens,
                ),
                torch.stack((pool.k[list(tokens), head], pool.v[list(tokens), head])),
            )
            for head in range(2)
        ]

    bank.stage(rows(0, (0, 1, 2, 3)))
    bank.install(0)
    bank.stage(rows(2, (3, 1)))
    bank.install(2)
    req = SimpleNamespace(req_to_token=torch.zeros(3, 16, dtype=torch.int64))
    req.req_to_token[1, :4] = -999  # sparse consumer must NOT read Prompt map
    req.req_to_token[1, 4:6] = torch.tensor([10, 11])
    consumer = CPUSparseDecodeConsumer(
        req, pool, layers=(0,), mapping=QueryHeadMapping(4, 2)
    )
    binding = SparseDecodeBinding(1, request_id, "inc", 5, bank)
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
        scaling=0.37,
        is_cross_attention=False,
        attn_type=AttentionTypeFixture.DECODER,
        logit_cap=0,
        sliding_window_size=-1,
    )
    q, k, v = (
        torch.arange(12, dtype=torch.float32).reshape(1, 12) / 9,
        pool.k[12:13].clone(),
        pool.v[13:14].clone(),
    )
    return consumer, binding, batch, layer, q, k, v, budget


def test_real_consumer_uses_head_specific_bank_and_generated_rows_and_layer_scale():
    consumer, binding, batch, layer, q, k, v, budget = fixture()
    pool = consumer.kv_pool
    prior = pool.k[10].clone(), pool.v[10].clone()
    with consumer.bind([binding]):
        with pytest.raises(SparsePayloadError, match="forward"):
            binding.bank.close()
        actual = consumer.forward_decode(q, k, v, layer, batch)
        with binding.bank.read() as groups:
            expected = []
            for head in range(4):
                data = groups[(0, head // 2)][1]
                keys = torch.cat(
                    (data[0], prior[0][head // 2].reshape(1, 3), k[:, head // 2])
                )
                vals = torch.cat(
                    (data[1], prior[1][head // 2].reshape(1, 3), v[:, head // 2])
                )
                expected.append(
                    ((keys @ q.reshape(4, 3)[head]) * layer.scaling).softmax(0) @ vals
                )
        torch.testing.assert_close(actual.reshape(4, 3), torch.stack(expected))
    torch.testing.assert_close(pool.k[10], prior[0], rtol=0, atol=0)
    torch.testing.assert_close(pool.v[10], prior[1], rtol=0, atol=0)
    torch.testing.assert_close(pool.k[11], k[0], rtol=0, atol=0)
    torch.testing.assert_close(pool.v[11], v[0], rtol=0, atol=0)
    assert pool.writes == 1
    binding.bank.close()
    assert budget.snapshot()["used_staging_bytes"] == 0


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
        "missing_slot",
    ],
)
def test_invalid_input_is_refused_before_generated_write(bad):
    consumer, binding, batch, layer, q, k, v, _ = fixture()
    if bad == "position":
        batch.positions[0] = 4
    elif bad == "length":
        batch.seq_lens[0] = 7
    elif bad == "destination":
        batch.out_cache_loc[0] = 15
    elif bad == "repeat_row":
        consumer.req_pool.req_to_token[1, 4] = 11
    elif bad == "zero_row":
        consumer.req_pool.req_to_token[1, 4] = 0
    elif bad == "spec":
        batch.spec_info = object()
    elif bad == "head":
        layer.tp_q_head_num = 2
    elif bad == "swa":
        layer.sliding_window_size = 2
    elif bad == "cap":
        layer.logit_cap = 2
    elif bad == "cross":
        layer.is_cross_attention = True
    elif bad == "dtype":
        q = q.half()
    elif bad == "extend":
        batch.forward_mode = SimpleNamespace(is_decode=lambda: False)
    else:
        batch.req_pool_indices[0] = 2
    with pytest.raises(SparsePayloadError), consumer.bind([binding]):
        consumer.forward_decode(q, k, v, layer, batch)
    assert consumer.kv_pool.writes == 0
    binding.bank.close()  # a refused forward must release its reader


def test_no_scope_and_stale_binding_are_rejected():
    consumer, binding, batch, layer, q, k, v, _ = fixture()
    with pytest.raises(SparsePayloadError, match="binding scope"):
        consumer.forward_decode(q, k, v, layer, batch)
    with (
        pytest.raises(SparsePayloadError, match="mismatch"),
        consumer.bind([replace(binding, incarnation="old")]),
    ):
        pytest.fail("stale binding accepted")
    binding.bank.close()


def test_scope_requires_every_layer_and_refuses_repeated_forward():
    consumer, binding, batch, layer, q, k, v, _ = fixture()
    with (
        pytest.raises(SparsePayloadError, match="every bound"),
        consumer.bind([binding]),
    ):
        pass
    with consumer.bind([binding]):
        consumer.forward_decode(q, k, v, layer, batch)
        with pytest.raises(SparsePayloadError, match="repeated layer"):
            consumer.forward_decode(q, k, v, layer, batch)
        with pytest.raises(SparsePayloadError, match="nest"), consumer.bind([binding]):
            pytest.fail("nested scope")
    assert consumer.kv_pool.writes == 1
    binding.bank.close()


def test_two_requests_cannot_share_generated_rows_before_any_write():
    consumer, binding, batch, layer, q, k, v, _ = fixture()
    _, second, _, _, _, _, _, _ = fixture("r2")
    second = replace(second, slot=2)
    consumer.req_pool.req_to_token[2] = consumer.req_pool.req_to_token[1]
    batch.req_pool_indices = torch.tensor([1, 2])
    batch.positions = torch.tensor([5, 5])
    batch.seq_lens = torch.tensor([6, 6])
    consumer.req_pool.req_to_token[2, 5] = 12
    batch.out_cache_loc = torch.tensor([11, 12])
    with (
        pytest.raises(SparsePayloadError, match="aliases another"),
        consumer.bind([binding, second]),
    ):
        consumer.forward_decode(
            q.repeat(2, 1), k.repeat(2, 1, 1), v.repeat(2, 1, 1), layer, batch
        )
    assert consumer.kv_pool.writes == 0
    binding.bank.close()
    second.bank.close()


def test_same_request_cannot_bind_two_slots():
    consumer, binding, _, _, _, _, _, _ = fixture()
    with (
        pytest.raises(SparsePayloadError, match="more than once"),
        consumer.bind([binding, replace(binding, slot=2)]),
    ):
        pytest.fail("duplicate instance accepted")
    binding.bank.close()


def test_foreign_thread_cannot_enter_scope():
    consumer, binding, _, _, _, _, _, _ = fixture()

    def work():
        with consumer.bind([binding]):
            pytest.fail("foreign thread admitted")

    with (
        ThreadPoolExecutor(1) as executor,
        pytest.raises(SparsePayloadError, match="main-thread-only"),
    ):
        executor.submit(work).result()
    binding.bank.close()


def test_reordered_batch_with_different_positions_matches_each_request_alone():
    consumer, first, batch, layer, q, k, v, _ = fixture()
    _, second, _, _, _, _, _, _ = fixture("second")
    second = replace(second, slot=2, query_position=6)
    consumer.req_pool.req_to_token[2, 4:7] = torch.tensor([18, 19, 20])
    second_batch = SimpleNamespace(**vars(batch))
    second_batch.req_pool_indices = torch.tensor([2])
    second_batch.positions = torch.tensor([6])
    second_batch.seq_lens = torch.tensor([7])
    second_batch.out_cache_loc = torch.tensor([20])
    with consumer.bind([first]):
        expected_first = consumer.forward_decode(q, k, v, layer, batch)
    with consumer.bind([second]):
        expected_second = consumer.forward_decode(
            q + 1, k + 1, v + 1, layer, second_batch
        )
    batch.req_pool_indices = torch.tensor([2, 1])
    batch.positions = torch.tensor([6, 5])
    batch.seq_lens = torch.tensor([7, 6])
    batch.out_cache_loc = torch.tensor([20, 11])
    with consumer.bind([first, second]):
        actual = consumer.forward_decode(
            torch.cat((q + 1, q)),
            torch.cat((k + 1, k)),
            torch.cat((v + 1, v)),
            layer,
            batch,
        )
    torch.testing.assert_close(actual, torch.cat((expected_second, expected_first)))
    first.bank.close()
    second.bank.close()
