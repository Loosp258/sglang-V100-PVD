"""Group-gated model reads; actual Llama execution has a separate strict smoke."""

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from pvd_controlled_prefetch import ControlledFixture
from sglang.srt.disaggregation.pvd.sparse_cpu_backend import (
    CPUSparseDecodeConsumer,
    SparseDecodeBinding,
)
from sglang.srt.disaggregation.pvd.sparse_install import (
    CPUInstalledPromptView,
    InstallProtocolError,
)
from sglang.srt.disaggregation.pvd.sparse_payload import (
    SparseKVPayload,
    SparsePayloadError,
)
from test_pvd_controlled_prefetch import components


@pytest.fixture
def env():
    fixture = ControlledFixture(*components())
    try:
        yield fixture
    finally:
        fixture.close()


def test_view_joins_all_heads_and_holds_every_reader_until_exit(env):
    view = CPUInstalledPromptView(env.group, 0)
    with view.read() as groups:
        assert set(groups) == {(0, 0), (0, 1), (1, 0), (1, 1)}
        assert len({s.operation_id for s, _ in groups.values()}) == 1
        assert all(bank._readers == 1 for bank in env.group._banks.values())
    assert all(bank._readers == 0 for bank in env.group._banks.values())


@pytest.mark.parametrize("count", [4, 5, -1, True, 0.5])
def test_view_refuses_boundary_overrun_and_invalid_count(env, count):
    with pytest.raises(ValueError), CPUInstalledPromptView(env.group, count).read():
        pytest.fail("invalid count admitted")
    assert all(bank._readers == 0 for bank in env.group._banks.values())


def test_view_cannot_reopen_after_cancel(env):
    view = CPUInstalledPromptView(env.group, 0)
    env.request.cancel()
    with pytest.raises(InstallProtocolError), view.read():
        pytest.fail("cancelled group was exposed")


def test_later_shard_read_failure_releases_earlier_shard(env, monkeypatch):
    original = env.group.read

    @contextmanager
    def failing(rank, count):
        if rank == 1:
            raise RuntimeError("shard failure")
        with original(rank, count) as rows:
            yield rows

    monkeypatch.setattr(env.group, "read", failing)
    with (
        pytest.raises(RuntimeError, match="shard failure"),
        CPUInstalledPromptView(env.group, 0).read(),
    ):
        pytest.fail("failure swallowed")
    assert all(bank._readers == 0 for bank in env.group._banks.values())


@pytest.mark.parametrize("change", ["operation_id", "target_tokens"])
def test_mixed_installed_generation_is_refused(env, monkeypatch, change):
    original = env.group.read

    @contextmanager
    def corrupt(rank, count):
        with original(rank, count) as rows:
            yield (
                {
                    key: (
                        replace(
                            spec, **{change: "wrong" if change == "operation_id" else 4}
                        ),
                        data,
                    )
                    for key, (spec, data) in rows.items()
                }
                if rank == 1
                else rows
            )

    monkeypatch.setattr(env.group, "read", corrupt)
    with (
        pytest.raises(InstallProtocolError, match="mixed-generation"),
        CPUInstalledPromptView(env.group, 0).read(),
    ):
        pytest.fail("mixed generations admitted")


def test_stale_count_view_refuses_after_progress(env):
    view = CPUInstalledPromptView(env.group, 0)
    assert env.request.can_decode(1)
    with pytest.raises(ValueError, match="regressed"), view.read():
        pytest.fail("stale count admitted")


def test_whole_forward_view_blocks_every_shard_install(env):
    epoch = env.group.begin(3)
    with CPUInstalledPromptView(env.group, 3).read() as groups:
        for rank, meta in env.group.describe_banks().items():
            payloads = [
                SparseKVPayload(
                    replace(
                        groups[key][0], operation_id=epoch.operation_id, target_tokens=4
                    ),
                    groups[key][1].clone(),
                )
                for key in meta["groups"]
            ]
            try:
                env.group.stage(epoch, rank, payloads)
            finally:
                for payload in payloads:
                    payload.close()
        assert not env.group.try_install(epoch, {0: 4, 1: 4})
        assert env.group.coordinator.snapshot()["applied"] == ()
        assert {spec.target_tokens for spec, _ in groups.values()} == {0}
    assert env.group.try_install(epoch, {0: 4, 1: 4})
    with CPUInstalledPromptView(env.group, 4).read() as groups:
        assert {spec.operation_id for spec, _ in groups.values()} == {
            epoch.operation_id
        }


@pytest.mark.parametrize("kind", ["overlap", "shape", "layout"])
def test_incompatible_metadata_refused(env, monkeypatch, kind):
    metadata = env.group.describe_banks()
    if kind == "overlap":
        metadata[1]["groups"] = metadata[0]["groups"]
    elif kind == "shape":
        metadata[1]["head_dim"] += 1
    else:
        metadata[1]["identity"] = (*metadata[1]["identity"][:3], "other")
    monkeypatch.setattr(env.group, "describe_banks", lambda: metadata)
    with pytest.raises(InstallProtocolError, match="metadata differ"):
        CPUInstalledPromptView(env.group, 0)


def test_consumer_validates_count_position_and_holds_group_across_layers(env):
    class Pool:
        def __init__(self):
            self.k = [torch.zeros(16, 2, 8) for _ in range(2)]
            self.v = [torch.zeros(16, 2, 8) for _ in range(2)]
            self.writes = 0

        def get_key_buffer(self, layer):
            return self.k[layer]

        def get_value_buffer(self, layer):
            return self.v[layer]

        def set_kv_buffer(self, layer, locations, k, v):
            self.writes += 1
            self.k[layer.layer_id][locations] = k
            self.v[layer.layer_id][locations] = v

    pool = Pool()
    req = SimpleNamespace(req_to_token=torch.full((2, 16), -1, dtype=torch.int64))
    req.req_to_token[1, 5] = 7
    consumer = CPUSparseDecodeConsumer(req, pool, layers=(0, 1), mapping=env.mapping)
    view = CPUInstalledPromptView(env.group, 0)
    binding = SparseDecodeBinding(1, "r", env.incarnation, 5, view)
    with (
        pytest.raises(SparsePayloadError, match="position"),
        consumer.bind([replace(binding, query_position=6)]),
    ):
        pytest.fail("count/position mismatch admitted")
    assert pool.writes == 0
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        encoder_lens=None,
        req_pool_indices=torch.tensor([1]),
        positions=torch.tensor([5]),
        seq_lens=torch.tensor([6]),
        out_cache_loc=torch.tensor([7]),
    )
    with consumer.bind([binding]):
        for layer_id in (0, 1):
            assert all(b._readers == 1 for b in env.group._banks.values())
            layer = SimpleNamespace(
                layer_id=layer_id,
                is_cross_attention=False,
                attn_type="decoder",
                logit_cap=0,
                sliding_window_size=-1,
                tp_q_head_num=4,
                tp_k_head_num=2,
                tp_v_head_num=2,
                qk_head_dim=8,
                v_head_dim=8,
                scaling=0.5,
            )
            output = consumer.forward_decode(
                torch.ones(1, 32), torch.ones(1, 16), torch.ones(1, 16), layer, batch
            )
            assert output.shape == (1, 32) and torch.isfinite(output).all()
    assert pool.writes == 2
    assert all(b._readers == 0 for b in env.group._banks.values())
