"""Exercise PVD HCA selection at the model-server and V launcher boundaries."""

import json
from types import SimpleNamespace

import pytest
from sglang.srt.arg_groups.pvd_disaggregation_hook import handle_pvd_disaggregation
from sglang.srt.disaggregation.pvd.preflight import (
    PVDPreflightError,
    resolve_rank_rails,
    validate_dual_rail_names,
    validate_rank_rail_names,
)
from sglang.srt.disaggregation.pvd.server import _validate_args, build_parser


def model_args(**overrides):
    values = dict(
        disaggregation_topology="pvd",
        disaggregation_mode="decode",
        pvd_kv_refresh_interval=16,
        pvd_vector_coordinator_url="http://v:9100",
        pvd_vector_groups=None,
        tp_size=2,
        dp_size=1,
        enable_dp_attention=False,
        pp_size=1,
        pvd_rank_rails=None,
        disaggregation_ib_device=None,
        disaggregation_transfer_backend="mooncake",
        pvd_strict_rdma_preflight=True,
        speculative_algorithm=None,
        enable_hierarchical_cache=False,
        enable_hisparse=False,
        enable_prefill_context_parallel=False,
        disaggregation_decode_enable_radix_cache=False,
        pvd_model_instance_id="model",
        pvd_transfer_staging_budget_bytes=1 << 30,
        pvd_transfer_max_inflight=64,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "devices,tp,mode,expected",
    [
        ("mlx5_2,mlx5_3", 2, "decode", {"0": "mlx5_2", "1": "mlx5_3"}),
        ("mlx5_3,mlx5_2", 2, "prefill", {"0": "mlx5_3", "1": "mlx5_2"}),
        ("rdma_a,rdma_b", 2, "decode", {"0": "rdma_a", "1": "rdma_b"}),
        ("mlx5_2", 1, "prefill", {"0": "mlx5_2"}),
        ("mlx5_2", 2, "decode", {"0": "mlx5_2", "1": "mlx5_2"}),
        (
            "mlx5_2,mlx5_3,mlx5_2,mlx5_3",
            4,
            "decode",
            {"0": "mlx5_2", "1": "mlx5_3", "2": "mlx5_2", "3": "mlx5_3"},
        ),
    ],
)
def test_model_ib_device_becomes_rank_local_mooncake_mapping(
    devices, tp, mode, expected
):
    args = model_args(
        disaggregation_ib_device=devices, tp_size=tp, disaggregation_mode=mode
    )
    handle_pvd_disaggregation(args)
    assert json.loads(args.disaggregation_ib_device) == expected
    assert args.pvd_rank_rails == ",".join(expected.values())
    assert args.pvd_strict_rdma_preflight


@pytest.mark.parametrize(
    "base,step,tp,mode,rails,expected",
    [
        (1, 1, 1, "prefill", "mlx5_0", {"1": "mlx5_0"}),
        (1, 1, 1, "decode", "mlx5_0", {"1": "mlx5_0"}),
        (2, 2, 2, "prefill", "mlx5_2,mlx5_3", {"2": "mlx5_2", "4": "mlx5_3"}),
    ],
)
def test_mooncake_mapping_uses_physical_gpu_ids(base, step, tp, mode, rails, expected):
    overrides = {
        "base_gpu_id": base,
        "gpu_id_step": step,
        "tp_size": tp,
        "disaggregation_mode": mode,
        "pvd_rank_rails": rails,
    }
    if mode == "decode":
        overrides.update(
            pvd_waiting_queue_bootstrap=True,
            pvd_full_kv_fanin_max_slices=4,
            pvd_full_kv_fanin_response_bytes=4096,
        )
    args = model_args(**overrides)
    handle_pvd_disaggregation(args)
    assert json.loads(args.disaggregation_ib_device) == expected
    handle_pvd_disaggregation(args)
    assert json.loads(args.disaggregation_ib_device) == expected


def test_nondefault_gpu_mapping_accepts_physical_json_and_rejects_missing_gpu():
    args = model_args(
        base_gpu_id=1,
        tp_size=1,
        disaggregation_mode="prefill",
        disaggregation_ib_device='{"1":"mlx5_0"}',
    )
    handle_pvd_disaggregation(args)
    assert args.pvd_rank_rails == "mlx5_0"
    assert json.loads(args.disaggregation_ib_device) == {"1": "mlx5_0"}
    with pytest.raises(ValueError, match="physical GPU IDs"):
        handle_pvd_disaggregation(
            model_args(
                base_gpu_id=1,
                tp_size=1,
                disaggregation_mode="prefill",
                disaggregation_ib_device='{"2":"mlx5_0"}',
            )
        )


def test_empty_gpu_json_remains_invalid_for_v_without_physical_mapping():
    with pytest.raises(ValueError, match="exactly ranks"):
        resolve_rank_rails(None, "{}", 2)


def test_model_preserves_legacy_default_and_custom_rank_rails():
    args = model_args()
    handle_pvd_disaggregation(args)
    assert json.loads(args.disaggregation_ib_device) == {"0": "mlx5_0", "1": "mlx5_1"}
    args = model_args(pvd_rank_rails="mlx5_8,mlx5_9")
    handle_pvd_disaggregation(args)
    assert json.loads(args.disaggregation_ib_device) == {"0": "mlx5_8", "1": "mlx5_9"}


def test_model_rejects_conflicting_flags_but_accepts_equivalent_mapping():
    args = model_args(
        pvd_rank_rails="mlx5_0,mlx5_1", disaggregation_ib_device="mlx5_2,mlx5_3"
    )
    with pytest.raises(ValueError, match="conflict"):
        handle_pvd_disaggregation(args)
    args = model_args(
        pvd_rank_rails=" mlx5_2 , mlx5_2 ", disaggregation_ib_device="mlx5_2"
    )
    handle_pvd_disaggregation(args)
    assert json.loads(args.disaggregation_ib_device) == {"0": "mlx5_2", "1": "mlx5_2"}


def test_decode_tp1_native_receive_rails_are_explicit_and_distinct():
    args = model_args(
        tp_size=1,
        pvd_rank_rails="mlx5_2",
        pvd_d_receive_rails=" mlx5_2 , mlx5_3 ",
        pvd_waiting_queue_bootstrap=True,
        pvd_full_kv_fanin_max_slices=4,
        pvd_full_kv_fanin_response_bytes=4096,
    )
    handle_pvd_disaggregation(args)
    assert args.pvd_d_receive_rails == "mlx5_2,mlx5_3"

    for invalid in ("mlx5_3,mlx5_4", "mlx5_2,mlx5_2"):
        other = model_args(
            tp_size=1,
            pvd_rank_rails="mlx5_2",
            pvd_d_receive_rails=invalid,
            pvd_waiting_queue_bootstrap=True,
            pvd_full_kv_fanin_max_slices=4,
            pvd_full_kv_fanin_response_bytes=4096,
        )
        with pytest.raises(ValueError, match="distinct HCAs including"):
            handle_pvd_disaggregation(other)

    with pytest.raises(ValueError, match="Decode TP1"):
        handle_pvd_disaggregation(model_args(pvd_d_receive_rails="mlx5_2,mlx5_3"))


def test_decode_manager_assembles_optional_receive_engine(monkeypatch):
    from sglang.srt.disaggregation.pvd import conn, multi_rail_receive

    existing, budget, shared = (
        object(),
        object(),
        SimpleNamespace(hostname="D", gpu_id=0),
    )
    assert (
        conn._build_sparse_receive_engine(SimpleNamespace(), shared, existing, budget)
        is existing
    )
    calls = []
    group = object()
    monkeypatch.setattr(
        multi_rail_receive,
        "create_native_receive_group",
        lambda **kwargs: calls.append(kwargs) or group,
    )
    assert (
        conn._build_sparse_receive_engine(
            SimpleNamespace(pvd_d_receive_rails="mlx5_2,mlx5_3"),
            shared,
            existing,
            budget,
        )
        is group
    )
    assert calls == [
        {
            "hostname": "D",
            "gpu_id": 0,
            "rails": ("mlx5_2", "mlx5_3"),
            "transfer_budget": budget,
            "existing_adapter": existing,
        }
    ]


def test_decode_manager_assembles_optional_cuda_receive_registry(monkeypatch):
    from sglang.srt.disaggregation.pvd import conn, cuda_sparse_receiver

    engine, budget, registry = object(), object(), object()
    assert (
        conn._build_sparse_receive_registry(
            SimpleNamespace(), engine, budget, epoch="worker", device="cuda:1"
        )
        is None
    )
    calls = []

    def make_registry(*args, **kwargs):
        calls.append((args, kwargs))
        return registry

    monkeypatch.setattr(
        cuda_sparse_receiver, "CUDASparseReceiveRegistry", make_registry
    )
    assert (
        conn._build_sparse_receive_registry(
            SimpleNamespace(pvd_d_receive_rails="mlx5_2,mlx5_3"),
            engine,
            budget,
            epoch="worker",
            device="cuda:1",
        )
        is registry
    )
    assert calls == [
        ((engine, budget), {"receiver_epoch": "worker", "device": "cuda:1"})
    ]


@pytest.mark.parametrize(
    "devices", ["", "mlx5_2,", ",mlx5_3", "mlx5_2,mlx5_3,mlx5_4", "../mlx5_2", "mlx5 2"]
)
def test_model_rejects_malformed_mapping(devices):
    with pytest.raises(ValueError):
        handle_pvd_disaggregation(model_args(disaggregation_ib_device=devices))


@pytest.mark.parametrize(
    "devices,expected",
    [("mlx5_2,mlx5_3", ["mlx5_2", "mlx5_3"]), ("mlx5_3", ["mlx5_3", "mlx5_3"])],
)
def test_v_accepts_ib_device_flag(devices, expected):
    args = build_parser().parse_args(
        [
            "--advertise-host",
            "127.0.0.1",
            "--transfer-staging-budget-bytes",
            "1073741824",
            "--transfer-max-inflight",
            "64",
            "--total-pages",
            "8",
            "--page-bytes",
            "256",
            "--disaggregation-ib-device",
            devices,
        ]
    )
    assert _validate_args(args) == expected
    assert args.strict_rdma_preflight


def test_v_rejects_conflicting_flags():
    args = build_parser().parse_args(
        [
            "--advertise-host",
            "127.0.0.1",
            "--transfer-staging-budget-bytes",
            "1073741824",
            "--transfer-max-inflight",
            "64",
            "--total-pages",
            "8",
            "--page-bytes",
            "256",
            "--disaggregation-ib-device",
            "mlx5_2,mlx5_3",
            "--rails",
            "mlx5_0,mlx5_1",
        ]
    )
    with pytest.raises(ValueError, match="conflict"):
        _validate_args(args)


def test_rail_modes_depend_on_distinct_devices_not_hca_spelling():
    assert validate_rank_rail_names(["mlx5_2", "mlx5_3"]) == "dual-rail"
    assert validate_rank_rail_names(["mlx5_3", "mlx5_3"]) == "single-rail-debug"
    assert validate_rank_rail_names(["hca0", "hca1", "hca2", "hca3"]) == "multi-rail"
    validate_dual_rail_names(["mlx5_3", "mlx5_2"])
    with pytest.raises(PVDPreflightError):
        validate_dual_rail_names(["mlx5_3", "mlx5_3"])


def test_pd_does_not_reinterpret_ib_device_list():
    args = model_args(
        disaggregation_topology="pd", disaggregation_ib_device="mlx5_2,mlx5_3"
    )
    handle_pvd_disaggregation(args)
    assert args.disaggregation_ib_device == "mlx5_2,mlx5_3"


def test_normalized_model_mapping_can_be_validated_again():
    args = model_args(disaggregation_ib_device="mlx5_2,mlx5_3")
    handle_pvd_disaggregation(args)
    handle_pvd_disaggregation(args)
    assert json.loads(args.disaggregation_ib_device) == {"0": "mlx5_2", "1": "mlx5_3"}


def test_model_accepts_explicit_json_mapping_and_file(tmp_path):
    mapping = '{"1":"mlx5_3","0":"mlx5_2"}'
    path = tmp_path / "hcas.json"
    path.write_text(mapping, encoding="utf-8")
    for config in (mapping, str(path)):
        args = model_args(disaggregation_ib_device=config)
        handle_pvd_disaggregation(args)
        assert args.pvd_rank_rails == "mlx5_2,mlx5_3"


@pytest.mark.parametrize(
    "mapping",
    [
        '{"0":"mlx5_2"}',
        '{"0":"mlx5_2","1":"mlx5_3,mlx5_4"}',
        '{"0":2,"1":"mlx5_3"}',
        "{bad json}",
    ],
)
def test_model_rejects_incomplete_or_ambiguous_json_mapping(mapping):
    with pytest.raises(ValueError):
        handle_pvd_disaggregation(model_args(disaggregation_ib_device=mapping))
