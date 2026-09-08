"""Exercise PVD's real adapter against a native-engine metadata-cache double.

Only the unavailable CUDA/RDMA boundary is replaced. The double snapshots
MC_DISABLE_METACACHE at construction, just as classic Mooncake loads its
process-global configuration once, and resolves writes by address, not region_id.
"""

import ast
import builtins
import importlib.metadata
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def transport(monkeypatch):
    root = Path(__file__).resolve().parents[3] / "python"
    for name in ("sglang.srt.distributed", "sglang.srt.utils"):
        package = types.ModuleType(name)
        package.__path__ = [str(root / Path(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.delitem(sys.modules, "mooncake.engine", raising=False)
    monkeypatch.delenv("MC_DISABLE_METACACHE", raising=False)
    real_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: (
            "0.3.13.post1" if name == "mooncake-transfer-engine" else real_version(name)
        ),
    )
    state = SimpleNamespace(
        live={}, writes=[], lookup_failure=False, engines=[], config_fresh=None
    )

    class NativeEngine:
        def __init__(self):
            self.cached = {}
            if state.config_fresh is None:
                state.config_fresh = "MC_DISABLE_METACACHE" in os.environ
            self.fresh = state.config_fresh
            state.engines.append(self)

        def initialize(self, *args):
            return 0

        def get_rpc_port(self):
            return 12345

        def transfer_sync_write(self, endpoint, source, destination, length):
            if self.fresh or endpoint not in self.cached:
                if state.lookup_failure:
                    return -1
                self.cached[endpoint] = dict(state.live)
            key = self.cached[endpoint].get(destination)
            state.writes.append((endpoint, destination, key, length))
            return 0 if key is not None and key == state.live[destination] else -1

    native = types.ModuleType("mooncake.engine")
    native.TransferEngine = NativeEngine
    original_import = builtins.__import__

    def native_import(name, *args, **kwargs):
        if name == "mooncake.engine":
            monkeypatch.setitem(sys.modules, name, native)
            return native
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", native_import)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)

    def load(name):
        spec = importlib.util.spec_from_file_location(
            name, root / Path(*name.split(".")).with_suffix(".py")
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    shared = load(
        "sglang.srt.distributed.device_communicators.mooncake_transfer_engine"
    )
    adapter = load("sglang.srt.disaggregation.pvd.mooncake_engine")
    return SimpleNamespace(shared=shared, adapter=adapter, state=state)


def put(adapter, region_id):
    from sglang.srt.disaggregation.pvd.protocol import RemoteRegionDescriptor
    from sglang.srt.disaggregation.pvd.transfer_engine import (
        MemorySlice,
        RegisteredMemory,
    )

    source = RemoteRegionDescriptor("v:1", "source", 4096, 16, "cuda:0", 0, "mlx5_2")
    target = RemoteRegionDescriptor("d:2", region_id, 8192, 16, "cuda:0", 0, "mlx5_2")
    return adapter.submit_put(
        MemorySlice(
            RegisteredMemory(source, torch.zeros(16, dtype=torch.uint8)), 0, 16
        ),
        target,
    )


def test_same_address_new_registration_uses_new_key_on_first_write(transport):
    from sglang.srt.disaggregation.pvd.transfer_engine import TransferStatus

    adapter = transport.adapter.MooncakePVDTransferEngine(
        hostname="v", gpu_id=0, rail="mlx5_2"
    )
    transport.state.live[8192] = 101
    assert put(adapter, "old-registration").status == TransferStatus.SUCCESS
    transport.state.live[8192] = 202
    assert put(adapter, "new-registration").status == TransferStatus.SUCCESS
    assert [w[2] for w in transport.state.writes] == [101, 202]


def test_metadata_lookup_failure_does_not_post_cached_write(transport):
    from sglang.srt.disaggregation.pvd.transfer_engine import TransferStatus

    adapter = transport.adapter.MooncakePVDTransferEngine(
        hostname="v", gpu_id=0, rail="mlx5_2"
    )
    transport.state.live[8192] = 101
    assert put(adapter, "first").status == TransferStatus.SUCCESS
    transport.state.lookup_failure = True
    assert put(adapter, "second").status == TransferStatus.FAILED
    assert len(transport.state.writes) == 1


def test_pvd_rejects_an_already_initialized_default_engine(transport):
    engine = transport.shared.MooncakeTransferEngine("d", gpu_id=0, ib_device="mlx5_2")
    with pytest.raises(RuntimeError, match="metadata|restart"):
        transport.adapter.MooncakePVDTransferEngine.from_existing(engine, rail="mlx5_2")


def test_pvd_rejects_late_configuration_even_for_a_new_engine(transport):
    transport.shared.MooncakeTransferEngine("d", gpu_id=0, ib_device="mlx5_2")
    with pytest.raises(RuntimeError, match="restart|already"):
        transport.adapter.MooncakePVDTransferEngine(
            hostname="v", gpu_id=0, rail="mlx5_2"
        )


def test_pvd_rejects_unverified_mooncake_version_before_native_init(
    transport, monkeypatch
):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.0.0")
    with pytest.raises(RuntimeError, match="version|verified"):
        transport.adapter.MooncakePVDTransferEngine(
            hostname="v", gpu_id=0, rail="mlx5_2"
        )
    assert not transport.state.engines


def test_two_v_ranks_share_preinit_policy_and_report_it(transport):
    ranks = [
        transport.adapter.MooncakePVDTransferEngine(hostname="v", gpu_id=i, rail=rail)
        for i, rail in enumerate(("mlx5_2", "mlx5_3"))
    ]
    for adapter in ranks:
        assert adapter.health().get("metadata_policy") == "fresh"
        assert adapter.health().get("mooncake_version") == "0.3.13.post1"


def test_regular_pd_does_not_change_process_environment(transport):
    engine = transport.shared.MooncakeTransferEngine("p", gpu_id=0, ib_device="mlx5_2")
    assert "MC_DISABLE_METACACHE" not in os.environ
    transport.state.live[8192] = 101
    assert engine.transfer_sync("d:2", 4096, 8192, 16) == 0
    transport.state.live[8192] = 202
    # Characterization: the native double reproduces the original stale-key risk.
    assert engine.transfer_sync("d:2", 4096, 8192, 16) == -1


def test_shared_pvd_initialization_and_reuse_enforce_policy(transport):
    engine = transport.shared.init_mooncake_transfer_engine(
        "d", gpu_id=0, ib_device="mlx5_2", require_fresh_metadata=True
    )
    adapter = transport.adapter.MooncakePVDTransferEngine.from_existing(
        engine, rail="mlx5_2"
    )
    assert adapter.health()["metadata_policy"] == "fresh"
    assert (
        transport.shared.init_mooncake_transfer_engine(
            "d", gpu_id=0, ib_device="mlx5_2", require_fresh_metadata=True
        )
        is engine
    )


def test_existing_shared_default_engine_cannot_be_silently_reused_by_pvd(transport):
    transport.shared.init_mooncake_transfer_engine("d", gpu_id=0, ib_device="mlx5_2")
    with pytest.raises(RuntimeError, match="metadata|restart"):
        transport.shared.init_mooncake_transfer_engine(
            "d", gpu_id=0, ib_device="mlx5_2", require_fresh_metadata=True
        )


def test_export_after_native_init_cannot_certify_an_old_engine(transport, monkeypatch):
    engine = transport.shared.MooncakeTransferEngine("d", gpu_id=0, ib_device="mlx5_2")
    monkeypatch.setenv("MC_DISABLE_METACACHE", "1")
    with pytest.raises(RuntimeError, match="metadata|restart"):
        transport.adapter.MooncakePVDTransferEngine.from_existing(engine, rail="mlx5_2")


@pytest.mark.parametrize("value", ["0", "", "1"])
def test_preexisting_env_is_normalized_before_native_load(
    transport, monkeypatch, value
):
    monkeypatch.setenv("MC_DISABLE_METACACHE", value)
    adapter = transport.adapter.MooncakePVDTransferEngine(
        hostname="v", gpu_id=0, rail="mlx5_2"
    )
    transport.state.live[8192] = 101
    put(adapter, "one")
    transport.state.live[8192] = 202
    put(adapter, "two")
    assert [w[2] for w in transport.state.writes] == [101, 202]
    assert os.environ["MC_DISABLE_METACACHE"] == "1"


def test_native_module_imported_outside_wrapper_is_rejected(transport, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "mooncake.engine", types.ModuleType("mooncake.engine")
    )
    with pytest.raises(RuntimeError, match="before|restart"):
        transport.adapter.MooncakePVDTransferEngine(
            hostname="v", gpu_id=0, rail="mlx5_2"
        )
    assert not transport.state.engines


def test_missing_package_rejected_before_environment_change(transport, monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="version"):
        transport.adapter.MooncakePVDTransferEngine(
            hostname="v", gpu_id=0, rail="mlx5_2"
        )
    assert "MC_DISABLE_METACACHE" not in os.environ
    assert not transport.state.engines


def test_regular_pd_accepts_versions_not_pinned_by_pvd(transport, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.0.0")
    engine = transport.shared.MooncakeTransferEngine("p", gpu_id=0, ib_device="mlx5_2")
    transport.state.live[8192] = 101
    assert engine.transfer_sync("d:2", 4096, 8192, 16) == 0
    assert "MC_DISABLE_METACACHE" not in os.environ


def test_v_launcher_accepts_debug_logging_for_mr_diagnostics():
    from sglang.srt.disaggregation.pvd.server import build_parser

    args = build_parser().parse_args(
        [
            "--advertise-host",
            "v",
            "--total-pages",
            "8",
            "--page-bytes",
            "32",
            "--log-level",
            "debug",
        ]
    )
    assert args.log_level == "debug"


@pytest.mark.parametrize("topology", ["pvd", "pd"])
def test_model_runner_initializes_policy_before_shared_engine(
    transport, monkeypatch, topology
):
    # Execute the real ModelRunner method without importing its CUDA-only
    # frontend dependencies. Only its hardware preflight is replaced.
    from sglang.srt.disaggregation.pvd import preflight

    monkeypatch.setattr(
        preflight,
        "run_rank_preflight",
        lambda **kw: SimpleNamespace(to_dict=lambda: {}),
    )
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/model_executor/model_runner.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "init_shared_mooncake_transfer_engine"
    )
    namespace = {
        "get_local_ip_auto": lambda: "d",
        "logger": logging.getLogger(__name__),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    runner = SimpleNamespace(
        gpu_id=0,
        tp_rank=0,
        server_args=SimpleNamespace(
            disaggregation_topology=topology,
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
            disaggregation_ib_device="mlx5_2",
            mooncake_ib_device=None,
            pvd_rank_rails="mlx5_2,mlx5_3",
            pvd_strict_rdma_preflight=True,
        ),
    )
    namespace["init_shared_mooncake_transfer_engine"](runner)
    engine = transport.shared.get_mooncake_transfer_engine()
    assert engine is not None
    if topology == "pvd":
        engine.require_pvd_metadata_policy()
        assert os.environ["MC_DISABLE_METACACHE"] == "1"
    else:
        assert "MC_DISABLE_METACACHE" not in os.environ
