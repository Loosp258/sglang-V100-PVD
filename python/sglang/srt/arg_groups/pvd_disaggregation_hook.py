"""Validation and normalization for the three-node PVD topology."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


def _validate_http_url(name: str, value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"{name} must be an absolute HTTP(S) URL, got {value!r}")


def handle_pvd_disaggregation(server_args: "ServerArgs") -> None:
    """Keep legacy PD untouched unless ``--disaggregation-topology pvd`` is set."""
    topology = server_args.disaggregation_topology
    if topology not in ("pd", "pvd"):
        raise ValueError(f"invalid disaggregation topology: {topology!r}")
    if topology == "pd":
        return

    if server_args.disaggregation_mode not in ("prefill", "decode"):
        raise ValueError(
            "PVD model servers must use --disaggregation-mode prefill or decode; "
            "the V role uses `python -m sglang.srt.disaggregation.pvd.server`"
        )
    if not server_args.pvd_vector_coordinator_url:
        raise ValueError("PVD requires --pvd-vector-coordinator-url")
    _validate_http_url(
        "--pvd-vector-coordinator-url", server_args.pvd_vector_coordinator_url
    )

    supported_tp = (
        (1, 2) if server_args.disaggregation_mode == "prefill" else (2, 4)
    )
    if server_args.tp_size not in supported_tp:
        raise ValueError(
            f"PVD 2.0 {server_args.disaggregation_mode} currently supports "
            f"--tp-size {supported_tp}"
        )
    if server_args.dp_size != 1 or server_args.enable_dp_attention:
        raise ValueError("PVD requires one TP group (dp-size=1, DP attention off)")
    if server_args.pp_size != 1:
        raise ValueError("PVD requires --pp-size 1")
    rails = [item.strip() for item in server_args.pvd_rank_rails.split(",")]
    if len(rails) != server_args.tp_size:
        raise ValueError(
            "PVD requires exactly one --pvd-rank-rails value per TP rank; "
            f"got {len(rails)} values for TP={server_args.tp_size}"
        )
    from sglang.srt.disaggregation.pvd.preflight import validate_rank_rail_names

    rail_mode = validate_rank_rail_names(rails)
    server_args.pvd_rank_rails = ",".join(rails)
    if rail_mode == "single-rail-debug":
        logger.warning(
            "PVD single-rail debug mode is active: all TP ranks use mlx5_0; "
            "this mode has no rail redundancy or dual-rail bandwidth"
        )
    if server_args.disaggregation_transfer_backend != "mooncake":
        raise ValueError("PVD currently requires the mooncake transfer backend")
    if not server_args.pvd_strict_rdma_preflight:
        raise ValueError("PVD P/D roles require strict rank/rail GPUDirect preflight")
    if server_args.speculative_algorithm is not None:
        raise ValueError("PVD does not support speculative decoding")
    if server_args.enable_hierarchical_cache:
        raise ValueError("PVD does not support hierarchical KV cache")
    if server_args.enable_hisparse:
        raise ValueError("PVD does not support HiSparse decode destinations")
    if server_args.enable_prefill_context_parallel:
        raise ValueError("PVD does not support Prefill context parallelism")
    if envs.SGLANG_DISAGG_STAGING_BUFFER.get():
        raise ValueError(
            "PVD uses its own full-prompt staging layout; "
            "SGLANG_DISAGG_STAGING_BUFFER must be disabled"
        )
    if server_args.disaggregation_decode_enable_radix_cache:
        raise ValueError("PVD requires decode radix cache to remain disabled")

    # Feed the existing Mooncake GPU->HCA selector an explicit per-GPU map.
    server_args.disaggregation_ib_device = json.dumps(
        {str(rank): rail for rank, rail in enumerate(rails)}, separators=(",", ":")
    )
    # Every Entry owns a complete prompt KV allocation. Prefix reuse on P would
    # make the exported allocation partial and violate the Entry manifest.
    server_args.disable_radix_cache = True
    server_args.disaggregation_decode_enable_radix_cache = False
    if not server_args.pvd_model_instance_id:
        revision = server_args.revision or "default"
        server_args.pvd_model_instance_id = f"{server_args.model_path}@{revision}"
