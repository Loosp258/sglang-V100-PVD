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


def _build_vector_group_map(server_args: "ServerArgs") -> dict[str, str]:
    groups: dict[str, str] = {}
    if server_args.pvd_vector_coordinator_url:
        url = server_args.pvd_vector_coordinator_url.rstrip("/")
        _validate_http_url("--pvd-vector-coordinator-url", url)
        groups["default"] = url

    for spec in server_args.pvd_vector_groups or []:
        if "=" not in spec:
            raise ValueError(f"--pvd-vector-group must use ID=URL syntax, got {spec!r}")
        group_id, url = (part.strip() for part in spec.split("=", 1))
        if not group_id or not url:
            raise ValueError(
                f"--pvd-vector-group requires non-empty ID and URL, got {spec!r}"
            )
        if group_id in groups:
            raise ValueError(f"duplicate PVD vector group id {group_id!r}")
        _validate_http_url("--pvd-vector-group", url)
        groups[group_id] = url.rstrip("/")
    return groups


def handle_pvd_disaggregation(server_args: "ServerArgs") -> None:
    """Keep legacy PD untouched unless ``--disaggregation-topology pvd`` is set."""
    topology = server_args.disaggregation_topology
    if topology not in ("pd", "pvd"):
        raise ValueError(f"invalid disaggregation topology: {topology!r}")
    if topology == "pd":
        return
    if (
        isinstance(server_args.pvd_kv_refresh_interval, bool)
        or not isinstance(server_args.pvd_kv_refresh_interval, int)
        or server_args.pvd_kv_refresh_interval <= 0
    ):
        raise ValueError("--pvd-kv-refresh-interval must be a positive integer")
    if server_args.disaggregation_mode == "decode":
        server_args.disable_overlap_schedule = True
        logger.info(
            "PVD 3.0 uses a synchronous KV refresh barrier every %s Decode tokens",
            server_args.pvd_kv_refresh_interval,
        )

    if server_args.disaggregation_mode not in ("prefill", "decode"):
        raise ValueError(
            "PVD model servers must use --disaggregation-mode prefill or decode; "
            "the V role uses `python -m sglang.srt.disaggregation.pvd.server`"
        )
    vector_groups = _build_vector_group_map(server_args)
    if not vector_groups:
        raise ValueError(
            "PVD requires --pvd-vector-coordinator-url or --pvd-vector-group"
        )
    server_args.pvd_vector_coordinator_map = vector_groups

    supported_tp = (1, 2) if server_args.disaggregation_mode == "prefill" else (2, 4)
    if server_args.tp_size not in supported_tp:
        raise ValueError(
            f"PVD 2.0 {server_args.disaggregation_mode} currently supports "
            f"--tp-size {supported_tp}"
        )
    if server_args.dp_size != 1 or server_args.enable_dp_attention:
        raise ValueError("PVD requires one TP group (dp-size=1, DP attention off)")
    if server_args.pp_size != 1:
        raise ValueError("PVD requires --pp-size 1")
    from sglang.srt.disaggregation.pvd.preflight import (
        resolve_rank_rails,
        validate_rank_rail_names,
    )

    rails = resolve_rank_rails(
        server_args.pvd_rank_rails,
        getattr(server_args, "disaggregation_ib_device", None),
        server_args.tp_size,
    )
    rail_mode = validate_rank_rail_names(rails)
    server_args.pvd_rank_rails = ",".join(rails)
    if rail_mode == "single-rail-debug":
        logger.warning(
            "PVD single-rail debug mode is active: all TP ranks use %s; "
            "this mode has no rail redundancy or dual-rail bandwidth",
            rails[0],
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
