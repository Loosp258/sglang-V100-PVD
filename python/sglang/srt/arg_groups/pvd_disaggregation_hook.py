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


def _require_positive_int(name: str, value: object) -> int:
    """Accept only an explicit positive integer.

    ``None`` means the operator did not choose a budget, and PVD will not guess
    one. ``bool`` is rejected explicitly because it is an ``int`` subclass and
    ``--flag true`` must not silently become a budget of 1.
    """
    if value is None:
        raise ValueError(f"{name} is required with --disaggregation-topology pvd")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_decode_token_reservation(server_args: "ServerArgs") -> None:
    """Reject a Decode pool that cannot admit even an empty new request."""
    capacity = getattr(server_args, "max_total_tokens", None)
    reserved = getattr(server_args, "num_reserved_decode_tokens", None)
    if capacity is not None and reserved is not None and reserved >= capacity:
        raise ValueError(
            "PVD Decode --num-reserved-decode-tokens must be smaller than "
            "--max-total-tokens; otherwise every request waits forever for KV"
        )


def _validate_predictive_retrieval_config(server_args: "ServerArgs") -> bool:
    """Validate retrieval settings and the optional CUDA-serving preflight."""
    enabled = getattr(server_args, "pvd_predictive_retrieval_config", False)
    if not isinstance(enabled, bool):
        raise ValueError("--pvd-predictive-retrieval-config must be a boolean")
    serving_enabled = getattr(server_args, "pvd_cuda_predictive_serving", False)
    if not isinstance(serving_enabled, bool):
        raise ValueError("--pvd-cuda-predictive-serving must be a boolean")
    serving_config = getattr(server_args, "pvd_cuda_serving_config", None)
    if serving_config is not None and not serving_enabled:
        raise ValueError(
            "--pvd-cuda-serving-config requires --pvd-cuda-predictive-serving"
        )
    if serving_enabled and not enabled:
        raise ValueError(
            "--pvd-cuda-predictive-serving requires --pvd-predictive-retrieval-config"
        )
    if serving_enabled and (
        not isinstance(serving_config, str) or not serving_config.strip()
    ):
        raise ValueError(
            "--pvd-cuda-serving-config is required with --pvd-cuda-predictive-serving"
        )

    names = (
        "pvd_retrieval_vector_space",
        "pvd_retrieval_top_k",
        "pvd_retrieval_max_union_tokens",
        "pvd_retrieval_bank_budget_bytes",
        "pvd_retrieval_scratch_budget_bytes",
    )
    if not enabled:
        supplied = [
            name for name in names if getattr(server_args, name, None) is not None
        ]
        if supplied:
            flags = ", ".join(f"--{name.replace('_', '-')}" for name in supplied)
            raise ValueError(
                f"{flags} require --pvd-predictive-retrieval-config; "
                "configuration values are otherwise ignored"
            )
        return False

    if server_args.disaggregation_topology != "pvd":
        raise ValueError("PVD predictive-retrieval configuration requires topology pvd")
    if server_args.disaggregation_mode != "decode":
        raise ValueError("PVD predictive-retrieval configuration is Decode-only")
    refresh_interval = getattr(server_args, "pvd_kv_refresh_interval", None)
    if (
        isinstance(refresh_interval, bool)
        or not isinstance(refresh_interval, int)
        or refresh_interval < 2
    ):
        raise ValueError(
            "PVD predictive retrieval requires --pvd-kv-refresh-interval >= 2 "
            "because the CUDA retrieval group needs lead_tokens < interval"
        )

    vector_space = getattr(server_args, "pvd_retrieval_vector_space", None)
    if (
        not isinstance(vector_space, str)
        or not vector_space
        or vector_space.strip() != vector_space
    ):
        raise ValueError(
            "--pvd-retrieval-vector-space is required and must be a non-empty "
            "exact identity"
        )
    if getattr(server_args, "pvd_retrieval_metric", "ip") != "ip":
        raise ValueError("PVD predictive retrieval currently supports metric ip only")
    top_k = _require_positive_int(
        "--pvd-retrieval-top-k", getattr(server_args, "pvd_retrieval_top_k", None)
    )
    if top_k > 512:
        raise ValueError("--pvd-retrieval-top-k must not exceed 512")
    max_union_tokens = _require_positive_int(
        "--pvd-retrieval-max-union-tokens",
        getattr(server_args, "pvd_retrieval_max_union_tokens", None),
    )
    if max_union_tokens < top_k:
        raise ValueError(
            "--pvd-retrieval-max-union-tokens must be at least --pvd-retrieval-top-k"
        )
    _require_positive_int(
        "--pvd-retrieval-bank-budget-bytes",
        getattr(server_args, "pvd_retrieval_bank_budget_bytes", None),
    )
    _require_positive_int(
        "--pvd-retrieval-scratch-budget-bytes",
        getattr(server_args, "pvd_retrieval_scratch_budget_bytes", None),
    )

    # This is the current executable CUDA sparse backend's declared envelope.
    # The checks are necessary but not sufficient: model architecture,
    # quantization, actual pools and backend objects are verified at runtime by
    # the lower-level factories. The CUDA opt-in installs them in Scheduler startup.
    if server_args.tp_size != 1:
        raise ValueError("PVD predictive retrieval currently requires Decode TP1")
    if server_args.pp_size != 1:
        raise ValueError("PVD predictive retrieval currently requires --pp-size 1")
    if server_args.dp_size != 1 or server_args.enable_dp_attention:
        raise ValueError("PVD predictive retrieval requires DP1 with DP attention off")
    if not getattr(server_args, "pvd_waiting_queue_bootstrap", False):
        raise ValueError(
            "PVD predictive retrieval requires --pvd-waiting-queue-bootstrap"
        )
    if (
        getattr(server_args, "pvd_full_kv_fanin_max_slices", None) is None
        or getattr(server_args, "pvd_full_kv_fanin_response_bytes", None) is None
    ):
        raise ValueError(
            "PVD predictive retrieval requires bounded initial full-KV fan-in"
        )
    device = getattr(server_args, "device", None)
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise ValueError("PVD predictive retrieval currently requires a CUDA device")
    if getattr(server_args, "page_size", None) != 1:
        raise ValueError(
            "PVD predictive retrieval currently requires explicit --page-size 1"
        )
    decode_backend = getattr(server_args, "decode_attention_backend", None) or getattr(
        server_args, "attention_backend", None
    )
    if decode_backend != "torch_native":
        raise ValueError(
            "PVD predictive retrieval currently requires explicit "
            "torch_native Decode attention"
        )
    if not getattr(server_args, "disable_cuda_graph", False):
        raise ValueError("PVD predictive retrieval currently requires CUDA graphs off")

    # Candidate generation is part of predictive retrieval, and its two
    # lifetimes must stay separately budgeted from the copied KV bank/scratch.
    if not getattr(server_args, "pvd_draft_model_path", None):
        raise ValueError("PVD predictive retrieval requires --pvd-draft-model-path")
    _require_positive_int(
        "--pvd-draft-scratch-budget-bytes",
        getattr(server_args, "pvd_draft_scratch_budget_bytes", None),
    )
    _require_positive_int(
        "--pvd-draft-persistent-budget-bytes",
        getattr(server_args, "pvd_draft_persistent_budget_bytes", None),
    )
    fraction = getattr(server_args, "pvd_draft_mem_fraction_static", None)
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0 < fraction < 1
    ):
        raise ValueError(
            "--pvd-draft-mem-fraction-static is required and must be between 0 and 1"
        )
    if not serving_enabled:
        logger.warning(
            "PVD predictive-retrieval configuration passed validation, but this is "
            "configuration only: the production Scheduler does not construct the "
            "predictive pipeline and continues to use full-Prompt refresh"
        )
        return True

    # The serving opt-in is narrower than the existing configuration-only
    # surface. These conditions are consumed by the CUDA Scheduler binding and
    # must be proven before startup is allowed to construct it.
    if not getattr(server_args, "pvd_d_receive_rails", None):
        raise ValueError(
            "--pvd-cuda-predictive-serving requires --pvd-d-receive-rails "
            "to initialize the D sparse receive registry"
        )
    if server_args.speculative_algorithm is not None:
        raise ValueError(
            "--pvd-cuda-predictive-serving requires native speculative decoding off"
        )
    if not getattr(server_args, "disable_overlap_schedule", False):
        raise ValueError(
            "--pvd-cuda-predictive-serving requires overlap scheduling off"
        )
    attention_backend = getattr(server_args, "attention_backend", None)
    decode_attention_backend = getattr(server_args, "decode_attention_backend", None)
    if attention_backend != "torch_native" or decode_attention_backend not in (
        None,
        "torch_native",
    ):
        raise ValueError(
            "--pvd-cuda-predictive-serving requires attention_backend and "
            "decode_attention_backend to be torch_native"
        )
    if (
        getattr(server_args, "disaggregation_decode_enable_radix_cache", False)
        or getattr(server_args, "disaggregation_decode_enable_offload_kvcache", False)
        or getattr(server_args, "enable_hierarchical_cache", False)
        or getattr(server_args, "enable_hisparse", False)
        or getattr(server_args, "enable_prefill_context_parallel", False)
    ):
        raise ValueError(
            "--pvd-cuda-predictive-serving requires radix/offload/hierarchical "
            "cache, HiSparse and prefill context parallelism off"
        )
    _require_positive_int(
        "--pvd-draft-persistent-budget-bytes",
        getattr(server_args, "pvd_draft_persistent_budget_bytes", None),
    )
    predict_tokens = _require_positive_int(
        "--pvd-draft-predict-tokens",
        getattr(server_args, "pvd_draft_predict_tokens", None),
    )
    if predict_tokens > 16:
        raise ValueError(
            "--pvd-cuda-predictive-serving supports at most 16 draft tokens"
        )
    logger.info(
        "PVD CUDA predictive-serving arguments passed preflight; startup still "
        "must construct and install the model-specific CUDA serving binding "
        "before admitting requests"
    )
    return True


def handle_pvd_disaggregation(server_args: "ServerArgs") -> None:
    """Keep legacy PD untouched unless ``--disaggregation-topology pvd`` is set."""
    topology = server_args.disaggregation_topology
    if topology not in ("pd", "pvd"):
        raise ValueError(f"invalid disaggregation topology: {topology!r}")
    if topology == "pd":
        if getattr(server_args, "pvd_full_kv_fanin_triton_scatter", False):
            raise ValueError(
                "--pvd-full-kv-fanin-triton-scatter requires PVD Decode"
            )
        _validate_predictive_retrieval_config(server_args)
        return
    # PVD Decode always runs without the overlap scheduler. Normalize this
    # before validating the stricter opt-in so the accepted ServerArgs object
    # already satisfies the CUDA binding's runtime precondition.
    if getattr(server_args, "disaggregation_mode", None) == "decode":
        server_args.disable_overlap_schedule = True
        _validate_decode_token_reservation(server_args)
    _validate_predictive_retrieval_config(server_args)
    # The generic PD HTTP warmup posts a synthetic /generate without the
    # Gateway-selected PVD transfer/delivery/vector IDs. PVD must reject that
    # request, so the generic warmup would mark an otherwise ready worker
    # UnHealthy. ModelRunner's own kernel/graph warmup still runs at startup;
    # the first *end-to-end* warmup must be issued through the PVD Gateway.
    if not getattr(server_args, "skip_server_warmup", False):
        logger.warning(
            "PVD skips the generic PD HTTP warmup because it has no Gateway "
            "identities; validate generation through the PVD Gateway"
        )
    server_args.skip_server_warmup = True
    if (
        isinstance(server_args.pvd_kv_refresh_interval, bool)
        or not isinstance(server_args.pvd_kv_refresh_interval, int)
        or server_args.pvd_kv_refresh_interval <= 0
    ):
        raise ValueError("--pvd-kv-refresh-interval must be a positive integer")
    waiting_queue_bootstrap = getattr(server_args, "pvd_waiting_queue_bootstrap", False)
    if not isinstance(waiting_queue_bootstrap, bool):
        raise ValueError("--pvd-waiting-queue-bootstrap must be a boolean")
    fanin = getattr(server_args, "pvd_full_kv_fanin_max_slices", None)
    fanin_bytes = getattr(server_args, "pvd_full_kv_fanin_response_bytes", None)
    rank_packed = getattr(server_args, "pvd_full_kv_fanin_rank_packed", False)
    fused_scatter = getattr(server_args, "pvd_full_kv_fanin_triton_scatter", False)
    if type(fused_scatter) is not bool:
        raise ValueError("--pvd-full-kv-fanin-triton-scatter must be a boolean")
    if fused_scatter and not rank_packed:
        raise ValueError(
            "--pvd-full-kv-fanin-triton-scatter requires rank-packed fan-in"
        )
    if type(rank_packed) is not bool:
        raise ValueError("--pvd-full-kv-fanin-rank-packed must be a boolean")
    if rank_packed and (fanin is None or fanin_bytes is None):
        raise ValueError(
            "--pvd-full-kv-fanin-rank-packed requires bounded full-KV fan-in"
        )
    if (fanin, fanin_bytes) != (None, None):
        _require_positive_int("--pvd-full-kv-fanin-max-slices", fanin)
        _require_positive_int("--pvd-full-kv-fanin-response-bytes", fanin_bytes)
        if server_args.disaggregation_mode != "decode" or not waiting_queue_bootstrap:
            raise ValueError(
                "full-KV fan-in requires Decode with waiting-queue bootstrap"
            )
    if server_args.disaggregation_mode == "decode":
        server_args.disable_overlap_schedule = True
        logger.info(
            "PVD 3.0 uses a synchronous KV refresh barrier every %s Decode tokens",
            server_args.pvd_kv_refresh_interval,
        )
        if waiting_queue_bootstrap:
            logger.info(
                "PVD initial KV is pulled at final-waiting-queue entry; a "
                "request is not runnable until installation and ACK complete. "
                "Delivery/ACK waits are asynchronous; TP coordination and "
                "GPU installation remain on the scheduler thread."
            )

    # Reserve-before-allocate needs a budget before any staging tensor exists.
    server_args.pvd_transfer_staging_budget_bytes = _require_positive_int(
        "--pvd-transfer-staging-budget-bytes",
        server_args.pvd_transfer_staging_budget_bytes,
    )
    server_args.pvd_transfer_max_inflight = _require_positive_int(
        "--pvd-transfer-max-inflight", server_args.pvd_transfer_max_inflight
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
    if fanin is not None:
        supported_tp = (1, 2, 4)
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

    base_gpu_id = getattr(server_args, "base_gpu_id", 0)
    gpu_id_step = getattr(server_args, "gpu_id_step", 1)
    if (
        type(base_gpu_id) is not int
        or base_gpu_id < 0
        or type(gpu_id_step) is not int
        or gpu_id_step <= 0
    ):
        raise ValueError("PVD requires a nonnegative base GPU ID and positive GPU step")
    gpu_ids = tuple(
        base_gpu_id + rank * gpu_id_step for rank in range(server_args.tp_size)
    )
    rails = resolve_rank_rails(
        server_args.pvd_rank_rails,
        getattr(server_args, "disaggregation_ib_device", None),
        server_args.tp_size,
        gpu_ids=gpu_ids,
    )
    rail_mode = validate_rank_rail_names(rails)
    server_args.pvd_rank_rails = ",".join(rails)
    receive_rails = getattr(server_args, "pvd_d_receive_rails", None)
    if receive_rails is not None:
        if (
            server_args.disaggregation_mode != "decode"
            or server_args.tp_size != 1
            or fanin is None
            or not isinstance(receive_rails, str)
        ):
            raise ValueError("--pvd-d-receive-rails requires Decode TP1 full-KV fan-in")
        selected = tuple(item.strip() for item in receive_rails.split(","))
        validate_rank_rail_names(selected)
        if (
            not 1 <= len(selected) <= 8
            or len(set(selected)) != len(selected)
            or rails[0] not in selected
        ):
            raise ValueError(
                "--pvd-d-receive-rails needs distinct HCAs including the D rank rail"
            )
        server_args.pvd_d_receive_rails = ",".join(selected)
        if getattr(server_args, "pvd_cuda_predictive_serving", False):
            logger.info(
                "PVD D receive HCA sessions will be preflighted for CUDA "
                "predictive serving"
            )
        else:
            logger.warning(
                "PVD D multi-HCA receive sessions will be preflighted; this does "
                "not enable predictive retrieval in the serving Scheduler"
            )
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
        # Unchanged and unconditional. PVD's prediction-only draft path is
        # configured through --pvd-draft-* and never sets this, so reaching
        # SGLang's speculative generation loop stays impossible under PVD.
        raise ValueError("PVD does not support speculative decoding")
    if getattr(server_args, "pvd_draft_model_path", None):
        # A prediction branch competes for the same device as the committed
        # path, so its scratch is bounded explicitly rather than guessed.
        if getattr(server_args, "pvd_draft_scratch_budget_bytes", None) is None:
            raise ValueError(
                "--pvd-draft-scratch-budget-bytes is required with "
                "--pvd-draft-model-path: a prediction branch that is not "
                "bounded can starve the committed decode path"
            )
        server_args.pvd_draft_scratch_budget_bytes = _require_positive_int(
            "--pvd-draft-scratch-budget-bytes",
            server_args.pvd_draft_scratch_budget_bytes,
        )
        if getattr(server_args, "pvd_draft_persistent_budget_bytes", None) is not None:
            server_args.pvd_draft_persistent_budget_bytes = _require_positive_int(
                "--pvd-draft-persistent-budget-bytes",
                server_args.pvd_draft_persistent_budget_bytes,
            )
        server_args.pvd_draft_predict_tokens = _require_positive_int(
            "--pvd-draft-predict-tokens", server_args.pvd_draft_predict_tokens
        )
        fraction = getattr(server_args, "pvd_draft_mem_fraction_static", None)
        if fraction is not None and (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not 0 < fraction < 1
        ):
            raise ValueError("--pvd-draft-mem-fraction-static must be between 0 and 1")
        if getattr(server_args, "pvd_cuda_predictive_serving", False):
            logger.info(
                "PVD draft configuration recorded (%s); CUDA predictive serving "
                "will be installed during Scheduler startup. Native speculative "
                "decoding remains off.",
                server_args.pvd_draft_model_path,
            )
        else:
            logger.warning(
                "PVD draft configuration recorded (%s), but production predictive "
                "retrieval is not active: --pvd-draft-* does not instantiate the "
                "CPU reference pipeline in this serving Scheduler. The full-Prompt "
                "refresh path remains in use; speculative decoding remains off.",
                server_args.pvd_draft_model_path,
            )
    elif getattr(server_args, "pvd_draft_scratch_budget_bytes", None) is not None:
        raise ValueError(
            "--pvd-draft-scratch-budget-bytes has no meaning without "
            "--pvd-draft-model-path"
        )
    elif getattr(server_args, "pvd_draft_persistent_budget_bytes", None) is not None:
        raise ValueError(
            "--pvd-draft-persistent-budget-bytes has no meaning without "
            "--pvd-draft-model-path"
        )
    elif getattr(server_args, "pvd_draft_mem_fraction_static", None) is not None:
        raise ValueError(
            "--pvd-draft-mem-fraction-static has no meaning without "
            "--pvd-draft-model-path"
        )
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

    # Mooncake selects by physical GPU ID, not TP rank. The two are equal only
    # for the default base=0, step=1 placement.
    server_args.disaggregation_ib_device = json.dumps(
        {str(gpu_id): rail for gpu_id, rail in zip(gpu_ids, rails, strict=True)},
        separators=(",", ":"),
    )
    # Every Entry owns a complete prompt KV allocation. Prefix reuse on P would
    # make the exported allocation partial and violate the Entry manifest.
    server_args.disable_radix_cache = True
    server_args.disaggregation_decode_enable_radix_cache = False
    if not server_args.pvd_model_instance_id:
        revision = server_args.revision or "default"
        server_args.pvd_model_instance_id = f"{server_args.model_path}@{revision}"
