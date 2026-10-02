"""Real-shape CUDA byte gate and bounded sparse-packing microbenchmark.

Run from a repository checkout with its Python dependencies, on two idle GPUs.
This uses synthetic bytes with the current Qwen2.5-7B Prompt-KV layout. It does
not run CAGRA, Mooncake, a target model, serving cancellation, or an online test.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time

# Also works when launched directly from benchmark/ without installing this repo.
_REPO = Path(__file__).resolve().parents[1]
if (_REPO / "python" / "sglang").is_dir():
    sys.path.insert(0, str(_REPO / "python"))

import torch

from sglang.srt.disaggregation.pvd.kv_packer import PVD_TENSOR_LAYOUT
from sglang.srt.disaggregation.pvd.protocol import KVLayoutSignature, KVShardManifest
from sglang.srt.disaggregation.pvd.sparse_copy import copy_sparse_kv_into
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_pack_plan import (
    SparsePackCompletionUnknown,
    build_sparse_pack_plan,
)
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec, SparsePayloadError
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from sglang.srt.disaggregation.pvd.triton_sparse_pack import SparsePackWorkspace


LAYERS, HEADS, HEAD_DIM, VALID_TOKENS = 28, 2, 128, 2159
ELEMENT_BYTES = 2
_QUARANTINED_OWNERS = []


def ids(count):
    """Unique, deliberately unordered IDs; every group includes the last valid row."""
    values = [VALID_TOKENS - 1, 0, VALID_TOKENS // 2, 1, VALID_TOKENS - 2]
    candidate = 0
    while len(values) < count:
        token = (candidate * 701 + 37) % VALID_TOKENS
        candidate += 1
        if token not in values:
            values.append(token)
    return tuple(values[:count])


def layout_and_shard(rank, page_size):
    pages = (VALID_TOKENS + page_size - 1) // page_size
    rows = pages * page_size
    per_token = HEADS * HEAD_DIM * ELEMENT_BYTES
    layout = KVLayoutSignature(
        model_id="Qwen2.5-7B-shape-only",
        model_revision="synthetic-byte-gate",
        kv_dtype="torch.float16",
        page_size=page_size,
        num_layers=LAYERS,
        total_kv_heads=2 * HEADS,
        kv_heads_per_rank=HEADS,
        head_dim=HEAD_DIM,
        tp_size=2,
        pp_size=1,
        tensor_layout=PVD_TENSOR_LAYOUT,
        extra={
            "component_count": 2 * LAYERS,
            "component_dtypes": ["torch.float16"] * (2 * LAYERS),
            "component_token_shapes": [[HEADS, HEAD_DIM]] * (2 * LAYERS),
            "component_bytes_per_token": [per_token] * (2 * LAYERS),
        },
    )
    shard = KVShardManifest(
        rank=rank,
        rail="synthetic-no-rdma",
        expected_bytes=2 * LAYERS * rows * per_token,
        page_count=pages,
        last_page_valid_tokens=VALID_TOKENS - (pages - 1) * page_size,
        layer_start=0,
        layer_end=LAYERS,
    )
    return layout, shard


def manifest_for(layout, rank, name, groups):
    specs = tuple(
        SparseKVSpec(
            "synthetic-request", "synthetic-incarnation", name, 16,
            "synthetic-entry", "synthetic-index", "synthetic-mapping",
            layout.fingerprint, layer, rank * HEADS + head, ids(count),
        )
        for layer, head, count in groups
    )
    return SparseDeliveryManifest(specs, "torch.float16", HEAD_DIM)


def kwargs_for(manifest, layout, shard):
    return dict(
        manifest=manifest, layout=layout, shard=shard,
        entry_transfer_id="synthetic-entry", index_version="synthetic-index",
        id_mapping_version="synthetic-mapping", allow_cuda=True,
    )


def summary(values):
    ordered = sorted(values)
    return dict(
        count=len(values), median=statistics.median(values),
        p95=ordered[min(len(ordered) - 1, (95 * len(ordered) + 99) // 100 - 1)],
        minimum=ordered[0], maximum=ordered[-1],
        observations=[round(value, 6) for value in values],
    )


def memory(device):
    return dict(
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )


def owned_fence(device, owners):
    """A failed fence stays UNKNOWN even if an incidental later sync succeeds."""
    try:
        torch.cuda.synchronize(device)
    except BaseException as exc:
        _QUARANTINED_OWNERS.append(owners)
        raise SparsePackCompletionUnknown("probe CUDA completion unknown; owners retained") from exc


def copy_once(source, destination, *, mode, kwargs, budget, owner, prove_budget=False):
    """Include new metadata's whole lifecycle and identical two-device-fence policy.

    Destination/source are already owned. Host time excludes their allocation,
    GPU-event creation, byte validation, registration and RDMA. Event span also
    includes host pacing between GPU operations; it is not a pure kernel time.
    """
    device = source.device
    caller_device = torch.cuda.current_device()
    with torch.cuda.device(device):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)
    workspace = None
    complete = False
    started = time.perf_counter()
    try:
        with torch.cuda.device(device):
            start.record(stream)
        if mode == "triton":
            workspace = SparsePackWorkspace(
                kwargs["manifest"], shard=kwargs["shard"], layout=kwargs["layout"],
                device=device, budget=budget, owner=owner,
            )
        if prove_budget:
            expected = destination.numel() + (workspace.bytes if workspace else 0)
            if budget.snapshot()["used_staging_bytes"] != expected:
                raise AssertionError("staging/metadata budget is not held before launch")
        copy_sparse_kv_into(source, destination, **kwargs, fused_workspace=workspace)
        owned_fence(device, (source, destination, workspace, budget, start, end))
        if workspace is not None:
            workspace.release_after_fence()
        with torch.cuda.device(device):
            end.record(stream)
        # The serving adapter retains an outer pre-PUT device fence as well.
        owned_fence(device, (source, destination, workspace, budget, start, end))
        complete = True
        elapsed_ms = (time.perf_counter() - started) * 1000
        event_ms = start.elapsed_time(end)
        if torch.cuda.current_device() != caller_device:
            raise AssertionError("packing did not restore the caller's current CUDA device")
        if budget.snapshot()["used_staging_bytes"] != destination.numel():
            raise AssertionError("metadata charge did not return to the staging-only charge")
        return elapsed_ms, event_ms
    except BaseException as exc:
        if isinstance(exc, SparsePackCompletionUnknown):
            _QUARANTINED_OWNERS.append((source, destination, workspace, budget, start, end))
            raise
        if not complete:
            try:
                torch.cuda.synchronize(device)
            except BaseException as fence_error:
                _QUARANTINED_OWNERS.append((source, destination, workspace, budget, start, end))
                raise SparsePackCompletionUnknown("probe packing completion unknown; owners retained") from fence_error
        if workspace is not None:
            workspace.release_after_fence()
        raise


def byte_gate(source, layout, shard):
    manifest = manifest_for(layout, shard.rank, "endpoints", (
        (0, 0, 1), (0, 1, 3), (LAYERS - 1, 0, 28), (LAYERS - 1, 1, 32),
    ))
    kwargs = kwargs_for(manifest, layout, shard)
    plan = build_sparse_pack_plan(manifest, layout, shard)
    invalid_kind = (
        "padding" if shard.page_count * layout.page_size > VALID_TOKENS
        else "out_of_range"
    )
    budget = TransferBudget(2 * manifest.nbytes + plan.metadata_bytes, 1)
    budget.reserve("gate-staging", 2 * manifest.nbytes, 0)
    actual = reference = workspace = None
    safe = False
    try:
        actual = torch.empty(manifest.nbytes, dtype=torch.uint8, device=source.device)
        reference = torch.empty_like(actual)
        # Independent existing CUDA row-copy oracle, including K/V pairing.
        copy_sparse_kv_into(source, reference, **kwargs)
        workspace = SparsePackWorkspace(
            manifest, shard=shard, layout=layout, device=source.device,
            budget=budget, owner="gate-metadata",
        )
        if budget.snapshot()["used_staging_bytes"] != 2 * manifest.nbytes + plan.metadata_bytes:
            raise AssertionError("byte gate metadata was not budgeted")
        copy_sparse_kv_into(source, actual, **kwargs, fused_workspace=workspace)
        owned_fence(source.device, (source, actual, reference, workspace, budget))
        if not torch.equal(actual, reference):
            mismatch = int(torch.nonzero(actual != reference).flatten()[0].item())
            raise AssertionError(f"FP16 byte mismatch at destination offset {mismatch}")
        workspace.release_after_fence()
        workspace = None
        rejection_checks = []
        for group_index in (0, len(manifest.specs) - 1):
            specs = list(manifest.specs)
            spec = specs[group_index]
            specs[group_index] = replace(spec, token_ids=(VALID_TOKENS,) + spec.token_ids[1:])
            invalid = SparseDeliveryManifest(tuple(specs), manifest.dtype, manifest.head_dim)
            for mode in ("torch", "triton"):
                actual.fill_(0xA7)
                if mode == "triton":
                    workspace = SparsePackWorkspace(
                        manifest, shard=shard, layout=layout, device=source.device,
                        budget=budget, owner=f"padding-metadata-{group_index}",
                    )
                owned_fence(source.device, (source, actual, reference, workspace, budget))
                try:
                    copy_sparse_kv_into(source, actual, **{**kwargs, "manifest": invalid}, fused_workspace=workspace)
                except SparsePayloadError:
                    pass
                else:
                    raise AssertionError(f"{invalid_kind} token was accepted")
                owned_fence(source.device, (source, actual, reference, workspace, budget))
                if not bool(torch.all(actual == 0xA7).item()):
                    raise AssertionError(f"{invalid_kind} rejection occurred after destination writes")
                if workspace is not None:
                    workspace.release_after_fence()
                    workspace = None
                rejection_checks.append(dict(mode=mode, group=group_index, token=VALID_TOKENS,
                                             reason=invalid_kind))
        owned_fence(source.device, (source, actual, reference, workspace, budget))
        safe = True
        budget.release("gate-staging")
        if budget.snapshot()["used_staging_bytes"] != 0:
            raise AssertionError("byte gate budget did not return to zero")
        return dict(
            equal_bytes=True, compared_bytes=manifest.nbytes,
            groups=[dict(layer=s.layer, global_head=s.kv_head, count=len(s.token_ids),
                         contains_last_valid=(VALID_TOKENS - 1 in s.token_ids)) for s in manifest.specs],
            metadata_bytes=plan.metadata_bytes, invalid_token_rejections=rejection_checks,
            final_used_staging_bytes=0,
        )
    except SparsePackCompletionUnknown:
        _QUARANTINED_OWNERS.append((source, actual, reference, workspace, budget))
        raise
    finally:
        if not safe and not any(bundle[0] is source for bundle in _QUARANTINED_OWNERS):
            try:
                torch.cuda.synchronize(source.device)
            except BaseException:
                _QUARANTINED_OWNERS.append((source, actual, reference, workspace, budget))
                raise
            if workspace is not None:
                workspace.release_after_fence()
            budget.release("gate-staging")


def benchmark_case(source, layout, shard, name, groups, warmup, repeats):
    manifest = manifest_for(layout, shard.rank, name, groups)
    kwargs = kwargs_for(manifest, layout, shard)
    plan = build_sparse_pack_plan(manifest, layout, shard)
    budget = TransferBudget(manifest.nbytes + plan.metadata_bytes, 1)
    budget.reserve("bench-staging", manifest.nbytes, 0)
    destination = None
    blocks = []
    try:
        destination = torch.empty(manifest.nbytes, dtype=torch.uint8, device=source.device)
        # Balanced block order; same warmup count in each block. Compilation and
        # first allocator growth are warmed, not silently charged to only one arm.
        for block_index, mode in enumerate(("torch", "triton", "triton", "torch")):
            for iteration in range(warmup):
                copy_once(source, destination, mode=mode, kwargs=kwargs, budget=budget,
                          owner=f"warm-{block_index}-{iteration}")
            torch.cuda.reset_peak_memory_stats(source.device)
            host_ms, event_ms = [], []
            for iteration in range(repeats):
                wall, event = copy_once(
                    source, destination, mode=mode, kwargs=kwargs, budget=budget,
                    owner=f"bench-{block_index}-{iteration}",
                )
                host_ms.append(wall)
                event_ms.append(event)
            blocks.append(dict(
                block=block_index, mode=mode, host_wall_ms=summary(host_ms),
                cuda_event_span_ms=summary(event_ms), torch_memory=memory(source.device),
            ))
        owned_fence(source.device, (source, destination, None, budget))
        budget.release("bench-staging")
        if budget.snapshot()["used_staging_bytes"] != 0:
            raise AssertionError("benchmark budget did not return to zero")
        aggregate = {}
        for mode in ("torch", "triton"):
            selected = [block for block in blocks if block["mode"] == mode]
            aggregate[mode] = {
                key: summary([value for block in selected for value in block[key]["observations"]])
                for key in ("host_wall_ms", "cuda_event_span_ms")
            }
        return dict(
            name=name, token_counts=[len(s.token_ids) for s in manifest.specs],
            staging_bytes=manifest.nbytes, metadata_bytes=plan.metadata_bytes,
            budget_limit_bytes=manifest.nbytes + plan.metadata_bytes,
            blocks=blocks, aggregate=aggregate, final_used_staging_bytes=0,
        )
    except BaseException:
        # Even a failed assertion cannot retire a buffer still referenced by CUDA.
        if not any(bundle[0] is source for bundle in _QUARANTINED_OWNERS):
            try:
                torch.cuda.synchronize(source.device)
            except BaseException:
                _QUARANTINED_OWNERS.append((source, destination, None, budget))
                raise
            budget.release("bench-staging")
        raise


def run_device(rank, device_index, caller_index, warmup, repeats, page_size):
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(caller_index)
    caller_before = torch.cuda.current_device()
    layout, shard = layout_and_shard(rank, page_size)
    padded_rows = shard.page_count * layout.page_size
    expected_source_bytes = 2 * LAYERS * padded_rows * HEADS * HEAD_DIM * ELEMENT_BYTES
    if (shard.expected_bytes != expected_source_bytes
            or (shard.page_count - 1) * layout.page_size + shard.last_page_valid_tokens != VALID_TOKENS):
        raise AssertionError("source shape/page extent does not match the requested fixture")
    before = memory(device)
    torch.cuda.reset_peak_memory_stats(device)
    source = generator = None
    try:
        source = torch.empty(shard.expected_bytes, dtype=torch.uint8, device=device)
        generator = torch.Generator(device=device).manual_seed(1603 + rank)
        # Arbitrary raw byte patterns exercise bit preservation, including FP16
        # encodings of signed zeros, infinities and NaNs; no float math is performed.
        source.random_(0, 256, generator=generator)
        owned_fence(device, (source, None, None, None, generator))
        gate = byte_gate(source, layout, shard)
        gate_memory = memory(device)
        cases = [
            benchmark_case(source, layout, shard, "layer0_1x3", ((0, 0, 1), (0, 1, 3)), warmup, repeats),
            benchmark_case(source, layout, shard, "layer14_4x4", ((14, 0, 4), (14, 1, 4)), warmup, repeats),
            benchmark_case(source, layout, shard, "layer27_28x32", ((27, 0, 28), (27, 1, 32)), warmup, repeats),
        ]
        owned_fence(device, (source, None, None, None, generator))
        if torch.cuda.current_device() != caller_before:
            raise AssertionError("cross-device probe did not restore the caller CUDA device")
    except BaseException as exc:
        # Source generation can itself fail after enqueuing GPU work. Retain
        # both the byte allocation and explicit RNG owner through its proof.
        if isinstance(exc, SparsePackCompletionUnknown) or any(
            bundle[0] is source for bundle in _QUARANTINED_OWNERS
        ):
            _QUARANTINED_OWNERS.append((source, None, None, None, generator))
            raise
        owned_fence(device, (source, None, None, None, generator))
        raise
    del source, generator
    torch.cuda.synchronize(device)
    after = memory(device)
    if after["allocated_bytes"] != before["allocated_bytes"]:
        raise AssertionError("probe source/staging/workspace allocations remain live")
    return dict(
        rank=rank, device=str(device), caller_device=caller_before,
        device_name=torch.cuda.get_device_name(device),
        capability=list(torch.cuda.get_device_capability(device)),
        shape=dict(layers=LAYERS, heads_per_rank=HEADS, head_dim=HEAD_DIM,
                   dtype="torch.float16", valid_tokens=VALID_TOKENS,
                   page_size=layout.page_size, padded_rows=padded_rows),
        fixture_scope=("current-launcher-page1-shape" if page_size == 1
                       else "additional-page2-partial-page-shape"),
        source_bytes=shard.expected_bytes, padded_rows=padded_rows,
        byte_gate=gate, byte_gate_memory=gate_memory, benchmark=cases,
        memory_before=before, memory_after=after,
        retained_allocator_cache_bytes=after["reserved_bytes"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1", help="Two real device indices, first then reversed")
    parser.add_argument("--page-sizes", default="1,2",
                        help="Comma-separated 1/2: current page1 shape and additional padded page2 shape")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    try:
        devices = tuple(int(value) for value in args.devices.split(","))
    except ValueError:
        parser.error("--devices must contain two integer indices")
    if len(devices) != 2 or len(set(devices)) != 2 or min(devices) < 0:
        parser.error("two different non-negative CUDA devices are required")
    try:
        page_sizes = tuple(int(value) for value in args.page_sizes.split(","))
    except ValueError:
        parser.error("--page-sizes must contain 1 and/or 2")
    if not page_sizes or len(set(page_sizes)) != len(page_sizes) or any(value not in (1, 2) for value in page_sizes):
        parser.error("--page-sizes must contain unique values from 1/2")
    if not 1 <= args.warmup <= 100 or not 3 <= args.repeats <= 100:
        parser.error("warmup must be 1..100 and repeats 3..100")
    if not torch.cuda.is_available() or max(devices) >= torch.cuda.device_count():
        parser.error("both requested real CUDA devices must be available")
    import triton
    result = dict(
        kind="synthetic-real-shape-byte-gate-and-microbenchmark", torch_version=torch.__version__,
        triton_version=triton.__version__, warmup_per_block=args.warmup,
        measured_repeats_per_block=args.repeats, order=["torch", "triton", "triton", "torch"],
        shape=dict(layers=LAYERS, heads_per_rank=HEADS, head_dim=HEAD_DIM,
                   dtype="torch.float16", valid_tokens=VALID_TOKENS,
                   page_sizes=list(page_sizes)),
        timing_scope="Preallocated source/staging; includes validation, fresh Triton metadata allocation/upload/release, copy, two device fences. Excludes source/staging allocation, event creation, registration, RDMA. CUDA event span includes host pacing; not pure kernel time.",
        memory_scope="Torch allocator counters only; reserved cache is not live ownership and is not refunded physical GPU memory. Source represents an already-owned Entry; only staging and metadata use TransferBudget here.",
        limits=["synthetic bytes, not actual model KV or retrieval-quality evidence",
                "requires idle devices; no concurrent serving-load measurement",
                "not CAGRA, RDMA, cancellation/unknown-fault, online latency or physical process peak validation",
                "28x32 groups are capacity/boundary coverage, not the typical steady max_new16 payload",
                "1x3/4x4 are specified small-payload probes, not an observed live payload distribution",
                "two-fence policy retained; no stream/event serving optimization"],
        devices=[],
    )
    for page_size in page_sizes:
        for rank, device in enumerate(devices):
            result["devices"].append(run_device(rank, device, devices[1 - rank],
                                               args.warmup, args.repeats, page_size))
    result["status"] = "passed"
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            raise
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__, error=str(exc),
                              quarantined_owner_bundles=len(_QUARANTINED_OWNERS))))
        raise
