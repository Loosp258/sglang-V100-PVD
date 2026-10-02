"""Native sparse scatter gate using two registered immutable original pools.

48 local-session cases qualify exact bytes, shared original MR ownership,
concurrent workers and receive ordering. This is not a cross-node performance
benchmark, and does not inject unresolved native failures. On any exception all
possibly active registrations, authorizations and budgets remain quarantined.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import types
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
package = types.ModuleType("sglang")
package.__path__ = [str(ROOT / "python" / "sglang")]
sys.modules["sglang"] = package

from sglang.srt.disaggregation.pvd.cuda_receive_ordering import CUDAReceiveOrdering
from sglang.srt.disaggregation.pvd.kv_packer import PVD_TENSOR_LAYOUT
from sglang.srt.disaggregation.pvd.multi_rail_receive import create_native_receive_group
from sglang.srt.disaggregation.pvd.oasis_receive_slots import OasisReceiveSlotPool
from sglang.srt.disaggregation.pvd.protocol import (
    KVEntryKey, KVLayoutSignature, KVShardManifest,
    PVD_TRANSFER_LIFECYCLE_PROTOCOL, WriteIdentity,
)
from sglang.srt.disaggregation.pvd.sparse_batch_plan import build_sparse_batch_plan
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
from sglang.srt.disaggregation.pvd.transfer_authorization import WriteAuthorization
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard, TransferBudget, TransportState,
)

_QUARANTINE = []
_PROGRESS = {}
ROWS = (1, 2, 8, 16, 32, 64)
LAYERS, HEADS, HEAD_DIM, PAGE_SIZE, PROMPT_TOKENS = 28, 2, 128, 16, 83


def fixture(rank, rail):
    """A real FP16 component-major shard, with the final 13 padded rows."""
    page_count = (PROMPT_TOKENS + PAGE_SIZE - 1) // PAGE_SIZE
    padded_rows = page_count * PAGE_SIZE
    head_bytes = HEAD_DIM * 2
    component_bpt = HEADS * head_bytes
    extra = dict(
        component_count=2 * LAYERS,
        component_dtypes=["torch.float16"] * (2 * LAYERS),
        component_token_shapes=[[HEADS, HEAD_DIM] for _ in range(2 * LAYERS)],
        component_bytes_per_token=[component_bpt] * (2 * LAYERS),
    )
    layout = KVLayoutSignature(
        model_id="native-Qwen2.5-7B-shape", model_revision="fixture-v1",
        kv_dtype="torch.float16", page_size=PAGE_SIZE, num_layers=LAYERS,
        total_kv_heads=2 * HEADS, kv_heads_per_rank=HEADS, head_dim=HEAD_DIM,
        tp_size=2, pp_size=1, tensor_layout=PVD_TENSOR_LAYOUT, extra=extra,
    )
    count = 2 * LAYERS * padded_rows * HEADS * HEAD_DIM
    values = ((torch.arange(count, dtype=torch.int64) + rank * 83) % 997).to(torch.float16)
    values.div_(16)
    # Independent shaped CPU source, not planner-derived offsets.
    components = values.reshape(2 * LAYERS, padded_rows, HEADS, HEAD_DIM)
    shard = KVShardManifest(
        rank=rank, rail=rail, expected_bytes=values.numel() * 2,
        page_count=page_count, last_page_valid_tokens=PROMPT_TOKENS % PAGE_SIZE,
        layer_start=0, layer_end=LAYERS,
    )
    page_bytes = shard.expected_bytes // shard.page_count
    return layout, shard, components, 2 * page_bytes, page_bytes


def manifest_for(rank, rows, executor_round, worker_slot, *, layout, key):
    counts = (rows,) if rows == 1 else (rows // 2, rows - rows // 2)
    layer = (executor_round * 13 + worker_slot * 7 + rows) % LAYERS
    operation = uuid.uuid4().hex
    specs = tuple(SparseKVSpec(
        "native-req", "native-inc", operation, 1, key.transfer_id,
        "native-index", "native-mapping", layout.fingerprint, layer,
        rank * HEADS + head,
        tuple((PROMPT_TOKENS - 1 - 2 * j - worker_slot * 3) % PROMPT_TOKENS
              for j in range(count)),
    ) for head, count in enumerate(counts))
    return SparseDeliveryManifest(specs, "torch.float16", HEAD_DIM)


def oracle_bytes(components, manifest, rank):
    groups = []
    for spec in manifest.specs:
        local_head = spec.kv_head - rank * HEADS
        groups.append(torch.stack((
            components[spec.layer, list(spec.token_ids), local_head],
            components[spec.layer + LAYERS, list(spec.token_ids), local_head],
        )).contiguous().view(torch.uint8).reshape(-1))
    return torch.cat(groups)


def poll_success(adapter, handle, expected_bytes, timeout):
    deadline = time.monotonic() + timeout
    while not handle.transport_state.is_locally_safe_to_release:
        adapter.poll(handle)
        if handle.transport_state == TransportState.UNKNOWN:
            raise RuntimeError("native aggregate terminal is UNKNOWN")
        if time.monotonic() >= deadline:
            raise TimeoutError("native scatter batch did not become terminal")
        time.sleep(0.001)
    if (
        handle.transport_state != TransportState.TERMINAL_SUCCESS
        or handle.transferred_bytes != expected_bytes
        or not adapter.cleanup_complete(handle)
    ):
        raise RuntimeError("native batch lacks exact aggregate terminal/cleanup proof")


def run(args):
    from sglang.srt.disaggregation.pvd.mooncake_engine import MooncakePVDTransferEngine

    device, caller = torch.device(args.device), torch.device(args.current_device)
    if (
        device.type != "cuda" or device.index is None
        or caller.type != "cuda" or caller.index is None or device == caller
        or not torch.cuda.is_available()
        or max(device.index, caller.index) >= torch.cuda.device_count()
    ):
        raise ValueError("two explicit distinct valid CUDA devices required")
    torch.cuda.set_device(caller)
    baseline_current = torch.cuda.current_device()
    budget = TransferBudget(32 << 20, 16)
    receiver = create_native_receive_group(
        hostname=args.hostname, gpu_id=device.index,
        rails=tuple(dict.fromkeys(args.rails)), transfer_budget=budget,
    )
    pool = OasisReceiveSlotPool(
        receiver, budget, device=device, receiver_epoch="native-D",
        slots_per_rank=2, capacity_bytes=64 * 512,
    )
    _QUARANTINE.append((pool, receiver, budget))
    _PROGRESS.update(observations=[], source_registrations=[])
    endpoints = {rank: receiver.health()["rails"][rail]["session_id"]
                 for rank, rail in enumerate(args.rails)}
    source_devices = (device, caller)
    sources = {}
    key = KVEntryKey.new("native-model", "native-prompt")
    for rank, rail in enumerate(args.rails):
        source_device = source_devices[rank]
        adapter = MooncakePVDTransferEngine(
            hostname=args.hostname, gpu_id=source_device.index, rail=rail, budget=budget,
        )
        adapter.require_native_batch()
        layout, shard, host, offset, suffix = fixture(rank, rail)
        owner = f"native-original-pool:{rank}"
        capacity = offset + shard.expected_bytes + suffix
        budget.reserve(owner, capacity, 0)
        with torch.cuda.device(source_device):
            buffer = torch.full((capacity,), 211, dtype=torch.uint8, device=source_device)
            _QUARANTINE.append((adapter, buffer, owner))
            buffer[offset:offset + shard.expected_bytes].copy_(host.view(torch.uint8).reshape(-1))
            torch.cuda.synchronize(source_device)
            registration = adapter.register_memory(
                buffer, endpoint=adapter._engine.get_session_id(), rank=rank, rail=rail,
                metadata={"role": "vector", "page_bytes": suffix, "native_original_pool": True},
            )
        released = []
        entry_guard = ResourceGuard((registration, offset, shard.expected_bytes),
                                    lambda released=released: released.append(True))
        source = dict(adapter=adapter, registration=registration, layout=layout,
                      shard=shard, host=host, allocation_offset=offset, owner=owner,
                      entry_guard=entry_guard, released=released, device=source_device)
        sources[rank] = source
        _QUARANTINE.append(source)
        _PROGRESS["source_registrations"].append(registration.descriptor.to_dict())

    inventory = [source["registration"].descriptor.to_dict() for source in sources.values()]
    generations, physical_ids = set(), set()
    for executor_round in range(2):
        with ThreadPoolExecutor(max_workers=2) as executor:
            for rows in ROWS:
                acquired = threading.Barrier(3)
                submitted = {rank: threading.Barrier(2) for rank in (0, 1)}
                pinned = {rank: threading.Barrier(2) for rank in (0, 1)}
                def worker(worker_slot):
                    ordering = CUDAReceiveOrdering(device)
                    leases, operations = [], []
                    try:
                        for rank, rail in enumerate(args.rails):
                            source = sources[rank]
                            manifest = manifest_for(rank, rows, executor_round, worker_slot,
                                                    layout=source["layout"], key=key)
                            identity = WriteIdentity(
                                PVD_TRANSFER_LIFECYCLE_PROTOCOL, f"native-V{rank}",
                                "native-D", uuid.uuid4().hex, "pending-registration",
                                uuid.uuid4().hex, rank, key,
                            )
                            lease = pool.acquire(manifest, identity, endpoint=endpoints[rank],
                                rail=rail, device=device, ordering=ordering)
                            # Each exclusive slot has no old writer/reader. A
                            # checked tail sentinel proves bounded native writes.
                            with torch.cuda.device(device):
                                lease._slot.buffer.fill_(197)
                                torch.cuda.synchronize(device)
                            receive_owner = "native-receive:" + lease.identity.generation
                            budget.reserve(receive_owner, 0, 1)
                            leases.append((rank, rail, lease, manifest, receive_owner))
                        acquired.wait(timeout=args.timeout)
                        observations = []
                        for rank, rail, lease, manifest, receive_owner in leases:
                            source = sources[rank]
                            adapter, registration = source["adapter"], source["registration"]
                            authorization = WriteAuthorization(lease.identity, source["entry_guard"])
                            operations.append((authorization, source, lease, receive_owner))
                            _QUARANTINE.append(authorization)
                            with torch.cuda.device(source["device"]):
                                plan = build_sparse_batch_plan(
                                    manifest, source["layout"], source["shard"],
                                    entry_transfer_id=key.transfer_id,
                                    index_version="native-index", id_mapping_version="native-mapping",
                                    allocation_offset=source["allocation_offset"], registration=registration,
                                )
                                if any(local.registration is not registration for local in plan.slices):
                                    raise AssertionError("scatter plan replaced the original source MR")
                                lease.identity.validate_destination(lease.registration.descriptor)
                                authorization.begin(lease.identity)
                                started = time.perf_counter()
                                handle = adapter.submit_batch_put(
                                    plan.slices, lease.registration.descriptor,
                                    remote_offsets=plan.remote_offsets,
                                )
                                _QUARANTINE.append((plan, handle, authorization, lease))
                                if handle.transport_state != TransportState.IN_FLIGHT:
                                    raise RuntimeError("scatter native submission was not admitted")
                                submitted[rank].wait(timeout=args.timeout)
                                # Both callbacks own the original MR until poll.
                                with source["entry_guard"]._lock:
                                    entry_pins = len(source["entry_guard"]._owners)
                                tracked = adapter.health()["lifecycle"]["tracked_transfers"]
                                pinned[rank].wait(timeout=args.timeout)
                                if entry_pins != 2 or tracked != 2:
                                    raise AssertionError("native source lifetime was not retained")
                                poll_success(adapter, handle, manifest.nbytes, args.timeout)
                                batch_seconds = time.perf_counter() - started
                            authorization.close()
                            authorization.observe_terminal(lease.identity, handle.transport_state)
                            fence = authorization.fence(lease.identity)
                            if not fence["fenced"] or not authorization.cleanup_complete:
                                raise AssertionError("native source authorization did not close")
                            with torch.cuda.device(device):
                                ordering.after_remote_write(lease.registration)
                                observed = lease.buffer.cpu()
                                tail = lease._slot.buffer[manifest.nbytes:].cpu()
                                oracle = oracle_bytes(source["host"], manifest, rank)
                                if not torch.equal(observed, oracle):
                                    raise AssertionError("direct scatter payload differs from CPU oracle")
                                if not bool(tail.eq(197).all()):
                                    raise AssertionError("native scatter wrote beyond the bounded destination")
                                payloads = manifest.payload_views(lease.buffer)
                                try:
                                    if sum(payload.nbytes for payload in payloads) != manifest.nbytes:
                                        raise AssertionError("sparse K/V payload extent differs")
                                finally:
                                    for payload in payloads:
                                        payload.close()
                                torch.cuda.synchronize(device)
                            observation = dict(
                                executor_round=executor_round, worker_slot=worker_slot,
                                worker_thread=threading.get_ident(), rank=rank, rows=rows,
                                manifest=manifest.to_dict(), nbytes=manifest.nbytes,
                                expected_bytes=manifest.nbytes,
                                device=str(device), current_device=str(caller),
                                source_region_id=registration.descriptor.region_id,
                                original_source_region_id=registration.descriptor.region_id,
                                source_registration_count=adapter.health()["registered_regions"],
                                staging_registration_count=0,
                                source_device=str(source["device"]), source_original_mr=True,
                                source_address=registration.descriptor.address,
                                allocation_offset=source["allocation_offset"],
                                source_entry_bytes=source["shard"].expected_bytes,
                                source_slices=[dict(offset=local.offset, length=local.length)
                                               for local in plan.slices],
                                remote_offsets=plan.remote_offsets, slice_count=len(plan.slices),
                                slices=len(plan.slices),
                                generation=lease.identity.generation, region_id=lease.identity.region_id,
                                destination_address=lease.registration.descriptor.address,
                                physical_register_calls=lease.physical_register_calls,
                                terminal_state=handle.transport_state.value,
                                terminal_success=handle.transport_state == TransportState.TERMINAL_SUCCESS,
                                cleanup_complete=adapter.cleanup_complete(handle),
                                transferred_bytes=handle.transferred_bytes, exact_bytes=True,
                                destination_sentinel_exact=True,
                                write_fence=fence, source_pins_observed=entry_pins,
                                tracked_transfers_observed=tracked, batch_seconds=batch_seconds,
                                payload_sha256=hashlib.sha256(observed.numpy().tobytes()).hexdigest(),
                            )
                            lease.release_after_proof()
                            budget.release(receive_owner)
                            observations.append(observation)
                        return observations
                    except BaseException:
                        acquired.abort()
                        for barrier in submitted.values():
                            barrier.abort()
                        for barrier in pinned.values():
                            barrier.abort()
                        for _, _, lease, _, _ in leases:
                            if lease._active and lease._slot.unknown is None:
                                lease.quarantine("native scatter callback lacks completed ownership proof")
                        raise

                futures = [executor.submit(worker, slot) for slot in (0, 1)]
                acquired.wait(timeout=args.timeout)
                if pool.snapshot()["physical_registrations"] != 4:
                    raise AssertionError("native destinations did not retain four bounded MRs")
                for future in futures:
                    observations = future.result(timeout=args.timeout * 4)
                    for observation in observations:
                        if observation["generation"] in generations:
                            raise AssertionError("destination generation reused")
                        generations.add(observation["generation"])
                        physical_ids.add(observation["region_id"])
                    _PROGRESS["observations"].extend(observations)

    observations = _PROGRESS["observations"]
    before_close = pool.snapshot()
    source_before = {rank: source["adapter"].health() for rank, source in sources.items()}
    if (
        len(observations) != 48 or len(generations) != 48 or len(physical_ids) != 4
        or before_close["physical_register_calls"] != 4
        or before_close["leased_slots"] or before_close["unknown_slots"]
        or any(source["registration"].descriptor.to_dict() != inventory[rank]
               for rank, source in sources.items())
        or any(state["registered_regions"] != 1
               or state["lifecycle"]["tracked_transfers"]
               or state["lifecycle"]["unknown_transfers"]
               or state["submit_timing"]["batch_submit_calls"] != 24
               or state["submit_timing"]["native_submit_calls"]
               for state in source_before.values())
    ):
        raise AssertionError("native scatter inventory/generation gate did not pass")
    for source in sources.values():
        observed_pool = source["registration"].buffer.cpu()
        offset, extent = source["allocation_offset"], source["shard"].expected_bytes
        if (
            not bool(observed_pool[:offset].eq(211).all())
            or not bool(observed_pool[offset + extent:].eq(211).all())
            or not torch.equal(observed_pool[offset:offset + extent],
                               source["host"].view(torch.uint8).reshape(-1))
        ):
            raise AssertionError("native scatter changed immutable source Entry/pool sentinels")
    torch.cuda.set_device(caller)
    pool.close()
    if torch.cuda.current_device() != baseline_current:
        raise AssertionError("receive pool close changed caller CUDA device")
    for source in sources.values():
        source["entry_guard"].request_release()
        if source["entry_guard"].value is not None or source["released"] != [True]:
            raise AssertionError("original Entry pin was not completely retired")
        source["adapter"].release_memory(source["registration"])
        budget.release(source["owner"])
    after_close = pool.snapshot()
    accounting = budget.snapshot()
    source_after = {rank: source["adapter"].health() for rank, source in sources.items()}
    receive_after = receiver.health()
    if (
        after_close["physical_releases"] != 4 or not after_close["closed"]
        or accounting["used_staging_bytes"] or accounting["used_inflight"]
        or receive_after["registered_destinations"]
        or any(state["registered_regions"] or state["lifecycle"]["tracked_transfers"]
               or state["lifecycle"]["unknown_transfers"]
               for state in receive_after["rails"].values())
        or any(state["registered_regions"] or state["lifecycle"]["tracked_transfers"]
               or state["lifecycle"]["unknown_transfers"] for state in source_after.values())
    ):
        raise AssertionError("native source/receive owners or budgets did not retire")
    _QUARANTINE.clear()
    return dict(
        status="passed", mode="direct_sparse_batch_put", transport="mooncake_local_session_scatter",
        device=str(device), current_device=str(caller), source_devices=list(map(str, source_devices)),
        exact_byte_cases=48, two_executor_pools=True, rows=ROWS,
        fixture=dict(layers=LAYERS, local_kv_heads=HEADS, head_dim=HEAD_DIM,
                     page_size=PAGE_SIZE, prompt_tokens=PROMPT_TOKENS,
                     padded_rows=96, nonzero_allocation_pages=2),
        source_registrations=inventory, source_physical_register_calls=2,
        staging_registration_count=0, immutable_source_bytes_exact=True,
        all_sources_unregistered=True, all_destinations_unregistered=True,
        all_owners_retired=True,
        source_before_close=source_before, source_after_close=source_after,
        before_close=before_close, after_close=after_close, transfer_budget=accounting,
        receive_health=receive_after, cases=observations, observations=observations,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--current-device", default="cuda:0")
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--rails", nargs=2, required=True)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "artifacts").resolve()) or args.timeout <= 0:
        raise ValueError("positive timeout and project artifacts output required")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
    except BaseException as error:
        result = dict(status="failed", error_type=type(error).__name__, error=str(error),
                      mode="direct_sparse_batch_put",
                      cases=_PROGRESS.get("observations", []),
                      source_registrations=_PROGRESS.get("source_registrations", []))
        if _QUARANTINE and isinstance(_QUARANTINE[0], tuple):
            pool, receiver, budget = _QUARANTINE[0]
            result.update(pool=pool.snapshot(), receive_health=receiver.health(),
                          transfer_budget=budget.snapshot())
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "observations")}, sort_keys=True))


if __name__ == "__main__":
    main()
