"""Real local Mooncake CUDA MR/PUT acceptance for request-owned receive slots.

This bounded isolated-node gate exercises native local-session PUTs, exact bytes,
SYNC_MEMOPS, repeated logical generations and physical unregister. It does not
replace online V->D RDMA testing or inject real unknown native operations.
"""

import argparse
import dataclasses
import json
import pathlib
import sys
import threading
import time
import types
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
# The gate imports PVD modules only; avoid the optional user-facing top-level
# package imports when running in an isolated validation environment.
package = types.ModuleType("sglang")
package.__path__ = [str(ROOT / "python" / "sglang")]
sys.modules["sglang"] = package

from sglang.srt.disaggregation.pvd.cuda_receive_ordering import CUDAReceiveOrdering
from sglang.srt.disaggregation.pvd.multi_rail_receive import create_native_receive_group
from sglang.srt.disaggregation.pvd.oasis_receive_slots import OasisReceiveSlotPool
from sglang.srt.disaggregation.pvd.protocol import (
    KVEntryKey,
    PVD_TRANSFER_LIFECYCLE_PROTOCOL,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.sparse_delivery import SparseDeliveryManifest
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
from sglang.srt.disaggregation.pvd.transfer_engine import MemorySlice
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget, TransportState

_QUARANTINE = []


def write_request(rank, rows, *, key):
    # Sum is the exact combined head-row count, not K/V components twice.
    counts = (rows,) if rows <= 32 else (32, rows - 32)
    delivery_id = uuid.uuid4().hex
    identity = WriteIdentity(PVD_TRANSFER_LIFECYCLE_PROTOCOL, f"native-V{rank}",
        "native-D", delivery_id, "pending-registration", uuid.uuid4().hex, rank, key)
    specs = tuple(SparseKVSpec("native-req", "native-inc", delivery_id, 1,
        key.transfer_id, "native-index", "native-mapping", "native-layout", 27,
        rank * 2 + i, tuple(2158 - j for j in range(n))) for i, n in enumerate(counts))
    return SparseDeliveryManifest(specs, "torch.float16", 128), identity


def poll_success(engine, handle, expected_bytes, *, timeout):
    deadline = time.monotonic() + timeout
    while not handle.transport_state.is_locally_safe_to_release:
        engine.poll(handle)
        if time.monotonic() >= deadline:
            raise TimeoutError("native local-session PUT did not become terminal")
        time.sleep(0.001)
    if (handle.transport_state != TransportState.TERMINAL_SUCCESS
            or handle.transferred_bytes != expected_bytes
            or not engine.cleanup_complete(handle)):
        raise RuntimeError("native local-session PUT lacks exact terminal/cleanup proof")


def run(args):
    device, current_device = torch.device(args.device), torch.device(args.current_device)
    if (device.type != "cuda" or device.index is None
            or current_device.type != "cuda" or current_device.index is None
            or current_device.index == device.index
            or not torch.cuda.is_available()
            or max(device.index, current_device.index) >= torch.cuda.device_count()):
        raise ValueError("two explicit valid indexed CUDA device arguments required")
    torch.cuda.set_device(current_device)
    baseline_current = torch.cuda.current_device()
    capacity_bytes = 64 * 512
    budget = TransferBudget(8 << 20, 16)
    engine = create_native_receive_group(hostname=args.hostname, gpu_id=device.index,
        rails=tuple(dict.fromkeys(args.rails)), transfer_budget=budget)
    pool = OasisReceiveSlotPool(engine, budget, device=device,
        receiver_epoch="native-D", slots_per_rank=2, capacity_bytes=capacity_bytes)
    _QUARANTINE.append((pool, engine))
    health = engine.health()
    endpoints = {rank: health["rails"][rail]["session_id"]
                 for rank, rail in enumerate(args.rails)}
    key = KVEntryKey("native-model", "native-prompt", uuid.uuid4().hex)
    sources, results = {}, []
    for rank, rail in enumerate(args.rails):
        owner = f"native-source:{rank}"
        budget.reserve(owner, capacity_bytes, 0)
        buffer = torch.empty(capacity_bytes, dtype=torch.uint8, device=device)
        registration = engine.adapters[rail].register_memory(buffer,
            endpoint=endpoints[rank], rank=rank, rail=rail,
            metadata={"native_probe_source": True})
        sources[rank] = (owner, registration)
        _QUARANTINE.append(registration)

    previous_generations, physical_ids = set(), set()
    native_lock = threading.Lock()
    for executor_round in range(2):
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Separate bootstrap/Decode executors must share the same four MRs.
            sizes = (1, 2, 8, 16, 32, 64)
            for rows in sizes:
                acquired = threading.Barrier(3)

                def worker(worker_slot):
                    ordering = CUDAReceiveOrdering(device)
                    leases = []
                    try:
                        for rank, rail in enumerate(args.rails):
                            manifest, identity = write_request(rank, rows, key=key)
                            lease = pool.acquire(manifest, identity, endpoint=endpoints[rank],
                                rail=rail, device=device, ordering=ordering)
                            leases.append((rank, rail, lease, manifest))
                        acquired.wait(timeout=args.timeout)
                        observations = []
                        for rank, rail, lease, manifest in leases:
                            # The source registration belongs to the same native
                            # per-HCA engine. Serialize source reuse through its
                            # terminal proof and consuming device fence.
                            with native_lock, torch.cuda.device(device):
                                source = sources[rank][1]
                                marker = executor_round * 71 + worker_slot * 37 + rank * 19 + rows
                                oracle = ((torch.arange(manifest.nbytes, dtype=torch.int64)
                                    + marker) % 251).to(torch.uint8)
                                source.buffer[:manifest.nbytes].copy_(oracle)
                                handle = engine.adapters[rail].submit_put(
                                    MemorySlice(source, 0, manifest.nbytes), lease.registration.descriptor)
                                try:
                                    poll_success(engine.adapters[rail], handle, manifest.nbytes,
                                                 timeout=args.timeout)
                                except BaseException:
                                    lease.quarantine("native local-session WRITE terminal unknown")
                                    raise
                                try:
                                    ordering.after_remote_write(lease.registration)
                                    observed = lease.buffer.cpu()
                                    if not torch.equal(observed, oracle):
                                        raise AssertionError("reused receive prefix differs from exact PUT bytes")
                                    payloads = manifest.payload_views(lease.buffer)
                                    try:
                                        if sum(p.nbytes for p in payloads) != manifest.nbytes:
                                            raise AssertionError("FP16 K/V payload view extent differs")
                                    finally:
                                        for payload in payloads:
                                            payload.close()
                                    # No local reader or native operation escapes
                                    # this synchronized, terminal-success scope.
                                    torch.cuda.synchronize(device)
                                except BaseException:
                                    lease.quarantine("native receive consumer completion unknown")
                                    raise
                            observations.append(dict(executor_round=executor_round,
                                worker_slot=worker_slot, worker_thread=threading.get_ident(), rank=rank, rows=rows,
                                nbytes=manifest.nbytes, generation=lease.identity.generation,
                                region_id=lease.identity.region_id,
                                physical_register_calls=lease.physical_register_calls,
                                allocate_seconds=lease.allocate_seconds,
                                register_seconds=lease.register_seconds,
                                terminal_state=handle.transport_state.value,
                                transferred_bytes=handle.transferred_bytes, exact_bytes=True))
                            lease.release_after_proof()
                        return observations
                    except BaseException:
                        # Hold every unreleased destination. An exception or a
                        # barrier cancellation is not a remote WRITE fence.
                        for _, _, lease, _ in leases:
                            if lease._active and lease._slot.unknown is None:
                                lease.quarantine("native probe worker did not retire with proof")
                        raise

                futures = [executor.submit(worker, slot) for slot in range(2)]
                acquired.wait(timeout=args.timeout)
                # Workers may already start returning leases, but physical
                # registration inventory is fixed at four from this point.
                if pool.snapshot()["physical_registrations"] != 4:
                    raise AssertionError("rank slots expanded or failed to register exactly four MRs")
                for future in futures:
                    observations = future.result(timeout=args.timeout * 4)
                    for observation in observations:
                        generation = observation["generation"]
                        if generation in previous_generations:
                            raise AssertionError("logical receive generation reused")
                        previous_generations.add(generation)
                        physical_ids.add(observation["region_id"])
                    results.extend(observations)
    before_close = pool.snapshot()
    if (len(results) != 48 or len(physical_ids) != 4
            or before_close["physical_register_calls"] != 4
            or before_close["leased_slots"] or before_close["unknown_slots"]):
        raise AssertionError("native slot ownership, reuse or byte gate did not pass")
    torch.cuda.set_device(current_device)
    pool.close()  # Different caller device; pass original physical handle only.
    if torch.cuda.current_device() != baseline_current:
        raise AssertionError("receive pool close changed caller CUDA device")
    for rank, rail in enumerate(args.rails):
        owner, registration = sources[rank]
        engine.adapters[rail].release_memory(registration)
        budget.release(owner)
    after_close = pool.snapshot()
    accounting = budget.snapshot()
    if (after_close["physical_releases"] != 4 or not after_close["closed"]
            or accounting["used_staging_bytes"] or accounting["used_inflight"]
            or engine.health()["registered_destinations"]
            or any(s["registered_regions"] or s["lifecycle"]["tracked_transfers"]
                   or s["lifecycle"]["unknown_transfers"]
                   for s in engine.health()["rails"].values())):
        raise AssertionError("native slot/source owners or budgets did not retire")
    _QUARANTINE.clear()
    return dict(status="passed", transport="mooncake_local_session",
        device=str(device), current_device=str(current_device), exact_byte_cases=len(results),
        two_executor_pools=True, bootstrap_and_decode_thread_ids_reusable=True,
        rows=(1, 2, 8, 16, 32, 64), before_close=before_close,
        after_close=after_close, transfer_budget=accounting,
        receive_health=engine.health(), observations=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--current-device", default="cuda:1")
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--rails", nargs=2, required=True)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("native probe output must stay inside checkout artifacts/")
    if args.timeout <= 0:
        raise ValueError("positive timeout required")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
    except BaseException as error:
        partial = dict(status="failed", error_type=type(error).__name__, error=str(error))
        if _QUARANTINE and isinstance(_QUARANTINE[0], tuple):
            pool, engine = _QUARANTINE[0]
            partial.update(pool=pool.snapshot(), receive_health=engine.health(),
                           transfer_budget=pool.budget.snapshot())
        output.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
        raise  # No exceptional cleanup can manufacture native completion.
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "observations"}, sort_keys=True))


if __name__ == "__main__":
    main()
